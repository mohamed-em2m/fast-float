"""timing.py - lightweight per-stage timing for the FLOAT pipeline.

Collects wall-clock (``perf_counter``) durations for named pipeline stages and,
when requested, the matching CUDA-event duration so GPU compute time can be
told apart from host stalls. One summary table is printed at the end of a run;
debug mode also logs each individual measurement as it happens.

Usage::

    from fast_float.utils.timing import timed, timing_run

    with timing_run("request"):  # outermost scope owns the summary
        with timed("preprocess"):
            ...
        with timed("decode", gpu=True):  # CUDA-event time in addition to wall
            ...

Enable per-call debug lines with ``FLOAT_DEBUG=1`` (or ``FLOAT_TIMING=1``), or
by setting the ``fast_float`` logger to DEBUG. The summary always prints at INFO.

Nested ``timing_run`` scopes are re-entrant: the first (outermost) scope owns the
collector and prints the summary, so a request that fans out into worker threads
still produces a single consolidated report.
"""

from __future__ import annotations

import functools
import logging
import math
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from fast_float.utils.logger import debug_enabled as _env_debug_enabled

try:
    import torch
except Exception:  # pragma: no cover - torch is a hard dependency in practice
    torch = None

logger = logging.getLogger("fast_float.timing")


def debug_timing_enabled() -> bool:
    """True when per-call timing lines should be logged.

    Driven by ``$FLOAT_DEBUG``/``$FLOAT_TIMING`` or by the ``fast_float`` logger
    being at DEBUG (so ``--debug`` in the app turns it on automatically).
    """
    if _env_debug_enabled():
        return True
    return logging.getLogger("fast_float").isEnabledFor(logging.DEBUG)


def _cuda_available() -> bool:
    return torch is not None and torch.cuda.is_available()


@dataclass
class _Stage:
    walls: list[float] = field(default_factory=list)
    gpu_events: list[tuple[Any, Any]] = field(default_factory=list)


class TimingCollector:
    """Thread-safe accumulator of per-stage wall and GPU durations.

    Stages are aggregated by name: hundreds of decode calls collapse into one
    ``decode`` row with count/total/mean/max/p95.
    """

    def __init__(self, name: str = "run", debug: bool | None = None):
        self.name = name
        self.debug = debug_timing_enabled() if debug is None else debug
        self._stages: dict[str, _Stage] = {}
        self._lock = threading.Lock()

    def add(
        self,
        stage: str,
        wall_s: float,
        gpu_events: tuple[Any, Any] | None = None,
        fields: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            st = self._stages.get(stage)
            if st is None:
                st = self._stages[stage] = _Stage()
            st.walls.append(wall_s)
            if gpu_events is not None:
                st.gpu_events.append(gpu_events)

        if self.debug:
            gpu_note = ""
            if gpu_events is not None:
                gpu_note = f" gpu={self._elapsed_ms(gpu_events):.3f}ms"
            extra = ""
            if fields:
                extra = " " + " ".join(f"{k}={v}" for k, v in fields.items())
            logger.debug(f"[timing] {stage}: {wall_s * 1000:.3f} ms{gpu_note}{extra}")

    @staticmethod
    def _elapsed_ms(events: tuple[Any, Any]) -> float:
        start, end = events
        try:
            end.synchronize()
            return float(start.elapsed_time(end))
        except Exception:
            return float("nan")

    def _gpu_ms(self, stage: _Stage) -> list[float]:
        return [self._elapsed_ms(ev) for ev in stage.gpu_events]

    def flush(self) -> None:
        """Emit the aggregated summary via the ``fast_float`` logger (INFO)."""
        with self._lock:
            stages = dict(self._stages)
        if not stages:
            return

        rows: list[tuple[str, int, float, float, float, float, float]] = []
        total_wall = total_gpu = 0.0
        for stage, st in stages.items():
            n = len(st.walls)
            if not n:
                continue
            tw = sum(st.walls)
            tg = sum(self._gpu_ms(st))
            ordered = sorted(st.walls)
            p95 = ordered[min(n - 1, max(0, math.ceil(0.95 * n) - 1))]
            total_wall += tw
            total_gpu += tg
            rows.append((stage, n, tw, tw / n, max(st.walls), p95, tg))
        rows.sort(key=lambda r: r[2], reverse=True)

        head = f"[timing] {self.name}: {total_wall:.3f}s wall"
        if total_gpu:
            head += f", {total_gpu / 1000.0:.3f}s gpu"
        logger.info(head)
        logger.info(f"  {'stage':<16}{'n':>6}{'total':>11}{'mean':>11}{'max':>11}{'p95':>11}{'gpu':>11}")
        for stage, n, tw, mean, mx, p95, tg in rows:
            logger.info(
                f"  {stage:<16}{n:>6}{_fmt_s(tw):>11}{_fmt_ms(mean):>11}"
                f"{_fmt_ms(mx):>11}{_fmt_ms(p95):>11}{(_fmt_s(tg / 1000.0) if tg else '-'):>11}"
            )


def _fmt_s(seconds: float) -> str:
    return f"{seconds:.3f}s"


def _fmt_ms(seconds: float) -> str:
    return f"{seconds * 1000:.2f}ms"


# --------------------------------------------------------------------------- #
# Active-collector plumbing (process-global + lock: worker threads see it too,
# and the request lock serialises runs so only one collector is live at a time).
# --------------------------------------------------------------------------- #
_CURRENT: TimingCollector | None = None
_CURRENT_LOCK = threading.Lock()


def current_collector() -> TimingCollector | None:
    return _CURRENT


@contextmanager
def timing_run(name: str = "run", debug: bool | None = None) -> Iterator[TimingCollector]:
    """Open (or join) the active timing scope.

    The outermost scope creates the collector, prints the summary on exit and
    resets the global; nested scopes reuse it. Worker threads share the global.
    """
    global _CURRENT
    with _CURRENT_LOCK:
        existing = _CURRENT
        owner = existing is None
        if owner:
            _CURRENT = existing = TimingCollector(name, debug)

    try:
        yield existing
    finally:
        if owner:
            with _CURRENT_LOCK:
                _CURRENT = None
            existing.flush()


@contextmanager
def timed(stage: str, *, gpu: bool = False, **fields: Any) -> Iterator[None]:
    """Time a block under ``stage`` (no-op when no run is active).

    ``gpu=True`` records a CUDA-event pair on the current stream so the summary
    can report real GPU time; reading it is deferred to :meth:`TimingCollector.flush`
    so the mid-run pipeline never stalls on a synchronize.
    """
    collector = _CURRENT
    if collector is None:
        yield
        return

    want_gpu = gpu and _cuda_available()
    start_event = None
    if want_gpu:
        start_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

    t0 = time.perf_counter()
    try:
        yield
    finally:
        wall = time.perf_counter() - t0
        gpu_events = None
        if want_gpu:
            try:
                end_event = torch.cuda.Event(enable_timing=True)
                end_event.record()
                gpu_events = (start_event, end_event)
            except Exception:
                gpu_events = None
        collector.add(stage, wall, gpu_events, fields or None)


def timed_fn(stage: str, *, gpu: bool = False):
    """Decorator form of :func:`timed` for whole functions/methods."""

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with timed(stage, gpu=gpu):
                return fn(*args, **kwargs)

        return wrapper

    return decorator


def timing_scope(name: str = "run", debug: bool | None = None):
    """Decorator form of :func:`timing_run` for a whole (sync) function/method."""

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with timing_run(name, debug):
                return fn(*args, **kwargs)

        return wrapper

    return decorator


__all__ = [
    "TimingCollector",
    "current_collector",
    "debug_timing_enabled",
    "timed",
    "timed_fn",
    "timing_run",
    "timing_scope",
]
