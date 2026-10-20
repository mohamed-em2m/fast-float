"""RuntimeSession: per-device serving wrapper for one component backend.

Implementations wrap eager modules, ``torch.compile`` callables, or ORT
sessions. The contract, so call sites can stay backend-agnostic:

* torch in (any device/dtype) -> torch out on the caller's device;
* never let a backend failure crash the pipeline -- follow the
  ``degrade_on_error`` policy (sticky / transient / raise);
* ``replicate`` for the components that run on more than one GPU (only
  ``decoder`` today, per the registry).

See ``docs/BUILD_RUNTIME_SESSIONS.md``.
"""

from __future__ import annotations

import contextlib
import functools
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from types import FunctionType
from typing import Any


def _timed_forward(fn):
    """Count one backend call and report it to the active timing collector.

    ``stats.calls``/``stats.errors`` are always updated (the inspect CLI reads
    them); the timing row is only added when a ``timing_run`` is open. Wall time
    only -- ORT runs on its own streams, so CUDA events recorded here would not
    bracket the real work. Telemetry failures never reach the pipeline.
    """

    @functools.wraps(fn)
    def wrapper(self, *args: Any, **kwargs: Any) -> Any:
        stats = getattr(self, "stats", None)
        if stats is not None:
            stats.calls += 1
        try:
            from fast_float.utils.timing import current_collector

            collector = current_collector()
        except Exception:
            collector = None
        if collector is None:
            try:
                return fn(self, *args, **kwargs)
            except Exception:
                if stats is not None:
                    stats.errors += 1
                raise
        t0 = time.perf_counter()
        try:
            return fn(self, *args, **kwargs)
        except Exception:
            if stats is not None:
                stats.errors += 1
            raise
        finally:
            with contextlib.suppress(Exception):
                collector.add(
                    f"runtime:{self._label()}",
                    time.perf_counter() - t0,
                    None,
                    {"backend": self.backend, "device": self.device_id},
                )

    wrapper._runtime_timed = True
    return wrapper


@dataclass
class RuntimeStats:
    """Per-session counters surfaced by the runtime ``inspect`` CLI."""

    calls: int = 0
    errors: int = 0
    io_bindings: int = 0
    degraded: bool = False
    reason: str | None = None


class RuntimeSession(ABC):
    """Serves one component's forward pass on one device."""

    component: str = ""
    backend: str = "eager"

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Wrap each subclass's own ``forward`` once so every session reports
        # into utils/timing.py without per-implementation code. Subclasses that
        # only inherit ``forward`` are covered by the class that defines it.
        fwd = cls.__dict__.get("forward")
        if isinstance(fwd, FunctionType) and not getattr(fwd, "_runtime_timed", False):
            cls.forward = _timed_forward(fwd)

    def __init__(self, device_id: int | None = None, degrade_on_error: str = "transient"):
        self.device_id = device_id
        self.degrade_on_error = degrade_on_error
        self.stats = RuntimeStats()
        self._degraded = False
        self._closed = False
        # Cooperative: in the ``(RuntimeSession, nn.Module)`` mixins this chains
        # to ``nn.Module.__init__`` so torch bookkeeping is initialised.
        super().__init__()

    @property
    def providers(self) -> list:
        """Active ORT providers (empty for eager/compile runtimes)."""
        return []

    @property
    def degraded(self) -> bool:
        return self._degraded

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Run one inference call."""

    def replicate(self, device_id: int) -> RuntimeSession:
        """Return an equivalent session bound to ``device_id``.

        Only registry-``replicable`` components (``decoder`` today) may
        replicate; the gate fails loudly instead of silently sharing a session
        across GPUs.
        """
        if not self._replicable():
            raise NotImplementedError(f"{self._label()} is not replicable")
        return self._replicate(device_id)

    def _replicable(self) -> bool:
        if not self.component:
            return False
        try:
            from fast_float.onnx.registry import get

            return get(self.component).replicable
        except Exception:
            return False

    def _replicate(self, device_id: int) -> RuntimeSession:
        raise NotImplementedError(f"{type(self).__name__} has no replicate implementation")

    def _label(self) -> str:
        return self.component or type(self).__name__

    def degrade(self, reason: str) -> None:
        """Mark the session degraded (sticky) after a backend failure."""
        self._degraded = True
        self.stats.degraded = True
        self.stats.reason = reason

    def close(self) -> None:
        """Release resources (ORT sessions, CUDA memory). Best-effort."""
        self._closed = True


__all__ = ["RuntimeSession", "RuntimeStats"]
