"""Per-request frame tap for live transports (WebSocket, WebRTC, custom sinks).

The decode pipeline calls :meth:`FrameDispatcher.submit` from its writer thread.
In the default ``live`` mode ``submit`` never blocks: a slow consumer costs
frames, never throughput.  Every delivered frame keeps generation order -- a
frame is dispatched only after every earlier frame has been dispatched or
reported lost through :meth:`gap`.

Modes
-----
``live`` (default)
    Bounded queue, drops the oldest frames when full or too stale.  Gaps are
    allowed but reported, and ``idx`` is the frame's true position in the video.

``lossless``
    Bounded queue that blocks the producer when full (up to
    ``submit_timeout_ms``), so every submitted frame is delivered in order.
    This throttles generation by design.  On timeout the tap is disabled rather
    than silently reordering.

Hooks are called on the dispatcher's delivery thread, in order::

    open(meta) -> dispatch(idx, frames, pts_ms)* -> [gap(start, end, reason)*] -> close()

``error(exc)`` replaces the remainder of the stream after a hook raises.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import numpy as np

from fast_float.streaming.types import StreamMeta

logger = logging.getLogger(__name__)

_MODE_LIVE = "live"
_MODE_LOSSLESS = "lossless"


class FrameDispatcher:
    """Subclass it, override the hooks, pass the instance to inference."""

    # ---- policy ----
    mode: str = _MODE_LIVE
    max_queue_frames: int = 50
    max_lag_ms: float = 1000.0
    """How far the oldest queued frame may trail the newest one, in media time.

    Bounds consumer lag independently of generation speed; ``0`` disables it.
    """
    gap_timeout_ms: float = 0.0
    submit_timeout_ms: float = 2000.0
    close_timeout: float = 2.0
    max_dispatch_ms: float = 250.0
    stall_limits: int = 3
    dispatch_granularity: str = "batch"
    abort_on_error: bool = False
    fail_fast_on_open: bool = False

    def __init__(self) -> None:
        self._cv = threading.Condition()
        self._meta: StreamMeta | None = None
        self._thread: threading.Thread | None = None
        self._stop_event: threading.Event | None = None

        self._bound = False
        self._closed = False
        self._opened = False
        self._close_called = False
        self._disabled_reason: str | None = None

        self._pending: dict[int, np.ndarray] = {}
        self._pending_frames = 0
        self._dropped: list[tuple[int, int]] = []
        self._next_expected = 0
        self._cursor = 0
        self._gap_deadline: float | None = None
        self._slow_streak = 0
        self._stats: dict[str, Any] = {
            "submitted": 0,
            "delivered": 0,
            "dropped_overflow": 0,
            "dropped_lag": 0,
            "skipped_gap": 0,
            "gap_calls": 0,
            "dispatch_calls": 0,
            "max_backlog": 0,
            "last_idx": -1,
        }

    # ------------------------------------------------------------------ #
    # Framework API (called by the pipeline / service)
    # ------------------------------------------------------------------ #
    def bind(self, meta: StreamMeta, stop_event: threading.Event | None = None) -> FrameDispatcher:
        """Attach stream metadata and start delivery.  Idempotent."""
        if self.mode not in (_MODE_LIVE, _MODE_LOSSLESS):
            logger.warning("[dispatch] unknown mode %r; using %r", self.mode, _MODE_LIVE)
            self.mode = _MODE_LIVE
        if self.dispatch_granularity not in ("batch", "frame"):
            logger.warning("[dispatch] unknown granularity %r; using 'batch'", self.dispatch_granularity)
            self.dispatch_granularity = "batch"

        with self._cv:
            if self._bound:
                return self
            self._bound = True
            self._meta = meta
            self._stop_event = stop_event
            self._next_expected = int(meta.first_idx)
            self._cursor = int(meta.first_idx)
            self._stats["last_idx"] = int(meta.first_idx) - 1

        try:
            self.open(meta)
        except BaseException as exc:
            logger.exception("[dispatch] open() failed; tap disabled")
            with self._cv:
                self._disabled_reason = "open_failed"
                self._cv.notify_all()
            self.error(exc)
            if self.fail_fast_on_open:
                raise
            return self

        with self._cv:
            self._opened = True
            self._thread = threading.Thread(
                target=self._deliver_loop,
                name=f"dispatch-{meta.request_id or 'stream'}",
                daemon=True,
            )
            self._thread.start()
        return self

    def submit(self, frames: Any, idx: int | None = None) -> bool:
        """Offer a batch of uint8 frames.  Never blocks in ``live`` mode.

        ``frames`` is ``[n, H, W, C]``; the data is copied before returning, so
        the caller may recycle its buffer immediately.  ``idx`` defaults to the
        next frame number in submission order, which is what the pipeline uses.
        """
        with self._cv:
            if not self._bound or self._closed or self._disabled_reason is not None:
                return False
            n = len(frames)
            if n <= 0:
                return False

            if idx is None:
                idx = self._cursor
            idx = int(idx)
            end = idx + n
            if end <= self._next_expected:
                return False  # already delivered
            self._cursor = max(self._cursor, end)

            skip = 0
            if idx < self._next_expected:
                skip = self._next_expected - idx
                idx = self._next_expected
                n = end - idx
            if idx in self._pending:
                return False  # duplicate batch
            self._stats["submitted"] += n

            meta = self._meta
            assert meta is not None
            if self.mode == _MODE_LIVE:
                if self._pending and self.max_lag_ms > 0:
                    # Bound the queued backlog in media time: a consumer that
                    # cannot keep up loses the oldest frames, never the newest.
                    # Generation speed is irrelevant here (it can be slower than
                    # real time without starving the stream).
                    newest_pts = meta.pts_ms(end - 1)
                    while self._pending and newest_pts - meta.pts_ms(min(self._pending)) > self.max_lag_ms:
                        self._drop_locked(min(self._pending), "dropped_lag")
                while self._pending_frames + n > self.max_queue_frames and self._pending:
                    self._drop_locked(min(self._pending), "dropped_overflow")
                if self._pending_frames + n > self.max_queue_frames:
                    self._record_drop(idx, idx + n, "dropped_overflow")
                    return False
            elif self.mode == _MODE_LOSSLESS:
                deadline = time.monotonic() + max(0.0, self.submit_timeout_ms) / 1000.0
                while self._pending_frames + n > self.max_queue_frames:
                    if self._closed or self._disabled_reason is not None:
                        return False
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        self._disable_locked("submit_timeout")
                        return False
                    self._cv.wait(remaining)
                if self._closed or self._disabled_reason is not None:
                    return False

            self._pending_frames += n

        try:
            data = np.array(frames[skip:], copy=True) if skip else np.array(frames, copy=True)
        except BaseException:
            with self._cv:
                self._pending_frames -= n
                self._cv.notify_all()
            raise

        with self._cv:
            if self._closed or self._disabled_reason is not None:
                self._pending_frames -= n
                self._cv.notify_all()
                return False
            self._pending[idx] = data
            if self._pending_frames > self._stats["max_backlog"]:
                self._stats["max_backlog"] = self._pending_frames
            self._cv.notify_all()
        return True

    def shutdown(self, timeout: float | None = None) -> None:
        """Stop accepting frames, drain in order, then call ``close()`` once."""
        with self._cv:
            self._closed = True
            self._cv.notify_all()
            thread = self._thread
        if thread is None:
            self._finish()
            return
        thread.join(self.close_timeout if timeout is None else timeout)
        if thread.is_alive():
            with self._cv:
                self._disable_locked("close_timeout")
            self._finish()

    def stats(self) -> dict[str, Any]:
        with self._cv:
            out = dict(self._stats)
            out["disabled_reason"] = self._disabled_reason
            out["pending_frames"] = self._pending_frames
            out["next_expected"] = self._next_expected
            return out

    def stop(self) -> None:
        """Ask the pipeline to stop generating (e.g. the client left)."""
        if self._stop_event is not None:
            self._stop_event.set()

    def set_audio_path(self, path: str | None) -> None:
        """Optional hint for transports that also need the request audio."""

    @property
    def disabled(self) -> bool:
        return self._disabled_reason is not None

    def __enter__(self) -> FrameDispatcher:
        return self

    def __exit__(self, *exc_info) -> None:
        self.shutdown()

    # ------------------------------------------------------------------ #
    # Hooks -- override these
    # ------------------------------------------------------------------ #
    def open(self, meta: StreamMeta) -> None:
        """Called once before the first frame."""

    def dispatch(self, idx: int, frames: np.ndarray, pts_ms: float) -> None:
        """Deliver frames starting at ``idx``.  Must return quickly."""

    def gap(self, start: int, end: int, reason: str) -> None:
        """Frames ``[start, end)`` were lost (``overflow``, ``lag``, ``missing``)."""

    def error(self, exc: BaseException) -> None:
        logger.error("[dispatch] %s: %s", type(exc).__name__, exc)

    def close(self) -> None:
        """Called once after the last dispatch."""

    # ------------------------------------------------------------------ #
    # Delivery
    # ------------------------------------------------------------------ #
    def _deliver_loop(self) -> None:
        while True:
            task = self._next_task()
            kind = task[0]
            if kind == "done":
                break
            if kind == "frames":
                self._dispatch_batch(task[1], task[2])
            elif kind == "gap":
                self._call_gap(task[1], task[2], task[3])
        self._finish()

    def _next_task(self) -> tuple:
        with self._cv:
            while True:
                if self._disabled_reason is not None:
                    self._pending.clear()
                    self._pending_frames = 0
                    return ("done",)

                lost_end = self._lost_end(self._next_expected)
                if lost_end is not None:
                    start = self._next_expected
                    self._next_expected = lost_end
                    self._gap_deadline = None
                    self._stats["skipped_gap"] += lost_end - start
                    self._prune_dropped()
                    return ("gap", start, lost_end, "dropped")

                if self._next_expected in self._pending:
                    frames = self._pending.pop(self._next_expected)
                    self._pending_frames -= len(frames)
                    idx = self._next_expected
                    self._next_expected += len(frames)
                    self._gap_deadline = None
                    self._cv.notify_all()
                    return ("frames", idx, frames)

                if self._closed and not self._pending:
                    return ("done",)

                if self._pending:
                    smallest = min(self._pending)
                    if smallest > self._next_expected:
                        if not self._closed:
                            if self.gap_timeout_ms > 0:
                                if self._gap_deadline is None:
                                    self._gap_deadline = time.monotonic() + self.gap_timeout_ms / 1000.0
                                remaining = self._gap_deadline - time.monotonic()
                                if remaining > 0:
                                    self._cv.wait(remaining)
                                    continue
                            else:
                                # No declared loss: wait for the earlier frames
                                # (the producer is ordered; a real hole shows up
                                # only on close).
                                self._cv.wait(0.1)
                                continue
                        start = self._next_expected
                        self._next_expected = smallest
                        self._gap_deadline = None
                        self._stats["skipped_gap"] += smallest - start
                        return ("gap", start, smallest, "missing")

                self._cv.wait(0.1)

    def _dispatch_batch(self, idx: int, frames: np.ndarray) -> None:
        meta = self._meta
        assert meta is not None
        n = len(frames)
        t0 = time.perf_counter()
        try:
            if self.dispatch_granularity == "frame":
                for i in range(n):
                    self.dispatch(idx + i, frames[i], meta.pts_ms(idx + i))
            else:
                self.dispatch(idx, frames, meta.pts_ms(idx))
        except BaseException as exc:
            with self._cv:
                self._disable_locked("dispatch_error")
            self.error(exc)
            return

        with self._cv:
            self._stats["dispatch_calls"] += 1
            self._stats["delivered"] += n
            self._stats["last_idx"] = idx + n - 1
        if self.mode != _MODE_LIVE:
            return

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if elapsed_ms > self.max_dispatch_ms:
            self._slow_streak += 1
            if self._slow_streak >= max(1, self.stall_limits):
                with self._cv:
                    self._disable_locked("dispatch_stall")
                self.error(TimeoutError(f"dispatch took {elapsed_ms:.0f}ms"))
        else:
            self._slow_streak = 0

    def _call_gap(self, start: int, end: int, reason: str) -> None:
        with self._cv:
            self._stats["gap_calls"] += 1
        try:
            self.gap(start, end, reason)
        except BaseException as exc:
            with self._cv:
                self._disable_locked("gap_error")
            self.error(exc)

    # ------------------------------------------------------------------ #
    # Internals -- call with the lock held
    # ------------------------------------------------------------------ #
    def _record_drop(self, start: int, end: int, key: str) -> None:
        if end <= start:
            return
        self._stats[key] += end - start
        self._dropped.append((start, end))
        self._dropped.sort()
        merged: list[tuple[int, int]] = []
        for s, e in self._dropped:
            if merged and s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))
        self._dropped[:] = merged

    def _drop_locked(self, start: int, key: str) -> None:
        frames = self._pending.pop(start)
        n = len(frames)
        self._pending_frames -= n
        self._record_drop(start, start + n, key)

    def _lost_end(self, idx: int) -> int | None:
        for start, end in self._dropped:
            if start <= idx < end:
                return end
            if start > idx:
                break
        return None

    def _prune_dropped(self) -> None:
        limit = self._next_expected
        self._dropped[:] = [(s, e) for s, e in self._dropped if e > limit]

    def _disable_locked(self, reason: str) -> None:
        if self._disabled_reason is None:
            self._disabled_reason = reason
        self._cv.notify_all()
        if self.abort_on_error and self._stop_event is not None:
            self._stop_event.set()

    def _finish(self) -> None:
        with self._cv:
            if self._close_called:
                return
            self._close_called = True
            opened = self._opened
        if not opened:
            return
        try:
            self.close()
        except BaseException:
            logger.exception("[dispatch] close() failed")


class FrameQueue(FrameDispatcher):
    """Feeds an ``asyncio.Queue`` so async transports can ``await queue.get()``.

    ``maxsize`` bounds the hand-off; the oldest item is dropped when full.
    """

    def __init__(self, loop: Any, maxsize: int = 8) -> None:
        super().__init__()
        self._loop = loop
        self._maxsize = max(1, int(maxsize))
        self._queue: Any = None
        self._adapter_dropped = 0

    def open(self, meta: StreamMeta) -> None:
        import asyncio

        self._queue = asyncio.Queue(maxsize=self._maxsize)

    def dispatch(self, idx: int, frames: np.ndarray, pts_ms: float) -> None:
        self._loop.call_soon_threadsafe(self._push, (idx, frames, pts_ms))

    def _push(self, item: tuple) -> None:
        if self._queue.full():
            try:
                self._queue.get_nowait()
                self._adapter_dropped += 1
            except Exception:
                pass
        self._queue.put_nowait(item)

    async def get(self) -> tuple:
        return await self._queue.get()
