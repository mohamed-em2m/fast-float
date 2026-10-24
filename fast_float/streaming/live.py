"""Live fan-out: one dispatcher, many subscribers (WebSocket, WebRTC)."""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import suppress
from dataclasses import replace
from typing import Any

import numpy as np

from fast_float.core.perf_metrics import FrameClock
from fast_float.streaming.dispatcher import FrameDispatcher
from fast_float.streaming.types import StreamMeta

logger = logging.getLogger(__name__)

CLOSE = object()

# Log a subscriber-overflow warning every this many dropped frames.
_DROP_WARN_EVERY = 25


class LiveQueue(asyncio.Queue):
    """Subscriber queue with drop accounting.

    :class:`LiveStream` is a lossy fan-out by design: when a transport cannot
    keep up, the *oldest* item is dropped so playback stays near-live. The
    counters below let a transport (and the logs) explain a jump in the
    stream instead of silently skipping frames.
    """

    def __init__(self, maxsize: int = 64) -> None:
        super().__init__(maxsize=maxsize)
        self.dropped_frames = 0
        self.last_dropped_idx: int | None = None
        # Recent dropped ranges, merged when contiguous (frames only).
        self.dropped_ranges: list[tuple[int, int]] = []

    def note_dropped(self, item: Any) -> None:
        if not isinstance(item, tuple) or not item or item[0] != "frame":
            return
        idx = int(item[1])
        self.dropped_frames += 1
        self.last_dropped_idx = idx
        if self.dropped_ranges and self.dropped_ranges[-1][1] == idx:
            start, _ = self.dropped_ranges[-1]
            self.dropped_ranges[-1] = (start, idx + 1)
        else:
            self.dropped_ranges.append((idx, idx + 1))
        if len(self.dropped_ranges) > 64:  # keep the bookkeeping bounded
            del self.dropped_ranges[:-64]


class LiveStream:
    """A finished-or-running request that any number of transports can consume.

    Items pushed to subscribers are tuples::

        ("meta", StreamMeta)
        ("frame", idx, pts_ms, np.ndarray)  # ndarray is H x W x C uint8 RGB
        ("gap", start, end, reason)

    The final ``CLOSE`` sentinel is enqueued by :meth:`finish`.
    """

    def __init__(self, request_id: str, loop: asyncio.AbstractEventLoop) -> None:
        self.request_id = request_id
        self.meta: StreamMeta | None = None
        self.audio_path: str | None = None
        self.result: dict[str, Any] | None = None
        self.error: str | None = None
        self.done = False
        self._loop = loop
        self._subs: set[asyncio.Queue] = set()
        self._meta_event = asyncio.Event()
        self._drop_warn_total = 0
        # Delivery-side clock: how late each frame reached the transport, plus
        # the dropped-frame count (see fast_float.core.perf_metrics).
        self.clock: FrameClock | None = None

    # ---- lifecycle ---------------------------------------------------- #
    def start(self, meta: StreamMeta) -> None:
        """Publish metadata. Called from the pipeline thread (via ``bind``)."""
        self.meta = meta
        if meta.audio_path:
            self.audio_path = meta.audio_path
        self.clock = FrameClock(
            fps=meta.fps,
            start_s=meta.started_at_s if meta.started_at_s is not None else time.monotonic(),
            first_idx=meta.first_idx,
        )
        # Waking an asyncio.Event is not thread-safe; hop onto the loop that
        # owns it (``opened`` runs on the generation thread).
        with suppress(RuntimeError):
            self._loop.call_soon_threadsafe(self._meta_event.set)

    async def wait_meta(self, timeout: float = 30.0) -> StreamMeta | None:
        if self.meta is not None:
            return self.meta
        try:
            await asyncio.wait_for(self._meta_event.wait(), timeout)
        except asyncio.TimeoutError:
            return None
        return self.meta

    def finish(self, result: dict[str, Any] | None = None, error: str | None = None) -> None:
        if self.done:
            return
        self.result = result
        self.error = error
        self.done = True
        # Queue the sentinel through the loop instead of pushing it inline:
        # frame publishes already scheduled by the (now joined) delivery thread
        # must land *before* CLOSE, or the tail of the stream is lost.
        with suppress(RuntimeError):
            self._loop.call_soon_threadsafe(self._close_subscribers)

    def _close_subscribers(self) -> None:
        # A stream that finished before bind() (e.g. the request failed during
        # preprocessing) must not leave watchers blocked in wait_meta().
        self._meta_event.set()
        for q in list(self._subs):
            self._push(q, CLOSE)

    # ---- fan-out ------------------------------------------------------ #
    def subscribe(self, maxsize: int = 64) -> asyncio.Queue:
        q: LiveQueue = LiveQueue(maxsize=maxsize)
        self._subs.add(q)
        if self.done:
            self._push(q, CLOSE)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subs.discard(q)

    @property
    def subscriber_count(self) -> int:
        return len(self._subs)

    def publish(self, item: tuple) -> None:
        # No ``done`` guard here: publishes scheduled before finish() must be
        # delivered ahead of the (also queued) CLOSE sentinel.
        if self.clock is not None and item[0] == "frame":
            # item == ("frame", idx, pts_ms, ndarray)
            self.clock.on_frame(item[1], item[2], time.monotonic())
        for q in list(self._subs):
            self._push(q, item)
        self._warn_drops()

    def dropped_frames(self) -> int:
        """Frames this stream dropped across all slow subscribers."""
        return sum(getattr(q, "dropped_frames", 0) for q in self._subs)

    def _warn_drops(self) -> None:
        total = self.dropped_frames()
        if total >= self._drop_warn_total + _DROP_WARN_EVERY:
            self._drop_warn_total = total
            logger.warning(
                "[live] %s: a subscriber fell behind; %d frame(s) dropped (playback jumps)",
                self.request_id,
                total,
            )

    def metrics(self) -> dict[str, Any]:
        """Delivery metrics of this stream (TTFF, latency percentiles, drops)."""
        data = self.clock.to_dict() if self.clock is not None else {}
        data["subscriber_dropped_frames"] = self.dropped_frames()
        return data

    def publish_threadsafe(self, item: tuple) -> None:
        with suppress(RuntimeError):  # loop already closed (shutdown)
            self._loop.call_soon_threadsafe(self.publish, item)

    @staticmethod
    def _push(q: asyncio.Queue, item: Any) -> None:
        if q.full():
            with suppress(asyncio.QueueEmpty):
                dropped = q.get_nowait()
                if isinstance(q, LiveQueue):
                    q.note_dropped(dropped)
        with suppress(asyncio.QueueFull):
            q.put_nowait(item)


class LiveStreamDispatcher(FrameDispatcher):
    """Dispatcher that publishes every frame to :class:`LiveStream` subscribers."""

    mode = "live"
    max_queue_frames = 64
    max_lag_ms = 1500.0
    dispatch_granularity = "frame"

    def __init__(self, request_id: str, loop: asyncio.AbstractEventLoop) -> None:
        super().__init__()
        self.request_id = request_id
        self.stream = LiveStream(request_id, loop)
        self._audio_hint: str | None = None
        self._trim_remaining = 0

    def set_audio_path(self, path: str | None) -> None:
        """Provide the full-request audio before bind (chunk paths are partial)."""
        self._audio_hint = path
        self.stream.audio_path = path

    def begin_chunk(self, trim_frames: int = 0) -> None:
        """Skip the overlapping head of the next chunk.

        ``_generate_chunked`` renders overlapping windows so each chunk has
        context at its start. The merge trims that head from the MP4; live
        transports must do the same before publishing, otherwise every chunk
        repeats half a second of video while the full-request audio keeps
        moving forward (and later audio slices fall off the end).
        """
        self._trim_remaining = max(0, int(trim_frames))

    def submit(self, frames: Any, idx: int | None = None) -> bool:
        if self._trim_remaining:
            drop = min(self._trim_remaining, len(frames))
            self._trim_remaining -= drop
            frames = frames[drop:]
            if idx is not None:
                idx += drop
            if len(frames) == 0:
                return True
        return super().submit(frames, idx=idx)

    def open(self, meta: StreamMeta) -> None:
        if self._audio_hint:
            meta = replace(meta, audio_path=self._audio_hint)
        self.stream.start(meta)
        logger.info(
            "[live] stream %s ready (%sx%s @ %sfps)",
            self.request_id,
            meta.width,
            meta.height,
            meta.fps,
        )

    def dispatch(self, idx: int, frames: np.ndarray, pts_ms: float) -> None:
        self.stream.publish_threadsafe(("frame", idx, pts_ms, frames))

    def gap(self, start: int, end: int, reason: str) -> None:
        self.stream.publish_threadsafe(("gap", start, end, reason))
