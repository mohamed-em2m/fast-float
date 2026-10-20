"""Live fan-out: one dispatcher, many subscribers (WebSocket, WebRTC)."""

from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from dataclasses import replace
from typing import Any

import numpy as np

from fast_float.streaming.dispatcher import FrameDispatcher
from fast_float.streaming.types import StreamMeta

logger = logging.getLogger(__name__)

CLOSE = object()


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

    # ---- lifecycle ---------------------------------------------------- #
    def start(self, meta: StreamMeta) -> None:
        """Publish metadata. Called from the pipeline thread (via ``bind``)."""
        self.meta = meta
        if meta.audio_path:
            self.audio_path = meta.audio_path
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
        q: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
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
        for q in list(self._subs):
            self._push(q, item)

    def publish_threadsafe(self, item: tuple) -> None:
        with suppress(RuntimeError):  # loop already closed (shutdown)
            self._loop.call_soon_threadsafe(self.publish, item)

    @staticmethod
    def _push(q: asyncio.Queue, item: Any) -> None:
        if q.full():
            with suppress(asyncio.QueueEmpty):
                q.get_nowait()
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

    def set_audio_path(self, path: str | None) -> None:
        """Provide the full-request audio before bind (chunk paths are partial)."""
        self._audio_hint = path
        self.stream.audio_path = path

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
