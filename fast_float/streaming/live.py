"""Live fan-out: one dispatcher, many subscribers (WebSocket, WebRTC)."""

from __future__ import annotations

import asyncio
import collections
import logging
import os
import struct
import tempfile
import time
from contextlib import suppress
from dataclasses import replace
from typing import Any

import numpy as np

from fast_float.core.perf_metrics import FrameClock
from fast_float.streaming.dispatcher import FrameDispatcher
from fast_float.streaming.types import StreamMeta

try:  # OpenCV is a hard dependency of the pipeline; keep this module usable without it.
    import cv2
except ImportError:  # pragma: no cover - exercised only in minimal installs
    cv2 = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

CLOSE = object()

# Log a subscriber-overflow warning every this many dropped frames.
_DROP_WARN_EVERY = 25

# A fast GPU renders far ahead of real time (100+ fps for a 25 fps stream).
# Transports that play at ``meta.fps`` (LiveKit, WebRTC) must buffer the
# in-flight render instead of dropping everything that does not fit in a
# near-live queue; size that buffer by bytes so a long request still has a
# hard memory ceiling.
LIVE_BUFFER_BYTES = 1024 * 1024 * 1024  # 1 GiB
LIVE_QUEUE_MIN_FRAMES = 128
LIVE_QUEUE_MAX_FRAMES = 4096  # ~164 s at 25 fps, ~3.2 GiB at 512x512

# Frames that do not fit in the in-memory window are appended, in order, to a
# temporary spool file.  A 512x512 RGB frame is 768 KiB raw but only ~50-90 KiB
# as JPEG, so a multi-minute preview costs hundreds of MiB on disk instead of
# several GiB of RAM.  The cap is a last-resort guard against a runaway request
# filling the disk; it is far above any normal clip (4 GiB ~ 55 min at 25 fps).
LIVE_SPILL_BYTES = 4 * 1024 * 1024 * 1024  # 4 GiB
LIVE_SPILL_JPEG_QUALITY = 85

# Spool record headers.  ``kind`` is 0=frame, 1=gap, 2=close.
_SPILL_FRAME = struct.Struct(">BIdHHBBI")  # kind, idx, pts_ms, h, w, c, codec, nbytes
_SPILL_GAP = struct.Struct(">BIIH")  # kind, start, end, reason_len


def live_queue_size(meta: StreamMeta | None) -> int:
    """Frames to keep in RAM for a transport that plays at real time.

    Sized from the stream's frame dimensions so the queue holds the near-live
    part of the render (e.g. ~1365 frames / 55 s at 512x512).  Anything beyond
    this window is spooled to disk by :class:`LiveQueue` instead of dropped, so
    the memory ceiling no longer bounds the clip length.
    """
    if meta is None or meta.width <= 0 or meta.height <= 0:
        return LIVE_QUEUE_MIN_FRAMES
    frame_bytes = meta.width * meta.height * max(1, meta.channels)
    return max(
        LIVE_QUEUE_MIN_FRAMES,
        min(LIVE_QUEUE_MAX_FRAMES, LIVE_BUFFER_BYTES // frame_bytes),
    )


class LiveQueue:
    """In-order subscriber queue with a compressed disk overflow.

    The in-memory deque holds the near-live window (``maxsize`` items).  Once
    it is full, further items are appended to a temporary spool file and read
    back in order, so a fast render that is much longer than real time still
    plays every frame.  Ordering is strict: while the spool holds anything,
    new items go to the spool too; memory is drained first.

    Dropping is now only a fallback for a failed/oversized spool.  The
    ``dropped_*`` counters still let a transport explain a jump, while
    ``spilled_frames`` reports how much of the clip was buffered on disk.
    """

    def __init__(
        self,
        maxsize: int = 64,
        *,
        request_id: str | None = None,
        spill: bool = True,
    ) -> None:
        self._maxsize = max(1, int(maxsize))
        self._mem: collections.deque[Any] = collections.deque()
        self._waiters: set[asyncio.Future[None]] = set()
        self._request_id = request_id or ""
        self._spill_enabled = spill
        self._spool: Any = None
        self._spool_fd: int | None = None
        self._spool_write = 0
        self._spool_read_off = 0
        self._spool_records = 0
        self._spool_read = 0
        self._spool_bytes = 0
        self._spilling = False
        self._pending_close = False
        self._disposed = False
        self._spill_error_logged = False
        self.dropped_frames = 0
        self.last_dropped_idx: int | None = None
        # Recent dropped ranges, merged when contiguous (frames only).
        self.dropped_ranges: list[tuple[int, int]] = []
        self.spilled_frames = 0

    # ---- asyncio.Queue-compatible surface ----------------------------- #
    @property
    def maxsize(self) -> int:
        return self._maxsize

    def resize(self, maxsize: int) -> None:
        """Change the in-memory window once the stream's frame size is known."""
        self._maxsize = max(1, int(maxsize))
        self._wakeup()

    def qsize(self) -> int:
        pending_close = 1 if self._pending_close else 0
        return len(self._mem) + (self._spool_records - self._spool_read) + pending_close

    def empty(self) -> bool:
        return self.qsize() == 0

    def full(self) -> bool:
        return len(self._mem) >= self._maxsize

    def put_nowait(self, item: Any) -> None:
        if self._disposed:
            return
        if item is CLOSE:
            # End-of-stream is a flag rather than a spool record: it must be
            # delivered after every buffered frame, and it must survive a full
            # spool (otherwise the consumer would wait forever).
            if self._spilling or self._spool_records > self._spool_read:
                self._pending_close = True
            else:
                self._mem.append(CLOSE)
            self._wakeup()
            return
        if self._spilling or self.full():
            if self._spill_enabled and self._spill(item):
                self._wakeup()
                return
            if self._spool_records > self._spool_read:
                # Already serving older frames from disk: putting this one in
                # RAM would make it jump ahead of them, so drop it instead.
                self.note_dropped(item)
                return
            # No spool (disabled, full, or I/O failure): keep the old
            # bounded, near-live behaviour.
            self._drop_oldest()
        self._mem.append(item)
        self._wakeup()

    async def put(self, item: Any) -> None:
        self.put_nowait(item)

    def get_nowait(self) -> Any:
        if self._mem:
            return self._mem.popleft()
        if self._spool_records > self._spool_read:
            item = self._read_spill()
            if self._spool_records == self._spool_read:
                self._reset_spool()
            return item
        if self._pending_close:
            self._pending_close = False
            return CLOSE
        raise asyncio.QueueEmpty

    async def get(self) -> Any:
        while True:
            try:
                return self.get_nowait()
            except asyncio.QueueEmpty:
                pass
            if self._disposed:
                return CLOSE
            waiter: asyncio.Future[None] = asyncio.get_running_loop().create_future()
            self._waiters.add(waiter)
            try:
                await waiter
            finally:
                self._waiters.discard(waiter)

    # ---- diagnostics -------------------------------------------------- #
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

    def dispose(self) -> None:
        """Release the spool.  Safe to call more than once."""
        self._disposed = True
        self._mem.clear()
        self._close_spool()
        self._wakeup()

    # ---- overflow spool ----------------------------------------------- #
    def _drop_oldest(self) -> None:
        if self._mem:
            self.note_dropped(self._mem.popleft())

    def _wakeup(self) -> None:
        for waiter in tuple(self._waiters):
            if not waiter.done():
                waiter.set_result(None)
        self._waiters.clear()

    def _ensure_spool(self) -> bool:
        if self._spool is not None:
            return True
        try:
            # TemporaryFile is unlinked on POSIX, so a crash cannot leave a
            # multi-hundred-MiB spool behind.
            self._spool = tempfile.TemporaryFile(prefix="float-live-", suffix=".spool")  # noqa: SIM115
            self._spool_fd = self._spool.fileno()
            return True
        except OSError as exc:
            if not self._spill_error_logged:
                self._spill_error_logged = True
                logger.warning(
                    "[live] %s: could not create frame spool (%s); falling back to dropping",
                    self._request_id,
                    exc,
                )
            return False

    def _close_spool(self) -> None:
        if self._spool is not None:
            with suppress(OSError):
                self._spool.close()
        self._spool = None
        self._spool_fd = None
        self._spool_write = 0
        self._spool_read_off = 0
        self._spool_records = 0
        self._spool_read = 0
        self._spool_bytes = 0
        self._spilling = False
        self._pending_close = False

    def _reset_spool(self) -> None:
        """Reclaim the spool once the consumer has read everything in it."""
        if self._spool is None:
            self._spilling = False
            return
        with suppress(OSError):
            self._spool.seek(0)
            self._spool.truncate(0)
        self._spool_write = 0
        self._spool_read_off = 0
        self._spool_records = 0
        self._spool_read = 0
        self._spool_bytes = 0
        self._spilling = False

    def _write_all(self, data: bytes, offset: int) -> None:
        assert self._spool_fd is not None
        view = memoryview(data)
        written = 0
        while written < len(view):
            written += os.pwrite(self._spool_fd, view[written:], offset + written)

    def _append_record(self, header: bytes, payload: bytes = b"") -> bool:
        total = len(header) + len(payload)
        if self._spool_bytes + total > LIVE_SPILL_BYTES:
            if not self._spill_error_logged:
                self._spill_error_logged = True
                logger.warning(
                    "[live] %s: live frame spool reached %.1f GiB; older frames are buffered but new ones will be dropped",
                    self._request_id,
                    LIVE_SPILL_BYTES / (1024**3),
                )
            return False
        if not self._ensure_spool():
            return False
        try:
            self._write_all(header, self._spool_write)
            if payload:
                self._write_all(payload, self._spool_write + len(header))
        except OSError as exc:
            if not self._spill_error_logged:
                self._spill_error_logged = True
                logger.warning("[live] %s: frame spool write failed (%s); falling back to dropping", self._request_id, exc)
            return False
        self._spool_write += total
        self._spool_bytes += total
        self._spool_records += 1
        return True

    def _spill(self, item: Any) -> bool:
        if not isinstance(item, tuple) or not item:
            return False
        if item[0] == "frame":
            _, idx, pts_ms, frame = item
            encoded = self._encode_frame(frame)
            if encoded is None:
                return False
            h, w, c, codec, data = encoded
            header = _SPILL_FRAME.pack(0, int(idx), float(pts_ms), h, w, c, codec, len(data))
            if not self._append_record(header, data):
                return False
            self.spilled_frames += 1
            self._spilling = True
            return True
        if item[0] == "gap":
            _, start, end, reason = item
            payload = str(reason).encode("utf-8")[:65535]
            header = _SPILL_GAP.pack(1, int(start), int(end), len(payload))
            if not self._append_record(header, payload):
                return False
            self._spilling = True
            return True
        return False

    @staticmethod
    def _encode_frame(frame: Any) -> tuple[int, int, int, int, bytes] | None:
        """Return (h, w, channels, codec, payload); codec 1=JPEG, 0=raw."""
        arr = np.asarray(frame)
        if arr.ndim == 2:
            arr = arr[:, :, None]
        if arr.ndim != 3:
            return None
        h, w, c = arr.shape
        if c == 3:
            rgb = np.ascontiguousarray(arr)
            if cv2 is not None:
                ok, buf = cv2.imencode(
                    ".jpg",
                    rgb,
                    [cv2.IMWRITE_JPEG_QUALITY, LIVE_SPILL_JPEG_QUALITY],
                )
                if ok:
                    return h, w, 3, 1, buf.tobytes()
            return h, w, 3, 0, rgb.tobytes()
        # The pipeline emits RGB24.  Other layouts (alpha, grayscale) are
        # rare; store them raw so their shape/channel count is preserved.
        return h, w, c, 0, np.ascontiguousarray(arr).tobytes()

    def _read_exact(self, nbytes: int) -> bytes:
        assert self._spool_fd is not None
        chunks: list[bytes] = []
        remaining = nbytes
        offset = self._spool_read_off
        while remaining > 0:
            chunk = os.pread(self._spool_fd, remaining, offset)
            if not chunk:
                raise OSError("frame spool is truncated")
            chunks.append(chunk)
            offset += len(chunk)
            remaining -= len(chunk)
        self._spool_read_off = offset
        return b"".join(chunks)

    def _read_spill(self) -> Any:
        # Read the kind byte first: frame and gap headers have different
        # lengths, and over-reading would eat into the next record.
        kind = self._read_exact(1)[0]
        if kind == 1:
            head = bytes([kind]) + self._read_exact(_SPILL_GAP.size - 1)
            _, start, end, reason_len = _SPILL_GAP.unpack(head)
            reason = self._read_exact(reason_len).decode("utf-8", "replace")
            self._spool_read += 1
            return ("gap", start, end, reason)
        head = bytes([kind]) + self._read_exact(_SPILL_FRAME.size - 1)
        _, idx, pts_ms, h, w, c, codec, nbytes = _SPILL_FRAME.unpack(head)
        data = self._read_exact(nbytes)
        self._spool_read += 1
        if codec == 1 and cv2 is not None:
            decoded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            # OpenCV's colour convention round-trips: the array was encoded as
            # if it were BGR and decodes back to the same RGB bytes.
            frame = np.ascontiguousarray(decoded) if decoded is not None else np.zeros((h, w, c), np.uint8)
        else:
            frame = np.frombuffer(data, np.uint8).reshape(h, w, c).copy()
        return ("frame", idx, pts_ms, frame)


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
        self._subs: set[LiveQueue] = set()
        # Counters of subscribers that already left, so metrics() keeps
        # reporting the whole request after its transports disconnect.
        self._retired_dropped = 0
        self._retired_spilled = 0
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
    def subscribe(self, maxsize: int = 64) -> LiveQueue:
        q = LiveQueue(maxsize=maxsize, request_id=self.request_id)
        self._subs.add(q)
        if self.done:
            self._push(q, CLOSE)
        return q

    def unsubscribe(self, q: LiveQueue) -> None:
        """Stop fanning out to ``q``.

        The queue is not disposed here: a consumer may still be draining
        frames that were buffered for it.  Consumers call ``q.dispose()`` when
        they are done (or let the temporary spool be collected).
        """
        if q in self._subs:
            self._subs.discard(q)
            self._retired_dropped += getattr(q, "dropped_frames", 0)
            self._retired_spilled += getattr(q, "spilled_frames", 0)

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
        """Frames this stream dropped across all (current and past) subscribers."""
        return self._retired_dropped + sum(getattr(q, "dropped_frames", 0) for q in self._subs)

    def spilled_frames(self) -> int:
        """Frames buffered on disk for subscribers that could not keep up."""
        return self._retired_spilled + sum(getattr(q, "spilled_frames", 0) for q in self._subs)

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
        data["subscriber_spilled_frames"] = self.spilled_frames()
        return data

    def publish_threadsafe(self, item: tuple) -> None:
        with suppress(RuntimeError):  # loop already closed (shutdown)
            self._loop.call_soon_threadsafe(self.publish, item)

    @staticmethod
    def _push(q: LiveQueue, item: Any) -> None:
        # LiveQueue spills overflow to disk, so a fast render that is longer
        # than the in-memory window still delivers every frame in order.
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
