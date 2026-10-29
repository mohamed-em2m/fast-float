"""Zero-copy shared memory frame ring buffer for cross-process video streaming.

Layout per slot:
- frame_idx: int64 (8 bytes)
- pts_ms: float64 (8 bytes)
- width: int32 (4 bytes)
- height: int32 (4 bytes)
- audio_len: int32 (4 bytes)
- state: int32 (4 bytes) — 0 = free (consumer read it), 1 = written and unread
- audio_data: 2048 bytes (fixed capacity, enough for up to 1024 16-bit samples = 64 ms @ 16 kHz)
- image_data: width * height * 3 bytes (RGB24)

Flow control
------------
A plain round-robin ring lets the writer lap the reader: the producer renders
faster than the consumer (which copies every frame into Python objects on the
event loop), so within one chunk the writer overwrites slots the consumer has
not read yet. The consumer then reads a *newer* frame at an old index — the
picture and the audio jump forward, and because each chunk restarts the slot
counter at 0, the next chunk's frames land on top of the previous chunk's
unread tail ("a chunk comes over the old one").

``write_frame`` therefore waits for the slot it is about to reuse to be free
(``state == 0``), and ``read_frame``/``release_slot`` free it. The writer can
never be more than ``num_slots`` frames ahead of the reader, so a slot is never
overwritten while it still holds an unread frame. The wait is bounded: a
consumer that died mid-stream costs one timeout, after which the writer forces
the write and reports it (``forced_overwrites``) instead of stalling the GPU.
"""

from __future__ import annotations

import contextlib
import logging
import struct
import time
from multiprocessing import shared_memory

import numpy as np

logger = logging.getLogger(__name__)


class WorkerUnavailable(RuntimeError):
    """The inference worker has not published its shared-memory ring.

    Raised by :meth:`FrameRingBuffer.attach` when the consumer (the API) looks
    for the worker's segment and it does not exist: the worker is not running,
    is still loading its weights, or died. It subclasses ``RuntimeError`` on
    purpose so the HTTP layer maps it to a readable 503 (``/livekit/...``
    already converts ``RuntimeError`` into one) instead of an ASGI 500 with a
    bare ``FileNotFoundError``.
    """


def segment_exists(name: str) -> bool:
    """Whether shared-memory segment ``name`` is currently published."""
    try:
        shm = shared_memory.SharedMemory(name=name)
    except (FileNotFoundError, OSError):
        return False
    with contextlib.suppress(Exception):
        shm.close()
    return True


# Header struct: frame_idx(q), pts_ms(d), width(i), height(i), audio_len(i), state(i)
HEADER_FORMAT = "<qdiiii"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
MAX_AUDIO_BYTES_PER_FRAME = 2048  # 1024 samples @ 16 kHz = 64ms (standard 40ms frame = 1280 bytes)

SLOT_FREE = 0
SLOT_BUSY = 1

# How long the writer waits for the consumer to release a slot before it forces
# the write (a dead/slow consumer must not wedge the GPU process).
DEFAULT_SLOT_WAIT_S = 2.0
_SLOT_POLL_S = 0.0005


class FrameRingBuffer:
    """Manages a pre-allocated shared-memory circular ring buffer of video & audio frames."""

    def __init__(
        self,
        name: str,
        num_slots: int = 64,
        width: int = 512,
        height: int = 512,
        create: bool = False,
    ) -> None:
        self.name = name
        self.num_slots = max(4, int(num_slots))
        self.width = int(width)
        self.height = int(height)
        self.image_bytes = self.width * self.height * 3
        self.slot_size = HEADER_SIZE + MAX_AUDIO_BYTES_PER_FRAME + self.image_bytes
        self.total_size = self.slot_size * self.num_slots
        self.create = create

        if create:
            # Clean up any stale shared memory with the same name first
            try:
                stale = shared_memory.SharedMemory(name=self.name)
                stale.close()
                stale.unlink()
            except FileNotFoundError:
                pass

            self.shm = shared_memory.SharedMemory(create=True, size=self.total_size, name=self.name)
            # Initialize with zeros
            self.shm.buf[: self.total_size] = b"\x00" * self.total_size
        else:
            self.shm = shared_memory.SharedMemory(name=self.name)

        self._buf = self.shm.buf
        self.forced_overwrites = 0

        # A segment created by a different layout (an API restarted against an
        # old worker, or the reverse) would be read at the wrong offsets and
        # silently corrupt every frame. Fail loudly instead.
        actual_size = int(self.shm.size)
        if actual_size < self.total_size:
            with contextlib.suppress(Exception):
                self.shm.close()
            raise RuntimeError(
                f"shared-memory segment {name!r} is {actual_size} bytes but this build "
                f"expects {self.total_size} (num_slots={self.num_slots}, "
                f"{self.width}x{self.height}). Restart the inference worker and the API "
                "together so both use the same ring layout."
            )

    @classmethod
    def attach(
        cls,
        name: str,
        num_slots: int = 64,
        width: int = 512,
        height: int = 512,
        timeout: float = 0.0,
        poll_interval: float = 0.25,
    ) -> FrameRingBuffer:
        """Attach to the worker-owned ring, waiting up to ``timeout`` seconds.

        The worker creates the segment only *after* it has loaded the model, so
        an API that is already serving when the worker boots (or restarts)
        races with it. Polling here turns that race into a short wait instead
        of a hard failure, and an empty segment into an actionable
        :class:`WorkerUnavailable` instead of a bare ``FileNotFoundError``.
        """
        delay = max(0.01, float(poll_interval))
        deadline = time.monotonic() + max(0.0, float(timeout))
        while True:
            try:
                return cls(name=name, num_slots=num_slots, width=width, height=height, create=False)
            except FileNotFoundError:
                if time.monotonic() >= deadline:
                    raise WorkerUnavailable(cls._missing_message(name, timeout)) from None
                time.sleep(delay)

    @staticmethod
    def _missing_message(name: str, waited: float) -> str:
        waited_txt = f" (waited {waited:.0f}s)" if waited and waited > 0 else ""
        return (
            f"FLOAT inference worker is not running: shared-memory segment {name!r} "
            f"does not exist{waited_txt}. Live conversations render through that worker, "
            "so start both processes with ./scripts/run_dual_process.sh, or run the API "
            "in single-process mode by setting FLOAT_IPC_ENABLED=false "
            "(./scripts/run_all.sh does that automatically)."
        )

    def _slot_offset(self, slot_idx: int) -> int:
        return (slot_idx % self.num_slots) * self.slot_size

    def _state_offset(self, slot_idx: int) -> int:
        return self._slot_offset(slot_idx) + HEADER_SIZE - 4

    def _slot_state(self, slot_idx: int) -> int:
        return struct.unpack_from("<i", self._buf, self._state_offset(slot_idx))[0]

    def _set_slot_state(self, slot_idx: int, state: int) -> None:
        struct.pack_into("<i", self._buf, self._state_offset(slot_idx), int(state))

    def _wait_for_free(self, slot_idx: int, timeout: float | None) -> bool:
        """Block until ``slot_idx`` has been read (or ``timeout`` elapses)."""
        wait_s = DEFAULT_SLOT_WAIT_S if timeout is None else max(0.0, float(timeout))
        if self._slot_state(slot_idx) == SLOT_FREE:
            return True
        if wait_s <= 0:
            return False
        deadline = time.monotonic() + wait_s
        while True:
            time.sleep(_SLOT_POLL_S)
            if self._slot_state(slot_idx) == SLOT_FREE:
                return True
            if time.monotonic() >= deadline:
                return False

    def release_slot(self, slot_idx: int) -> None:
        """Mark a slot free without reading it (the consumer skipped this frame)."""
        self._set_slot_state(slot_idx, SLOT_FREE)

    def write_frame(
        self,
        slot_idx: int,
        frame_idx: int,
        pts_ms: float,
        image_array: np.ndarray,
        audio_bytes: bytes | None = None,
        wait_s: float | None = None,
    ) -> bool:
        """Write a video frame (RGB24) and paired audio slice into the specified slot.

        Zero-copy into the underlying memory-mapped buffer. Blocks (bounded by
        ``wait_s``) until the consumer has released the slot, so the ring can
        never overwrite a frame that has not been read yet. Returns ``True``
        when the write was safe and ``False`` when it had to force an overwrite
        of a still-unread slot.
        """
        offset = self._slot_offset(slot_idx)
        clean = self._wait_for_free(slot_idx, wait_s)
        if not clean and (wait_s is None or wait_s > 0):
            self.forced_overwrites += 1
            logger.warning(
                "[shm] slot %d still unread after %.2fs; overwriting it (consumer is "
                "%.0f frames behind) — frames may be skipped",
                slot_idx,
                DEFAULT_SLOT_WAIT_S if wait_s is None else wait_s,
                self.num_slots,
            )

        # 1. Header. Only publish it as free when the previous frame was really
        # read: on a forced overwrite the slot stays busy so nothing else
        # mistakes it for a clean slot.
        audio_data = audio_bytes or b""
        audio_len = min(len(audio_data), MAX_AUDIO_BYTES_PER_FRAME)
        h, w = image_array.shape[0], image_array.shape[1]
        header = struct.pack(
            HEADER_FORMAT,
            frame_idx,
            float(pts_ms),
            w,
            h,
            audio_len,
            SLOT_FREE if clean else SLOT_BUSY,
        )
        self._buf[offset : offset + HEADER_SIZE] = header

        # 2. Audio
        audio_offset = offset + HEADER_SIZE
        if audio_len > 0:
            self._buf[audio_offset : audio_offset + audio_len] = audio_data[:audio_len]

        # 3. Image
        img_offset = audio_offset + MAX_AUDIO_BYTES_PER_FRAME
        arr_contiguous = np.ascontiguousarray(image_array, dtype=np.uint8)
        img_raw = arr_contiguous.tobytes()
        n_img = min(len(img_raw), self.image_bytes)
        self._buf[img_offset : img_offset + n_img] = img_raw

        # 4. Publish: the consumer may read the slot from now on.
        self._set_slot_state(slot_idx, SLOT_BUSY)
        return clean

    def read_frame(self, slot_idx: int) -> tuple[int, float, np.ndarray, bytes]:
        """Read a frame out of the specified slot without reallocating ring structures.

        Returns (frame_idx, pts_ms, rgb_image_array, audio_bytes). The slot is
        released before returning, so the writer may reuse it immediately.
        """
        offset = self._slot_offset(slot_idx)

        # 1. Header
        header_bytes = bytes(self._buf[offset : offset + HEADER_SIZE])
        frame_idx, pts_ms, w, h, audio_len, _state = struct.unpack(HEADER_FORMAT, header_bytes)

        # 2. Audio
        audio_offset = offset + HEADER_SIZE
        audio_bytes = bytes(self._buf[audio_offset : audio_offset + audio_len]) if audio_len > 0 else b""

        # 3. Image: return a detached copy (or copy view) so the shared memory buffer
        # can be closed/unlinked without 'cannot close exported pointers exist' errors.
        img_offset = audio_offset + MAX_AUDIO_BYTES_PER_FRAME
        img_size = w * h * 3
        raw_bytes = bytes(self._buf[img_offset : img_offset + img_size])
        arr = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((h, w, 3))

        # 4. Hand the slot back to the writer.
        self._set_slot_state(slot_idx, SLOT_FREE)

        return frame_idx, pts_ms, arr, audio_bytes

    def close(self) -> None:
        """Close the shared memory handle in this process."""
        with contextlib.suppress(Exception):
            self.shm.close()

    def unlink(self) -> None:
        """Destroy the shared memory segment (call once by the creator/master on shutdown)."""
        if self.create:
            with contextlib.suppress(Exception):
                self.shm.unlink()
