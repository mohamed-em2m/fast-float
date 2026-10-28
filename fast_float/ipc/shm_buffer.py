"""Zero-copy shared memory frame ring buffer for cross-process video streaming.

Layout per slot:
- frame_idx: int64 (8 bytes)
- pts_ms: float64 (8 bytes)
- width: int32 (4 bytes)
- height: int32 (4 bytes)
- audio_len: int32 (4 bytes)
- audio_data: 2048 bytes (fixed capacity, enough for up to 1024 16-bit samples = 64 ms @ 16 kHz)
- image_data: width * height * 3 bytes (RGB24)
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


# Header struct: frame_idx(q), pts_ms(d), width(i), height(i), audio_len(i)
HEADER_FORMAT = "<qdi i i"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
MAX_AUDIO_BYTES_PER_FRAME = 2048  # 1024 samples @ 16 kHz = 64ms (standard 40ms frame = 1280 bytes)


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

    def write_frame(
        self,
        slot_idx: int,
        frame_idx: int,
        pts_ms: float,
        image_array: np.ndarray,
        audio_bytes: bytes | None = None,
    ) -> None:
        """Write a video frame (RGB24) and paired audio slice into the specified slot.

        Zero-copy into the underlying memory-mapped buffer.
        """
        offset = self._slot_offset(slot_idx)

        # 1. Header
        audio_data = audio_bytes or b""
        audio_len = min(len(audio_data), MAX_AUDIO_BYTES_PER_FRAME)
        h, w = image_array.shape[0], image_array.shape[1]
        header = struct.pack(HEADER_FORMAT, frame_idx, float(pts_ms), w, h, audio_len)
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

    def read_frame(self, slot_idx: int) -> tuple[int, float, np.ndarray, bytes]:
        """Read a frame out of the specified slot without reallocating ring structures.

        Returns (frame_idx, pts_ms, rgb_image_array, audio_bytes).
        """
        offset = self._slot_offset(slot_idx)

        # 1. Header
        header_bytes = bytes(self._buf[offset : offset + HEADER_SIZE])
        frame_idx, pts_ms, w, h, audio_len = struct.unpack(HEADER_FORMAT, header_bytes)

        # 2. Audio
        audio_offset = offset + HEADER_SIZE
        audio_bytes = bytes(self._buf[audio_offset : audio_offset + audio_len]) if audio_len > 0 else b""

        # 3. Image: return a detached copy (or copy view) so the shared memory buffer
        # can be closed/unlinked without 'cannot close exported pointers exist' errors.
        img_offset = audio_offset + MAX_AUDIO_BYTES_PER_FRAME
        img_size = w * h * 3
        raw_bytes = bytes(self._buf[img_offset : img_offset + img_size])
        arr = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((h, w, 3))

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
