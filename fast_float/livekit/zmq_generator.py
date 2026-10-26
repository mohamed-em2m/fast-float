"""ZeroMQ-backed FLOAT video generator (Process 1 / streaming-backend side).

Drop-in replacement for ``FloatVideoGenerator`` that delegates GPU inference to
the dedicated ``InferenceWorker`` (Process 2) over ZeroMQ + shared memory.

Usage
-----
Swap ``FloatVideoGenerator`` for ``ZmqFloatVideoGenerator`` in the LiveKit
session when ``[ipc] enabled = true`` in ``settings.toml``::

    from fast_float.livekit.zmq_generator import ZmqFloatVideoGenerator

    gen = ZmqFloatVideoGenerator(ref_image_path=ref, nfe=5)
    await gen.prepare_silence()

The generator connects to the worker's two sockets:

* ``REQ`` on ``zmq_control_endpoint`` (default ``tcp://127.0.0.1:5555``)
  — synchronous request/reply for control commands and infer requests.
* ``SUB`` on ``zmq_frames_endpoint`` (default ``tcp://127.0.0.1:5556``)
  — asynchronous per-frame notifications published by the worker.

Shared memory
-------------
The generator attaches to the ``FrameRingBuffer`` created by the worker
(``create=False``). Each ``IPCFrameNotification`` carries a ``slot_idx``; the
generator calls ``ring.read_frame(slot_idx)`` to retrieve the RGB24 image and
PCM16 audio, then wraps them in ``rtc.VideoFrame`` / ``rtc.AudioFrame``.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import uuid
from collections.abc import AsyncIterator
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional heavy imports
# ---------------------------------------------------------------------------
try:
    import zmq
    import zmq.asyncio as azmq
except ImportError:
    zmq = None  # type: ignore[assignment]
    azmq = None  # type: ignore[assignment]

try:
    from livekit import rtc

    RTC_AVAILABLE = True
except ImportError:
    rtc = None  # type: ignore[assignment]
    RTC_AVAILABLE = False

import contextlib

from fast_float.ipc.messages import IPCFrameNotification, IPCRequest, IPCResponse
from fast_float.ipc.shm_buffer import FrameRingBuffer

# ---------------------------------------------------------------------------
# Constants — kept in sync with generator.py and inference_worker.py
# ---------------------------------------------------------------------------
TARGET_FPS = 25.0
TARGET_WIDTH = 512
TARGET_HEIGHT = 512
SAMPLE_RATE = 16_000
SAMPLES_PER_FRAME = int(SAMPLE_RATE / TARGET_FPS)  # 640
BYTES_PER_FRAME = SAMPLES_PER_FRAME * 2  # 1280 bytes PCM16

# How long to buffer incoming audio before dispatching an infer request (sec).
DEFAULT_CHUNK_SEC = 5.0
DEFAULT_FIRST_CHUNK_SEC = 5.0

_SILENCE_SENTINEL = object()


class ZmqFloatVideoGenerator:
    """Async video generator that proxies inference to the FLOAT worker over IPC.

    Parameters
    ----------
    ref_image_path:
        Path to the reference portrait image (passed to the worker on first
        infer request and cached thereafter).
    nfe:
        Number of flow-matching evaluation steps (speed/quality knob).
    chunk_sec:
        Audio buffer threshold before launching an infer request (seconds).
    first_chunk_sec:
        Buffer threshold for the *first* render (may differ for latency tuning).
    control_endpoint:
        ZeroMQ endpoint for the REP/REQ control socket.
    frames_endpoint:
        ZeroMQ endpoint for the PUB/SUB frame notification socket.
    shm_name:
        Shared-memory segment name (must match the worker).
    shm_slots:
        Ring-buffer slot count (must match the worker).
    session_id:
        Unique identifier for this generator session (auto-generated if omitted).
    """

    def __init__(
        self,
        ref_image_path: str = "",
        nfe: int = 5,
        chunk_sec: float = DEFAULT_CHUNK_SEC,
        first_chunk_sec: float = DEFAULT_FIRST_CHUNK_SEC,
        control_endpoint: str = "tcp://127.0.0.1:5555",
        frames_endpoint: str = "tcp://127.0.0.1:5556",
        shm_name: str = "float_frame_ring",
        shm_slots: int = 64,
        session_id: str = "",
    ) -> None:
        if zmq is None:
            raise RuntimeError("pyzmq is required: pip install pyzmq")

        self._ref_image_path = ref_image_path
        self._nfe = nfe
        self._chunk_sec = chunk_sec
        self._first_chunk_sec = first_chunk_sec
        self._control_ep = control_endpoint
        self._frames_ep = frames_endpoint
        self._shm_name = shm_name
        self._shm_slots = shm_slots
        self._session_id = session_id or str(uuid.uuid4())

        # Asyncio + ZMQ context (created in _ensure_connected)
        self._zmq_ctx: Any = None
        self._req_sock: Any = None  # DEALER/REQ to control port
        self._sub_sock: Any = None  # SUB to frames port

        # Shared memory (attached once, consumer side)
        self._ring: FrameRingBuffer | None = None

        # Audio accumulation
        self._audio_buf: list[bytes] = []
        self._audio_samples: int = 0
        self._chunk_samples = int(SAMPLE_RATE * chunk_sec)
        self._first_chunk_samples = int(SAMPLE_RATE * first_chunk_sec)
        self._first_chunk_done = False

        # Generation counter — incremented each push_audio cycle.
        self._gen = 0

        # Frame output queue (filled by _frame_listener_task).
        self._frame_queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=256)

        # Pre-encoded silence frames (list of (rtc.VideoFrame, rtc.AudioFrame)).
        self._silence_frames: list[Any] = []
        self._silence_ready = asyncio.Event()

        # Background tasks
        self._listener_task: asyncio.Task | None = None
        self._background_tasks: set[asyncio.Task] = set()

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------

    async def _ensure_connected(self) -> None:
        """Lazily create ZMQ context, sockets, and SHM attachment."""
        if self._zmq_ctx is not None:
            return

        self._zmq_ctx = azmq.Context()

        # REQ socket for synchronous control commands.
        self._req_sock = self._zmq_ctx.socket(zmq.REQ)
        self._req_sock.connect(self._control_ep)

        # SUB socket for per-frame notifications from the worker.
        self._sub_sock = self._zmq_ctx.socket(zmq.SUB)
        self._sub_sock.connect(self._frames_ep)
        self._sub_sock.setsockopt(zmq.SUBSCRIBE, b"")  # receive all topics

        # Attach to the shared-memory ring (worker must already own it).
        self._ring = FrameRingBuffer(
            name=self._shm_name,
            num_slots=self._shm_slots,
            create=False,
        )

        # Start the background frame-listener.
        self._listener_task = asyncio.create_task(self._frame_listener(), name="zmq-frame-listener")
        logger.info(
            "[zmq-gen] connected session=%s ctrl=%s frames=%s shm=%s",
            self._session_id,
            self._control_ep,
            self._frames_ep,
            self._shm_name,
        )

    async def _send_request(self, req: IPCRequest) -> IPCResponse:
        """Send a synchronous IPC request and await the reply."""
        await self._req_sock.send_string(req.to_json())
        raw = await self._req_sock.recv()
        return IPCResponse.from_json(raw)

    # ------------------------------------------------------------------
    # Silence pre-rendering
    # ------------------------------------------------------------------

    async def prepare_silence(self) -> None:
        """Ask the worker to pre-render a silence loop and cache the frames.

        Must be called once per session before the first ``push_audio()`` /
        ``__aiter__`` usage to populate ``_silence_frames``.
        """
        await self._ensure_connected()

        req = IPCRequest(
            cmd="prepare_silence",
            session_id=self._session_id,
            req_id=str(uuid.uuid4()),
            ref_image_path=self._ref_image_path,
            gen=0,
        )
        resp = await self._send_request(req)
        if resp.status != "ok":
            logger.warning("[zmq-gen] prepare_silence failed: %s", resp.error_msg)
            return

        # Collect the published frames.
        silence_frames: list[Any] = []
        for _ in range(resp.total_frames + 1):  # +1 for the is_last sentinel
            try:
                raw = await asyncio.wait_for(self._sub_sock.recv_string(), timeout=30.0)
            except TimeoutError:
                logger.warning("[zmq-gen] timeout waiting for silence frame")
                break
            notif = IPCFrameNotification.from_json(raw)
            if notif.is_last:
                break
            if self._ring is None:
                continue
            _, _, img_arr, audio_bytes = self._ring.read_frame(notif.slot_idx)
            vf = _make_video_frame(img_arr)
            af = _make_audio_frame(audio_bytes)
            if vf is not None and af is not None:
                silence_frames.append((vf, af))

        self._silence_frames = silence_frames
        self._silence_ready.set()
        logger.info("[zmq-gen] %d silence frames cached", len(silence_frames))

    # ------------------------------------------------------------------
    # Audio input
    # ------------------------------------------------------------------

    def push_audio(self, pcm_bytes: bytes) -> None:
        """Buffer raw PCM16 mono audio from the Gemini bridge.

        When enough audio is accumulated (``chunk_sec`` threshold), an async
        infer request is queued.
        """
        self._audio_buf.append(pcm_bytes)
        self._audio_samples += len(pcm_bytes) // 2  # int16 → samples

        threshold = self._first_chunk_samples if not self._first_chunk_done else self._chunk_samples

        if self._audio_samples >= threshold:
            chunk_bytes = b"".join(self._audio_buf)
            self._audio_buf.clear()
            self._audio_samples = 0
            self._first_chunk_done = True
            self._gen += 1

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.get_event_loop()

            if loop.is_running():
                asyncio.run_coroutine_threadsafe(self._dispatch_infer(chunk_bytes, self._gen), loop)
            else:
                task = loop.create_task(self._dispatch_infer(chunk_bytes, self._gen))
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)

    async def _dispatch_infer(self, pcm_bytes: bytes, gen: int) -> None:
        """Send an infer request to the worker for the given audio chunk."""
        await self._ensure_connected()
        pcm_b64 = base64.b64encode(pcm_bytes).decode()
        n_samples = len(pcm_bytes) // 2

        req = IPCRequest(
            cmd="infer",
            session_id=self._session_id,
            req_id=str(uuid.uuid4()),
            audio_pcm=pcm_b64,
            valid_samples=n_samples,
            sample_rate=SAMPLE_RATE,
            nfe=self._nfe,
            ref_image_path=self._ref_image_path,
            gen=gen,
        )
        resp = await self._send_request(req)
        if resp.status not in ("ok", "interrupted"):
            logger.warning("[zmq-gen] infer failed (gen=%d): %s", gen, resp.error_msg)

    def clear_buffer(self) -> None:
        """Discard buffered audio and interrupt any in-flight inference."""
        self._audio_buf.clear()
        self._audio_samples = 0
        self._first_chunk_done = False
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.get_event_loop()

        if loop.is_running():
            asyncio.run_coroutine_threadsafe(self._send_interrupt(), loop)
        else:
            task = loop.create_task(self._send_interrupt())
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

    async def _send_interrupt(self) -> None:
        await self._ensure_connected()
        req = IPCRequest(
            cmd="interrupt",
            session_id=self._session_id,
            req_id=str(uuid.uuid4()),
        )
        await self._send_request(req)

    # ------------------------------------------------------------------
    # Frame listener (background async task)
    # ------------------------------------------------------------------

    async def _frame_listener(self) -> None:
        """Receive IPCFrameNotifications from the PUB socket and enqueue frames."""
        while True:
            try:
                raw = await self._sub_sock.recv_string()
            except Exception as exc:
                logger.debug("[zmq-gen] listener recv error: %s", exc)
                await asyncio.sleep(0.01)
                continue

            try:
                notif = IPCFrameNotification.from_json(raw)
            except Exception:
                continue

            # Skip notifications from other sessions.
            if notif.session_id != self._session_id:
                continue
            if notif.is_last:
                continue

            if self._ring is None:
                continue

            _, _, img_arr, audio_bytes = self._ring.read_frame(notif.slot_idx)
            vf = _make_video_frame(img_arr)
            af = _make_audio_frame(audio_bytes)
            if vf is None or af is None:
                continue

            try:
                self._frame_queue.put_nowait((vf, af))
            except asyncio.QueueFull:
                # Drop the oldest to maintain latency.
                with contextlib.suppress(asyncio.QueueEmpty):
                    self._frame_queue.get_nowait()
                self._frame_queue.put_nowait((vf, af))

    # ------------------------------------------------------------------
    # AsyncIterator — yields (VideoFrame, AudioFrame) pairs
    # ------------------------------------------------------------------

    def __aiter__(self) -> AsyncIterator[tuple[Any, Any]]:
        return self._iter()

    async def _iter(self) -> AsyncIterator[tuple[Any, Any]]:
        """Yield live frames from the worker; fall back to silence while idle."""
        await self._ensure_connected()

        frame_interval = 1.0 / TARGET_FPS
        silence_idx = 0

        while True:
            # Try to get a live frame (non-blocking).
            try:
                pair = self._frame_queue.get_nowait()
                yield pair
                continue
            except asyncio.QueueEmpty:
                pass

            # No live frame — yield silence if available.
            if self._silence_frames:
                pair = self._silence_frames[silence_idx % len(self._silence_frames)]
                silence_idx += 1
                yield pair
            else:
                await asyncio.sleep(frame_interval)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Shut down sockets and release shared memory."""
        self._stopped = True
        if self._listener_task is not None:
            self._listener_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._listener_task

        if self._req_sock is not None:
            self._req_sock.close()
        if self._sub_sock is not None:
            self._sub_sock.close()
        if self._zmq_ctx is not None:
            self._zmq_ctx.term()
        if self._ring is not None:
            self._ring.close()
            # Do NOT unlink — the worker owns the SHM segment.

        logger.info("[zmq-gen] session %s closed", self._session_id)

    def stop(self) -> None:
        """Synchronous stop method matching FloatVideoGenerator's interface."""
        self._stopped = True
        self.clear_buffer()
        if self._listener_task is not None and not self._listener_task.done():
            self._listener_task.cancel()


# ---------------------------------------------------------------------------
# Frame construction helpers
# ---------------------------------------------------------------------------


def _make_video_frame(img_arr: np.ndarray) -> Any | None:
    """Wrap a numpy RGB24 array in an ``rtc.VideoFrame``."""
    if not RTC_AVAILABLE or rtc is None:
        return None
    try:
        h, w = img_arr.shape[0], img_arr.shape[1]
        return rtc.VideoFrame(
            width=w,
            height=h,
            type=rtc.VideoBufferType.RGB24,
            data=img_arr.tobytes(),
        )
    except Exception as exc:
        logger.debug("[zmq-gen] VideoFrame creation failed: %s", exc)
        return None


def _make_audio_frame(audio_bytes: bytes) -> Any | None:
    """Wrap raw PCM16 bytes in an ``rtc.AudioFrame``."""
    if not RTC_AVAILABLE or rtc is None:
        return None
    try:
        n_samples = len(audio_bytes) // 2
        if n_samples == 0:
            n_samples = SAMPLES_PER_FRAME
            audio_bytes = b"\x00" * (n_samples * 2)
        return rtc.AudioFrame(
            data=audio_bytes,
            sample_rate=SAMPLE_RATE,
            num_channels=1,
            samples_per_channel=n_samples,
        )
    except Exception as exc:
        logger.debug("[zmq-gen] AudioFrame creation failed: %s", exc)
        return None
