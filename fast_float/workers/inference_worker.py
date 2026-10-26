"""FLOAT GPU inference worker — Process 2 in the dual-process architecture.

Run as a standalone process::

    python -m fast_float.workers.inference_worker

Responsibilities
----------------
* Load the FLOAT model once on startup (singleton ``OptimizedInferenceAgent``).
* Open two ZeroMQ sockets (tcp, Windows-safe):
  - ``REP`` on ``zmq_control_port`` — receives ``IPCRequest``, replies ``IPCResponse``.
  - ``PUB`` on ``zmq_frames_port``  — pushes ``IPCFrameNotification`` per frame.
* Create and own the ``FrameRingBuffer`` shared-memory segment (``create=True``).
* On ``cmd="infer"``:
  - Decode base64 PCM16, write a temp WAV.
  - Build a ``_ShmFrameDispatcher`` that writes every decoded frame into a SHM
    slot and immediately publishes an ``IPCFrameNotification`` over PUB.
  - Call ``agent.run_inference(...)`` with that dispatcher.
  - Reply ``IPCResponse(status="ok", total_frames=N)``.
* On ``cmd="interrupt"``: set ``_interrupt_event`` so the in-flight dispatcher
  raises ``_Cancelled``; reply immediately.
* On ``cmd="status"``: reply with VRAM / model info.
* On ``cmd="prepare_silence"``: pre-render a silence loop (same wav/ref as last
  ``infer``); reply with total frames produced.
* Graceful shutdown on SIGINT / SIGTERM.

Design notes
------------
* The REP socket is synchronous — the backend must wait for the reply before
  sending the next request.  This is intentional: inference is single-GPU, so
  there is never more than one outstanding job.  The PUB socket is fire-and-forget
  and carries the per-frame payload concurrently with the reply.
* ``_interrupt_event`` is a plain ``threading.Event``.  The dispatcher checks it
  after writing each frame; when set it raises ``_Cancelled`` which bubbles up
  through ``run_inference`` and is caught here.
* All ZeroMQ endpoints use ``tcp://`` because ``ipc://`` (Unix domain sockets) is
  not reliably supported on Windows.
"""

from __future__ import annotations

import base64
import logging
import os
import signal
import tempfile
import threading
import time
import wave
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional heavy imports — workers only; keep importable without GPU/torch.
# ---------------------------------------------------------------------------
try:
    import zmq
except ImportError:
    zmq = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# IPC layer
# ---------------------------------------------------------------------------
import contextlib

from fast_float.ipc.messages import IPCFrameNotification, IPCRequest, IPCResponse
from fast_float.ipc.shm_buffer import FrameRingBuffer

# ---------------------------------------------------------------------------
# FLOAT inference
# ---------------------------------------------------------------------------
from fast_float.streaming.dispatcher import FrameDispatcher

# Default socket ports — mirror settings.toml [ipc] section.
DEFAULT_CONTROL_PORT = 5555
DEFAULT_FRAMES_PORT = 5556
DEFAULT_SHM_NAME = "float_frame_ring"
DEFAULT_SHM_SLOTS = 64

SAMPLE_RATE = 16_000
FPS = 25
SAMPLES_PER_FRAME = SAMPLE_RATE // FPS  # 640


class _Cancelled(Exception):
    """Raised by the SHM dispatcher when the current request is interrupted."""


# ---------------------------------------------------------------------------
# SHM-backed dispatcher
# ---------------------------------------------------------------------------


class _ShmFrameDispatcher(FrameDispatcher):
    """FrameDispatcher subclass that writes every frame into a SHM ring buffer
    and publishes an ``IPCFrameNotification`` over a ZeroMQ PUB socket.

    Policy: ``live`` mode, no max-lag drop (the consumer is on the same
    machine and reads from SHM; it should never be slower than inference).
    """

    mode = "live"
    max_queue_frames = 128  # generous — SHM slots are pre-allocated
    max_lag_ms = 0.0  # disable lag-based drops
    abort_on_error = True  # stop inference on dispatch error

    def __init__(
        self,
        ring: FrameRingBuffer,
        pub_socket: Any,  # zmq.Socket PUB
        session_id: str,
        req_id: str,
        gen: int,
        interrupt_event: threading.Event,
        audio_pcm_int16: np.ndarray | None = None,  # full audio for A/V pairing
        sample_rate: int = SAMPLE_RATE,
        fps: float = FPS,
    ) -> None:
        super().__init__()
        self._ring = ring
        self._pub = pub_socket
        self._session_id = session_id
        self._req_id = req_id
        self._gen = gen
        self._interrupt = interrupt_event
        self._audio = audio_pcm_int16
        self._sample_rate = sample_rate
        self._fps = fps
        self._slot_counter = 0
        self._frame_total = 0

    # ---- hooks ----------------------------------------------------------

    def dispatch(self, idx: int, frames: np.ndarray, pts_ms: float) -> None:
        """Write each frame in the batch to SHM and notify the consumer.

        ``frames`` is ``[N, H, W, C]`` uint8 RGB24 from the decoder.
        ``idx``    is the first frame index in this batch.
        ``pts_ms`` is the media time of ``frames[0]`` in milliseconds.
        """
        if self._interrupt.is_set():
            raise _Cancelled("request interrupted")

        n = len(frames)
        for i in range(n):
            frame_idx = idx + i
            frame_pts = pts_ms + i * (1000.0 / self._fps)

            # Pair frame with its 40 ms audio slice (640 samples @ 16 kHz).
            audio_bytes = self._audio_for_frame(frame_idx)

            # Round-robin slot; the consumer must read before N more frames.
            slot_idx = self._slot_counter % self._ring.num_slots
            self._slot_counter += 1
            self._frame_total += 1

            img = frames[i]  # [H, W, C]
            h, w = img.shape[0], img.shape[1]

            self._ring.write_frame(slot_idx, frame_idx, frame_pts, img, audio_bytes)

            notif = IPCFrameNotification(
                session_id=self._session_id,
                gen=self._gen,
                slot_idx=slot_idx,
                frame_idx=frame_idx,
                pts_ms=frame_pts,
                width=w,
                height=h,
                audio_bytes_len=len(audio_bytes),
                is_last=False,  # set on the very last frame in close()
            )
            self._pub.send_string(notif.to_json())

    def close(self) -> None:
        """Send a final notification with ``is_last=True``."""
        # Resend the last notification with is_last flag — zero cost because
        # the consumer just needs the signal, not a new frame.
        notif = IPCFrameNotification(
            session_id=self._session_id,
            gen=self._gen,
            slot_idx=-1,  # sentinel: not a real slot
            frame_idx=self._frame_total,
            pts_ms=0.0,
            width=0,
            height=0,
            audio_bytes_len=0,
            is_last=True,
        )
        self._pub.send_string(notif.to_json())

    # ---- helpers --------------------------------------------------------

    def _audio_for_frame(self, frame_idx: int) -> bytes:
        """Return the 40 ms (1280-byte) PCM16 slice for ``frame_idx``.

        Returns silence (zeros) when there is no audio or the slice falls
        past the end of the buffer.
        """
        if self._audio is None:
            return b"\x00" * (SAMPLES_PER_FRAME * 2)
        start = frame_idx * SAMPLES_PER_FRAME
        end = start + SAMPLES_PER_FRAME
        if start >= len(self._audio):
            return b"\x00" * (SAMPLES_PER_FRAME * 2)
        chunk = self._audio[start:end]
        if len(chunk) < SAMPLES_PER_FRAME:
            chunk = np.pad(chunk, (0, SAMPLES_PER_FRAME - len(chunk)))
        return chunk.astype("<i2").tobytes()


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


class InferenceWorker:
    """ZeroMQ-based FLOAT inference server.

    Lifecycle::

        worker = InferenceWorker(cfg)
        worker.start()  # blocks until SIGINT / SIGTERM / stop()

    Parameters
    ----------
    cfg:
        Namespace / object with the same attributes that ``InferenceOptions``
        provides (ckpt_path, rank, optimization_mode, use_fp16, fps, …).
    control_port:
        TCP port for the REP control socket.
    frames_port:
        TCP port for the PUB frame-notification socket.
    shm_name:
        Name of the shared-memory ring buffer (must match the consumer).
    shm_slots:
        Number of slots in the ring buffer (must match the consumer).
    """

    def __init__(
        self,
        cfg: Any,
        control_port: int = DEFAULT_CONTROL_PORT,
        frames_port: int = DEFAULT_FRAMES_PORT,
        shm_name: str = DEFAULT_SHM_NAME,
        shm_slots: int = DEFAULT_SHM_SLOTS,
    ) -> None:
        if zmq is None:
            raise RuntimeError("pyzmq is not installed; run: pip install pyzmq")

        self._cfg = cfg
        self._control_port = control_port
        self._frames_port = frames_port
        self._shm_name = shm_name
        self._shm_slots = shm_slots

        self._agent: Any = None  # OptimizedInferenceAgent (loaded in start())
        self._ring: FrameRingBuffer | None = None
        self._interrupt = threading.Event()
        self._shutdown = threading.Event()

        # Keep the last ref image + audio so prepare_silence can reuse them.
        self._last_ref_path: str = ""
        self._last_audio_pcm: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Load the model, open sockets, enter the request loop."""
        self._load_model()
        self._ring = FrameRingBuffer(
            name=self._shm_name,
            num_slots=self._shm_slots,
            create=True,
        )

        ctx = zmq.Context()
        rep = ctx.socket(zmq.REP)
        pub = ctx.socket(zmq.PUB)

        rep.bind(f"tcp://127.0.0.1:{self._control_port}")
        pub.bind(f"tcp://127.0.0.1:{self._frames_port}")
        # Give subscribers a moment to connect before the first publish.
        time.sleep(0.05)

        logger.info(
            "[worker] FLOAT worker listening on tcp://127.0.0.1:%d (ctrl) + tcp://127.0.0.1:%d (frames)",
            self._control_port,
            self._frames_port,
        )

        try:
            self._request_loop(rep, pub)
        finally:
            rep.close()
            pub.close()
            ctx.term()
            if self._ring is not None:
                self._ring.close()
                self._ring.unlink()
            logger.info("[worker] shutdown complete")

    def stop(self) -> None:
        """Ask the request loop to exit after the current request."""
        self._shutdown.set()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Import and initialise OptimizedInferenceAgent (heavy; GPU alloc)."""
        from fast_float.core.inference_optimized import OptimizedInferenceAgent

        logger.info("[worker] loading FLOAT model …")
        t0 = time.perf_counter()
        self._agent = OptimizedInferenceAgent(self._cfg)
        elapsed = time.perf_counter() - t0
        logger.info("[worker] model ready in %.2f s", elapsed)

    def _request_loop(self, rep: Any, pub: Any) -> None:
        """Poll the REP socket and handle one request at a time."""
        # Use a short poll timeout so we can check _shutdown periodically.
        poller = zmq.Poller()
        poller.register(rep, zmq.POLLIN)

        while not self._shutdown.is_set():
            events = dict(poller.poll(timeout=200))  # ms
            if rep not in events:
                continue

            raw = rep.recv()
            try:
                req = IPCRequest.from_json(raw)
            except Exception as exc:
                logger.warning("[worker] malformed request: %s", exc)
                rep.send_string(IPCResponse(status="error", error_msg=str(exc)).to_json())
                continue

            try:
                resp = self._handle(req, pub)
            except Exception as exc:
                logger.exception("[worker] unhandled error during %s", req.cmd)
                resp = IPCResponse(
                    status="error",
                    req_id=req.req_id,
                    session_id=req.session_id,
                    error_msg=str(exc),
                )

            rep.send_string(resp.to_json())

    def _handle(self, req: IPCRequest, pub: Any) -> IPCResponse:
        """Dispatch a single IPC request."""
        cmd = req.cmd

        if cmd == "infer":
            return self._handle_infer(req, pub)

        if cmd == "interrupt":
            self._interrupt.set()
            return IPCResponse(status="ok", req_id=req.req_id, session_id=req.session_id)

        if cmd == "prepare_silence":
            return self._handle_prepare_silence(req, pub)

        if cmd == "status":
            return self._handle_status(req)

        if cmd == "shutdown":
            self._shutdown.set()
            return IPCResponse(status="ok", req_id=req.req_id, session_id=req.session_id)

        return IPCResponse(
            status="error",
            req_id=req.req_id,
            session_id=req.session_id,
            error_msg=f"unknown cmd: {cmd!r}",
        )

    def _handle_infer(self, req: IPCRequest, pub: Any) -> IPCResponse:
        """Decode audio, run FLOAT, write frames into SHM, notify via PUB."""
        # Clear any stale interrupt from a previous request.
        self._interrupt.clear()

        # ---- decode audio ----
        pcm_bytes = base64.b64decode(req.audio_pcm) if req.audio_pcm else b""
        pcm_int16 = (
            np.frombuffer(pcm_bytes, dtype="<i2").copy()
            if pcm_bytes
            else np.zeros(req.valid_samples or SAMPLE_RATE * 2, dtype="<i2")
        )
        self._last_audio_pcm = pcm_int16

        # ---- write temp WAV ----
        wav_fd, wav_path = tempfile.mkstemp(suffix=".wav", prefix="float_ipc_")
        try:
            os.close(wav_fd)
            _write_wav(wav_path, pcm_int16, req.sample_rate)

            ref_path = req.ref_image_path or self._last_ref_path
            if not ref_path:
                return IPCResponse(
                    status="error",
                    req_id=req.req_id,
                    session_id=req.session_id,
                    error_msg="ref_image_path is required for first infer request",
                )
            self._last_ref_path = ref_path

            dispatcher = _ShmFrameDispatcher(
                ring=self._ring,
                pub_socket=pub,
                session_id=req.session_id,
                req_id=req.req_id,
                gen=req.gen,
                interrupt_event=self._interrupt,
                audio_pcm_int16=pcm_int16,
                sample_rate=req.sample_rate,
                fps=getattr(self._cfg, "fps", FPS),
            )

            t0 = time.perf_counter()
            try:
                self._agent.run_inference(
                    res_video_path=None,
                    ref_path=ref_path,
                    audio_path=wav_path,
                    nfe=req.nfe,
                    verbose=False,
                    dispatcher=dispatcher,
                )
            except _Cancelled:
                logger.info("[worker] inference interrupted (session=%s)", req.session_id)
                dispatcher.shutdown(timeout=1.0)
                return IPCResponse(
                    status="interrupted",
                    req_id=req.req_id,
                    session_id=req.session_id,
                    total_frames=dispatcher._frame_total,
                )
            finally:
                dispatcher.shutdown(timeout=2.0)

            elapsed = time.perf_counter() - t0
            audio_dur = len(pcm_int16) / req.sample_rate
            rtf = elapsed / audio_dur if audio_dur > 0 else 0.0

            logger.info(
                "[worker] infer done session=%s frames=%d rtf=%.2fx",
                req.session_id,
                dispatcher._frame_total,
                rtf,
            )
            return IPCResponse(
                status="ok",
                req_id=req.req_id,
                session_id=req.session_id,
                total_frames=dispatcher._frame_total,
                rtf=rtf,
            )
        finally:
            with contextlib.suppress(OSError):
                os.unlink(wav_path)

    def _handle_prepare_silence(self, req: IPCRequest, pub: Any) -> IPCResponse:
        """Pre-render a silence loop (reuses the last ref image).

        The frames are published over PUB just like normal infer frames so the
        consumer can cache them for idle playback.
        """
        ref_path = req.ref_image_path or self._last_ref_path
        if not ref_path:
            return IPCResponse(
                status="error",
                req_id=req.req_id,
                session_id=req.session_id,
                error_msg="no ref_image_path available for prepare_silence",
            )

        # Silence is a WAV of zeros at the configured duration.
        silence_sec = getattr(self._cfg, "livekit_silence_sec", 10.0)
        n_samples = int(silence_sec * SAMPLE_RATE)
        silence_pcm = np.zeros(n_samples, dtype="<i2")

        wav_fd, wav_path = tempfile.mkstemp(suffix=".wav", prefix="float_silence_")
        try:
            os.close(wav_fd)
            _write_wav(wav_path, silence_pcm, SAMPLE_RATE)

            self._interrupt.clear()
            dispatcher = _ShmFrameDispatcher(
                ring=self._ring,
                pub_socket=pub,
                session_id=req.session_id,
                req_id=req.req_id,
                gen=req.gen,
                interrupt_event=self._interrupt,
                audio_pcm_int16=silence_pcm,
            )
            try:
                self._agent.run_inference(
                    res_video_path=None,
                    ref_path=ref_path,
                    audio_path=wav_path,
                    nfe=req.nfe or getattr(self._cfg, "livekit_nfe", 5),
                    verbose=False,
                    dispatcher=dispatcher,
                )
            except _Cancelled:
                pass
            finally:
                dispatcher.shutdown(timeout=2.0)

            return IPCResponse(
                status="ok",
                req_id=req.req_id,
                session_id=req.session_id,
                total_frames=dispatcher._frame_total,
            )
        finally:
            with contextlib.suppress(OSError):
                os.unlink(wav_path)

    def _handle_status(self, req: IPCRequest) -> IPCResponse:
        """Return basic worker / VRAM status."""
        info: dict[str, Any] = {"model_ready": self._agent is not None}
        try:
            import torch

            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                free, total = torch.cuda.mem_get_info(0)
                info["gpu"] = props.name
                info["vram_free_mb"] = free // (1024 * 1024)
                info["vram_total_mb"] = total // (1024 * 1024)
        except Exception:
            pass
        import json

        return IPCResponse(
            status="ok",
            req_id=req.req_id,
            session_id=req.session_id,
            error_msg=json.dumps(info),  # piggyback JSON in error_msg field
        )


# ---------------------------------------------------------------------------
# WAV helper
# ---------------------------------------------------------------------------


def _write_wav(path: str, pcm_int16: np.ndarray, sample_rate: int) -> None:
    """Write a mono PCM-16 WAV file to *path*."""
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_int16.astype("<i2").tobytes())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _build_cfg():
    """Build a config namespace from settings.toml + InferenceOptions defaults."""
    try:
        from fast_float.config.float_settings import load_float_options

        opt = load_float_options()
        opt.rank = 0
        return opt
    except Exception as exc:
        logger.warning(
            "Could not load float options via load_float_options (%s); falling back to InferenceOptions", exc
        )
        from fast_float.core.options import InferenceOptions

        opt = InferenceOptions().parse()
        opt.rank = 0
        return opt


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Read IPC settings from env / defaults.
    control_port = int(os.environ.get("FLOAT_ZMQ_CONTROL_PORT", DEFAULT_CONTROL_PORT))
    frames_port = int(os.environ.get("FLOAT_ZMQ_FRAMES_PORT", DEFAULT_FRAMES_PORT))
    shm_name = os.environ.get("FLOAT_SHM_NAME", DEFAULT_SHM_NAME)
    shm_slots = int(os.environ.get("FLOAT_SHM_SLOTS", DEFAULT_SHM_SLOTS))

    cfg = _build_cfg()
    worker = InferenceWorker(
        cfg,
        control_port=control_port,
        frames_port=frames_port,
        shm_name=shm_name,
        shm_slots=shm_slots,
    )

    # Graceful shutdown on SIGINT / SIGTERM.
    def _sig_handler(sig, frame):
        logger.info("[worker] caught signal %s — shutting down …", sig)
        worker.stop()

    signal.signal(signal.SIGINT, _sig_handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _sig_handler)

    worker.start()


if __name__ == "__main__":
    main()
