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
import functools
import logging
import math
import os
import uuid
from collections.abc import AsyncIterator
from typing import Any

import numpy as np

from fast_float.core.state import app_state
from fast_float.utils.gpu import estimate_stream_capacity

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

# How long to wait for the worker's shared-memory ring at attach time (sec).
# The worker creates it only after loading the model, so a backend that is
# already serving can be up before it.
DEFAULT_SHM_WAIT_S = 15.0

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
        shm_wait_s: float | None = None,
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

        # How long to wait for the worker's segment to appear. The worker owns
        # it and creates it only after loading the model, so a backend that is
        # already up races with a worker that is still booting.
        if shm_wait_s is None:
            raw_wait = os.environ.get("FLOAT_SHM_WAIT_S", "").strip()
            try:
                shm_wait_s = float(raw_wait) if raw_wait else DEFAULT_SHM_WAIT_S
            except ValueError:
                shm_wait_s = DEFAULT_SHM_WAIT_S
        self._shm_wait_s = max(0.0, float(shm_wait_s))

        # Asyncio + ZMQ context (created in _ensure_connected)
        self._zmq_ctx: Any = None
        self._req_sock: Any = None  # DEALER/REQ to control port
        self._sub_sock: Any = None  # SUB to frames port

        # REQ/REQ sockets are strictly stateful (send -> reply -> send ...):
        # control requests MUST be serialized, or the second in-flight
        # ``send_string`` raises ZMQError EFSM and the chunk is silently
        # dropped — every chunk submitted while the previous one still waits
        # for its reply used to vanish ("avatar talks the first 5 s chunk,
        # then stops").
        self._req_lock: asyncio.Lock | None = None

        # Shared memory (attached once, consumer side)
        self._ring: FrameRingBuffer | None = None

        # Audio accumulation
        self._audio_buf: list[bytes] = []
        self._audio_samples: int = 0
        self._chunk_samples = int(SAMPLE_RATE * chunk_sec)
        self._first_chunk_samples = int(SAMPLE_RATE * first_chunk_sec)
        self._first_chunk_done = False

        # Generation counter — incremented each push cycle; `_sentinel_gen` is
        # the last generation the worker finished (its ``is_last`` sentinel).
        # The idle event mirrors FloatVideoGenerator: it starts set ("nothing
        # outstanding") and the agent runner's ``wait_until_idle`` awaits it.
        self._gen = 0
        self._sentinel_gen = 0
        self._idle_event = asyncio.Event()
        self._idle_event.set()  # nothing outstanding
        self._stopped = False

        # Highest generation abandoned by ``clear_buffer()`` (barge-in). The
        # worker only reads the interrupt once the chunk it is *already*
        # rendering is done (its REP loop is blocked inside that infer), so the
        # cancelled chunk's frames keep arriving; they must never reach the
        # playout queue or the interrupted sentence keeps talking over the next
        # reply.
        self._cleared_gen = 0
        # Frames a generation really covers. The worker renders whole windows
        # (``wav2vec_sec * fps`` frames), so a chunk is padded up to the next
        # window boundary with silence; the in-process generator trims those
        # with ``valid_samples`` and this path has to do the same, or every
        # chunk ends with up to 2 s of silent video and the avatar drifts
        # behind the conversation.
        self._valid_frames: dict[int, int] = {}
        # Frames dropped because the consumer could not keep up (diagnostics).
        self._dropped_frames = 0

        # Frame output queue (filled by _frame_listener_task).
        self._frame_queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=256)

        # Pre-encoded silence frames (list of (rtc.VideoFrame, rtc.AudioFrame)).
        self._silence_frames: list[Any] = []
        self._silence_ready = asyncio.Event()
        # While paused, ``__aiter__`` stops emitting silence frames and waits for
        # the next real one. The agent runner pauses the idle loop while it drains
        # `AVSynchronizer`: continuous idle frames keep the playout queue non-empty,
        # so ``wait_for_playout()`` could only ever time out.
        self._idle_paused = False

        # Last real-time factor reported by the worker (render seconds per
        # audio second), used once to report the measured stream capacity.
        self.last_rtf: float | None = None
        self._capacity_logged = False

        # Silence collection state. While ``_collecting`` is True the frame
        # listener routes THIS session's gen-0 notifications into
        # ``_silence_collected`` instead of the live queue; ``_silence_done``
        # fires on the ``is_last`` sentinel. The SUB socket must have exactly
        # one reader — a second receiver (e.g. an inline loop here) would steal
        # messages from the listener and starve one side into a timeout.
        self._collecting = False
        self._silence_collected: list[Any] = []
        self._silence_done = asyncio.Event()

        # Background tasks
        self._listener_task: asyncio.Task | None = None
        self._background_tasks: set[asyncio.Task] = set()

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------

    async def _ensure_connected(self) -> None:
        """Lazily create ZMQ context, sockets, and SHM attachment.

        Every resource is built locally and committed at the end. A failed
        attach (worker not started, still loading, or crashed) must not leave
        ``_zmq_ctx`` set: the old code assigned it first, so the *next* call
        returned early with no ring, no sockets and no frame listener while
        looking "connected".
        """
        if self._zmq_ctx is not None:
            return

        ctx = azmq.Context()
        req = sub = None

        try:
            # REQ socket for synchronous control commands.
            req = ctx.socket(zmq.REQ)
            req.connect(self._control_ep)

            # SUB socket for per-frame notifications from the worker.
            sub = ctx.socket(zmq.SUB)
            sub.connect(self._frames_ep)
            sub.setsockopt(zmq.SUBSCRIBE, b"")  # receive all topics

            # Attach to the shared-memory ring (the worker owns it). Blocking
            # poll, so run it off the loop: it can wait for a booting worker.
            ring = await asyncio.to_thread(
                functools.partial(
                    FrameRingBuffer.attach,
                    name=self._shm_name,
                    num_slots=self._shm_slots,
                    timeout=self._shm_wait_s,
                )
            )
        except BaseException:
            with contextlib.suppress(Exception):
                if req is not None:
                    req.close(linger=0)
                if sub is not None:
                    sub.close(linger=0)
                ctx.term()
            raise

        self._zmq_ctx = ctx
        self._req_sock = req
        self._sub_sock = sub
        self._ring = ring

        # Start the background frame-listener.
        self._listener_task = asyncio.create_task(self._frame_listener(), name="zmq-frame-listener")
        # Bound to this event loop (started here); bind it on the same thread
        # that runs the request dispatchers.
        self._req_lock = asyncio.Lock()
        logger.info(
            "[zmq-gen] connected session=%s ctrl=%s frames=%s shm=%s",
            self._session_id,
            self._control_ep,
            self._frames_ep,
            self._shm_name,
        )

    async def _send_request(self, req: IPCRequest) -> IPCResponse:
        """Send a synchronous IPC request and await the reply.

        Serialized: a REQ socket cannot accept a second request before the
        previous reply was received, and the control channel carries infer
        requests *and* interrupts that can overlap while audio keeps streaming.
        """
        if self._req_lock is None:
            # _ensure_connected was bypassed by a stub in a test; run unlocked.
            await self._req_sock.send_string(req.to_json())
            return IPCResponse.from_json(await self._req_sock.recv())
        async with self._req_lock:
            await self._req_sock.send_string(req.to_json())
            raw = await self._req_sock.recv()
            return IPCResponse.from_json(raw)

    # ------------------------------------------------------------------
    # Silence pre-rendering
    # ------------------------------------------------------------------

    async def prepare_silence(self, duration_sec: float = 2.0) -> None:
        """Ask the worker to pre-render a silence loop and cache the frames.

        Must be called once per session before the first ``push_audio()`` /
        ``__aiter__`` usage to populate ``_silence_frames``. Idempotent — a
        repeated call after a successful prepare is a no-op.

        ``duration_sec`` is the idle-loop length the worker renders. Frames
        arriving while the request is in flight are collected by the frame
        listener (the SUB socket's single reader): collection mode is switched
        on BEFORE the request is sent, so no notification can be misrouted.
        """
        await self._ensure_connected()
        if self._silence_ready.is_set():
            return

        self._silence_collected.clear()
        self._silence_done.clear()
        self._collecting = True
        try:
            req = IPCRequest(
                cmd="prepare_silence",
                session_id=self._session_id,
                req_id=str(uuid.uuid4()),
                ref_image_path=self._ref_image_path,
                gen=0,
                duration_sec=max(0.0, float(duration_sec or 0.0)),
            )
            resp = await self._send_request(req)
            if resp.status != "ok":
                logger.warning("[zmq-gen] prepare_silence failed: %s", resp.error_msg)
                return

            # The ``is_last`` sentinel (published by the worker) fires
            # ``_silence_done`` through the frame listener.
            # NOTE: catch ``asyncio.TimeoutError``, *not* the builtin
            # ``TimeoutError`` — they are distinct classes on Python 3.10, so
            # ``except TimeoutError`` would let the session restart crash.
            try:
                await asyncio.wait_for(self._silence_done.wait(), timeout=30.0)
            except asyncio.TimeoutError:
                logger.warning(
                    "[zmq-gen] timeout waiting for silence frames (got %d/%d)",
                    len(self._silence_collected),
                    resp.total_frames,
                )
        finally:
            self._collecting = False

        self._silence_frames = list(self._silence_collected)
        self._silence_ready.set()
        logger.info("[zmq-gen] %d silence frames cached", len(self._silence_frames))

    # ------------------------------------------------------------------
    # Audio input
    # ------------------------------------------------------------------

    def push_audio(self, audio: Any) -> None:
        """Buffer a synthesized-audio chunk from the agent bridge.

        Accepts the same payload types as ``FloatVideoGenerator``: raw 16 kHz
        mono PCM16 bytes (the direct bridge), ``rtc.AudioFrame`` (the
        ``DataStreamAudioReceiver`` output the Gemini runner pushes — arrives
        at the agent's sample rate, e.g. 24 kHz) and numpy arrays. Everything
        is normalized to 16 kHz mono PCM16; a bare ``len()`` on an
        ``rtc.AudioFrame`` used to kill the runner's audio loop ("object of
        type 'AudioFrame' has no len()").

        When enough audio is accumulated (``chunk_sec`` threshold), an async
        infer request is queued.
        """
        pcm_bytes = self._to_pcm16_mono(audio)
        self._audio_buf.append(pcm_bytes)
        self._audio_samples += len(pcm_bytes) // 2  # int16 → samples

        threshold = self._first_chunk_samples if not self._first_chunk_done else self._chunk_samples

        if self._audio_samples >= threshold:
            chunk_bytes = b"".join(self._audio_buf)
            self._audio_buf.clear()
            self._audio_samples = 0
            self._first_chunk_done = True
            self._submit_chunk(chunk_bytes)

    def _submit_chunk(self, chunk_bytes: bytes) -> None:
        """Send a buffered chunk to the worker as the next generation."""
        self._gen += 1
        n_samples = len(chunk_bytes) // 2
        # Frames covering real audio. Anything past this index is FLOAT's
        # replicate-pad tail for the last 2 s window: silence over a still
        # face. ``frame_idx`` is 0-based per chunk (each request gets its own
        # FrameDispatcher, whose cursor starts at ``StreamMeta.first_idx`` = 0).
        self._valid_frames[self._gen] = math.ceil(n_samples / SAMPLES_PER_FRAME) if n_samples else 0
        self._idle_event.clear()  # work now outstanding for the runner

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

    def flush(self) -> None:
        """Dispatch buffered tail audio even when below the chunk threshold.

        Called at a TTS segment's end (AgentAvatarRunner) so the last chunk is
        rendered instead of stranded until the next threshold crossing.
        """
        if self._stopped or not self._audio_buf:
            return
        chunk_bytes = b"".join(self._audio_buf)
        self._audio_buf.clear()
        self._audio_samples = 0
        self._first_chunk_done = True
        self._submit_chunk(chunk_bytes)

    def _to_pcm16_mono(self, audio: Any) -> bytes:
        """Normalize an inbound audio item to 16 kHz mono PCM16 bytes.

        Mirrors ``FloatVideoGenerator._to_pcm16_mono``: raw bytes pass through;
        ``rtc.AudioFrame`` is downmixed (mono-converted) and resampled from the
        agent's sample rate to the worker's 16 kHz input.
        """
        if RTC_AVAILABLE and isinstance(audio, rtc.AudioFrame):
            samples = np.frombuffer(audio.data, dtype=np.int16)
            ch, sr = audio.num_channels, audio.sample_rate
            if ch > 1:
                samples = samples.reshape(-1, ch).mean(axis=1)
            if sr != SAMPLE_RATE:
                samples = self._resample(samples.astype(np.float32), sr)
            return np.asarray(samples).astype(np.int16).tobytes()
        if isinstance(audio, (bytes, bytearray, memoryview)):
            return bytes(audio)
        if isinstance(audio, np.ndarray):
            if audio.dtype in (np.float32, np.float64):
                return (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
            return audio.astype(np.int16).tobytes()
        raise TypeError(f"Unsupported audio type: {type(audio)}")

    def _resample(self, x: np.ndarray, src_sr: int) -> np.ndarray:
        """Cheap resample to the worker's SAMPLE_RATE.

        Integer ratios (48k -> 16k) use box-filter decimation, which also
        low-passes; otherwise fall back to linear interpolation.
        """
        if src_sr % SAMPLE_RATE == 0:
            r = src_sr // SAMPLE_RATE
            n = len(x) // r * r
            return x[:n].reshape(-1, r).mean(axis=1)
        n_out = int(len(x) * SAMPLE_RATE / src_sr)
        return np.interp(np.linspace(0, len(x) - 1, n_out), np.arange(len(x)), x)

    async def _dispatch_infer(self, pcm_bytes: bytes, gen: int) -> None:
        """Send an infer request to the worker for the given audio chunk.

        Runs as a fire-and-forget task, so failure MUST be accounted for here:
        otherwise ``_idle_event`` stays cleared („work outstanding" that will
        never finish) and the runner's ``wait_until_idle`` waits out the full
        timeout, freezing the segment watcher mid-conversation.
        """
        try:
            await self._ensure_connected()
        except Exception as exc:
            logger.warning("[zmq-gen] connect failed before infer (gen=%d): %s", gen, exc)
            self._account_failed_gen(gen)
            return

        pcm_b64 = base64.b64encode(pcm_bytes).decode()
        n_samples = len(pcm_bytes) // 2

        if gen <= self._cleared_gen:
            # A barge-in landed while this chunk was waiting for the control
            # socket: its audio belongs to the interrupted sentence, so there is
            # nothing to render (and its frames would be dropped anyway).
            logger.debug("[zmq-gen] dropping stale infer (gen=%d)", gen)
            return

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
        try:
            resp = await self._send_request(req)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # EFSM (overlapping request), dead socket, worker restart … the
            # chunk is lost either way — log it and free the runner instead of
            # wedging idle forever.
            logger.warning("[zmq-gen] infer dispatch failed (gen=%d): %s", gen, exc)
            self._account_failed_gen(gen)
            return

        if resp.status == "ok":
            rtf = float(getattr(resp, "rtf", 0.0) or 0.0)
            logger.info(
                "[zmq-gen] infer ok (gen=%d frames=%s rtf=%.2fx)",
                gen,
                getattr(resp, "total_frames", -1) or -1,
                rtf,
            )
            if rtf > 0:
                self._record_rtf(rtf)
        else:
            logger.warning("[zmq-gen] infer failed (gen=%d): %s", gen, resp.error_msg)

    def _record_rtf(self, rtf: float) -> None:
        """Remember the worker's measured render speed and report capacity once.

        The first real render is the only honest measurement of how many
        streams this GPU can sustain: VRAM alone says 8, but a stream that
        renders slower than real time cannot be served at all. The estimate is
        published in ``app_state`` so ``GET /gpu`` can report it without
        importing the LiveKit stack.
        """
        self.last_rtf = float(rtf)
        app_state.live_rtf = float(rtf)
        if self._capacity_logged:
            return
        self._capacity_logged = True
        logger.info(
            "[zmq-gen] measured live capacity: %s",
            estimate_stream_capacity(observed_rtf=rtf).summary(),
        )

    def _account_failed_gen(self, gen: int) -> None:
        """Treat a generation that will never produce a sentinel as completed.

        Marks it done in the idle bookkeeping so ``wait_until_idle`` resolves
        (bounded by the queue drain) instead of hanging until the timeout. The
        speech in that chunk is already lost — playback of the rest continues.
        """
        self._sentinel_gen = max(self._sentinel_gen, gen)
        self._refresh_idle()

    def clear_buffer(self) -> None:
        """Discard buffered audio and abandon the in-flight generation."""
        self._audio_buf.clear()
        self._audio_samples = 0
        self._first_chunk_done = False

        # A barge-in abandons every generation submitted so far: those frames
        # belong to the interrupted sentence. The worker cannot be stopped
        # mid-chunk (its REP loop is blocked inside the infer it is already
        # running, so ``interrupt`` is only read once that chunk is done), hence
        # the frames still arrive — mark the watermark so the listener drops
        # them, and drain what is already queued. Without this the interrupted
        # reply kept playing for up to ~10 s while the next reply's frames
        # queued up behind it, which sounds like two chunks overlapping.
        self._cleared_gen = self._gen
        for gen in [g for g in self._valid_frames if g <= self._cleared_gen]:
            del self._valid_frames[gen]
        dropped = self._drain_frames()
        if dropped:
            logger.info(
                "[zmq-gen] barge-in: dropped %d queued frames (gen<=%d)",
                dropped,
                self._cleared_gen,
            )
        # The abandoned generations will never deliver a usable sentinel, so
        # account them as finished: ``wait_until_idle`` would otherwise stall
        # for its full timeout after every barge-in, and the segment watcher
        # would never report playback_finished to the agent.
        self._sentinel_gen = max(self._sentinel_gen, self._cleared_gen)
        self._refresh_idle()

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

    def _drain_frames(self) -> int:
        """Drop every queued (video, audio) pair; returns how many were dropped."""
        dropped = 0
        while True:
            try:
                self._frame_queue.get_nowait()
            except asyncio.QueueEmpty:
                return dropped
            dropped += 1

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

    def _read_frame_pair(self, notif: IPCFrameNotification) -> tuple[Any, Any] | None:
        """Load a (VideoFrame, AudioFrame) pair for a notification from SHM."""
        if self._ring is None:
            return None
        _, _, img_arr, audio_bytes = self._ring.read_frame(notif.slot_idx)
        vf = _make_video_frame(img_arr)
        af = _make_audio_frame(audio_bytes)
        if vf is None or af is None:
            return None
        return vf, af

    def _is_live_frame(self, notif: IPCFrameNotification) -> bool:
        """True when ``notif`` carries a frame of the current, unabandoned chunk.

        Mirrors ``FloatVideoGenerator``'s two guards:

        * ``_cleared_gen`` — a barge-in abandons every generation submitted
          before it. The worker finishes the chunk it is already rendering, so
          its frames still arrive and must not be played over the new reply.
        * ``_valid_frames`` — the tail of a chunk past the real audio is
          FLOAT's replicate-pad silence and is trimmed exactly like the
          in-process bridge's ``valid_samples`` cut.
        """
        if notif.gen > 0 and notif.gen <= self._cleared_gen:
            return False
        limit = self._valid_frames.get(notif.gen)
        return limit is None or notif.frame_idx < limit

    async def _frame_listener(self) -> None:
        """Receive IPCFrameNotifications from the PUB socket and enqueue frames.

        Sole consumer of ``_sub_sock`` — a second reader would siphon off
        messages nondeterministically, e.g. swallow the ``is_last`` sentinel
        that ends silence pre-rendering.
        """
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

            # Silence pre-render: while collecting, route the worker's gen-0
            # replay into the silence bucket instead of the live queue.
            if self._collecting and notif.gen == 0:
                if notif.is_last:
                    self._collecting = False
                    self._silence_done.set()
                    continue
                pair = self._read_frame_pair(notif)
                if pair is not None:
                    self._silence_collected.append(pair)
                continue

            if notif.is_last:
                # A generation finished rendering server-side. Track the
                # highest completed generation; the idle event resolves only
                # when the live queue has drained too (checked here and from
                # the iterator, which re-runs while the consumer paces).
                self._sentinel_gen = max(self._sentinel_gen, notif.gen)
                self._valid_frames.pop(notif.gen, None)
                self._refresh_idle()
                continue

            # Stale (barge-in) frames and the padded tail of a chunk never
            # reach the playout queue.
            if not self._is_live_frame(notif):
                continue

            pair = self._read_frame_pair(notif)
            if pair is None:
                continue

            try:
                self._frame_queue.put_nowait(pair)
            except asyncio.QueueFull:
                # Drop the oldest to maintain latency.
                with contextlib.suppress(asyncio.QueueEmpty):
                    self._frame_queue.get_nowait()
                self._dropped_frames += 1
                if self._dropped_frames % 25 == 1:
                    logger.warning(
                        "[zmq-gen] consumer behind; dropped %d frames total",
                        self._dropped_frames,
                    )
                self._frame_queue.put_nowait(pair)

    # ------------------------------------------------------------------
    # Idle tracking (AgentAvatarRunner.wait_until_idle)
    # ------------------------------------------------------------------

    def _refresh_idle(self) -> None:
        """Idle == every submitted generation has rendered and the live queue is drained."""
        if self._gen == self._sentinel_gen and self._frame_queue.empty():
            self._idle_event.set()
        else:
            self._idle_event.clear()

    async def wait_until_idle(self, timeout: float | None = None) -> bool:
        """Wait until every pushed sample has been generated and consumed.

        Returns ``True`` when idle, ``False`` on timeout. Used by the agent
        runner to know when a TTS segment has finished rendering, so it can
        notify the agent that playback is over. On Python 3.10,
        ``asyncio.wait_for`` raises ``asyncio.TimeoutError``, which is distinct
        from the builtin ``TimeoutError`` until 3.11.
        """
        try:
            await asyncio.wait_for(self._idle_event.wait(), timeout)
        except asyncio.TimeoutError:
            return False
        return True

    def pause_idle(self) -> None:
        """Stop emitting silence frames so the playout queue can drain.

        Real frames queued by the worker are still yielded immediately; only
        the idle loop is suspended.
        """
        self._idle_paused = True

    def resume_idle(self) -> None:
        """Resume emitting silence frames (undo :meth:`pause_idle`)."""
        self._idle_paused = False

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
            # The consumer re-enters this loop at 25 fps while pacing, so this
            # is where idle (queue drained after the last sentinel) resolves.
            self._refresh_idle()

            # Try to get a live frame (non-blocking).
            try:
                pair = self._frame_queue.get_nowait()
                yield pair
                continue
            except asyncio.QueueEmpty:
                pass

            if self._idle_paused:
                # Idle loop paused (playout drain in progress): block until a real
                # frame arrives instead of feeding silence to AVSynchronizer.
                try:
                    pair = await asyncio.wait_for(self._frame_queue.get(), timeout=frame_interval)
                except asyncio.TimeoutError:
                    continue
                yield pair
                continue

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
