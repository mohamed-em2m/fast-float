"""FloatVideoGenerator: audio in -> synchronized (VideoFrame, AudioFrame) out.

Implements the custom avatar video generator interface for LiveKit.
FLOAT conditions on a 2-second audio window (50 frames at 25 fps, 16 kHz audio)
and is autoregressive over chunks.

This module provides:
1. `FloatVideoGenerator`: generates video from live audio input in 2.0 s chunks,
   pairing each 40 ms video frame with its exact 40 ms audio slice (640 samples @ 16 kHz)
   for drift-free A/V sync.
2. `LiveStreamVideoGenerator`: taps an in-flight `LiveStream` from `/generate`,
   pairing decoded frames with audio from `stream.audio_path`.
3. `MediaFileVideoGenerator`: plays back an existing rendered MP4 file into LiveKit.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import tempfile
from collections.abc import AsyncIterator
from typing import Any

import numpy as np

try:
    from livekit import rtc

    RTC_AVAILABLE = True
except ImportError:  # pragma: no cover
    rtc = None  # type: ignore[assignment]
    RTC_AVAILABLE = False

from fast_float.streaming.dispatcher import FrameDispatcher
from fast_float.streaming.live import CLOSE, LiveStream

logger = logging.getLogger(__name__)

TARGET_FPS = 25.0
TARGET_WIDTH = 512
TARGET_HEIGHT = 512
SAMPLE_RATE = 16000
MIN_AUDIO_SEC = 2.0
MIN_AUDIO_SAMPLES = int(SAMPLE_RATE * MIN_AUDIO_SEC)  # 32,000 samples for 2s
SAMPLES_PER_FRAME = int(SAMPLE_RATE / TARGET_FPS)  # 640 samples per 40ms frame
BYTES_PER_FRAME = SAMPLES_PER_FRAME * 2  # 16-bit mono = 1280 bytes

_QUEUE_SENTINEL = object()


class _LiveKitDispatcherBridge(FrameDispatcher):
    """Bridge FrameDispatcher events to an asyncio.Queue on the event loop."""

    def __init__(self, queue: asyncio.Queue, loop: asyncio.AbstractEventLoop) -> None:
        super().__init__()
        self._queue = queue
        self._loop = loop

    def dispatch(self, idx: int, frames: np.ndarray, pts_ms: float) -> None:
        if frames.ndim == 3:
            frames_to_push = [(idx, frames, pts_ms)]
        else:
            frames_to_push = [(idx + i, frames[i], pts_ms + (i * (1000.0 / TARGET_FPS))) for i in range(len(frames))]

        for item in frames_to_push:
            if not self._loop.is_closed():
                self._loop.call_soon_threadsafe(self._push_sync, item)

    def _push_sync(self, item: tuple[int, np.ndarray, float]) -> None:
        with contextlib.suppress(asyncio.QueueFull):
            self._queue.put_nowait(item)

    def close(self) -> None:
        if not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._close_sync)

    def _close_sync(self) -> None:
        with contextlib.suppress(asyncio.QueueFull):
            self._queue.put_nowait(_QUEUE_SENTINEL)


class FloatVideoGenerator:
    """Consumes audio chunks and yields synchronized (rtc.VideoFrame, rtc.AudioFrame) tuples."""

    def __init__(
        self,
        *,
        agent: Any = None,
        image_path: str | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
        width: int = TARGET_WIDTH,
        height: int = TARGET_HEIGHT,
        fps: float = TARGET_FPS,
        sample_rate: int = SAMPLE_RATE,
        semaphore: asyncio.Semaphore | None = None,
    ) -> None:
        self.agent = agent
        self.image_path = image_path
        # Prefer the running loop (inside async context) to avoid the
        # DeprecationWarning from get_event_loop() in Python >= 3.10.
        if loop is not None:
            self.loop = loop
        else:
            try:
                self.loop = asyncio.get_running_loop()
            except RuntimeError:
                # Called from sync context (e.g. tests); create a fresh loop.
                self.loop = asyncio.new_event_loop()
        self.width = width
        self.height = height
        self.fps = fps
        self.sample_rate = sample_rate
        self.semaphore = semaphore

        self._audio_pcm = bytearray()
        self._frame_queue: asyncio.Queue = asyncio.Queue(maxsize=128)
        self._current_chunk_audio: bytes | None = None
        self._stopped = False
        self._inference_running = False
        self._frame_count = 0
        self._first_frame_emitted = False
        self._bg_tasks: set[asyncio.Task[Any]] = set()
        self.on_first_frame: Any | None = None

    def push_audio(self, audio: Any) -> None:
        """Accumulate incoming audio (rtc.AudioFrame, raw bytes, or numpy array)."""
        if self._stopped:
            return

        if RTC_AVAILABLE and isinstance(audio, rtc.AudioFrame):
            data = bytes(audio.data)
        elif isinstance(audio, (bytes, bytearray, memoryview)):
            data = bytes(audio)
        elif isinstance(audio, np.ndarray):
            if audio.dtype in (np.float32, np.float64):
                data = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
            else:
                data = audio.astype(np.int16).tobytes()
        else:
            raise TypeError(f"Unsupported audio type: {type(audio)}")

        self._audio_pcm.extend(data)

        # Trigger chunk inference if we have >= 2.0 s of audio
        total_samples = len(self._audio_pcm) // 2
        if total_samples >= MIN_AUDIO_SAMPLES and not self._inference_running:
            self._schedule_chunk()

    def _schedule_chunk(self) -> None:
        if self._stopped or self._inference_running:
            return
        self._inference_running = True
        chunk_bytes = bytes(self._audio_pcm[: MIN_AUDIO_SAMPLES * 2])
        del self._audio_pcm[: MIN_AUDIO_SAMPLES * 2]
        self._current_chunk_audio = chunk_bytes

        task = self.loop.create_task(self._run_chunk_inference(chunk_bytes))
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)

    async def _run_chunk_inference(self, pcm_bytes: bytes) -> None:
        """Run FLOAT inference in a background thread and deliver frames."""
        try:
            if self.semaphore:
                await self.semaphore.acquire()
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
                    tmp_audio_path = tmp_audio.name
                    import wave

                    with wave.open(tmp_audio_path, "wb") as wav_file:
                        wav_file.setnchannels(1)
                        wav_file.setsampwidth(2)
                        wav_file.setframerate(self.sample_rate)
                        wav_file.writeframes(pcm_bytes)

                try:
                    bridge = _LiveKitDispatcherBridge(self._frame_queue, self.loop)
                    if self.agent is not None and self.image_path:
                        from fast_float.core.service import run_inference_blocking

                        req_data = {
                            "a_cfg_scale": 2.0,
                            "r_cfg_scale": 1.0,
                            "e_cfg_scale": 1.0,
                            "emotion": "S2E",
                            "nfe": 10,
                            "no_crop": False,
                            "seed": 25,
                        }
                        await self.loop.run_in_executor(
                            None,
                            run_inference_blocking,
                            self.agent,
                            None,
                            self.image_path,
                            tmp_audio_path,
                            req_data,
                            bridge,
                        )
                    else:
                        # Fallback test pattern: 50 frames
                        for i in range(50):
                            pts_ms = (self._frame_count + i) * (1000.0 / self.fps)
                            pattern = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                            pattern[:, :, 0] = (i * 5) % 255
                            pattern[:, :, 1] = 120
                            pattern[:, :, 2] = 180
                            await self._frame_queue.put((self._frame_count + i, pattern, pts_ms))
                finally:
                    if os.path.exists(tmp_audio_path):
                        with contextlib.suppress(OSError):
                            os.remove(tmp_audio_path)
            finally:
                if self.semaphore:
                    self.semaphore.release()
        except Exception as exc:
            logger.warning(f"[livekit-gen] inference error: {exc}")
        finally:
            self._inference_running = False
            if len(self._audio_pcm) // 2 >= MIN_AUDIO_SAMPLES:
                self._schedule_chunk()

    def flush(self) -> None:
        """Trigger generation on remaining audio in buffer."""
        if len(self._audio_pcm) > 0 and not self._inference_running:
            remaining_samples = len(self._audio_pcm) // 2
            if remaining_samples < MIN_AUDIO_SAMPLES:
                padding = bytes(MIN_AUDIO_SAMPLES * 2 - len(self._audio_pcm))
                self._audio_pcm.extend(padding)
            self._schedule_chunk()

    def stop(self) -> None:
        self._stopped = True
        with contextlib.suppress(asyncio.QueueFull):
            self._frame_queue.put_nowait(_QUEUE_SENTINEL)

    async def __aiter__(self) -> AsyncIterator[Any]:
        chunk_frame_idx = 0
        while True:
            item = await self._frame_queue.get()
            if item is _QUEUE_SENTINEL or item is CLOSE:
                break

            idx, frame_array, pts_ms = item

            if not self._first_frame_emitted:
                self._first_frame_emitted = True
                if callable(self.on_first_frame):
                    try:
                        res = self.on_first_frame()
                        if asyncio.iscoroutine(res):
                            await res
                    except Exception as exc:
                        logger.warning(f"[livekit-gen] on_first_frame callback failed: {exc}")

            self._frame_count += 1

            if RTC_AVAILABLE:
                h, w, c = frame_array.shape
                if c == 4:
                    frame_array = frame_array[:, :, :3]
                data = np.ascontiguousarray(frame_array).tobytes()
                video_frame = rtc.VideoFrame(w, h, rtc.VideoBufferType.RGB24, data)

                # Pair with matching 40ms audio slice
                audio_frame = None
                if self._current_chunk_audio:
                    start_b = (chunk_frame_idx % 50) * BYTES_PER_FRAME
                    end_b = start_b + BYTES_PER_FRAME
                    slice_data = self._current_chunk_audio[start_b:end_b]
                    if len(slice_data) == BYTES_PER_FRAME:
                        audio_frame = rtc.AudioFrame(
                            slice_data,
                            self.sample_rate,
                            1,
                            SAMPLES_PER_FRAME,
                        )
                chunk_frame_idx += 1

                yield (video_frame, audio_frame) if audio_frame else video_frame
            else:
                yield (idx, frame_array, pts_ms)


class LiveStreamVideoGenerator:
    """Video generator that taps an in-flight LiveStream and synchronizes audio."""

    def __init__(self, stream: LiveStream) -> None:
        self.stream = stream
        self._queue = stream.subscribe(maxsize=128)
        self._stopped = False
        self._audio_data: np.ndarray | None = None
        self._audio_loaded = False
        self._first_emitted = False
        self.on_first_frame: Any | None = None

    def _ensure_audio_loaded(self) -> None:
        if self._audio_loaded:
            return
        self._audio_loaded = True
        audio_path = getattr(self.stream, "audio_path", None)
        if audio_path and os.path.exists(audio_path):
            try:
                import soundfile as sf

                data, sr = sf.read(audio_path, dtype="int16")
                if sr != SAMPLE_RATE:
                    import librosa

                    data_f32, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
                    data = (np.clip(data_f32, -1.0, 1.0) * 32767).astype(np.int16)
                if data.ndim > 1:
                    data = data[:, 0]
                self._audio_data = data
            except Exception as exc:
                logger.warning(f"[livekit-gen] failed to load audio from {audio_path}: {exc}")

    def stop(self) -> None:
        self._stopped = True
        self.stream.unsubscribe(self._queue)
        with contextlib.suppress(asyncio.QueueFull):
            self._queue.put_nowait(CLOSE)

    async def __aiter__(self) -> AsyncIterator[Any]:
        self._ensure_audio_loaded()
        while not self._stopped:
            item = await self._queue.get()
            if item is CLOSE:
                break
            if item[0] != "frame":
                continue

            _, idx, pts_ms, frame_array = item

            if not self._first_emitted:
                self._first_emitted = True
                if callable(self.on_first_frame):
                    try:
                        res = self.on_first_frame()
                        if asyncio.iscoroutine(res):
                            await res
                    except Exception as exc:
                        logger.warning(f"[livekit-gen] on_first_frame error: {exc}")

            if RTC_AVAILABLE:
                h, w, c = frame_array.shape
                if c == 4:
                    frame_array = frame_array[:, :, :3]
                data = np.ascontiguousarray(frame_array).tobytes()
                video_frame = rtc.VideoFrame(w, h, rtc.VideoBufferType.RGB24, data)

                audio_frame = None
                if self._audio_data is not None:
                    start_s = idx * SAMPLES_PER_FRAME
                    end_s = start_s + SAMPLES_PER_FRAME
                    if start_s < len(self._audio_data):
                        chunk = self._audio_data[start_s:end_s]
                        if len(chunk) < SAMPLES_PER_FRAME:
                            chunk = np.pad(chunk, (0, SAMPLES_PER_FRAME - len(chunk)))
                        audio_frame = rtc.AudioFrame(
                            chunk.tobytes(),
                            SAMPLE_RATE,
                            1,
                            SAMPLES_PER_FRAME,
                        )

                yield (video_frame, audio_frame) if audio_frame else video_frame
            else:
                yield (idx, frame_array, pts_ms)


class MediaFileVideoGenerator:
    """Decodes an existing rendered video file (MP4) into synchronized LiveKit frames."""

    def __init__(self, file_path: str, loop: bool = False) -> None:
        self.file_path = file_path
        self.loop = loop
        self._stopped = False

    def stop(self) -> None:
        self._stopped = True

    async def __aiter__(self) -> AsyncIterator[Any]:
        import av

        while not self._stopped:
            container = av.open(self.file_path)
            try:
                video_stream = container.streams.video[0] if container.streams.video else None
                audio_stream = container.streams.audio[0] if container.streams.audio else None

                if not video_stream:
                    break

                resampler = None
                if audio_stream:
                    resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)

                audio_samples_buf = bytearray()

                for packet in container.demux():
                    if self._stopped:
                        break

                    if packet.stream == audio_stream and resampler:
                        for a_frame in packet.decode():
                            for resampled in resampler.resample(a_frame):
                                audio_samples_buf.extend(bytes(resampled.planes[0]))

                    elif packet.stream == video_stream:
                        for v_frame in packet.decode():
                            arr = v_frame.to_ndarray(format="rgb24")
                            h, w, _ = arr.shape
                            video_frame = rtc.VideoFrame(w, h, rtc.VideoBufferType.RGB24, arr.tobytes())

                            audio_frame = None
                            if len(audio_samples_buf) >= BYTES_PER_FRAME:
                                chunk = bytes(audio_samples_buf[:BYTES_PER_FRAME])
                                del audio_samples_buf[:BYTES_PER_FRAME]
                                audio_frame = rtc.AudioFrame(chunk, SAMPLE_RATE, 1, SAMPLES_PER_FRAME)

                            yield (video_frame, audio_frame) if audio_frame else video_frame
                            await asyncio.sleep(0.001)  # Cooperate with event loop
            finally:
                container.close()

            if not self.loop:
                break


__all__ = [
    "BYTES_PER_FRAME",
    "MIN_AUDIO_SEC",
    "SAMPLES_PER_FRAME",
    "SAMPLE_RATE",
    "TARGET_FPS",
    "TARGET_HEIGHT",
    "TARGET_WIDTH",
    "FloatVideoGenerator",
    "LiveStreamVideoGenerator",
    "MediaFileVideoGenerator",
]
