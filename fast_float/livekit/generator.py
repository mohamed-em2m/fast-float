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
import time
import wave
from collections.abc import AsyncIterator, Callable
from concurrent.futures import ThreadPoolExecutor
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


# Scratch dir for the temp WAVs: RAM-backed when available, so no disk I/O per chunk.
_TMP_DIR = "/dev/shm" if os.path.isdir("/dev/shm") else None


class _Cancelled(Exception):
    """Raised inside the inference thread when the chunk's generation is stale."""


class _LiveKitDispatcherBridge(FrameDispatcher):
    """Runs on the inference thread. Pairs each frame with its own 40 ms audio slice.

    Real-time changes vs. the old version:
      * `valid_samples`: the chunk is zero-padded to 2 s, but only frames that
        cover real audio are emitted (no dead silent frames at the tail).
      * `is_current(gen)`: after clear_buffer()/stop(), the next dispatch raises
        _Cancelled so the GPU stops working on audio nobody will hear.
    """

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        push: Callable[[int, tuple], None],
        audio_pcm: bytes,
        valid_samples: int,
        gen: int,
        is_current: Callable[[int], bool],
    ) -> None:
        super().__init__()
        self._loop = loop
        self._push = push
        self._audio_pcm = audio_pcm
        self._valid_samples = valid_samples
        self._gen = gen
        self._is_current = is_current
        self._cursor = 0  # in samples; only touched from the inference thread

    def dispatch(self, idx: int, frames: np.ndarray, pts_ms: float) -> None:
        if self._loop.is_closed():
            return
        if not self._is_current(self._gen):
            raise _Cancelled
        if frames.ndim == 3:
            frames = frames[None]
        for i in range(len(frames)):
            if self._cursor >= self._valid_samples:
                return  # padded tail: drop
            s = self._cursor * 2
            audio = self._audio_pcm[s : s + BYTES_PER_FRAME]
            if len(audio) < BYTES_PER_FRAME:
                audio = audio + bytes(BYTES_PER_FRAME - len(audio))
            self._cursor += SAMPLES_PER_FRAME
            item = (idx + i, frames[i], pts_ms + i * (1000.0 / TARGET_FPS), audio)
            self._loop.call_soon_threadsafe(self._push, self._gen, item)

    def close(self) -> None:
        pass  # the generator owns the sentinel; a per-chunk close must not end the stream


class FloatVideoGenerator:
    """Audio in -> synchronized (rtc.VideoFrame, rtc.AudioFrame) out, tuned for low latency."""

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
        first_chunk_sec: float = 1.0,
        idle_flush_sec: float | None = 0.25,
        max_queue_frames: int = 128,
        executor: ThreadPoolExecutor | None = None,
    ) -> None:
        self.agent = agent
        self.image_path = image_path
        self.loop = loop or asyncio.get_running_loop()
        self.width = width
        self.height = height
        self.fps = fps
        self.sample_rate = sample_rate
        self.semaphore = semaphore

        # Start a segment after `first_chunk_sec` of audio instead of waiting for a full 2 s.
        self._first_chunk_samples = int(sample_rate * min(first_chunk_sec, MIN_AUDIO_SAMPLES / sample_rate))
        self.idle_flush_sec = idle_flush_sec

        # One dedicated worker: serialises GPU work and doesn't compete with the
        # default executor (DNS, file I/O, ...) used by the rest of the agent.
        self._owns_executor = executor is None
        self._executor = executor or ThreadPoolExecutor(max_workers=1, thread_name_prefix="float-infer")

        self._audio_pcm = bytearray()
        self._frame_queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_frames)
        self._stopped = False
        self._inference_running = False
        self._segment_started = False
        self._flush_requested = False
        self._gen = 0  # bumped by clear_buffer(); stale work is dropped
        self._idle_handle: asyncio.TimerHandle | None = None
        self._frame_count = 0
        self._first_frame_emitted = False
        self._dropped_frames = 0
        self._bg_tasks: set[asyncio.Task[Any]] = set()
        self.on_first_frame: Any | None = None
        self.last_chunk_rtf: float | None = None  # inference_time / audio_time; must stay < 1

    # ------------------------------------------------------------------ input

    def _to_pcm16_mono(self, audio: Any) -> bytes:
        if RTC_AVAILABLE and isinstance(audio, rtc.AudioFrame):
            samples = np.frombuffer(audio.data, dtype=np.int16)
            ch, sr = audio.num_channels, audio.sample_rate
            if ch > 1:
                samples = samples.reshape(-1, ch).mean(axis=1)
            if sr != self.sample_rate:
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
        # Cheap path. Integer ratios (48k/24k -> 16k) use box-filter decimation,
        # which also low-passes; otherwise fall back to linear interpolation.
        if src_sr % self.sample_rate == 0:
            r = src_sr // self.sample_rate
            n = len(x) // r * r
            return x[:n].reshape(-1, r).mean(axis=1)
        n_out = int(len(x) * self.sample_rate / src_sr)
        return np.interp(np.linspace(0, len(x) - 1, n_out), np.arange(len(x)), x)

    def push_audio(self, audio: Any) -> None:
        """Call from the event-loop thread."""
        if self._stopped:
            return
        self._audio_pcm.extend(self._to_pcm16_mono(audio))
        self._pump()
        self._arm_idle_flush()

    def _arm_idle_flush(self) -> None:
        if self.idle_flush_sec is None:
            return
        if self._idle_handle:
            self._idle_handle.cancel()
        self._idle_handle = self.loop.call_later(self.idle_flush_sec, self.flush)

    # ------------------------------------------------------------- scheduling

    def _pump(self) -> None:
        """Start the next chunk if there is enough audio (or a flush is pending)."""
        if self._stopped or self._inference_running:
            return
        n = len(self._audio_pcm) // 2
        if n == 0:
            self._flush_requested = False
            self._segment_started = False
            return
        need = MIN_AUDIO_SAMPLES if self._segment_started else self._first_chunk_samples
        if n >= need or self._flush_requested:
            self._launch_chunk()

    def _launch_chunk(self) -> None:
        self._inference_running = True
        self._segment_started = True

        take = min(len(self._audio_pcm), MIN_AUDIO_SAMPLES * 2)
        pcm = bytes(self._audio_pcm[:take])
        del self._audio_pcm[:take]
        valid_samples = take // 2
        pcm += bytes(MIN_AUDIO_SAMPLES * 2 - len(pcm))  # FLOAT wants exactly 2 s

        if not self._audio_pcm and self._flush_requested:  # tail consumed: segment over
            self._flush_requested = False
            self._segment_started = False

        task = self.loop.create_task(self._run_chunk_inference(pcm, valid_samples, self._gen))
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)

    def flush(self) -> None:
        """Generate whatever audio is buffered (end of utterance / idle). Also fine to
        call from an AudioSegmentEnd handler."""
        self._idle_handle = None
        if self._audio_pcm:
            self._flush_requested = True
            self._pump()  # if inference is running, _run_chunk_inference's finally picks it up

    def clear_buffer(self) -> None:
        """Barge-in / interrupt: drop buffered audio, queued frames and in-flight inference."""
        self._gen += 1
        self._audio_pcm.clear()
        self._flush_requested = False
        self._segment_started = False
        if self._idle_handle:
            self._idle_handle.cancel()
            self._idle_handle = None
        while True:
            try:
                self._frame_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    # -------------------------------------------------------------- inference

    def _push(self, gen: int, item: tuple) -> None:
        """Loop thread. Drops *stale* frames; on overflow drops the OLDEST pair so we
        stay near live instead of playing old frames late."""
        if gen != self._gen or self._stopped:
            return
        if self._frame_queue.full():
            with contextlib.suppress(asyncio.QueueEmpty):
                self._frame_queue.get_nowait()
            self._dropped_frames += 1
            if self._dropped_frames % 25 == 1:
                logger.warning(f"[livekit-gen] consumer behind; dropped {self._dropped_frames} frames total")
        self._frame_queue.put_nowait(item)

    def _write_wav(self, pcm: bytes) -> str:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=_TMP_DIR) as f:
            path = f.name
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.sample_rate)
            w.writeframes(pcm)
        return path

    async def _run_chunk_inference(self, pcm: bytes, valid_samples: int, gen: int) -> None:
        wav_path = None
        try:
            wav_path = self._write_wav(pcm)  # before the semaphore: don't hold the GPU slot for I/O
            if self.semaphore:
                await self.semaphore.acquire()
            try:
                if gen != self._gen:
                    return
                t0 = time.perf_counter()
                if self.agent is not None and self.image_path:
                    from fast_float.core.service import run_inference_blocking

                    bridge = _LiveKitDispatcherBridge(
                        self.loop, self._push, pcm, valid_samples, gen, lambda g: g == self._gen and not self._stopped
                    )
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
                        self._executor,
                        run_inference_blocking,
                        self.agent,
                        None,
                        self.image_path,
                        wav_path,
                        req_data,
                        bridge,
                    )
                else:  # test pattern, one frame per 40 ms of real audio
                    n_frames = -(-valid_samples // SAMPLES_PER_FRAME)
                    for i in range(n_frames):
                        pattern = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                        pattern[:, :, 0] = (i * 5) % 255
                        pattern[:, :, 1] = 120
                        pattern[:, :, 2] = 180
                        s = i * BYTES_PER_FRAME
                        audio = pcm[s : s + BYTES_PER_FRAME].ljust(BYTES_PER_FRAME, b"\0")
                        self._push(gen, (i, pattern, i * 1000.0 / self.fps, audio))

                audio_s = valid_samples / self.sample_rate
                self.last_chunk_rtf = (time.perf_counter() - t0) / audio_s
                if self.last_chunk_rtf >= 1.0:
                    logger.warning(
                        f"[livekit-gen] slower than real time: RTF={self.last_chunk_rtf:.2f} "
                        f"(consider lower nfe / smaller chunks)"
                    )
            finally:
                if self.semaphore:
                    self.semaphore.release()
        except _Cancelled:
            logger.debug("[livekit-gen] chunk cancelled")
        except Exception as exc:
            logger.warning(f"[livekit-gen] inference error: {exc}")
        finally:
            if wav_path:
                with contextlib.suppress(OSError):
                    os.remove(wav_path)
            self._inference_running = False
            self._pump()  # next full chunk, or a flush that arrived while we were busy

    # ----------------------------------------------------------------- output

    def stop(self) -> None:
        self._stopped = True
        self._gen += 1
        if self._idle_handle:
            self._idle_handle.cancel()
        if self._owns_executor:
            self._executor.shutdown(wait=False, cancel_futures=True)
        # make room for the sentinel so the consumer always wakes up
        while self._frame_queue.full():
            with contextlib.suppress(asyncio.QueueEmpty):
                self._frame_queue.get_nowait()
        self._frame_queue.put_nowait(_QUEUE_SENTINEL)

    async def __aiter__(self) -> AsyncIterator[Any]:
        while True:
            item = await self._frame_queue.get()
            if item is _QUEUE_SENTINEL or item is CLOSE:
                break

            idx, frame_array, pts_ms, audio_data = item

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
                video_frame = rtc.VideoFrame(
                    w, h, rtc.VideoBufferType.RGB24, np.ascontiguousarray(frame_array).tobytes()
                )
                if audio_data:
                    audio_frame = rtc.AudioFrame(audio_data, self.sample_rate, 1, SAMPLES_PER_FRAME)
                    yield (video_frame, audio_frame)
                else:
                    yield video_frame
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
