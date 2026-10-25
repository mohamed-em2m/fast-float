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

from fast_float.livekit.settings import DEFAULT_LIVE_NFE, MAX_LIVE_NFE, MIN_LIVE_NFE
from fast_float.streaming.dispatcher import FrameDispatcher
from fast_float.streaming.live import CLOSE, LiveStream, live_queue_size
from fast_float.streaming.types import StreamMeta
from fast_float.utils.audio import frame_sample_bounds, load_pcm16_mono, slice_audio_at_time, slice_audio_for_frame

logger = logging.getLogger(__name__)

TARGET_FPS = 25.0
TARGET_WIDTH = 512
TARGET_HEIGHT = 512
SAMPLE_RATE = 16000
MIN_AUDIO_SEC = 2.0
MIN_AUDIO_SAMPLES = int(SAMPLE_RATE * MIN_AUDIO_SEC)  # 32,000 samples for 2s
SAMPLES_PER_FRAME = int(SAMPLE_RATE / TARGET_FPS)  # 640 samples per 40ms frame
BYTES_PER_FRAME = SAMPLES_PER_FRAME * 2  # 16-bit mono = 1280 bytes

# Live rendering must keep up with the audio arriving from Gemini. NFE is the
# one knob that trades FLOAT quality for latency: the file API keeps its
# balanced default (10), while the conversational path defaults to 6. At a
# measured RTF of 1.24 (see the warning below), 6 leaves ~25% headroom.
# Configure it in settings.toml as ``[float].livekit_nfe`` (5..25).
_QUEUE_SENTINEL = object()


def _resolve_live_nfe(value: int | str | None) -> int:
    """Clamp a requested NFE to the range the public generation API accepts."""
    try:
        nfe = int(value) if value is not None else DEFAULT_LIVE_NFE
    except (TypeError, ValueError):
        nfe = DEFAULT_LIVE_NFE
    return max(MIN_LIVE_NFE, min(MAX_LIVE_NFE, nfe))


# Scratch dir for the temp WAVs: RAM-backed when available, so no disk I/O per chunk.
_TMP_DIR = "/dev/shm" if os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK) else None


class _Cancelled(Exception):
    """Raised on the dispatch thread when the chunk's generation is stale."""


class _LiveKitDispatcherBridge(FrameDispatcher):
    """Runs on the dispatcher delivery thread. Pairs each frame with its own 40 ms audio slice.

    Real-time changes vs. the old version:
      * `valid_samples`: the chunk is zero-padded to 2 s, but only frames that
        cover real audio are emitted (no dead silent frames at the tail).
      * `is_current(gen)`: after clear_buffer()/stop(), the next dispatch raises
        _Cancelled so the pipeline's stop event aborts the chunk.
      * The audio position comes from the frame index, so frames dropped by the
        dispatcher (a ``gap``) do not shift every later frame's audio.
    """

    # A stale/cancelled chunk must stop the inference pipeline, not keep
    # generating frames nobody will hear.
    abort_on_error = True
    # A 2 s chunk is submitted as bursts of up to 50 frames. The default
    # FrameDispatcher live policy (50 frames / 1000 ms lag) drops the oldest
    # half of such a burst before the delivery thread can drain it, so disable
    # lag drops and let the generator's own bounded queue handle backpressure.
    max_queue_frames = 128
    max_lag_ms = 0.0

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        push: Callable[[int, tuple], None],
        audio_pcm: bytes,
        valid_samples: int,
        gen: int,
        is_current: Callable[[int], bool],
        sample_rate: int = SAMPLE_RATE,
    ) -> None:
        super().__init__()
        self._loop = loop
        self._push = push
        self._audio_pcm = audio_pcm
        self._valid_samples = valid_samples
        self._gen = gen
        self._is_current = is_current
        self._sample_rate = int(sample_rate)
        self._first_idx = 0
        self._fps = TARGET_FPS

    def open(self, meta: StreamMeta) -> None:
        # ``idx`` is relative to the stream's first frame; each chunk has its
        # own dispatcher, so the audio chunk starts at the stream's first frame.
        self._first_idx = int(meta.first_idx)
        self._fps = float(meta.fps or TARGET_FPS)

    def dispatch(self, idx: int, frames: np.ndarray, pts_ms: float) -> None:
        if self._loop.is_closed():
            return
        if not self._is_current(self._gen):
            raise _Cancelled
        if frames.ndim == 3:
            frames = frames[None]
        first = int(idx) - self._first_idx
        frame_ms = 1000.0 / self._fps
        for i in range(len(frames)):
            frame_idx = first + i
            if frame_idx < 0:
                continue
            start_sample, end_sample = frame_sample_bounds(frame_idx, self._fps, self._sample_rate)
            if start_sample >= self._valid_samples:
                return  # padded tail: drop
            s = start_sample * 2
            e = end_sample * 2
            audio = self._audio_pcm[s:e]
            if len(audio) < e - s:
                audio = audio + bytes(e - s - len(audio))
            item = (frame_idx, frames[i], pts_ms + i * frame_ms, audio)
            self._loop.call_soon_threadsafe(self._push, self._gen, item)

    def error(self, exc: BaseException) -> None:
        # FrameDispatcher reports a dispatch exception through error(); a stale
        # chunk is a normal barge-in, not an error.
        if isinstance(exc, _Cancelled):
            logger.debug("[livekit-gen] chunk cancelled")
            return
        super().error(exc)

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
        nfe: int | str | None = None,
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
        # Explicit argument > ``[float].livekit_nfe`` > the built-in default.
        # Read once so every chunk in a session reuses the same sampling budget.
        self.nfe = _resolve_live_nfe(nfe)

        # Start a segment after `first_chunk_sec` of audio instead of waiting for a full 2 s.
        self._first_chunk_samples = int(sample_rate * min(first_chunk_sec, MIN_AUDIO_SAMPLES / sample_rate))
        self.idle_flush_sec = idle_flush_sec

        # One dedicated worker: serialises GPU work and doesn't compete with the
        # default executor (DNS, file I/O, ...) used by the rest of the agent.
        self._owns_executor = executor is None
        self._executor = executor or ThreadPoolExecutor(max_workers=1, thread_name_prefix="float-infer")

        self._audio_pcm = bytearray()
        self._frame_queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_frames)
        # Largest burst one chunk can add: a full 2 s window rendered at
        # ``TARGET_FPS``. Production is paced so that a new chunk only starts
        # when this many slots are free, i.e. the queue never has to drop the
        # oldest frames of the utterance being played.
        self._max_chunk_frames = int(np.ceil(MIN_AUDIO_SAMPLES / SAMPLES_PER_FRAME))
        self._pump_retry: asyncio.TimerHandle | None = None
        self._queue_retry_s = 0.1
        self._stopped = False
        self._stop_pending = False
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
        # Signals that every sample pushed so far has been generated and the
        # consumer has drained the queue. The external-agent runner uses it to
        # report `lk.playback_finished` after a TTS segment has fully rendered.
        self._idle_event = asyncio.Event()
        self._idle_event.set()

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
        self._refresh_idle()

    def _arm_idle_flush(self) -> None:
        if self._idle_handle:
            self._idle_handle.cancel()
            self._idle_handle = None
        if self.idle_flush_sec is None:
            return
        self._idle_handle = self.loop.call_later(self.idle_flush_sec, self.flush)

    # --------------------------------------------------------------- idle state

    @property
    def idle(self) -> bool:
        """True when no audio is buffered, no inference is running and the queue is empty."""
        return (
            not self._audio_pcm
            and not self._flush_requested
            and not self._inference_running
            and self._frame_queue.empty()
        )

    def _refresh_idle(self) -> None:
        if self.idle:
            self._idle_event.set()
        else:
            self._idle_event.clear()

    async def wait_until_idle(self, timeout: float | None = None) -> bool:
        """Wait until every pushed sample has been generated and consumed.

        Returns ``True`` when idle, ``False`` on timeout. Used by the
        conversational runner to know when a TTS segment has finished
        rendering, so it can notify the agent that playback is over.
        """
        try:
            await asyncio.wait_for(self._idle_event.wait(), timeout)
        except asyncio.TimeoutError:
            return False
        return True

    # ------------------------------------------------------------- scheduling

    def _pump(self) -> None:
        """Start the next chunk if there is enough audio (or a flush is pending)."""
        if self._stopped or self._inference_running:
            return
        n = len(self._audio_pcm) // 2
        if n == 0:
            self._flush_requested = False
            self._segment_started = False
            self._refresh_idle()
            return
        need = MIN_AUDIO_SAMPLES if self._segment_started else self._first_chunk_samples
        if n < need and not self._flush_requested:
            self._refresh_idle()
            return
        # Backpressure: don't let production outrun playout. Queuing the whole
        # utterance at inference speed overflows the frame queue, which then
        # drops the *oldest* frames — the viewer hears a jump cut in the middle
        # of the sentence. Wait for the consumer instead; inference is faster
        # than real time, so the backlog still drains before playback needs it.
        # One extra chunk of headroom covers the frames this chunk already
        # handed to the loop via call_soon_threadsafe but that have not landed
        # in the queue yet. A queue smaller than a single chunk cannot be paced
        # (waiting would deadlock), so it keeps the old drop-oldest behaviour.
        maxsize = self._frame_queue.maxsize
        if maxsize >= self._max_chunk_frames:
            margin = self._max_chunk_frames if maxsize >= 2 * self._max_chunk_frames else 0
            if self._frame_queue.qsize() + self._max_chunk_frames + margin > maxsize:
                self._schedule_pump_retry()
                return
        self._launch_chunk()

    def _schedule_pump_retry(self) -> None:
        if self._pump_retry is not None or self._stopped:
            return
        self._pump_retry = self.loop.call_later(self._queue_retry_s, self._on_pump_retry)

    def _on_pump_retry(self) -> None:
        self._pump_retry = None
        self._pump()

    def _launch_chunk(self) -> None:
        self._inference_running = True
        self._segment_started = True
        self._idle_event.clear()

        take = min(len(self._audio_pcm), MIN_AUDIO_SAMPLES * 2) & ~1  # whole int16 samples only
        if take <= 0:
            self._inference_running = False
            return
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
        # Also cancels a still-pending idle timer: an explicit flush replaces it,
        # otherwise the old timer could fire into the next utterance's audio.
        if self._idle_handle:
            self._idle_handle.cancel()
            self._idle_handle = None
        if self._audio_pcm:
            self._flush_requested = True
            self._pump()  # if inference is running, _run_chunk_inference's finally picks it up
        self._refresh_idle()

    def clear_buffer(self) -> None:
        """Barge-in / interrupt: drop buffered audio, queued frames and in-flight inference."""
        self._gen += 1
        self._audio_pcm.clear()
        self._flush_requested = False
        self._segment_started = False
        if self._idle_handle:
            self._idle_handle.cancel()
            self._idle_handle = None
        if self._pump_retry:
            self._pump_retry.cancel()
            self._pump_retry = None
        while True:
            try:
                self._frame_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        self._refresh_idle()

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
        self._refresh_idle()

    def _write_wav(self, pcm: bytes) -> str:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=_TMP_DIR) as f:
            path = f.name
        try:
            with wave.open(path, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(self.sample_rate)
                w.writeframes(pcm)
        except BaseException:
            with contextlib.suppress(OSError):
                os.remove(path)
            raise
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
                        self.loop,
                        self._push,
                        pcm,
                        valid_samples,
                        gen,
                        lambda g: g == self._gen and not self._stopped,
                        sample_rate=self.sample_rate,
                    )
                    req_data = {
                        "a_cfg_scale": 2.0,
                        "r_cfg_scale": 1.0,
                        "e_cfg_scale": 1.0,
                        "emotion": "S2E",
                        "nfe": self.nfe,
                        "no_crop": False,
                        "seed": 25,
                    }

                    def _infer_and_close() -> str:
                        # The bridge owns a dispatcher delivery thread (started by
                        # bind() inside run_inference). It must be shut down per
                        # chunk or every chunk leaks a thread for the whole session.
                        try:
                            return run_inference_blocking(
                                self.agent,
                                None,
                                self.image_path,
                                wav_path,
                                req_data,
                                bridge,
                            )
                        finally:
                            with contextlib.suppress(Exception):
                                bridge.shutdown()

                    await self.loop.run_in_executor(self._executor, _infer_and_close)
                else:  # test pattern, one frame per 40 ms of real audio
                    n_frames = int(np.ceil(valid_samples * self.fps / self.sample_rate))
                    for i in range(n_frames):
                        pattern = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                        pattern[:, :, 0] = (i * 5) % 255
                        pattern[:, :, 1] = 120
                        pattern[:, :, 2] = 180
                        start_sample, end_sample = frame_sample_bounds(i, self.fps, self.sample_rate)
                        s = start_sample * 2
                        audio = pcm[s : end_sample * 2].ljust((end_sample - start_sample) * 2, b"\0")
                        self._push(gen, (i, pattern, i * 1000.0 / self.fps, audio))

                if gen != self._gen:  # barge-in/stop: the chunk was discarded
                    return
                audio_s = valid_samples / self.sample_rate
                self.last_chunk_rtf = (time.perf_counter() - t0) / audio_s
                if self.last_chunk_rtf >= 1.0:
                    logger.warning(
                        f"[livekit-gen] slower than real time: RTF={self.last_chunk_rtf:.2f} "
                        f"(nfe={self.nfe}; lower [float].livekit_nfe in settings.toml or use a faster backend)"
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
            self._refresh_idle()

    # ----------------------------------------------------------------- output

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        self._gen += 1
        if self._idle_handle:
            self._idle_handle.cancel()
            self._idle_handle = None
        if self._pump_retry:
            self._pump_retry.cancel()
            self._pump_retry = None
        if self._owns_executor:
            self._executor.shutdown(wait=False, cancel_futures=True)
        # Do NOT flush the queue here: frames already generated are real audio
        # and the consumer must play them out (dropping them cut the tail of
        # the last sentence). If the queue is full the sentinel can't be
        # enqueued; ``_stop_pending`` makes the consumer exit as soon as it has
        # drained what is queued.
        self._stop_pending = True
        with contextlib.suppress(asyncio.QueueFull):
            self._frame_queue.put_nowait(_QUEUE_SENTINEL)

    async def __aiter__(self) -> AsyncIterator[Any]:
        while True:
            if self._stop_pending and self._frame_queue.empty():
                break
            item = await self._frame_queue.get()
            self._refresh_idle()
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
                    audio_frame = rtc.AudioFrame(audio_data, self.sample_rate, 1, len(audio_data) // 2)
                    yield (video_frame, audio_frame)
                else:
                    yield video_frame
            else:
                yield (idx, frame_array, pts_ms)


class LiveStreamVideoGenerator:
    """Video generator that taps an in-flight LiveStream and synchronizes audio."""

    def __init__(self, stream: LiveStream) -> None:
        self.stream = stream
        # Fast renders produce many more frames than real-time playout can
        # consume; buffer them (bounded by ``live_queue_size``) so the preview
        # plays the whole clip instead of skipping the frames that did not fit.
        self._queue = stream.subscribe(maxsize=live_queue_size(stream.meta))
        self._stopped = False
        self._stop_pending = False
        self._audio_data: np.ndarray | None = None
        self._audio_loaded = False
        self._first_emitted = False
        self._fps = float(stream.meta.fps) if stream.meta and stream.meta.fps else TARGET_FPS
        self.on_first_frame: Any | None = None

    def _ensure_audio_loaded(self) -> None:
        if self._audio_loaded:
            return
        self._audio_loaded = True
        self._audio_data = load_pcm16_mono(getattr(self.stream, "audio_path", None), SAMPLE_RATE)

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        dropped = getattr(self._queue, "dropped_frames", 0)
        if dropped:
            logger.warning(
                "[livekit-gen] %s: live preview fell %d frame(s) behind and skipped them",
                self.stream.request_id,
                dropped,
            )
        spilled = getattr(self._queue, "spilled_frames", 0)
        if spilled:
            logger.info(
                "[livekit-gen] %s: buffered %d frame(s) on disk for real-time playback",
                self.stream.request_id,
                spilled,
            )
        # Unsubscribe first so no new frames arrive, then queue the sentinel:
        # frames already queued are real audio/video and must be played out
        # before the publisher disconnects.
        self.stream.unsubscribe(self._queue)
        self._stop_pending = True
        with contextlib.suppress(asyncio.QueueFull):
            self._queue.put_nowait(CLOSE)

    async def __aiter__(self) -> AsyncIterator[Any]:
        self._ensure_audio_loaded()
        # The subscription is created before meta is known in some callers;
        # grow it now so a fast render is buffered instead of dropped.
        if self.stream.meta is not None:
            self._fps = float(self.stream.meta.fps or TARGET_FPS)
            self._queue.resize(live_queue_size(self.stream.meta))
        try:
            while True:
                if self._stop_pending and self._queue.empty():
                    break
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
                        chunk, samples = slice_audio_for_frame(self._audio_data, idx, self._fps, SAMPLE_RATE)
                        audio_frame = rtc.AudioFrame(chunk, SAMPLE_RATE, 1, samples)

                    yield (video_frame, audio_frame) if audio_frame else video_frame
                else:
                    yield (idx, frame_array, pts_ms)
        finally:
            # Release the overflow spool as soon as the consumer is done
            # (including cancellation / early teardown).
            self._queue.dispose()


class MediaFileVideoGenerator:
    """Decodes an existing rendered video file (MP4) into synchronized LiveKit frames."""

    def __init__(self, file_path: str, loop: bool = False) -> None:
        self.file_path = file_path
        self.loop = loop
        self._stopped = False
        # Filled in by probe(): a file rendered at 30 fps published at the
        # default 25 fps would play slow and drift against its own audio.
        self.fps: float = TARGET_FPS
        self.width: int = TARGET_WIDTH
        self.height: int = TARGET_HEIGHT
        self._probed = False

    def probe(self) -> tuple[int, int, float]:
        """Read the container's frame size and rate (defaults if unreadable)."""
        if self._probed:
            return self.width, self.height, self.fps
        self._probed = True
        try:
            import av

            with av.open(self.file_path) as container:
                stream = container.streams.video[0] if container.streams.video else None
                if stream is not None:
                    self.width = int(stream.codec_context.width or self.width)
                    self.height = int(stream.codec_context.height or self.height)
                    rate = stream.average_rate or stream.guessed_rate
                    if rate:
                        self.fps = float(rate)
        except Exception as exc:
            logger.warning(f"[livekit-gen] could not probe {self.file_path}: {exc}")
        return self.width, self.height, self.fps

    def stop(self) -> None:
        self._stopped = True

    def _load_audio(self) -> tuple[np.ndarray | None, float]:
        """Decode the file's audio to mono 16 kHz PCM once.

        The audio is paired with video by PTS below, so it must be available
        before the first video frame.  FIFO pairing (one slice per decoded
        frame) drifts whenever the container is not CFR or the audio stream
        starts with encoder priming.
        """
        try:
            import av

            with av.open(self.file_path) as container:
                if not container.streams.audio:
                    return None, 0.0
                stream = container.streams.audio[0]
                resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
                chunks: list[bytes] = []
                first_pts: float | None = None
                for packet in container.demux(stream):
                    for a_frame in packet.decode():
                        if first_pts is None and a_frame.pts is not None and a_frame.time_base is not None:
                            first_pts = float(a_frame.pts * a_frame.time_base)
                        chunks.extend(bytes(resampled.planes[0]) for resampled in resampler.resample(a_frame))
                chunks.extend(bytes(resampled.planes[0]) for resampled in resampler.resample(None))
                # The first decoded sample sits at the first audio frame's PTS;
                # the stream start time is only a fallback for containers that
                # do not tag individual frames.
                start_s = first_pts if first_pts is not None else 0.0
                if stream.start_time is not None and stream.time_base is not None and first_pts is None:
                    start_s = float(stream.start_time * stream.time_base)
                if not chunks:
                    return None, start_s
                return np.frombuffer(b"".join(chunks), dtype=np.int16), start_s
        except Exception as exc:
            logger.warning(f"[livekit-gen] could not decode audio from {self.file_path}: {exc}")
            return None, 0.0

    async def __aiter__(self) -> AsyncIterator[Any]:
        import av

        # Probe before the first frame so direct users get the file's real fps
        # (the publisher already probes to size its VideoSource).
        self.probe()
        audio_data, audio_start_s = self._load_audio()
        frame_interval = 1.0 / max(self.fps, 1e-6)

        while not self._stopped:
            container = av.open(self.file_path)
            try:
                video_stream = container.streams.video[0] if container.streams.video else None

                if not video_stream:
                    break

                frame_index = 0
                for packet in container.demux(video_stream):
                    if self._stopped:
                        break

                    if packet.stream != video_stream:
                        continue
                    for v_frame in packet.decode():
                        arr = v_frame.to_ndarray(format="rgb24")
                        h, w, _ = arr.shape
                        video_frame = rtc.VideoFrame(w, h, rtc.VideoBufferType.RGB24, arr.tobytes())

                        audio_frame = None
                        if audio_data is not None:
                            if v_frame.pts is not None and v_frame.time_base is not None:
                                media_t = float(v_frame.pts * v_frame.time_base)
                            else:
                                media_t = frame_index * frame_interval
                            frame_index += 1
                            chunk, samples = slice_audio_at_time(
                                audio_data,
                                media_t,
                                self.fps,
                                SAMPLE_RATE,
                                audio_start_s,
                            )
                            audio_frame = rtc.AudioFrame(chunk, SAMPLE_RATE, 1, samples)

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
