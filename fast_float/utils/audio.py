"""Small PCM helpers shared by the live audio/video transports.

The live preview paths (LiveKit, raw WebRTC) pair each video frame with the
audio samples for that frame's media time.  Keeping the decoding/slicing here
means both transports use the same timeline and cannot drift apart.
"""

from __future__ import annotations

import logging
import os

import numpy as np

logger = logging.getLogger(__name__)


def resample_mono(data: np.ndarray, src_rate: int, target_rate: int) -> np.ndarray:
    """Resample a mono int16 array with linear interpolation.

    Linear interpolation is enough for the 48 kHz -> 16 kHz downsample the
    browser/WebRTC path can produce, and avoids pulling librosa (and its
    numba/LLVM startup cost) into the live request path.
    """
    if src_rate == target_rate or len(data) == 0:
        return data
    n_out = max(1, round(len(data) * target_rate / src_rate))
    src_idx = np.linspace(0, len(data) - 1, n_out)
    out = np.interp(src_idx, np.arange(len(data)), data.astype(np.float32))
    return np.clip(out, -32768, 32767).astype(np.int16)


def load_pcm16_mono(path: str | None, sample_rate: int = 16000) -> np.ndarray | None:
    """Decode ``path`` to mono int16 PCM at ``sample_rate``.

    Returns ``None`` (and logs a warning) when the file is missing or cannot be
    decoded, so a live stream keeps publishing video without audio instead of
    failing the whole request.
    """
    if not path or not os.path.exists(path):
        return None
    try:
        import soundfile as sf

        data, sr = sf.read(path, dtype="int16", always_2d=True)
        if data.size == 0:
            return None
        mono = np.ascontiguousarray(data[:, 0])
        if sr != sample_rate:
            mono = resample_mono(mono, int(sr), int(sample_rate))
        return mono
    except Exception as exc:
        logger.warning("[audio] failed to load %s: %s", path, exc)
        return None


def frame_sample_bounds(idx: int, fps: float, sample_rate: int = 16000) -> tuple[int, int]:
    """Sample range ``[start, end)`` covered by frame ``idx`` at ``fps``.

    Rounding the *boundaries* (instead of a fixed samples-per-frame count)
    keeps non-integer ratios exact: at 30 fps the slices alternate 533/534
    samples instead of accumulating 0.33 samples of drift per frame.
    """
    rate = max(1e-6, float(fps or 25.0))
    start = round(int(idx) * sample_rate / rate)
    end = round((int(idx) + 1) * sample_rate / rate)
    return start, max(start + 1, end)


def slice_audio_for_frame(
    audio: np.ndarray | None,
    idx: int,
    fps: float,
    sample_rate: int = 16000,
) -> tuple[bytes, int]:
    """PCM bytes for frame ``idx``, zero-padded when the audio ran out."""
    start, end = frame_sample_bounds(idx, fps, sample_rate)
    n = end - start
    if audio is None or start >= len(audio):
        return bytes(n * 2), n
    chunk = audio[start:end]
    if len(chunk) < n:
        chunk = np.pad(chunk, (0, n - len(chunk)))
    return np.ascontiguousarray(chunk).tobytes(), n


def slice_audio_at_time(
    audio: np.ndarray | None,
    media_t: float,
    fps: float,
    sample_rate: int = 16000,
    audio_start_s: float = 0.0,
) -> tuple[bytes, int]:
    """PCM bytes for the frame presented at ``media_t`` seconds.

    Used by the file publisher, where the video frame's own PTS (not its
    decode order) decides which audio samples belong to it.  A positive
    ``audio_start_s`` shifts the audio timeline to the container's start time;
    video before it gets silence instead of the first audio samples.
    """
    interval = 1.0 / max(1e-6, float(fps or 25.0))
    rel = float(media_t) - float(audio_start_s)
    if rel < 0:
        n = max(1, round(interval * sample_rate))
        return bytes(n * 2), n
    start = round(rel * sample_rate)
    end = round((rel + interval) * sample_rate)
    n = max(1, end - start)
    if audio is None or start >= len(audio):
        return bytes(n * 2), n
    chunk = audio[start:end]
    if len(chunk) < n:
        chunk = np.pad(chunk, (0, n - len(chunk)))
    return np.ascontiguousarray(chunk).tobytes(), n


__all__ = [
    "frame_sample_bounds",
    "load_pcm16_mono",
    "resample_mono",
    "slice_audio_at_time",
    "slice_audio_for_frame",
]
