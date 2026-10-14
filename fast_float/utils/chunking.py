"""Audio chunking and chunk-video merging for long-form generation.

Splitting the driving audio into overlapping windows lets the server render a
long clip as several independent, progress-reportable generation runs, then
stitch the per-chunk MP4s back into one file. Chunks overlap so the merge can
drop the duplicated head of every later chunk, keeping timing continuous.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from typing import Any, Dict, List, Optional

import numpy as np
import soundfile as sf

from fast_float.utils.logger import logger

# A trailing chunk shorter than this gets folded into the previous one, since
# the flow-matching model needs a minimum window to produce sensible motion.
MIN_CHUNK_SECONDS = 0.5


def get_audio_duration_seconds(wav_path: str) -> float:
    """Return the duration of a WAV file in seconds (0.0 on failure)."""
    try:
        info = sf.info(wav_path)
        if info.samplerate <= 0:
            return 0.0
        return float(info.frames) / float(info.samplerate)
    except Exception as e:
        logger.warning(f"Could not probe audio duration for {wav_path}: {e}")
        return 0.0


def split_wav_into_chunks(
    wav_path: str,
    chunk_seconds: float,
    overlap_seconds: float,
    out_dir: str,
) -> List[Dict[str, Any]]:
    """Slice a WAV into overlapping windows written as separate WAV files.

    Returns a list of ``{"path", "start", "duration", "index"}``. When chunking
    is disabled (``chunk_seconds <= 0``) or the audio is already short enough,
    the original file is returned as a single-element list.
    """
    data, sr = sf.read(wav_path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1).astype(np.float32)

    total = len(data) / sr if sr > 0 else 0.0
    if chunk_seconds <= 0 or total <= chunk_seconds:
        return [{"path": wav_path, "start": 0.0, "duration": total, "index": 0, "synthetic": False}]

    overlap = max(0.0, min(float(overlap_seconds), chunk_seconds - 0.1))
    step = max(chunk_seconds - overlap, 0.1)

    spans: List[List[float]] = []
    start = 0.0
    while start < total - 1e-6:
        end = min(start + chunk_seconds, total)
        spans.append([start, end])
        if end >= total - 1e-6:
            break
        start += step

    # Fold a too-short tail into the previous chunk so we never render a 0.1s clip.
    if len(spans) >= 2 and (spans[-1][1] - spans[-1][0]) < MIN_CHUNK_SECONDS:
        spans[-2][1] = spans[-1][1]
        spans.pop()

    results: List[Dict[str, Any]] = []
    for i, (s, e) in enumerate(spans):
        i0 = max(int(round(s * sr)), 0)
        i1 = min(int(round(e * sr)), len(data))
        if i1 <= i0:
            continue
        out_path = os.path.join(out_dir, f"chunk_{i:03d}.wav")
        sf.write(out_path, data[i0:i1], sr, format="WAV", subtype="PCM_16")
        results.append(
            {
                "path": out_path,
                "start": s,
                "duration": (e - s),
                "index": i,
                "synthetic": True,
            }
        )

    logger.info(
        f"✂️  Split {total:.2f}s audio into {len(results)} chunks "
        f"(window={chunk_seconds}s, overlap={overlap}s)"
    )
    return results


def pad_wav_to_duration(wav_path: str, target_seconds: float, out_path: str) -> str:
    """Append silence so the WAV reaches ``target_seconds`` (no-op if already long)."""
    data, sr = sf.read(wav_path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1).astype(np.float32)
    cur = len(data) / sr if sr > 0 else 0.0
    need = int(round((target_seconds - cur) * sr)) if sr > 0 else 0
    if need > 0:
        data = np.concatenate([data, np.zeros(need, dtype=np.float32)])
    sf.write(out_path, data, sr, format="WAV", subtype="PCM_16")
    return out_path


def group_chunks_for_batch(
    chunks: List[Dict[str, Any]],
    batch_size: int = 1,
    max_pad_seconds: float = 2.0,
) -> List[List[Dict[str, Any]]]:
    """Greedily group adjacent chunks for batched sampling.

    A chunk joins the current group only if padding it up to the group's max
    duration needs at most ``max_pad_seconds`` of silence. Chunks that would
    need more padding run solo so we never burn whole ODE solves on silence.
    ``batch_size <= 1`` returns every chunk solo (today's sequential path).
    """
    if batch_size <= 1 or len(chunks) <= 1:
        return [[c] for c in chunks]
    groups: List[List[Dict[str, Any]]] = []
    cur: List[Dict[str, Any]] = []
    for ch in chunks:
        trial = cur + [ch]
        mx = max(c["duration"] for c in trial)
        fits = len(trial) <= batch_size and all(
            mx - c["duration"] <= max_pad_seconds + 1e-9 for c in trial
        )
        if fits:
            cur = trial
        else:
            if cur:
                groups.append(cur)
            cur = [ch]
        if len(cur) >= batch_size:
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)
    return groups
def _has_audio_stream(path: str) -> bool:
    """True if the file contains at least one audio stream (ffprobe)."""
    try:
        res = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "a",
                "-show_entries", "stream=index", "-of", "csv=p=0", path,
            ],
            capture_output=True,
            text=True,
            timeout=20,
        )
        return res.returncode == 0 and bool(res.stdout.strip())
    except Exception:
        return False


def _concat_stream_copy(paths: List[str], output_path: str) -> bool:
    """Concatenate MP4s by stream copy (fast, lossless). False on any failure."""
    list_path = None
    try:
        fd, list_path = tempfile.mkstemp(suffix=".txt")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            for p in paths:
                # ffmpeg concat lists want forward slashes and single quotes.
                safe = os.path.abspath(p).replace("\\", "/").replace("'", r"'\''")
                fh.write(f"file '{safe}'\n")
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "concat", "-safe", "0", "-i", list_path,
            "-c", "copy", "-movflags", "+faststart", output_path,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        ok = res.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0
        if not ok:
            logger.warning(f"Stream-copy concat failed: {res.stderr[-400:]}")
        return ok
    except Exception as e:
        logger.warning(f"Stream-copy concat error: {e}")
        return False
    finally:
        if list_path and os.path.exists(list_path):
            os.remove(list_path)


def _concat_filter(
    paths: List[str],
    output_path: str,
    overlap_seconds: float,
    encoder: Optional[str],
    fps: int,
    with_audio: bool,
) -> str:
    """Re-encode and concatenate, trimming the overlapping head of later chunks."""
    n = len(paths)
    inputs: List[str] = []
    for p in paths:
        inputs += ["-i", p]

    parts: List[str] = []
    labels: List[str] = []
    for i in range(n):
        trim = overlap_seconds if i > 0 else 0.0
        v_filter = "trim=start=%.4f,setpts=PTS-STARTPTS" % trim if trim > 0.01 else "setpts=PTS-STARTPTS"
        parts.append(f"[{i}:v]{v_filter}[v{i}];")
        if with_audio:
            a_filter = "atrim=start=%.4f,asetpts=PTS-STARTPTS" % trim if trim > 0.01 else "asetpts=PTS-STARTPTS"
            parts.append(f"[{i}:a]{a_filter}[a{i}];")
        labels += [f"[v{i}]"] + ([f"[a{i}]"] if with_audio else [])

    concat = f"{''.join(labels)}concat=n={n}:v=1:a={1 if with_audio else 0}[outv]"
    filter_complex = "".join(parts) + concat + ("[outa]" if with_audio else "")

    if encoder and "nvenc" in encoder.lower():
        vopts = ["-c:v", encoder, "-preset", "p1", "-rc", "constqp", "-qp", "23", "-pix_fmt", "yuv420p"]
    else:
        vopts = ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "23", "-pix_fmt", "yuv420p"]

    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *inputs,
           "-filter_complex", filter_complex, "-map", "[outv]"]
    if with_audio:
        cmd += ["-map", "[outa]", "-c:a", "aac", "-b:a", "128k"]
    cmd += [*vopts, "-r", str(fps), "-movflags", "+faststart", output_path]

    subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=1800)
    return output_path


def merge_mp4_chunks(
    chunk_paths: List[str],
    output_path: str,
    overlap_seconds: float = 0.0,
    encoder: Optional[str] = None,
    fps: int = 25,
) -> str:
    """Merge per-chunk MP4s into one file, trimming overlap between chunks."""
    if not chunk_paths:
        raise ValueError("No chunk videos to merge")
    if len(chunk_paths) == 1:
        shutil.copy(chunk_paths[0], output_path)
        return output_path

    overlap = max(0.0, float(overlap_seconds))
    with_audio = all(_has_audio_stream(p) for p in chunk_paths)

    # Fastest path: no trim needed and identical codecs -> straight stream copy.
    if overlap < 0.01 and with_audio and _concat_stream_copy(chunk_paths, output_path):
        logger.info(f"🔗 Merged {len(chunk_paths)} chunks via stream copy")
        return output_path

    logger.info(f"🔗 Merging {len(chunk_paths)} chunks via filter (overlap={overlap}s, audio={with_audio})")
    _concat_filter(chunk_paths, output_path, overlap, encoder, fps, with_audio)
    return output_path
