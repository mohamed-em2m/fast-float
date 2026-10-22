"""Merge-path tests that use a real ffmpeg (skipped when it is unavailable)."""

import shutil
import subprocess

import pytest

from fast_float.utils.chunking import _has_audio_stream, merge_mp4_chunks

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not installed",
)


def _ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *args], check=True)


def _make_chunk(path, color: str, seconds: float = 0.4) -> None:
    _ffmpeg(
        "-f",
        "lavfi",
        "-i",
        f"color=c={color}:s=64x64:r=10:d={seconds}",
        "-c:v",
        "mpeg4",
        "-q:v",
        "5",
        str(path),
    )


def _duration(path) -> float:
    res = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(res.stdout.strip())


def test_video_only_chunks_merge_by_stream_copy(tmp_path):
    a, b = tmp_path / "a.mp4", tmp_path / "b.mp4"
    _make_chunk(a, "red")
    _make_chunk(b, "blue")

    out = merge_mp4_chunks([str(a), str(b)], str(tmp_path / "out.mp4"))

    assert _duration(out) == pytest.approx(0.8, abs=0.2)
    assert not _has_audio_stream(out)


def test_merge_muxes_audio_into_video_only_chunks(tmp_path):
    a, b = tmp_path / "a.mp4", tmp_path / "b.mp4"
    _make_chunk(a, "red")
    _make_chunk(b, "blue")
    wav = tmp_path / "audio.wav"
    _ffmpeg("-f", "lavfi", "-i", "sine=frequency=440:duration=0.8", "-c:a", "pcm_s16le", str(wav))

    out = merge_mp4_chunks(
        [str(a), str(b)],
        str(tmp_path / "out.mp4"),
        overlap_seconds=0.0,
        fps=10,
        audio_path=str(wav),
    )

    assert _duration(out) == pytest.approx(0.8, abs=0.2)
    assert _has_audio_stream(out)
