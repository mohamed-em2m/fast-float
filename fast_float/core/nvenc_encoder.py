"""FFmpeg NVENC video writer with sub-process management and retry logic."""

from __future__ import annotations

import logging
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Constant-quality NVENC settings. Kept in one place so the per-chunk writer and
# the merge fallback cannot drift apart.
_ENCODE_OPTS = [
    "-c:v",
    "h264_nvenc",
    "-preset",
    "p1",
    "-tune",
    "hq",
    "-rc",
    "constqp",
    "-qp",
    "23",
    "-delay",
    "0",
    "-no-scenecut",
    "1",
    "-forced-idr",
    "1",
    "-pix_fmt",
    "yuv420p",
    "-movflags",
    "+faststart",
]

# Frames per pipe write. One write per frame costs a syscall + a `.tobytes()`
# per frame; 32 frames is a few MB — large enough to amortize, small enough to
# keep the transient buffer off the critical memory path.
_WRITE_BLOCK = 32


def _to_uint8_hwc(vid: torch.Tensor) -> torch.Tensor:
    """Device-side half of the export: ``(T, C|H, H|W, W|C)`` -> uint8 NHWC."""
    if vid.dim() != 4:
        raise ValueError(f"Expected a 4D frame tensor, got shape {tuple(vid.shape)}")
    if vid.shape[1] in (3, 4) and vid.shape[-1] not in (3, 4):
        vid = vid.permute(0, 2, 3, 1)
    vid = vid.contiguous()

    if torch.is_floating_point(vid):
        vid = (vid.clamp(-1.0, 1.0) * 127.5 + 127.5).to(torch.uint8)
    else:
        vid = vid.to(torch.uint8)
    return vid


def frames_to_uint8_hwc(vid: torch.Tensor) -> np.ndarray:
    """Convert a frame tensor to a contiguous ``(T, H, W, 3|4)`` uint8 array.

    Accepts ``(T, C, H, W)`` or ``(T, H, W, C)``, float in ``[-1, 1]`` or
    integer. The device copy (``.cpu()``) synchronizes only the stream that
    produced the tensor, so this is safe to call from a decode loop that is
    itself running on a non-default CUDA stream.
    """
    arr = _to_uint8_hwc(vid).cpu().numpy()
    if arr.ndim != 4 or arr.shape[-1] not in (3, 4):
        raise ValueError(f"Expected (T, H, W, 3|4) frames, got {arr.shape}")
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    return arr


@dataclass
class StagedFrames:
    """Host-visible frames whose device->host copy may still be in flight.

    ``frames`` aliases pinned host memory, so the encoder thread must call
    :meth:`wait` before reading it and keep this object alive until the encode
    finishes (that reference is what keeps the pinned buffer from being freed).
    """

    frames: np.ndarray
    host: torch.Tensor | None = None
    event: torch.cuda.Event | None = None

    def wait(self) -> np.ndarray:
        if self.event is not None:
            self.event.synchronize()
            self.event = None
        return self.frames


def stage_frames_for_nvenc(
    vid: torch.Tensor,
    stream: torch.cuda.Stream | None = None,
) -> StagedFrames:
    """Cast and copy frames out of device memory on the caller's stream.

    Decoding runs on the default stream, so doing the cast + D2H there stalls
    the next chunk for ~60 ms. Passing a private stream issues both there and
    returns immediately instead; the encode thread blocks on the returned
    event, which overlaps the copy with the next decode.

    With ``stream=None`` (or a CPU tensor) this is a synchronous conversion and
    the returned object is already safe to read.
    """
    if stream is not None and vid.is_cuda:
        stream.wait_stream(torch.cuda.current_stream(vid.device))
        with torch.cuda.stream(stream):
            u8 = _to_uint8_hwc(vid)
            host = torch.empty(u8.shape, dtype=torch.uint8, pin_memory=True)
            host.copy_(u8, non_blocking=True)
            event = torch.cuda.Event()
            event.record(stream)
        # ``vid`` was allocated on the decode stream; tell the caching allocator
        # that the side stream still reads it so its block is not reused early.
        vid.record_stream(stream)
        return StagedFrames(host.numpy(), host, event)

    return StagedFrames(frames_to_uint8_hwc(vid))


def encode_numpy_nvenc(
    frames: np.ndarray,
    video_path: str,
    fps: int = 25,
    max_retries: int = 2,
) -> str:
    """Encode an in-memory ``(T, H, W, 3|4)`` uint8 array to MP4 with NVENC.

    Pure CPU/NVENC work (no torch calls), so callers may run it on a worker
    thread while the main thread keeps decoding.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found in PATH")
    if frames.ndim != 4 or frames.shape[-1] not in (3, 4):
        raise ValueError(f"Expected (T, H, W, 3|4) frames, got {frames.shape}")

    T, H, W, C = frames.shape
    pix_fmt_in = "rgba" if C == 4 else "rgb24"

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        pix_fmt_in,
        "-s",
        f"{W}x{H}",
        "-r",
        str(fps),
        "-i",
        "-",
        *_ENCODE_OPTS,
        video_path,
    ]

    for retry_count in range(max_retries):
        proc: subprocess.Popen | None = None
        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                bufsize=0,
            )

            stderr_lines: list[str] = []

            def read_stderr(proc=proc, stderr_lines=stderr_lines) -> None:
                assert proc is not None
                assert proc.stderr is not None
                for line in proc.stderr:
                    line_str = line.decode(errors="ignore")
                    stderr_lines.append(line_str)
                    if "error" in line_str.lower() or "failed" in line_str.lower():
                        logger.warning(f"FFmpeg stderr: {line_str.strip()}")

            stderr_thread = threading.Thread(target=read_stderr, daemon=True)
            stderr_thread.start()

            for start in range(0, T, _WRITE_BLOCK):
                assert proc.stdin is not None
                try:
                    proc.stdin.write(frames[start : start + _WRITE_BLOCK].tobytes())
                except BrokenPipeError:
                    proc.wait(timeout=5)
                    stderr_thread.join(timeout=1)
                    full_stderr = "".join(stderr_lines)
                    logger.error(f"❌ FFmpeg died at frame {start}/{T}")
                    logger.error(f"FFmpeg stderr:\n{full_stderr[-1000:]}")
                    raise RuntimeError(f"FFmpeg broken pipe at frame {start}: {full_stderr[-500:]}") from None

            assert proc.stdin is not None
            proc.stdin.close()

            ret = proc.wait(timeout=300)
            stderr_thread.join(timeout=2)
            full_stderr = "".join(stderr_lines)

            if ret != 0:
                if (
                    "CUDA_ERROR" in full_stderr or "initialization error" in full_stderr
                ) and retry_count < max_retries - 1:
                    logger.warning(f"⚠️ NVENC init error, retry {retry_count + 1}/{max_retries}")
                    time.sleep(0.5)
                    continue

                logger.error(f"FFmpeg stderr:\n{full_stderr[-1000:]}")
                raise RuntimeError(f"ffmpeg exited with code {ret}") from None

            return video_path

        except subprocess.TimeoutExpired:
            if proc:
                proc.kill()
            logger.error("❌ FFmpeg encoding timeout")
            if retry_count < max_retries - 1:
                continue
            raise RuntimeError("FFmpeg encoding timeout") from None

        except Exception as e:
            if proc and proc.poll() is None:
                proc.kill()
            if retry_count < max_retries - 1:
                logger.warning(f"⚠️ Encoding attempt {retry_count + 1} failed, retrying: {e}")
                time.sleep(0.5)
                continue
            logger.error(f"❌ Encoding failed after {max_retries} retries: {e}")
            raise

    return video_path


def encode_video_nvenc(
    vid: torch.Tensor,
    video_path: str,
    fps: int = 25,
    max_retries: int = 2,
) -> str:
    """Save a video tensor with FFmpeg's hardware NVENC encoder.

    Args:
        vid: Video tensor of shape (T, C, H, W) or (T, H, W, C).
        video_path: Destination path for the MP4 video.
        fps: Frames per second.
        max_retries: Number of retry attempts on hardware encoder failure.

    Returns:
        The output video path.
    """
    t0 = time.perf_counter()
    frames = frames_to_uint8_hwc(vid)
    logger.info(
        "⏱️ Tensor -> numpy: %.3fs (T=%d, H=%d, W=%d, C=%d)",
        time.perf_counter() - t0,
        *frames.shape,
    )

    t1 = time.perf_counter()
    encode_numpy_nvenc(frames, video_path, fps=fps, max_retries=max_retries)
    logger.info(f"✅ Direct FFmpeg encode: {time.perf_counter() - t1:.3f}s")
    return video_path
