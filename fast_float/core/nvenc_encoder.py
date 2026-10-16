"""FFmpeg NVENC video writer with sub-process management and retry logic."""

from __future__ import annotations

import logging
import shutil
import subprocess
import threading
import time
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


def encode_video_nvenc(
    vid: torch.Tensor,
    video_path: str,
    fps: int = 25,
    max_retries: int = 2,
) -> str:
    """Save video using FFmpeg hardware NVENC encoder with CUDA synchronization.

    Args:
        vid: Video tensor of shape (T, C, H, W) or (T, H, W, C).
        video_path: Destination path for the MP4 video.
        fps: Frames per second.
        max_retries: Number of retry attempts on hardware encoder failure.

    Returns:
        The output video path.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found in PATH")

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        time.sleep(0.05)

    t0 = time.time()

    # Ensure shape (T, H, W, C)
    if vid.is_cuda:
        vid = vid.permute(0, 2, 3, 1).contiguous()

    if torch.is_floating_point(vid):
        vid_uint8_gpu = (vid.clamp(-1.0, 1.0) * 127.5 + 127.5).to(torch.uint8)
    else:
        vid_uint8_gpu = vid.to(torch.uint8)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    vid_np = vid_uint8_gpu.cpu().numpy()
    T, H, W, C = vid_np.shape

    if not vid_np.flags["C_CONTIGUOUS"]:
        vid_np = np.ascontiguousarray(vid_np)

    if C not in (3, 4):
        raise ValueError(f"Expected 3 or 4 channels, got {C}")

    tensor_prep_time = time.time() - t0
    logger.info(f"⏱️ Tensor -> numpy: {tensor_prep_time:.3f}s (T={T}, H={H}, W={W}, C={C})")

    pix_fmt_in = "rgba" if C == 4 else "rgb24"
    size_str = f"{W}x{H}"

    base_cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        pix_fmt_in,
        "-s",
        size_str,
        "-r",
        str(fps),
        "-i",
        "-",
    ]

    encode_opts = [
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

    cmd = base_cmd + encode_opts + [video_path]
    encode_start = time.time()

    for retry_count in range(max_retries):
        proc: subprocess.Popen | None = None
        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=65536,
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

            for i in range(T):
                frame = vid_np[i]
                try:
                    assert proc.stdin is not None
                    proc.stdin.write(frame.tobytes())
                except BrokenPipeError:
                    proc.wait(timeout=5)
                    stderr_thread.join(timeout=1)
                    full_stderr = "".join(stderr_lines)
                    logger.error(f"❌ FFmpeg died at frame {i}/{T}")
                    logger.error(f"FFmpeg stderr:\n{full_stderr[-1000:]}")
                    raise RuntimeError(f"FFmpeg broken pipe at frame {i}: {full_stderr[-500:]}") from None

                if i % 100 == 0:
                    assert proc.stdin is not None
                    proc.stdin.flush()

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
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    continue

                logger.error(f"FFmpeg stderr:\n{full_stderr[-1000:]}")
                raise RuntimeError(f"ffmpeg exited with code {ret}") from None

            encode_time = time.time() - encode_start
            logger.info(f"✅ Direct FFmpeg encode: {encode_time:.3f}s")
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
