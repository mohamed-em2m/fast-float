import contextlib
import os
import shutil
import subprocess
import tempfile
import time
import uuid

from fast_float.utils.logger import logger


class VideoSaveOptimizer:
    """Smart video encoder with hardware acceleration autodetection and CPU fallback."""

    def __init__(self):
        self.encoder_type = None
        self.selected_encoder = None
        self._check_nvidia_gpu()
        self.encoder_available = self._detect_best_encoder()
        logger.info(f"🎬 Video Encoder: {self.encoder_type}")

    def _check_nvidia_gpu(self) -> bool:
        """Check if NVIDIA GPU and drivers are present."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=3,
            )
            if result.returncode == 0 and result.stdout.strip():
                gpu_name = result.stdout.strip()
                logger.info(f"🎮 NVIDIA GPU detected: {gpu_name}")

                result2 = subprocess.run(
                    ["nvidia-smi", "--query-gpu=encoder.h264", "--format=csv,noheader"],
                    capture_output=True,
                    text=True,
                    timeout=3,
                )
                if result2.returncode == 0:
                    logger.info(f"   NVENC support: {result2.stdout.strip()}")
                return True
            else:
                logger.warning("⚠️ No NVIDIA GPU detected (nvidia-smi returned no output)")
                return False
        except FileNotFoundError:
            logger.warning("⚠️ nvidia-smi not found - NVIDIA drivers may not be installed")
            return False
        except Exception as e:
            logger.warning(f"⚠️ GPU check failed: {e}")
            return False

    def _detect_best_encoder(self) -> bool:
        """Auto-detect best available video encoder (NVENC > QSV > VideoToolbox > AMF > libx264)."""
        gpu_encoders = [
            ("h264_nvenc", "NVENC (NVIDIA GPU)"),
            ("hevc_nvenc", "NVENC HEVC (NVIDIA GPU)"),
            ("h264_qsv", "QuickSync (Intel GPU)"),
            ("h264_videotoolbox", "VideoToolbox (Apple)"),
            ("h264_amf", "AMF (AMD GPU)"),
        ]

        try:
            result = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                text=True,
                timeout=3,
            )
            if result.returncode != 0:
                logger.error("❌ FFmpeg not found!")
                self.encoder_type = "libx264 (CPU - fallback)"
                self.selected_encoder = "libx264"
                return False

            result = subprocess.run(
                ["ffmpeg", "-hide_banner", "-encoders"],
                capture_output=True,
                text=True,
                timeout=3,
            )
            encoders_output = result.stdout
            logger.info("🔍 Checking available GPU encoders...")

            for encoder, name in gpu_encoders:
                if encoder in encoders_output:
                    logger.info(f"   ✓ Found {encoder} in encoders list")
                    if self._test_encoder(encoder):
                        logger.info(f"   ✓ {encoder} test PASSED")
                        self.encoder_type = name
                        self.selected_encoder = encoder
                        return True
                    else:
                        logger.warning(f"   ✗ {encoder} test FAILED (encoder exists but cannot encode)")
                else:
                    logger.debug(f"   ✗ {encoder} not available")

            logger.warning("⚠️ No working GPU encoder found, falling back to CPU")
            if "libx264" in encoders_output:
                self.encoder_type = "libx264 (CPU - ultrafast)"
                self.selected_encoder = "libx264"
                return True

            logger.error("❌ No suitable video encoder found!")
            return False

        except FileNotFoundError:
            logger.error("❌ FFmpeg not installed or not in PATH!")
            self.encoder_type = "libx264 (CPU - fallback)"
            self.selected_encoder = "libx264"
            return False
        except Exception as e:
            logger.error(f"❌ Encoder detection failed: {e}")
            self.encoder_type = "libx264 (CPU - fallback)"
            self.selected_encoder = "libx264"
            return False

    def _test_encoder(self, encoder: str) -> bool:
        """Test whether the encoder works with a minimal color frame."""
        try:
            logger.info(f"      🧪 Testing {encoder}...")
            test_cmd = [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=256x256:d=0.04",
                "-c:v",
                encoder,
                "-frames:v",
                "1",
                "-f",
                "null",
                "-",
            ]
            result = subprocess.run(
                test_cmd,
                capture_output=True,
                timeout=10,
            )
            if result.returncode == 0:
                logger.info(f"      ✅ {encoder} working!")
                return True
            else:
                error_output = result.stderr.decode("utf-8", errors="ignore") if result.stderr else "Unknown error"
                error_lines = [
                    line for line in error_output.split("\n") if "error" in line.lower() or "failed" in line.lower()
                ]
                error_msg = error_lines[0] if error_lines else error_output[:200]
                logger.warning(f"      ❌ {encoder} failed: {error_msg}")
                return False
        except subprocess.TimeoutExpired:
            logger.warning(f"      ⏱️ {encoder} test timeout")
            return False
        except Exception as e:
            logger.warning(f"      ❌ {encoder} test error: {e}")
            return False

    def save_video_optimized(
        self,
        input_path: str,
        output_path: str,
        fps: int = 25,
        quality: str = "fast",
        resolution: tuple[int, int] = (256, 256),
    ) -> tuple[bool, float, str]:
        """Convert or compress video using detected hardware or CPU encoder."""
        start_time = time.time()
        width, height = resolution

        # tmp location (prefer tmpfs on Linux, tempdir on Windows)
        tmp_dir = "/dev/shm" if os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK) else tempfile.gettempdir()
        tmp_out = os.path.join(tmp_dir, f"tmp_opt_{uuid.uuid4().hex}.mp4")

        use_gpu = bool(self.selected_encoder and "nvenc" in self.selected_encoder)

        if use_gpu:
            cmd = [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-hwaccel",
                "cuda",
                "-hwaccel_output_format",
                "cuda",
                "-i",
                input_path,
                "-vf",
                f"hwupload_cuda,scale_cuda={width}:{height}:interp_algo=linear,fps={fps}",
                "-c:v",
                "h264_nvenc",
                "-preset",
                "p1",
                "-rc",
                "constqp",
                "-qp",
                "28",
                "-profile:v",
                "baseline",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-b:a",
                "64k",
                "-movflags",
                "+faststart",
                tmp_out,
            ]
            encoder_label = "h264_nvenc_p1"
        else:
            cmd = [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                input_path,
                "-vf",
                f"scale={width}:{height}:flags=bilinear,fps={fps}",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-tune",
                "zerolatency",
                "-crf",
                "30",
                "-threads",
                "0",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-b:a",
                "64k",
                "-movflags",
                "+faststart",
                tmp_out,
            ]
            encoder_label = "libx264_ultrafast"

        try:
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                timeout=120,
            )
            elapsed = time.time() - start_time

            try:
                os.replace(tmp_out, output_path)
            except Exception:
                shutil.copy(tmp_out, output_path)
                if os.path.exists(tmp_out):
                    os.remove(tmp_out)

            return True, elapsed, encoder_label

        except subprocess.CalledProcessError as e:
            err = e.stderr or ""
            if use_gpu:
                logger.warning(f"⚠️ NVENC encode failed: {err.splitlines()[:5]} -- retrying with CPU fallback")
                self.selected_encoder = "libx264"
                return self.save_video_optimized(input_path, output_path, fps, quality, resolution)
            logger.error(f"❌ Encoding failed: {err}")
            return False, time.time() - start_time, "failed"

        except subprocess.TimeoutExpired:
            logger.error("❌ Encoding timeout")
            return False, time.time() - start_time, "timeout"

        except Exception as e:
            logger.error(f"❌ Unexpected error during encoding: {e}")
            return False, time.time() - start_time, "error"
        finally:
            if os.path.exists(tmp_out):
                with contextlib.suppress(Exception):
                    os.remove(tmp_out)
