import os, torch, random, cv2, librosa, datetime, tempfile, face_alignment

torch.backends.cudnn.benchmark = True
import numpy as np
import albumentations as A
import albumentations.pytorch.transforms as A_pytorch
from tqdm import tqdm
from pathlib import Path
from transformers import Wav2Vec2FeatureExtractor

try:
    from fast_float.models.float.FLOAT import FLOAT
except ImportError:
    from models.float.FLOAT import FLOAT

import contextlib

try:
    from fast_float.schemas.base_options import BaseOptions
except ImportError:
    try:
        from schemas.base_options import BaseOptions
    except ImportError:
        from options.base_options import BaseOptions
import time
import uuid
import logging
import struct
import wave
import subprocess
import torch
import numpy as np
import time
import shutil
import os
import traceback

logger = logging.getLogger(__name__)
import asyncio


def setup_logger(name: str = __name__, level: int = logging.INFO) -> logging.Logger:
    """Setup a logger with console output."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


logger = setup_logger(__name__, level=logging.INFO)


class SimpleVideoEncoder:
    """
    A from-scratch video encoder that creates AVI files with Motion JPEG encoding.
    This is a simple, pure-Python implementation that doesn't rely on ffmpeg.
    """

    def __init__(self, width: int, height: int, fps: int, output_path: str):
        self.width = width
        self.height = height
        self.fps = fps
        self.output_path = output_path
        self.frame_count = 0
        self.frames = []

    def add_frame(self, frame: np.ndarray):
        """Add a frame to the video. Frame should be RGB uint8 numpy array (H, W, 3)"""
        if frame.shape != (self.height, self.width, 3):
            raise ValueError(
                f"Frame shape {frame.shape} doesn't match video size ({self.height}, {self.width}, 3)"
            )

        # Convert RGB to BGR for OpenCV
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self.frames.append(bgr_frame)
        self.frame_count += 1

    def write_avi_with_audio(self, audio_path: str = None):
        """Write frames to AVI file using OpenCV VideoWriter"""
        if self.frame_count == 0:
            raise ValueError("No frames to write!")

        # Use MJPEG codec (widely supported, no external dependencies)
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")

        # Create temporary video file without audio
        temp_video = self.output_path.replace(".mp4", "_temp.avi").replace(
            ".avi", "_temp.avi"
        )

        out = cv2.VideoWriter(temp_video, fourcc, self.fps, (self.width, self.height))

        if not out.isOpened():
            raise RuntimeError("Failed to open video writer")

        logger.info(f"Writing {self.frame_count} frames to {temp_video}...")
        for frame in tqdm(self.frames, desc="Encoding video"):
            out.write(frame)

        out.release()
        logger.info(f"✅ Video encoding complete: {temp_video}")

        # If audio is provided, combine using OpenCV (limited) or just save video-only
        if audio_path and os.path.exists(audio_path):
            logger.warning(
                "Audio merging without ffmpeg is limited. Saving video-only version."
            )
            logger.info(
                f"To add audio manually, use: ffmpeg -i {temp_video} -i {audio_path} -c:v copy -c:a aac {self.output_path}"
            )

        # Rename temp to final output
        if os.path.exists(self.output_path):
            os.remove(self.output_path)
        os.rename(temp_video, self.output_path)

        return self.output_path

    def write(self):
        """Write frames without audio"""
        return self.write_avi_with_audio(audio_path=None)


class DataProcessor:
    def __init__(self, opt):
        self.opt = opt
        self.fps = opt.fps
        self.sampling_rate = opt.sampling_rate
        self.input_size = opt.input_size
        self.fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D, flip_input=False
        )

        if os.path.exists(opt.wav2vec_model_path):
            self.wav2vec_preprocessor = Wav2Vec2FeatureExtractor.from_pretrained(
                opt.wav2vec_model_path, local_files_only=True
            )
        else:
            self.wav2vec_preprocessor = Wav2Vec2FeatureExtractor.from_pretrained(
                opt.wav2vec_model_path
            )

        self.transform = A.Compose(
            [
                A.Resize(
                    height=opt.input_size,
                    width=opt.input_size,
                    interpolation=cv2.INTER_AREA,
                ),
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                A_pytorch.ToTensorV2(),
            ]
        )

    @torch.no_grad()
    def process_img(self, img: np.ndarray) -> np.ndarray:
        mult = 360.0 / img.shape[0]
        resized_img = cv2.resize(
            img,
            dsize=(0, 0),
            fx=mult,
            fy=mult,
            interpolation=cv2.INTER_AREA if mult < 1.0 else cv2.INTER_CUBIC,
        )
        bboxes = self.fa.face_detector.detect_from_image(resized_img)
        bboxes = [
            (int(x1 / mult), int(y1 / mult), int(x2 / mult), int(y2 / mult), score)
            for (x1, y1, x2, y2, score) in bboxes
            if score > 0.95
        ]
        bboxes = bboxes[0]

        bsy = int((bboxes[3] - bboxes[1]) / 2)
        bsx = int((bboxes[2] - bboxes[0]) / 2)
        my = int((bboxes[1] + bboxes[3]) / 2)
        mx = int((bboxes[0] + bboxes[2]) / 2)
        bs = int(max(bsy, bsx) * 1.6)

        img = cv2.copyMakeBorder(img, bs, bs, bs, bs, cv2.BORDER_CONSTANT, value=0)
        my, mx = my + bs, mx + bs
        crop_img = img[my - bs : my + bs, mx - bs : mx + bs]
        crop_img = cv2.resize(
            crop_img,
            dsize=(self.input_size, self.input_size),
            interpolation=cv2.INTER_AREA if mult < 1.0 else cv2.INTER_CUBIC,
        )
        return crop_img

    def default_img_loader(self, path) -> np.ndarray:
        img = cv2.imread(path)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def default_aud_loader(self, path: str) -> torch.Tensor:
        speech_array, sampling_rate = librosa.load(path, sr=self.sampling_rate)
        return self.wav2vec_preprocessor(
            speech_array, sampling_rate=sampling_rate, return_tensors="pt"
        ).input_values[0]

    def preprocess(self, ref_path, audio_path, no_crop: bool = False) -> dict:
        if isinstance(ref_path, (list, tuple)):
            s_list = []
            for rp in ref_path:
                img = self.default_img_loader(rp)
                if not no_crop:
                    img = self.process_img(img)
                s_list.append(self.transform(image=img)["image"])
            s = torch.stack(s_list, dim=0)
        else:
            s = self.default_img_loader(ref_path)
            if not no_crop:
                s = self.process_img(s)
            s = self.transform(image=s)["image"].unsqueeze(0)

        if isinstance(audio_path, (list, tuple)):
            a_list = [self.default_aud_loader(ap) for ap in audio_path]
            max_len = max(aud.shape[-1] for aud in a_list)
            a_padded = [
                F.pad(aud, (0, max_len - aud.shape[-1]), mode="constant", value=0)
                if aud.shape[-1] < max_len
                else aud
                for aud in a_list
            ]
            a = torch.stack(a_padded, dim=0)
        else:
            a = self.default_aud_loader(audio_path).unsqueeze(0)

        return {"s": s, "a": a, "p": None, "e": None}


class OptimizedInferenceAgent:
    def __init__(self, opt):
        torch.cuda.empty_cache()
        self.opt = opt
        self.rank = opt.rank
        self.optimization_mode = (
            opt.optimization_mode
            if hasattr(opt, "optimization_mode")
            else "torch_compile"
        )
        self.use_fp16 = opt.use_fp16 if hasattr(opt, "use_fp16") else False

        # Load Model
        self.load_model()
        self.load_weight(opt.ckpt_path, rank=self.rank)
        self.G.to(self.rank)
        self.G.eval()

        # Apply optimization
        self.apply_optimization()

        # Load Data Processor
        self.data_processor = DataProcessor(opt)

        print(f"✓ Model loaded with {self.optimization_mode} optimization")

    def load_model(self) -> None:
        self.G = FLOAT(self.opt)

    def load_weight(self, checkpoint_path: str, rank: int) -> None:
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        with torch.no_grad():
            for model_name, model_param in self.G.named_parameters():
                if model_name in state_dict:
                    model_param.copy_(state_dict[model_name].to(rank))
                elif "wav2vec2" in model_name:
                    pass
                else:
                    print(f"! Warning; {model_name} not found in state_dict.")
        del state_dict

    def apply_optimization(self):
        """Apply various optimization strategies"""

        if self.optimization_mode == "torch_compile":
            try:
                if hasattr(torch, "compile"):
                    print("Applying torch.compile optimization...")
                    self.G_opt = torch.compile(
                        self.G, mode="max-autotune", fullgraph=False
                    )
                    print("✓ torch.compile applied successfully")
                else:
                    print("! torch.compile not available (requires PyTorch 2.0+)")
                    self.G_opt = self.G
            except Exception as e:
                print(f"! torch.compile failed: {e}")
                self.G_opt = self.G

        elif self.optimization_mode == "channels_last":
            print("Applying channels_last memory format...")
            try:
                for module in self.G.modules():
                    if isinstance(module, (torch.nn.Conv2d, torch.nn.Conv3d)):
                        module = module.to(memory_format=torch.channels_last)
                self.G_opt = self.G
                print("✓ Channels last format applied")
            except Exception as e:
                print(f"! Channels last conversion failed: {e}, skipping...")
                self.G_opt = self.G

        elif self.optimization_mode == "amp":
            print("Using Automatic Mixed Precision (AMP)...")
            self.G_opt = self.G
            self.use_amp = True
            print("✓ AMP enabled")

        elif self.optimization_mode == "cudnn_benchmark":
            print("Enabling cuDNN benchmark mode...")
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            self.G_opt = self.G
            print("✓ cuDNN benchmark enabled")

        elif self.optimization_mode == "all":
            print("Applying all optimizations...")

            try:
                for module in self.G.modules():
                    if isinstance(module, (torch.nn.Conv2d, torch.nn.Conv3d)):
                        module = module.to(memory_format=torch.channels_last)
                print("✓ Channels last applied to conv layers")
            except Exception as e:
                print(f"! Channels last skipped: {e}")

            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            print("✓ cuDNN benchmark enabled")

            if hasattr(torch, "compile"):
                try:
                    self.G_opt = torch.compile(self.G, mode="max-autotune")
                    print("✓ All optimizations applied")
                except Exception as e:
                    print(f"! torch.compile failed: {e}")
                    self.G_opt = self.G
                    print("✓ Partial optimizations applied")
            else:
                self.G_opt = self.G
                print("✓ Partial optimizations applied")

        else:
            print("No optimization applied")
            self.G_opt = self.G

        if self.use_fp16:
            print("Enabling FP16 inference...")
            self.use_amp_fp16 = True
            print("✓ FP16 will be applied via autocast during inference")

    def save_video(
        self,
        vid: "torch.Tensor",
        video_path: str,
        audio_path: str,
        video_encoder: Optional[str] = None,
    ) -> str:
        """
        Save video with proper CUDA synchronization to prevent corruption.
        Supports FFmpeg NVENC, CPU PyAV, and Torchcodec alternatives.
        """
        try:
            start_time = time.time()

            encoder_choice = str(video_encoder or getattr(self.opt, "video_encoder", "FFmpeg NVENC")).strip().lower()
            if encoder_choice in ("cpu pyav", "pyav", "pyav_cpu"):
                try:
                    from fast_float.utils.pyav_writer import save_video_pyav
                except ImportError:
                    from utils.pyav_writer import save_video_pyav
                fps = int(getattr(self.opt, "fps", 25))
                return save_video_pyav(vid, video_path, audio_path=audio_path, fps=fps)

            if encoder_choice in ("torchcodec", "torch codec", "tc"):
                try:
                    from fast_float.utils.pyav_writer import save_video_torchcodec
                except ImportError:
                    from utils.pyav_writer import save_video_torchcodec
                fps = int(getattr(self.opt, "fps", 25))
                return save_video_torchcodec(vid, video_path, audio_path=audio_path, fps=fps)

            # Sanity: ffmpeg present
            if shutil.which("ffmpeg") is None:
                raise RuntimeError("ffmpeg not found in PATH")

            # ===== FIX 1: Proper GPU synchronization =====
            if torch.cuda.is_available():
                torch.cuda.synchronize()  # FIXED: was just "1"
                time.sleep(0.05)

            # Step 1: Prepare tensor -> CPU uint8 numpy efficiently
            t0 = time.time()

            # 1: permute & make contiguous on GPU
            if vid.is_cuda:
                vid = vid.permute(0, 2, 3, 1).contiguous()  # (T, H, W, C)

            # 2: convert to uint8 on GPU
            print("vid is floating:", torch.is_floating_point(vid))
            if torch.is_floating_point(vid):
                vid_uint8_gpu = (vid.clamp(-1.0, 1.0) * 127.5 + 127.5).to(torch.uint8)
            else:
                vid_uint8_gpu = vid.to(torch.uint8)

            # 3: move to CPU with synchronization
            if torch.cuda.is_available():
                torch.cuda.synchronize()  # FIXED: ensure GPU work is done

            vid_np = vid_uint8_gpu.cpu().numpy()

            T, H, W, C = vid_np.shape

            # Verify numpy array is valid
            if not vid_np.flags["C_CONTIGUOUS"]:
                vid_np = np.ascontiguousarray(vid_np)

            if C not in (3, 4):
                raise ValueError(f"Expected 3 or 4 channels, got {C}")

            fps = int(getattr(self.opt, "fps", 25))
            tensor_prep_time = time.time() - t0
            logger.info(
                f"⏱️ Tensor -> numpy: {tensor_prep_time:.3f}s (T={T}, H={H}, W={W}, C={C})"
            )

            # Build ffmpeg command
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
                "-",  # read video from stdin
            ]

            # More conservative NVENC settings
            vcodec = "h264_nvenc"
            encode_opts = [
                "-c:v",
                vcodec,
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

            # ===== FIX 2: Better error handling =====
            encode_start = time.time()
            max_retries = 2

            for retry_count in range(max_retries):
                proc = None
                try:
                    proc = subprocess.Popen(
                        cmd,
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,  # Capture stdout too
                        stderr=subprocess.PIPE,
                        bufsize=65536,
                    )

                    # ===== FIX 3: Monitor stderr in separate thread =====
                    stderr_lines = []

                    def read_stderr():
                        for line in proc.stderr:
                            line_str = line.decode(errors="ignore")
                            stderr_lines.append(line_str)
                            if (
                                "error" in line_str.lower()
                                or "failed" in line_str.lower()
                            ):
                                logger.warning(f"FFmpeg stderr: {line_str.strip()}")

                    import threading

                    stderr_thread = threading.Thread(target=read_stderr, daemon=True)
                    stderr_thread.start()

                    # Write frames
                    write_start = time.time()

                    for i in range(T):
                        frame = vid_np[i]
                        frame_bytes = frame.tobytes()

                        try:
                            proc.stdin.write(frame_bytes)
                        except BrokenPipeError:
                            # FFmpeg died - check why
                            proc.wait(timeout=5)
                            stderr_thread.join(timeout=1)
                            full_stderr = "".join(stderr_lines)
                            logger.error(f"❌ FFmpeg died at frame {i}/{T}")
                            logger.error(
                                f"FFmpeg stderr:\n{full_stderr[-1000:]}"
                            )  # Last 1000 chars
                            raise RuntimeError(
                                f"FFmpeg broken pipe at frame {i}: {full_stderr[-500:]}"
                            )

                        # Flush periodically
                        if i % 100 == 0:
                            proc.stdin.flush()

                    proc.stdin.close()

                    # Wait for completion
                    ret = proc.wait(timeout=300)
                    stderr_thread.join(timeout=2)
                    full_stderr = "".join(stderr_lines)

                    if ret != 0:
                        # Check if NVENC initialization error
                        if (
                            "CUDA_ERROR" in full_stderr
                            or "initialization error" in full_stderr
                        ):
                            if retry_count < max_retries - 1:
                                logger.warning(
                                    f"⚠️ NVENC init error, retry {retry_count + 1}/{max_retries}"
                                )
                                time.sleep(0.5)
                                if torch.cuda.is_available():
                                    torch.cuda.synchronize()
                                continue

                        logger.error(f"FFmpeg stderr:\n{full_stderr[-1000:]}")
                        raise RuntimeError(f"ffmpeg exited with code {ret}")

                    encode_time = time.time() - encode_start
                    logger.info(f"✅ Direct FFmpeg encode: {encode_time:.3f}s")
                    break  # Success!

                except subprocess.TimeoutExpired:
                    if proc:
                        proc.kill()
                    logger.error("❌ FFmpeg encoding timeout")
                    if retry_count < max_retries - 1:
                        continue
                    raise RuntimeError("FFmpeg encoding timeout")

                except Exception as e:
                    if proc and proc.poll() is None:
                        proc.kill()

                    if retry_count < max_retries - 1:
                        logger.warning(
                            f"⚠️ Encoding attempt {retry_count + 1} failed, retrying..."
                        )
                        time.sleep(0.5)
                        continue

                    logger.error(f"❌ Encoding failed after {max_retries} retries: {e}")
                    raise

            total_time = time.time() - start_time
            logger.info(f"✅ Total save: {total_time:.3f}s -> {video_path}")
            return video_path

        except Exception as e:
            logger.error(f"Error: {e}\n{traceback.format_exc()}")
            raise

    @torch.no_grad()
    def run_inference(
        self,
        res_video_path: str,
        ref_path: str,
        audio_path: str,
        a_cfg_scale: float = 2.0,
        r_cfg_scale: float = 1.0,
        e_cfg_scale: float = 1.0,
        emo: str = "S2E",
        nfe: int = 10,
        no_crop: bool = False,
        seed: int = 25,
        verbose: bool = True,
        benchmark: bool = False,
        video_encoder: Optional[str] = None,
    ) -> str:
        timings = {}
        total_start = time.perf_counter()

        # ---- preprocess ----
        t0 = time.perf_counter()
        data = self.data_processor.preprocess(ref_path, audio_path, no_crop=no_crop)
        timings["preprocess"] = time.perf_counter() - t0
        if verbose:
            print(f"> [Done] Preprocess ({timings['preprocess']:.3f}s)")

        gen_kwargs = dict(
            a_cfg_scale=a_cfg_scale,
            r_cfg_scale=r_cfg_scale,
            e_cfg_scale=e_cfg_scale,
            emo=emo,
            nfe=nfe,
            seed=seed,
        )

        # Autocast: same behaviour as before. Only use_amp_fp16 turns it on
        # (autocast(dtype=float32) is a no-op on CUDA). stream_pipeline copies this autocast
        # state into its sampler/decode threads, so wrapping the call here is enough.
        def make_amp():
            if getattr(self, "use_amp_fp16", False):
                return torch.autocast("cuda", dtype=torch.float16)
            return contextlib.nullcontext()

        def generate(out_path: str) -> str:
            """Sampling + decoding + video encoding + audio mux, fully overlapped."""
            with make_amp():
                return self.G_opt.inference_and_save(
                    data=data,
                    video_path=out_path,
                    audio_path=audio_path,
                    video_encoder=video_encoder,
                    **gen_kwargs,
                )

        # ---- benchmark (measures the whole generate+encode pipeline) ----
        if benchmark:
            warmup_runs, benchmark_runs = 3, 10
            fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
            os.close(fd)
            try:
                print("Warming up (first run also runs the one-time GPU auto-tune)...")
                for _ in range(warmup_runs):
                    generate(tmp_path)

                print(f"Running {benchmark_runs} benchmark iterations...")
                frames = 0
                t0 = time.perf_counter()
                for _ in range(benchmark_runs):
                    generate(tmp_path)
                    frames += self.G_opt.last_stats["frames"]
                elapsed = time.perf_counter() - t0
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

            print(
                f"Average generate+encode time: {elapsed / benchmark_runs * 1000:.2f} ms/run"
            )
            print(f"Throughput: {frames / elapsed:.2f} frames/s")

        # ---- actual run ----
        first_run = getattr(self.G_opt, "_decode_profiles", None) is None
        t0 = time.perf_counter()
        res_video_path = generate(res_video_path)
        timings["generate_and_save"] = time.perf_counter() - t0
        timings["total"] = time.perf_counter() - total_start

        stats = getattr(self.G_opt, "last_stats", {})
        frames = stats.get("frames", 0)

        if verbose:
            gs = timings["generate_and_save"]
            note = "  (includes one-time GPU auto-tune)" if first_run else ""
            print(f"> [Done] Generate + save ({gs:.3f}s){note}")
            print(f"\n{'=' * 50}")
            print("⏱️  PERFORMANCE SUMMARY")
            print(f"{'=' * 50}")
            print(f"  Preprocess:       {timings['preprocess']:>8.3f}s")
            print(f"  Generate + save:  {gs:>8.3f}s")
            print(f"  {'─' * 40}")
            print(f"  Total Time:       {timings['total']:>8.3f}s")
            if frames:
                print(f"  Frames:           {frames:>8d}")
                print(f"  FPS (gen+save):   {frames / gs:>8.2f}")
            print(f"{'=' * 50}")
            print(f"> Result saved at {res_video_path}")

        return res_video_path

    @torch.no_grad()
    def run_inference_old(
        self,
        res_video_path: str,
        ref_path: str,
        audio_path: str,
        a_cfg_scale: float = 2.0,
        r_cfg_scale: float = 1.0,
        e_cfg_scale: float = 1.0,
        emo: str = "S2E",
        nfe: int = 10,
        no_crop: bool = False,
        seed: int = 25,
        verbose: bool = True,
        benchmark: bool = False,
    ) -> str:

        timings = {}
        total_start = time.time()

        # Preprocess data
        preprocess_start = time.time()
        data = self.data_processor.preprocess(ref_path, audio_path, no_crop=no_crop)
        timings["preprocess"] = time.time() - preprocess_start

        if verbose:
            print(f"> [Done] Preprocess ({timings['preprocess']:.3f}s)")

        # Benchmark mode
        if benchmark:
            warmup_runs = 3
            benchmark_runs = 10

            print("Warming up...")
            for _ in range(warmup_runs):
                _ = self.G_opt.inference(
                    data=data,
                    a_cfg_scale=a_cfg_scale,
                    r_cfg_scale=r_cfg_scale,
                    e_cfg_scale=e_cfg_scale,
                    emo=emo,
                    nfe=nfe,
                    seed=seed,
                )

            1

            print(f"Running {benchmark_runs} benchmark iterations...")
            start_time = time.time()
            for _ in range(benchmark_runs):
                _ = self.G_opt.inference(
                    data=data,
                    a_cfg_scale=a_cfg_scale,
                    r_cfg_scale=r_cfg_scale,
                    e_cfg_scale=e_cfg_scale,
                    emo=emo,
                    nfe=nfe,
                    seed=seed,
                )

            end_time = time.time()

            avg_time = (end_time - start_time) / benchmark_runs
            print(f"Average inference time: {avg_time*1000:.2f} ms")
            print(f"FPS: {1.0/avg_time:.2f}")

        # Actual inference
        use_amp = hasattr(self, "use_amp") and self.use_amp
        use_amp_fp16 = hasattr(self, "use_amp_fp16") and self.use_amp_fp16

        inference_start = time.time()
        1 if torch.cuda.is_available() else None

        if use_amp or use_amp_fp16:
            with torch.cuda.amp.autocast(
                dtype=torch.float16 if use_amp_fp16 else torch.float32
            ):
                d_hat = self.G_opt.inference(
                    data=data,
                    a_cfg_scale=a_cfg_scale,
                    r_cfg_scale=r_cfg_scale,
                    e_cfg_scale=e_cfg_scale,
                    emo=emo,
                    nfe=nfe,
                    seed=seed,
                )["d_hat"]
        else:
            d_hat = self.G_opt.inference(
                data=data,
                a_cfg_scale=a_cfg_scale,
                r_cfg_scale=r_cfg_scale,
                e_cfg_scale=e_cfg_scale,
                emo=emo,
                nfe=nfe,
                seed=seed,
            )["d_hat"]

        1 if torch.cuda.is_available() else None
        timings["inference"] = time.time() - inference_start

        if verbose:
            print(f"> [Done] Inference ({timings['inference']:.3f}s)")

        # Save result
        save_start = time.time()
        res_video_path = self.save_video(d_hat.float(), res_video_path, audio_path)
        timings["save"] = time.time() - save_start

        timings["total"] = time.time() - total_start

        if verbose:
            print(f"> [Done] Save video ({timings['save']:.3f}s)")
            print(f"\n{'='*50}")
            print(f"⏱️  PERFORMANCE SUMMARY")
            print(f"{'='*50}")
            print(f"  Preprocess:  {timings['preprocess']:>8.3f}s")
            print(f"  Inference:   {timings['inference']:>8.3f}s")
            print(f"  Save Video:  {timings['save']:>8.3f}s")
            print(f"  {'─'*40}")
            print(f"  Total Time:  {timings['total']:>8.3f}s")
            print(f"  FPS:         {1.0/timings['inference']:>8.2f}")
            print(f"{'='*50}")
            print(f"> Result saved at {res_video_path}")

        return res_video_path


class InferenceOptions(BaseOptions):
    def __init__(self):
        super().__init__()

    def initialize(self, parser):
        super().initialize(parser)
        parser.add_argument(
            "--ref_path", default=None, type=str, help="reference image"
        )
        parser.add_argument("--aud_path", default=None, type=str, help="audio file")
        parser.add_argument(
            "--emo",
            default=None,
            type=str,
            help="emotion",
            choices=["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"],
        )
        parser.add_argument("--no_crop", action="store_true", help="skip face cropping")
        parser.add_argument(
            "--res_video_path", default=None, type=str, help="output video path"
        )
        parser.add_argument(
            "--ckpt_path",
            default="./checkpoints/float.pth",
            type=str,
            help="checkpoint path",
        )
        parser.add_argument(
            "--res_dir", default="./results", type=str, help="result directory"
        )

        parser.add_argument(
            "--optimization_mode",
            default="all",
            type=str,
            choices=[
                "none",
                "torch_compile",
                "channels_last",
                "amp",
                "cudnn_benchmark",
                "all",
            ],
            help="Optimization strategy to use",
        )
        parser.add_argument(
            "--use_fp16", action="store_true", help="Use FP16 precision"
        )
        parser.add_argument(
            "--benchmark", action="store_true", help="Run benchmark mode"
        )
        parser.add_argument(
            "--video_encoder",
            default="FFmpeg NVENC",
            type=str,
            choices=["FFmpeg NVENC", "CPU PyAV", "Torchcodec"],
            help="Video encoding backend ('FFmpeg NVENC', 'CPU PyAV', or 'Torchcodec')",
        )

        return parser


if __name__ == "__main__":
    opt = InferenceOptions().parse()
    opt.rank = 0

    print(f"\n{'='*60}")
    print(f"FLOAT Optimized Inference (No FFmpeg)")
    print(f"Optimization: {opt.optimization_mode}")
    print(f"FP16: {opt.use_fp16}")
    print(f"{'='*60}\n")

    agent = OptimizedInferenceAgent(opt)

    os.makedirs(opt.res_dir, exist_ok=True)

    ref_path = opt.ref_path
    aud_path = opt.aud_path

    if opt.res_video_path is None:
        video_name = os.path.splitext(os.path.basename(ref_path))[0]
        audio_name = os.path.splitext(os.path.basename(aud_path))[0]
        call_time = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        opt_suffix = f"{opt.optimization_mode}"
        if opt.use_fp16:
            opt_suffix += "-fp16"
        res_video_path = os.path.join(
            opt.res_dir,
            "%s-%s-%s-nfe%s-seed%s-acfg%s-ecfg%s-%s-%s.avi"
            % (
                call_time,
                video_name,
                audio_name,
                opt.nfe,
                opt.seed,
                opt.a_cfg_scale,
                opt.e_cfg_scale,
                opt.emo,
                opt_suffix,
            ),
        )
    else:
        res_video_path = opt.res_video_path

    agent.run_inference(
        res_video_path,
        ref_path,
        aud_path,
        a_cfg_scale=opt.a_cfg_scale,
        r_cfg_scale=opt.r_cfg_scale,
        e_cfg_scale=opt.e_cfg_scale,
        emo=opt.emo,
        nfe=opt.nfe,
        no_crop=opt.no_crop,
        seed=opt.seed,
        verbose=True,
        benchmark=opt.benchmark,
    )
