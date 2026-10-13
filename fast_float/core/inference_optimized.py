"""Optimized inference agent for FLOAT with multi-backend encoding and GPU optimizations."""

from __future__ import annotations

import contextlib
import datetime
import logging
import os
import tempfile
import time
import traceback
from typing import Any, Optional

import torch

try:
    from fast_float.models.float.FLOAT import FLOAT
except ImportError:
    from models.float.FLOAT import FLOAT

try:
    from fast_float.utils.pyav_writer import save_video_pyav, save_video_torchcodec
except ImportError:
    from utils.pyav_writer import save_video_pyav, save_video_torchcodec

from fast_float.core.data_processor import DataProcessor
from fast_float.core.nvenc_encoder import encode_video_nvenc
from fast_float.core.options import InferenceOptions
from fast_float.core.simple_encoder import SimpleVideoEncoder

torch.backends.cudnn.benchmark = True

logger = logging.getLogger(__name__)


class OptimizedInferenceAgent:
    """Manages optimized inference execution, model weights, optimizations, and video encoding."""

    def __init__(self, opt: Any) -> None:
        torch.cuda.empty_cache()
        self.opt = opt
        self.rank = getattr(opt, "rank", 0)
        self.optimization_mode = getattr(opt, "optimization_mode", "torch_compile")
        self.use_fp16 = getattr(opt, "use_fp16", False)

        # Load Model
        self.load_model()
        self.load_weight(opt.ckpt_path, rank=self.rank)
        self.G.to(self.rank)
        self.G.eval()

        # Apply optimization
        self.apply_optimization()

        # Load Data Processor
        self.data_processor = DataProcessor(opt)

        logger.info(f"✓ Model loaded with {self.optimization_mode} optimization")

    def load_model(self) -> None:
        self.G = FLOAT(self.opt)

    def load_weight(self, checkpoint_path: str, rank: int) -> None:
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        with torch.no_grad():
            for model_name, model_param in self.G.named_parameters():
                clean_name = model_name.replace("._orig_mod", "")
                if model_name in state_dict:
                    model_param.copy_(state_dict[model_name].to(rank))
                elif clean_name in state_dict:
                    model_param.copy_(state_dict[clean_name].to(rank))
                elif "wav2vec2" in model_name or "wav2vec2" in clean_name:
                    pass
                else:
                    logger.warning(f"! Warning: {clean_name} not found in state_dict.")
        del state_dict

    def apply_optimization(self) -> None:
        """Apply various GPU optimization strategies."""
        compile_mode = getattr(self.opt, "compile_mode", "default")

        if self.optimization_mode == "torch_compile":
            try:
                if hasattr(torch, "compile"):
                    logger.info(f"Applying torch.compile optimization (mode='{compile_mode}')...")
                    self.G.compile_models(mode=compile_mode)
                    self.G_opt = self.G
                    logger.info("✓ torch.compile applied to FLOAT submodules successfully")
                else:
                    logger.warning(
                        "! torch.compile not available (requires PyTorch 2.0+)"
                    )
                    self.G_opt = self.G
            except Exception as e:
                logger.warning(f"! torch.compile failed: {e}")
                self.G_opt = self.G

        elif self.optimization_mode == "channels_last":
            logger.info("Applying channels_last memory format...")
            try:
                for module in self.G.modules():
                    if isinstance(module, (torch.nn.Conv2d, torch.nn.Conv3d)):
                        module = module.to(memory_format=torch.channels_last)
                self.G_opt = self.G
                logger.info("✓ Channels last format applied")
            except Exception as e:
                logger.warning(f"! Channels last conversion failed: {e}, skipping...")
                self.G_opt = self.G

        elif self.optimization_mode == "amp":
            logger.info("Using Automatic Mixed Precision (AMP)...")
            self.G_opt = self.G
            self.use_amp = True
            logger.info("✓ AMP enabled")

        elif self.optimization_mode == "cudnn_benchmark":
            logger.info("Enabling cuDNN benchmark mode...")
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            self.G_opt = self.G
            logger.info("✓ cuDNN benchmark enabled")

        elif self.optimization_mode == "all":
            logger.info("Applying all optimizations...")
            try:
                for module in self.G.modules():
                    if isinstance(module, (torch.nn.Conv2d, torch.nn.Conv3d)):
                        module = module.to(memory_format=torch.channels_last)
                logger.info("✓ Channels last applied to conv layers")
            except Exception as e:
                logger.warning(f"! Channels last skipped: {e}")

            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            logger.info("✓ cuDNN benchmark enabled")

            if hasattr(torch, "compile"):
                try:
                    self.G.compile_models(mode=compile_mode)
                    logger.info("✓ torch.compile applied to FLOAT submodules")
                except Exception as e:
                    logger.warning(f"! torch.compile failed: {e}")
            self.G_opt = self.G
            logger.info("✓ All optimizations applied")
        else:
            logger.info("No optimization applied")
            self.G_opt = self.G

        if self.use_fp16:
            logger.info("Enabling FP16 inference...")
            self.use_amp_fp16 = True
            logger.info("✓ FP16 will be applied via autocast during inference")

    def save_video(
        self,
        vid: torch.Tensor,
        video_path: str,
        audio_path: str,
        video_encoder: Optional[str] = None,
    ) -> str:
        """Save video supporting FFmpeg NVENC, CPU PyAV, and Torchcodec alternatives."""
        try:
            start_time = time.time()
            encoder_choice = (
                str(video_encoder or getattr(self.opt, "video_encoder", "FFmpeg NVENC"))
                .strip()
                .lower()
            )
            fps = int(getattr(self.opt, "fps", 25))

            if encoder_choice in ("cpu pyav", "pyav", "pyav_cpu"):
                return save_video_pyav(vid, video_path, audio_path=audio_path, fps=fps)

            if encoder_choice in ("torchcodec", "torch codec", "tc"):
                try:
                    return save_video_torchcodec(
                        vid, video_path, audio_path=audio_path, fps=fps
                    )
                except Exception as e:
                    # torchcodec missing or binary-incompatible with installed
                    # torch (OSError: undefined symbol) — fall back to PyAV.
                    logger.warning(
                        f"⚠️ torchcodec writer failed ({e}); falling back to PyAV."
                    )
                    return save_video_pyav(
                        vid, video_path, audio_path=audio_path, fps=fps
                    )

            # Default: FFmpeg NVENC
            encode_video_nvenc(vid, video_path, fps=fps)

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
        timings: dict[str, float] = {}
        total_start = time.perf_counter()

        # ---- preprocess ----
        t0 = time.perf_counter()
        data = self.data_processor.preprocess(ref_path, audio_path, no_crop=no_crop)
        timings["preprocess"] = time.perf_counter() - t0
        if verbose:
            logger.info(f"> [Done] Preprocess ({timings['preprocess']:.3f}s)")

        gen_kwargs = dict(
            a_cfg_scale=a_cfg_scale,
            r_cfg_scale=r_cfg_scale,
            e_cfg_scale=e_cfg_scale,
            emo=emo,
            nfe=nfe,
            seed=seed,
        )

        def make_amp() -> contextlib.AbstractContextManager[None]:
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

        # ---- benchmark ----
        if benchmark:
            warmup_runs, benchmark_runs = 3, 10
            fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
            os.close(fd)
            try:
                logger.info(
                    "Warming up (first run also runs the one-time GPU auto-tune)..."
                )
                for _ in range(warmup_runs):
                    generate(tmp_path)

                logger.info(f"Running {benchmark_runs} benchmark iterations...")
                frames = 0
                t_bench = time.perf_counter()
                for _ in range(benchmark_runs):
                    generate(tmp_path)
                    frames += self.G_opt.last_stats["frames"]
                elapsed = time.perf_counter() - t_bench
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

            logger.info(
                f"Average generate+encode time: {elapsed / benchmark_runs * 1000:.2f} ms/run"
            )
            logger.info(f"Throughput: {frames / elapsed:.2f} frames/s")

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
            logger.info(f"> [Done] Generate + save ({gs:.3f}s){note}")
            logger.info(f"\n{'=' * 50}")
            logger.info("⏱️  PERFORMANCE SUMMARY")
            logger.info(f"{'=' * 50}")
            logger.info(f"  Preprocess:       {timings['preprocess']:>8.3f}s")
            logger.info(f"  Generate + save:  {gs:>8.3f}s")
            logger.info(f"  {'─' * 40}")
            logger.info(f"  Total Time:       {timings['total']:>8.3f}s")
            if frames:
                logger.info(f"  Frames:           {frames:>8d}")
                logger.info(f"  FPS (gen+save):   {frames / gs:>8.2f}")
            logger.info(f"{'=' * 50}")
            logger.info(f"> Result saved at {res_video_path}")

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
        """Legacy sequential inference path (kept for backwards compatibility)."""
        timings: dict[str, float] = {}
        total_start = time.time()

        preprocess_start = time.time()
        data = self.data_processor.preprocess(ref_path, audio_path, no_crop=no_crop)
        timings["preprocess"] = time.time() - preprocess_start

        if verbose:
            logger.info(f"> [Done] Preprocess ({timings['preprocess']:.3f}s)")

        if benchmark:
            for _ in range(3):
                _ = self.G_opt.inference(
                    data=data,
                    a_cfg_scale=a_cfg_scale,
                    r_cfg_scale=r_cfg_scale,
                    e_cfg_scale=e_cfg_scale,
                    emo=emo,
                    nfe=nfe,
                    seed=seed,
                )

        use_amp = getattr(self, "use_amp", False)
        use_amp_fp16 = getattr(self, "use_amp_fp16", False)

        inference_start = time.time()
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

        timings["inference"] = time.time() - inference_start

        save_start = time.time()
        res_video_path = self.save_video(d_hat.float(), res_video_path, audio_path)
        timings["save"] = time.time() - save_start
        timings["total"] = time.time() - total_start

        if verbose:
            logger.info(f"> [Done] Save video ({timings['save']:.3f}s)")
            logger.info(f"> Result saved at {res_video_path}")

        return res_video_path


__all__ = [
    "OptimizedInferenceAgent",
    "InferenceOptions",
    "DataProcessor",
    "SimpleVideoEncoder",
    "encode_video_nvenc",
]


if __name__ == "__main__":
    opt = InferenceOptions().parse()
    opt.rank = 0

    print(f"\n{'=' * 60}")
    print("FLOAT Optimized Inference")
    print(f"Optimization: {opt.optimization_mode}")
    print(f"FP16: {opt.use_fp16}")
    print(f"{'=' * 60}\n")

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
            f"{call_time}-{video_name}-{audio_name}-nfe{opt.nfe}-seed{opt.seed}-acfg{opt.a_cfg_scale}-ecfg{opt.e_cfg_scale}-{opt.emo}-{opt_suffix}.mp4",
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
