"""Optimized inference agent for FLOAT with multi-backend encoding and GPU optimizations."""

from __future__ import annotations

import contextlib
import datetime
import logging
import math
import os
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any

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
from fast_float.core.nvenc_encoder import (
    encode_numpy_nvenc,
    encode_video_nvenc,
    stage_frames_for_nvenc,
)
from fast_float.core.options import InferenceOptions
from fast_float.core.simple_encoder import SimpleVideoEncoder
from fast_float.models.float.sampler import stack_latent_parts
from fast_float.utils.timing import timed, timing_scope

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

        # Route each component (direction/refenc/decoder) to its configured
        # backend before the global optimisation pass. All-eager config = no-op.
        try:
            from fast_float.onnx.dispatch import apply_backends
            from fast_float.runtimes.inspect import warn_degraded

            apply_backends(self.G, self.opt)
            warn_degraded(self.G, logger)
        except Exception as e:
            logger.warning(f"! Per-component backend dispatch skipped: {e}")

        # Apply optimization
        self.apply_optimization()

        # Load Data Processor
        self.data_processor = DataProcessor(opt)

        logger.info(f"✓ Model loaded with {self.optimization_mode} optimization")

    def load_model(self) -> None:
        self.G = FLOAT(self.opt)

    def load_weight(self, checkpoint_path: str, rank: int) -> None:
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        dev = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            for model_name, model_param in self.G.named_parameters():
                if model_name in state_dict:
                    model_param.copy_(state_dict[model_name].to(dev))
                elif "wav2vec2" in model_name:
                    pass
                else:
                    logger.warning(f"! Warning: {model_name} not found in state_dict.")
            for buf_name, buf in self.G.named_buffers():
                if buf_name in state_dict:
                    buf.copy_(state_dict[buf_name].to(buf.device))
                elif "wav2vec2" in buf_name:
                    pass
        del state_dict

    def apply_optimization(self) -> None:
        """Apply various GPU optimization strategies."""
        if self.optimization_mode == "torch_compile":
            try:
                if hasattr(torch, "compile"):
                    logger.info("Applying torch.compile optimization...")
                    self.G_opt = torch.compile(self.G, mode="max-autotune", fullgraph=False)
                    logger.info("✓ torch.compile applied successfully")
                else:
                    logger.warning("! torch.compile not available (requires PyTorch 2.0+)")
                    self.G_opt = self.G
            except Exception as e:
                logger.warning(f"! torch.compile failed: {e}")
                self.G_opt = self.G

        elif self.optimization_mode == "channels_last":
            logger.info("Applying channels_last memory format...")
            try:
                # NOTE: Module.to() converts params/buffers in place and returns
                # self; do not rebind the loop variable. Inputs still arrive as
                # contiguous NCHW, so each conv pays a layout conversion per op
                # unless the pipeline is fed channels-last end to end.
                for module in self.G.modules():
                    if isinstance(module, (torch.nn.Conv2d, torch.nn.Conv3d)):
                        module.to(memory_format=torch.channels_last)
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
                # Same caveat as above: in-place conversion, but inputs stay
                # contiguous NCHW, so mixed layouts convert per op.
                for module in self.G.modules():
                    if isinstance(module, (torch.nn.Conv2d, torch.nn.Conv3d)):
                        module.to(memory_format=torch.channels_last)
                logger.info("✓ Channels last applied to conv layers")
            except Exception as e:
                logger.warning(f"! Channels last skipped: {e}")

            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            logger.info("✓ cuDNN benchmark enabled")

            if hasattr(torch, "compile"):
                try:
                    self.G_opt = torch.compile(self.G, mode="max-autotune")
                    logger.info("✓ All optimizations applied")
                except Exception as e:
                    logger.warning(f"! torch.compile failed: {e}")
                    self.G_opt = self.G
                    logger.info("✓ Partial optimizations applied")
            else:
                self.G_opt = self.G
                logger.info("✓ Partial optimizations applied")
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
        video_encoder: str | None = None,
    ) -> str:
        """Save video supporting FFmpeg NVENC, CPU PyAV, and Torchcodec alternatives."""
        try:
            start_time = time.time()
            encoder_choice = str(video_encoder or getattr(self.opt, "video_encoder", "FFmpeg NVENC")).strip().lower()
            fps = int(getattr(self.opt, "fps", 25))

            if encoder_choice in ("cpu pyav", "pyav", "pyav_cpu"):
                with timed("save"):
                    return save_video_pyav(vid, video_path, audio_path=audio_path, fps=fps)

            if encoder_choice in ("torchcodec", "torch codec", "tc"):
                with timed("save"):
                    return save_video_torchcodec(vid, video_path, audio_path=audio_path, fps=fps)

            # Default: FFmpeg NVENC
            with timed("save"):
                encode_video_nvenc(vid, video_path, fps=fps)

            total_time = time.time() - start_time
            logger.info(f"✅ Total save: {total_time:.3f}s -> {video_path}")
            return video_path

        except Exception as e:
            logger.error(f"Error: {e}\n{traceback.format_exc()}")
            raise

    @staticmethod
    def _uses_nvenc(video_encoder: str | None) -> bool:
        """True when the encoder is the direct FFmpeg/NVENC pipe (CPU-side work)."""
        choice = str(video_encoder or "").strip().lower()
        return choice not in ("cpu pyav", "pyav", "pyav_cpu", "torchcodec", "torch codec", "tc")

    @staticmethod
    def _encode_nvenc_chunk(staged, video_path: str, fps: int) -> str:
        """Worker-thread body: wait for the async copy, then run ffmpeg."""
        with timed("save"):
            return encode_numpy_nvenc(staged.wait(), video_path, fps=fps)

    @torch.no_grad()
    @timing_scope("run_inference")
    def run_inference(
        self,
        res_video_path: str | None,
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
        video_encoder: str | None = None,
        dispatcher=None,
    ) -> str | None:
        timings: dict[str, float] = {}
        total_start = time.perf_counter()

        # ---- preprocess ----
        t0 = time.perf_counter()
        with timed("preprocess"):
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

        def generate(out_path: str | None, tap=None) -> str | None:
            """Sampling + decoding + video encoding + audio mux, fully overlapped."""
            with make_amp():
                return self.G_opt.inference_and_save(
                    data=data,
                    video_path=out_path,
                    audio_path=audio_path,
                    video_encoder=video_encoder,
                    dispatcher=tap,
                    **gen_kwargs,
                )

        # ---- benchmark ----
        if benchmark:
            warmup_runs, benchmark_runs = 3, 10
            fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
            os.close(fd)
            try:
                logger.info("Warming up (first run also runs the one-time GPU auto-tune)...")
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

            logger.info(f"Average generate+encode time: {elapsed / benchmark_runs * 1000:.2f} ms/run")
            logger.info(f"Throughput: {frames / elapsed:.2f} frames/s")

        # ---- actual run ----
        first_run = getattr(self.G_opt, "_decode_profiles", None) is None
        t0 = time.perf_counter()
        with timed("generate_and_save"):
            res_video_path = generate(res_video_path, tap=dispatcher)
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
            logger.info(f"> Result saved at {res_video_path}" if res_video_path else "> Streamed (no output file)")

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
            with torch.cuda.amp.autocast(dtype=torch.float16 if use_amp_fp16 else torch.float32):
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

    @torch.no_grad()
    @timing_scope("run_inference_batched")
    def run_inference_batched(
        self,
        items: list,
        ref_path: str,
        a_cfg_scale: float = 2.0,
        r_cfg_scale: float = 1.0,
        e_cfg_scale: float = 1.0,
        emo: str = "S2E",
        nfe: int = 10,
        no_crop: bool = False,
        seed: int = 25,
        seeds: list[int] | None = None,
        verbose: bool = True,
        video_encoder: str | None = None,
        dispatcher=None,
    ) -> list:
        """Sample several padded audio chunks in ONE model batch, then decode each.

        ``items`` is a list of ``{"audio_path", "padded_path", "duration",
        "out_path"}`` where every ``padded_path`` WAV has the same length.
        Returns the output video paths in order.

        Each chunk keeps an independent temporal chain (all start from zero
        prev-frames, exactly like the sequential path), so batching only shares
        the ODE solve across items. Per-item latents are trimmed to their real
        (unpadded) length before decoding. Emotion follows the tail rule: tails
        longer than 30% of the window are silence-padded into the batched read,
        shorter tails reuse the previous chunk's emotion.
        """
        if dispatcher is not None:
            logger.warning("[dispatch] batched chunk generation does not support a frame dispatcher yet; ignoring it")
        G = self.G_opt
        rank = getattr(self.opt, "rank", 0)
        device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
        fps = int(getattr(self.opt, "fps", 25))

        # data["a"] is [N, L] (preprocess zero-pads the list to equal length).
        with timed("preprocess"):
            data = self.data_processor.preprocess(ref_path, [it["padded_path"] for it in items], no_crop=no_crop)
        s = data["s"]

        s_r, r_s_lambda, s_r_feats = G.encode_image_into_latent(s.to(device))
        data["r_s"] = G.motion_autoencoder.dec.direction(r_s_lambda)

        emo_arg = emo
        if isinstance(emo, str) and emo == "S2E":
            # Tail rule: chunks longer than 30% of the batch window are padded
            # with silence up to full length and join the batched forward (they
            # carry enough real speech to dominate the pooled prediction);
            # chunks shorter than 30% reuse the previous chunk's emotion — too
            # little speech for a reliable read, and the feeling rarely flips
            # inside one window. Full windows are never padded, so their reads
            # stay exact.
            auds = [self.data_processor.default_aud_loader(it["audio_path"]) for it in items]
            full_len = max(a.shape[-1] for a in auds)
            pct30 = full_len * 0.30
            use_idx, copy_from, pad_list = [], {}, []
            for k, a0 in enumerate(auds):
                L = a0.shape[-1]
                if full_len <= L:
                    use_idx.append(k)
                    pad_list.append(a0)
                elif pct30 < L:
                    need = full_len - L
                    pad_list.append(torch.cat([a0, torch.zeros(need, dtype=a0.dtype)], dim=-1))
                    use_idx.append(k)
                else:
                    copy_from[k] = k - 1  # previous chunk in time order
            emo_mat = [None] * len(items)
            if pad_list:
                batch_a = torch.stack(pad_list, dim=0).to(device)
                with timed("emotion_encode", gpu=True):
                    logits = G.emotion_encoder.predict_emotion(batch_a)  # [m, dim_e]
                for k, m in zip(use_idx, range(len(use_idx)), strict=False):
                    emo_mat[k] = logits[m]
            for k, src in copy_from.items():
                if src >= 0 and emo_mat[src] is not None:
                    emo_mat[k] = emo_mat[src]
                else:
                    # Defensive: no previous chunk (only a tail can be short) —
                    # fall back to a solo read of the unpadded audio.
                    with timed("emotion_encode", gpu=True):
                        emo_mat[k] = G.emotion_encoder.predict_emotion(auds[k].unsqueeze(0).to(device))[0]
            emo_arg = torch.stack(emo_mat, dim=0).unsqueeze(1)  # [N, 1, dim_e]

        def make_amp() -> contextlib.AbstractContextManager[None]:
            if getattr(self, "use_amp_fp16", False):
                return torch.autocast("cuda", dtype=torch.float16)
            return contextlib.nullcontext()

        decode_bs = getattr(self.opt, "decode_batch_size", None)
        encoder_choice = video_encoder or getattr(self.opt, "video_encoder", "FFmpeg NVENC")
        parallel_save = self._uses_nvenc(encoder_choice)

        # This path never calls `video_pipeline.inference_and_save`, which is
        # where the per-request cuDNN policy is applied. Mirror it here: with
        # benchmark=True every fresh executor thread re-autotunes ~40 conv
        # shapes (~1.2 s) on its first decode.
        if os.getenv("FLOAT_CUDNN_BENCHMARK", "0") != "1":
            torch.backends.cudnn.benchmark = False

        out_paths: list = [None] * len(items)
        pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="vidsave") if parallel_save else None
        # Cast + D2H run on their own stream so the next chunk can decode while
        # the frames are still being copied out; the encode thread waits on the
        # per-chunk event instead of stalling this loop.
        save_stream = (
            torch.cuda.Stream(device=torch.device(device)) if pool is not None and torch.cuda.is_available() else None
        )
        # Sampling AND decoding stay under one autocast context, exactly like the
        # sequential path (run_inference wraps the whole generate() call). The
        # latents leave the ODE solve in half precision when AMP is on, so the
        # decoder must run under autocast too — otherwise half latents hit fp32
        # weights and torch raises a dtype mismatch.
        try:
            with make_amp():
                parts = list(
                    G.sample(
                        data,
                        a_cfg_scale=a_cfg_scale,
                        r_cfg_scale=r_cfg_scale,
                        e_cfg_scale=e_cfg_scale,
                        emo=emo_arg,
                        nfe=nfe,
                        seed=seed,
                        seeds=seeds,
                        s_r=s_r,
                    )
                )  # each piece is [N, T_slice, dim_w] (s_r already added), or [T_slice, dim_w] when N == 1
                lat_all = stack_latent_parts(parts, len(items))  # [N, T_full, dim_w]

                for i, it in enumerate(items):
                    # duration is seconds -> frames (samples*fps/sr == seconds*fps).
                    t_i = max(1, math.ceil(it["duration"] * fps))
                    # The merge used to trim the overlapping head by re-encoding;
                    # skip those frames before decoding instead. One frame is kept
                    # even for a pathologically short chunk.
                    trim = min(max(0, int(it.get("trim_frames", 0) or 0)), t_i - 1)
                    lat_i = lat_all[i, trim:t_i, :]
                    with timed("decode", gpu=True):
                        d_hat = G.decode_latent_into_image(s_r, s_r_feats, lat_i, batch_size=decode_bs)
                    if pool is not None:
                        # Hand the cast + copy to the save stream (returns
                        # immediately) and let the worker run ffmpeg/NVENC
                        # while the next chunk decodes.
                        out_paths[i] = pool.submit(
                            self._encode_nvenc_chunk,
                            stage_frames_for_nvenc(d_hat, save_stream),
                            it["out_path"],
                            fps,
                        )
                    else:
                        out_paths[i] = self.save_video(
                            d_hat.float(),
                            it["out_path"],
                            it["audio_path"],
                            video_encoder=video_encoder,
                        )
                    del d_hat, lat_i
        finally:
            if pool is not None:
                pool.shutdown(wait=True)

        if pool is not None:
            out_paths = [future.result() for future in out_paths]

        if verbose:
            logger.info(f"> [Done] Batched inference: {len(items)} chunk(s) in one sample batch")
        return out_paths


__all__ = [
    "DataProcessor",
    "InferenceOptions",
    "OptimizedInferenceAgent",
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
