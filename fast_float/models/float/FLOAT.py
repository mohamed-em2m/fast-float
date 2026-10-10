"""
stream_pipeline.py - auto-tuned, multi-stream, multi-GPU streaming decode + ffmpeg pipeline for FLOAT.

                         +--> GPU0: N0 decode streams --+
  sampler (GPU0, ODE) -> |--> GPU1: N1 decode streams --|-> reorder -> ffmpeg (NVENC/x264)
  re-batched latents     +--> GPU2: N2 decode streams --+     (pinned host buffers)

* The ODE sampler is sequential (chunk t needs chunk t-1), so it stays on the primary GPU.
* Decoding is per-frame and independent, so micro-batches are spread over all GPUs. Every decode
  worker pulls from ONE shared queue, so faster / less-busy GPUs automatically take more work.
* Latents are tiny, so they are handed to other GPUs through host memory (no P2P required).
  Each GPU copies its finished uint8 frames straight into pinned host buffers.
* Each GPU is auto-tuned separately (batch size + stream count from its own free VRAM).

Options read from self.opt (all optional):
    ngpus              number of GPUs for decoding (default 1; clamped to what is visible)
    decode_on_primary  False -> keep GPU0 for the sampler only when ngpus > 1 (default True)
    decode_streams     force streams PER GPU (0/absent -> auto)
    decode_max_streams cap for auto mode (default 3)
    decode_max_batch   largest micro-batch the tuner may try (default 32)
    decode_reserve_gib VRAM kept free per GPU (default: auto)
    use_nvenc          False -> libx264 (default True if ffmpeg has h264_nvenc)

Usage in your model file:
    from stream_pipeline import StreamingVideoSaver
    class FLOAT(BaseModel, StreamingVideoSaver): ...
"""

import os
import math
import time
import uuid
import logging
import struct
import wave
import subprocess
import shutil
import contextlib
import copy
import queue
import threading
import traceback
from dataclasses import dataclass, field
from typing import Callable, Dict, Generator, Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from torchdiffeq import odeint
except ImportError:
    odeint = None

try:
    import torchvision
except Exception:
    import sys
    sys.modules["torchvision"] = None
    sys.modules["torchvision.io"] = None

from transformers import Wav2Vec2Config
from transformers.modeling_outputs import BaseModelOutput

from fast_float.models.wav2vec2 import Wav2VecModel
from fast_float.models.wav2vec2_ser import Wav2Vec2ForSpeechClassification
from fast_float.models import BaseModel
from fast_float.models.float.generator import Generator
from fast_float.models.float.FMT import FlowMatchingTransformer

torch.backends.cudnn.benchmark = True

logger = logging.getLogger(__name__)
GiB = 1024**3
_ABORT = object()


# --------------------------------------------------------------------------- #
# Auto-tuner (one instance per GPU)
# --------------------------------------------------------------------------- #
@dataclass
class DecodeProfile:
    batch_size: int
    transient_bytes: int  # peak extra GPU bytes of ONE batch at batch_size
    height: int
    width: int
    channels: int
    transient_by_batch: Dict[int, int] = field(default_factory=dict)
    fps_by_batch: Dict[int, float] = field(default_factory=dict)


class DecodeAutoTuner:
    def __init__(
        self,
        device,
        primary=True,
        max_batch=16,
        max_streams=1,
        reserve_gib=None,
        plateau=0.93,
        safety=1.3,
        repeats=1,
        verbose=True,
        opt=None,
    ):
        self.device = device
        self.primary = (
            primary  # primary GPU also hosts sampler + NVENC -> bigger reserve
        )
        self.max_batch = max_batch
        self.max_streams = max_streams
        self.reserve_gib = reserve_gib
        self.plateau = plateau  # "good enough" fraction of best throughput
        self.safety = safety  # headroom for allocator fragmentation
        self.repeats = repeats
        self.verbose = verbose
        self.opt = opt

    def _budget(self) -> int:
        """Bytes we may spend on decode streams right now (call with this device current)."""
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info(self.device)
        if self.reserve_gib is not None:
            reserve = int(self.reserve_gib * GiB)
        elif self.primary:
            reserve = max(
                int(0.8 * GiB), int(0.08 * total)
            )  # sampler + NVENC + context
        else:
            reserve = max(int(0.35 * GiB), int(0.03 * total))  # decode only
        return int(free) - reserve

    def _measure(self, fn: Callable[[int], torch.Tensor], b: int):
        out = fn(b)  # warm-up (cudnn.benchmark picks algos)
        torch.cuda.synchronize(self.device)
        del out
        torch.cuda.empty_cache()
        base = torch.cuda.memory_allocated(self.device)
        best_dt, peak, shape = float("inf"), 0, None
        for _ in range(self.repeats):
            torch.cuda.reset_peak_memory_stats(self.device)
            t0 = time.perf_counter()
            out = fn(b)
            torch.cuda.synchronize(self.device)
            best_dt = min(best_dt, time.perf_counter() - t0)
            peak = max(peak, torch.cuda.max_memory_allocated(self.device))
            shape = tuple(out.shape[1:])  # (H, W, C)
            del out
        return peak - base, b / best_dt, shape

    @torch.no_grad()
    def tune(self, decode_fn: Callable[[int], torch.Tensor]) -> DecodeProfile:
        """decode_fn(b) -> uint8 [b, H, W, C] tensor on self.device (no OOM fallback inside)."""
        with torch.cuda.device(self.device):
            results, shape, b = {}, None, 1
            while b <= self.max_batch:
                if results:
                    prev = max(results)
                    projected = results[prev][0] * (b / prev) * self.safety
                    if projected > self._budget():
                        break
                try:
                    transient, fps, shape = self._measure(decode_fn, b)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    break
                results[b] = (transient, fps)
                b *= 2
            if not results:
                raise RuntimeError(
                    f"{self.device}: decoding even 1 frame does not fit in free VRAM"
                )

            budget = self._budget()
            feasible = {
                k: v for k, v in results.items() if v[0] * self.safety <= budget
            }
            if not feasible:
                k = min(results)
                feasible = {k: results[k]}

            opt_bs = getattr(self.opt, "decode_batch_size", None) if self.opt else None
            if opt_bs is not None and int(opt_bs) > 0:
                req_bs = int(opt_bs)
                if req_bs in results:
                    chosen = req_bs
                else:
                    chosen = min(req_bs, max(feasible.keys()))
            else:
                best = max(f for _, f in feasible.values())
                # Choose the smallest batch size that achieves plateau throughput
                # to minimize VRAM usage, lower latency, and prevent cuDNN grouped-conv alignment issues.
                chosen = min(
                    k for k, (_, f) in feasible.items() if f >= self.plateau * best
                )

            h, w, c = shape
            prof = DecodeProfile(
                chosen,
                results[chosen][0],
                h,
                w,
                c,
                {k: v[0] for k, v in results.items()},
                {k: round(v[1], 1) for k, v in results.items()},
            )
            if self.verbose:
                print(
                    f"[TUNER {self.device}] frames/s by batch {prof.fps_by_batch} -> batch={chosen}, "
                    f"{prof.transient_bytes / GiB:.2f} GiB per in-flight batch"
                )
            return prof

    def num_streams(self, prof: DecodeProfile, batch: int = None) -> int:
        t = (
            prof.transient_by_batch.get(batch, prof.transient_bytes)
            if batch
            else prof.transient_bytes
        )
        per_stream = max(1, int(t * self.safety))
        with torch.cuda.device(self.device):
            n = self._budget() // per_stream
        return int(max(1, min(self.max_streams, n)))


# --------------------------------------------------------------------------- #
# Streaming pipeline mixin
# --------------------------------------------------------------------------- #
def _qget(q: queue.Queue, stop: threading.Event):
    while not stop.is_set():
        try:
            return q.get(timeout=0.1)
        except queue.Empty:
            pass
    return _ABORT


class StreamingVideoSaver:
    """
    Mixin. Expects on self: opt, sample(), encode_image_into_latent(),
    encode_identity_into_motion(), motion_autoencoder (with .dec).
    """

    # ---------------- devices / replicas ---------------- #
    def _cuda_device(self) -> torch.device:
        r = getattr(self.opt, "rank", 0)
        if not torch.cuda.is_available():
            return torch.device("cpu")
        dev = torch.device(f"cuda:{r}") if isinstance(r, int) else torch.device(r)
        if dev.index is None:
            dev = torch.device("cuda", torch.cuda.current_device())
        return dev

    def _decode_devices(self):
        primary = self._cuda_device()
        visible = torch.cuda.device_count()
        want = max(1, int(getattr(self.opt, "ngpus", 1)))
        n = min(want, visible)
        p_idx = primary.index if primary.index is not None else 0
        if want > visible:
            logger.warning(
                f"[MULTI-GPU] ngpus={want} requested but only {visible} visible; using {n}"
            )
        else:
            logger.info(
                f"[MULTI-GPU] using {n}/{visible} GPU(s) (primary=cuda:{p_idx})"
            )
        others = [
            torch.device("cuda", i) for i in range(visible) if i != primary.index
        ][: max(0, n - 1)]
        devs = [primary] + others
        if n > 1 and not getattr(self.opt, "decode_on_primary", True):
            devs = others
        return primary, devs

    def _get_ae(self, dev, primary):
        """Motion autoencoder on `dev`: the original on the primary GPU, cached replicas elsewhere.
        If you reload weights into self.motion_autoencoder, call self._ae_replicas.clear().
        """
        if dev == primary or (dev.index is not None and primary.index is not None and dev.index == primary.index):
            return self.motion_autoencoder
        reps = getattr(self, "_ae_replicas", None)
        if reps is None:
            reps = self._ae_replicas = {}
        idx = dev.index if dev.index is not None else str(dev)
        if idx not in reps:
            ae = copy.deepcopy(self.motion_autoencoder).to(dev).eval()
            ae.requires_grad_(False)
            reps[idx] = ae
        return reps[idx]

    # ---------------- decode ---------------- #
    def _decode_u8(self, ae, feats_in, lat, allow_split=True) -> torch.Tensor:
        """lat [b, dim] -> uint8 [b, H, W, C] on lat's device/current stream. Halves batch on OOM."""
        b = lat.shape[0]
        # expand is a zero-copy view; harmless if the decoder already broadcasts
        feats = [
            f.expand(b, *f.shape[1:]) if (f.shape[0] == 1 and b > 1) else f
            for f in feats_in
        ]
        try:
            out = ae.dec(
                lat, alpha=None, feats=feats
            )  # same call as decode_latent_into_image
            if isinstance(out, (tuple, list)):
                out = out[0]
            out = out.float()
            out = out.add_(1.0).mul_(127.5).clamp_(0.0, 255.0).to(torch.uint8)
            return out.permute(0, 2, 3, 1).contiguous()
        except torch.cuda.OutOfMemoryError:
            if not allow_split or b <= 1:
                raise
        # (outside the except block so the failed tensors are released first)
        torch.cuda.empty_cache()
        h = b // 2
        return torch.cat(
            [
                self._decode_u8(ae, feats_in, lat[:h]),
                self._decode_u8(ae, feats_in, lat[h:]),
            ],
            0,
        )

    def _get_decode_profile(self, dev, primary, ae, feats, dtype):
        if getattr(self, "_decode_profiles", None) is None:
            self._decode_profiles, self._decode_tuners = {}, {}
        if dev.index not in self._decode_profiles:
            tuner = DecodeAutoTuner(
                dev,
                primary=(dev.index == primary.index),
                max_batch=int(getattr(self.opt, "decode_max_batch", 16)),
                max_streams=int(getattr(self.opt, "decode_max_streams", 1)),
                reserve_gib=getattr(self.opt, "decode_reserve_gib", None),
                opt=self.opt,
            )
            dim = self.opt.dim_w

            def fn(b):
                lat = torch.randn(b, dim, device=dev, dtype=dtype)
                return self._decode_u8(ae, feats, lat, allow_split=False)

            self._decode_profiles[dev.index] = tuner.tune(fn)
            self._decode_tuners[dev.index] = tuner
        return self._decode_tuners[dev.index], self._decode_profiles[dev.index]

    # ---------------- ffmpeg ---------------- #
    def _encoder_args(self):
        if getattr(self, "_has_nvenc", None) is None:
            try:
                txt = subprocess.run(
                    ["ffmpeg", "-hide_banner", "-encoders"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                ).stdout
                self._has_nvenc = "h264_nvenc" in txt
            except Exception:
                self._has_nvenc = False
        if self._has_nvenc and getattr(self.opt, "use_nvenc", True):
            return [
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
                "-forced-idr",
                "1",
                "-pix_fmt",
                "yuv420p",
            ]
        return [
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
        ]

    def _start_ffmpeg(self, video_path, audio_path, H, W, C, fps):
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgba" if C == 4 else "rgb24",
            "-s",
            f"{W}x{H}",
            "-r",
            str(fps),
            "-i",
            "-",
        ]
        if audio_path:
            cmd += ["-i", audio_path]
        cmd += self._encoder_args()
        if audio_path:
            cmd += ["-c:a", "aac", "-shortest"]
        cmd += ["-movflags", "+faststart", video_path]
        return subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    # ---------------- main entry ---------------- #
    @torch.no_grad()
    def inference_and_save(
        self,
        data: dict,
        video_path: str,
        audio_path: str = None,
        a_cfg_scale=None,
        r_cfg_scale=None,
        e_cfg_scale=None,
        emo=None,
        nfe=10,
        seed=None,
    ) -> str:
        t_start = time.perf_counter()
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg not found in PATH")

        primary, devs = self._decode_devices()
        torch.cuda.set_device(primary)

        s = data["s"]
        s_r, r_s_lambda, s_r_feats = self.encode_image_into_latent(s.to(primary))
        if "s_r" in data:
            r_s = self.encode_identity_into_motion(data["s_r"])
        else:
            r_s = self.motion_autoencoder.dec.direction(r_s_lambda)
        data["r_s"] = r_s

        if a_cfg_scale is None:
            a_cfg_scale = self.opt.a_cfg_scale
        if r_cfg_scale is None:
            r_cfg_scale = self.opt.r_cfg_scale
        if e_cfg_scale is None:
            e_cfg_scale = self.opt.e_cfg_scale
        fps = int(getattr(self.opt, "fps", 25))

        # autocast state is thread-local: capture it here, re-enter it in every pipeline thread
        try:
            amp_on = torch.is_autocast_enabled("cuda")
        except TypeError:  # older torch: no device argument
            amp_on = torch.is_autocast_enabled()
        try:
            amp_dtype = torch.get_autocast_dtype("cuda")
        except AttributeError:
            amp_dtype = torch.get_autocast_gpu_dtype()

        def with_amp(fn):
            def run(*a, **k):
                ctx = (
                    torch.autocast("cuda", dtype=amp_dtype)
                    if amp_on
                    else contextlib.nullcontext()
                )
                with ctx:
                    return fn(*a, **k)

            return run

        # ---- per-GPU context: model replica, reference features, tuned profile ----
        ctx = {}
        for dev in devs:
            ae = self._get_ae(dev, primary)
            if dev.index == primary.index:
                feats = s_r_feats
            else:
                feats = [f.to(dev) for f in s_r_feats]
                if torch.cuda.is_available():
                    torch.cuda.synchronize(dev)
                    torch.cuda.synchronize(primary)
            tuner, prof = self._get_decode_profile(dev, primary, ae, feats, s_r.dtype)
            ctx[dev.index] = dict(dev=dev, ae=ae, feats=feats, tuner=tuner, prof=prof)

        # Select micro-batch size: user override takes precedence, otherwise use maximum tuned size
        # to maximize GPU utilization and tensor core saturation.
        override_bs = getattr(self.opt, "decode_batch_size", None)
        if override_bs is not None and int(override_bs) > 0:
            B = int(override_bs)
        else:
            B = max(c["prof"].batch_size for c in ctx.values())
        override = int(getattr(self.opt, "decode_streams", 0))
        for c in ctx.values():
            c["n_streams"] = override or c["tuner"].num_streams(c["prof"], B)
        any_prof = next(iter(ctx.values()))["prof"]
        H, W, C = any_prof.height, any_prof.width, any_prof.channels
        n_workers = sum(c["n_streams"] for c in ctx.values())
        need_host_latents = any(i != primary.index for i in ctx)

        logger.info(
            f"[PLAN] batch={B} frame={W}x{H}x{C} @ {fps}fps, "
            f"{len(ctx)} GPU(s), {n_workers} decode stream(s)"
        )
        for i, c in ctx.items():
            logger.info(
                f"       cuda:{i} {torch.cuda.get_device_name(i)}: {c['n_streams']} stream(s)"
            )

        # ---- shared state ----
        stop = threading.Event()
        errors = []
        state = {"total": None, "frames": 0}
        work_q: queue.Queue = queue.Queue()  # latents only: tiny, unbounded is fine
        done_q: queue.Queue = queue.Queue()
        pool: queue.Queue = queue.Queue()  # pinned host buffers = backpressure
        for _ in range(max(16, n_workers * 4 + 4)):
            pool.put(torch.empty((B, H, W, C), dtype=torch.uint8, pin_memory=True))

        def fail(where, e):
            errors.append((where, e, traceback.format_exc()))
            stop.set()

        # ---- producer: ODE sampling on primary GPU, immediately dispatched to fixed micro-batches ----
        def producer():
            try:
                torch.cuda.set_device(primary)
                primary_stream = torch.cuda.current_stream(primary)

                def emit_one(lat_slice, n_valid):
                    """Record a per-item event and (if needed) copy latents to host non-blocking.

                    Each work item gets its own Event so workers on secondary GPUs
                    can call ev.synchronize() before touching lat_host without
                    incorrectly racing against the *next* producer step.
                    """
                    if need_host_latents and torch.cuda.is_available():
                        lat_host = torch.empty(lat_slice.shape, dtype=lat_slice.dtype, pin_memory=True)
                        lat_host.copy_(lat_slice, non_blocking=True)
                    else:
                        lat_host = None
                    ev = torch.cuda.Event()
                    ev.record(primary_stream)
                    return ev, lat_host

                seq = 0
                for chunk in self.sample(
                    data,
                    a_cfg_scale=a_cfg_scale,
                    r_cfg_scale=r_cfg_scale,
                    e_cfg_scale=e_cfg_scale,
                    emo=emo,
                    nfe=nfe,
                    seed=seed,
                    s_r=s_r,
                ):
                    if stop.is_set():
                        return
                    lat = chunk.reshape(-1, chunk.shape[-1])
                    n_total = lat.shape[0]
                    # Immediately emit ALL frames in this chunk so decoder never starves.
                    for i in range(0, n_total, B):
                        sl = lat[i : i + B]
                        n_valid = sl.shape[0]
                        if n_valid < B:
                            # Pad remainder to fixed batch size B to maintain uniform pinned host buffer shape
                            sl = torch.cat([sl, sl[-1:].expand(B - n_valid, -1)], 0)
                        ev, lat_host = emit_one(sl, n_valid)
                        work_q.put((seq, sl, lat_host, n_valid, ev))
                        seq += 1

                state["total"] = seq
            except BaseException as e:  # noqa: BLE001
                fail("producer", e)
            finally:
                for _ in range(n_workers):
                    work_q.put(None)

        # ---- decode workers: one CUDA stream each, spread over all GPUs ----
        def worker(c, stream):
            dev = c["dev"]
            is_primary = dev.index == primary.index
            try:
                torch.cuda.set_device(dev)
                with torch.no_grad():  # grad mode is thread-local
                    while True:
                        item = _qget(work_q, stop)
                        if item is _ABORT or item is None:
                            return
                        seq, lat_gpu, lat_host, n_valid, ev = item

                        buf = _qget(pool, stop)
                        if buf is _ABORT:
                            return

                        with torch.cuda.stream(stream):
                            if is_primary:
                                # Wait for the sampler stream to finish writing
                                # this micro-batch before we decode it.
                                stream.wait_event(ev)
                                lat_gpu.record_stream(stream)
                                lat = lat_gpu
                            else:
                                # For secondary GPUs the latents live in pinned
                                # host memory. We must ensure the producer has
                                # finished the CPU copy (recorded in ev) before
                                # we upload them. ev.synchronize() is a host-side
                                # wait so it does NOT block the CUDA stream.
                                ev.synchronize()
                                lat = lat_host.to(dev, non_blocking=True)
                                lat.record_stream(stream)
                            u8 = self._decode_u8(c["ae"], c["feats"], lat)
                            buf[:n_valid].copy_(u8[:n_valid], non_blocking=True)
                            done = torch.cuda.Event()
                            done.record(stream)

                        # Do NOT call done.synchronize() here to avoid stalling the worker
                        # thread and starving the GPU stream. The writer synchronizes before reading buf.
                        done_q.put((seq, buf, n_valid, done))
            except BaseException as e:  # noqa: BLE001
                fail(f"decode_worker[{dev}]", e)

        # ---- writer: restore order, feed ffmpeg ----
        def writer():
            proc, err_chunks, drain = None, [], None
            try:
                proc = self._start_ffmpeg(video_path, audio_path, H, W, C, fps)
                drain = threading.Thread(
                    target=lambda: err_chunks.append(proc.stderr.read()), daemon=True
                )
                drain.start()
                pending, nxt = {}, 0
                while not stop.is_set():
                    total = state["total"]
                    if total is not None and nxt >= total:
                        break
                    try:
                        seq, buf, n, done = done_q.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    pending[seq] = (buf, n, done)
                    while nxt in pending:
                        buf, n, done = pending.pop(nxt)
                        if done is not None:
                            done.synchronize()
                        proc.stdin.write(buf[:n].numpy().reshape(-1))
                        state["frames"] += n
                        pool.put(buf)
                        nxt += 1
                if stop.is_set():
                    return
                proc.stdin.close()
                rc = proc.wait(timeout=300)
                drain.join(timeout=5)
                if rc != 0:
                    raise RuntimeError(
                        f"ffmpeg exited with {rc}: "
                        f"{b''.join(err_chunks).decode(errors='ignore')[:1000]}"
                    )
            except BrokenPipeError:
                try:
                    proc.wait(timeout=5)
                    drain.join(timeout=5)
                except Exception:
                    pass
                msg = b"".join(err_chunks).decode(errors="ignore")[:1000]
                fail("writer", RuntimeError(f"ffmpeg pipe closed: {msg}"))
            except BaseException as e:  # noqa: BLE001
                fail("writer", e)
            finally:
                if proc is not None and proc.poll() is None:
                    proc.kill()

        threads = [threading.Thread(target=with_amp(producer), name="sampler")]
        for i, c in ctx.items():
            for k in range(c["n_streams"]):
                threads.append(
                    threading.Thread(
                        target=with_amp(worker),
                        args=(c, torch.cuda.Stream(device=c["dev"])),
                        name=f"decode-cuda{i}-{k}",
                    )
                )
        threads.append(threading.Thread(target=writer, name="writer"))
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        if errors:
            where, exc, tb = errors[0]
            raise RuntimeError(f"[{where}] {exc}\n{tb}") from exc

        dt = time.perf_counter() - t_start
        self.last_stats = {"frames": state["frames"], "seconds": dt}
        logger.info(
            f"[DONE] {state['frames']} frames in {dt:.2f}s "
            f"({state['frames'] / dt:.1f} fps end-to-end) -> {video_path}"
        )
        return video_path


class FLOAT(BaseModel, StreamingVideoSaver):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        logger.info(opt)
        rank = getattr(self.opt, "rank", 0)
        self.rank = rank
        device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available():
            self.stream_decode_2 = torch.cuda.Stream(device=device)
            self.stream_decode = torch.cuda.Stream(device=device)
            self.stream_encode = torch.cuda.Stream(device=device)
        else:
            self.stream_decode_2 = None
            self.stream_decode = None
            self.stream_encode = None

        self.num_frames_for_clip = int(self.opt.wav2vec_sec * self.opt.fps)
        self.num_prev_frames = int(self.opt.num_prev_frames)

        # motion latent auto-encoder
        self.motion_autoencoder = Generator(
            size=opt.input_size, style_dim=opt.dim_w, motion_dim=opt.dim_m
        )
        self.motion_autoencoder.requires_grad_(False)

        # condition encoders
        self.audio_encoder = AudioEncoder(opt)
        self.emotion_encoder = Audio2Emotion(opt)
        # FMT; Flow Matching Transformer
        self.fmt = FlowMatchingTransformer(opt)

        # ODE options
        self.odeint_kwargs = {
            "atol": self.opt.ode_atol,
            "rtol": self.opt.ode_rtol,
            "method": self.opt.torchdiffeq_ode_method,
        }

    ######## Motion Encoder - Decoder ########
    @torch.no_grad()
    def encode_image_into_latent(self, x: torch.Tensor) -> list:
        if x.ndim == 3:
            x = x.unsqueeze(0)
        x_r, _, x_r_feats = self.motion_autoencoder.enc(x, input_target=None)
        x_r_lambda = self.motion_autoencoder.enc.fc(x_r)
        return x_r, x_r_lambda, x_r_feats

    @torch.no_grad()
    def encode_identity_into_motion(self, x_r: torch.Tensor) -> torch.Tensor:
        if x_r.ndim == 1:
            x_r = x_r.unsqueeze(0)
        x_r_lambda = self.motion_autoencoder.enc.fc(x_r)
        r_x = self.motion_autoencoder.dec.direction(x_r_lambda)
        return r_x

    def clear_decode_cache(self):
        """Clears memoized batch size and tuner profiles so GPU VRAM is re-evaluated."""
        if hasattr(self, "_decode_bs_cache") and self._decode_bs_cache:
            self._decode_bs_cache.clear()
        if hasattr(self, "_decode_profiles") and self._decode_profiles:
            self._decode_profiles.clear()
        if hasattr(self, "_decode_tuners") and self._decode_tuners:
            self._decode_tuners.clear()

    def get_optimal_decode_batch_size(self, device=None) -> int:
        """Automatically select optimal frame decode batch size based on GPU memory.
        Maximized for high GPU utilization and throughput.

        The result is memoized per device index in ``self._decode_bs_cache`` so that
        ``torch.cuda.mem_get_info`` is only called once per device, not once per chunk.

        Tiers (total VRAM / free VRAM):
            * >= 6 GiB free OR (>= 14 GiB total & >= 3.8 GiB free): batch 16
            * >= 3 GiB free OR (>= 10 GiB total & >= 1.8 GiB free): batch 8
            * >= 1.2 GiB free OR >= 6 GiB total                   : batch 4
            * >= 0.7 GiB free                                     : batch 2
            * otherwise / CPU                                     : batch 1
        """
        if hasattr(self, "opt") and getattr(self.opt, "decode_batch_size", None) is not None:
            val = int(self.opt.decode_batch_size)
            if val > 0:
                return val

        if not torch.cuda.is_available():
            return 4

        try:
            if device is None:
                rank = getattr(self.opt, "rank", 0) if hasattr(self, "opt") else 0
                device = getattr(self, "device", None) or torch.device(f"cuda:{rank}")
            if isinstance(device, str):
                device = torch.device(device)
            if device.type != "cuda":
                return 4

            idx = device.index if device.index is not None else torch.cuda.current_device()

            # Return memoized result to avoid repeated cudaMemGetInfo calls
            cache = getattr(self, "_decode_bs_cache", None)
            if cache is None:
                self._decode_bs_cache = {}
                cache = self._decode_bs_cache
            if idx in cache:
                return cache[idx]

            free_b, total_b = torch.cuda.mem_get_info(idx)
            gib = 1024 ** 3
            free_gib = free_b / gib
            total_gib = total_b / gib

            # StyleDecoder decoding takes ~20-40 MB per frame (e.g. ~300-500 MB for batch 16).
            # When ~5 GiB free out of 15 GiB is available, batch 16 is easily accommodated!
            if free_gib >= 6.0 or (total_gib >= 14.0 and free_gib >= 3.8):
                bs = 16
            elif free_gib >= 3.0 or (total_gib >= 10.0 and free_gib >= 1.8):
                bs = 8
            elif free_gib >= 1.2 or total_gib >= 6.0:
                bs = 4
            elif free_gib >= 0.7:
                bs = 2
            else:
                bs = 1

            cache[idx] = bs
            return bs
        except Exception:
            return 4

    @torch.no_grad()
    def _decode_latent_batched(
        self,
        ae,
        r_d: torch.Tensor,
        s_r_feats: list,
        batch_size: int = None,
        device=None,
    ) -> torch.Tensor:
        """Decode motion latents into images in mini-batches (e.g., 4 frames gathered together)
        to prevent OOM and maximize GPU throughput. Supports 2D [N, D] and 3D [B, T, D] latents.
        """
        if device is None:
            device = r_d.device if (hasattr(r_d, "device") and r_d.device.type == "cuda") else None

        bs = batch_size if batch_size is not None else self.get_optimal_decode_batch_size(device)
        bs = max(1, int(bs))

        if r_d.ndim == 3:
            b_dim, t_dim, d_dim = r_d.shape
            n_total = b_dim * t_dim
            r_d_flat = r_d.reshape(n_total, d_dim)

            aligned_feats = []
            for f in s_r_feats:
                if f.shape[0] == 1:
                    aligned_feats.append(("broadcast", f))
                elif f.shape[0] == b_dim:
                    full_f = f.unsqueeze(1).expand(-1, t_dim, *f.shape[1:]).reshape(n_total, *f.shape[1:])
                    aligned_feats.append(("slice", full_f))
                elif f.shape[0] == n_total:
                    aligned_feats.append(("slice", f))
                else:
                    aligned_feats.append(("broadcast", f[:1]))

            if n_total <= bs:
                chunk_feats = [
                    f.expand(n_total, *f.shape[1:]) if mode == "broadcast" else f
                    for mode, f in aligned_feats
                ]
                out = ae.dec(r_d_flat, alpha=None, feats=chunk_feats)
                if isinstance(out, (tuple, list)):
                    out = out[0]
                return out.view(b_dim, t_dim, *out.shape[1:])
            else:
                out_chunks = []
                for i in range(0, n_total, bs):
                    chunk_r_d = r_d_flat[i : i + bs]
                    m = chunk_r_d.shape[0]
                    chunk_feats = [
                        f.expand(m, *f.shape[1:]) if mode == "broadcast" else f[i : i + m]
                        for mode, f in aligned_feats
                    ]
                    chunk_out = ae.dec(chunk_r_d, alpha=None, feats=chunk_feats)
                    if isinstance(chunk_out, (tuple, list)):
                        chunk_out = chunk_out[0]
                    out_chunks.append(chunk_out)
                out = torch.cat(out_chunks, dim=0)
                return out.view(b_dim, t_dim, *out.shape[1:])
        else:
            is_1d = (r_d.ndim == 1)
            if is_1d:
                r_d = r_d.unsqueeze(0)
            n_total = r_d.shape[0]

            aligned_feats = []
            for f in s_r_feats:
                if f.shape[0] == 1:
                    aligned_feats.append(("broadcast", f))
                elif f.shape[0] == n_total:
                    aligned_feats.append(("slice", f))
                elif n_total % f.shape[0] == 0:
                    t = n_total // f.shape[0]
                    full_f = f.unsqueeze(1).expand(-1, t, *f.shape[1:]).reshape(n_total, *f.shape[1:])
                    aligned_feats.append(("slice", full_f))
                else:
                    aligned_feats.append(("broadcast", f[:1]))

            if n_total <= bs:
                chunk_feats = [
                    f.expand(n_total, *f.shape[1:]) if mode == "broadcast" else f
                    for mode, f in aligned_feats
                ]
                out = ae.dec(r_d, alpha=None, feats=chunk_feats)
                if isinstance(out, (tuple, list)):
                    out = out[0]
                return out.squeeze(0) if is_1d else out
            else:
                out_chunks = []
                for i in range(0, n_total, bs):
                    chunk_r_d = r_d[i : i + bs]
                    m = chunk_r_d.shape[0]
                    chunk_feats = [
                        f.expand(m, *f.shape[1:]) if mode == "broadcast" else f[i : i + m]
                        for mode, f in aligned_feats
                    ]
                    chunk_out = ae.dec(chunk_r_d, alpha=None, feats=chunk_feats)
                    if isinstance(chunk_out, (tuple, list)):
                        chunk_out = chunk_out[0]
                    out_chunks.append(chunk_out)
                out = torch.cat(out_chunks, dim=0)
                return out.squeeze(0) if is_1d else out

    @torch.no_grad()
    def decode_latent_into_image(
        self,
        s_r: torch.Tensor,
        s_r_feats: list,
        r_d: torch.Tensor,
        batch_size: int = None,
    ) -> torch.Tensor:
        """Decode latent into image with broadcasted features (supports 2D [N, D] and 3D [B, T, D])
        Processes frames in mini-batches (e.g. 4 frames at a time, auto-selected based on GPU VRAM)
        to prevent OOM and maximize decoding throughput.
        """
        return self._decode_latent_batched(
            self.motion_autoencoder,
            r_d,
            s_r_feats,
            batch_size=batch_size,
        )

    @torch.no_grad()
    def sample(
        self,
        data: dict,
        a_cfg_scale: float = 1.0,
        r_cfg_scale: float = 1.0,
        e_cfg_scale: float = 1.0,
        emo: str = None,
        nfe: int = 10,
        seed: int = None,
        s_r=None,
    ) -> torch.Tensor:
        rank = getattr(self.opt, "rank", 0)
        device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
        cur_stream = torch.cuda.current_stream(device) if torch.cuda.is_available() else None
        encode_stream = self.stream_encode if (self.stream_encode is not None and torch.cuda.is_available()) else None

        encode_ctx = (
            torch.cuda.stream(encode_stream)
            if encode_stream is not None
            else contextlib.nullcontext()
        )
        with encode_ctx:
            r_s, a = data["r_s"], data["a"]
            if a.ndim == 1:
                a = a.unsqueeze(0)
            B_a = a.shape[0]
            B_r = r_s.shape[0] if (r_s is not None and r_s.ndim > 1) else 1
            B_s = s_r.shape[0] if (s_r is not None and s_r.ndim > 1) else 1
            B = max(B_a, B_r, B_s)

            if a.shape[0] != B and a.shape[0] == 1:
                a = a.expand(B, -1)
            if r_s is not None:
                if r_s.ndim == 1:
                    r_s = r_s.unsqueeze(0)
                if r_s.shape[0] != B and r_s.shape[0] == 1:
                    r_s = r_s.expand(B, -1)
            if s_r is not None:
                if s_r.ndim == 1:
                    s_r = s_r.unsqueeze(0)
                if s_r.shape[0] != B and s_r.shape[0] == 1:
                    s_r = s_r.expand(B, -1)

            # make time
            steps = nfe if nfe is not None else getattr(self.opt, "nfe", 10)
            time = torch.linspace(0, 1, steps, device=device)

            # encoding audio first with whole audio
            a = a.to(device)
            T = math.ceil(a.shape[-1] * self.opt.fps / self.opt.sampling_rate)
            wa = self.audio_encoder.inference(a, seq_len=T)

            # encoding emotion first
            if isinstance(emo, torch.Tensor):
                we = emo.to(device=device, dtype=torch.float32)
                if we.ndim == 1:
                    we = we.unsqueeze(0).unsqueeze(1)
                elif we.ndim == 2:
                    we = we.unsqueeze(1)
                if we.shape[0] != B and we.shape[0] == 1:
                    we = we.expand(B, -1, -1)
            elif isinstance(emo, (list, tuple)):
                emo_tensors = []
                for e in emo:
                    idx = self.emotion_encoder.label2id.get(str(e).lower(), None)
                    if idx is not None:
                        one_hot = F.one_hot(
                            torch.tensor(idx, device=device),
                            num_classes=self.opt.dim_e,
                        ).float()
                    else:
                        one_hot = torch.zeros(self.opt.dim_e, device=device)
                    emo_tensors.append(one_hot)
                we = torch.stack(emo_tensors, dim=0).unsqueeze(1)
                if we.shape[0] != B and we.shape[0] == 1:
                    we = we.expand(B, -1, -1)
            elif emo is not None and str(emo).lower() in self.emotion_encoder.label2id:
                emo_idx = self.emotion_encoder.label2id[str(emo).lower()]
                we = (
                    F.one_hot(
                        torch.tensor(emo_idx, device=device),
                        num_classes=self.opt.dim_e,
                    )
                    .float()
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .expand(B, -1, -1)
                )
            else:
                we = self.emotion_encoder.predict_emotion(a).unsqueeze(1)
                if we.shape[0] != B and we.shape[0] == 1:
                    we = we.expand(B, -1, -1)

        # Synchronize sampling stream so ODE starts with audio & emotion features ready.
        # This keeps encoding isolated to encode_stream and lets ODE sampling run on cur_stream
        # without interfering with or stalling concurrent decode streams.
        if cur_stream is not None and encode_stream is not None and cur_stream != encode_stream:
            cur_stream.wait_stream(encode_stream)

        decoded_length = 0

        # ------------------------------------------------------------------
        # Pre-build CFG constant tensors that are invariant across ALL ODE
        # steps and ALL chunks (ref / emotion never change during inference).
        # Avoids 2 × torch.cat allocations per ODE step for free.
        # ------------------------------------------------------------------
        _cfg_active = (
            a_cfg_scale != 1.0 or r_cfg_scale != 1.0 or e_cfg_scale != 1.0
        )
        _ref_cat = None
        _emotion_cat = None
        if _cfg_active and r_s is not None:
            # Mirror the shape normalisation that forward_with_cfv would do.
            wr_ready = r_s
            if wr_ready.ndim == 1:
                wr_ready = wr_ready.unsqueeze(0).unsqueeze(1)
            elif wr_ready.ndim == 2:
                wr_ready = wr_ready.unsqueeze(1)
            if wr_ready.shape[0] != B and wr_ready.shape[0] == 1:
                wr_ready = wr_ready.expand(B, -1, -1)
            _ref_cat = torch.cat([wr_ready, wr_ready, wr_ready], dim=0)

        if _cfg_active:
            # we is already shaped [B, 1, dim_e] at this point.
            we_ready = we
            if we_ready.ndim == 1:
                we_ready = we_ready.unsqueeze(0).unsqueeze(1)
            elif we_ready.ndim == 2:
                we_ready = we_ready.unsqueeze(1)
            if we_ready.shape[0] != B and we_ready.shape[0] == 1:
                we_ready = we_ready.expand(B, -1, -1)
            _null_we = torch.zeros_like(we_ready)
            _emotion_cat = torch.cat([_null_we, we_ready, _null_we], dim=0)

        # sampling chunk by chunk
        for t in range(0, int(math.ceil(T / self.num_frames_for_clip))):
            if getattr(self.opt, "fix_noise_seed", False):
                seed = getattr(self.opt, "seed", 25) if seed is None else seed
                g = torch.Generator(device=device)
                g.manual_seed(seed)
                x0 = torch.randn(
                    B,
                    self.num_frames_for_clip,
                    self.opt.dim_w,
                    device=device,
                    generator=g,
                )
            else:
                x0 = torch.randn(
                    B,
                    self.num_frames_for_clip,
                    self.opt.dim_w,
                    device=device,
                )

            if t == 0:  # should define the previous
                prev_x_t = torch.zeros(
                    B, self.num_prev_frames, self.opt.dim_w, device=device
                )
                prev_wa_t = torch.zeros(
                    B, self.num_prev_frames, self.opt.dim_w, device=device
                )
            else:
                prev_x_t = sample_t[:, -self.num_prev_frames :]
                prev_wa_t = wa_t[:, -self.num_prev_frames :]

            wa_t = wa[
                :, t * self.num_frames_for_clip : (t + 1) * self.num_frames_for_clip
            ]
            if wa_t.shape[1] < self.num_frames_for_clip:  # padding by replicate
                wa_t = F.pad(
                    wa_t,
                    (0, 0, 0, self.num_frames_for_clip - wa_t.shape[1]),
                    mode="replicate",
                )

            # ------------------------------------------------------------------
            # Pre-build per-chunk CFG tensors (invariant within one ODE solve).
            # audio_cat / prev_x_cat / prev_wa_cat change between chunks but
            # are constant across the NFE ODE steps inside a single chunk.
            # ------------------------------------------------------------------
            _audio_cat = None
            _prev_x_cat = None
            _prev_wa_cat = None
            if _cfg_active:
                wa_t_exp = wa_t
                if wa_t_exp.shape[0] != B and wa_t_exp.shape[0] == 1:
                    wa_t_exp = wa_t_exp.expand(B, -1, -1)
                _null_wa = torch.zeros_like(wa_t_exp)
                _audio_cat = torch.cat([_null_wa, wa_t_exp, wa_t_exp], dim=0)

                _prev_x_cat = torch.cat(
                    [prev_x_t, prev_x_t, prev_x_t], dim=0
                )
                _prev_wa_cat = torch.cat(
                    [prev_wa_t, prev_wa_t, prev_wa_t], dim=0
                )

            def sample_chunk(tt, zt,
                             _ac=_audio_cat, _rc=_ref_cat, _ec=_emotion_cat,
                             _pxc=_prev_x_cat, _pwc=_prev_wa_cat):
                out = self.fmt.forward_with_cfv(
                    t=tt.unsqueeze(0),
                    x=zt,
                    wa=wa_t,
                    wr=r_s,
                    we=we,
                    prev_x=prev_x_t,
                    prev_wa=prev_wa_t,
                    a_cfg_scale=a_cfg_scale,
                    r_cfg_scale=r_cfg_scale,
                    e_cfg_scale=e_cfg_scale,
                    _audio_cat=_ac,
                    _ref_cat=_rc,
                    _emotion_cat=_ec,
                    _prev_x_cat=_pxc,
                    _prev_wa_cat=_pwc,
                )
                out_current = out[:, self.num_prev_frames :]
                return out_current

            # solve ODE
            if odeint is None:
                raise ImportError("torchdiffeq is required for ODE sampling. Install with: pip install torchdiffeq")
            trajectory_t = odeint(sample_chunk, x0, time, **self.odeint_kwargs)
            sample_t = trajectory_t[-1]

            # Trim to exact length T on the last chunk
            frames_remaining = T - decoded_length
            if frames_remaining < sample_t.shape[1]:
                sample_t_trim = sample_t[:, :frames_remaining]
            else:
                sample_t_trim = sample_t
            samples_len = sample_t_trim.shape[1]
            decoded_length += samples_len

            # Add reference and yield chunk
            s_r_b = (
                s_r.unsqueeze(1)
                if (s_r is not None and s_r.ndim == 2)
                else s_r
            )
            s_r_d_t = (
                (s_r_b + sample_t_trim)
                if s_r_b is not None
                else sample_t_trim
            )
            if B == 1:
                yield s_r_d_t.squeeze(0)  # [T, dim_w]
            else:
                yield s_r_d_t  # [B, T, dim_w]

            # Stop if we've reached the target length
            if decoded_length >= T:
                break

    @torch.no_grad()
    def inference_stream(
        self,
        data: dict,
        a_cfg_scale=None,
        r_cfg_scale=None,
        e_cfg_scale=None,
        emo=None,
        nfe=10,
        seed=None,
    ):
        """Generator that streams sampling latents sequentially and decodes them concurrently
        on dedicated CUDA stream(s), yielding each decoded video chunk [T_chunk, 3, H, W]
        (or [B, T_chunk, 3, H, W]) as soon as it finishes decoding.
        """
        rank = getattr(self.opt, "rank", 0)
        device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
        primary = torch.device(device)

        s, a = data["s"], data["a"]
        if s.ndim == 3:
            s = s.unsqueeze(0)
        if a.ndim == 1:
            a = a.unsqueeze(0)

        B_s = s.shape[0]
        B_a = a.shape[0]
        B = max(B_s, B_a)
        if s.shape[0] != B and s.shape[0] == 1:
            s = s.expand(B, *s.shape[1:])
        if a.shape[0] != B and a.shape[0] == 1:
            a = a.expand(B, *a.shape[1:])
        data["s"] = s
        data["a"] = a

        s_r, r_s_lambda, s_r_feats = self.encode_image_into_latent(s.to(device))

        if "s_r" in data:
            r_s = self.encode_identity_into_motion(data["s_r"])
        else:
            r_s = self.motion_autoencoder.dec.direction(r_s_lambda)
        data["r_s"] = r_s

        if a_cfg_scale is None:
            a_cfg_scale = getattr(self.opt, "a_cfg_scale", 1.0)
        if r_cfg_scale is None:
            r_cfg_scale = getattr(self.opt, "r_cfg_scale", 1.0)
        if e_cfg_scale is None:
            e_cfg_scale = getattr(self.opt, "e_cfg_scale", 1.0)

        # Multi-GPU / multi-stream decode support
        if torch.cuda.is_available():
            _, devs = self._decode_devices()
        else:
            devs = [primary]

        gpu_ctx = []
        for dev in devs:
            ae = self._get_ae(dev, primary)
            is_prim = (
                dev == primary
                or (
                    dev.index is not None
                    and primary.index is not None
                    and dev.index == primary.index
                )
            )
            feats = (
                s_r_feats
                if is_prim
                else [f.to(dev) for f in s_r_feats]
            )
            if is_prim:
                prim_streams = [
                    st for st in (self.stream_decode, self.stream_decode_2)
                    if st is not None
                ]
                if not prim_streams and torch.cuda.is_available():
                    prim_streams = [torch.cuda.Stream(device=dev)]
                for st in prim_streams:
                    gpu_ctx.append((dev, ae, feats, st))
            else:
                st = (
                    torch.cuda.Stream(device=dev)
                    if torch.cuda.is_available()
                    else None
                )
                gpu_ctx.append((dev, ae, feats, st))

        num_devs = len(gpu_ctx)

        # Synchronize decode streams with initial reference features
        if torch.cuda.is_available():
            init_ev = torch.cuda.Event()
            init_ev.record(torch.cuda.current_stream(primary))
            for _, _, _, st in gpu_ctx:
                if st is not None and st != torch.cuda.current_stream(primary):
                    st.wait_event(init_ev)

        # Autocast state is thread-local: capture to re-enter inside producer thread
        try:
            amp_on = torch.is_autocast_enabled("cuda")
        except TypeError:
            amp_on = torch.is_autocast_enabled()
        try:
            amp_dtype = torch.get_autocast_dtype("cuda")
        except AttributeError:
            amp_dtype = torch.get_autocast_gpu_dtype()

        def with_amp(fn):
            def run(*args, **kwargs):
                ctx = (
                    torch.autocast("cuda", dtype=amp_dtype)
                    if amp_on
                    else contextlib.nullcontext()
                )
                with ctx:
                    return fn(*args, **kwargs)

            return run

        # Producer-consumer queue for streaming latents immediately to decode streams
        work_q = queue.Queue(maxsize=max(2, num_devs * 2))
        stop_event = threading.Event()
        prod_error = []

        def producer():
            try:
                if torch.cuda.is_available():
                    torch.cuda.set_device(primary)
                    prod_stream = torch.cuda.current_stream(primary)
                else:
                    prod_stream = None

                seq = 0
                for frame_latent in self.sample(
                    data,
                    a_cfg_scale=a_cfg_scale,
                    r_cfg_scale=r_cfg_scale,
                    e_cfg_scale=e_cfg_scale,
                    emo=emo,
                    nfe=nfe,
                    seed=seed,
                    s_r=s_r,
                ):
                    if stop_event.is_set():
                        break

                    ev = None
                    if prod_stream is not None:
                        ev = torch.cuda.Event()
                        ev.record(prod_stream)

                    while not stop_event.is_set():
                        try:
                            work_q.put((seq, frame_latent, ev), timeout=0.1)
                            break
                        except queue.Full:
                            continue
                    seq += 1

                while not stop_event.is_set():
                    try:
                        work_q.put((None, None, None), timeout=0.1)
                        break
                    except queue.Full:
                        continue
            except BaseException as e:
                prod_error.append((e, traceback.format_exc()))
                while not stop_event.is_set():
                    try:
                        work_q.put(("ERROR", None, None), timeout=0.1)
                        break
                    except queue.Full:
                        continue

        prod_thread = threading.Thread(
            target=with_amp(producer), name="sampler_producer", daemon=True
        )
        prod_thread.start()

        dev_idx = 0
        cur = (
            torch.cuda.current_stream(device=primary)
            if torch.cuda.is_available()
            else None
        )

        try:
            while True:
                try:
                    seq, frame_latent, ev = work_q.get(timeout=0.1)
                except queue.Empty:
                    if not prod_thread.is_alive() and work_q.empty():
                        break
                    continue

                if seq == "ERROR":
                    if prod_error:
                        err, tb = prod_error[0]
                        raise RuntimeError(f"Error in sampling producer thread:\n{tb}") from err
                    raise RuntimeError("Error in sampling producer thread")

                if seq is None or frame_latent is None:
                    break

                dev, ae, feats, stream = gpu_ctx[dev_idx % num_devs]
                dev_idx += 1

                ctx_mgr = (
                    torch.cuda.stream(stream)
                    if (stream is not None and torch.cuda.is_available())
                    else contextlib.nullcontext()
                )
                with ctx_mgr:
                    if stream is not None and ev is not None:
                        stream.wait_event(ev)

                    lat = (
                        frame_latent.to(dev, non_blocking=True)
                        if dev != primary
                        else frame_latent
                    )
                    if stream is not None and torch.cuda.is_available():
                        lat.record_stream(stream)

                    out = self._decode_latent_batched(
                        ae,
                        lat,
                        feats,
                        batch_size=getattr(self.opt, "decode_batch_size", None),
                        device=dev,
                    )
                    if not out.is_cuda and torch.cuda.is_available():
                        out = out.to(device=primary, non_blocking=True)
                    elif out.device != primary:
                        out = out.to(device=primary, non_blocking=True)

                    done_ev = None
                    if stream is not None and torch.cuda.is_available():
                        done_ev = torch.cuda.Event()
                        done_ev.record(stream)

                if cur is not None:
                    if done_ev is not None:
                        cur.wait_event(done_ev)
                    elif stream is not None:
                        cur.wait_stream(stream)

                yield out
        finally:
            stop_event.set()
            prod_thread.join(timeout=2.0)

    @torch.no_grad()
    def inference(
        self,
        data: dict,
        a_cfg_scale=None,
        r_cfg_scale=None,
        e_cfg_scale=None,
        emo=None,
        nfe=10,
        seed=None,
    ) -> dict:
        try:
            s, a = data["s"], data["a"]
            B_s = s.shape[0] if s.ndim >= 4 else 1
            B_a = a.shape[0] if a.ndim >= 2 else 1
            is_batched = max(B_s, B_a) > 1

            stacked_list = []
            for chunk in self.inference_stream(
                data=data,
                a_cfg_scale=a_cfg_scale,
                r_cfg_scale=r_cfg_scale,
                e_cfg_scale=e_cfg_scale,
                emo=emo,
                nfe=nfe,
                seed=seed,
            ):
                stacked_list.append(chunk)

            if is_batched:
                d_hat = torch.cat(stacked_list, dim=1)  # [B, total_T, 3, H, W]
            else:
                d_hat = torch.cat(stacked_list, dim=0)  # [total_T, 3, H, W]
            return {"d_hat": d_hat}

        except Exception as e:
            logger.error(f"Error in inference: {e}\n{traceback.format_exc()}")
            raise


@dataclass
class FrameData:
    """Container for frame data passed between producer and consumer"""

    frame_np: np.ndarray
    frame_idx: int
    is_last: bool = False


# Usage example:
"""
# The run_inference method now uses producer-consumer pattern internally:
result_path = model.run_inference(
    res_video_path='output.mp4',
    ref_path='reference.jpg',
    audio_path='audio.wav',
    a_cfg_scale=2.0,
    r_cfg_scale=1.0,
    e_cfg_scale=1.0,
    emo='happy',
    nfe=10,
    verbose=True
)
"""


################ Condition Encoders ################
class AudioEncoder(BaseModel):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.only_last_features = opt.only_last_features

        self.num_frames_for_clip = int(opt.wav2vec_sec * self.opt.fps)
        self.num_prev_frames = int(opt.num_prev_frames)
        if os.path.exists(opt.wav2vec_model_path):

            self.wav2vec2 = Wav2VecModel.from_pretrained(
                opt.wav2vec_model_path, local_files_only=True
            )
        else:
            self.wav2vec2 = Wav2VecModel.from_pretrained(opt.wav2vec_model_path)

        self.wav2vec2.feature_extractor._freeze_parameters()

        for name, param in self.wav2vec2.named_parameters():
            param.requires_grad = False

        audio_input_dim = 768 if opt.only_last_features else 12 * 768

        self.audio_projection = nn.Sequential(
            nn.Linear(audio_input_dim, opt.dim_w), nn.LayerNorm(opt.dim_w), nn.SiLU()
        )

    def get_wav2vec2_feature(self, a: torch.Tensor, seq_len: int) -> torch.Tensor:
        if a.ndim == 1:
            a = a.unsqueeze(0)
        a = self.wav2vec2(
            a, seq_len=seq_len, output_hidden_states=not self.only_last_features
        )
        if self.only_last_features:
            a = a.last_hidden_state
        else:
            a = torch.stack(a.hidden_states[1:], dim=1).permute(0, 2, 1, 3)
            a = a.reshape(a.shape[0], a.shape[1], -1)
        return a

    def forward(self, a: torch.Tensor, prev_a: torch.Tensor = None) -> torch.Tensor:
        if a.ndim == 1:
            a = a.unsqueeze(0)
        if prev_a is not None:
            if prev_a.ndim == 1:
                prev_a = prev_a.unsqueeze(0)
            a = torch.cat([prev_a, a], dim=1)
            target_len = int(
                (self.num_frames_for_clip + self.num_prev_frames)
                * self.opt.sampling_rate
                / self.opt.fps
            )
            if a.shape[1] % target_len != 0:
                pad_amt = target_len - (a.shape[1] % target_len)
                a = F.pad(a, (0, pad_amt), mode="replicate")
            a = self.get_wav2vec2_feature(
                a, seq_len=self.num_frames_for_clip + self.num_prev_frames
            )
        else:
            target_len = int(
                self.num_frames_for_clip * self.opt.sampling_rate / self.opt.fps
            )
            if a.shape[1] % target_len != 0:
                pad_amt = target_len - (a.shape[1] % target_len)
                a = F.pad(a, (0, pad_amt), mode="replicate")
            a = self.get_wav2vec2_feature(a, seq_len=self.num_frames_for_clip)

        return self.audio_projection(a)  # frame by frame

    @torch.no_grad()
    def inference(self, a: torch.Tensor, seq_len: int) -> torch.Tensor:
        if a.ndim == 1:
            a = a.unsqueeze(0)
        expected_len = int(seq_len * self.opt.sampling_rate / self.opt.fps)
        if a.shape[1] < expected_len:
            a = F.pad(
                a,
                (0, expected_len - a.shape[1]),
                mode="replicate",
            )
        elif a.shape[1] > expected_len:
            a = a[:, :expected_len]
        a = self.get_wav2vec2_feature(a, seq_len=seq_len)
        return self.audio_projection(a)


class Audio2Emotion(nn.Module):
    def __init__(self, opt):
        super().__init__()

        # Load pretrained model (local preferred)
        if os.path.exists(opt.audio2emotion_path):
            self.wav2vec2_for_emotion = Wav2Vec2ForSpeechClassification.from_pretrained(
                opt.audio2emotion_path, local_files_only=True
            )
        else:
            self.wav2vec2_for_emotion = Wav2Vec2ForSpeechClassification.from_pretrained(
                opt.audio2emotion_path
            )

        self.wav2vec2_for_emotion.eval()

        # seven labels
        self.id2label = {
            0: "angry",
            1: "disgust",
            2: "fear",
            3: "happy",
            4: "neutral",
            5: "sad",
            6: "surprise",
        }

        self.label2id = {v: k for k, v in self.id2label.items()}

    @torch.no_grad()
    def predict_emotion(
        self, a: torch.Tensor, prev_a: torch.Tensor = None
    ) -> torch.Tensor:
        if a.ndim == 1:
            a = a.unsqueeze(0)
        # Concatenate previous audio if provided
        if prev_a is not None:
            if prev_a.ndim == 1:
                prev_a = prev_a.unsqueeze(0)
            a = torch.cat([prev_a, a], dim=1)

        logits = self.wav2vec2_for_emotion(a).logits
        return F.softmax(logits, dim=1)
