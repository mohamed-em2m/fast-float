"""
stream_pipeline.py - auto-tuned, multi-stream streaming decode + ffmpeg pipeline for FLOAT.

Pipeline (all stages overlap):

  producer thread          decode workers (1 CUDA stream each)        writer thread
  ----------------         ------------------------------------      ----------------
  sample() ODE chunks  ->  decode micro-batch -> uint8 NHWC ->       reorder -> ffmpeg stdin
  re-batched to B          async copy to pinned host buffer          (NVENC / x264)

The tuner measures, once per process, on THIS GPU:
  * the decode micro-batch size B where throughput stops improving
  * the real peak GPU memory of one in-flight batch
and then, before every run, sizes the number of concurrent decode streams from the
memory that is actually free right now. OOM at runtime halves the batch and retries.

Usage in your model file:
    from stream_pipeline import StreamingVideoSaver
    class FLOAT(BaseModel, StreamingVideoSaver): ...
"""

import os
import torch, math
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
import torch.nn as nn
import torch.nn.functional as F
import asyncio
from torchdiffeq import odeint
from transformers import Wav2Vec2Config
from transformers.modeling_outputs import BaseModelOutput
import torch
import subprocess
import time
import shutil
import numpy as np
from typing import Generator
import traceback
import threading
import queue
from dataclasses import dataclass

import torch
import subprocess
import time
import shutil
import numpy as np
from typing import Generator
import traceback
import threading
import queue
from dataclasses import dataclass


@dataclass
class FrameData:
    """Container for frame data passed between producer and consumer"""

    frame_np: np.ndarray
    frame_idx: int
    is_last: bool = False


from fast_float.models.wav2vec2 import Wav2VecModel
from fast_float.models.wav2vec2_ser import Wav2Vec2ForSpeechClassification

from fast_float.models import BaseModel
from fast_float.models.float.generator import Generator
from fast_float.models.float.FMT import FlowMatchingTransformer
import traceback

torch.backends.cudnn.benchmark = True


import contextlib  # noqa: F401  (handy for the sample() patch: contextlib.nullcontext())
import queue
import shutil
import subprocess
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Callable, Dict

import torch

GiB = 1024**3
_ABORT = object()


# --------------------------------------------------------------------------- #
# Auto-tuner
# --------------------------------------------------------------------------- #
@dataclass
class DecodeProfile:
    batch_size: int
    transient_bytes: int  # peak extra GPU bytes for ONE in-flight batch
    height: int
    width: int
    channels: int
    fps_by_batch: Dict[int, float] = field(default_factory=dict)


class DecodeAutoTuner:
    def __init__(
        self,
        device,
        max_batch=32,
        max_streams=3,
        reserve_gib=None,
        plateau=0.93,
        safety=1.3,
        repeats=2,
        verbose=True,
    ):
        self.device = device
        self.max_batch = max_batch
        self.max_streams = max_streams
        self.reserve_gib = reserve_gib  # None -> max(1.5 GiB, 15% of VRAM)
        self.plateau = plateau  # "good enough" fraction of best throughput
        self.safety = safety  # headroom for allocator fragmentation
        self.repeats = repeats
        self.verbose = verbose

    # memory we are allowed to spend on decode streams right now
    def _budget(self) -> int:
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info(self.device)
        if self.reserve_gib is not None:
            reserve = int(self.reserve_gib * GiB)
        else:
            # headroom for the sampler (ODE + wav2vec), NVENC session, CUDA context growth
            reserve = max(int(1.5 * GiB), int(0.15 * total))
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
        """decode_fn(b) -> uint8 [b, H, W, C] CUDA tensor (no OOM fallback inside)."""
        results = {}
        shape = None
        b = 1
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
                "Decoding even a single frame does not fit in free GPU memory"
            )

        budget = self._budget()
        feasible = {k: v for k, v in results.items() if v[0] * self.safety <= budget}
        if not feasible:
            k = min(results)
            feasible = {k: results[k]}
        best = max(f for _, f in feasible.values())
        chosen = min(k for k, (_, f) in feasible.items() if f >= self.plateau * best)

        h, w, c = shape
        prof = DecodeProfile(
            chosen,
            results[chosen][0],
            h,
            w,
            c,
            {k: round(v[1], 1) for k, v in results.items()},
        )
        if self.verbose:
            print(
                f"[TUNER] frames/s by batch: {prof.fps_by_batch} -> batch={chosen}, "
                f"{prof.transient_bytes / GiB:.2f} GiB per in-flight batch"
            )
        return prof

    def num_streams(self, prof: DecodeProfile) -> int:
        per_stream = max(1, int(prof.transient_bytes * self.safety))
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
    encode_identity_into_motion(), motion_autoencoder, decode_latent_into_image().
    """

    # ---------------- helpers ---------------- #
    def _cuda_device(self) -> torch.device:
        r = self.opt.rank
        dev = torch.device(f"cuda:{r}") if isinstance(r, int) else torch.device(r)
        if dev.index is None:
            dev = torch.device("cuda", torch.cuda.current_device())
        return dev

    def _decode_u8(self, s_r, s_r_feats, lat, allow_split=True) -> torch.Tensor:
        """lat [b, dim] -> uint8 [b, H, W, C] on the current stream. Halves the batch on OOM."""
        b = lat.shape[0]
        # expand is a zero-copy view; harmless if the decoder already broadcasts
        feats = [
            f.expand(b, *f.shape[1:]) if (f.shape[0] == 1 and b > 1) else f
            for f in s_r_feats
        ]
        try:
            out = self.decode_latent_into_image(s_r=s_r, s_r_feats=feats, r_d=lat)
            if isinstance(out, (tuple, list)):
                out = out[0]
            return (
                (out.clamp(-1, 1) * 127.5 + 127.5)
                .to(torch.uint8)
                .permute(0, 2, 3, 1)
                .contiguous()
            )
        except torch.cuda.OutOfMemoryError:
            if not allow_split or b == 1:
                raise
        # (outside the except block so the failed tensors are released first)
        torch.cuda.empty_cache()
        h = b // 2
        return torch.cat(
            [
                self._decode_u8(s_r, s_r_feats, lat[:h]),
                self._decode_u8(s_r, s_r_feats, lat[h:]),
            ],
            0,
        )

    def _get_decode_profile(self, s_r, s_r_feats):
        cached = getattr(self, "_decode_profile", None)
        if cached is not None:
            return self._decode_tuner, cached
        dev = self._cuda_device()
        tuner = DecodeAutoTuner(
            dev,
            max_batch=int(getattr(self.opt, "decode_max_batch", 32)),
            max_streams=int(getattr(self.opt, "decode_max_streams", 3)),
            reserve_gib=getattr(self.opt, "decode_reserve_gib", None),
        )
        dim = self.opt.dim_w

        def fn(b):
            lat = torch.randn(b, dim, device=dev, dtype=s_r.dtype)
            return self._decode_u8(s_r, s_r_feats, lat, allow_split=False)

        prof = tuner.tune(fn)
        self._decode_tuner, self._decode_profile = tuner, prof
        return tuner, prof

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

        dev = self._cuda_device()
        torch.cuda.set_device(dev)

        s = data["s"]
        s_r, r_s_lambda, s_r_feats = self.encode_image_into_latent(s.to(self.opt.rank))
        if "s_r" in data:
            r_s = self.encode_identity_into_motion(s_r)
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

        # ---- plan: batch size (cached) + stream count (from memory free right now) ----
        tuner, prof = self._get_decode_profile(s_r, s_r_feats)
        B = prof.batch_size
        n_streams = int(getattr(self.opt, "decode_streams", 0)) or tuner.num_streams(
            prof
        )
        H, W, C = prof.height, prof.width, prof.channels
        print(f"[PLAN] batch={B} streams={n_streams} frame={W}x{H}x{C} @ {fps}fps")

        # ---- shared state ----
        stop = threading.Event()
        errors = []
        state = {"total": None, "frames": 0, "next_alloc": 0}
        cond = threading.Condition()
        work_q: queue.Queue = queue.Queue()  # latents only: tiny, unbounded is fine
        done_q: queue.Queue = queue.Queue()
        pool: queue.Queue = queue.Queue()  # pinned host buffers = backpressure
        for _ in range(n_streams * 2 + 2):
            pool.put(torch.empty((B, H, W, C), dtype=torch.uint8, pin_memory=True))
        streams = [torch.cuda.Stream(device=dev) for _ in range(n_streams)]

        def fail(where, e):
            errors.append((where, e, traceback.format_exc()))
            stop.set()

        # ---- producer: ODE sampling, re-batched into fixed-size micro-batches ----
        def producer():
            try:
                torch.cuda.set_device(dev)
                seq, carry = 0, None
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
                    if carry is not None:
                        lat = torch.cat([carry, lat], 0)
                        carry = None
                    n_full = (lat.shape[0] // B) * B
                    if n_full:
                        ev = torch.cuda.Event()
                        ev.record(torch.cuda.current_stream(dev))
                        for i in range(0, n_full, B):
                            work_q.put((seq, lat[i : i + B], B, ev))
                            seq += 1
                    if n_full < lat.shape[0]:
                        carry = lat[n_full:]
                if carry is not None and carry.shape[0]:
                    n = carry.shape[0]
                    padded = torch.cat(
                        [carry, carry[-1:].expand(B - n, -1)], 0
                    )  # fixed shape
                    ev = torch.cuda.Event()
                    ev.record(torch.cuda.current_stream(dev))
                    work_q.put((seq, padded, n, ev))
                    seq += 1
                state["total"] = seq
            except BaseException as e:  # noqa: BLE001
                fail("producer", e)
            finally:
                for _ in range(n_streams):
                    work_q.put(None)

        # ---- decode workers: one CUDA stream each ----
        def worker(stream):
            try:
                torch.cuda.set_device(dev)
                with torch.no_grad():  # grad mode is thread-local
                    while True:
                        item = _qget(work_q, stop)
                        if item is _ABORT or item is None:
                            return
                        seq, lat, n_valid, ev = item

                        # take buffers strictly in sequence order -> no pool deadlock
                        with cond:
                            while state["next_alloc"] != seq and not stop.is_set():
                                cond.wait(0.1)
                        buf = _qget(pool, stop)
                        if buf is _ABORT:
                            return
                        with cond:
                            state["next_alloc"] += 1
                            cond.notify_all()

                        with torch.cuda.stream(stream):
                            stream.wait_event(
                                ev
                            )  # latents were made on the sampler stream
                            lat.record_stream(stream)
                            u8 = self._decode_u8(s_r, s_r_feats, lat)
                            buf[:n_valid].copy_(u8[:n_valid], non_blocking=True)
                            done = torch.cuda.Event()
                            done.record(stream)
                        done.synchronize()  # blocks this thread only
                        done_q.put((seq, buf, n_valid))
            except BaseException as e:  # noqa: BLE001
                fail("decode_worker", e)

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
                        seq, buf, n = done_q.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    pending[seq] = (buf, n)
                    while nxt in pending:
                        buf, n = pending.pop(nxt)
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

        threads = [threading.Thread(target=producer, name="sampler")]
        threads += [
            threading.Thread(target=worker, args=(st,), name=f"decode-{i}")
            for i, st in enumerate(streams)
        ]
        threads += [threading.Thread(target=writer, name="writer")]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        if errors:
            where, exc, tb = errors[0]
            raise RuntimeError(f"[{where}] {exc}\n{tb}") from exc

        dt = time.perf_counter() - t_start
        print(
            f"[DONE] {state['frames']} frames in {dt:.2f}s "
            f"({state['frames'] / dt:.1f} fps end-to-end) -> {video_path}"
        )
        return video_path


class FLOAT(BaseModel, StreamingVideoSaver):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.stream_decode_2 = torch.cuda.Stream(
            device=self.opt.rank
        )  # device optional
        self.stream_decode = torch.cuda.Stream(device=self.opt.rank)  # device optional
        self.stream_encode = torch.cuda.Stream(device=self.opt.rank)  # device optional

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
        self.decode_semaphore = asyncio.Semaphore(10)
        # FMT; Flow Matchingssss Transformer
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
        x_r, _, x_r_feats = self.motion_autoencoder.enc(x, input_target=None)
        x_r_lambda = self.motion_autoencoder.enc.fc(x_r)
        return x_r, x_r_lambda, x_r_feats

    @torch.no_grad()
    def encode_identity_into_motion(self, x_r: torch.Tensor) -> torch.Tensor:
        x_r_lambda = self.motion_autoencoder.enc.fc(x_r)
        r_x = self.motion_autoencoder.dec.direction(x_r_lambda)
        return r_x

    @torch.no_grad()
    def decode_latent_into_image(
        self, s_r: torch.Tensor, s_r_feats: list, r_d: torch.Tensor
    ) -> torch.Tensor:
        """Decode a single frame (not a chunk with time dimension)"""
        # r_d is now [B, dim] - a single frame
        img, _ = self.motion_autoencoder.dec(r_d, alpha=None, feats=s_r_feats)
        return img  # Return single decoded frame

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
        with torch.cuda.stream(self.stream_encode):
            r_s, a = data["r_s"], data["a"]
            B = a.shape[0]
            # make time
            time = torch.linspace(0, 1, self.opt.nfe, device=self.opt.rank)

            # encoding audio first with whole audio
            a = a.to(self.opt.rank)
            T = math.ceil(a.shape[-1] * self.opt.fps / self.opt.sampling_rate)
            wa = self.audio_encoder.inference(a, seq_len=T)

            # encoding emotion first
            emo_idx = self.emotion_encoder.label2id.get(str(emo).lower(), None)
            if emo_idx is None:
                we = self.emotion_encoder.predict_emotion(a).unsqueeze(1)
            else:
                we = (
                    F.one_hot(
                        torch.tensor(emo_idx, device=a.device),
                        num_classes=self.opt.dim_e,
                    )
                    .unsqueeze(0)
                    .unsqueeze(0)
                )

            decoded_length = 0

            # sampling chunk by chunk
            for t in range(0, int(math.ceil(T / self.num_frames_for_clip))):
                if self.opt.fix_noise_seed:
                    seed = self.opt.seed if seed is None else seed
                    g = torch.Generator(self.opt.rank)
                    g.manual_seed(seed)
                    x0 = torch.randn(
                        B,
                        self.num_frames_for_clip,
                        self.opt.dim_w,
                        device=self.opt.rank,
                        generator=g,
                    )
                else:
                    x0 = torch.randn(
                        B,
                        self.num_frames_for_clip,
                        self.opt.dim_w,
                        device=self.opt.rank,
                    )

                if t == 0:  # should define the previous
                    prev_x_t = torch.zeros(B, self.num_prev_frames, self.opt.dim_w).to(
                        self.opt.rank
                    )
                    prev_wa_t = torch.zeros(B, self.num_prev_frames, self.opt.dim_w).to(
                        self.opt.rank
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

                def sample_chunk(tt, zt):
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
                    )
                    out_current = out[:, self.num_prev_frames :]
                    return out_current

                # solve ODE
                trajectory_t = odeint(sample_chunk, x0, time, **self.odeint_kwargs)
                sample_t = trajectory_t[-1]

                # Trim to exact length T on the last chunk
                frames_remaining = T - decoded_length
                if frames_remaining < sample_t.shape[1]:
                    sample_t_trim = sample_t[:, :frames_remaining]
                else:
                    sample_t_trim = sample_t
                # sample_t_trim = sample_t
                samples_len = sample_t_trim.shape[1]
                decoded_length += samples_len
                # Add reference and yield frame by frame
                s_r_d_t = s_r + sample_t_trim

                # Yield frame by frame for streaming
                # print("shape of sample_t before squeeze",s_r_d_t.shape)

                s_r_d_t_squeezed = s_r_d_t.squeeze(0)  # [T, C, H, W]
                T_ = s_r_d_t_squeezed.shape[0]

                # Divide into 3 parts
                part_size = T_ // 2
                remainder = T_ % 2

                # Split points
                split1 = part_size + (1 if remainder > 0 else 0)
                split2 = split1 + part_size + (1 if remainder > 1 else 0)

                # Yield each part
                yield s_r_d_t_squeezed

                # print("shape of sample_t after squeeze",s_r_d_t.squeeze(0).shape)

                # Stop if we've reached the target length
                if decoded_length >= T:
                    break

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
        s, a = data["s"], data["a"]
        s_r, r_s_lambda, s_r_feats = self.encode_image_into_latent(s.to(self.opt.rank))

        if "s_r" in data:
            r_s = self.encode_identity_into_motion(s_r)
        else:
            r_s = self.motion_autoencoder.dec.direction(r_s_lambda)
        data["r_s"] = r_s

        if a_cfg_scale is None:
            a_cfg_scale = self.opt.a_cfg_scale
        if r_cfg_scale is None:
            r_cfg_scale = self.opt.r_cfg_scale
        if e_cfg_scale is None:
            e_cfg_scale = self.opt.e_cfg_scale

        try:
            # list of (tensor_on_gpu, stream)
            decode_results = []

            # s_r_feats = [f.expand(50, -1, -1, -1) for f in s_r_feats]

            # If sample yields lazily, we will create per-frame streams and decode on them.
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
                # create and keep a reference to the stream
                # launch decode on that stream
                with torch.cuda.stream(self.stream_decode):

                    # IMPORTANT: decode_latent_into_image must perform CUDA ops only,
                    # and must NOT move tensors to CPU / PIL or call .item() etc.
                    try:
                        out = self.decode_latent_into_image(
                            s_r=s_r, s_r_feats=s_r_feats, r_d=frame_latent
                        )
                        # out should be a torch.Tensor located on GPU (e.g. [B, C, H, W])
                        # If decode returns (tensor, other) adapt accordingly.
                        if isinstance(out, tuple) or isinstance(out, list):
                            # assume primary tensor is first element
                            out_tensor = out[0]
                        else:
                            out_tensor = out

                        # ensure it's a CUDA tensor
                        if not out_tensor.is_cuda:
                            # If your decode runs on CPU, you lose streaming benefits. Move to GPU (non-blocking)
                            out_tensor = out_tensor.to(
                                device=self.opt.rank, non_blocking=True
                            )

                        # keep pair for later synchronization/stack
                        decode_results.append((out_tensor, self.stream_decode))
                    except Exception as e:
                        print(f"Error: {e}\n{traceback.format_exc()}")
                        raise

            # now wait for each stream to finish and move outputs to the current/default stream
            stacked_list = []
            cur = torch.cuda.current_stream(device=self.opt.rank)
            for out_tensor, stream in decode_results:
                # make sure default stream waits for worker stream before we touch `out_tensor`
                cur.wait_stream(stream)
                # Clone to default stream
                out_on_default = out_tensor.clone()  # shape: [T, C, H, W]
                stacked_list.append(out_on_default)

            # Concatenate along temporal dimension (dim=0): [T_total, C, H, W]
            d_hat = torch.cat(stacked_list, dim=0)  # [156, 3, 512, 512] in your example

            # If you really need [B, T, H, W, C], permute:
            print("d_hat", d_hat.shape)
            # if B==1 and you want remove batch dim:
            if d_hat.shape[1] == 1:
                d_hat = d_hat.squeeze(1)

            return {"d_hat": d_hat}

        except Exception as e:
            print(f"Error: {e}\n{traceback.format_exc()}")
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
        if prev_a is not None:
            a = torch.cat([prev_a, a], dim=1)
            if (
                a.shape[1]
                % int(
                    (self.num_frames_for_clip + self.num_prev_frames)
                    * self.opt.sampling_rate
                    / self.opt.fps
                )
                != 0
            ):
                a = F.pad(
                    a,
                    (
                        0,
                        int(
                            (self.num_frames_for_clip + self.num_prev_frames)
                            * self.opt.sampling_rate
                            / self.opt.fps
                        )
                        - a.shape[1],
                    ),
                    mode="replicate",
                )
            a = self.get_wav2vec2_feature(
                a, seq_len=self.num_frames_for_clip + self.num_prev_frames
            )
        else:
            if (
                a.shape[1]
                % int(self.num_frames_for_clip * self.opt.sampling_rate / self.opt.fps)
                != 0
            ):
                a = F.pad(
                    a,
                    (
                        0,
                        int(
                            self.num_frames_for_clip
                            * self.opt.sampling_rate
                            / self.opt.fps
                        )
                        - a.shape[1],
                    ),
                    mode="replicate",
                )
            a = self.get_wav2vec2_feature(a, seq_len=self.num_frames_for_clip)

        return self.audio_projection(a)  # frame by frame

    @torch.no_grad()
    def inference(self, a: torch.Tensor, seq_len: int) -> torch.Tensor:
        if a.shape[1] % int(seq_len * self.opt.sampling_rate / self.opt.fps) != 0:
            a = F.pad(
                a,
                (0, int(seq_len * self.opt.sampling_rate / self.opt.fps) - a.shape[1]),
                mode="replicate",
            )
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
        # Concatenate previous audio if provided
        if prev_a is not None:
            a = torch.cat([prev_a, a], dim=1)

        logits = self.wav2vec2_for_emotion(a).logits
        return F.softmax(logits, dim=1)
