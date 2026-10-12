"""Inference and ODE sampling operations mixin for FLOAT.

Contains:
  - FullInferenceMixin:
      - clear_decode_cache: Cache invalidation for dynamic VRAM tuning
      - get_optimal_decode_batch_size: Tiered batch size determination based on free VRAM
      - _decode_latent_batched: High-throughput micro-batch decoding
      - decode_latent_into_image: High-level frame synthesis
      - sample: Generator solving the continuous ODE flow matching problem chunk-by-chunk
      - inference_stream: Pipelined streaming inference yielding decoded visual clips
      - inference: Monolithic end-to-end evaluation entry point
"""

from __future__ import annotations

import contextlib
import logging
import math
import queue
import threading
import traceback
from typing import Generator

import torch
import torch.nn.functional as F

try:
    from torchdiffeq import odeint
except ImportError:
    odeint = None

logger = logging.getLogger(__name__)


class FullInferenceMixin:
    """Mixin containing full-sequence sampling and batched frame decoding operations."""

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

            cache = getattr(self, "_decode_bs_cache", None)
            if cache is None:
                self._decode_bs_cache = {}
                cache = self._decode_bs_cache
            if idx in cache:
                return cache[idx]

            free_b, total_b = torch.cuda.mem_get_info(idx)
            gib = 1024**3
            free_gib = free_b / gib
            total_gib = total_b / gib

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
        """Decode motion latents into images in mini-batches to prevent OOM and maximize GPU throughput."""
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
        """Decode latent into image with broadcasted features in optimized mini-batches."""
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
    ) -> Generator[torch.Tensor, None, None]:
        """Continuous ODE Flow Matching sampling loop."""
        rank = getattr(self.opt, "rank", 0)
        device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
        cur_stream = torch.cuda.current_stream(device) if torch.cuda.is_available() else None
        encode_stream = getattr(self, "stream_encode", None)
        if cur_stream is not None and encode_stream is not None and cur_stream != encode_stream:
            encode_stream.wait_stream(cur_stream)

        encode_ctx = (
            torch.cuda.stream(encode_stream)
            if (encode_stream is not None and torch.cuda.is_available())
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

            steps = nfe if nfe is not None else getattr(self.opt, "nfe", 10)
            time_steps = torch.linspace(0, 1, steps, device=device)

            T = int(math.floor(a.shape[1] / self.opt.sampling_rate * self.opt.fps))
            wa = self.audio_encoder.inference(a.to(device), seq_len=T)

            if emo is not None:
                if emo == "S2E":
                    we = self.emotion_encoder.predict_emotion(a.to(device))
                elif isinstance(emo, str):
                    emo_id = self.emotion_encoder.label2id.get(emo.lower(), None)
                    if emo_id is None:
                        raise ValueError(f"Emotion {emo} is not valid.")
                    we = (
                        F.one_hot(
                            torch.tensor(emo_id, device=device),
                            num_classes=self.opt.dim_e,
                        )
                        .float()
                        .unsqueeze(0)
                    )
                elif isinstance(emo, list):
                    emo_id = [self.emotion_encoder.label2id.get(e.lower(), None) for e in emo]
                    if None in emo_id:
                        raise ValueError(f"Emotion {emo} contains invalid emotion.")
                    we = (
                        F.one_hot(
                            torch.tensor(emo_id, device=device),
                            num_classes=self.opt.dim_e,
                        )
                        .float()
                    )
                elif isinstance(emo, torch.Tensor):
                    we = emo.to(device)
                else:
                    raise ValueError(f"Emotion {emo} is not valid.")
            else:
                we = self.emotion_encoder.predict_emotion(a.to(device))

            if we.ndim == 2:
                we = we.unsqueeze(1)
            if we.shape[0] != B and we.shape[0] == 1:
                we = we.expand(B, -1, -1)

        if cur_stream is not None and encode_stream is not None and cur_stream != encode_stream:
            cur_stream.wait_stream(encode_stream)

        decoded_length = 0
        _cfg_active = a_cfg_scale != 1.0 or r_cfg_scale != 1.0 or e_cfg_scale != 1.0

        _ref_cat = None
        _emotion_cat = None
        if _cfg_active:
            wr_ready = r_s
            if wr_ready.ndim == 1:
                wr_ready = wr_ready.unsqueeze(0).unsqueeze(1)
            elif wr_ready.ndim == 2:
                wr_ready = wr_ready.unsqueeze(1)
            if wr_ready.shape[0] != B and wr_ready.shape[0] == 1:
                wr_ready = wr_ready.expand(B, -1, -1)
            _ref_cat = torch.cat([wr_ready, wr_ready, wr_ready], dim=0)

            we_ready = we
            if we_ready.ndim == 1:
                we_ready = we_ready.unsqueeze(0).unsqueeze(1)
            elif we_ready.ndim == 2:
                we_ready = we_ready.unsqueeze(1)
            if we_ready.shape[0] != B and we_ready.shape[0] == 1:
                we_ready = we_ready.expand(B, -1, -1)
            _null_we = torch.zeros_like(we_ready)
            _emotion_cat = torch.cat([_null_we, we_ready, _null_we], dim=0)

        sample_t = None
        wa_t = None
        for t in range(0, int(math.ceil(T / self.num_frames_for_clip))):
            if getattr(self.opt, "fix_noise_seed", False):
                noise_seed = getattr(self.opt, "seed", 25) if seed is None else seed
                g = torch.Generator(device=device)
                g.manual_seed(noise_seed)
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

            if t == 0:
                prev_x_t = torch.zeros(B, self.num_prev_frames, self.opt.dim_w, device=device)
                prev_wa_t = torch.zeros(B, self.num_prev_frames, self.opt.dim_w, device=device)
            else:
                prev_x_t = sample_t[:, -self.num_prev_frames :]
                prev_wa_t = wa_t[:, -self.num_prev_frames :]

            wa_t = wa[:, t * self.num_frames_for_clip : (t + 1) * self.num_frames_for_clip]
            if wa_t.shape[1] < self.num_frames_for_clip:
                wa_t = F.pad(
                    wa_t,
                    (0, 0, 0, self.num_frames_for_clip - wa_t.shape[1]),
                    mode="replicate",
                )

            _audio_cat = None
            _prev_x_cat = None
            _prev_wa_cat = None
            if _cfg_active:
                wa_t_exp = wa_t
                if wa_t_exp.shape[0] != B and wa_t_exp.shape[0] == 1:
                    wa_t_exp = wa_t_exp.expand(B, -1, -1)
                _null_wa = torch.zeros_like(wa_t_exp)
                _audio_cat = torch.cat([_null_wa, wa_t_exp, wa_t_exp], dim=0)
                _prev_x_cat = torch.cat([prev_x_t, prev_x_t, prev_x_t], dim=0)
                _prev_wa_cat = torch.cat([prev_wa_t, prev_wa_t, prev_wa_t], dim=0)

            def sample_chunk(
                tt, zt,
                _ac=_audio_cat, _rc=_ref_cat, _ec=_emotion_cat,
                _pxc=_prev_x_cat, _pwc=_prev_wa_cat,
            ):
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
                return out[:, self.num_prev_frames :]

            if odeint is None:
                raise ImportError("torchdiffeq is required for ODE sampling.")
            trajectory_t = odeint(sample_chunk, x0, time_steps, **self.odeint_kwargs)
            sample_t = trajectory_t[-1]

            frames_remaining = T - decoded_length
            if frames_remaining < sample_t.shape[1]:
                sample_t_trim = sample_t[:, :frames_remaining]
            else:
                sample_t_trim = sample_t
            decoded_length += sample_t_trim.shape[1]

            s_r_b = s_r.unsqueeze(1) if (s_r is not None and s_r.ndim == 2) else s_r
            s_r_d_t = (s_r_b + sample_t_trim) if s_r_b is not None else sample_t_trim
            if B == 1:
                yield s_r_d_t.squeeze(0)
            else:
                yield s_r_d_t

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
    ) -> Generator[torch.Tensor, None, None]:
        """Streams sampling latents and decodes them concurrently on dedicated CUDA stream(s)."""
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

        if torch.cuda.is_available():
            _, devs = self._decode_devices()
        else:
            devs = [primary]

        gpu_ctx = []
        for dev in devs:
            ae = self._get_ae(dev, primary)
            is_prim = (dev == primary or (dev.index is not None and primary.index is not None and dev.index == primary.index))
            feats = s_r_feats if is_prim else [f.to(dev) for f in s_r_feats]
            if is_prim:
                prim_streams = [
                    st for st in (getattr(self, "stream_decode", None), getattr(self, "stream_decode_2", None))
                    if st is not None
                ]
                if not prim_streams and torch.cuda.is_available():
                    prim_streams = [torch.cuda.Stream(device=dev)]
                for st in prim_streams:
                    gpu_ctx.append((dev, ae, feats, st))
            else:
                st = torch.cuda.Stream(device=dev) if torch.cuda.is_available() else None
                gpu_ctx.append((dev, ae, feats, st))

        num_devs = len(gpu_ctx)

        if torch.cuda.is_available():
            init_ev = torch.cuda.Event()
            init_ev.record(torch.cuda.current_stream(primary))
            for _, _, _, st in gpu_ctx:
                if st is not None and st != torch.cuda.current_stream(primary):
                    st.wait_event(init_ev)

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
                    s_r=r_s,
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
        cur = torch.cuda.current_stream(device=primary) if torch.cuda.is_available() else None

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

                    lat = frame_latent.to(dev, non_blocking=True) if dev != primary else frame_latent
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
                d_hat = torch.cat(stacked_list, dim=1)
            else:
                d_hat = torch.cat(stacked_list, dim=0)
            return {"d_hat": d_hat}

        except Exception as e:
            logger.error(f"Error in inference: {e}\n{traceback.format_exc()}")
            raise
