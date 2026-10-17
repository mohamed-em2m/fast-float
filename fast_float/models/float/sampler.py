"""sampler.py - Flow matching ODE sampling and streaming chunk generation for FLOAT."""

import contextlib
import logging
import math
import queue
import threading
import traceback

import torch
import torch.nn.functional as F

from fast_float.utils.timing import timed

try:
    from torchdiffeq import odeint
except ImportError:
    odeint = None

logger = logging.getLogger(__name__)


class SamplerMixin:
    """Mixin providing ODE sampling (sample) and streaming inference (inference_stream) for FLOAT."""

    @torch.no_grad()
    def sample(
        self,
        data: dict,
        a_cfg_scale: float = 1.0,
        r_cfg_scale: float = 1.0,
        e_cfg_scale: float = 1.0,
        emo: str | None = None,
        nfe: int = 10,
        seed: int | None = None,
        s_r=None,
    ) -> torch.Tensor:
        rank = getattr(self.opt, "rank", 0)
        device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
        cur_stream = torch.cuda.current_stream(device) if torch.cuda.is_available() else None
        encode_stream = self.stream_encode if (self.stream_encode is not None and torch.cuda.is_available()) else None

        if cur_stream is not None and encode_stream is not None and cur_stream != encode_stream:
            encode_stream.wait_stream(cur_stream)

        encode_ctx = torch.cuda.stream(encode_stream) if encode_stream is not None else contextlib.nullcontext()
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
            with timed("audio_encode", gpu=True):
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
                with timed("emotion_encode", gpu=True):
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
        _cfg_active = a_cfg_scale != 1.0 or r_cfg_scale != 1.0 or e_cfg_scale != 1.0
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
        for t in range(0, math.ceil(T / self.num_frames_for_clip)):
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
                prev_x_t = torch.zeros(B, self.num_prev_frames, self.opt.dim_w, device=device)
                prev_wa_t = torch.zeros(B, self.num_prev_frames, self.opt.dim_w, device=device)
            else:
                # sample_t / wa_t come from the previous chunk iteration; the loop
                # always starts at t == 0 (which takes the branch above), so both
                # are defined before any use (ruff cannot see this: noqa F821).
                prev_x_t = sample_t[:, -self.num_prev_frames :]  # noqa: F821
                prev_wa_t = wa_t[:, -self.num_prev_frames :]  # noqa: F821

            wa_t = wa[:, t * self.num_frames_for_clip : (t + 1) * self.num_frames_for_clip]
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

                _prev_x_cat = torch.cat([prev_x_t, prev_x_t, prev_x_t], dim=0)
                _prev_wa_cat = torch.cat([prev_wa_t, prev_wa_t, prev_wa_t], dim=0)

            def sample_chunk(
                tt,
                zt,
                _ac=_audio_cat,
                _rc=_ref_cat,
                _ec=_emotion_cat,
                _pxc=_prev_x_cat,
                _pwc=_prev_wa_cat,
                _wa=wa_t,
                _px=prev_x_t,
                _pwa=prev_wa_t,
            ):
                out = self.fmt.forward_with_cfv(
                    t=tt.unsqueeze(0),
                    x=zt,
                    wa=_wa,
                    wr=r_s,
                    we=we,
                    prev_x=_px,
                    prev_wa=_pwa,
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
            with timed("ode_solve", gpu=True, chunk=t, batch=B):
                trajectory_t = odeint(sample_chunk, x0, time, **self.odeint_kwargs)
            sample_t = trajectory_t[-1]

            # Trim to exact length T on the last chunk
            frames_remaining = T - decoded_length
            sample_t_trim = sample_t[:, :frames_remaining] if frames_remaining < sample_t.shape[1] else sample_t
            samples_len = sample_t_trim.shape[1]
            decoded_length += samples_len

            # Add reference and yield chunk
            s_r_b = s_r.unsqueeze(1) if (s_r is not None and s_r.ndim == 2) else s_r
            s_r_d_t = (s_r_b + sample_t_trim) if s_r_b is not None else sample_t_trim
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

        # Reference features live on the primary GPU; move them to any secondary
        # GPU on a dedicated transfer stream so the cross-device copy is enqueued
        # asynchronously and the primary GPU's stream is never drained to do it.
        if torch.cuda.is_available() and len(devs) > 1:
            feats_ready = torch.cuda.Event()
            feats_ready.record(torch.cuda.current_stream(primary))
            feats_ready.synchronize()  # features are final; wait once, not per device

        gpu_ctx = []
        for dev in devs:
            ae = self._get_ae(dev, primary)
            is_prim = dev == primary or (
                dev.index is not None and primary.index is not None and dev.index == primary.index
            )
            if is_prim:
                feats = s_r_feats
            elif torch.cuda.is_available():
                xfer = torch.cuda.Stream(device=dev)
                with torch.cuda.stream(xfer):
                    feats = [f.to(dev, non_blocking=True) for f in s_r_feats]
                xfer.synchronize()  # drain only this device's transfer stream
            else:
                feats = [f.to(dev) for f in s_r_feats]
            if is_prim:
                prim_streams = [st for st in (self.stream_decode, self.stream_decode_2) if st is not None]
                if not prim_streams and torch.cuda.is_available():
                    prim_streams = [torch.cuda.Stream(device=dev)]
                gpu_ctx.extend((dev, ae, feats, st) for st in prim_streams)
            else:
                st = torch.cuda.Stream(device=dev) if torch.cuda.is_available() else None
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
                ctx = torch.autocast("cuda", dtype=amp_dtype) if amp_on else contextlib.nullcontext()
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

        prod_thread = threading.Thread(target=with_amp(producer), name="sampler_producer", daemon=True)
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

                    with timed("decode", gpu=True, device=str(dev)):
                        out = self._decode_latent_batched(
                            ae,
                            lat,
                            feats,
                            batch_size=getattr(self.opt, "decode_batch_size", None),
                            device=dev,
                        )
                    if (not out.is_cuda and torch.cuda.is_available()) or out.device != primary:
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


class FloatSampler(SamplerMixin):
    """Standalone sampler component that coordinates condition encoders and FMT ODE solving."""

    def __init__(
        self,
        opt,
        fmt=None,
        audio_encoder=None,
        emotion_encoder=None,
        motion_autoencoder=None,
        stream_encode=None,
    ):
        self.opt = opt
        self.fmt = fmt
        self.audio_encoder = audio_encoder
        self.emotion_encoder = emotion_encoder
        self.motion_autoencoder = motion_autoencoder
        self.stream_encode = stream_encode
        self.num_frames_for_clip = int(self.opt.wav2vec_sec * self.opt.fps)
        self.num_prev_frames = int(self.opt.num_prev_frames)
        self.odeint_kwargs = {
            "atol": self.opt.ode_atol,
            "rtol": self.opt.ode_rtol,
            "method": self.opt.torchdiffeq_ode_method,
        }


__all__ = ["FloatSampler", "SamplerMixin"]
