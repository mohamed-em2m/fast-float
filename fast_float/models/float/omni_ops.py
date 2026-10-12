"""Omni streaming and chunk-based decoupled execution mixin for FLOAT.

Contains:
  - OmniStreamingMixin:
      - prepare_omni_conditions: Invariant feature pre-extraction (image identity, audio, emotion)
      - sample_omni_chunk: ODE latent sampling for a single chunk with temporal continuity
      - decode_omni_chunk: StyleDecoder synthesis from latent tasks to uint8 RGB frames
"""

from __future__ import annotations

import math
import uuid
from typing import Any, Optional

import torch
import torch.nn.functional as F

try:
    from torchdiffeq import odeint
except ImportError:
    odeint = None

from fast_float.sched.types import FloatOmniRequest, LatentChunkTask


class OmniStreamingMixin:
    """Decoupled streaming and chunked generation methods for the Omni pipeline."""

    @torch.no_grad()
    def prepare_omni_conditions(
        self,
        ref_image: torch.Tensor,
        audio: torch.Tensor,
        emotion: Optional[str] = None,
        request_id: Optional[str] = None,
        callback: Optional[Any] = None,
        a_cfg_scale: float = 2.0,
        r_cfg_scale: float = 1.0,
        e_cfg_scale: float = 1.0,
        nfe: int = 10,
        output_format: str = "tensor",
    ) -> FloatOmniRequest:
        """Pre-extract all invariant conditioning features for a streaming request once."""
        rank = getattr(self.opt, "rank", 0)
        device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"

        if ref_image.ndim == 3:
            ref_image = ref_image.unsqueeze(0)
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)

        ref_image = ref_image.to(device)
        audio = audio.to(device)

        # 1. Image identity
        x_r, x_r_lambda, x_r_feats = self.encode_image_into_latent(ref_image)
        r_s = self.motion_autoencoder.dec.direction(x_r_lambda)

        # 2. Audio features
        T_frames = int(math.floor(audio.shape[1] / self.opt.sampling_rate * self.opt.fps))
        wa = self.audio_encoder.inference(audio, seq_len=T_frames)

        # 3. Emotion features
        if emotion is not None and emotion not in ("S2E", None):
            emo_idx = self.emotion_encoder.label2id.get(str(emotion).lower(), None)
            if emo_idx is not None:
                we = (
                    F.one_hot(
                        torch.tensor(emo_idx, device=device),
                        num_classes=self.opt.dim_e,
                    )
                    .float()
                    .unsqueeze(0)
                    .unsqueeze(0)
                )
            else:
                we = torch.zeros(1, 1, self.opt.dim_e, device=device)
        else:
            we = self.emotion_encoder.predict_emotion(audio)
            if we.ndim == 2:
                we = we.unsqueeze(1)

        total_chunks = max(1, int(math.ceil(T_frames / self.num_frames_for_clip)))
        req_id = request_id or f"omni_{uuid.uuid4().hex[:8]}"

        return FloatOmniRequest(
            request_id=req_id,
            ref_image=ref_image,
            audio=audio,
            ref_latent=r_s,
            ref_feats=x_r_feats,
            wa=wa,
            we=we,
            total_chunks=total_chunks,
            total_frames=T_frames,
            current_chunk=0,
            callback=callback,
            a_cfg_scale=a_cfg_scale,
            r_cfg_scale=r_cfg_scale,
            e_cfg_scale=e_cfg_scale,
            nfe=nfe,
            output_format=output_format,
        )

    @torch.no_grad()
    def sample_omni_chunk(self, req: Any) -> LatentChunkTask:
        """Executes ODE latent sampling for a single chunk of req (50 frames)."""
        rank = getattr(self.opt, "rank", 0)
        device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"

        t_idx = req.current_chunk
        B = req.wa.shape[0]

        # Noise initialization
        x0 = torch.randn(
            B,
            self.num_frames_for_clip,
            self.opt.dim_w,
            device=device,
        )

        if t_idx == 0 or req.prev_x is None:
            prev_x_t = torch.zeros(B, self.num_prev_frames, self.opt.dim_w, device=device)
            prev_wa_t = torch.zeros(B, self.num_prev_frames, self.opt.dim_w, device=device)
        else:
            prev_x_t = req.prev_x
            prev_wa_t = req.prev_wa

        wa_t = req.wa[:, t_idx * self.num_frames_for_clip : (t_idx + 1) * self.num_frames_for_clip]
        if wa_t.shape[1] < self.num_frames_for_clip:
            wa_t = F.pad(
                wa_t,
                (0, 0, 0, self.num_frames_for_clip - wa_t.shape[1]),
                mode="replicate",
            )

        # Pre-build CFG tensors
        _cfg_active = req.a_cfg_scale != 1.0 or req.r_cfg_scale != 1.0 or req.e_cfg_scale != 1.0
        _audio_cat = None
        _prev_x_cat = None
        _prev_wa_cat = None
        _ref_cat = None
        _emotion_cat = None

        if _cfg_active:
            wr_ready = req.ref_latent
            if wr_ready.ndim == 1:
                wr_ready = wr_ready.unsqueeze(0).unsqueeze(1)
            elif wr_ready.ndim == 2:
                wr_ready = wr_ready.unsqueeze(1)
            if wr_ready.shape[0] != B and wr_ready.shape[0] == 1:
                wr_ready = wr_ready.expand(B, -1, -1)
            _ref_cat = torch.cat([wr_ready, wr_ready, wr_ready], dim=0)

            we_ready = req.we
            if we_ready.ndim == 1:
                we_ready = we_ready.unsqueeze(0).unsqueeze(1)
            elif we_ready.ndim == 2:
                we_ready = we_ready.unsqueeze(1)
            if we_ready.shape[0] != B and we_ready.shape[0] == 1:
                we_ready = we_ready.expand(B, -1, -1)
            _null_we = torch.zeros_like(we_ready)
            _emotion_cat = torch.cat([_null_we, we_ready, _null_we], dim=0)

            wa_t_exp = wa_t
            if wa_t_exp.shape[0] != B and wa_t_exp.shape[0] == 1:
                wa_t_exp = wa_t_exp.expand(B, -1, -1)
            _null_wa = torch.zeros_like(wa_t_exp)
            _audio_cat = torch.cat([_null_wa, wa_t_exp, wa_t_exp], dim=0)

            _prev_x_cat = torch.cat([prev_x_t, prev_x_t, prev_x_t], dim=0)
            _prev_wa_cat = torch.cat([prev_wa_t, prev_wa_t, prev_wa_t], dim=0)

        # Bind all closed-over tensors as default args to guarantee thread-safety and immutable snapshot
        def sample_fn(
            tt, zt,
            _wa=wa_t, _wr=req.ref_latent, _we=req.we,
            _px=prev_x_t, _pwa=prev_wa_t,
            _ac=_audio_cat, _rc=_ref_cat, _ec=_emotion_cat,
            _pxc=_prev_x_cat, _pwc=_prev_wa_cat,
        ):
            out = self.fmt.forward_with_cfv(
                t=tt.unsqueeze(0),
                x=zt,
                wa=_wa,
                wr=_wr,
                we=_we,
                prev_x=_px,
                prev_wa=_pwa,
                a_cfg_scale=req.a_cfg_scale,
                r_cfg_scale=req.r_cfg_scale,
                e_cfg_scale=req.e_cfg_scale,
                _audio_cat=_ac,
                _ref_cat=_rc,
                _emotion_cat=_ec,
                _prev_x_cat=_pxc,
                _prev_wa_cat=_pwc,
            )
            return out[:, self.num_prev_frames:]

        time_steps = torch.linspace(0, 1, req.nfe, device=device)
        if odeint is None:
            raise ImportError("torchdiffeq is required for ODE sampling.")
        trajectory_t = odeint(sample_fn, x0, time_steps, **self.odeint_kwargs)
        sample_t = trajectory_t[-1]

        # Trim last chunk to exact remaining frames
        if req.total_frames > 0:
            frames_remaining = req.total_frames - req.decoded_frames
            if frames_remaining < sample_t.shape[1]:
                sample_t = sample_t[:, :frames_remaining]
        req.decoded_frames += sample_t.shape[1]

        # Update context for next chunk
        req.prev_x = sample_t[:, -self.num_prev_frames:]
        req.prev_wa = wa_t[:, -self.num_prev_frames:]
        is_last = (req.current_chunk + 1) >= req.total_chunks
        req.current_chunk += 1

        # Add reference motion
        s_r = req.ref_latent
        s_r_b = s_r.unsqueeze(1) if (s_r is not None and s_r.ndim == 2) else s_r
        s_r_d_t = (s_r_b + sample_t) if s_r_b is not None else sample_t
        latents_out = s_r_d_t.squeeze(0) if B == 1 else s_r_d_t

        return LatentChunkTask(
            request_id=req.request_id,
            chunk_idx=t_idx,
            latents=latents_out,
            ref_feats=req.ref_feats,
            is_last=is_last,
            callback=req.callback,
            output_format=req.output_format,
        )

    @torch.no_grad()
    def decode_omni_chunk(self, task: Any, dev: Optional[torch.device] = None):
        """Decodes one LatentChunkTask into uint8 RGB video frames [T, H, W, 3]."""
        rank = getattr(self.opt, "rank", 0)
        primary = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
        dev = dev or primary

        ae = self._get_ae(dev, primary)
        lat = task.latents
        if lat.device != dev:
            lat = lat.to(dev, non_blocking=True)

        frames_u8 = self._decode_u8(ae, task.ref_feats, lat)
        if task.output_format == "numpy":
            return frames_u8.cpu().numpy()
        return frames_u8.cpu()
