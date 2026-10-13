"""
FLOAT: Audio-Driven Talking Face Generation Model with Flow Matching.

This module coordinates:
  - Motion Autoencoder (Generator) for identity and expression latent spaces
  - Flow Matching Transformer (FMT) for continuous ODE trajectory modeling
  - Wav2Vec2 Audio & Emotion conditioning encoders
  - Multi-stream, multi-GPU decode optimization and hardware video encoding
  - Decoupled Omni streaming scheduling
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from fast_float.models import BaseModel
from fast_float.models.float.FMT import FlowMatchingTransformer
from fast_float.models.float.generator import Generator

# Sub-module imports & backward-compatible re-exports
from fast_float.models.float.audio_encoder import (
    AudioEncoder,
    Audio2Emotion,
    FrameData,
)
from fast_float.models.float.video_saver import (
    DecodeProfile,
    DecodeAutoTuner,
    StreamingVideoSaver,
)
from fast_float.models.float.omni_ops import OmniStreamingMixin
from fast_float.models.float.inference_ops import FullInferenceMixin

logger = logging.getLogger(__name__)


class FLOAT(BaseModel, StreamingVideoSaver, OmniStreamingMixin, FullInferenceMixin):
    """The central Flow Matching Talking Face generation model."""

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

        # 1. Motion latent auto-encoder
        self.motion_autoencoder = Generator(
            size=opt.input_size, style_dim=opt.dim_w, motion_dim=opt.dim_m
        )
        self.motion_autoencoder.requires_grad_(False)

        # 2. Condition encoders
        self.audio_encoder = AudioEncoder(opt)
        self.emotion_encoder = Audio2Emotion(opt)

        # 3. Flow Matching Transformer (FMT)
        self.fmt = FlowMatchingTransformer(opt)

        # 4. ODE solver options
        self.odeint_kwargs = {
            "atol": self.opt.ode_atol,
            "rtol": self.opt.ode_rtol,
            "method": self.opt.torchdiffeq_ode_method,
        }

    def compile_models(
        self,
        mode: str = "default",
        fullgraph: bool = False,
        dynamic: Optional[bool] = None,
    ) -> bool:
        """Compile internal PyTorch submodules using torch.compile for high-throughput inference.

        Compiles:
          1. FlowMatchingTransformer (self.fmt) - the ODE sampling hotspot
          2. StyleDecoder (self.motion_autoencoder.dec) - per-frame image synthesis hotspot
          3. Motion Encoder (self.motion_autoencoder.enc) - source image encoder
          4. Audio projection layers (self.audio_encoder.audio_projection)
        """
        if not hasattr(torch, "compile"):
            logger.warning("torch.compile is not supported in this PyTorch version.")
            return False

        try:
            import torch._dynamo
            torch._dynamo.config.verbose = False
            torch._dynamo.config.suppress_errors = True
            logging.getLogger("torch._dynamo").setLevel(logging.WARNING)
            logging.getLogger("torch._inductor").setLevel(logging.WARNING)
            logging.getLogger("torch.fx").setLevel(logging.WARNING)
        except Exception:
            pass

        logger.info(f"Applying torch.compile (mode='{mode}') to FLOAT submodels...")

        # 1. FlowMatchingTransformer (FMT)
        if hasattr(self, "fmt") and self.fmt is not None:
            try:
                self.fmt = torch.compile(self.fmt, mode=mode, fullgraph=fullgraph, dynamic=dynamic)
                logger.info("✓ FlowMatchingTransformer compiled successfully")
            except Exception as e:
                logger.warning(f"! Failed to compile FlowMatchingTransformer: {e}")

        # 2. StyleDecoder (Synthesis)
        if hasattr(self, "motion_autoencoder") and hasattr(self.motion_autoencoder, "dec"):
            try:
                self.motion_autoencoder.dec = torch.compile(
                    self.motion_autoencoder.dec, mode=mode, fullgraph=fullgraph, dynamic=dynamic
                )
                logger.info("✓ StyleDecoder compiled successfully")
            except Exception as e:
                logger.warning(f"! Failed to compile StyleDecoder: {e}")

        # 3. Motion Encoder
        if hasattr(self, "motion_autoencoder") and hasattr(self.motion_autoencoder, "enc"):
            try:
                self.motion_autoencoder.enc = torch.compile(
                    self.motion_autoencoder.enc, mode=mode, fullgraph=fullgraph, dynamic=dynamic
                )
                logger.info("✓ Motion Encoder compiled successfully")
            except Exception as e:
                logger.warning(f"! Failed to compile Motion Encoder: {e}")

        # 4. Audio Projection
        if hasattr(self, "audio_encoder") and hasattr(self.audio_encoder, "audio_projection"):
            try:
                self.audio_encoder.audio_projection = torch.compile(
                    self.audio_encoder.audio_projection, mode=mode, fullgraph=fullgraph, dynamic=dynamic
                )
                logger.info("✓ Audio Projection compiled successfully")
            except Exception as e:
                logger.warning(f"! Failed to compile Audio Projection: {e}")

        return True

    @torch.no_grad()
    def encode_image_into_latent(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """Encodes portrait image into appearance latents, motion direction, and multi-scale feature pyramid."""
        if x.ndim == 3:
            x = x.unsqueeze(0)
        x_r, _, x_r_feats = self.motion_autoencoder.enc(x, input_target=None)
        x_r_lambda = self.motion_autoencoder.enc.fc(x_r)
        return x_r, x_r_lambda, x_r_feats

    @torch.no_grad()
    def encode_identity_into_motion(self, x_r: torch.Tensor) -> torch.Tensor:
        """Projects image identity latent into the motion direction vector."""
        if x_r.ndim == 1:
            x_r = x_r.unsqueeze(0)
        x_r_lambda = self.motion_autoencoder.enc.fc(x_r)
        return self.motion_autoencoder.dec.direction(x_r_lambda)


__all__ = [
    "FLOAT",
    "AudioEncoder",
    "Audio2Emotion",
    "FrameData",
    "DecodeProfile",
    "DecodeAutoTuner",
    "StreamingVideoSaver",
    "OmniStreamingMixin",
    "FullInferenceMixin",
]
