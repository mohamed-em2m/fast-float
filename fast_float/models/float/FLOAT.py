"""FLOAT.py - Audio-driven talking portrait video generation model architecture.

Orchestrates motion latent encoding/decoding, Flow Matching Transformer sampling,
condition encoders, and streaming video pipelines.
"""

import logging
import threading
import traceback

import torch

from fast_float.models import BaseModel
from fast_float.models.float.condition_encoders import Audio2Emotion, AudioEncoder
from fast_float.models.float.decoder import DecoderMixin, FloatDecoder
from fast_float.models.float.FMT import FlowMatchingTransformer
from fast_float.models.float.generator import Generator
from fast_float.models.float.sampler import FloatSampler, SamplerMixin
from fast_float.models.float.video_pipeline import SaveVideoPipeline, StreamingVideoSaver
from fast_float.utils.decode_tuner import DecodeAutoTuner, DecodeProfile, GiB
from fast_float.utils.stream_utils import _ABORT, FrameData, _qget

torch.backends.cudnn.benchmark = True
logger = logging.getLogger(__name__)


class FLOAT(BaseModel, DecoderMixin, SamplerMixin, StreamingVideoSaver):
    """FLOAT: Flow Matching for Audio-Driven Talking Portrait Video Generation.

    Composes motion latent autoencoders, condition encoders, flow matching transformer,
    ODE sampler, multi-GPU batch decoder, and streaming video saving pipeline.
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        logger.info(opt)
        rank = getattr(self.opt, "rank", 0)
        self.rank = rank

        # Guards _ae_replicas / _decode_profiles creation (decode worker setup
        # is single-threaded today, but replicas/profiles persist across calls).
        self._decode_lock = threading.Lock()
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
        self.motion_autoencoder = Generator(size=opt.input_size, style_dim=opt.dim_w, motion_dim=opt.dim_m)
        self.motion_autoencoder.requires_grad_(False)

        # condition encoders
        self.audio_encoder = AudioEncoder(opt)
        self.emotion_encoder = Audio2Emotion(opt)

        # FMT: Flow Matching Transformer
        self.fmt = FlowMatchingTransformer(opt)

        # ODE options
        self.odeint_kwargs = {
            "atol": self.opt.ode_atol,
            "rtol": self.opt.ode_rtol,
            "method": self.opt.torchdiffeq_ode_method,
        }

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
        seeds=None,
    ) -> dict:
        try:
            s, a = data["s"], data["a"]
            B_s = s.shape[0] if s.ndim >= 4 else 1
            B_a = a.shape[0] if a.ndim >= 2 else 1
            is_batched = max(B_s, B_a) > 1

            stacked_list = list(
                self.inference_stream(
                    data=data,
                    a_cfg_scale=a_cfg_scale,
                    r_cfg_scale=r_cfg_scale,
                    e_cfg_scale=e_cfg_scale,
                    emo=emo,
                    nfe=nfe,
                    seed=seed,
                    seeds=seeds,
                )
            )

            dim = 1 if is_batched else 0  # batched -> [B, total_T, 3, H, W] else [total_T, 3, H, W]
            d_hat = torch.cat(stacked_list, dim=dim)
            return {"d_hat": d_hat}

        except Exception as e:
            logger.error(f"Error in inference: {e}\n{traceback.format_exc()}")
            raise


__all__ = [
    "FLOAT",
    "_ABORT",
    "Audio2Emotion",
    "AudioEncoder",
    "DecodeAutoTuner",
    "DecodeProfile",
    "DecoderMixin",
    "FloatDecoder",
    "FloatSampler",
    "FrameData",
    "GiB",
    "SamplerMixin",
    "SaveVideoPipeline",
    "StreamingVideoSaver",
    "_qget",
]
