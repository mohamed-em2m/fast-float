"""condition_encoders.py - Condition encoders (AudioEncoder and Audio2Emotion) for FLOAT."""

import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from fast_float.models import BaseModel
from fast_float.models.wav2vec2 import Wav2VecModel

logger = logging.getLogger(__name__)


class AudioEncoder(BaseModel):
    """Audio condition encoder that transforms audio waveforms into motion latent conditioning features.

    Exports and utilizes an ONNX backend if configured and available, falling back to PyTorch
    eager or torch.compile inference.
    """

    DEFAULT_ONNX_PATH = "./onnx/wav2vec_v1.onnx"

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.only_last_features = opt.only_last_features

        self.num_frames_for_clip = int(opt.wav2vec_sec * self.opt.fps)
        self.num_prev_frames = int(opt.num_prev_frames)
        if os.path.exists(opt.wav2vec_model_path):
            self.wav2vec2 = Wav2VecModel.from_pretrained(opt.wav2vec_model_path, local_files_only=True)
        else:
            self.wav2vec2 = Wav2VecModel.from_pretrained(opt.wav2vec_model_path)

        self.wav2vec2.feature_extractor._freeze_parameters()

        for _name, param in self.wav2vec2.named_parameters():
            param.requires_grad = False

        audio_input_dim = 768 if opt.only_last_features else 12 * 768

        self.audio_projection = nn.Sequential(nn.Linear(audio_input_dim, opt.dim_w), nn.LayerNorm(opt.dim_w), nn.SiLU())

        # Prefer the ONNX backend for inference (fast, no 362 MB torch
        # weight load) when the config selects it. Skipped when the 12-layer
        # stack is not in use.
        self._onnx_backend = None
        self._compiled_inference = None
        try:
            from fast_float.onnx.backends import resolve_mode

            self._mode = resolve_mode("audio")
        except Exception:
            self._mode = "eager"
        onnx_path = getattr(opt, "wav2vec_onnx_path", self.DEFAULT_ONNX_PATH)
        if self._mode == "onnx" and onnx_path and os.path.exists(onnx_path) and not opt.only_last_features:
            try:
                from fast_float.onnx.runtime import Wav2vecOnnxBackend

                feat_frames = self.num_frames_for_clip + self.num_prev_frames
                backend = Wav2vecOnnxBackend(
                    onnx_path,
                    opt_shapes={"conv_feats": [1, feat_frames, 512]},
                    extra_dynamic={"conv_feats": {1: (10, 2000)}},
                )
                if backend.out_dim is not None and backend.out_dim != opt.dim_w:
                    raise ValueError(f"ONNX out_dim={backend.out_dim} != dim_w={opt.dim_w}")
                self._onnx_backend = backend
                logger.info(f"AudioEncoder: using ONNX backend {onnx_path} (providers={self._onnx_backend.providers})")
            except Exception as e:
                logger.warning(f"AudioEncoder: ONNX backend failed ({e}); falling back to torch weights")
                self._onnx_backend = None

    def get_wav2vec2_feature(self, a: torch.Tensor, seq_len: int) -> torch.Tensor:
        if a.ndim == 1:
            a = a.unsqueeze(0)
        a = self.wav2vec2(a, seq_len=seq_len, output_hidden_states=not self.only_last_features)
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
            target_len = int((self.num_frames_for_clip + self.num_prev_frames) * self.opt.sampling_rate / self.opt.fps)
            if a.shape[1] % target_len != 0:
                pad_amt = target_len - (a.shape[1] % target_len)
                a = F.pad(a, (0, pad_amt), mode="replicate")
            a = self.get_wav2vec2_feature(a, seq_len=self.num_frames_for_clip + self.num_prev_frames)
        else:
            target_len = int(self.num_frames_for_clip * self.opt.sampling_rate / self.opt.fps)
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
        if self._onnx_backend is not None:
            feats = self.wav2vec2.feature_extract(a, seq_len)
            return self._onnx_backend.run_features(feats)
        if self._mode == "compile":
            if self._compiled_inference is None:
                try:
                    from fast_float.onnx.backends import maybe_compile

                    self._compiled_inference = maybe_compile(self._inference_body, "audio")
                except Exception as e:
                    logger.warning(f"AudioEncoder: torch.compile failed ({e}); using eager")
                    self._compiled_inference = self._inference_body
            return self._compiled_inference(a, seq_len)
        return self._inference_body(a, seq_len)

    def _inference_body(self, a: torch.Tensor, seq_len: int) -> torch.Tensor:
        a = self.get_wav2vec2_feature(a, seq_len=seq_len)
        return self.audio_projection(a)

    @property
    def uses_onnx(self) -> bool:
        return self._onnx_backend is not None


class Audio2Emotion(nn.Module):
    """Speech emotion recognition network predicting emotion category distribution.

    Uses an ONNX backend when available or falls back to local PyTorch wav2vec2-ser core.
    """

    DEFAULT_ONNX_PATH = "./onnx/emotion_v1.onnx"

    def __init__(self, opt):
        super().__init__()

        # Prefer the ONNX export (fast, no 1.2 GB torch weight load).
        # Plain attribute on purpose: nn.Module.to()/compile ignore it,
        # and the ORT call below is dynamo-disabled for torch.compile safety.
        self._onnx_backend = None
        self._compiled_body = None
        try:
            from fast_float.onnx.backends import resolve_mode

            self._mode = resolve_mode("emotion")
        except Exception:
            self._mode = "eager"
        onnx_path = getattr(opt, "emotion_onnx_path", self.DEFAULT_ONNX_PATH)
        if self._mode == "onnx" and onnx_path and os.path.exists(onnx_path):
            try:
                from fast_float.onnx.runtime import EmotionOnnxBackend

                emo_len = int(16000 * 3.0)
                self._onnx_backend = EmotionOnnxBackend(
                    onnx_path,
                    opt_shapes={"input_values": [1, emo_len]},
                    extra_dynamic={"input_values": {1: (16000, 160000)}},
                )
                logger.info(f"Audio2Emotion: using ONNX backend {onnx_path} (providers={self._onnx_backend.providers})")
            except Exception as e:
                logger.warning(f"Audio2Emotion: ONNX backend failed ({e}); falling back to torch weights")
                self._onnx_backend = None

        # Torch fallback (local preferred). Skipped when ONNX is up.
        # build_core handles new-transformers incompatibility
        # (from_pretrained -> manual load with tie_weights no-op).
        self.wav2vec2_for_emotion = None
        if self._onnx_backend is None:
            from fast_float.onnx.convert_emotion import build_core

            self.wav2vec2_for_emotion = build_core(opt.audio2emotion_path)
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
    def predict_emotion(self, a: torch.Tensor, prev_a: torch.Tensor = None) -> torch.Tensor:
        if a.ndim == 1:
            a = a.unsqueeze(0)
        # Concatenate previous audio if provided
        if prev_a is not None:
            if prev_a.ndim == 1:
                prev_a = prev_a.unsqueeze(0)
            a = torch.cat([prev_a, a], dim=1)

        if self._onnx_backend is not None:
            # ONNX graph already outputs softmax probabilities.
            return self._onnx_backend.predict_probs(a)
        if self._mode == "compile":
            if self._compiled_body is None:
                try:
                    from fast_float.onnx.backends import maybe_compile

                    self._compiled_body = maybe_compile(self._predict_body, "emotion")
                except Exception as e:
                    logger.warning(f"Audio2Emotion: torch.compile failed ({e}); using eager")
                    self._compiled_body = self._predict_body
            return self._compiled_body(a)
        return self._predict_body(a)

    def _predict_body(self, a: torch.Tensor) -> torch.Tensor:
        logits = self.wav2vec2_for_emotion(a).logits
        return F.softmax(logits, dim=1)

    @property
    def uses_onnx(self) -> bool:
        return self._onnx_backend is not None


__all__ = ["Audio2Emotion", "AudioEncoder"]
