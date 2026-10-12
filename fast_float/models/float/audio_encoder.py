"""Audio and emotion condition encoders for FLOAT.

Contains:
  - AudioEncoder: Wav2Vec2 backbone with linear motion projection
  - Audio2Emotion: Speech emotion classification & prediction
  - FrameData: Data structure for video frame chunks
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fast_float.models import BaseModel
from fast_float.models.wav2vec2 import Wav2VecModel
from fast_float.models.wav2vec2_ser import Wav2Vec2ForSpeechClassification


@dataclass
class FrameData:
    """Container for frame data passed between producer and consumer."""

    frame_np: np.ndarray
    frame_idx: int
    is_last: bool = False


class AudioEncoder(BaseModel):
    """Audio feature encoder extracting Wav2Vec2 speech representations projected into motion space."""

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.only_last_features = getattr(opt, "only_last_features", False)

        self.num_frames_for_clip = int(opt.wav2vec_sec * self.opt.fps)
        self.num_prev_frames = int(opt.num_prev_frames)
        wav2vec_path = getattr(opt, "wav2vec_model_path", "facebook/wav2vec2-base-960h")
        if os.path.exists(wav2vec_path):
            self.wav2vec2 = Wav2VecModel.from_pretrained(
                wav2vec_path, local_files_only=True
            )
        else:
            self.wav2vec2 = Wav2VecModel.from_pretrained(wav2vec_path)

        self.wav2vec2.feature_extractor._freeze_parameters()

        for param in self.wav2vec2.parameters():
            param.requires_grad = False

        audio_input_dim = 768 if self.only_last_features else 12 * 768

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

    def forward(self, a: torch.Tensor, prev_a: Optional[torch.Tensor] = None) -> torch.Tensor:
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

        return self.audio_projection(a)

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
    """Audio emotion classifier producing 7-class emotion probabilities from speech audio."""

    def __init__(self, opt):
        super().__init__()
        emo_path = getattr(opt, "audio2emotion_path", "")
        if emo_path and os.path.exists(emo_path):
            self.wav2vec2_for_emotion = Wav2Vec2ForSpeechClassification.from_pretrained(
                emo_path, local_files_only=True
            )
        elif emo_path:
            self.wav2vec2_for_emotion = Wav2Vec2ForSpeechClassification.from_pretrained(
                emo_path
            )
        else:
            self.wav2vec2_for_emotion = None

        if self.wav2vec2_for_emotion is not None:
            self.wav2vec2_for_emotion.eval()

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
        self, a: torch.Tensor, prev_a: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.wav2vec2_for_emotion is None:
            return torch.zeros(a.shape[0] if a.ndim > 1 else 1, len(self.id2label), device=a.device)
        if a.ndim == 1:
            a = a.unsqueeze(0)
        if prev_a is not None:
            if prev_a.ndim == 1:
                prev_a = prev_a.unsqueeze(0)
            a = torch.cat([prev_a, a], dim=1)

        logits = self.wav2vec2_for_emotion(a).logits
        return F.softmax(logits, dim=1)
