"""ONNX Runtime backend for the S2E emotion classifier.

Replaces the torch ``Wav2Vec2ForSpeechClassification`` forward at inference
time with an ``onnxruntime.InferenceSession`` over ``onnx/emotion_v1.onnx``
(waveform ``[B, L]`` -> probabilities ``[B, 7]``).

The exported graph already applies softmax, so session outputs are final
probabilities — no extra normalization needed.
"""
from __future__ import annotations

import os

import torch

from fast_float.onnx.export import make_ort_session

LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


def _dynamo_disable(fn):
    """No-op decorator when torch._dynamo is unavailable."""
    try:
        from torch._dynamo import disable

        return disable(fn)
    except Exception:
        return fn


class EmotionOnnxBackend:
    """Thin wrapper around an ORT session for emotion probabilities.

    Keeps torch semantics for callers: torch in (any device/dtype) ->
    float32 probs on the input's device. Always runs fp32 on CPU numpy,
    so AMP/autocast contexts in the caller are harmless. The graph has
    symbolic batch/length, so one ``sess.run`` covers any ``[B, L]``.
    """

    def __init__(self, onnx_path: str, use_gpu: bool | None = None):
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"emotion ONNX not found: {onnx_path}")
        if use_gpu is None:
            use_gpu = torch.cuda.is_available()
        self.path = os.path.abspath(onnx_path)
        self.session = make_ort_session(self.path, use_gpu=use_gpu)
        self.input_name = self.session.get_inputs()[0].name

    @property
    def providers(self) -> list:
        return self.session.get_providers()

    @_dynamo_disable
    def _run(self, wav_np):
        return self.session.run(None, {self.input_name: wav_np})[0]

    @torch.no_grad()
    def predict_probs(self, a: torch.Tensor) -> torch.Tensor:
        """Waveform ``[L]`` or ``[B, L]`` -> probs ``[B, 7]`` (float32)."""
        device = a.device
        if a.ndim == 1:
            a = a.unsqueeze(0)
        wav_np = a.detach().to("cpu", dtype=torch.float32).numpy()
        probs = self._run(wav_np)
        return torch.from_numpy(probs).to(device=device, dtype=torch.float32)


class Wav2vecOnnxBackend:
    """Thin wrapper around an ORT session for FLOAT audio features.

    Covers feature-projection + the 12-layer transformer + the trained
    ``audio_projection`` head (``onnx/wav2vec_v1.onnx``)::

        conv_feats [B, T, 512] -> audio feats [B, T, dim_w] (float32)

    The caller (``AudioEncoder.inference``) produces ``conv_feats`` in
    torch via ``Wav2VecModel.feature_extract`` (conv front-end +
    interpolation). Inference-only: gradients cannot flow through ORT,
    so training paths must keep using the torch module.
    """

    def __init__(self, onnx_path: str, use_gpu: bool | None = None):
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"wav2vec ONNX not found: {onnx_path}")
        if use_gpu is None:
            use_gpu = torch.cuda.is_available()
        self.path = os.path.abspath(onnx_path)
        self.session = make_ort_session(self.path, use_gpu=use_gpu)
        self.input_name = self.session.get_inputs()[0].name

    @property
    def providers(self) -> list:
        return self.session.get_providers()

    @property
    def out_dim(self) -> int | None:
        """Static trailing dim of ``audio_feats``, or None if dynamic."""
        shape = self.session.get_outputs()[0].shape
        dim = shape[-1] if shape else None
        return int(dim) if isinstance(dim, int) else None

    @_dynamo_disable
    def _run(self, feats_np):
        return self.session.run(None, {self.input_name: feats_np})[0]

    @torch.no_grad()
    def run_features(self, conv_feats: torch.Tensor) -> torch.Tensor:
        """``[B, T, 512]`` -> ``[B, T, dim_w]`` (float32, input device)."""
        device = conv_feats.device
        feats_np = conv_feats.detach().to("cpu", dtype=torch.float32).numpy()
        out = self._run(feats_np)
        return torch.from_numpy(out).to(device=device, dtype=torch.float32)


class FMTStepBackend:
    """Thin wrapper around an ORT session for one FMT vector-field step.

    Covers ``FlowMatchingTransformer.forward(train=False)``::

        (t[1], x[B,50,512], wa[B,50,512], wr[B,1,512], we[B,1,7],
         prev_x[B,10,512], prev_wa[B,10,512]) -> vec [B,60,512]

    Batch is dynamic: the same graph serves plain (B) and CFG (3B) calls.
    The caller slices ``vec[:, 10:]`` for the current chunk.
    """

    INPUTS = ("t", "x", "wa", "wr", "we", "prev_x", "prev_wa")

    def __init__(self, onnx_path: str, use_gpu: bool | None = None):
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"fmt ONNX not found: {onnx_path}")
        if use_gpu is None:
            use_gpu = torch.cuda.is_available()
        self.path = os.path.abspath(onnx_path)
        self.session = make_ort_session(self.path, use_gpu=use_gpu)

    @property
    def providers(self) -> list:
        return self.session.get_providers()

    @_dynamo_disable
    def _run(self, feed):
        return self.session.run(None, feed)[0]

    @torch.no_grad()
    def run_step(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        wa: torch.Tensor,
        wr: torch.Tensor,
        we: torch.Tensor,
        prev_x: torch.Tensor,
        prev_wa: torch.Tensor,
    ) -> torch.Tensor:
        """Single vector-field eval -> ``[B, 60, 512]`` (input device)."""
        device = x.device
        feed = {
            name: val.detach().to("cpu", dtype=torch.float32).numpy()
            for name, val in zip(
                self.INPUTS, (t, x, wa, wr, we, prev_x, prev_wa)
            )
        }
        out = self._run(feed)
        return torch.from_numpy(out).to(device=device, dtype=torch.float32)
