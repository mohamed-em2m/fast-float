"""ONNX Runtime backend for the S2E emotion classifier.

Replaces the torch ``Wav2Vec2ForSpeechClassification`` forward at inference
time with an ``onnxruntime.InferenceSession`` over ``onnx/emotion_v1.onnx``
(waveform ``[B, L]`` -> probabilities ``[B, 7]``).

The exported graph already applies softmax, so session outputs are final
probabilities — no extra normalization needed.
"""
from __future__ import annotations

import os

import numpy as np
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
    so AMP/autocast contexts in the caller are harmless.
    """

    def __init__(self, onnx_path: str, use_gpu: bool | None = None):
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"emotion ONNX not found: {onnx_path}")
        if use_gpu is None:
            # CPU default: the CUDA EP miscomputes conv workspace shapes for
            # inputs shorter than the 3 s trace length (bogus ~17 GB alloc,
            # hard FAIL for L<48000). CPU is correct for all tested lengths.
            use_gpu = False
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
        """Waveform ``[L]`` or ``[B, L]`` -> probs ``[B, 7]`` (float32).

        The traced graph bakes batch=1 (a reshape captures the sample
        batch dim), so B>1 runs row-by-row and stacks. Chunk counts per
        inference window are small, so this costs nothing measurable.
        """
        device = a.device
        if a.ndim == 1:
            a = a.unsqueeze(0)
        rows = [
            self._run(a[i : i + 1].detach().to("cpu", dtype=torch.float32).numpy())
            for i in range(a.shape[0])
        ]
        probs = np.concatenate(rows, axis=0)
        return torch.from_numpy(probs).to(device=device, dtype=torch.float32)
