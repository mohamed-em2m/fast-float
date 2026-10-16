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

from fast_float.onnx.export import make_ort_session, run_ort_iobinding

LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


def _gpu_active(session) -> bool:
    try:
        provs = session.get_providers()
    except Exception:
        return False
    return bool(provs) and provs[0] != "CPUExecutionProvider"


def _dynamo_disable(fn):
    """No-op decorator when torch._dynamo is unavailable."""
    try:
        from torch._dynamo import disable

        return disable(fn)
    except Exception:
        return fn


def _vec_out_len(x: torch.Tensor, prev_x) -> int:
    """Sequence length of one FMT step's output.

    The graph concatenates ``prev_x``/``prev_wa`` onto ``x``/``wa`` before the
    transformer (see ``FlowMatchingTransformer.forward``), so the vector field
    is ``x.shape[1] + prev_x.shape[1]`` long — not ``x.shape[1]``. Binding the
    IOBinding output at ``x.shape[1]`` alone under-sizes the buffer.
    """
    n = int(x.shape[1])
    if prev_x is not None and torch.is_tensor(prev_x):
        n += int(prev_x.shape[1])
    return n


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
        out = self.session.get_outputs()[0]
        self.output_name = out.name
        shape = tuple(out.shape or ())
        self.output_dim = shape[-1] if shape and isinstance(shape[-1], int) else None
        self._io_failed = False

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
        if (
            device.type == "cuda"
            and not self._io_failed
            and self.output_dim is not None
            and _gpu_active(self.session)
        ):
            try:
                (out,) = run_ort_iobinding(
                    self.session,
                    {self.input_name: a.to(dtype=torch.float32)},
                    [(self.output_name, (a.shape[0], int(self.output_dim)))],
                    device=device,
                )
                return out.to(device=device, dtype=torch.float32)
            except Exception as e:  # noqa: BLE001
                self._io_failed = True
                print(f"[backends:iobinding] disabled ({e}); using host path")
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
        out = self.session.get_outputs()[0]
        self.output_name = out.name
        shape = tuple(out.shape or ())
        self._static_out_dim = (
            shape[-1] if shape and isinstance(shape[-1], int) else None
        )
        self._io_failed = False

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
        if (
            device.type == "cuda"
            and not self._io_failed
            and self._static_out_dim is not None
            and _gpu_active(self.session)
        ):
            try:
                (out,) = run_ort_iobinding(
                    self.session,
                    {self.input_name: conv_feats.to(dtype=torch.float32)},
                    [
                        (
                            self.output_name,
                            (
                                conv_feats.shape[0],
                                conv_feats.shape[1],
                                int(self._static_out_dim),
                            ),
                        )
                    ],
                    device=device,
                )
                return out.to(device=device, dtype=torch.float32)
            except Exception as e:  # noqa: BLE001
                self._io_failed = True
                print(f"[backends:iobinding] disabled ({e}); using host path")
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
        out = self.session.get_outputs()[0]
        self.output_name = out.name
        shape = tuple(out.shape or ())
        self._static_last = shape[-1] if shape and isinstance(shape[-1], int) else None
        self._io_failed = False

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
        """Single vector-field eval -> ``[B, L+Lp, 512]`` (input device).

        The graph concatenates ``prev_x``/``prev_wa`` onto ``x``/``wa`` before
        the transformer (see ``FlowMatchingTransformer.forward``), so the output
        length is ``x.shape[1] + prev_x.shape[1]`` — not ``x.shape[1]``. Bind the
        output at that length or the IOBinding buffer is too small.
        """
        device = x.device
        out_len = _vec_out_len(x, prev_x)
        if (
            device.type == "cuda"
            and not self._io_failed
            and self._static_last is not None
            and _gpu_active(self.session)
            and all(
                torch.is_tensor(v) and v.device.type == "cuda"
                for v in (t, x, wa, wr, we, prev_x, prev_wa)
            )
        ):
            try:
                (out,) = run_ort_iobinding(
                    self.session,
                    {
                        name: val.to(dtype=torch.float32)
                        for name, val in zip(
                            self.INPUTS, (t, x, wa, wr, we, prev_x, prev_wa)
                        )
                    },
                    [(self.output_name, (x.shape[0], out_len, int(self._static_last)))],
                    device=device,
                )
                return out.to(device=device, dtype=torch.float32)
            except Exception as e:  # noqa: BLE001
                self._io_failed = True
                print(f"[backends:iobinding] disabled ({e}); using host path")
        feed = {
            name: val.detach().to("cpu", dtype=torch.float32).numpy()
            for name, val in zip(self.INPUTS, (t, x, wa, wr, we, prev_x, prev_wa))
        }
        out = self._run(feed)
        return torch.from_numpy(out).to(device=device, dtype=torch.float32)
