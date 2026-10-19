"""ONNX runtime backends for the constructor-wired components.

``emotion``/``audio``/``fmt`` self-wire inside their constructors and use these
sessions (``fast_float.onnx.runtime`` re-exports them for compatibility).
Moved here from ``fast_float/onnx/runtime.py`` (Phase 2 of
``docs/BUILD_RUNTIME_SESSIONS.md``), so both ONNX runner families live under one
``RuntimeSession`` umbrella.
"""

from __future__ import annotations

import os

import torch

from fast_float.onnx.export import make_runtime_session, run_ort_iobinding
from fast_float.runtimes.base_runtime import RuntimeSession
from fast_float.runtimes.onnx.runner import _dynamo_disable

LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


def _gpu_active(session) -> bool:
    try:
        provs = session.get_providers()
    except Exception:
        return False
    return bool(provs) and provs[0] != "CPUExecutionProvider"


def _trt_active(session) -> bool:
    """True when the session is served by the TensorRT EP.

    Only TRT bakes rigid optimization-profile bounds (min/max on the dynamic
    dims); the CUDA and CPU EPs accept any length. Callers that clamp an input
    to a fixed window must do so *only* under TRT, or they silently change the
    result relative to the eager path on the other EPs.
    """
    try:
        return "TensorrtExecutionProvider" in session.get_providers()
    except Exception:
        return False


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


class EmotionOnnxBackend(RuntimeSession):
    """Thin wrapper around an ORT session for emotion probabilities.

    Keeps torch semantics for callers: torch in (any device/dtype) ->
    float32 probs on the input's device. Always runs fp32 on CPU numpy,
    so AMP/autocast contexts in the caller are harmless. The graph has
    symbolic batch/length, so one ``sess.run`` covers any ``[B, L]``.
    """

    component = "emotion"
    backend = "onnx"

    def __init__(
        self,
        onnx_path: str,
        use_gpu: bool | None = None,
        opt_shapes: dict | None = None,
        extra_dynamic: dict | None = None,
        max_batch: int | None = None,
        device_id: int | None = None,
    ):
        super().__init__(device_id=device_id)
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"emotion ONNX not found: {onnx_path}")
        self.path = os.path.abspath(onnx_path)
        self.session, _ = make_runtime_session(
            self.path,
            opt_shapes=opt_shapes,
            extra_dynamic=extra_dynamic,
            max_batch=max_batch,
            use_gpu=use_gpu,
            device_id=device_id,
        )
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
        """Waveform ``[L]`` or ``[B, L]`` -> probs ``[B, 7]`` (float32).

        The waveform is passed through unchanged unless TensorRT is serving the
        session: the emotion graph is exported with symbolic batch/length, so
        the CUDA/CPU EPs accept the whole clip exactly like the eager path.
        Clamping to a fixed window under TRT is unavoidable (its optimization
        profile bounds length to ``[16000, 160000]``), but doing it on the other
        EPs made emotion depend on *which backend was active* -- the same clip
        read a different (and for long clips, a stale) emotion than eager.
        """
        device = a.device
        if a.ndim == 1:
            a = a.unsqueeze(0)
        # TensorRT only: keep the length inside the built profile. Anywhere else
        # the full waveform matches the eager SER read.
        if _trt_active(self.session):
            if a.shape[1] < 16000:
                a = torch.nn.functional.pad(a, (0, 16000 - a.shape[1]), mode="replicate")
            elif a.shape[1] > 160000:
                a = a[:, :160000]
        if device.type == "cuda" and not self._io_failed and self.output_dim is not None and _gpu_active(self.session):
            try:
                (out,) = run_ort_iobinding(
                    self.session,
                    {self.input_name: a.to(dtype=torch.float32)},
                    [(self.output_name, (a.shape[0], int(self.output_dim)))],
                    device=device,
                )
                self.stats.io_bindings += 1
                return out.to(device=device, dtype=torch.float32)
            except Exception as e:
                self._io_failed = True
                print(f"[backends:iobinding] disabled ({e}); using host path")
        wav_np = a.detach().to("cpu", dtype=torch.float32).numpy()
        probs = self._run(wav_np)
        return torch.from_numpy(probs).to(device=device, dtype=torch.float32)

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        return self.predict_probs(a)


class Wav2vecOnnxBackend(RuntimeSession):
    """Thin wrapper around an ORT session for FLOAT audio features.

    Covers feature-projection + the 12-layer transformer + the trained
    ``audio_projection`` head (``onnx/wav2vec_v1.onnx``)::

        conv_feats [B, T, 512] -> audio feats [B, T, dim_w] (float32)

    The caller (``AudioEncoder.inference``) produces ``conv_feats`` in
    torch via ``Wav2VecModel.feature_extract`` (conv front-end +
    interpolation). Inference-only: gradients cannot flow through ORT,
    so training paths must keep using the torch module.
    """

    component = "audio"
    backend = "onnx"

    def __init__(
        self,
        onnx_path: str,
        use_gpu: bool | None = None,
        opt_shapes: dict | None = None,
        extra_dynamic: dict | None = None,
        max_batch: int | None = None,
        device_id: int | None = None,
    ):
        super().__init__(device_id=device_id)
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"wav2vec ONNX not found: {onnx_path}")
        self.path = os.path.abspath(onnx_path)
        self.session, _ = make_runtime_session(
            self.path,
            opt_shapes=opt_shapes,
            extra_dynamic=extra_dynamic,
            max_batch=max_batch,
            use_gpu=use_gpu,
            device_id=device_id,
        )
        self.input_name = self.session.get_inputs()[0].name
        out = self.session.get_outputs()[0]
        self.output_name = out.name
        shape = tuple(out.shape or ())
        self._static_out_dim = shape[-1] if shape and isinstance(shape[-1], int) else None
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
                self.stats.io_bindings += 1
                return out.to(device=device, dtype=torch.float32)
            except Exception as e:
                self._io_failed = True
                print(f"[backends:iobinding] disabled ({e}); using host path")
        feats_np = conv_feats.detach().to("cpu", dtype=torch.float32).numpy()
        out = self._run(feats_np)
        return torch.from_numpy(out).to(device=device, dtype=torch.float32)

    def forward(self, conv_feats: torch.Tensor) -> torch.Tensor:
        return self.run_features(conv_feats)


class FMTStepBackend(RuntimeSession):
    """Thin wrapper around an ORT session for one FMT vector-field step.

    Covers ``FlowMatchingTransformer.forward(train=False)``::

        (t[1], x[B,50,512], wa[B,50,512], wr[B,1,512], we[B,1,7],
         prev_x[B,10,512], prev_wa[B,10,512]) -> vec [B,60,512]

    Batch is dynamic: the same graph serves plain (B) and CFG (3B) calls.
    The caller slices ``vec[:, 10:]`` for the current chunk.
    """

    component = "fmt"
    backend = "onnx"

    INPUTS = ("t", "x", "wa", "wr", "we", "prev_x", "prev_wa")

    def __init__(
        self,
        onnx_path: str,
        use_gpu: bool | None = None,
        opt_shapes: dict | None = None,
        extra_dynamic: dict | None = None,
        max_batch: int | None = None,
        device_id: int | None = None,
    ):
        super().__init__(device_id=device_id, degrade_on_error="sticky")
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"fmt ONNX not found: {onnx_path}")
        self.path = os.path.abspath(onnx_path)
        self.session, _ = make_runtime_session(
            self.path,
            opt_shapes=opt_shapes,
            extra_dynamic=extra_dynamic,
            max_batch=max_batch,
            use_gpu=use_gpu,
            device_id=device_id,
        )
        out = self.session.get_outputs()[0]
        self.output_name = out.name
        shape = tuple(out.shape or ())
        self._static_last = shape[-1] if shape and isinstance(shape[-1], int) else None
        self._io_failed = False
        # TensorRT refuses any batch above the optimization profile's max with
        # "Set dimension [24,50,512] for tensor x does not satisfy any
        # optimization profiles" -- and the runtime used to disable this backend
        # for the rest of the process on that first failure. The graph has no
        # cross-batch ops, so serving a larger call as several profile-sized
        # calls is numerically identical. Batched sampling needs it: CFG sends
        # 3 * chunk_batch_size samples, e.g. 24 for 8 chunks with a profile
        # capped at 16.
        self._batch_cap = None
        if "TensorrtExecutionProvider" in self.session.get_providers():
            cap = max_batch if max_batch is not None else int(os.environ.get("FLOAT_TRT_MAX_BATCH", "16"))
            self._batch_cap = int(cap) if int(cap) > 0 else None
        self._split_warned = False
        self._failures = 0

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

        Batches larger than the TensorRT profile are split: the graph has no
        cross-batch ops, so ``cat(run(chunk))`` is identical to one big call.
        """
        cap = self._batch_cap
        batch = int(x.shape[0])
        if cap is not None and batch > cap:
            if not self._split_warned:
                self._split_warned = True
                print(
                    f"[backends:fmt] batch {batch} exceeds the TensorRT profile max "
                    f"({cap}); splitting into {cap}-sized calls (numerically identical). "
                    f"Raise FLOAT_TRT_MAX_BATCH or lower chunk_batch_size to avoid the split."
                )
            parts = []
            for i in range(0, batch, cap):
                j = min(i + cap, batch)
                sl = slice(i, j)
                parts.append(
                    self.run_step(
                        # ``t`` is exported static ([1]) and broadcast, but if a
                        # caller ever hands us a batch-sized ``t`` it must follow
                        # the split like every other input or the parts disagree.
                        t[sl] if (torch.is_tensor(t) and t.ndim > 0 and t.shape[0] == batch) else t,
                        x[sl],
                        wa[sl] if wa.shape[0] == batch else wa,
                        wr[sl] if wr.shape[0] == batch else wr,
                        we[sl] if we.shape[0] == batch else we,
                        prev_x[sl] if prev_x.shape[0] == batch else prev_x,
                        prev_wa[sl] if prev_wa.shape[0] == batch else prev_wa,
                    )
                )
            return torch.cat(parts, dim=0)
        return self._run_once(t, x, wa, wr, we, prev_x, prev_wa)

    def _run_once(self, t, x, wa, wr, we, prev_x, prev_wa) -> torch.Tensor:
        """One ORT call, guaranteed to fit the profile (see ``run_step``)."""
        device = x.device
        out_len = _vec_out_len(x, prev_x)
        if (
            device.type == "cuda"
            and not self._io_failed
            and self._static_last is not None
            and _gpu_active(self.session)
            and all(torch.is_tensor(v) and v.device.type == "cuda" for v in (t, x, wa, wr, we, prev_x, prev_wa))
        ):
            try:
                (out,) = run_ort_iobinding(
                    self.session,
                    {
                        name: val.to(dtype=torch.float32)
                        for name, val in zip(self.INPUTS, (t, x, wa, wr, we, prev_x, prev_wa), strict=False)
                    },
                    [(self.output_name, (x.shape[0], out_len, int(self._static_last)))],
                    device=device,
                )
                self.stats.io_bindings += 1
                return out.to(device=device, dtype=torch.float32)
            except Exception as e:
                self._io_failed = True
                print(f"[backends:iobinding] disabled ({e}); using host path")
        feed = {
            name: val.detach().to("cpu", dtype=torch.float32).numpy()
            for name, val in zip(self.INPUTS, (t, x, wa, wr, we, prev_x, prev_wa), strict=False)
        }
        out = self._run(feed)
        return torch.from_numpy(out).to(device=device, dtype=torch.float32)

    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        return self.run_step(*args)


__all__ = ["LABELS", "EmotionOnnxBackend", "FMTStepBackend", "Wav2vecOnnxBackend"]
