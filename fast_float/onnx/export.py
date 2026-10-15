"""Shared torch -> ONNX export helpers.

Strategy: dynamo export via torch.export (symbolic batch/length), no
torchscript fallback. ``torch.export`` + ``torch.onnx.export(ExportedProgram)``
is the modern flow: it decomposes SDPA/LayerNorm/diag_embed natively and
marks dynamic dims symbolically. Converters keep their export-safe
substitutions (manual attention, plain norm, matmul reformulation) — all
dynamo-safe and equivalence-proven — plus static-batch modconv freezing,
which stays because ONNX Conv ``group`` is static regardless of exporter.
"""
from __future__ import annotations

import os
from typing import Dict, Optional, Sequence, Tuple

import torch


def _as_tuple(inputs):
    if isinstance(inputs, torch.Tensor):
        return (inputs,)
    return tuple(inputs)


def _dynamo_mark(args, dynamic_dims=(0, 1)):
    """Mark dims dynamic for torch.export.

    Uses ``maybe_mark_dynamic`` (Dim-dicts misbehave under
    ``assume_static_by_default`` in some builds). Size-1 dims get
    specialized by dynamo, so the trace sample must already have ``>1``
    on every dim we want dynamic — callers pass batch>=2 samples.
    """
    try:
        from torch._dynamo import maybe_mark_dynamic
    except Exception:
        return False
    marked = False
    for a in args:
        if isinstance(a, torch.Tensor):
            for d in dynamic_dims:
                if d < a.ndim and a.shape[d] > 1:
                    maybe_mark_dynamic(a, d)
                    marked = True
    return marked


def _ensure_batch2(args):
    """Duplicate a batch-1 sample so dynamo sees batch=2 (shapes only)."""
    out = []
    for a in args:
        if isinstance(a, torch.Tensor) and a.ndim >= 1 and a.shape[0] == 1:
            a = torch.cat([a, a], dim=0)
        out.append(a)
    return tuple(out)


def export_module(
    module: torch.nn.Module,
    sample_inputs,
    out_path: str,
    input_names: Sequence[str],
    output_names: Sequence[str],
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
    opset: int = 17,
    fixed_batch: bool = False,
    mark_dims: Tuple[int, ...] = (0, 1),
) -> Dict[str, str]:
    """Export ``module`` to ``out_path`` via dynamo. Returns info dict.

    ``dynamic_axes``/``opset`` are accepted for API compatibility; dynamism
    comes from ``mark_dims`` and dynamo's own opset selection.
    ``fixed_batch=True`` skips sample duplication + dynamic marking (for
    graphs with baked batch, e.g. static-batch grouped conv).
    Raises RuntimeError when dynamo export is unavailable/fails.
    """
    module = module.eval()
    inputs = _as_tuple(sample_inputs)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    notes = []
    dynamo_err = None
    try:
        if fixed_batch:
            trace_args = inputs
            notes.append("fixed batch: no dynamic marking")
        else:
            trace_args = _ensure_batch2(inputs)
            marked = _dynamo_mark(trace_args, dynamic_dims=mark_dims)
            notes.append(f"dynamo dynamic dims={mark_dims} marked={marked}")
        if hasattr(torch, "export") and hasattr(torch.export, "export"):
            ep = torch.export.export(module, trace_args, strict=False)
            torch.onnx.export(
                ep,
                f=out_path,
                input_names=list(input_names),
                output_names=list(output_names),
                # external_data=False: keeps all weights inline so ORT
                # InferenceSession can load without extra path hints
                # (external_data=True writes a .data sidecar that ORT
                # older than 1.24 cannot locate via a plain path string,
                # causing the cls.__new__ crash on Kaggle / cp310 boxes).
                external_data=False,
            )
            notes.append("dynamo export with symbolic batch/length")
            return {
                "path": os.path.abspath(out_path),
                "exporter": "dynamo",
                "notes": notes,
            }
        else:
            raise NotImplementedError("torch.export not available")
    except Exception as e:
        dynamo_err = e
        notes.append(f"dynamo export bypassed ({e}); using classic torch.onnx.export")

    # Robust fallback: classic torch.onnx.export (compatible across all torch 2.x versions)
    try:
        with torch.no_grad():
            torch.onnx.export(
                module,
                inputs if len(inputs) > 1 else inputs[0],
                out_path,
                input_names=list(input_names),
                output_names=list(output_names),
                dynamic_axes=dynamic_axes or {},
                opset_version=opset,
                do_constant_folding=True,
            )
        notes.append("classic torch.onnx.export successful")
        return {
            "path": os.path.abspath(out_path),
            "exporter": "classic",
            "notes": notes,
        }
    except Exception as fallback_e:
        raise RuntimeError(
            f"Both dynamo export ({dynamo_err}) and classic export ({fallback_e}) failed."
        ) from fallback_e


def make_ort_session(
    onnx_path: str, use_gpu: bool = True, require_gpu: bool = False
):
    """ORT InferenceSession, CUDA first, CPU fallback. Raises if file missing.

    CUDA uses ``cudnn_conv_algo_search=DEFAULT``: with dynamic-shape convs
    (e.g. waveform inputs of varying length) the default EXHAUSTIVE search
    can pick a workspace-hungry algorithm and hard-FAIL allocation
    (observed: bogus ~17 GB request for <3 s inputs). DEFAULT stays correct
    for all lengths at negligible speed cost for auxiliary classifiers.

    With ``require_gpu=True`` (use on CUDA boxes) a missing CUDA/TRT
    provider raises instead of silently benchmarking CPU.
    """
    import onnxruntime as ort

    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")
    available = set(ort.get_available_providers())
    gpu_providers = []
    if use_gpu:
        if "CUDAExecutionProvider" in available:
            gpu_providers.append(
                ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"})
            )
        elif "TensorrtExecutionProvider" in available:
            gpu_providers.append("TensorrtExecutionProvider")
    if use_gpu and require_gpu and not gpu_providers:
        raise RuntimeError(
            "GPU requested but no CUDA execution provider available "
            f"(available={sorted(available)}). Install onnxruntime-gpu "
            "matching this box's CUDA instead of plain onnxruntime."
        )
    # SessionOptions: always use abs-path so ORT resolves any external-data
    # sibling files correctly regardless of CWD.
    abs_path = os.path.abspath(onnx_path)
    sess_opts = ort.SessionOptions()
    sess = None
    if gpu_providers:
        try:
            sess = ort.InferenceSession(
                abs_path,
                sess_options=sess_opts,
                providers=gpu_providers + ["CPUExecutionProvider"],
            )
        except Exception as e:
            if require_gpu:
                raise
            print(f"[ort] WARNING: GPU provider session creation failed ({e}); falling back to CPU")
    if sess is None:
        sess = ort.InferenceSession(
            abs_path,
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )
    active = sess.get_providers()
    if use_gpu and (not active or active[0] == "CPUExecutionProvider"):
        print(
            f"[ort] WARNING: session runs on CPU (active={active}, "
            f"available={sorted(available)}). Expect torch-CUDA to win."
        )
    return sess
