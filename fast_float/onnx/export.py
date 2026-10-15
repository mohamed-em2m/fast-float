"""Shared torch -> ONNX export helpers.

Strategy: dynamo export first (torch.export + onnx, symbolic batch/length),
fall back to the classic torchscript exporter (works back to torch 2.0,
which is what this repo pins).
"""
from __future__ import annotations

import os
from typing import Dict, Optional, Sequence

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
) -> Dict[str, str]:
    """Export ``module`` to ``out_path``. Returns info dict (path, exporter)."""
    module = module.eval()
    inputs = _as_tuple(sample_inputs)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    notes = []

    # 1) Dynamo exporter (symbolic batch/length via torch.export).
    try:
        trace_args = _ensure_batch2(inputs)
        _dynamo_mark(trace_args)
        ep = torch.export.export(module, trace_args, strict=False)
        torch.onnx.export(
            ep,
            f=out_path,
            input_names=list(input_names),
            output_names=list(output_names),
            external_data=True,
        )
        notes.append("dynamo export with symbolic batch/length")
        return {
            "path": os.path.abspath(out_path),
            "exporter": "dynamo",
            "notes": notes,
        }
    except Exception as e:  # noqa: BLE001 - fallback path below
        notes.append(f"dynamo exporter failed ({e}); falling back to torchscript")

    # 2) Classic torchscript exporter fallback with explicit dynamic axes.
    with torch.no_grad():
        torch.onnx.export(
            module,
            inputs,
            out_path,
            input_names=list(input_names),
            output_names=list(output_names),
            dynamic_axes=dynamic_axes or {},
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )
    return {
        "path": os.path.abspath(out_path),
        "exporter": "torchscript",
        "notes": notes,
    }


def make_ort_session(onnx_path: str, use_gpu: bool = True):
    """ORT InferenceSession, CUDA first, CPU fallback. Raises if file missing.

    CUDA uses ``cudnn_conv_algo_search=DEFAULT``: with dynamic-shape convs
    (e.g. waveform inputs of varying length) the default EXHAUSTIVE search
    can pick a workspace-hungry algorithm and hard-FAIL allocation
    (observed: bogus ~17 GB request for <3 s inputs). DEFAULT stays correct
    for all lengths at negligible speed cost for auxiliary classifiers.
    """
    import onnxruntime as ort

    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")
    available = set(ort.get_available_providers())
    providers = []
    if use_gpu and "CUDAExecutionProvider" in available:
        providers.append(("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}))
    providers.append("CPUExecutionProvider")
    return ort.InferenceSession(onnx_path, providers=providers)
