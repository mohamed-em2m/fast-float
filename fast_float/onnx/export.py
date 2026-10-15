"""Shared torch -> ONNX export helpers.

Strategy: try the dynamo exporter first (best dynamic-shape support on modern
torch, e.g. Kaggle images), fall back to the classic exporter with explicit
``dynamic_axes`` (works back to torch 2.0, which is what this repo pins).
"""
from __future__ import annotations

import os
from typing import Dict, Optional, Sequence

import torch


def _as_tuple(inputs):
    if isinstance(inputs, torch.Tensor):
        return (inputs,)
    return tuple(inputs)


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

    # 1) Dynamo exporter (torch >= 2.1 recommended; harmless to attempt).
    try:
        from torch.onnx import dynamo_export

        try:
            from torch.onnx import DynamoExportOptions

            program = dynamo_export(
                module, *inputs, export_options=DynamoExportOptions(dynamic_shapes=True)
            )
        except Exception:
            program = dynamo_export(module, *inputs)
        program.save(out_path)
        return {
            "path": os.path.abspath(out_path),
            "exporter": "dynamo",
            "notes": notes,
        }
    except Exception as e:  # noqa: BLE001 - fallback path below
        notes.append(f"dynamo exporter failed ({e}); falling back to torchscript")

    # 2) Classic exporter fallback with explicit dynamic axes.
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
        )
    return {
        "path": os.path.abspath(out_path),
        "exporter": "torchscript",
        "notes": notes,
    }


def make_ort_session(onnx_path: str, use_gpu: bool = True):
    """ORT InferenceSession, CUDA first, CPU fallback. Raises if file missing."""
    import onnxruntime as ort

    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")
    available = set(ort.get_available_providers())
    providers = []
    if use_gpu:
        for cand in ("CUDAExecutionProvider", "TensorrtExecutionProvider"):
            if cand in available:
                providers.append(cand)
    providers.append("CPUExecutionProvider")
    return ort.InferenceSession(onnx_path, providers=providers)
