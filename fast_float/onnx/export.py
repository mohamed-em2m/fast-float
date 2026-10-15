"""Shared torch -> ONNX export helpers.

Strategy: dynamo exporter only (torch >= 2.1, e.g. Kaggle images). Dynamo
decomposes modern ops (SDPA, LayerNorm variants, diag_embed) into portable
ONNX natively and has first-class dynamic-shape support. The classic
torchscript exporter is retired: it cannot translate SDPA/diag_embed and
rejects valid graphs at the checker. Converters keep their export-safe
substitutions (manual attention, plain norm, matmul reformulation) — all
dynamo-safe and equivalence-proven — except static-batch modconv freezing,
which stays because ONNX Conv ``group`` is static regardless of exporter.
"""
from __future__ import annotations

import os
from typing import Dict, Optional, Sequence

import torch

MIN_TORCH_DYNAMO = (2, 1)


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
    """Export ``module`` to ``out_path`` via dynamo. Returns info dict.

    ``dynamic_axes``/``opset`` are accepted for API compatibility but the
    dynamo exporter manages dynamic shapes (dynamic_shapes=True) and its
    own opset selection. Raises RuntimeError on torch < 2.1.
    """
    try:
        from torch.onnx import dynamo_export
    except ImportError as e:
        raise RuntimeError(
            f"ONNX export requires torch>={'.'.join(map(str, MIN_TORCH_DYNAMO))} "
            "for the dynamo exporter (installed torch has no "
            f"torch.onnx.dynamo_export): {e}. Upgrade torch on the conversion "
            "box; the inference venv is unaffected."
        ) from e
    module = module.eval()
    inputs = _as_tuple(sample_inputs)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

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
        "notes": [f"input_names={list(input_names)}",
                  f"output_names={list(output_names)}"],
    }


def make_ort_session(
    onnx_path: str, use_gpu: bool = True, require_gpu: bool = False
):
    """ORT InferenceSession, CUDA first, CPU fallback.

    With ``require_gpu=True`` (use on CUDA boxes) a missing CUDA/TRT
    provider raises instead of silently benchmarking CPU — a silent CPU
    session is the #1 cause of "ONNX is slow" reports.
    """
    import onnxruntime as ort

    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")
    available = set(ort.get_available_providers())
    gpu_providers = []
    if use_gpu:
        for cand in ("CUDAExecutionProvider", "TensorrtExecutionProvider"):
            if cand in available:
                gpu_providers.append(cand)
    if use_gpu and require_gpu and not gpu_providers:
        raise RuntimeError(
            "GPU requested but no CUDA/TRT execution provider available "
            f"(available={sorted(available)}). Install onnxruntime-gpu "
            "matching this box's CUDA instead of plain onnxruntime."
        )
    sess = ort.InferenceSession(onnx_path, providers=gpu_providers + ["CPUExecutionProvider"])
    active = sess.get_providers()
    if use_gpu and (not active or active[0] == "CPUExecutionProvider"):
        print(
            f"[ort] WARNING: session runs on CPU (active={active}, "
            f"available={sorted(available)}). Expect torch-CUDA to win."
        )
    return sess
