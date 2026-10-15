"""torch-vs-ONNX parity checks used by every converter."""
from __future__ import annotations

from typing import Dict

import numpy as np


def _to_numpy(x) -> np.ndarray:
    import torch

    if isinstance(x, torch.Tensor):
        return x.detach().cpu().float().numpy()
    return np.asarray(x, dtype=np.float32)


def compare_outputs(a, b, atol: float = 1e-4, rtol: float = 1e-4) -> Dict[str, float]:
    """Compare two outputs. Returns metrics dict with an ``ok`` flag."""
    a, b = _to_numpy(a), _to_numpy(b)
    if a.shape != b.shape:
        return {
            "ok": False,
            "reason": f"shape mismatch torch={a.shape} onnx={b.shape}",
            "max_abs": float("inf"),
            "mean_abs": float("inf"),
            "max_rel": float("inf"),
            "cosine": 0.0,
        }
    abs_diff = np.abs(a - b)
    denom = np.maximum(np.abs(b), 1e-12)
    ok = bool(np.allclose(a, b, atol=atol, rtol=rtol))
    na, nb = np.linalg.norm(a.ravel()), np.linalg.norm(b.ravel())
    cosine = float(np.dot(a.ravel(), b.ravel()) / (na * nb + 1e-12))
    return {
        "ok": ok,
        "max_abs": float(abs_diff.max()),
        "mean_abs": float(abs_diff.mean()),
        "max_rel": float((abs_diff / denom).max()),
        "cosine": cosine,
    }


def check_parity(
    torch_out, ort_out, name: str = "output", atol: float = 1e-4, rtol: float = 1e-4
) -> Dict[str, float]:
    """Compare and print one named output pair. Raises on mismatch."""
    rep = compare_outputs(torch_out, ort_out, atol=atol, rtol=rtol)
    status = "PASS" if rep["ok"] else "FAIL"
    print(
        f"[parity:{name}] {status} "
        f"max_abs={rep['max_abs']:.3e} mean_abs={rep['mean_abs']:.3e} "
        f"max_rel={rep['max_rel']:.3e} cosine={rep['cosine']:.6f} "
        f"(atol={atol:g}, rtol={rtol:g})"
    )
    if not rep["ok"]:
        raise RuntimeError(f"parity FAILED for {name}: {rep}")
    return rep
