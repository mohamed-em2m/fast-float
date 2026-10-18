"""torch-vs-ONNX parity checks used by every converter."""

from __future__ import annotations

import numpy as np


def _to_numpy(x) -> np.ndarray:
    import torch

    if isinstance(x, torch.Tensor):
        return x.detach().cpu().float().numpy()
    return np.asarray(x, dtype=np.float32)


def compare_outputs(
    a,
    b,
    atol: float = 1e-4,
    rtol: float = 1e-4,
    min_cosine: float | None = None,
    max_mean: float | None = None,
) -> dict[str, float]:
    """Compare two outputs. Returns metrics dict with an ``ok`` flag.

    ``min_cosine``/``max_mean`` add a *functional-equivalence* branch for the
    TensorRT FP16 path: an output that misses the elementwise gate still passes
    when it is the same map (``cosine >= min_cosine`` and ``mean_abs <=
    max_mean``). TRT FP16 rounds every accumulation in a deep stack, so on
    bounded outputs (probability vectors, conv feature maps) a tight
    elementwise gate measures FP16 rounding, not export correctness. The
    decoder has relied on this branch since the first TRT run; pass ``None``
    (the default) to keep the strict elementwise gate only.
    """
    a, b = _to_numpy(a), _to_numpy(b)
    if a.shape != b.shape:
        return {
            "ok": False,
            "elementwise_ok": False,
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
    mean_abs = float(abs_diff.mean())
    functional = min_cosine is not None and max_mean is not None and cosine >= min_cosine and mean_abs <= max_mean
    return {
        "ok": bool(ok or functional),
        "elementwise_ok": ok,
        "max_abs": float(abs_diff.max()),
        "mean_abs": mean_abs,
        "max_rel": float((abs_diff / denom).max()),
        "cosine": cosine,
    }


def check_parity(
    torch_out,
    ort_out,
    name: str = "output",
    atol: float = 1e-4,
    rtol: float = 1e-4,
    min_cosine: float | None = None,
    max_mean: float | None = None,
) -> dict[str, float]:
    """Compare and print one named output pair. Raises on mismatch."""
    rep = compare_outputs(torch_out, ort_out, atol=atol, rtol=rtol, min_cosine=min_cosine, max_mean=max_mean)
    status = "PASS" if rep["ok"] else "FAIL"
    if rep["ok"] and not rep.get("elementwise_ok", True):
        status = "PASS(functional)"
    print(
        f"[parity:{name}] {status} "
        f"max_abs={rep['max_abs']:.3e} mean_abs={rep['mean_abs']:.3e} "
        f"max_rel={rep['max_rel']:.3e} cosine={rep['cosine']:.6f} "
        f"(atol={atol:g}, rtol={rtol:g})"
    )
    if not rep["ok"]:
        raise RuntimeError(f"parity FAILED for {name}: {rep}")
    return rep


def benchmark(torch_fn, ort_fn, name: str = "run", iters: int = 50, warmup: int = 10) -> dict[str, float]:
    """Warmed median-latency shootout: torch vs ORT. Returns report dict.

    Session-creation cost is excluded by construction (session already
    built); callables must run exactly one inference each. Prints speedup
    (torch_med / ort_med); <1.0 means torch wins and the unit stays torch.
    """
    import time

    for _ in range(max(0, warmup)):
        torch_fn()
        ort_fn()

    def _med(fn) -> float:
        ts = []
        for _ in range(max(1, iters)):
            t0 = time.perf_counter()
            fn()
            ts.append((time.perf_counter() - t0) * 1000.0)
        ts.sort()
        return ts[len(ts) // 2]

    torch_med, ort_med = _med(torch_fn), _med(ort_fn)
    speedup = torch_med / max(ort_med, 1e-9)
    print(
        f"[bench:{name}] torch_med={torch_med:.2f}ms ort_med={ort_med:.2f}ms "
        f"speedup={speedup:.2f}x (iters={iters}, warmup={warmup})"
    )
    return {"torch_med_ms": torch_med, "ort_med_ms": ort_med, "speedup": speedup}
