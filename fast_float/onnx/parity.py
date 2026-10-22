"""torch-vs-ONNX parity checks used by every converter."""

from __future__ import annotations

import numpy as np

# Deployed-frame gate for the decoder: at 512x512, 2 LSB max / 0.5 LSB mean is
# invisible on screen, while a structurally wrong graph (the TRT engine measured
# max_abs=4.375, cosine=0.983) is not. The exporter uses both floors (it
# compares the graph against the model in a common precision, so the residual
# is the graph's own error); the runtime install self-test uses the mean floor
# plus a cosine floor, because there the *provider* is in the loop and TF32
# alone moves this 22-conv stack by ~40 LSB between two implementations.
U8_MAX_LSB = 2
U8_MAX_MEAN_LSB = 0.5

# Provider-level floor for the decoder: the *provider*, not the graph, sets
# the precision here. The ORT CUDA EP runs fp32 matmuls/convs on TF32
# tensor cores, and through the decoder's 22-conv stack that alone moves the
# frames by ~0.5 uint8 LSB on average (max ~50 LSB on a few pixels) -- so the
# runtime install self-test cannot use the 0.5 LSB floor: it would reject an
# artifact the exporter already proved exact. 2 LSB mean (0.8% of the range) is
# still invisible on a 512x512 frame, while a structurally wrong graph (the
# broken TensorRT engine: max_abs 4.375) measures ~23 LSB mean and a cosine of
# 0.983, so both floors still reject it.
U8_MAX_PROVIDER_MEAN_LSB = 2.0


def u8_lsb(ref, out) -> tuple[int, float]:
    """Quantized-frame error: what the decode path actually writes.

    ``_quantize_u8_hwc`` clamps to [-1, 1], scales by 127.5 and truncates, so a
    sub-LSB float drift is invisible while a 4.4 drift is a broken face. Gating
    here is meaningful at every precision; a float-space ``atol`` (40x for TRT
    FP16) accepts any bounded output, which is how a decoder graph that was off
    by 4.375 passed as "max_abs=4.375 cosine=0.98".
    """

    def q(x):
        import torch

        arr = np.asarray(x.detach().cpu() if torch.is_tensor(x) else x, dtype=np.float32)
        return np.clip((arr + 1.0) * 127.5, 0.0, 255.0).astype(np.uint8)

    d = np.abs(q(ref).astype(np.int16) - q(out).astype(np.int16))
    return int(d.max()), float(d.mean())


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

    ``min_cosine``/``max_mean`` are *hard floors*, not alternatives to the
    elementwise gate: an output passes only when every floor that was supplied
    holds. They let the TensorRT FP16 path pass on the functional branch when
    the elementwise gate is too strict for absolute rounding drift (probability
    vectors, conv feature maps), but they can never be skipped by a loose
    ``atol``. That is exactly how a decoder graph off by 4.375 with cosine
    0.9829 was reported as "PASS (atol=40)": the old code ORed the branches and
    the cosine floor was never evaluated.
    """
    a, b = _to_numpy(a), _to_numpy(b)
    if a.shape != b.shape:
        return {
            "ok": False,
            "elementwise_ok": False,
            "cosine_ok": False,
            "mean_ok": False,
            "reason": f"shape mismatch torch={a.shape} onnx={b.shape}",
            "max_abs": float("inf"),
            "mean_abs": float("inf"),
            "max_rel": float("inf"),
            "cosine": 0.0,
        }
    abs_diff = np.abs(a - b)
    denom = np.maximum(np.abs(b), 1e-12)
    na, nb = np.linalg.norm(a.ravel()), np.linalg.norm(b.ravel())
    cosine = float(np.dot(a.ravel(), b.ravel()) / (na * nb + 1e-12))
    mean_abs = float(abs_diff.mean())
    elementwise_ok = bool(np.allclose(a, b, atol=atol, rtol=rtol))
    cosine_ok = min_cosine is None or cosine >= min_cosine
    mean_ok = max_mean is None or mean_abs <= max_mean
    has_floor = min_cosine is not None or max_mean is not None
    functional_ok = has_floor and cosine_ok and mean_ok
    return {
        "ok": bool((elementwise_ok or functional_ok) and cosine_ok and mean_ok),
        "elementwise_ok": elementwise_ok,
        "cosine_ok": bool(cosine_ok),
        "mean_ok": bool(mean_ok),
        "max_abs": float(abs_diff.max()),
        "mean_abs": mean_abs,
        "max_rel": float((abs_diff / denom).max()),
        "cosine": cosine,
    }


def is_significant(
    candidate_ms: float,
    candidate_mad: float,
    baseline_ms: float,
    baseline_mad: float,
    min_gain: float = 1.10,
    spread_factor: float = 3.0,
) -> bool:
    """Is ``candidate_ms`` a real win over ``baseline_ms``?

    A speedup must clear two bars: the configured ``min_gain`` *and* more than
    ``spread_factor`` times the run-to-run spread (median absolute deviation)
    of both measurements. Without the second bar a 1.05x "win" inside a 2-3%
    jitter band reselects a backend every run -- the refenc case, where
    ``torch.compile`` was picked at 1.05x and paid a guard/recompile storm for
    nothing.
    """
    if candidate_ms <= 0 or baseline_ms <= 0:
        return False
    if candidate_ms * min_gain >= baseline_ms:
        return False
    return (baseline_ms - candidate_ms) > spread_factor * max(candidate_mad, baseline_mad)


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
