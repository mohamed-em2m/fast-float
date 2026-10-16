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
    if not (hasattr(torch, "export") and hasattr(torch.export, "export")):
        raise RuntimeError(
            "torch.export.export is required (torch>=2.1) and the classic "
            f"torch.onnx.export fallback is retired. This box has torch "
            f"{torch.__version__}. Run conversion on the GPU box (torch>=2.1)."
        )
    if fixed_batch:
        trace_args = inputs
        notes.append("fixed batch: no dynamic marking")
    else:
        trace_args = _ensure_batch2(inputs)
        marked = _dynamo_mark(trace_args, dynamic_dims=mark_dims)
        notes.append(f"dynamo dynamic dims={mark_dims} marked={marked}")
    try:
        ep = torch.export.export(module, trace_args, strict=False)
        torch.onnx.export(
            ep,
            f=out_path,
            input_names=list(input_names),
            output_names=list(output_names),
            # external_data=False: keeps all weights inline so ORT
            # InferenceSession can load without extra path hints
            # (external_data=True writes a .data sidecar that ORT older
            # than 1.24 cannot locate via a plain path string).
            external_data=False,
        )
    except Exception as e:
        raise RuntimeError(
            "dynamo ONNX export failed and the classic torch.onnx.export "
            "fallback is retired (it cannot translate SDPA/diag_embed, and "
            f"torch>=2.9 rejects its dynamic_axes): {e}. Fix the graph (see "
            "the converter substitution notes) instead of falling back."
        ) from e
    notes.append("dynamo export with symbolic batch/length")
    return {
        "path": os.path.abspath(out_path),
        "exporter": "dynamo",
        "notes": notes,
    }


def _parse_shape(v):
    if isinstance(v, str):
        return [int(x) for x in v.replace(",", "x").split("x")]
    return [int(x) for x in v]


def trt_optimization_profiles(opt_shapes, min_batch=1, max_batch=None, extra=None):
    """Build ``(min, opt, max)`` input-shape dicts for the TensorRT EP.

    ``opt_shapes`` maps each *dynamic* input to a representative shape; dim 0
    (batch) is ranged over ``[min_batch, max_batch]`` while every other dim is
    frozen to the representative value. Static inputs must be omitted (a TRT
    profile dim that disagrees with a static dim is rejected). Per-input
    non-batch dims can be ranged via ``extra={name: {dim: (lo, hi)}}``.
    Returns ``(None, None, None)`` when ``opt_shapes`` is empty.
    """
    if not opt_shapes:
        return None, None, None
    if max_batch is None:
        max_batch = max(_parse_shape(s)[0] for s in opt_shapes.values())
    mn, op, mx = {}, {}, {}
    for name, shape in opt_shapes.items():
        d = _parse_shape(shape)
        op[name] = d
        lo, hi = list(d), list(d)
        lo[0], hi[0] = min_batch, max_batch
        for dim, (a, b) in (extra or {}).get(name, {}).items():
            lo[dim], hi[dim] = int(a), int(b)
        mn[name], mx[name] = lo, hi
    return mn, op, mx


def _trt_libs_loadable() -> bool:
    """True only when ORT's TensorRT EP shared library can actually load.

    ``ort.get_available_providers()`` advertises ``TensorrtExecutionProvider``
    even when ``libnvinfer`` is absent (it is compiled into onnxruntime-gpu).
    ORT then fails provider init and — crucially — silently degrades the
    *entire* session to ``CPUExecutionProvider``, discarding the CUDA EP too.
    Loading the library up front avoids that and the EP-error log spew.
    """
    import ctypes

    for soname in (
        "libnvinfer.so",
        "libnvinfer.so.10",
        "libnvinfer.so.9",
        "libnvinfer.so.8",
    ):
        try:
            ctypes.CDLL(soname)
            return True
        except OSError:
            continue
    return False


def _trt_provider_options(
    fp16, cache_dir, min_shapes, opt_shapes, max_shapes, workspace_gb=2.0
):
    """OR-TensorRT EP provider options (FP16, on-disk engine+timing cache,
    shape profiles). TRT engine build is minutes-long, so the cache avoids
    rebuilding per process; engines are tied to GPU/TRT/CUDA versions."""
    opts = {
        "trt_fp16_enable": bool(fp16),
        "trt_builder_optimization_level": 3,
        "trt_max_workspace_size": int(workspace_gb * (1 << 30)),
    }
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_abs = os.path.abspath(cache_dir)
        opts["trt_engine_cache_enable"] = True
        opts["trt_engine_cache_path"] = cache_abs
        opts["trt_timing_cache_enable"] = True
        opts["trt_timing_cache_path"] = cache_abs
    if min_shapes and opt_shapes and max_shapes:
        def _fmt(d):
            return ",".join(f"{k}:{'x'.join(map(str, v))}" for k, v in d.items())

        opts["trt_profile_min_shapes"] = _fmt(min_shapes)
        opts["trt_profile_opt_shapes"] = _fmt(opt_shapes)
        opts["trt_profile_max_shapes"] = _fmt(max_shapes)
    return opts


def make_ort_session(
    onnx_path: str,
    use_gpu: bool = True,
    require_gpu: bool = False,
    use_trt: bool = True,
    trt_fp16: bool = True,
    trt_cache_dir: Optional[str] = None,
    trt_min_shapes: Optional[Dict[str, Sequence[int]]] = None,
    trt_opt_shapes: Optional[Dict[str, Sequence[int]]] = None,
    trt_max_shapes: Optional[Dict[str, Sequence[int]]] = None,
):
    """ORT InferenceSession, provider chain TRT -> CUDA -> CPU.

    TensorRT EP is inserted first (when ``use_trt`` and available) so its
    conv kernels replace ORT-CUDA's slow "fallback mode" path; anything TRT
    can't take falls through to CUDA EP automatically. ``trt_fp16`` uses T4
    tensor cores. ``trt_cache_dir`` (default ``<onnx_dir>/trt_cache``) caches
    built engines. Dynamic-shape graphs need ``trt_{min,opt,max}_shapes``
    (see :func:`trt_optimization_profiles`) or TRT cannot build an engine.

    CUDA uses ``cudnn_conv_algo_search=DEFAULT``: with dynamic-shape convs
    the default EXHAUSTIVE search can pick a workspace-hungry algorithm and
    hard-FAIL allocation (bogus ~17 GB request observed). DEFAULT stays
    correct for all lengths at negligible speed cost.

    With ``require_gpu=True`` a missing CUDA/TRT provider raises instead of
    silently benchmarking CPU.
    """
    import onnxruntime as ort

    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")
    abs_path = os.path.abspath(onnx_path)
    available = set(ort.get_available_providers())
    if trt_cache_dir is None:
        trt_cache_dir = os.path.join(os.path.dirname(abs_path), "trt_cache")
    trt_listed = "TensorrtExecutionProvider" in available
    used_trt = use_trt and trt_listed and _trt_libs_loadable()
    if use_gpu and use_trt and trt_listed and not used_trt:
        print(
            "[ort] NOTE: TensorRT EP is listed but libnvinfer is not loadable "
            "(pip install tensorrt, or set LD_LIBRARY_PATH) — using the CUDA EP."
        )
    gpu_providers = []
    if use_gpu:
        if used_trt:
            gpu_providers.append(
                (
                    "TensorrtExecutionProvider",
                    _trt_provider_options(
                        trt_fp16, trt_cache_dir,
                        trt_min_shapes, trt_opt_shapes, trt_max_shapes,
                    ),
                )
            )
        if "CUDAExecutionProvider" in available:
            gpu_providers.append(
                ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"})
            )
    if use_gpu and require_gpu and not gpu_providers:
        raise RuntimeError(
            "GPU requested but no CUDA/TRT execution provider available "
            f"(available={sorted(available)}). Install onnxruntime-gpu "
            "matching this box's CUDA instead of plain onnxruntime."
        )
    if use_gpu and use_trt and not trt_listed:
        print(
            "[ort] NOTE: TensorRT EP not available "
            f"(available={sorted(available)}); using the CUDA EP. Install "
            "tensorrt / libnvinfer for TRT."
        )

    def _new_session(provs):
        return ort.InferenceSession(
            abs_path,
            sess_options=ort.SessionOptions(),
            providers=list(provs) + ["CPUExecutionProvider"],
        )

    def _on_gpu(s):
        active = s.get_providers()
        return bool(active) and active[0] != "CPUExecutionProvider"

    # Candidate GPU chains, most-preferred first. A TensorRT build without
    # libnvinfer (or without matching shape profiles) makes ORT silently drop
    # ALL GPU providers — so we always keep a CUDA-only retry.
    attempts = []
    if gpu_providers:
        attempts.append(gpu_providers)
    cuda_only = [p for p in gpu_providers if p[0] == "CUDAExecutionProvider"]
    if used_trt and cuda_only:
        attempts.append(cuda_only)

    sess = None
    for provs in attempts:
        try:
            candidate = _new_session(provs)
        except Exception as e:
            print(f"[ort] WARNING: GPU provider session creation raised ({e})")
            continue
        if _on_gpu(candidate):
            sess = candidate
            break
        # Session came back CPU-only: ORT swallowed a provider-load failure.
        requested_trt = any(p[0] == "TensorrtExecutionProvider" for p in provs)
        if requested_trt:
            print("[ort] NOTE: TensorRT EP failed to initialize — retrying with the CUDA EP.")
        else:
            print("[ort] WARNING: GPU execution providers did not initialize.")
    if sess is None:
        if use_gpu and require_gpu:
            raise RuntimeError(
                "GPU requested but the ORT session fell back to CPU "
                f"(available={sorted(available)})."
            )
        sess = _new_session([])
    active = sess.get_providers()
    trt_active = "TensorrtExecutionProvider" in active
    trt_label = ("fp16" if trt_fp16 else "fp32") if trt_active else "off"
    print(f"[ort] providers={active} (trt={trt_label})")
    if use_gpu and (not active or active[0] == "CPUExecutionProvider"):
        print(
            f"[ort] WARNING: session runs on CPU (active={active}, "
            f"available={sorted(available)}). Expect torch-CUDA to win."
        )
    return sess


def add_ort_args(ap, default_max_batch: int = 8):
    """Register the shared ORT/TensorRT CLI flags on an argparse parser."""
    ap.add_argument("--cpu-ort", action="store_true",
                    help="allow a CPU ORT session on a CUDA box (dev only)")
    ap.add_argument("--no-trt", action="store_true",
                    help="disable the TensorRT EP (use the CUDA EP instead)")
    ap.add_argument("--trt-fp32", action="store_true",
                    help="TensorRT in FP32 (default FP16)")
    ap.add_argument("--trt-cache", default=None,
                    help="TensorRT engine cache dir (default <onnx_dir>/trt_cache)")
    ap.add_argument("--trt-min-batch", type=int, default=1,
                    help="TensorRT profile min batch")
    ap.add_argument("--trt-max-batch", type=int, default=default_max_batch,
                    help="TensorRT profile max batch")


def make_session_from_args(
    onnx_path: str, args, opt_shapes=None, extra_dynamic=None
):
    """Build an ORT session from shared CLI flags (chain TRT -> CUDA -> CPU).

    ``opt_shapes`` maps each *dynamic* input to a representative shape so the
    TensorRT EP can build shape profiles; omit it for fully-static graphs (the
    decoder). Returns ``(session, fp16_active)`` where ``fp16_active`` tells
    callers to relax parity tolerances.
    """
    use_gpu = torch.cuda.is_available() and not getattr(args, "cpu_ort", False)
    use_trt = use_gpu and not getattr(args, "no_trt", False)
    fp16 = use_trt and not getattr(args, "trt_fp32", False)
    mn = op = mx = None
    if use_trt and opt_shapes:
        mn, op, mx = trt_optimization_profiles(
            opt_shapes,
            min_batch=getattr(args, "trt_min_batch", 1),
            max_batch=getattr(args, "trt_max_batch", None),
            extra=extra_dynamic,
        )
    sess = make_ort_session(
        onnx_path, use_gpu=use_gpu, require_gpu=use_gpu,
        use_trt=use_trt, trt_fp16=fp16,
        trt_cache_dir=getattr(args, "trt_cache", None),
        trt_min_shapes=mn, trt_opt_shapes=op, trt_max_shapes=mx,
    )
    return sess, fp16


def fp16_tol(base: float, fp16: bool, factor: float = 40.0) -> float:
    """Relax a parity tolerance when TensorRT runs in FP16 (tensor cores
    trade ~1e-3 relative precision for speed; FP32 keeps the strict gate)."""
    return base * factor if fp16 else base
