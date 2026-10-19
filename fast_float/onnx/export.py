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

import contextlib
import os
from collections.abc import Sequence

import torch


def _as_tuple(inputs):
    if isinstance(inputs, torch.Tensor):
        return (inputs,)
    return tuple(inputs)


def _dynamo_mark(args, dynamic_dims=(0, 1), skip=()):
    """Mark dims dynamic for torch.export.

    Uses ``maybe_mark_dynamic`` (Dim-dicts misbehave under
    ``assume_static_by_default`` in some builds). Size-1 dims get
    specialized by dynamo, so the trace sample must already have ``>1``
    on every dim we want dynamic — callers pass batch>=2 samples.
    ``skip`` holds positional indices of inputs that are NOT batch-first
    (e.g. the FMT scalar time ``t``), which must stay static.
    """
    try:
        from torch._dynamo import maybe_mark_dynamic
    except Exception:
        return False
    marked = False
    for i, a in enumerate(args):
        if i in skip or not isinstance(a, torch.Tensor):
            continue
        for d in dynamic_dims:
            if d < a.ndim and a.shape[d] > 1:
                maybe_mark_dynamic(a, d)
                marked = True
    return marked


def _ensure_batch2(args, skip=()):
    """Duplicate a batch-1 sample so dynamo sees batch=2 (shapes only).

    ``skip`` holds positional indices that must not be touched: duplicating a
    *shared* input (the FMT scalar time ``t``, which is deliberately shape
    ``[1]`` and broadcast inside the graph) would both falsify its shape and
    get dim 0 marked dynamic, which TensorRT then mis-resolves across the
    rank-1/rank-3 inputs.
    """
    out = []
    for i, a in enumerate(args):
        if i not in skip and isinstance(a, torch.Tensor) and a.ndim >= 1 and a.shape[0] == 1:
            a = torch.cat([a, a], dim=0)
        out.append(a)
    return tuple(out)


def export_module(
    module: torch.nn.Module,
    sample_inputs,
    out_path: str,
    input_names: Sequence[str],
    output_names: Sequence[str],
    dynamic_axes: dict[str, dict[int, str]] | None = None,
    opset: int = 17,
    fixed_batch: bool = False,
    mark_dims: tuple[int, ...] = (0, 1),
    shared_inputs: Sequence[str] = (),
) -> dict[str, str]:
    """Export ``module`` to ``out_path`` via dynamo. Returns info dict.

    ``dynamic_axes``/``opset`` are accepted for API compatibility; dynamism
    comes from ``mark_dims`` and dynamo's own opset selection.
    ``fixed_batch=True`` skips sample duplication + dynamic marking (for
    graphs with baked batch, e.g. static-batch grouped conv).
    ``shared_inputs`` names inputs that are broadcast across the batch rather
    than batch-first (the FMT scalar time ``t``): they keep their traced shape
    and no dim of theirs is marked dynamic, so TensorRT never has to reconcile
    a rank-1 shared dim against the rank-3 batch dims.
    Raises RuntimeError when dynamo export is unavailable/fails.
    """
    module = module.eval()
    inputs = _as_tuple(sample_inputs)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    notes = []
    marked = False
    if not (hasattr(torch, "export") and hasattr(torch.export, "export")):
        raise RuntimeError(
            "torch.export.export is required (torch>=2.1) and the classic "
            f"torch.onnx.export fallback is retired. This box has torch "
            f"{torch.__version__}. Run conversion on the GPU box (torch>=2.1)."
        )
    skip = {input_names.index(n) for n in shared_inputs if n in input_names}
    if fixed_batch:
        trace_args = inputs
        notes.append("fixed batch: no dynamic marking")
    else:
        trace_args = _ensure_batch2(inputs, skip=skip)
        marked = _dynamo_mark(trace_args, dynamic_dims=mark_dims, skip=skip)
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
    # Report the dynamics honestly: a fixed-batch graph (the decoder, whose
    # grouped convs bake ``groups=batch``) has no symbolic axes at all, while
    # other units may mark only some dims (e.g. ``audio`` bakes length and
    # marks batch only). Claiming "symbolic batch/length" unconditionally was
    # wrong for both.
    if fixed_batch:
        notes.append("dynamo export: static batch/length (baked at trace)")
    elif marked:
        notes.append(f"dynamo export with symbolic dims={mark_dims}")
    else:
        notes.append("dynamo export: static shapes (no dims marked dynamic)")
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
        lo, hi = list(d), list(d)
        lo[0], hi[0] = min_batch, max_batch
        d_opt = list(d)
        d_opt[0] = max(min_batch, min(d_opt[0], max_batch))
        for dim, (a, b) in (extra or {}).get(name, {}).items():
            lo[dim], hi[dim] = int(a), int(b)
            d_opt[dim] = max(lo[dim], min(d_opt[dim], hi[dim]))
        mn[name], mx[name] = lo, hi
        op[name] = d_opt
    return mn, op, mx


def _trt_libs_loadable() -> bool:
    """True only when ORT's TensorRT EP shared library can actually load.

    ``ort.get_available_providers()`` advertises ``TensorrtExecutionProvider``
    even when ``libnvinfer`` is absent (it is compiled into onnxruntime-gpu).
    ORT then fails provider init and — crucially — silently degrades the
    *entire* session to ``CPUExecutionProvider``, discarding the CUDA EP too.
    Loading the library up front avoids that and the EP-error log spew.

    The honest test is loading ORT's own provider library: that resolves the
    exact ``libnvinfer`` version ORT was built against (plus the cuDNN/CUDA
    sonames on ``LD_LIBRARY_PATH``). A pip ``tensorrt`` whose major disagrees
    with that version (e.g. ``libnvinfer.so.11`` vs ORT's expected ``.10``)
    imports fine yet is unusable, so a bare ``import tensorrt`` is not enough.
    """
    import ctypes
    import sys

    # 1. Best effort: the python tensorrt module registers its lib dirs so the
    #    provider library can resolve them (mandatory on Windows).
    with contextlib.suppress(Exception):
        import tensorrt  # noqa: F401

    # 2. Definitive: load ORT's TensorRT provider library itself.
    try:
        import onnxruntime

        capi = os.path.join(os.path.dirname(onnxruntime.__file__), "capi")
        provider_name = (
            "onnxruntime_providers_tensorrt.dll" if sys.platform == "win32" else "libonnxruntime_providers_tensorrt.so"
        )
        provider_path = os.path.join(capi, provider_name)
        if os.path.exists(provider_path):
            ctypes.CDLL(provider_path)
            return True
    except Exception:
        return False

    # 3. Fallback: soname probe for installs without a bundled provider file.
    candidates = (
        ("nvinfer.dll", "nvinfer_10.dll", "nvinfer_9.dll", "nvinfer_8.dll")
        if sys.platform == "win32"
        else (
            "libnvinfer.so",
            "libnvinfer.so.10",
            "libnvinfer.so.9",
            "libnvinfer.so.8",
        )
    )
    for soname in candidates:
        try:
            ctypes.CDLL(soname)
            return True
        except OSError:
            continue
    return False


def _trt_provider_options(
    fp16,
    cache_dir,
    min_shapes,
    opt_shapes,
    max_shapes,
    workspace_gb=2.0,
    device_id: int = 0,
):
    """OR-TensorRT EP provider options (FP16, on-disk engine+timing cache,
    shape profiles). TRT engine build is minutes-long, so the cache avoids
    rebuilding per process; engines are tied to GPU/TRT/CUDA versions."""
    opts = {
        "device_id": int(device_id),
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
        opts["trt_timing_cache_path"] = os.path.join(cache_abs, "timing.cache")
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
    trt_cache_dir: str | None = None,
    trt_min_shapes: dict[str, Sequence[int]] | None = None,
    trt_opt_shapes: dict[str, Sequence[int]] | None = None,
    trt_max_shapes: dict[str, Sequence[int]] | None = None,
    device_id: int | None = None,
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
    dev_id = device_id if device_id is not None else (torch.cuda.current_device() if torch.cuda.is_available() else 0)
    gpu_providers = []
    if use_gpu:
        if used_trt:
            gpu_providers.append(
                (
                    "TensorrtExecutionProvider",
                    _trt_provider_options(
                        trt_fp16,
                        trt_cache_dir,
                        trt_min_shapes,
                        trt_opt_shapes,
                        trt_max_shapes,
                        device_id=dev_id,
                    ),
                )
            )
        if "CUDAExecutionProvider" in available:
            gpu_providers.append(
                (
                    "CUDAExecutionProvider",
                    {"device_id": dev_id, "cudnn_conv_algo_search": "DEFAULT"},
                )
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
            providers=[*list(provs), "CPUExecutionProvider"],
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
            raise RuntimeError(f"GPU requested but the ORT session fell back to CPU (available={sorted(available)}).")
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
    ap.add_argument(
        "--cpu-ort",
        action="store_true",
        help="allow a CPU ORT session on a CUDA box (dev only)",
    )
    ap.add_argument(
        "--no-trt",
        action="store_true",
        help="disable the TensorRT EP (use the CUDA EP instead)",
    )
    ap.add_argument("--trt-fp32", action="store_true", help="TensorRT in FP32 (default FP16)")
    ap.add_argument(
        "--trt-cache",
        default=None,
        help="TensorRT engine cache dir (default <onnx_dir>/trt_cache)",
    )
    ap.add_argument("--trt-min-batch", type=int, default=1, help="TensorRT profile min batch")
    ap.add_argument(
        "--trt-max-batch",
        type=int,
        default=default_max_batch,
        help="TensorRT profile max batch",
    )


def make_session_from_args(onnx_path: str, args, opt_shapes=None, extra_dynamic=None):
    """Build an ORT session from shared CLI flags (chain TRT -> CUDA -> CPU).

    ``opt_shapes`` maps each *dynamic* input to a representative shape so the
    TensorRT EP can build shape profiles; omit it for fully-static graphs (the
    decoder). Returns ``(session, fp16_active)`` where ``fp16_active`` tells
    callers to relax parity tolerances -- it is True only when the TensorRT
    FP16 EP is *actually* active (a requested-but-unloadable TRT must not relax
    the CUDA/CPU parity gate).
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
        onnx_path,
        use_gpu=use_gpu,
        require_gpu=use_gpu,
        use_trt=use_trt,
        trt_fp16=fp16,
        trt_cache_dir=getattr(args, "trt_cache", None),
        trt_min_shapes=mn,
        trt_opt_shapes=op,
        trt_max_shapes=mx,
    )
    return sess, (fp16 and "TensorrtExecutionProvider" in sess.get_providers())


def make_runtime_session(
    onnx_path: str,
    opt_shapes: dict[str, Sequence[int]] | None = None,
    extra_dynamic: dict[str, dict[int, tuple[int, int]]] | None = None,
    min_batch: int = 1,
    max_batch: int | None = None,
    use_gpu: bool | None = None,
    fp16: bool = True,
    device_id: int | None = None,
):
    """ORT session for a *runtime* component with automatic TensorRT profiles.

    Runtime ONNX backends get the same ``TRT -> CUDA -> CPU`` chain the
    converters use, so TensorRT engages automatically when its libraries load.
    TensorRT can only build an engine for a *dynamic-shape* graph once it is
    handed shape profiles, so ``opt_shapes`` maps every dynamic input to a
    representative shape: batch (dim 0) is ranged over ``[min_batch, max_batch]``
    while each other dim is frozen to its representative value unless ranged via
    ``extra_dynamic={name: {dim: (lo, hi)}}``. Static graphs (the decoder) pass
    ``None`` and need no profiles.

    ``max_batch`` defaults to ``$FLOAT_TRT_MAX_BATCH`` (or 16). Returns
    ``(session, fp16_active)``; ``fp16_active`` is True only when the TensorRT
    EP is actually on. Never raises for GPU setup -- the CUDA/CPU fallback lives
    inside :func:`make_ort_session`.
    """
    if use_gpu is None:
        use_gpu = torch.cuda.is_available()
    if max_batch is None:
        max_batch = int(os.environ.get("FLOAT_TRT_MAX_BATCH", "16"))
    mn = op = mx = None
    if use_gpu and opt_shapes:
        mn, op, mx = trt_optimization_profiles(
            opt_shapes,
            min_batch=min_batch,
            max_batch=max_batch,
            extra=extra_dynamic,
        )
    sess = make_ort_session(
        onnx_path,
        use_gpu=use_gpu,
        use_trt=use_gpu,
        trt_fp16=fp16,
        trt_min_shapes=mn,
        trt_opt_shapes=op,
        trt_max_shapes=mx,
        device_id=device_id,
    )
    return sess, ("TensorrtExecutionProvider" in sess.get_providers())


def fp16_tol(base: float, fp16: bool, factor: float = 40.0) -> float:
    """Relax a parity tolerance when TensorRT runs in FP16 (tensor cores
    trade ~1e-3 relative precision for speed; FP32 keeps the strict gate).

    The factor covers units whose outputs are *large-magnitude* (the relative
    term then absorbs the drift). Units whose outputs are bounded or
    small-magnitude cannot be fixed by any factor -- their FP16 drift is
    absolute, so they pass the additional functional gate below instead.
    """
    return base * factor if fp16 else base


# Functional-equivalence gate for the TRT FP16 path (see
# :func:`fast_float.onnx.parity.compare_outputs`). Measured worst case on this
# box (Tesla T4, TRT 10.9, torch 2.7.1+cu126) for the two units whose
# elementwise gate FP16 cannot meet:
#   emotion  (probs in [0, 1])  cosine 0.9985  mean_abs 8.9e-3  max_abs 3.1e-2
#   refenc   (conv feature maps, |b| up to ~39)  cosine 0.99999  mean_abs 3.1e-3
# The classifier's argmax held on 13/13 draws and refenc's maps keep their
# structure, so cosine + bounded mean error is the honest test; the
# elementwise gate stays active as the primary check. 0.995 matches the
# threshold the decoder has used since the first TRT run.
FP16_MIN_COSINE = 0.995
FP16_MAX_MEAN = 0.02


def run_ort_iobinding(session, feed, out_shapes, device=None):
    """Run an ORT session with CUDA IOBinding: zero host copies.

    ``feed`` maps input name -> CUDA tensor (any dtype; cast to fp32 on
    device, matching the exported graphs). ``out_shapes`` is a list of
    ``(output_name, shape)`` with fully concrete ints; outputs are allocated
    as CUDA fp32 tensors and returned on ``device``. Raises on any failure --
    callers must fall back to the numpy path.
    """
    import numpy as np
    import onnxruntime as ort  # noqa: F401 (ensures ORT present)

    if not feed:
        raise ValueError("iobinding needs at least one input")
    first = next(iter(feed.values()))
    dev = device if device is not None else first.device
    if dev.type != "cuda":
        raise RuntimeError("iobinding requires CUDA tensors")
    did = dev.index if dev.index is not None else 0
    refs = []
    binding = session.io_binding()
    for name, t in feed.items():
        if t.device.type != "cuda":
            raise RuntimeError(f"iobinding input {name!r} is not a CUDA tensor")
        tc = t.to(dtype=torch.float32).contiguous()
        refs.append(tc)
        binding.bind_input(name, "cuda", did, np.float32, tuple(tc.shape), tc.data_ptr())
    outs = []
    for name, shape in out_shapes:
        shape = tuple(int(d) for d in shape)
        o = torch.empty(shape, device=dev, dtype=torch.float32)
        binding.bind_output(name, "cuda", did, np.float32, shape, o.data_ptr())
        outs.append(o)
    session.run_with_iobinding(binding)
    return outs
