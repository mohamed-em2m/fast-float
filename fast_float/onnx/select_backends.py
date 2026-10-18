"""Per-component backend selector: measure eager vs torch.compile vs ONNX.

Runs on the target box *after* the converters have produced the ONNX
artifacts. For each component it builds the same torch module the converter
exported, then measures, on identical inputs at the deployed dtype:

  * ``eager``   -- the reference (and the fallback)
  * ``compile`` -- ``torch.compile`` (inductor) around the module
  * ``onnx``    -- the exported graph on the ORT session (TRT -> CUDA -> CPU)

Every accelerated candidate is parity-gated against eager, then the fastest
candidate that beats eager by ``--min-gain`` wins. The result is written to a
config YAML consumed at runtime by :mod:`fast_float.onnx.backends`.

Note on composition: isolated per-component medians are a *pre-filter*, not the
whole answer -- device round-trips and lost fusion mean a component that looks
faster alone can lose inside the pipeline. This tool also probes a whole-
``Generator`` compile as an advisory (``e2e`` block); a true end-to-end
objective is validated by the app's own bench. See ``docs/BACKEND_SELECTION.md``.

Usage (from repo root, on the GPU box):
    python -m fast_float.onnx.select_backends --onnx-dir ./onnx \
        --out ./onnx/backends.generated.yaml [--dry-run] [--hot-only]

``--dry-run`` only writes the safe all-eager config + an env fingerprint
(no model construction), so it also works on a CPU-only box.
"""

from __future__ import annotations

import argparse
import os
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from fast_float.builders.onnx import OnnxBuilder
from fast_float.builders.spec import BuildArtifact, BuildSpec
from fast_float.onnx.backends import (
    COMPONENTS,
    HOT_COMPONENTS,
    BackendConfig,
    ComponentConfig,
    env_fingerprint,
    save_backends,
)
from fast_float.onnx.export import fp16_tol
from fast_float.onnx.parity import compare_outputs

ONNX_FILES = {
    "emotion": "emotion_v1.onnx",
    "audio": "audio_enc_v1.onnx",
    "direction": "direction_v1.onnx",
    "fmt": "fmt_step_v1.onnx",
    "decoder": "decoder_v1.onnx",
    "refenc": "refenc_v1.onnx",
}
AE_PREFIX = "motion_autoencoder."


@dataclass
class Case:
    """One component's eager module, its real inputs, and its ORT feed."""

    name: str
    module: torch.nn.Module
    inputs: tuple
    feed: dict[str, np.ndarray]
    atol: float = 2e-4
    rtol: float = 2e-4
    opt_shapes: dict | None = None
    extra_dynamic: dict | None = None
    min_cosine: float | None = None
    max_mean: float | None = None

    def eager_outs(self) -> list:
        out = self.module(*self.inputs)
        return list(out) if isinstance(out, (tuple, list)) else [out]

    @property
    def input_names(self) -> tuple[str, ...]:
        """Graph input names, in module-input order (from the ORT feed)."""
        return tuple(self.feed)

    def runtime_outs(self, runtime) -> list:
        out = runtime(*self.inputs)
        return list(out) if isinstance(out, (tuple, list)) else [out]


# --------------------------------------------------------------------------- #
# Component builders (mirror each converter's deployed path; lazy imports so
# --dry-run stays importable without the model stack).
# --------------------------------------------------------------------------- #


def _build_emotion(args, device) -> Case:
    from fast_float.onnx.convert_emotion import build_emotion_module, probe_inputs

    module = build_emotion_module(args.emo_dir, args.ckpt).to(device)
    wav = probe_inputs(args.seconds)["sine"].to(device)
    return Case(
        "emotion",
        module,
        (wav,),
        {"input_values": wav.detach().cpu().numpy()},
        opt_shapes={"input_values": [1, int(wav.shape[1])]},
        extra_dynamic={"input_values": {1: (16000, 160000)}},
    )


def _build_audio(args, device) -> Case:
    from fast_float.onnx.convert_audio import build_audio_module

    module, _, _ = build_audio_module(
        args.ckpt,
        args.w2v_dir,
        args.audio_frames,
        512,
        False,
        device,
    )
    wav_len = int(args.audio_frames * 16000 / 25.0)
    g = torch.Generator().manual_seed(11)
    a = torch.randn(1, wav_len, generator=g).clamp(-1, 1).to(device)
    return Case(
        "audio",
        module,
        (a,),
        {"waveform": a.detach().cpu().numpy()},
        opt_shapes={"waveform": [1, wav_len]},
    )


def _build_direction(args, device) -> Case:
    from fast_float.onnx.convert_direction import build_direction_module

    module, _ = build_direction_module(args.ckpt, args.motion_dim, device)
    x = torch.randn(1, module.motion_dim, device=device)
    return Case(
        "direction",
        module,
        (x,),
        {"x": x.detach().cpu().numpy()},
        opt_shapes={"x": [1, module.motion_dim]},
    )


def _build_fmt(args, device) -> Case:
    from fast_float.onnx.convert_fmt import (
        CLIP_FRAMES,
        PREV_FRAMES,
        FMTStepWrapper,
        build_fmt,
        sample_inputs,
    )

    ns = SimpleNamespace(
        ckpt=args.ckpt,
        prev_frames=args.prev_frames,
        dim_h=args.fmt_dim_h,
        mlp_ratio=args.fmt_mlp_ratio,
        fmt_depth=args.fmt_depth,
        num_heads=args.fmt_heads,
    )
    fmt = build_fmt(ns).to(device)
    for blk in fmt.blocks:
        if hasattr(blk, "attn"):
            blk.attn.fused_attn = False
    module = FMTStepWrapper(fmt).eval().to(device)
    names = ["t", "x", "wa", "wr", "we", "prev_x", "prev_wa"]
    ins = sample_inputs(3, 0.5, 99, device)  # CFG 3B: the hot call
    feed = {k: v.detach().cpu().numpy() for k, v in zip(names, ins, strict=False)}
    opt = {
        "x": [1, CLIP_FRAMES, 512],
        "wa": [1, CLIP_FRAMES, 512],
        "wr": [1, 1, 512],
        "we": [1, 1, 7],
        "prev_x": [1, PREV_FRAMES, 512],
        "prev_wa": [1, PREV_FRAMES, 512],
    }
    return Case("fmt", module, tuple(ins), feed, opt_shapes=opt)


def _build_decoder(args, device) -> Case:
    from fast_float.onnx.convert_decoder import build_decoder_module

    module, _dec, feats1 = build_decoder_module(
        args.ckpt,
        args.image_size,
        1.0,
        args.dec_batch,
        device,
    )
    b = args.dec_batch
    with torch.no_grad():
        lat = torch.randn(b, 512, device=device)
        feats_b = [f.expand(b, *f.shape[1:]).contiguous() for f in feats1]
    feed = {"latent": lat.detach().cpu().numpy()}
    feed.update({f"feat{i}": f.detach().cpu().numpy() for i, f in enumerate(feats_b)})
    return Case(
        "decoder",
        module,
        (lat, *feats_b),
        feed,
        atol=1.0,
        rtol=1e-2,
        min_cosine=0.995,
        max_mean=0.05,
    )


def _build_refenc(args, device) -> Case:
    from fast_float.onnx.convert_refenc import build_refenc_module

    module, _enc, _n = build_refenc_module(
        args.ckpt,
        args.image_size,
        args.motion_dim,
        device,
    )
    img = torch.randn(1, 3, args.image_size, args.image_size, device=device)
    return Case(
        "refenc",
        module,
        (img,),
        {"img": img.detach().cpu().numpy()},
        opt_shapes={"img": [1, 3, args.image_size, args.image_size]},
    )


BUILDERS: dict[str, Callable] = {
    "emotion": _build_emotion,
    "audio": _build_audio,
    "direction": _build_direction,
    "fmt": _build_fmt,
    "decoder": _build_decoder,
    "refenc": _build_refenc,
}


# --------------------------------------------------------------------------- #
# Measurement helpers
# --------------------------------------------------------------------------- #


def _median_ms(fn, iters: int, warmup: int) -> float:
    for _ in range(max(0, warmup)):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    ts = []
    for _ in range(max(1, iters)):
        t0 = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000.0)
    ts.sort()
    return ts[len(ts) // 2]


def _provider_label(providers) -> str | None:
    try:
        p = providers[0]
    except Exception:
        return None
    if "Tensorrt" in p:
        return "trt"
    if "CUDA" in p:
        return "cuda"
    return "cpu"


def _precision(args) -> str:
    return "fp32" if getattr(args, "trt_fp32", False) else "fp16"


def _chain(args) -> tuple[str, ...]:
    """Provider chain implied by the selector's ORT flags."""
    if getattr(args, "cpu_ort", False):
        return ("cpu",)
    if getattr(args, "no_trt", False):
        return ("cuda", "cpu")
    return ("trt", "cuda", "cpu")


def _device_index(device) -> int | None:
    if getattr(device, "type", None) == "cuda":
        return device.index if device.index is not None else 0
    return None


def _parity(case: Case, runtime, fp16: bool) -> dict[str, Any]:
    atol, rtol = fp16_tol(case.atol, fp16), fp16_tol(case.rtol, fp16)
    ref, got = case.eager_outs(), case.runtime_outs(runtime)
    if len(ref) != len(got):
        return {"ok": False, "reason": f"outputs {len(got)} != {len(ref)}"}
    ok, worst = True, {"max_abs": 0.0, "mean_abs": 0.0, "cosine": 1.0}
    for r, o in zip(ref, got, strict=False):
        rep = compare_outputs(r, o, atol=atol, rtol=rtol)
        this_ok = rep["ok"]
        if case.min_cosine is not None and case.max_mean is not None:
            this_ok = this_ok or (rep["cosine"] >= case.min_cosine and rep["mean_abs"] <= case.max_mean)
        ok = ok and this_ok
        worst["max_abs"] = max(worst["max_abs"], rep["max_abs"])
        worst["mean_abs"] = max(worst["mean_abs"], rep["mean_abs"])
        worst["cosine"] = min(worst["cosine"], rep["cosine"])
    worst["ok"] = bool(ok)
    return worst


def _round(metrics: dict[str, Any]) -> dict[str, Any]:
    return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in metrics.items()}


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #


def _select_component(name, case, args, builder, device) -> tuple[ComponentConfig, float, float]:
    metrics: dict[str, Any] = {}
    eager_ms = _median_ms(lambda: case.module(*case.inputs), args.iters, args.warmup)
    metrics["eager_ms"] = eager_ms
    onnx_path = Path(args.onnx_dir) / ONNX_FILES[name]

    spec = BuildSpec(
        component=name,
        backend="onnx",
        input_names=case.input_names,
        opt_shapes=case.opt_shapes,
        extra_dynamic=case.extra_dynamic,
        precision=_precision(args),
        provider_chain=_chain(args),
        trt_min_batch=args.trt_min_batch,
        trt_max_batch=args.trt_max_batch,
        trt_cache_dir=getattr(args, "trt_cache", None),
        ckpt_path=args.ckpt,
        tolerance=case.atol,
    )

    best_mode, best_provider, best_ms = "eager", None, eager_ms

    if not args.no_compile and hasattr(torch, "compile"):
        try:
            compiled = torch.compile(case.module, mode=args.compile_mode)
            t0 = time.perf_counter()
            compiled(*case.inputs)
            compile_s = time.perf_counter() - t0
            cms = _median_ms(lambda: compiled(*case.inputs), args.iters, args.warmup)
            metrics["compile_ms"] = cms
            metrics["compile_s"] = round(compile_s, 3)
            if cms * args.min_gain < best_ms:
                best_mode, best_provider, best_ms = "compile", None, cms
        except Exception as e:
            metrics["compile_error"] = f"{type(e).__name__}: {e}"
            print(f"[select:{name}] compile FAILED: {e}")

    if not args.no_onnx and onnx_path.exists():
        try:
            runtime = builder.runtime(spec, device_id=_device_index(device))
            providers = runtime.providers
            fp16 = _precision(args) == "fp16" and any("Tensorrt" in p for p in providers)
            par = _parity(case, runtime, fp16)
            metrics["onnx_parity"] = _round(par)
            if par["ok"]:
                oms = _median_ms(lambda: runtime(*case.inputs), args.iters, args.warmup)
                provider = _provider_label(providers)
                metrics["onnx_ms"] = oms
                metrics["onnx_provider"] = provider
                metrics["onnx_providers"] = providers
                builder.record(
                    BuildArtifact(
                        component=name,
                        kind="onnx",
                        path=str(onnx_path),
                        provider=provider,
                        precision=_precision(args),
                        fingerprint=builder.fingerprint(spec),
                        metrics={"onnx_ms": round(oms, 6), "onnx_providers": providers},
                        parity=par,
                    )
                )
                if oms * args.min_gain < best_ms:
                    best_mode, best_provider, best_ms = "onnx", provider, oms
            else:
                metrics["onnx_reason"] = "parity failed"
        except Exception as e:
            metrics["onnx_error"] = f"{type(e).__name__}: {e}"
            print(f"[select:{name}] onnx FAILED: {e}")

    speedup = eager_ms / max(best_ms, 1e-9)
    reason = None
    if best_mode == "eager":
        reason = f"no candidate beat eager by {args.min_gain:.2f}x"
    label = f"{best_mode}/{best_provider}" if best_provider else best_mode
    print(f"[select:{name}] eager={eager_ms:.2f}ms -> {label} ({best_ms:.2f}ms, {speedup:.2f}x)")

    cfg = ComponentConfig(
        name=name,
        mode=best_mode,
        provider=best_provider,
        onnx_path=(str(onnx_path) if onnx_path.exists() else None),
        metrics=metrics,
        reason=reason,
    )
    return cfg, best_ms, eager_ms


def _build_generator(args, device):
    from fast_float.models.float.generator import Generator

    gen = Generator(size=args.image_size, style_dim=512, motion_dim=args.motion_dim).to(device).eval()
    if os.path.exists(args.ckpt):
        sd = torch.load(args.ckpt, map_location="cpu", weights_only=True)
        sub = {k[len(AE_PREFIX) :]: v for k, v in sd.items() if k.startswith(AE_PREFIX)}
        if sub:
            gen.load_state_dict(sub, strict=False)
    img = torch.randn(1, 3, args.image_size, args.image_size, device=device)
    return gen, (img, img)


def _select_global(args, device) -> dict[str, Any] | None:
    """Advisory whole-Generator compile vs eager (enc+dec fused)."""
    if args.no_compile or args.no_global or not hasattr(torch, "compile"):
        return None
    try:
        gen, inputs = _build_generator(args, device)
        with torch.no_grad():
            eager_ms = _median_ms(lambda: gen(*inputs), args.iters, args.warmup)
            compiled = torch.compile(gen, mode=args.compile_mode)
            t0 = time.perf_counter()
            compiled(*inputs)
            compile_s = time.perf_counter() - t0
            cms = _median_ms(lambda: compiled(*inputs), args.iters, args.warmup)
        return {
            "kind": "global_generator",
            "eager_ms": eager_ms,
            "compile_ms": cms,
            "compile_s": round(compile_s, 3),
            "speedup": eager_ms / max(cms, 1e-9),
        }
    except Exception as e:
        print(f"[select:global] global compile FAILED: {e}")
        return {"kind": "global_generator", "error": f"{type(e).__name__}: {e}"}


def _assemble(components, args, fingerprint, glob) -> BackendConfig:
    chosen = {n: c.mode for n, c in components.items() if c.mode != "eager"}
    return BackendConfig(
        version=2,
        source=str(Path(args.out)),
        target=fingerprint,
        objective="end_to_end_ms",
        default_mode="eager",
        chosen_combo=chosen,
        e2e=(glob or {}),
        components=components,
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Select per-component inference backends (writes a config YAML)")
    ap.add_argument("--onnx-dir", default="./onnx")
    ap.add_argument("--out", default="./onnx/backends.generated.yaml")
    ap.add_argument("--ckpt", default="./checkpoints/float.pth")
    ap.add_argument("--emo-dir", default="./checkpoints/wav2vec-english-speech-emotion-recognition")
    ap.add_argument("--w2v-dir", default="./checkpoints/wav2vec2-base-960h")
    ap.add_argument("--components", default=",".join(COMPONENTS))
    ap.add_argument(
        "--hot-only",
        action="store_true",
        help="only measure the hot components (fmt, decoder, audio)",
    )
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument(
        "--min-gain",
        type=float,
        default=1.03,
        help="candidate must beat eager by this factor to be chosen",
    )
    ap.add_argument("--compile-mode", default="default")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--no-onnx", action="store_true")
    ap.add_argument("--no-global", action="store_true")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="write the safe all-eager config + fingerprint only",
    )
    # component shape params (must match the deployed config)
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--motion-dim", type=int, default=20)
    ap.add_argument("--dec-batch", type=int, default=8)
    ap.add_argument("--audio-frames", type=int, default=300)
    ap.add_argument("--prev-frames", type=int, default=10)
    ap.add_argument("--fmt-depth", type=int, default=8)
    ap.add_argument("--fmt-dim-h", type=int, default=1024)
    ap.add_argument("--fmt-heads", type=int, default=8)
    ap.add_argument("--fmt-mlp-ratio", type=float, default=4.0)
    # ORT / TensorRT flags (same names as the converters so ${ORT_ARGS} forwards)
    ap.add_argument("--cpu-ort", action="store_true")
    ap.add_argument("--no-trt", action="store_true")
    ap.add_argument("--trt-fp32", action="store_true")
    ap.add_argument("--trt-cache", default=None)
    ap.add_argument("--trt-min-batch", type=int, default=1)
    ap.add_argument("--trt-max-batch", type=int, default=8)
    ap.add_argument(
        "--manifest",
        default=None,
        help="artifact cache manifest recorded per measured component (default onnx/.build_manifest.json)",
    )
    args = ap.parse_args(argv)

    fingerprint = env_fingerprint()
    print(f"[select] fingerprint: {fingerprint}")

    if args.dry_run:
        cfg = _assemble({}, args, fingerprint, None)
        cfg.components = {n: ComponentConfig(name=n) for n in COMPONENTS}
        path = save_backends(cfg, args.out)
        print(f"[select] dry-run wrote {path}")
        return 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[select] device={device} iters={args.iters} min_gain={args.min_gain}")
    # On a CUDA box whose CUDA EP is unusable (e.g. torch cu130 vs ORT cu12),
    # still measure ONNX so its (losing) result is on the record.
    if fingerprint.get("cuda") and not fingerprint.get("cuda_ep"):
        args.cpu_ort = True
    builder = OnnxBuilder(args.manifest)

    names = [n.strip() for n in args.components.split(",") if n.strip()]
    if args.hot_only:
        names = [n for n in names if n in HOT_COMPONENTS]

    components: dict[str, ComponentConfig] = {}
    for name in names:
        builder = BUILDERS.get(name)
        if builder is None:
            print(f"[select:{name}] no builder — skipping")
            continue
        try:
            case = builder(args, device)
        except Exception as e:
            print(f"[select:{name}] build FAILED: {e}")
            traceback.print_exc()
            components[name] = ComponentConfig(name=name, mode="eager", reason=f"build failed: {type(e).__name__}")
            continue
        try:
            cfg, _best, _eager = _select_component(name, case, args, builder, device)
            components[name] = cfg
        finally:
            del case
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    glob = _select_global(args, device)
    if glob:
        print(f"[select:global] {glob}")
        # Advisory heuristic: a whole-Generator compile win promotes the two
        # components it fuses, when they were left eager.
        if (
            glob.get("speedup", 1.0) >= args.min_gain
            and components.get("refenc", ComponentConfig("refenc")).mode == "eager"
            and components.get("decoder", ComponentConfig("decoder")).mode == "eager"
        ):
            for n in ("refenc", "decoder"):
                if n in components:
                    components[n].mode = "compile"
                    components[n].reason = "global Generator compile won"

    path = save_backends(_assemble(components, args, fingerprint, glob), args.out)
    print(f"\n[select] wrote {path}")
    for n, c in components.items():
        extra = f"/{c.provider}" if c.provider else ""
        print(f"  {n:9s} {c.mode}{extra}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
