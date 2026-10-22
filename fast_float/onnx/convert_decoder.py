"""Convert the frame decoder Synthesis (unit 5) to ONNX.

Exports the deployed decode path ``ae.dec(latent, alpha=None, feats)`` and
returns img only (decode uses out[0]; flow is unused there).

Constraints (baked at trace time, verified in parity):
- STATIC batch per artifact: the graph is traced at a fixed B, so each graph
  is valid for exactly one B. We export a *set* (``--batches``, default
  1,2,4,8,16): the primary ``--batch`` (also the selector's ``--dec-batch``)
  goes to ``--out`` and the rest to ``<stem>_b<N>.onnx`` siblings.
  ``DecoderOnnx`` then picks the smallest session covering a micro-batch and
  replicate-pads/trims to it — instead of always padding up to the single
  largest graph.
- The graph is the *model itself*: the deployed ``ModulatedConv2d`` is a dense
  modulated convolution (the style is folded into the input channels, no
  ``groups``). An earlier export substituted a grouped variant (``groups=batch``
  baked into the ONNX attribute); the ORT CUDA EP can only serve grouped convs
  through its fallback kernel, which drifts ~90 uint8 LSB from torch's and is
  not run-to-run reproducible — the artifact then could not reproduce the
  frames the eager path produces.
- Parity is judged in the *deployed* uint8 domain (2 LSB max / 0.5 LSB mean)
  with **both sides in fp32**: TF32 alone moves this 22-conv stack by ~40
  uint8 LSB, which would drown the graph's own error.
- alpha=None path: decode never blends directions; the direction branches
  are not in the graph.
- feats structure (count/shapes) is self-discovered by running the reference
  encoder once — no hardcoded map list.
- ToFlow grid caches are warmed pre-export so meshgrid/linspace freeze as
  constants; pass matching feat batches (runtime expands batch-1 feats).
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import os
from collections import Counter

import torch
import torch.nn as nn

from fast_float.models.float.generator import Generator
from fast_float.onnx.export import (
    TRT_DISABLED_MARKER,
    add_ort_args,
    export_module,
    make_session_from_args,
)
from fast_float.onnx.parity import U8_MAX_LSB, U8_MAX_MEAN_LSB, benchmark, compare_outputs, u8_lsb

AE_PREFIX = "motion_autoencoder."


@contextlib.contextmanager
def _deterministic_cudnn():
    """Force deterministic, full-precision cuDNN for parity (repeat-diff must be 0).

    Bench runs outside this context with deployed flags. Parity compares two
    deterministic computations (torch-det vs ORT), so residual diff is true
    graph divergence, not kernel noise.

    TF32 is disabled as well: it rounds every accumulation to a 10-bit
    mantissa, and through this 22-conv stack that alone moves the output by
    ~40 uint8 LSB. The ORT side of the comparison is pinned to fp32 as well
    (``use_tf32=False``), so both sides compute the *same* thing and the
    residual is the graph's divergence -- which is what the gate is for.
    """
    b, d = torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = b, d
        torch.backends.cudnn.allow_tf32 = cudnn_tf32
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32


def _fold_static_weights(dec) -> int:
    """Fold the static scales of the plain convs / linears (as the eager model does)."""
    folded = 0
    for m in dec.modules():
        if hasattr(m, "fold_weights"):
            m.fold_weights()
            folded += 1
    return folded


class DecoderExport(nn.Module):
    """Decode-path Synthesis: latent + feats -> img. All shapes static."""

    def __init__(self, dec, n_feats: int):
        super().__init__()
        self.dec = dec
        self.n_feats = n_feats

    def forward(self, latent, *feats):
        out = self.dec(latent, alpha=None, feats=list(feats))
        img = out[0] if isinstance(out, (tuple, list)) else out
        return img


def _build_generator(ckpt_path: str, image_size: int, channel_mult: float, device: torch.device):
    """Construct the Generator, overlay the ckpt, and self-discover the feats map."""
    # Integral channel multipliers (the deployed default "1") feed tensor
    # shapes and must be ints; a float (e.g. 1.0) crashes randn() deep in the
    # decoder. Coerce here so every caller (selector included) is safe.
    if float(channel_mult).is_integer():
        channel_mult = int(channel_mult)
    gen = (
        Generator(
            size=image_size,
            style_dim=512,
            motion_dim=20,
            channel_multiplier=channel_mult,
        )
        .to(device)
        .eval()
    )

    if os.path.exists(ckpt_path):
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        tops = Counter(k.split(".")[0] for k in sd)
        print(f"[weights] ckpt top-level prefixes: {dict(tops.most_common(10))}")
        sub = {k[len(AE_PREFIX) :]: v for k, v in sd.items() if k.startswith(AE_PREFIX)}
        if not sub:
            print("[weights] WARN: no motion_autoencoder.* keys — random init")
        else:
            missing, unexpected = gen.load_state_dict(sub, strict=False)
            print(
                f"[weights] overlaid {len(sub)} tensors from {ckpt_path} "
                f"(missing={len(missing)}, unexpected={len(unexpected)})"
            )
        gen.to(device)
    else:
        print(f"[weights] WARN: {ckpt_path} not found — random init (parity still valid)")

    # Self-discover feats structure from the reference encoder.
    with torch.no_grad():
        probe_img = torch.randn(1, 3, image_size, image_size, device=device)
        _, _, feats1 = gen.enc(probe_img, None)
    print(f"[model] feats: {len(feats1)} maps {[tuple(f.shape) for f in feats1]}")
    return gen, feats1


def freeze_decoder_for_batch(dec_raw, n_feats: int, batch: int) -> DecoderExport:
    """Export module for one static decode batch, built from the *deployed* dec.

    The deployed ``ModulatedConv2d`` already computes a **dense** convolution
    with the style folded into the input channels -- it does not use
    ``groups=batch`` (the model was refactored to "dense modulated convolution
    without groups=batch"). The old export substituted a grouped variant
    (``groups=batch`` baked into the ONNX attribute), which the ORT CUDA EP
    can only serve through its *fallback* grouped-conv kernel: that kernel
    drifts ~90 uint8 LSB from torch's grouped conv and is not even
    run-to-run reproducible, so the shipped artifact could not reproduce the
    frames the eager path produces.

    So the graph is now the model itself, traced at a fixed batch. Deep-copy
    first: folding the static scales mutates in place, and the raw decoder
    stays the parity reference.
    """
    clone = copy.deepcopy(dec_raw)
    _fold_static_weights(clone)
    return DecoderExport(clone, n_feats).eval()


def _img(out):
    return out[0] if isinstance(out, (tuple, list)) else out


def build_decoder_module(
    ckpt_path: str,
    image_size: int,
    channel_mult: float,
    batch: int,
    device: torch.device,
) -> tuple[DecoderExport, object, list]:
    """Single-batch decoder + the raw (dynamic) dec for the parity reference.

    Kept for the backend selector; ``main`` builds the whole batch set.
    """
    gen, feats1 = _build_generator(ckpt_path, image_size, channel_mult, device)
    dec_raw = gen.dec
    module = freeze_decoder_for_batch(dec_raw, len(feats1), batch)

    # Equivalence proof (the exported copy vs the model it was copied from).
    with torch.no_grad(), _deterministic_cudnn():
        lat_b = torch.randn(batch, 512, device=device)
        feats_b = [f.expand(batch, *f.shape[1:]).contiguous() for f in feats1]
        out_dyn = _img(dec_raw(lat_b, alpha=None, feats=feats_b))
        out_dyn2 = _img(dec_raw(lat_b, alpha=None, feats=feats_b))
    repeat = (out_dyn - out_dyn2).abs().max().item()
    with torch.no_grad(), _deterministic_cudnn():
        out_copy = _img(module.dec(lat_b, alpha=None, feats=feats_b))
    print(
        f"[model] decoder graph for static batch {batch}; "
        f"repeat-diff={repeat:.3e} "
        f"model-vs-export-copy max_abs={(out_dyn - out_copy).abs().max().item():.3e}"
    )
    return module, dec_raw, feats1


def _parse_batches(spec: str) -> list[int]:
    out = sorted({int(s) for s in spec.split(",") if s.strip()})
    if not out:
        raise ValueError("--batches is empty")
    return out


def _u8_lsb(ref, out) -> tuple[int, float]:
    """Quantized-frame error (see :func:`fast_float.onnx.parity.u8_lsb`)."""
    return u8_lsb(ref, out)


def _measure(dec_raw, sess, lat_t, feats_b, feed) -> dict:
    """torch (dynamic dec) vs this session, in the deployed uint8 domain."""
    with torch.no_grad(), _deterministic_cudnn():
        ref_img = _img(dec_raw(lat_t, alpha=None, feats=feats_b))
        ort_img = sess.run(None, feed)[0]
    max_lsb, mean_lsb = _u8_lsb(ref_img, ort_img)
    return {
        "max_lsb": max_lsb,
        "mean_lsb": mean_lsb,
        "cosine": compare_outputs(ref_img, ort_img)["cosine"],
    }


def _gate(rep: dict, args) -> bool:
    return rep["max_lsb"] <= args.max_lsb and rep["mean_lsb"] <= args.max_mean_lsb


def _mark_no_trt(onnx_path: str) -> str:
    marker = f"{onnx_path}{TRT_DISABLED_MARKER}"
    with open(marker, "w", encoding="utf-8") as f:
        f.write(
            "TensorRT EP is vetoed for this decoder artifact: it either failed the uint8 "
            "parity gate or the artifact was validated without it (see convert_decoder).\n"
        )
    return marker


def _veto_or_clear_trt(onnx_path: str, b: int, trt_on: bool) -> bool:
    """Keep the shipped artifact and the runtime on the same execution provider.

    The runtime (``make_runtime_session``) always tries TensorRT first, so a
    graph validated on the CUDA EP must leave the ``.no_trt`` sidecar behind or
    production would silently run the provider that was *not* checked - exactly
    how a 10x-slower, visibly wrong decoder shipped before.
    """
    marker = f"{onnx_path}{TRT_DISABLED_MARKER}"
    if trt_on:
        if os.path.exists(marker):
            os.remove(marker)
            return True
        return False
    if not os.path.exists(marker):
        _mark_no_trt(onnx_path)
        print(f"[parity:dec_b{b}] wrote {os.path.basename(marker)}: runtime will skip TensorRT")
    return False


def _run_parity(dec_raw, args, device, b, feats1, feat_names, onnx_path) -> None:
    """Eager (dynamic dec) vs the static ONNX graph for one batch, then bench."""
    g = torch.Generator(device="cpu").manual_seed(777)
    lat_t = torch.randn(b, 512, generator=g).to(device)
    feats_b = [f.expand(b, *f.shape[1:]).contiguous() for f in feats1]
    feed = {
        "latent": lat_t.detach().cpu().numpy(),
        **{n: f.detach().cpu().numpy() for n, f in zip(feat_names, feats_b, strict=False)},
    }
    gate_desc = f"gate: max_lsb<={args.max_lsb}, mean_lsb<={args.max_mean_lsb:g}"

    # Static batch -> fully static graph, no TRT shape profiles needed.
    # The parity session is pinned to fp32 (no TF32) so both sides of the
    # comparison compute the same thing: TF32 alone moves this 22-conv stack
    # by ~40 uint8 LSB, which would drown the graph's own error.
    sess, _fp16 = make_session_from_args(onnx_path, args, use_tf32=False)
    trt_on = "TensorrtExecutionProvider" in sess.get_providers()
    rep = _measure(dec_raw, sess, lat_t, feats_b, feed)
    print(
        f"[parity:dec_b{b}] max_lsb={rep['max_lsb']} mean_lsb={rep['mean_lsb']:.3f} "
        f"cosine={rep['cosine']:.6f} ({gate_desc}{'' if trt_on else ', no TRT'})"
    )
    if not _gate(rep, args):
        if not trt_on:
            raise RuntimeError(
                f"decoder parity FAILED (b{b}): max_lsb={rep['max_lsb']} (max={args.max_lsb}), "
                f"mean_lsb={rep['mean_lsb']:.3f} (max={args.max_mean_lsb:g}), "
                f"cosine={rep['cosine']:.6f}, providers={sess.get_providers()}"
            )
        # TRT cannot run this graph faithfully (and is far slower): rebuild the
        # session on the CUDA EP, re-gate, and leave a marker so the runtime
        # never re-enables TRT for the artifact we just shipped.
        print(f"[parity:dec_b{b}] FAIL on TensorRT (max_lsb={rep['max_lsb']}) - retrying on the CUDA EP")
        fallback = copy.copy(args)
        fallback.no_trt = True
        sess, _fp16 = make_session_from_args(onnx_path, fallback, use_tf32=False)
        trt_on = "TensorrtExecutionProvider" in sess.get_providers()
        rep = _measure(dec_raw, sess, lat_t, feats_b, feed)
        print(
            f"[parity:dec_b{b}] max_lsb={rep['max_lsb']} mean_lsb={rep['mean_lsb']:.3f} "
            f"cosine={rep['cosine']:.6f} ({gate_desc}, no TRT)"
        )
        if not _gate(rep, args):
            raise RuntimeError(
                f"decoder parity FAILED (b{b}) with and without TensorRT: "
                f"max_lsb={rep['max_lsb']} (max={args.max_lsb}), "
                f"mean_lsb={rep['mean_lsb']:.3f} (max={args.max_mean_lsb:g})"
            )
    cleared = _veto_or_clear_trt(onnx_path, b, trt_on)
    if cleared:
        print(f"[parity:dec_b{b}] cleared stale {TRT_DISABLED_MARKER} (TRT passes now)")
    print(f"[parity:dec_b{b}] PASS")

    # Bench OUTSIDE deterministic cuDNN so torch runs its deployed kernels
    # (deterministic disables cuDNN autotuning and would inflate speedup).
    if args.bench_iters > 0:
        with torch.no_grad():
            benchmark(
                lambda: dec_raw(lat_t, alpha=None, feats=feats_b),
                lambda: sess.run(None, feed),
                name=f"dec_b{b}",
                iters=args.bench_iters,
            )


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Convert frame decoder to ONNX")
    ap.add_argument("--ckpt", default="./checkpoints/float.pth")
    ap.add_argument("--out", default="./onnx/decoder_v1.onnx")
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--channel-mult", type=float, default=1.0)
    ap.add_argument(
        "--batches",
        default="4,8,16",
        help="static decode batches to export (one artifact each); the runtime "
        "picks the smallest session covering a micro-batch instead of always "
        "padding up to the largest",
    )
    ap.add_argument(
        "--batch",
        type=int,
        default=4,
        help="PRIMARY static batch written to --out (matches the selector's "
        "--dec-batch); extra batches land in <stem>_b<N>.onnx siblings",
    )
    ap.add_argument(
        "--model-seed",
        type=int,
        default=0,
        help="seed for random init (irrelevant when ckpt overlays all)",
    )
    ap.add_argument("--opset", type=int, default=17)
    # Parity is gated in the *deployed* uint8 domain: the decode path clamps to
    # [-1, 1], scales by 127.5 and truncates, so float-space max_abs is not what
    # the video shows. 2 LSB max / 0.5 LSB mean is invisible on a 512x512 frame
    # and fails a structurally wrong graph (the broken TensorRT engine measured
    # max_abs=4.375, cosine=0.983 while "passing" atol=40).
    ap.add_argument("--max-lsb", type=int, default=U8_MAX_LSB, help="max uint8 frame error vs eager torch")
    ap.add_argument("--max-mean-lsb", type=float, default=U8_MAX_MEAN_LSB, help="mean uint8 frame error vs eager torch")
    ap.add_argument("--skip-parity", action="store_true")
    ap.add_argument("--bench-iters", type=int, default=10)
    ap.add_argument(
        "--trt",
        action="store_true",
        help="opt in to the TensorRT EP for the parity session; off by default "
        "because TRT partitions this graph (10x slower) and misses the uint8 gate",
    )
    add_ort_args(ap)
    return ap


def _parse_args(argv=None):
    """Parse flags and make TensorRT opt-in for this converter.

    A TRT session for this decoder was measured at 0.09x eager with a 4.375
    max_abs error, so the default parity session (and therefore the shipped
    artifact's ``.no_trt`` sidecar) must be the CUDA EP. ``--trt`` restores the
    old behaviour for a box where TRT has been re-validated.
    """
    args = _build_parser().parse_args(argv)
    args.no_trt = bool(args.no_trt) or not args.trt
    return args


def main(argv=None) -> int:
    args = _parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batches = sorted(set(_parse_batches(args.batches)) | {int(args.batch)})
    print(f"[env] device={device} static_batches={batches} primary={args.batch}")
    torch.manual_seed(args.model_seed)
    # Deployed default is int 1; float channel sizes crash randn() exactly
    # like deployed code, so coerce integral values back to int.
    cm = int(args.channel_mult) if float(args.channel_mult).is_integer() else args.channel_mult
    gen, feats1 = _build_generator(args.ckpt, args.image_size, cm, device)
    dec_raw = gen.dec
    feat_names = [f"feat{i}" for i in range(len(feats1))]

    stem = os.path.splitext(os.path.basename(args.out))[0]
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)

    for b in batches:
        # The primary batch is the registry/selector artifact (--out); the rest
        # are siblings the runtime discovers by filename.
        out_path = args.out if b == args.batch else os.path.join(out_dir, f"{stem}_b{b}.onnx")
        module = freeze_decoder_for_batch(dec_raw, len(feats1), b)
        with torch.no_grad():
            lat = torch.randn(b, 512, device=device)
            feats_b = [f.expand(b, *f.shape[1:]).contiguous() for f in feats1]
            # Warmup: validates shapes + freezes ToFlow grid caches pre-export.
            warm_img = _img(module.dec(lat, alpha=None, feats=feats_b))
            print(f"[model] b{b} warmup img {tuple(warm_img.shape)}")

        # Export on CPU (tracer constant-folder trips on mixed devices).
        module.to("cpu")
        info = export_module(
            module,
            [lat.detach().cpu(), *[f.detach().cpu() for f in feats_b]],
            out_path,
            input_names=["latent", *feat_names],
            output_names=["img"],
            dynamic_axes={},  # static: grouped-conv groups baked per batch
            opset=args.opset,
            fixed_batch=True,  # skip batch-2 duplication + dynamic marking
        )
        print(f"[export] b{b} {info['path']} via {info['exporter']}")
        for note in info.get("notes") or []:
            print(f"[export] note: {note}")
        module.to(device)

        if not args.skip_parity:
            _run_parity(dec_raw, args, device, b, feats1, feat_names, info["path"])

    print(f"[done] decoder converter OK ({len(batches)} batches: {batches})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
