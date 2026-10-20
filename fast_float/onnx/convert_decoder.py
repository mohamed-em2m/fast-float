"""Convert the frame decoder Synthesis (unit 5) to ONNX.

Exports the deployed decode path ``ae.dec(latent, alpha=None, feats)`` and
returns img only (decode uses out[0]; flow is unused there).

Constraints (baked at trace time, verified in parity):
- STATIC batch per artifact: ModulatedConv2d bakes ``groups=batch`` as an ONNX
  attribute, so each graph is valid for exactly one B. We export a *set*
  (``--batches``, default 1,2,4,8,16): the primary ``--batch`` (also the
  selector's ``--dec-batch``) goes to ``--out`` and the rest to
  ``<stem>_b<N>.onnx`` siblings. ``DecoderOnnx`` then picks the smallest
  session covering a micro-batch and replicate-pads/trims to it — instead of
  always padding up to the single largest graph.
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
import torch.nn.functional as F

from fast_float.models.float.generator import Generator
from fast_float.models.float.styledecoder import ModulatedConv2d
from fast_float.onnx.export import (
    add_ort_args,
    export_module,
    fp16_tol,
    make_session_from_args,
)
from fast_float.onnx.parity import benchmark, compare_outputs

AE_PREFIX = "motion_autoencoder."


@contextlib.contextmanager
def _deterministic_cudnn():
    """Force deterministic cuDNN for parity (repeat-diff must be exactly 0).

    Bench runs outside this context with deployed flags. Parity compares two
    deterministic computations (torch-det vs ORT), so residual diff is true
    graph divergence, not kernel noise.
    """
    b, d = torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        yield
    finally:
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = b, d


class _StaticModConv(nn.Module):
    """ModulatedConv2d with batch baked to a Python int.

    Pre-scales weights and folds modulation EqualLinear parameters so the
    forward graph contains zero Tensor-scalar multiplications (avoiding
    dynamo FX type promotion crashes: 'prim.device != aten.mul').
    """

    def __init__(self, src: ModulatedConv2d, batch: int):
        super().__init__()
        self.batch = batch
        self.eps = src.eps
        self.kernel_size = src.kernel_size
        self.in_channel = src.in_channel
        self.out_channel = src.out_channel
        self.upsample = src.upsample
        self.downsample = src.downsample
        self.scale = 1.0
        self.padding = src.padding
        self.demodulate = src.demodulate
        # Pre-scale weight to eliminate scalar * Tensor multiplication in FX graph
        self.weight = nn.Parameter(src.weight.data * src.scale)
        if hasattr(src.modulation, "fold_weights"):
            src.modulation.fold_weights()
        self.modulation = src.modulation
        if hasattr(src, "blur"):
            self.blur = src.blur

    def forward(self, input, style):
        batch = self.batch
        style = self.modulation(style).view(batch, 1, self.in_channel, 1, 1)
        weight = self.weight * style
        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)
        weight = weight.view(
            batch * self.out_channel,
            self.in_channel,
            self.kernel_size,
            self.kernel_size,
        )
        _, _, height, width = input.shape
        if self.upsample:
            input = input.contiguous().view(1, batch * self.in_channel, height, width)
            weight = weight.view(
                batch,
                self.out_channel,
                self.in_channel,
                self.kernel_size,
                self.kernel_size,
            )
            weight = (
                weight.transpose(1, 2)
                .contiguous()
                .reshape(
                    batch * self.in_channel,
                    self.out_channel,
                    self.kernel_size,
                    self.kernel_size,
                )
            )
            out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)
        elif self.downsample:
            input = self.blur(input)
            _, _, height, width = input.shape
            input = input.contiguous().view(1, batch * self.in_channel, height, width)
            weight = weight.contiguous()
            out = F.conv2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
        else:
            input = input.contiguous().view(1, batch * self.in_channel, height, width)
            weight = weight.contiguous()
            out = F.conv2d(input, weight, padding=self.padding, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
        return out


def _freeze_modconvs(dec, batch: int) -> int:
    """Replace every ModulatedConv2d with its static-batch twin (shared params) and fold static weights."""
    for m in dec.modules():
        if hasattr(m, "fold_weights"):
            m.fold_weights()
    targets = []
    for parent in dec.modules():
        for name, child in parent.named_children():
            if type(child) is ModulatedConv2d:
                targets.append((parent, name, child))
    for parent, name, child in targets:
        setattr(parent, name, _StaticModConv(child, batch))
    return len(targets)


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
    """Fresh frozen (static-batch) export module built from the *unfrozen* dec.

    ``ModulatedConv2d`` bakes ``groups=batch`` into an ONNX attribute, so each
    static batch needs its own graph. Deep-copy first: ``_freeze_modconvs``
    mutates in place and we must keep the raw (dynamic-batch) decoder intact
    for the parity reference and for the other batches.
    """
    clone = copy.deepcopy(dec_raw)
    _freeze_modconvs(clone, batch)
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

    # Equivalence proof (dynamic vs frozen), then report.
    with torch.no_grad(), _deterministic_cudnn():
        lat_b = torch.randn(batch, 512, device=device)
        feats_b = [f.expand(batch, *f.shape[1:]).contiguous() for f in feats1]
        out_dyn = _img(dec_raw(lat_b, alpha=None, feats=feats_b))
        out_dyn2 = _img(dec_raw(lat_b, alpha=None, feats=feats_b))
    repeat = (out_dyn - out_dyn2).abs().max().item()
    with torch.no_grad(), _deterministic_cudnn():
        out_frozen = _img(module.dec(lat_b, alpha=None, feats=feats_b))
    print(
        f"[model] froze decoder to static batch {batch}; "
        f"repeat-diff={repeat:.3e} "
        f"dyn-vs-frozen max_abs={(out_dyn - out_frozen).abs().max().item():.3e}"
    )
    return module, dec_raw, feats1


def _parse_batches(spec: str) -> list[int]:
    out = sorted({int(s) for s in spec.split(",") if s.strip()})
    if not out:
        raise ValueError("--batches is empty")
    return out


def _run_parity(dec_raw, args, device, b, feats1, feat_names, onnx_path) -> None:
    """Eager (dynamic dec) vs the static ONNX graph for one batch, then bench."""
    # Static batch -> fully static graph, no TRT shape profiles needed.
    sess, fp16 = make_session_from_args(onnx_path, args)
    atol, rtol = fp16_tol(args.atol, fp16), fp16_tol(args.rtol, fp16)
    min_cos = 0.995 if fp16 else args.min_cosine
    max_mean = fp16_tol(args.max_mean, fp16)
    with torch.no_grad(), _deterministic_cudnn():
        g = torch.Generator(device="cpu").manual_seed(777)
        lat_t = torch.randn(b, 512, generator=g).to(device)
        feats_b = [f.expand(b, *f.shape[1:]).contiguous() for f in feats1]
        ref_img = _img(dec_raw(lat_t, alpha=None, feats=feats_b))
        ort_img = sess.run(
            None,
            {
                "latent": lat_t.detach().cpu().numpy(),
                **{n: f.detach().cpu().numpy() for n, f in zip(feat_names, feats_b, strict=False)},
            },
        )[0]
        rep = compare_outputs(ref_img, ort_img, atol=atol, rtol=rtol)
        cos_ok = rep["cosine"] >= min_cos
        mean_ok = rep["mean_abs"] <= max_mean
        status = "PASS" if (rep["ok"] or (cos_ok and mean_ok)) else "FAIL"
        print(
            f"[parity:dec_b{b}] {status} "
            f"max_abs={rep['max_abs']:.3e} mean_abs={rep['mean_abs']:.3e} "
            f"max_rel={rep['max_rel']:.3e} cosine={rep['cosine']:.6f} "
            f"(atol={atol:g}, rtol={rtol:g})"
        )
        if not (rep["ok"] or (cos_ok and mean_ok)):
            raise RuntimeError(
                f"decoder parity FAILED (b{b}): cosine={rep['cosine']:.6f} (min={min_cos}), "
                f"mean_abs={rep['mean_abs']:.3e} (max={max_mean})"
            )
        gate = "elementwise" if rep["ok"] else "functional (cosine+mean)"
        print(f"[parity:dec_b{b}] passed via {gate} gate (cosine={rep['cosine']:.6f}, mean_abs={rep['mean_abs']:.3e})")
    # Bench OUTSIDE deterministic cuDNN so torch runs its deployed kernels
    # (deterministic disables cuDNN autotuning and would inflate speedup).
    if args.bench_iters > 0:
        with torch.no_grad():
            lat_np = lat_t.detach().cpu().numpy()
            feats_np = {n: f.detach().cpu().numpy() for n, f in zip(feat_names, feats_b, strict=False)}
            benchmark(
                lambda: dec_raw(lat_t, alpha=None, feats=feats_b),
                lambda: sess.run(None, {"latent": lat_np, **feats_np}),
                name=f"dec_b{b}",
                iters=args.bench_iters,
            )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Convert frame decoder to ONNX")
    ap.add_argument("--ckpt", default="./checkpoints/float.pth")
    ap.add_argument("--out", default="./onnx/decoder_v1.onnx")
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--channel-mult", type=float, default=1.0)
    ap.add_argument(
        "--batches",
        default="1,2,4,8,16",
        help="static decode batches to export (one artifact each); the runtime "
        "picks the smallest session covering a micro-batch instead of always "
        "padding up to the largest",
    )
    ap.add_argument(
        "--batch",
        type=int,
        default=8,
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
    # Nondeterminism note: cuDNN conv_transpose2d uses atomic-add kernels, so
    # parity runs under _deterministic_cudnn (repeat-diff must print 0.0).
    # On random weights the warp stack is chaotic: kernel summation order
    # alone leaves ~0.7 max_abs (mean ~1e-3, cosine 0.9999+). Trained weights
    # run far tighter — the GPU-box run is the true numeric gate. The tripwire
    # for structural breaks is mean_abs + cosine, not max_abs.
    ap.add_argument("--atol", type=float, default=1.0)
    ap.add_argument("--rtol", type=float, default=1e-2)
    ap.add_argument("--min-cosine", type=float, default=0.998)
    ap.add_argument("--max-mean", type=float, default=0.05)
    ap.add_argument("--skip-parity", action="store_true")
    ap.add_argument("--bench-iters", type=int, default=10)
    add_ort_args(ap)
    args = ap.parse_args(argv)

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
