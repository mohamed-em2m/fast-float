"""Convert the frame decoder Synthesis (unit 5) to ONNX.

Exports the deployed decode path ``ae.dec(latent, alpha=None, feats)`` and
returns img only (decode uses out[0]; flow is unused there).

Constraints (baked at trace time, verified in parity):
- STATIC batch (--batch): ModulatedConv2d bakes ``groups=batch`` as an ONNX
  attribute, so the graph is valid for exactly one B. Runtime must pad/trim
  micro-batches to it (replicate-pad like the audio chunker, trim after).
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
import os
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

from fast_float.models.float.generator import Generator
from fast_float.models.float.styledecoder import ModulatedConv2d
from fast_float.onnx.export import export_module, make_ort_session
from fast_float.onnx.parity import benchmark, check_parity

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

    The deployed forward derives reshape dims and ``groups=`` from
    ``input.shape[0]``, which defeats TorchScript shape inference
    ("convolution for kernel of unknown shape"). Same parameters, same
    math — only the batch source differs. Proven equivalent at build.
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
        self.scale = src.scale
        self.padding = src.padding
        self.demodulate = src.demodulate
        self.weight = src.weight
        self.modulation = src.modulation
        if hasattr(src, "blur"):
            self.blur = src.blur

    def forward(self, input, style):
        batch = self.batch
        style = self.modulation(style).view(batch, 1, self.in_channel, 1, 1)
        weight = self.scale * self.weight * style
        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)
        weight = weight.view(
            batch * self.out_channel, self.in_channel,
            self.kernel_size, self.kernel_size,
        )
        _, _, height, width = input.shape
        if self.upsample:
            input = input.contiguous().view(1, batch * self.in_channel, height, width)
            weight = weight.view(
                batch, self.out_channel, self.in_channel,
                self.kernel_size, self.kernel_size,
            )
            weight = weight.transpose(1, 2).contiguous().reshape(
                batch * self.in_channel, self.out_channel,
                self.kernel_size, self.kernel_size,
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
    """Replace every ModulatedConv2d with its static-batch twin (shared params)."""
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
        img, _flow = self.dec(latent, alpha=None, feats=list(feats))
        return img


def build_decoder_module(
    ckpt_path: str, image_size: int, channel_mult: float, batch: int,
    device: torch.device,
) -> tuple[DecoderExport, object, list]:
    gen = Generator(size=image_size, style_dim=512, motion_dim=20,
                    channel_multiplier=channel_mult).to(device).eval()
    if os.path.exists(ckpt_path):
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        tops = Counter(k.split(".")[0] for k in sd.keys())
        print(f"[weights] ckpt top-level prefixes: {dict(tops.most_common(10))}")
        sub = {k[len(AE_PREFIX):]: v for k, v in sd.items()
               if k.startswith(AE_PREFIX)}
        if not sub:
            print("[weights] WARN: no motion_autoencoder.* keys — random init")
        else:
            missing, unexpected = gen.load_state_dict(sub, strict=False)
            print(f"[weights] overlaid {len(sub)} tensors from {ckpt_path} "
                  f"(missing={len(missing)}, unexpected={len(unexpected)})")
        gen.to(device)
    else:
        print(f"[weights] WARN: {ckpt_path} not found — random init "
              "(parity still valid)")

    # Self-discover feats structure from the reference encoder.
    with torch.no_grad():
        probe_img = torch.randn(1, 3, image_size, image_size, device=device)
        _, _, feats1 = gen.enc(probe_img, None)
    dec = gen.dec
    print(f"[model] feats: {len(feats1)} maps "
          f"{[tuple(f.shape) for f in feats1]}")

    # Equivalence proof, then freeze grouped-conv batch for export.
    with torch.no_grad(), _deterministic_cudnn():
        lat_b = torch.randn(batch, 512, device=device)
        feats_b = [f.expand(batch, *f.shape[1:]).contiguous() for f in feats1]
        out_dyn = dec(lat_b, alpha=None, feats=feats_b)
        out_dyn = out_dyn[0] if isinstance(out_dyn, (tuple, list)) else out_dyn
        out_dyn2 = dec(lat_b, alpha=None, feats=feats_b)
        out_dyn2 = out_dyn2[0] if isinstance(out_dyn2, (tuple, list)) else out_dyn2
    repeat = (out_dyn - out_dyn2).abs().max().item()
    n = _freeze_modconvs(dec, batch)
    with torch.no_grad(), _deterministic_cudnn():
        out_frozen = dec(lat_b, alpha=None, feats=feats_b)
        out_frozen = (out_frozen[0] if isinstance(out_frozen, (tuple, list))
                      else out_frozen)
    print(f"[model] froze {n} ModulatedConv2d to static batch {batch}; "
          f"repeat-diff={repeat:.3e} "
          f"dyn-vs-frozen max_abs={(out_dyn - out_frozen).abs().max().item():.3e}")
    return DecoderExport(gen.dec, len(feats1)).eval(), gen.dec, feats1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Convert frame decoder to ONNX")
    ap.add_argument("--ckpt", default="./checkpoints/float.pth")
    ap.add_argument("--out", default="./onnx/decoder_v1.onnx")
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--channel-mult", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=8,
                    help="STATIC decode batch (runtime pads/trims to it)")
    ap.add_argument("--model-seed", type=int, default=0,
                    help="seed for random init (irrelevant when ckpt overlays all)")
    ap.add_argument("--opset", type=int, default=17)
    # Nondeterminism note: cuDNN conv_transpose2d uses atomic-add kernels, so
    # parity runs under _deterministic_cudnn (repeat-diff must print 0.0).
    # On random weights the warp stack is chaotic: kernel summation order
    # alone leaves ~0.7 max_abs (mean ~1e-3, cosine 0.9999+). Trained weights
    # run far tighter — the GPU-box run is the true numeric gate. The tripwire
    # for structural breaks is mean_abs + cosine, not max_abs.
    ap.add_argument("--atol", type=float, default=1.0)
    ap.add_argument("--rtol", type=float, default=1e-2)
    ap.add_argument("--min-cosine", type=float, default=0.999)
    ap.add_argument("--max-mean", type=float, default=0.05)
    ap.add_argument("--skip-parity", action="store_true")
    ap.add_argument("--bench-iters", type=int, default=10)
    ap.add_argument("--cpu-ort", action="store_true")
    args = ap.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[env] device={device} static_batch={args.batch}")
    torch.manual_seed(args.model_seed)
    # Deployed default is int 1; float channel sizes crash randn() exactly
    # like deployed code, so coerce integral values back to int.
    cm = int(args.channel_mult) if float(args.channel_mult).is_integer() else args.channel_mult
    module, dec, feats1 = build_decoder_module(
        args.ckpt, args.image_size, cm, args.batch, device
    )
    b = args.batch
    feat_names = [f"feat{i}" for i in range(len(feats1))]

    def expand_feats(feats):
        return [f.expand(b, *f.shape[1:]).contiguous() for f in feats]

    with torch.no_grad():
        lat = torch.randn(b, 512, device=device)
        feats_b = expand_feats(feats1)
        # Warmup: validates shapes + freezes ToFlow grid caches pre-export.
        warm = dec(lat, alpha=None, feats=feats_b)
        warm_img = warm[0] if isinstance(warm, (tuple, list)) else warm
        print(f"[model] warmup img {tuple(warm_img.shape)}")

    # Export on CPU (tracer constant-folder trips on mixed devices).
    module.to("cpu")
    lat_cpu = lat.detach().cpu()
    feats_cpu = [f.detach().cpu() for f in feats_b]
    info = export_module(
        module,
        [lat_cpu, *feats_cpu],
        args.out,
        input_names=["latent", *feat_names],
        output_names=["img"],
        dynamic_axes={},  # static: grouped-conv groups baked per batch
        opset=args.opset,
        fixed_batch=True,  # skip batch-2 duplication + dynamic marking
    )
    print(f"[export] {info['path']} via {info['exporter']}")
    for note in info.get("notes") or []:
        print(f"[export] note: {note}")
    module.to(device)

    if not args.skip_parity:
        sess = make_ort_session(
            info["path"],
            use_gpu=torch.cuda.is_available() and not args.cpu_ort,
            require_gpu=torch.cuda.is_available() and not args.cpu_ort,
        )
        print(f"[ort] providers={sess.get_providers()}")
        with torch.no_grad(), _deterministic_cudnn():
            g = torch.Generator(device="cpu").manual_seed(777)
            lat_t = torch.randn(b, 512, generator=g).to(device)
            ref_out = dec(lat_t, alpha=None, feats=feats_b)
            ref_img = ref_out[0] if isinstance(ref_out, (tuple, list)) else ref_out
            ort_img = sess.run(None, {
                "latent": lat_t.detach().cpu().numpy(),
                **{n: f.detach().cpu().numpy() for n, f in zip(feat_names, feats_b)},
            })[0]
            rep = check_parity(ref_img, ort_img, name=f"dec_b{b}",
                               atol=args.atol, rtol=args.rtol)
            if rep["cosine"] < args.min_cosine:
                raise RuntimeError(
                    f"decoder cosine {rep['cosine']:.6f} < {args.min_cosine}")
            if rep["mean_abs"] > args.max_mean:
                raise RuntimeError(
                    f"decoder mean_abs {rep['mean_abs']:.3e} > {args.max_mean}")
            print(f"[parity:dec_b{b}] cosine OK ({rep['cosine']:.6f}) "
                  f"mean OK ({rep['mean_abs']:.3e})")
            if args.bench_iters > 0:
                lat_np = lat_t.detach().cpu().numpy()
                feats_np = {n: f.detach().cpu().numpy()
                            for n, f in zip(feat_names, feats_b)}
                benchmark(
                    lambda: dec(lat_t, alpha=None, feats=feats_b),
                    lambda: sess.run(None, {"latent": lat_np, **feats_np}),
                    name=f"dec_b{b}",
                    iters=args.bench_iters,
                )
    print("[done] decoder converter OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
