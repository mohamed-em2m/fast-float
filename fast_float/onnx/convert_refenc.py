"""Convert the reference encoder (unit 6) to ONNX.

Exports the deployed ``encode_image_into_latent`` path: img -> x_r latent,
x_r_lambda (fc motion code), and the 7 reference feature maps that feed the
frame decoder. All-plain convs (no grouped conv), so batch is dynamic.

Runs once per request, so the speedup value is ~zero on its own — this unit
exists to complete torch-free deployment alongside the other five.
"""
from __future__ import annotations

import argparse
import os
from collections import Counter

import torch
import torch.nn as nn

from fast_float.models.float.generator import Generator
from fast_float.onnx.export import (
    add_ort_args,
    export_module,
    fp16_tol,
    make_session_from_args,
)
from fast_float.onnx.parity import benchmark, check_parity

AE_PREFIX = "motion_autoencoder."


class RefEncExport(nn.Module):
    """Reference image -> (x_r, x_r_lambda, *feats). Static image, dyn batch."""

    def __init__(self, enc):
        super().__init__()
        self.enc = enc

    def forward(self, img):
        x_r, _, feats = self.enc(img, None)
        x_r_lambda = self.enc.fc(x_r)
        return (x_r, x_r_lambda, *feats)


def build_refenc_module(
    ckpt_path: str, image_size: int, motion_dim: int, device: torch.device
):
    gen = Generator(size=image_size, style_dim=512, motion_dim=motion_dim).to(device)
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
    enc = gen.enc.eval()
    with torch.no_grad():
        probe = torch.randn(1, 3, image_size, image_size, device=device)
        x_r, _, feats = enc(probe, None)
        lam = enc.fc(x_r)
    print(f"[model] x_r {tuple(x_r.shape)} lambda {tuple(lam.shape)} "
          f"feats {len(feats)} {[tuple(f.shape) for f in feats]}")
    return RefEncExport(enc).eval(), enc, len(feats)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Convert reference encoder to ONNX")
    ap.add_argument("--ckpt", default="./checkpoints/float.pth")
    ap.add_argument("--out", default="./onnx/refenc_v1.onnx")
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--motion-dim", type=int, default=20)
    ap.add_argument("--batches", default="1,2")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--atol", type=float, default=2e-4)
    ap.add_argument("--rtol", type=float, default=2e-4)
    ap.add_argument("--skip-parity", action="store_true")
    ap.add_argument("--bench-iters", type=int, default=10)
    add_ort_args(ap)
    args = ap.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[env] device={device}")
    module, enc, n_feats = build_refenc_module(
        args.ckpt, args.image_size, args.motion_dim, device)
    batches = [int(s) for s in args.batches.split(",") if s.strip()]
    out_names = ["x_r", "x_r_lambda", *[f"feat{i}" for i in range(n_feats)]]

    sample = torch.randn(1, 3, args.image_size, args.image_size, device=device)
    module.to("cpu")
    info = export_module(
        module,
        sample.detach().cpu(),
        args.out,
        input_names=["img"],
        output_names=out_names,
        dynamic_axes={"img": {0: "batch"}, **{n: {0: "batch"} for n in out_names}},
        opset=args.opset,
        mark_dims=(0,),  # spatial dims static; batch only
    )
    print(f"[export] {info['path']} via {info['exporter']}")
    for note in info.get("notes") or []:
        print(f"[export] note: {note}")
    module.to(device)

    if not args.skip_parity:
        sess, fp16 = make_session_from_args(
            info["path"], args,
            opt_shapes={"img": [1, 3, args.image_size, args.image_size]},
        )
        atol, rtol = fp16_tol(args.atol, fp16), fp16_tol(args.rtol, fp16)
        with torch.no_grad():
            for bi, b in enumerate(batches):
                g = torch.Generator(device="cpu").manual_seed(300 + bi)
                img = torch.randn(b, 3, args.image_size, args.image_size,
                                  generator=g).to(device)
                x_r, _, feats = enc(img, None)
                ref = (x_r, enc.fc(x_r), *feats)
                outs = sess.run(None, {"img": img.detach().cpu().numpy()})
                assert len(outs) == len(ref), (len(outs), len(ref))
                for name, r, o in zip(out_names, ref, outs):
                    check_parity(r, o, name=f"{name}_b{b}",
                                 atol=atol, rtol=rtol)
                if args.bench_iters > 0:
                    img_np = img.detach().cpu().numpy()

                    def torch_fn():
                        xr, _, fs = enc(img, None)
                        return (xr, enc.fc(xr), *fs)

                    benchmark(torch_fn,
                              lambda: sess.run(None, {"img": img_np}),
                              name=f"refenc_b{b}", iters=args.bench_iters)
    print("[done] refenc converter OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
