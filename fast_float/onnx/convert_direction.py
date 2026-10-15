"""Convert the Direction motion basis (unit 3) to ONNX.

Exports the QR-free path: Q is baked as a buffer (same cache the torch
model uses since the QR fix), forward is diag_embed + matmul + sum.
Canonical shapes: x (B, M) -> out (B, 512). Dynamic batch.
"""
from __future__ import annotations

import argparse
import os
from collections import Counter

import torch
import torch.nn as nn

from fast_float.models.float.styledecoder import Direction
from fast_float.onnx.export import export_module, make_ort_session
from fast_float.onnx.parity import benchmark, check_parity


class DirectionExport(nn.Module):
    """QR-free Direction: Q baked as a buffer, always-batched input.

    Note: the deployed ``(diag_embed(x) @ Q.T).sum(1)`` is algebraically
    ``x @ Q.T`` (verified to 0.0 at build); the matmul form avoids
    aten::diag_embed, which the classic exporter cannot translate.
    """

    def __init__(self, direction: Direction):
        super().__init__()
        q = direction._orthonormal_basis().detach().clone()
        self.register_buffer("Q", q)
        self.motion_dim = direction.weight.shape[1]
        with torch.no_grad():
            probe = torch.randn(4, self.motion_dim, device=q.device)
            a = (torch.diag_embed(probe) @ q.T).sum(1)
            b = probe @ q.T
            print(f"[model] diag_embed-form vs matmul max_abs="
                  f"{(a - b).abs().max().item():.3e}")

    def forward(self, x):
        return x @ self.Q.T


def build_direction_module(
    ckpt_path: str, motion_dim: int, device: torch.device
) -> tuple[DirectionExport, Direction]:
    direction = Direction(motion_dim).to(device).eval()
    if os.path.exists(ckpt_path):
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        tops = Counter(k.split(".")[0] for k in sd.keys())
        print(f"[weights] ckpt top-level prefixes: {dict(tops.most_common(10))}")
        # Direction lives at motion_autoencoder.dec.direction; fall back to a
        # suffix scan (there is exactly one Direction module in the graph).
        sub = {
            "weight": v
            for k, v in sd.items()
            if k == "motion_autoencoder.dec.direction.weight"
        }
        if not sub:
            cands = [k for k in sd.keys() if k.endswith("dec.direction.weight")]
            if len(cands) == 1:
                sub = {"weight": sd[cands[0]]}
                print(f"[weights] fallback prefix: {cands[0]}")
        if not sub:
            print(
                "[weights] WARN: no direction weight in ckpt — exporting "
                "randomly initialized Direction (parity still valid)"
            )
        else:
            w = sub["weight"]
            if tuple(w.shape) != tuple(direction.weight.shape):
                print(
                    f"[weights] WARN: ckpt direction shape {tuple(w.shape)} != "
                    f"model {tuple(direction.weight.shape)} — rebuilding"
                )
                direction = Direction(w.shape[1]).to(device).eval()
            missing, unexpected = direction.load_state_dict(sub, strict=False)
            print(
                f"[weights] overlaid {len(sub)} tensors from {ckpt_path} "
                f"(missing={len(missing)}, unexpected={len(unexpected)})"
            )
        direction.to(device)
    else:
        print(
            f"[weights] WARN: {ckpt_path} not found — exporting randomly "
            "initialized Direction (parity still valid)"
        )
    return DirectionExport(direction).eval(), direction


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Convert Direction basis to ONNX")
    ap.add_argument("--ckpt", default="./checkpoints/float.pth")
    ap.add_argument("--out", default="./onnx/direction_v1.onnx")
    ap.add_argument("--motion-dim", type=int, default=20)
    ap.add_argument("--batches", default="1,4")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--atol", type=float, default=2e-4)
    ap.add_argument("--rtol", type=float, default=2e-4)
    ap.add_argument("--skip-parity", action="store_true")
    ap.add_argument("--bench-iters", type=int, default=100)
    ap.add_argument("--cpu-ort", action="store_true")
    args = ap.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[env] device={device}")
    module, direction = build_direction_module(args.ckpt, args.motion_dim, device)
    batches = [int(s) for s in args.batches.split(",") if s.strip()]
    m = module.motion_dim

    sample = torch.randn(1, m, device=device)
    info = export_module(
        module,
        sample,
        args.out,
        input_names=["x"],
        output_names=["dir"],
        dynamic_axes={"x": {0: "batch"}, "dir": {0: "batch"}},
        opset=args.opset,
    )
    print(f"[export] {info['path']} via {info['exporter']}")
    for note in info.get("notes") or []:
        print(f"[export] note: {note}")

    if not args.skip_parity:
        sess = make_ort_session(
            info["path"],
            use_gpu=torch.cuda.is_available() and not args.cpu_ort,
            require_gpu=torch.cuda.is_available() and not args.cpu_ort,
        )
        print(f"[ort] providers={sess.get_providers()}")
        with torch.no_grad():
            for bi, b in enumerate(batches):
                g = torch.Generator(device="cpu").manual_seed(1000 + bi)
                x = torch.randn(b, m, generator=g).to(device)
                ref = direction(x)
                ort_out = sess.run(None, {"x": x.detach().cpu().numpy()})[0]
                check_parity(ref, ort_out, name=f"dir_b{b}",
                             atol=args.atol, rtol=args.rtol)
                if args.bench_iters > 0:
                    x_np = x.detach().cpu().numpy()
                    benchmark(
                        lambda: direction(x),
                        lambda: sess.run(None, {"x": x_np}),
                        name=f"dir_b{b}",
                        iters=args.bench_iters,
                    )
    print("[done] direction converter OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
