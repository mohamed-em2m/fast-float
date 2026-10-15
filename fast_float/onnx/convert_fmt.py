"""Unit 3: convert the FLOAT flow-matching transformer step to ONNX.

Wraps :class:`FlowMatchingTransformer.forward(train=False)` — a single
ODE vector-field evaluation::

    (t[1], x[B,50,512], wa[B,50,512], wr[B,1,512], we[B,1,7],
     prev_x[B,10,512], prev_wa[B,10,512]) -> vec [B,60,512]

The ODE loop, CFG cat/chunk/combine and chunk autoregression stay in
Python (see ``forward_with_cfv`` / ``sample``); only this inner step —
the dominant cost (~9 evals x 3B batch per 50-frame chunk) — runs in ORT.
Batch is dynamic, so one file serves both the plain (B) and CFG (3B)
calls. Internal length is fixed at 60 (50 clip + 10 prev) by construction.

Structural CLI flags (--fmt-depth, --dim-h, ...) must match the deploy
opt; the app runs a live torch-vs-ONNX probe at init and falls back to
torch on any mismatch.

Usage (from repo root):
    python -m fast_float.onnx.convert_fmt --out ./onnx/fmt_step_v1.onnx
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

# Repo-root bootstrap (same rule as the other converters: only REPO_ROOT
# may be added — never fast_float/ itself, whose onnx/ subpackage would
# shadow the pip `onnx` dependency).
CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fast_float.models.float.FMT import FlowMatchingTransformer  # noqa: E402
from fast_float.onnx.export import export_module, make_ort_session  # noqa: E402
from fast_float.onnx.parity import check_parity  # noqa: E402

FMT_PREFIX = "fmt."
CLIP_FRAMES = 50
PREV_FRAMES = 10


class FMTStepWrapper(torch.nn.Module):
    """Single vector-field step. ``train=False`` disables all dropout
    paths, keeping the graph tracer-friendly."""

    def __init__(self, fmt: FlowMatchingTransformer):
        super().__init__()
        self.fmt = fmt

    def forward(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        wa: torch.Tensor,
        wr: torch.Tensor,
        we: torch.Tensor,
        prev_x: torch.Tensor,
        prev_wa: torch.Tensor,
    ) -> torch.Tensor:
        return self.fmt.forward(
            t, x, wa, wr, we, prev_x, prev_wa, train=False
        )


def build_fmt(args) -> FlowMatchingTransformer:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    opt = SimpleNamespace(
        wav2vec_sec=2.0,
        fps=25.0,
        num_prev_frames=args.prev_frames,
        dim_h=args.dim_h,
        mlp_ratio=args.mlp_ratio,
        fmt_depth=args.fmt_depth,
        num_heads=args.num_heads,
        dim_w=512,
        dim_a=512,
        dim_e=7,
        attention_window=2,
        rank=device,
        audio_dropout_prob=0.1,
        ref_dropout_prob=0.1,
        emotion_dropout_prob=0.1,
    )
    fmt = FlowMatchingTransformer(opt)
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    sub = {k[len(FMT_PREFIX):]: v for k, v in sd.items() if k.startswith(FMT_PREFIX)}
    if not sub:
        raise KeyError(f"no {FMT_PREFIX}* keys in {args.ckpt}")
    missing, unexpected = fmt.load_state_dict(sub, strict=False)
    print(
        f"[weights] fmt: {len(sub)} tensors from {args.ckpt} "
        f"(missing={len(missing)}, unexpected={len(unexpected)})"
    )
    if missing or unexpected:
        raise RuntimeError(f"fmt weight mismatch: missing={missing} unexpected={unexpected}")
    return fmt.eval()


def sample_inputs(batch: int, t_val: float, seed: int, device: str):
    g = torch.Generator().manual_seed(seed)
    kw = dict(dtype=torch.float32)
    t = torch.tensor([t_val], **kw)
    x = torch.randn(batch, CLIP_FRAMES, 512, generator=g, **kw).clamp(-1, 1)
    wa = torch.randn(batch, CLIP_FRAMES, 512, generator=g, **kw).clamp(-1, 1)
    wr = torch.randn(batch, 1, 512, generator=g, **kw).clamp(-1, 1)
    we = torch.randn(batch, 1, 7, generator=g, **kw).clamp(-1, 1)
    prev_x = torch.randn(batch, PREV_FRAMES, 512, generator=g, **kw).clamp(-1, 1)
    prev_wa = torch.randn(batch, PREV_FRAMES, 512, generator=g, **kw).clamp(-1, 1)
    if device == "cuda":
        t, x, wa, wr, we, prev_x, prev_wa = (
            v.cuda() for v in (t, x, wa, wr, we, prev_x, prev_wa)
        )
    return t, x, wa, wr, we, prev_x, prev_wa


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Convert FLOAT FMT step to ONNX")
    ap.add_argument("--ckpt", default="./checkpoints/float.pth")
    ap.add_argument("--out", default="./onnx/fmt_step_v1.onnx")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--fmt-depth", type=int, default=8)
    ap.add_argument("--dim-h", type=int, default=1024)
    ap.add_argument("--num-heads", type=int, default=8)
    ap.add_argument("--mlp-ratio", type=float, default=4.0)
    ap.add_argument("--prev-frames", type=int, default=10)
    ap.add_argument("--atol", type=float, default=1e-4)
    ap.add_argument("--rtol", type=float, default=1e-4)
    ap.add_argument("--skip-parity", action="store_true")
    args = ap.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    fmt = build_fmt(args).to(device)
    module = FMTStepWrapper(fmt).eval().to(device)

    trace = sample_inputs(2, 0.5, 0, device)  # batch 2: dynamic batch mark
    info = export_module(
        module,
        trace,
        args.out,
        input_names=["t", "x", "wa", "wr", "we", "prev_x", "prev_wa"],
        output_names=["vec"],
        dynamic_axes={
            "t": {},
            "x": {0: "batch"},
            "wa": {0: "batch"},
            "wr": {0: "batch"},
            "we": {0: "batch"},
            "prev_x": {0: "batch"},
            "prev_wa": {0: "batch"},
            "vec": {0: "batch"},
        },
        opset=args.opset,
    )
    print(f"[export] {info['path']} via {info['exporter']}")
    for note in info.get("notes") or []:
        print(f"[export] note: {note}")

    if not args.skip_parity:
        sess = make_ort_session(info["path"], use_gpu=(device == "cuda"))
        print(f"[ort] providers={sess.get_providers()}")
        names = ["t", "x", "wa", "wr", "we", "prev_x", "prev_wa"]
        with torch.no_grad():
            # B=1 (plain call) and B=3 (CFG call), t across the trajectory.
            cases = [
                ("plain_t0", 1, 0.0, 1),
                ("plain_t05", 1, 0.5, 2),
                ("plain_t1", 1, 1.0, 3),
                ("cfg_t05", 3, 0.5, 4),
                ("cfg_t01", 3, 0.1, 5),
            ]
            for name, batch, t_val, seed in cases:
                args_in = sample_inputs(batch, t_val, seed, device)
                ref = module(*args_in)
                feed = {k: v.cpu().numpy() for k, v in zip(names, args_in)}
                ort_out = sess.run(None, feed)[0]
                check_parity(ref, ort_out, name=name, atol=args.atol, rtol=args.rtol)
                print(f"[probe:{name}] shape torch={tuple(ref.shape)} "
                      f"onnx={tuple(ort_out.shape)}")
    print("[done] fmt converter OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
