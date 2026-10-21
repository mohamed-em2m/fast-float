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

The exported graph must be the *deployed* graph: the module is traced
exactly as the application builds it, and the parity gate compares the
ONNX output against a snapshot of that deployed model — never against the
module that was traced. Comparing a graph with itself proves nothing,
which is how an inverted attention mask (a different model: per-step
cosine 0.94) once passed every gate and shipped.

Usage (from repo root):
    python -m fast_float.onnx.convert_fmt --out ./onnx/fmt_step_v1.onnx
"""

from __future__ import annotations

import argparse
import copy
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

from fast_float.models.float.FMT import FlowMatchingTransformer
from fast_float.onnx.export import (
    FP16_MAX_MEAN,
    FP16_MIN_COSINE,
    add_ort_args,
    export_module,
    fp16_tol,
    make_session_from_args,
)
from fast_float.onnx.parity import check_parity

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
        return self.fmt.forward(t, x, wa, wr, we, prev_x, prev_wa, train=False)


def build_fmt(args) -> FlowMatchingTransformer:
    # ``opt.rank`` is a GPU *index* (deployed wiring sets ``opt.rank = 0``), not
    # a device string: FMT builds ``torch.device(f"cuda:{opt.rank}")``. Passing
    # "cuda"/"cpu" here produced the invalid "cuda:cuda".
    rank = torch.cuda.current_device() if torch.cuda.is_available() else 0
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
        rank=rank,
        audio_dropout_prob=0.1,
        ref_dropout_prob=0.1,
        emotion_dropout_prob=0.1,
    )
    fmt = FlowMatchingTransformer(opt)
    if os.path.exists(args.ckpt):
        sd = torch.load(args.ckpt, map_location="cpu", weights_only=True)
        sub = {k[len(FMT_PREFIX) :]: v for k, v in sd.items() if k.startswith(FMT_PREFIX)}
        if not sub:
            print(f"[weights] WARN: no {FMT_PREFIX}* keys in {args.ckpt}")
        else:
            missing, unexpected = fmt.load_state_dict(sub, strict=False)
            print(
                f"[weights] fmt: {len(sub)} tensors from {args.ckpt} "
                f"(missing={len(missing)}, unexpected={len(unexpected)})"
            )
            if missing or unexpected:
                print(f"[weights] WARN: fmt weight mismatch: missing={len(missing)} unexpected={len(unexpected)}")
    else:
        print(f"[weights] WARN: {args.ckpt} not found — random init (parity still valid)")
    return fmt.eval()


def sample_inputs(batch: int, t_val: float, seed: int, device: str | torch.device):
    g = torch.Generator().manual_seed(seed)
    kw = dict(dtype=torch.float32)
    t = torch.tensor([t_val], **kw)
    x = torch.randn(batch, CLIP_FRAMES, 512, generator=g, **kw).clamp(-1, 1)
    wa = torch.randn(batch, CLIP_FRAMES, 512, generator=g, **kw).clamp(-1, 1)
    wr = torch.randn(batch, 1, 512, generator=g, **kw).clamp(-1, 1)
    we = torch.randn(batch, 1, 7, generator=g, **kw).clamp(-1, 1)
    prev_x = torch.randn(batch, PREV_FRAMES, 512, generator=g, **kw).clamp(-1, 1)
    prev_wa = torch.randn(batch, PREV_FRAMES, 512, generator=g, **kw).clamp(-1, 1)
    # `device` may be a str or a torch.device; `torch.device("cuda") == "cuda"`
    # is False, so normalise before testing (the selector passes a torch.device).
    if torch.device(device).type == "cuda":
        t, x, wa, wr, we, prev_x, prev_wa = (v.to(device) for v in (t, x, wa, wr, we, prev_x, prev_wa))
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
    ap.add_argument(
        "--bench-iters",
        type=int,
        default=30,
        help="warmed benchmark iters on the CFG case (0 = skip)",
    )
    add_ort_args(ap)
    args = ap.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    fmt = build_fmt(args).to(device)
    # Export the *deployed* graph: the attention implementation is whatever
    # ``use_fused_attn()`` selects, i.e. exactly what production runs. The
    # converter used to force the manual branch ("fused_attn = False"), but
    # that branch masked the opposite set of frames (see FMT.Attention), so
    # the artifact was a different model than the one that generated the
    # frames -- and the parity gate below could not see it, because it
    # compared the graph against the very module it had just exported.
    attn_mode = "unknown"
    if len(fmt.blocks) and hasattr(fmt.blocks[0], "attn"):
        attn_mode = "fused" if getattr(fmt.blocks[0].attn, "fused_attn", False) else "manual"
    print(f"[export] attention path: {attn_mode} (the deployed one)")
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
        # ``t`` is one shared scalar time value (shape [1], broadcast over the
        # batch inside the graph), not a batch-first input. Marking its dim
        # dynamic makes TensorRT freeze it at the profile's opt batch, so the
        # plain forward works but every CFG call (batch 3) dies with
        # "Dimensions with name s0 must be equal ... 1 != 3".
        shared_inputs=("t",),
        opset=args.opset,
    )
    print(f"[export] {info['path']} via {info['exporter']}")
    for note in info.get("notes") or []:
        print(f"[export] note: {note}")

    if not args.skip_parity:
        # The parity reference is a *snapshot of the deployed model*, taken
        # before the export, and never the module that was exported. Comparing
        # a graph against the module it was traced from proves nothing about
        # the model the application runs -- that blind spot is how a graph
        # with an inverted attention mask (a different model: cosine 0.94 per
        # step) passed every gate and shipped.
        deployed = copy.deepcopy(fmt)
        sess, fp16 = make_session_from_args(
            info["path"],
            args,
            opt_shapes={
                # ``t`` is exported static ([1], see shared_inputs above), so it
                # gets no profile entry: the TRT EP ignores/or rejects profile
                # dims for static inputs.
                "x": [1, CLIP_FRAMES, 512],
                "wa": [1, CLIP_FRAMES, 512],
                "wr": [1, 1, 512],
                "we": [1, 1, 7],
                "prev_x": [1, PREV_FRAMES, 512],
                "prev_wa": [1, PREV_FRAMES, 512],
            },
        )
        atol, rtol = fp16_tol(args.atol, fp16), fp16_tol(args.rtol, fp16)
        # The vector field is bounded/small-magnitude: TRT FP16 drift is
        # absolute (~9e-3 max, cosine 0.999997), so the elementwise gate can
        # never pass even though the field is correct. Add the same
        # functional-equivalence gate the emotion/refenc/audio units use.
        func = {"min_cosine": FP16_MIN_COSINE, "max_mean": FP16_MAX_MEAN} if fp16 else {}
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
                # Reference = the deployed torch model, never the traced module.
                ref = deployed.forward(*args_in, train=False)
                exported = module(*args_in)
                drift = (ref - exported).abs().max().item()
                if drift > 1e-4:
                    raise RuntimeError(
                        f"the exported module is not the deployed model "
                        f"(max_abs={drift:.3e}); the graph would not reproduce "
                        f"the frames the torch path generates"
                    )
                feed = {k: v.cpu().numpy() for k, v in zip(names, args_in, strict=False)}
                ort_out = sess.run(None, feed)[0]
                check_parity(ref, ort_out, name=name, atol=atol, rtol=rtol, **func)
                print(f"[probe:{name}] shape torch={tuple(ref.shape)} onnx={tuple(ort_out.shape)}")
            if args.bench_iters > 0:
                from fast_float.onnx.parity import benchmark

                bench_in = sample_inputs(3, 0.5, 99, device)
                bench_feed = {k: v.cpu().numpy() for k, v in zip(names, bench_in, strict=False)}
                benchmark(
                    lambda: module(*bench_in),
                    lambda: sess.run(None, bench_feed),
                    name="fmt_cfg_b3",
                    iters=args.bench_iters,
                )
    print("[done] fmt converter OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
