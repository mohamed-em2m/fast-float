"""Convert the FMT single ODE step (unit 2) to ONNX.

Exports ``FlowMatchingTransformer.forward`` at CFG-triplet batch (3B) with
canonical shapes. Python keeps cat/chunk/CFG-combine + prev-chaining + the
ODE loop — the same recipe the runtime will use, verified here in numpy.

Canonical per-step shapes (B = base batch, Lc = clip frames, Lp = prev):
  t (3B,) | x,wa (3B,Lc,512) | wr (3B,1,512) | we (3B,1,7)
  | prev_x,prev_wa (3B,Lp,512) -> out (3B,Lp+Lc,512)

L is static per config (pos_embed + alignment_mask are fixed buffers), so
only the batch axis is dynamic. Prev tensors are always passed (zeros for
chunk 0) — matching deployed behavior and avoiding Optional ONNX inputs.
"""
from __future__ import annotations

import argparse
import os
from types import SimpleNamespace

import torch
import torch.nn as nn

from fast_float.models.float.FMT import FlowMatchingTransformer
from fast_float.onnx.export import export_module, make_ort_session
from fast_float.onnx.parity import benchmark, check_parity

FMT_PREFIX = "fmt."
DIM_W = 512
DIM_E = 7


def default_fmt_opt(clip_frames: int = 50, prev_frames: int = 10) -> SimpleNamespace:
    """opt subset FMT needs, matching base_options.py deployed defaults."""
    return SimpleNamespace(
        wav2vec_sec=2.0,
        fps=25.0,
        num_prev_frames=prev_frames,
        dim_w=DIM_W,
        dim_a=DIM_W,
        dim_e=DIM_E,
        dim_h=1024,
        mlp_ratio=4.0,
        fmt_depth=8,
        num_heads=8,
        attention_window=2,
        audio_dropout_prob=0.1,
        ref_dropout_prob=0.1,
        emotion_dropout_prob=0.1,
        rank=0 if torch.cuda.is_available() else "cpu",
    )


class FMTStepExport(nn.Module):
    """Single ODE step at triplet batch. All CFG cats pre-built by caller."""

    def __init__(self, fmt: FlowMatchingTransformer):
        super().__init__()
        self.fmt = fmt

    def forward(self, t, x, wa, wr, we, prev_x, prev_wa):
        return self.fmt.forward(t, x, wa, wr, we, prev_x, prev_wa, train=False)


class _PlainNorm(nn.Module):
    """LayerNorm(affine=False) in pure elementwise ops (export-safe).

    The exporter maps affine-free nn.LayerNorm to a LayerNormalization node
    with empty scale/bias inputs, which the ONNX checker rejects. This
    formulation uses only ReduceMean/Sub/Mul/Div/Sqrt — supported everywhere.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        m = x.mean(dim=-1, keepdim=True)
        v = ((x - m) ** 2).mean(dim=-1, keepdim=True)
        return (x - m) / torch.sqrt(v + self.eps)


def cfv_combine_torch(out_3b, lp: int, a_scale: float, e_scale: float):
    """Runtime recipe: chunk triplet, CFV-combine, drop prev frames."""
    uncond, all_cond, audio_only = torch.chunk(out_3b, chunks=3, dim=0)
    cfv = uncond + a_scale * (audio_only - uncond) + e_scale * (all_cond - audio_only)
    return cfv[:, lp:]


def build_fmt_module(
    ckpt_path: str, clip_frames: int, prev_frames: int, device: torch.device
) -> tuple[FMTStepExport, FlowMatchingTransformer, SimpleNamespace]:
    opt = default_fmt_opt(clip_frames, prev_frames)
    # Honour explicit frame overrides by resizing the derived constants.
    opt.num_prev_frames = prev_frames
    opt.wav2vec_sec = clip_frames / float(opt.fps)
    fmt = FlowMatchingTransformer(opt).to(device).eval()

    # The classic exporter cannot translate aten::scaled_dot_product_attention
    # (dynamo on modern torch can, but the fallback path must work too), so
    # the export — and its parity reference — use the math-equivalent manual
    # attention path. Same weights; only kernel summation order differs.
    with torch.no_grad():
        p0 = probe_batch(1, clip_frames, prev_frames, device, 7)
        t0 = to_triplet(p0)
        out_fused = fmt.forward(
            t0["t"], t0["x"], t0["wa"], t0["wr"], t0["we"],
            t0["prev_x"], t0["prev_wa"], train=False,
        )
    had_fused = False
    for blk in fmt.blocks:
        had_fused = had_fused or blk.attn.fused_attn
        blk.attn.fused_attn = False
    with torch.no_grad():
        out_manual = fmt.forward(
            t0["t"], t0["x"], t0["wa"], t0["wr"], t0["we"],
            t0["prev_x"], t0["prev_wa"], train=False,
        )
    diff = (out_fused - out_manual).abs().max().item()
    print(f"[model] fused_attn was={had_fused} -> manual (export-safe); "
          f"sdpa-vs-manual max_abs={diff:.3e}")

    # Affine-free LayerNorm -> _PlainNorm (same equivalence proof pattern).
    n_sub = 0
    for blk in fmt.blocks:
        blk.norm1 = _PlainNorm(blk.norm1.eps)
        blk.norm2 = _PlainNorm(blk.norm2.eps)
        n_sub += 2
    fmt.decoder.norm_final = _PlainNorm(fmt.decoder.norm_final.eps)
    n_sub += 1
    with torch.no_grad():
        out_plain = fmt.forward(
            t0["t"], t0["x"], t0["wa"], t0["wr"], t0["we"],
            t0["prev_x"], t0["prev_wa"], train=False,
        )
    ndiff = (out_manual - out_plain).abs().max().item()
    print(f"[model] LayerNorm(affine=False) x{n_sub} -> _PlainNorm; "
          f"ln-vs-plain max_abs={ndiff:.3e}")

    n_total = sum(p.numel() for p in fmt.parameters())
    print(
        f"[model] FMT depth={opt.fmt_depth} heads={opt.num_heads} "
        f"Lc={clip_frames} Lp={prev_frames} params={n_total / 1e6:.1f}M"
    )
    if os.path.exists(ckpt_path):
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        from collections import Counter

        tops = Counter(k.split(".")[0] for k in sd.keys())
        print(f"[weights] ckpt top-level prefixes: {dict(tops.most_common(10))}")
        sub = {
            k[len(FMT_PREFIX):]: v
            for k, v in sd.items()
            if k.startswith(FMT_PREFIX)
        }
        if not sub:
            print(
                "[weights] WARN: no fmt.* keys in ckpt — exporting randomly "
                "initialized FMT (parity still valid, weights differ from deploy)"
            )
        missing, unexpected = fmt.load_state_dict(sub, strict=False)
        print(
            f"[weights] overlaid {len(sub)} tensors from {ckpt_path} "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )
        fmt.to(device)
    else:
        print(
            f"[weights] WARN: {ckpt_path} not found — exporting randomly "
            "initialized FMT (parity still valid, weights differ from deploy)"
        )
    return FMTStepExport(fmt).eval(), fmt, opt


def probe_batch(
    b: int, clip: int, prev: int, device: torch.device, seed: int
) -> dict:
    g = torch.Generator(device="cpu").manual_seed(seed)
    kw = {"generator": g}
    x = torch.randn(b, clip, DIM_W, **kw)
    wa = torch.randn(b, clip, DIM_W, **kw)
    wr = torch.randn(b, 1, DIM_W, **kw)
    we = torch.randn(b, 1, DIM_E, **kw).softmax(-1)
    prev_x = torch.randn(b, prev, DIM_W, **kw)
    prev_wa = torch.randn(b, prev, DIM_W, **kw)
    t = torch.full((1,), 0.35 + 0.05 * (seed % 5))
    return {
        "t": t.to(device),
        "x": x.to(device),
        "wa": wa.to(device),
        "wr": wr.to(device),
        "we": we.to(device),
        "prev_x": prev_x.to(device),
        "prev_wa": prev_wa.to(device),
    }


def to_triplet(p: dict) -> dict:
    b3 = p["x"].shape[0] * 3
    t3 = p["t"].expand(b3).contiguous()
    null_wa = torch.zeros_like(p["wa"])
    null_we = torch.zeros_like(p["we"])
    return {
        "t": t3,
        "x": p["x"].repeat(3, 1, 1),
        "wa": torch.cat([null_wa, p["wa"], p["wa"]], 0),
        "wr": p["wr"].repeat(3, 1, 1),
        "we": torch.cat([null_we, p["we"], null_we], 0),
        "prev_x": p["prev_x"].repeat(3, 1, 1),
        "prev_wa": p["prev_wa"].repeat(3, 1, 1),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Convert FMT single ODE step to ONNX")
    ap.add_argument("--ckpt", default="./checkpoints/float.pth")
    ap.add_argument("--out", default="./onnx/fmt_step_v1.onnx")
    ap.add_argument("--clip-frames", type=int, default=50)
    ap.add_argument("--prev-frames", type=int, default=10)
    ap.add_argument("--batches", default="1,2")
    ap.add_argument("--a-cfg", type=float, default=2.0)
    ap.add_argument("--e-cfg", type=float, default=1.0)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--atol", type=float, default=2e-4)
    ap.add_argument("--rtol", type=float, default=2e-4)
    ap.add_argument("--skip-parity", action="store_true")
    ap.add_argument("--bench-iters", type=int, default=30)
    ap.add_argument("--cpu-ort", action="store_true",
                    help="allow CPU ORT session on a CUDA box (dev only)")
    args = ap.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[env] device={device}")
    module, fmt, _ = build_fmt_module(
        args.ckpt, args.clip_frames, args.prev_frames, device
    )
    batches = [int(s) for s in args.batches.split(",") if s.strip()]

    # Export with B=1 triplet; dynamic batch axis proves B=2 in parity.
    # Export on CPU: the tracer/constant-folder trips on mixed devices when
    # the module sits on CUDA. The graph is device-agnostic; parity + bench
    # still run torch on the real device below.
    fmt.to("cpu")
    cpu = torch.device("cpu")
    sample = to_triplet(probe_batch(1, args.clip_frames, args.prev_frames, cpu, 0))
    order = ["t", "x", "wa", "wr", "we", "prev_x", "prev_wa"]
    info = export_module(
        module,
        [sample[k] for k in order],
        args.out,
        input_names=order,
        output_names=["step_out"],
        dynamic_axes={k: {0: "batch3"} for k in order} | {"step_out": {0: "batch3"}},
        opset=args.opset,
    )
    print(f"[export] {info['path']} via {info['exporter']}")
    for note in info.get("notes") or []:
        print(f"[export] note: {note}")
    fmt.to(device)

    if not args.skip_parity:
        sess = make_ort_session(
            info["path"],
            use_gpu=torch.cuda.is_available() and not args.cpu_ort,
            require_gpu=torch.cuda.is_available() and not args.cpu_ort,
        )
        print(f"[ort] providers={sess.get_providers()}")
        with torch.no_grad():
            for bi, b in enumerate(batches):
                p = probe_batch(b, args.clip_frames, args.prev_frames, device, 100 + bi)
                tri = to_triplet(p)
                # Deployed path: forward_with_cfv with pre-built cats (steady state).
                ref = fmt.forward_with_cfv(
                    t=p["t"],
                    x=p["x"],
                    wa=p["wa"],
                    wr=p["wr"],
                    we=p["we"],
                    prev_x=p["prev_x"],
                    prev_wa=p["prev_wa"],
                    a_cfg_scale=args.a_cfg,
                    r_cfg_scale=1.0,
                    e_cfg_scale=args.e_cfg,
                    _audio_cat=tri["wa"],
                    _ref_cat=tri["wr"],
                    _emotion_cat=tri["we"],
                    _prev_x_cat=tri["prev_x"],
                    _prev_wa_cat=tri["prev_wa"],
                )
                ort_out = sess.run(
                    None, {k: tri[k].detach().cpu().numpy() for k in order}
                )[0]
                # THE parity: torch full-step output vs the runtime recipe
                # (ORT triplet -> chunk + CFV combine -> drop prev frames).
                combined = cfv_combine_torch(
                    torch.from_numpy(ort_out), args.prev_frames,
                    args.a_cfg, args.e_cfg,
                )
                check_parity(
                    ref[:, args.prev_frames:],
                    combined,
                    name=f"fmt_b{b}",
                    atol=args.atol,
                    rtol=args.rtol,
                )
                if args.bench_iters > 0:
                    np_inputs = {k: tri[k].detach().cpu().numpy() for k in order}
                    benchmark(
                        lambda: fmt.forward_with_cfv(
                            t=p["t"], x=p["x"], wa=p["wa"], wr=p["wr"],
                            we=p["we"], prev_x=p["prev_x"], prev_wa=p["prev_wa"],
                            a_cfg_scale=args.a_cfg, r_cfg_scale=1.0,
                            e_cfg_scale=args.e_cfg,
                            _audio_cat=tri["wa"], _ref_cat=tri["wr"],
                            _emotion_cat=tri["we"], _prev_x_cat=tri["prev_x"],
                            _prev_wa_cat=tri["prev_wa"],
                        ),
                        lambda: sess.run(None, np_inputs),
                        name=f"fmt_b{b}",
                        iters=args.bench_iters,
                    )
    print("[done] fmt converter OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
