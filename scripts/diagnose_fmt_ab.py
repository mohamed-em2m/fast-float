"""A/B the deployed FMT ONNX step against torch on the GPU box.

The converter proves the *exported graph* matches torch when fed host numpy via
``sess.run`` (by name, on clamped randn inputs, batch 1/3).  Production differs
in three ways it never touches:

  1. the IOBinding device path (``run_ort_iobinding``) instead of ``sess.run``;
  2. *real* inputs (real latents/weights, one-hot emotion, real prev frames);
  3. the CFG call: 3 * chunk_batch samples, then ``torch.chunk`` + blend.

This script isolates all three.  It builds the production CFG batch, runs each
ODE step through torch and through the ONNX backend, and reports the error
*per CFG branch* (so "emotion is ignored" is distinguishable from "audio is
wrong"), for the host path and the IOBinding path separately.

Usage (from the repo root, GPU box):
    python scripts/diagnose_fmt_ab.py --ckpt ./checkpoints/float.pth
    python scripts/diagnose_fmt_ab.py --onnx ./onnx/fmt_step_v1.onnx --no-trt
    python scripts/diagnose_fmt_ab.py --steps 30 --chunk-batch 4 --prev random
"""

from __future__ import annotations

import argparse
import copy
import sys
from types import SimpleNamespace

import torch

from fast_float.onnx.convert_fmt import CLIP_FRAMES, PREV_FRAMES, build_fmt

LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Counts real backend calls vs exceptions. ``_forward_step`` swallows a backend
# exception and silently runs torch, which would make a broken graph look fine.
_COUNT = {"ok": 0, "err": 0}


def wrap_run_step(backend):
    """Wrap ``backend.run_step`` so fallbacks cannot hide a failing graph."""
    real = backend.run_step

    def counted(*a, **k):
        try:
            out = real(*a, **k)
        except Exception:
            _COUNT["err"] += 1
            raise
        _COUNT["ok"] += 1
        return out

    backend.run_step = counted


def make_backend(fmt, onnx_path: str, use_trt: bool | None, max_batch: int | None):
    """Same wiring the FMT constructor uses (see FMT.py, __init__)."""
    from fast_float.runtimes.onnx.backends import FMTStepBackend

    n_clip = int(fmt.opt.wav2vec_sec * fmt.opt.fps)
    n_prev = int(fmt.opt.num_prev_frames)
    dim_w = int(fmt.opt.dim_w)
    return FMTStepBackend(
        onnx_path,
        opt_shapes={
            "x": [1, n_clip, dim_w],
            "wa": [1, n_clip, dim_w],
            "wr": [1, 1, dim_w],
            "we": [1, 1, 7],
            "prev_x": [1, n_prev, dim_w],
            "prev_wa": [1, n_prev, dim_w],
        },
        max_batch=max_batch,
        use_trt=use_trt,
    )


def stats(ref: torch.Tensor, got: torch.Tensor) -> tuple[float, float, float]:
    d = (ref.float() - got.float()).abs()
    cos = torch.nn.functional.cosine_similarity(ref.flatten(), got.flatten(), dim=0).item()
    return d.max().item(), d.mean().item(), cos


def build_cats(x, wa, wr, we, prev_x, prev_wa):
    """Exactly the sampler's pre-built cats (sampler.py, _cfg_active block)."""
    null_wa = torch.zeros_like(wa)
    null_we = torch.zeros_like(we)
    return (
        torch.cat([null_wa, wa, wa], dim=0),  # audio:    [uncond, cond, cond]
        torch.cat([wr, wr, wr], dim=0),  # ref:      constant
        torch.cat([null_we, we, null_we], dim=0),  # emotion:  [uncond, cond, uncond]
        torch.cat([prev_x, prev_x, prev_x], dim=0),
        torch.cat([prev_wa, prev_wa, prev_wa], dim=0),
    )


def blend(model_output, a_cfg: float, e_cfg: float):
    """The deployed blend (FMT.py, forward_with_cfv)."""
    uncond, all_cond, audio_uncond_emotion = torch.chunk(model_output, chunks=3, dim=0)
    return uncond + a_cfg * (audio_uncond_emotion - uncond) + e_cfg * (all_cond - audio_uncond_emotion)


def run_ab(fmt_eager, fmt_onnx, inputs, steps: int, a_cfg: float, e_cfg: float, tag: str):
    x0, wa, wr, we, prev_x, prev_wa = inputs
    time = torch.linspace(0, 1, steps, device=x0.device)

    z_t = x0.clone()
    z_o = x0.clone()
    _COUNT["ok"] = 0
    _COUNT["err"] = 0
    shared_err, per_branch = [], []
    for i in range(len(time) - 1):
        t = time[i].unsqueeze(0)
        # ---- torch reference on the shared trajectory ----
        x_cat = torch.cat([z_t, z_t, z_t], dim=0)
        audio_cat, ref_cat, emotion_cat, prev_x_cat, prev_wa_cat = build_cats(z_t, wa, wr, we, prev_x, prev_wa)
        with torch.no_grad():
            ref3 = fmt_eager.forward(t, x_cat, audio_cat, ref_cat, emotion_cat, prev_x_cat, prev_wa_cat, train=False)
            got3 = fmt_onnx._forward_step(t, x_cat, audio_cat, ref_cat, emotion_cat, prev_x_cat, prev_wa_cat)

        ma, mn, cos = stats(ref3, got3)
        shared_err.append((t.item(), ma, mn, cos))
        ru, rc, ra = torch.chunk(ref3, 3, dim=0)
        gu, gc, ga = torch.chunk(got3, 3, dim=0)
        per_branch.append((t.item(), stats(ru, gu), stats(rc, gc), stats(ra, ga)))

        # ---- independent trajectories (integration divergence) ----
        v_t = blend(ref3, a_cfg, e_cfg)
        v_o = blend(got3, a_cfg, e_cfg)
        dt = (time[i + 1] - time[i]).item()
        z_t = z_t + dt * v_t[:, PREV_FRAMES:]
        z_o = z_o + dt * v_o[:, PREV_FRAMES:]

    print(f"\n=== {tag} : per-step raw CFG-batch error (torch vs onnx) ===")
    print(f"{'t':>6} {'max_abs':>10} {'mean_abs':>10} {'cosine':>10}   (branches: uncond / all_cond / audio_uncond)")
    for (t, ma, mn, cos), (_, s_uncond, s_cond, s_audio) in zip(shared_err, per_branch, strict=False):
        print(f"{t:6.3f} {ma:10.3e} {mn:10.3e} {cos:10.7f}   {s_uncond[2]:.7f} / {s_cond[2]:.7f} / {s_audio[2]:.7f}")
    fin = stats(z_t, z_o)
    print(
        f"[{tag}] final latent after {steps - 1} Euler steps: "
        f"max_abs={fin[0]:.3e} mean_abs={fin[1]:.3e} cosine={fin[2]:.8f}"
    )
    worst = max(shared_err, key=lambda r: r[2])
    print(f"[{tag}] worst step mean_abs={worst[2]:.3e} at t={worst[0]:.3f}")
    # ``_forward_step`` falls back to torch on any backend exception, which would
    # make a broken graph look identical to the reference. Surface that.
    print(
        f"[{tag}] backend calls: ok={_COUNT['ok']} raised={_COUNT['err']} "
        f"(backend still engaged: {fmt_onnx._onnx_backend is not None})"
    )
    if _COUNT["err"] or fmt_onnx._onnx_backend is None:
        print(f"[{tag}] WARNING: results above may be torch-vs-torch (fallback), not the graph.")
    # Branch 1 is all_cond (the only branch carrying the emotion vector).
    emo_err = max(r[2][1] for r in per_branch)
    print(f"[{tag}] worst all_cond(emotion) mean_abs={emo_err:.3e}  <- 'emotion is ignored' shows up here")
    return shared_err, per_branch, fin


def emotion_sensitivity(fmt_eager, fmt_onnx, x0, wa, wr, prev_x, prev_wa, a_cfg, e_cfg, t_val=0.5):
    """Does the graph respond to emotion the way torch does, for each label?"""
    print("\n=== emotion sensitivity (all_cond branch, one-hot per label) ===")
    B = x0.shape[0]
    t = torch.tensor([t_val], device=x0.device)
    x_cat = torch.cat([x0, x0, x0], dim=0)
    base = None
    for idx, label in enumerate(LABELS):
        we = torch.zeros(B, 1, 7, device=x0.device)
        we[:, :, idx] = 1.0
        audio_cat, ref_cat, emotion_cat, prev_x_cat, prev_wa_cat = build_cats(x0, wa, wr, we, prev_x, prev_wa)
        with torch.no_grad():
            ref3 = fmt_eager.forward(t, x_cat, audio_cat, ref_cat, emotion_cat, prev_x_cat, prev_wa_cat, train=False)
            got3 = fmt_onnx._forward_step(t, x_cat, audio_cat, ref_cat, emotion_cat, prev_x_cat, prev_wa_cat)
        _, rc, _ = torch.chunk(ref3, 3, dim=0)
        _, gc, _ = torch.chunk(got3, 3, dim=0)
        if base is None:
            base = rc.clone()
            torch_delta = 0.0
        else:
            torch_delta = (rc - base).abs().mean().item()
        ma, mn, cos = stats(rc, gc)
        print(
            f"  {label:>9}: torch_sensitivity(delta vs first)={torch_delta:.3e}  "
            f"onnx-vs-torch max_abs={ma:.3e} mean_abs={mn:.3e} cos={cos:.7f}"
        )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="./checkpoints/float.pth")
    ap.add_argument("--onnx", default="./onnx/fmt_step_v1.onnx")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--chunk-batch", type=int, default=1, help="chunk_batch_size (CFG batch = 3x)")
    ap.add_argument("--prev", choices=["zero", "random"], default="zero")
    ap.add_argument("--a-cfg", type=float, default=2.0, help="deployed --a_cfg_scale default")
    ap.add_argument("--e-cfg", type=float, default=1.0, help="deployed --e_cfg_scale default")
    ap.add_argument("--no-trt", action="store_true", help="build the ONNX session without TRT (fp32 CUDA EP)")
    ap.add_argument("--max-batch", type=int, default=None)
    ap.add_argument("--force-onnx", action="store_true", help="build the backend even if the yaml says eager")
    ap.add_argument("--skip-emotion-sweep", action="store_true")
    args = ap.parse_args(argv)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: no CUDA device; the IOBinding path and TRT will not be exercised.")

    ns = SimpleNamespace(
        ckpt=args.ckpt,
        prev_frames=PREV_FRAMES,
        dim_h=1024,
        mlp_ratio=4.0,
        fmt_depth=8,
        num_heads=8,
    )
    fmt = build_fmt(ns).to(device)
    # The torch reference must be the *deployed* model (same attention path
    # the application runs). Forcing the manual branch here compared the
    # graph against a different model, which is exactly the blind spot this
    # probe is meant to close.
    fmt_eager = copy.deepcopy(fmt)
    fmt_eager._onnx_backend = None
    fmt_eager._mode = "eager"
    print(f"yaml-wired backend: {'yes' if fmt._onnx_backend is not None else 'no'}")

    backend = fmt._onnx_backend
    if backend is None and args.force_onnx:
        backend = make_backend(fmt, args.onnx, not args.no_trt, args.max_batch)
    if backend is None:
        print("No ONNX backend wired (config says eager and --force-onnx not given). Nothing to compare.")
        return 2
    if args.no_trt:
        backend = make_backend(fmt, args.onnx, False, args.max_batch)
    fmt._onnx_backend = backend
    wrap_run_step(backend)
    print(f"onnx providers: {backend.providers}")
    print(f"batch cap: {backend._batch_cap} (None = no TRT split)")

    g = torch.Generator(device=device).manual_seed(7)
    B = args.chunk_batch
    x0 = torch.randn(B, CLIP_FRAMES, 512, generator=g, device=device)
    wa = torch.randn(B, CLIP_FRAMES, 512, generator=g, device=device) * 0.5
    wr = torch.randn(B, 1, 512, generator=g, device=device)
    we = torch.zeros(B, 1, 7, device=device)
    we[:, :, 3] = 1.0  # "happy"
    if args.prev == "zero":
        prev_x = torch.zeros(B, PREV_FRAMES, 512, device=device)
        prev_wa = torch.zeros(B, PREV_FRAMES, 512, device=device)
    else:
        prev_x = torch.randn(B, PREV_FRAMES, 512, generator=g, device=device)
        prev_wa = torch.randn(B, PREV_FRAMES, 512, generator=g, device=device)
    inputs = (x0, wa, wr, we, prev_x, prev_wa)

    before = backend.stats.io_bindings
    run_ab(fmt_eager, fmt, inputs, args.steps, args.a_cfg, args.e_cfg, tag="IOBinding/auto")
    used_io = backend.stats.io_bindings - before
    print(f"[io] IOBinding calls used by the run above: {used_io}")

    # Now force the host path on the *same* graph and compare to torch directly.
    backend._io_failed = True
    before_host = backend.stats.io_bindings
    run_ab(fmt_eager, fmt, inputs, args.steps, args.a_cfg, args.e_cfg, tag="host(sess.run) forced")
    print(f"[io] IOBinding calls used by the forced-host run: {backend.stats.io_bindings - before_host}")

    if not args.skip_emotion_sweep:
        backend._io_failed = not used_io
        emotion_sensitivity(fmt_eager, fmt, x0, wa, wr, prev_x, prev_wa, args.a_cfg, args.e_cfg)

    print(
        "\nRead the result:\n"
        "  - if the host-forced run matches torch but the IOBinding run does not -> the\n"
        "    deployed-only IOBinding path is the bug (graph is fine).\n"
        "  - if 'all_cond(emotion)' error is large while uncond/audio are fine -> the\n"
        "    graph's emotion conditioning is wrong ('emotion goes silent').\n"
        "  - if every branch is ~1e-3 mean_abs in both paths -> the graph and plumbing are\n"
        "    fine and the drift is TRT fp16/real-input amplitude (re-run with --no-trt)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
