"""bench_perf.py - measure the speed/throughput + latency matrix on the GPU box.

Prints (and optionally writes) the matrix the pipeline is judged on:

  Speed and throughput
    - throughput fps (model-only)
    - end-to-end fps (encode, composite, stream)
    - real-time factor (fps / target fps)
    - concurrent streams per GPU at real-time
    - per-stream fps under load (does it hold at 4, 8, 16 streams?)

  Latency
    - time to first frame (TTFF)
    - audio-in to video-out
    - conversational latency (stop -> avatar)
    - p50 / p95 / p99 latency
    - frame-time jitter and dropped frames

The measurement itself lives in ``fast_float.core.perf_bench`` so the API can
run exactly the same benchmark (see ``POST /perf/benchmark``).

Usage (from the repo root, on the GPU box):
    python scripts/bench_perf.py
    python scripts/bench_perf.py --concurrency 1,4,8,16 --out report.md
    python scripts/bench_perf.py --nfe 30 --json report.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fast_float.core.perf_bench import (
    DEFAULT_AUDIO,
    DEFAULT_REF,
    BenchmarkConfig,
    cleanup_outputs,
    run_benchmark,
)


def build_agent(args: argparse.Namespace):
    """Build the inference agent exactly like the API's lifespan does."""
    from fast_float.core.inference_optimized import InferenceOptions, OptimizedInferenceAgent

    orig_argv = sys.argv.copy()
    sys.argv = [sys.argv[0]]
    try:
        opt = InferenceOptions().parse()
    finally:
        sys.argv = orig_argv

    opt.rank = 0
    opt.ngpus = int(os.getenv("NGPUS", "1"))
    opt.res_dir = args.out_dir
    opt.no_crop = False
    opt.optimization_mode = os.getenv("OPTIMIZATION_MODE", "all")
    opt.use_fp16 = os.getenv("USE_FP16", "true").lower() in ("true", "1", "yes")
    opt.n_diff_steps = 100
    opt.diff_schedule = "cosine"
    opt.diffusion_mode = "sample"
    return OptimizedInferenceAgent(opt), opt


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", default=DEFAULT_REF, help="reference image")
    ap.add_argument("--audio", default=DEFAULT_AUDIO, help="driving audio")
    ap.add_argument("--target-fps", type=float, default=25.0, help="fps the stream must sustain")
    ap.add_argument("--concurrency", default="1,4,8,16", help="comma-separated stream counts")
    ap.add_argument("--repeats", type=int, default=1, help="runs per level (median is kept)")
    ap.add_argument("--nfe", type=int, default=10)
    ap.add_argument("--seed", type=int, default=25)
    ap.add_argument("--emotion", default="S2E")
    ap.add_argument("--a-cfg", type=float, default=2.0)
    ap.add_argument("--r-cfg", type=float, default=1.0)
    ap.add_argument("--e-cfg", type=float, default=1.0)
    ap.add_argument("--out-dir", default="./results/bench")
    ap.add_argument("--out", default=None, help="write the markdown matrix here")
    ap.add_argument("--json", dest="json_out", default=None, help="write the raw numbers here")
    ap.add_argument("--keep-files", action="store_true", help="keep the produced mp4s")
    args = ap.parse_args(argv)

    levels = tuple(int(part) for part in str(args.concurrency).replace(" ", "").split(",") if part)
    if not levels:
        print("no concurrency levels given", file=sys.stderr)
        return 2
    args.out_dir = os.path.abspath(args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    config = BenchmarkConfig(
        ref=args.ref,
        audio=args.audio,
        target_fps=args.target_fps,
        concurrency=levels,
        repeats=args.repeats,
        nfe=args.nfe,
        seed=args.seed,
        emotion=args.emotion,
        a_cfg=args.a_cfg,
        r_cfg=args.r_cfg,
        e_cfg=args.e_cfg,
        out_dir=args.out_dir,
        keep_files=args.keep_files,
    )
    for path in (config.ref, config.audio):
        if not os.path.exists(path):
            print(f"missing input: {path}", file=sys.stderr)
            return 2

    print(f"[bench] building agent (nfe={config.nfe}, target_fps={config.target_fps}) ...")
    agent, _opt = build_agent(args)

    print("[bench] warmup (pays the one-time GPU auto-tune, excluded from the numbers) ...")
    matrix = run_benchmark(agent, config)

    for point in matrix.points:
        print(
            f"[bench] streams={point.concurrency:>3}  aggregate={point.aggregate_fps:8.2f} fps  "
            f"per-stream={point.per_stream_fps:8.2f} fps  worst={point.worst_stream_fps:8.2f} fps  "
            f"rtf={point.rtf:5.2f}x  model-share={point.gpu_saturation * 100:5.1f}%  "
            f"dropped={point.dropped}"
        )

    markdown = matrix.to_markdown()
    print()
    print(markdown)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(markdown)
        print(f"[bench] markdown written to {args.out}")
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(matrix.to_dict(), fh, indent=2)
        print(f"[bench] json written to {args.json_out}")

    cleanup_outputs(config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
