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

How the numbers are obtained:
  * model-only vs end-to-end comes from the pipeline's own ``timed(...)``
    stages: the model stages (ODE solve, encoders, decode) are summed
    separately from the encode/mux stages, and end-to-end is the wall time of
    the whole request (preprocess included).
  * the delivery leg (TTFF, per-frame latency, jitter, drops) is measured at
    the dispatcher, i.e. when a frame is handed to the transport - not in the
    browser, so add the network/encode time on top.
  * frames are counted with ffprobe on the produced mp4, so the numbers
    describe what the user actually gets.

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
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fast_float.core.perf_metrics import (
    ENCODE_STAGES,
    MODEL_STAGES,
    ConcurrencyPoint,
    FrameClock,
    PerfMatrix,
    StreamMetrics,
    fold_stage_seconds,
)
from fast_float.core.service import _probe_frame_count
from fast_float.streaming.dispatcher import FrameDispatcher
from fast_float.utils.timing import timing_run


class RecordingDispatcher(FrameDispatcher):
    """Dispatcher that records delivery timings into a :class:`FrameClock`."""

    def __init__(self, clock: FrameClock) -> None:
        super().__init__()
        self.clock = clock

    def dispatch(self, idx: int, frames, pts_ms: float) -> None:
        # The base dispatcher hands over batches (granularity "batch"); the
        # clock counts frames, so expand the batch into its frames.
        count = len(frames) if hasattr(frames, "__len__") else 1
        now = time.monotonic()
        step_ms = 1000.0 / self.clock.fps if self.clock.fps else 0.0
        for i in range(count):
            self.clock.on_frame(idx + i, pts_ms + i * step_ms, now)


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


def run_level(agent, args: argparse.Namespace, concurrency: int) -> ConcurrencyPoint:
    """Run ``concurrency`` streams at once and measure the level."""
    streams: list[StreamMetrics] = []
    lock = threading.Lock()

    def one(index: int) -> None:
        out_path = os.path.join(args.out_dir, f"bench_c{concurrency}_{index:02d}.mp4")
        clock = FrameClock(fps=args.target_fps, start_s=time.monotonic())
        dispatcher = RecordingDispatcher(clock)
        started = time.monotonic()
        try:
            agent.run_inference(
                res_video_path=out_path,
                ref_path=args.ref,
                audio_path=args.audio,
                a_cfg_scale=args.a_cfg,
                r_cfg_scale=args.r_cfg,
                e_cfg_scale=args.e_cfg,
                emo=args.emotion,
                nfe=args.nfe,
                no_crop=False,
                seed=args.seed + index,
                verbose=False,
                dispatcher=dispatcher,
            )
        finally:
            wall = time.monotonic() - started
            dispatcher.shutdown(timeout=10.0)

        frames = _probe_frame_count(out_path)
        stats = dispatcher.stats()
        dropped = max(
            clock.dropped,
            int(stats.get("dropped_overflow", 0)) + int(stats.get("dropped_lag", 0)),
        )
        with lock:
            streams.append(
                StreamMetrics(
                    name=f"c{concurrency}-{index}",
                    target_fps=args.target_fps,
                    frames=frames,
                    dropped=dropped,
                    wall_seconds=wall,
                    clock=clock,
                )
            )

    level_started = time.monotonic()
    with timing_run(f"bench-c{concurrency}") as collector, ThreadPoolExecutor(max_workers=concurrency) as pool:
        list(pool.map(one, range(concurrency)))
    wall = time.monotonic() - level_started

    totals = collector.totals()
    return ConcurrencyPoint(
        concurrency=concurrency,
        streams=streams,
        wall_seconds=wall,
        model_seconds=fold_stage_seconds(totals, MODEL_STAGES),
        encode_seconds=fold_stage_seconds(totals, ENCODE_STAGES),
    )


def median_point(points: list[ConcurrencyPoint]) -> ConcurrencyPoint:
    """Pick the median run of a repeated level (by wall time)."""
    if len(points) == 1:
        return points[0]
    ordered = sorted(points, key=lambda p: p.wall_seconds)
    return ordered[len(ordered) // 2]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", default="assets/sam_altman_512x512.jpg", help="reference image")
    ap.add_argument("--audio", default="assets/aud-sample-vs-1.wav", help="driving audio")
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

    levels = [int(part) for part in str(args.concurrency).replace(" ", "").split(",") if part]
    if not levels:
        print("no concurrency levels given", file=sys.stderr)
        return 2
    args.out_dir = os.path.abspath(args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    for path in (args.ref, args.audio):
        if not os.path.exists(path):
            print(f"missing input: {path}", file=sys.stderr)
            return 2

    print(f"[bench] building agent (nfe={args.nfe}, target_fps={args.target_fps}) ...")
    agent, _opt = build_agent(args)

    print("[bench] warmup (pays the one-time GPU auto-tune, excluded from the numbers) ...")
    warm_path = os.path.join(args.out_dir, "bench_warmup.mp4")
    agent.run_inference(
        res_video_path=warm_path,
        ref_path=args.ref,
        audio_path=args.audio,
        a_cfg_scale=args.a_cfg,
        r_cfg_scale=args.r_cfg,
        e_cfg_scale=args.e_cfg,
        emo=args.emotion,
        nfe=args.nfe,
        no_crop=False,
        seed=args.seed,
        verbose=False,
    )
    if not args.keep_files and os.path.exists(warm_path):
        os.remove(warm_path)

    matrix = PerfMatrix(target_fps=args.target_fps)
    for level in levels:
        runs = [run_level(agent, args, level) for _ in range(max(1, args.repeats))]
        point = median_point(runs)
        if level == min(levels) and len(point.streams) == 1:
            # Single stream: the collector's stage totals belong to it, so the
            # model/encode split (and the model-only fps) are exact.
            stream = point.streams[0]
            stream.model_seconds = point.model_seconds
            stream.encode_seconds = point.encode_seconds
            stream.other_seconds = max(0.0, stream.wall_seconds - point.model_seconds - point.encode_seconds)
            matrix.baseline = stream
        matrix.points.append(point)
        print(
            f"[bench] streams={level:>3}  aggregate={point.aggregate_fps:8.2f} fps  "
            f"per-stream={point.per_stream_fps:8.2f} fps  worst={point.worst_stream_fps:8.2f} fps  "
            f"rtf={point.rtf:5.2f}x  model-share={point.gpu_saturation * 100:5.1f}%  "
            f"dropped={point.dropped}"
        )

    if matrix.baseline is not None:
        base = matrix.baseline
        ttff = base.ttff_ms
        base.audio_in_to_video_out_ms = ttff
        base.conversational_latency_ms = ttff

    matrix.notes = [
        (
            "model-only fps is the single-stream baseline: the ODE solve, encoders and "
            "decode, with encode/mux excluded. Under concurrency those stages overlap, "
            "so the sweep reports aggregate fps and the model time share instead."
        ),
        (
            "end-to-end fps is frames (ffprobe on the produced mp4) over the wall time of "
            "the whole request, including preprocess, encode, mux and write."
        ),
        (
            "TTFF and the per-frame latency are measured at the dispatcher (transport "
            "handoff). Browser-side latency adds the network and the WebRTC encode."
        ),
        (
            "offline mode: the whole audio is present at the trigger, so audio-in to "
            "video-out and the conversational latency both equal TTFF. A live TTS mode "
            "must measure them from the first/last audio chunk instead."
        ),
        "the warmup run (one-time GPU auto-tune) is excluded from every number.",
    ]

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

    if not args.keep_files:
        for name in os.listdir(args.out_dir):
            if name.startswith("bench_c") and name.endswith(".mp4"):
                with suppress(OSError):
                    os.remove(os.path.join(args.out_dir, name))
    return 0


if __name__ == "__main__":
    sys.exit(main())
