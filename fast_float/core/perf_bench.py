"""perf_bench.py - run the perf matrix against a live agent.

The benchmark reuses the *running* agent (the weights are loaded once) and has
to be serialized with live requests: "how many streams still run at real time"
is only meaningful when nothing else is using the GPU.

How the numbers are obtained:
  * model-only vs end-to-end comes from the pipeline's own ``timed(...)``
    stages: the model stages (ODE solve, encoders, decode) are summed
    separately from the encode/mux stages, and end-to-end is the wall time of
    the whole request (preprocess included).
  * the delivery leg (TTFF, per-frame latency, jitter, drops) is measured at
    the dispatcher, i.e. when a frame is handed to the transport.
  * frames are counted with ffprobe on the produced mp4, so the numbers
    describe what the user actually gets.
"""

from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

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

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REF = str(REPO_ROOT / "assets" / "sam_altman_512x512.jpg")
DEFAULT_AUDIO = str(REPO_ROOT / "assets" / "aud-sample-vs-1.wav")


@dataclass
class BenchmarkConfig:
    """What to measure."""

    ref: str = DEFAULT_REF
    audio: str = DEFAULT_AUDIO
    target_fps: float = 25.0
    concurrency: tuple[int, ...] = (1, 4, 8, 16)
    repeats: int = 1
    nfe: int = 10
    seed: int = 25
    emotion: str = "S2E"
    a_cfg: float = 2.0
    r_cfg: float = 1.0
    e_cfg: float = 1.0
    out_dir: str = "./results/bench"
    keep_files: bool = False
    warmup: bool = True


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


def run_level(agent, config: BenchmarkConfig, concurrency: int) -> ConcurrencyPoint:
    """Run ``concurrency`` streams at once and measure the level."""
    streams: list[StreamMetrics] = []
    lock = threading.Lock()

    def one(index: int) -> None:
        out_path = os.path.join(config.out_dir, f"bench_c{concurrency}_{index:02d}.mp4")
        clock = FrameClock(fps=config.target_fps, start_s=time.monotonic())
        dispatcher = RecordingDispatcher(clock)
        started = time.monotonic()
        try:
            agent.run_inference(
                res_video_path=out_path,
                ref_path=config.ref,
                audio_path=config.audio,
                a_cfg_scale=config.a_cfg,
                r_cfg_scale=config.r_cfg,
                e_cfg_scale=config.e_cfg,
                emo=config.emotion,
                nfe=config.nfe,
                no_crop=False,
                seed=config.seed + index,
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
                    target_fps=config.target_fps,
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


def _warmup(agent, config: BenchmarkConfig) -> None:
    """Pay the one-time GPU auto-tune outside the measured levels."""
    path = os.path.join(config.out_dir, "bench_warmup.mp4")
    agent.run_inference(
        res_video_path=path,
        ref_path=config.ref,
        audio_path=config.audio,
        a_cfg_scale=config.a_cfg,
        r_cfg_scale=config.r_cfg,
        e_cfg_scale=config.e_cfg,
        emo=config.emotion,
        nfe=config.nfe,
        no_crop=False,
        seed=config.seed,
        verbose=False,
    )
    if not config.keep_files and os.path.exists(path):
        os.remove(path)


def _notes() -> list[str]:
    return [
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


def run_benchmark(agent, config: BenchmarkConfig) -> PerfMatrix:
    """Measure the matrix. Blocks until every level is done."""
    os.makedirs(config.out_dir, exist_ok=True)
    if config.warmup:
        _warmup(agent, config)

    matrix = PerfMatrix(target_fps=config.target_fps)
    for level in config.concurrency:
        runs = [run_level(agent, config, level) for _ in range(max(1, config.repeats))]
        point = median_point(runs)
        if level == min(config.concurrency) and len(point.streams) == 1:
            # Single stream: the collector's stage totals belong to it, so the
            # model/encode split (and the model-only fps) are exact.
            stream = point.streams[0]
            stream.model_seconds = point.model_seconds
            stream.encode_seconds = point.encode_seconds
            stream.other_seconds = max(0.0, stream.wall_seconds - point.model_seconds - point.encode_seconds)
            matrix.baseline = stream
        matrix.points.append(point)

    if matrix.baseline is not None:
        base = matrix.baseline
        # Offline mode: the audio is fully present at the trigger, so both
        # numbers are the time to the first frame.
        base.audio_in_to_video_out_ms = base.ttff_ms
        base.conversational_latency_ms = base.ttff_ms
    matrix.notes = _notes()
    return matrix


def cleanup_outputs(config: BenchmarkConfig) -> None:
    """Remove the mp4s the benchmark produced (unless they are kept)."""
    if config.keep_files or not os.path.isdir(config.out_dir):
        return
    for name in os.listdir(config.out_dir):
        if name.startswith("bench_c") and name.endswith(".mp4"):
            with suppress(OSError):
                os.remove(os.path.join(config.out_dir, name))


__all__ = [
    "DEFAULT_AUDIO",
    "DEFAULT_REF",
    "BenchmarkConfig",
    "RecordingDispatcher",
    "cleanup_outputs",
    "median_point",
    "run_benchmark",
    "run_level",
]
