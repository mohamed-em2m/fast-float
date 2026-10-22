"""Tests for the speed/throughput + latency matrix math (no GPU needed)."""

from __future__ import annotations

import time

import pytest

from fast_float.core.perf_metrics import (
    ConcurrencyPoint,
    FrameClock,
    LatencySummary,
    PerfMatrix,
    RequestAccumulator,
    RequestSample,
    StreamMetrics,
    fold_stage_seconds,
    percentile,
)


def test_percentile_interpolates():
    assert percentile([1, 2, 3, 4], 0) == 1
    assert percentile([1, 2, 3, 4], 100) == 4
    assert percentile([1, 2, 3, 4], 50) == pytest.approx(2.5)
    assert percentile([10], 95) == 10
    assert percentile([], 50) == 0.0


def test_percentile_is_order_independent():
    assert percentile([4, 1, 3, 2], 50) == pytest.approx(2.5)


def test_latency_summary_reports_tail_percentiles():
    values = list(range(1, 101))  # 1..100 ms
    summary = LatencySummary.from_values(values)
    assert summary.count == 100
    assert summary.p50_ms == pytest.approx(50.5)
    assert summary.p95_ms == pytest.approx(95.05)
    assert summary.p99_ms == pytest.approx(99.01)
    assert summary.max_ms == 100


def test_latency_summary_of_nothing_is_empty():
    summary = LatencySummary.from_values([])
    assert summary.count == 0
    assert summary.p95_ms == 0.0


def test_frame_clock_measures_ttff_latency_and_drops():
    clock = FrameClock(fps=25.0, start_s=100.0)
    # frame 0 is due at 0 ms, arrives 50 ms late
    clock.on_frame(0, 0.0, 100.05)
    # frame 2 is due at 80 ms (25 fps), arrives 100 ms late; frame 1 was dropped
    clock.on_frame(2, 80.0, 100.18)
    # frame 3 is due at 120 ms, arrives 130 ms late
    clock.on_frame(3, 120.0, 100.25)

    assert clock.delivered == 3
    assert clock.dropped == 1
    assert clock.ttff_ms == pytest.approx(50.0)
    assert clock.latency.p50_ms == pytest.approx(100.0)
    assert clock.latency.p95_ms == pytest.approx(127.0)  # interpolated between 100 and 130
    assert clock.latency.max_ms == pytest.approx(130.0)
    assert clock.drop_ratio == pytest.approx(0.25)


def test_frame_clock_jitter_uses_delivery_intervals():
    clock = FrameClock(fps=25.0, start_s=0.0)
    for i in range(5):
        clock.on_frame(i, i * 40.0, i * 0.04)
    # perfectly paced delivery -> no jitter
    assert clock.jitter_ms == pytest.approx(0.0, abs=1e-6)
    assert clock.ideal_interval_ms == pytest.approx(40.0)


def test_stream_metrics_split_model_and_end_to_end():
    stream = StreamMetrics(
        name="s", target_fps=25.0, model_seconds=1.0, encode_seconds=1.0, wall_seconds=2.0, frames=50
    )
    assert stream.model_fps == pytest.approx(50.0)
    assert stream.e2e_fps == pytest.approx(25.0)
    assert stream.rtf == pytest.approx(1.0)
    assert stream.gpu_utilisation == pytest.approx(0.5)


def test_concurrency_point_reports_worst_stream_and_realtime():
    def stream(fps: float) -> StreamMetrics:
        s = StreamMetrics(target_fps=25.0, wall_seconds=1.0)
        s.frames = int(fps)
        return s

    point = ConcurrencyPoint(
        concurrency=2,
        streams=[stream(30.0), stream(20.0)],
        wall_seconds=1.0,
        model_seconds=1.5,
    )
    assert point.aggregate_fps == pytest.approx(50.0)
    assert point.per_stream_fps == pytest.approx(25.0)
    assert point.worst_stream_fps == pytest.approx(20.0)
    assert point.rtf == pytest.approx(1.0)
    assert point.at_realtime is False
    assert point.gpu_saturation == pytest.approx(0.75)
    # the UI reads the saturation out of the payload, so it has to be there
    assert point.to_dict()["gpu_saturation"] == pytest.approx(0.75)


def test_perf_matrix_counts_streams_at_realtime():
    def level(concurrency: int, fps: float) -> ConcurrencyPoint:
        s = StreamMetrics(target_fps=25.0, wall_seconds=1.0)
        s.frames = int(fps)
        return ConcurrencyPoint(concurrency=concurrency, streams=[s], wall_seconds=1.0, model_seconds=0.5)

    matrix = PerfMatrix(target_fps=25.0, baseline=level(1, 30.0).streams[0])
    matrix.points = [level(1, 30.0), level(4, 26.0), level(8, 18.0)]
    assert matrix.streams_at_realtime == 4
    assert matrix.model_fps == pytest.approx(0.0)  # baseline has no model time
    assert matrix.e2e_fps == pytest.approx(30.0)


def test_perf_matrix_renders_both_sections():
    clock = FrameClock(fps=25.0, start_s=0.0)
    clock.on_frame(0, 0.0, 0.4)
    baseline = StreamMetrics(
        name="s",
        target_fps=25.0,
        model_seconds=1.0,
        encode_seconds=1.0,
        wall_seconds=2.0,
        frames=50,
        clock=clock,
    )
    baseline.audio_in_to_video_out_ms = baseline.ttff_ms
    baseline.conversational_latency_ms = baseline.ttff_ms

    matrix = PerfMatrix(target_fps=25.0, baseline=baseline)
    matrix.points = [ConcurrencyPoint(concurrency=1, streams=[baseline], wall_seconds=2.0, model_seconds=1.0)]
    markdown = matrix.to_markdown()

    assert "## Speed and throughput" in markdown
    assert "## Latency" in markdown
    assert "Throughput fps (model-only)" in markdown
    assert "Real-time factor" in markdown
    assert "Concurrent streams per GPU at real-time" in markdown
    assert "Per-stream fps under load" in markdown
    assert "Time to first frame (TTFF)" in markdown
    assert "p50 / p95 / p99 latency" in markdown
    assert "Frame-time jitter" in markdown
    assert "Dropped frames" in markdown


def test_fold_stage_seconds_sums_only_the_named_stages():
    totals = {
        "ode_solve": {"total_s": 2.0},
        "decode": {"total_s": 1.0},
        "video_encode": {"total_s": 0.5},
    }
    assert fold_stage_seconds(totals, ("ode_solve", "decode")) == pytest.approx(3.0)
    assert fold_stage_seconds(totals, ("missing",)) == 0.0


def test_request_sample_computes_fps_and_rtf():
    sample = RequestSample(
        frames=50,
        model_seconds=0.5,
        wall_seconds=1.0,
        target_fps=25.0,
    )
    assert sample.model_fps == pytest.approx(100.0)
    assert sample.e2e_fps == pytest.approx(50.0)
    assert sample.rtf == pytest.approx(2.0)


def test_request_accumulator_reports_percentiles_across_requests():
    acc = RequestAccumulator(maxlen=5)
    for i in range(1, 6):
        acc.add(RequestSample(frames=50, model_seconds=1.0 / i, wall_seconds=1.0, target_fps=25.0))

    summary = acc.summary()
    assert summary["count"] == 5
    assert summary["window"] == 5
    assert summary["model_fps"]["p50"] == pytest.approx(150.0)  # sorted 50,100,150,200,250
    assert summary["e2e_fps"]["p50"] == pytest.approx(50.0)
    # every request had the same wall time, so the real-time factor is constant
    assert summary["rtf"]["p50"] == pytest.approx(2.0)
    assert summary["rtf"]["p95"] == pytest.approx(2.0)
    assert len(summary["recent"]) == 5


def test_request_accumulator_is_bounded():
    acc = RequestAccumulator(maxlen=3)
    for _ in range(10):
        acc.add(RequestSample(frames=1, wall_seconds=1.0))
    assert len(acc) == 3
    assert acc.summary()["count"] == 3


def test_request_accumulator_of_nothing_is_empty():
    summary = RequestAccumulator().summary()
    assert summary["count"] == 0
    assert summary["model_fps"]["count"] == 0
    assert summary["recent"] == []


def test_timing_collector_exposes_stage_totals():
    from fast_float.utils.timing import timed, timing_run

    with timing_run("test") as collector:
        with timed("ode_solve"):
            time.sleep(0.01)
        with timed("video_encode"):
            time.sleep(0.01)

    totals = collector.totals()
    assert totals["ode_solve"]["count"] == 1
    assert totals["ode_solve"]["total_s"] >= 0.005
    assert fold_stage_seconds(totals, ("ode_solve", "video_encode")) >= 0.01


def test_dispatcher_delivery_feeds_the_frame_clock():
    import numpy as np

    from fast_float.streaming.dispatcher import FrameDispatcher
    from fast_float.streaming.types import StreamMeta

    class ClockDispatcher(FrameDispatcher):
        def __init__(self, clock):
            super().__init__()
            self.clock = clock
            self.delivered = []

        def dispatch(self, idx, frames, pts_ms):
            # Mirror the bench harness: the base dispatcher delivers batches.
            count = len(frames) if hasattr(frames, "__len__") else 1
            now = time.monotonic()
            step_ms = 1000.0 / clock.fps
            for i in range(count):
                self.delivered.append((idx + i, pts_ms + i * step_ms))
                self.clock.on_frame(idx + i, pts_ms + i * step_ms, now)

    clock = FrameClock(fps=25.0, start_s=time.monotonic())
    dispatcher = ClockDispatcher(clock)
    dispatcher.bind(StreamMeta(fps=25.0, request_id="t", started_at_s=clock.start_s))
    try:
        for start in (0, 2):
            frames = np.zeros((2, 4, 4, 3), dtype=np.uint8)
            dispatcher.submit(frames, idx=start)
        deadline = time.monotonic() + 5.0
        while len(dispatcher.delivered) < 4 and time.monotonic() < deadline:
            time.sleep(0.01)
    finally:
        dispatcher.shutdown(timeout=5.0)

    assert len(dispatcher.delivered) == 4
    assert clock.delivered == 4
    assert clock.ttff_ms is not None
    assert clock.latency.count == 4
