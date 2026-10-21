"""perf_metrics.py - the speed/throughput + latency matrix, computed in code.

Every number the matrix reports comes from recorded samples, never from a single
average: latencies are reported as p50/p95/p99, frame pacing as jitter plus
dropped frames, and throughput is split into model-only vs end-to-end (encode,
composite, mux, write).

Three layers, each usable on its own:

``FrameClock``
    Delivery-side recorder. Feed it ``(idx, pts_ms, delivered_at)`` samples and
    it yields time-to-first-frame, per-frame latency, pacing jitter and dropped
    frames. This is the live/WebRTC leg.

``StreamMetrics`` / ``ConcurrencyPoint``
    Throughput accounting for one stream, and for a level of the concurrency
    sweep (how many streams still run at real time, and what per-stream fps
    collapses to under load).

``PerfMatrix``
    Holds the single-stream baseline plus the sweep and renders the matrix
    (markdown or dict) for reports.

The module is deliberately dependency-free (no torch, no numpy): it must be
importable in a plain API process and unit-testable without a GPU.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

# Stages that make up "the model" (encoders, ODE solve, decode to frames).
# Sourced from the ``timed(...)`` stage names in the pipeline.
MODEL_STAGES: tuple[str, ...] = (
    "audio_encode",
    "emotion_encode",
    "ode_solve",
    "decode",
    "encode_image",
    "encode_identity",
)

# Stages that turn frames into a file/stream (encode, gather, mux, write).
ENCODE_STAGES: tuple[str, ...] = (
    "frame_gather",
    "video_encode",
    "video_write",
    "video_finalize",
    "merge",
)


def percentile(values: Sequence[float], p: float) -> float:
    """Linear-interpolated percentile of ``values`` (``p`` in 0..100).

    Pure python on purpose: the metrics must not drag numpy (or torch) into a
    process that only reports numbers.
    """
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    p = min(100.0, max(0.0, float(p)))
    pos = (len(ordered) - 1) * (p / 100.0)
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return ordered[int(pos)]
    return ordered[low] * (high - pos) + ordered[high] * (pos - low)


def _stdev(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))


def _fmt(value: float | None, digits: int = 1, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}{suffix}"


@dataclass
class LatencySummary:
    """Latency distribution of a set of samples (milliseconds)."""

    count: int = 0
    mean_ms: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    max_ms: float = 0.0

    @classmethod
    def from_values(cls, values: Iterable[float]) -> LatencySummary:
        vals = [float(v) for v in values]
        if not vals:
            return cls()
        return cls(
            count=len(vals),
            mean_ms=sum(vals) / len(vals),
            p50_ms=percentile(vals, 50),
            p95_ms=percentile(vals, 95),
            p99_ms=percentile(vals, 99),
            max_ms=max(vals),
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "count": self.count,
            "mean_ms": self.mean_ms,
            "p50_ms": self.p50_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
            "max_ms": self.max_ms,
        }


@dataclass
class FrameClock:
    """Delivery-side clock: turns frame samples into the latency matrix.

    Feed ``(idx, pts_ms, delivered_at)`` where ``pts_ms`` is the frame's
    presentation timestamp relative to the stream start and ``delivered_at`` is
    a monotonic timestamp of when the frame reached the consumer. The latency of
    a frame is how late it arrived relative to when it was due.
    """

    fps: float = 25.0
    start_s: float = 0.0
    first_idx: int = 0
    delivered: int = 0
    dropped: int = 0
    ttff_ms: float | None = None
    latencies_ms: list[float] = field(default_factory=list)
    intervals_ms: list[float] = field(default_factory=list)
    _last_idx: int | None = field(default=None, repr=False)
    _last_delivered_s: float | None = field(default=None, repr=False)

    def on_frame(self, idx: int, pts_ms: float, delivered_s: float) -> None:
        """Record one delivered frame."""
        self.delivered += 1
        if self._last_idx is not None and idx > self._last_idx + 1:
            self.dropped += idx - self._last_idx - 1
        self._last_idx = idx

        elapsed_ms = (delivered_s - self.start_s) * 1000.0
        self.latencies_ms.append(elapsed_ms - float(pts_ms))
        if self.ttff_ms is None:
            self.ttff_ms = elapsed_ms
        if self._last_delivered_s is not None:
            self.intervals_ms.append((delivered_s - self._last_delivered_s) * 1000.0)
        self._last_delivered_s = delivered_s

    @property
    def latency(self) -> LatencySummary:
        return LatencySummary.from_values(self.latencies_ms)

    @property
    def jitter_ms(self) -> float:
        """Standard deviation of the delivery intervals (pacing jitter)."""
        return _stdev(self.intervals_ms)

    @property
    def ideal_interval_ms(self) -> float:
        return 1000.0 / self.fps if self.fps else 0.0

    @property
    def drop_ratio(self) -> float:
        total = self.delivered + self.dropped
        return self.dropped / total if total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "delivered": self.delivered,
            "dropped": self.dropped,
            "drop_ratio": self.drop_ratio,
            "ttff_ms": self.ttff_ms,
            "jitter_ms": self.jitter_ms,
            "ideal_interval_ms": self.ideal_interval_ms,
            "latency": self.latency.to_dict(),
        }


@dataclass
class StreamMetrics:
    """Throughput + latency accounting for one stream."""

    name: str = "stream"
    target_fps: float = 25.0
    model_seconds: float = 0.0
    encode_seconds: float = 0.0
    other_seconds: float = 0.0
    wall_seconds: float = 0.0
    frames: int = 0
    dropped: int = 0
    clock: FrameClock | None = None
    audio_in_to_video_out_ms: float | None = None
    conversational_latency_ms: float | None = None

    @property
    def model_fps(self) -> float:
        """Frames per second of *model* time (ODE solve + decode, no encode)."""
        return self.frames / self.model_seconds if self.model_seconds > 0 else 0.0

    @property
    def e2e_fps(self) -> float:
        """Frames per second of wall time (model + encode + composite + write)."""
        return self.frames / self.wall_seconds if self.wall_seconds > 0 else 0.0

    @property
    def rtf(self) -> float:
        """Real-time factor: end-to-end fps divided by the target fps."""
        return self.e2e_fps / self.target_fps if self.target_fps else 0.0

    @property
    def gpu_utilisation(self) -> float:
        """Share of the wall time spent in model stages (1.0 = saturated)."""
        return self.model_seconds / self.wall_seconds if self.wall_seconds > 0 else 0.0

    @property
    def latency(self) -> LatencySummary:
        return self.clock.latency if self.clock is not None else LatencySummary()

    @property
    def jitter_ms(self) -> float:
        return self.clock.jitter_ms if self.clock is not None else 0.0

    @property
    def ttff_ms(self) -> float | None:
        return self.clock.ttff_ms if self.clock is not None else None

    @property
    def drop_ratio(self) -> float:
        if self.clock is not None and (self.clock.delivered + self.clock.dropped) > 0:
            return self.clock.drop_ratio
        total = self.frames + self.dropped
        return self.dropped / total if total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "target_fps": self.target_fps,
            "frames": self.frames,
            "dropped": self.dropped,
            "model_seconds": self.model_seconds,
            "encode_seconds": self.encode_seconds,
            "other_seconds": self.other_seconds,
            "wall_seconds": self.wall_seconds,
            "model_fps": self.model_fps,
            "e2e_fps": self.e2e_fps,
            "rtf": self.rtf,
            "gpu_utilisation": self.gpu_utilisation,
            "ttff_ms": self.ttff_ms,
            "audio_in_to_video_out_ms": self.audio_in_to_video_out_ms,
            "conversational_latency_ms": self.conversational_latency_ms,
            "jitter_ms": self.jitter_ms,
            "drop_ratio": self.drop_ratio,
            "latency": self.latency.to_dict(),
        }


@dataclass
class ConcurrencyPoint:
    """One level of the concurrency sweep (N streams sharing the GPU)."""

    concurrency: int
    streams: list[StreamMetrics] = field(default_factory=list)
    wall_seconds: float = 0.0
    # Summed stage times of the level, as measured by the timing collector.
    # Under concurrency the model stages of different streams overlap, so
    # ``model_seconds`` is "GPU busy time", not elapsed time.
    model_seconds: float = 0.0
    encode_seconds: float = 0.0

    @property
    def total_frames(self) -> int:
        return sum(s.frames for s in self.streams)

    @property
    def aggregate_fps(self) -> float:
        """Total frames per second of wall time across the level."""
        return self.total_frames / self.wall_seconds if self.wall_seconds > 0 else 0.0

    @property
    def gpu_saturation(self) -> float:
        """Share of the level's wall-clock capacity spent in model stages.

        1.0 means the model stages saturated the whole window (the GPU is the
        bottleneck); well below 1.0 means the streams were waiting on
        something else (encode, IO, Python).
        """
        capacity = self.concurrency * self.wall_seconds
        return self.model_seconds / capacity if capacity > 0 else 0.0

    @property
    def per_stream_fps(self) -> float:
        """Mean per-stream fps (each stream's own frames over its own wall time)."""
        values = [s.e2e_fps for s in self.streams if s.wall_seconds > 0]
        return sum(values) / len(values) if values else 0.0

    @property
    def worst_stream_fps(self) -> float:
        values = [s.e2e_fps for s in self.streams if s.wall_seconds > 0]
        return min(values) if values else 0.0

    @property
    def rtf(self) -> float:
        target = self.streams[0].target_fps if self.streams else 0.0
        return self.per_stream_fps / target if target else 0.0

    @property
    def at_realtime(self) -> bool:
        """True when every stream in the level still runs at (or above) real time."""
        target = self.streams[0].target_fps if self.streams else 0.0
        return bool(target) and self.worst_stream_fps >= target

    @property
    def dropped(self) -> int:
        return sum(s.dropped for s in self.streams)

    def to_dict(self) -> dict[str, Any]:
        return {
            "concurrency": self.concurrency,
            "wall_seconds": self.wall_seconds,
            "total_frames": self.total_frames,
            "model_seconds": self.model_seconds,
            "encode_seconds": self.encode_seconds,
            "aggregate_fps": self.aggregate_fps,
            "per_stream_fps": self.per_stream_fps,
            "worst_stream_fps": self.worst_stream_fps,
            "rtf": self.rtf,
            "at_realtime": self.at_realtime,
            "dropped": self.dropped,
            "streams": [s.to_dict() for s in self.streams],
        }


@dataclass
class PerfMatrix:
    """The full report: a single-stream baseline plus the concurrency sweep."""

    target_fps: float = 25.0
    baseline: StreamMetrics | None = None
    points: list[ConcurrencyPoint] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def streams_at_realtime(self) -> int:
        """How many streams still run at real time (largest passing level)."""
        passing = [p.concurrency for p in self.points if p.at_realtime]
        return max(passing) if passing else 0

    @property
    def model_fps(self) -> float:
        """Model-only throughput of the single-stream baseline."""
        return self.baseline.model_fps if self.baseline is not None else 0.0

    @property
    def e2e_fps(self) -> float:
        return self.baseline.e2e_fps if self.baseline is not None else 0.0

    @property
    def rtf(self) -> float:
        return self.baseline.rtf if self.baseline is not None else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_fps": self.target_fps,
            "model_fps": self.model_fps,
            "e2e_fps": self.e2e_fps,
            "rtf": self.rtf,
            "streams_at_realtime": self.streams_at_realtime,
            "baseline": self.baseline.to_dict() if self.baseline is not None else None,
            "points": [p.to_dict() for p in self.points],
            "notes": list(self.notes),
        }

    # ------------------------------------------------------------------ report
    def to_markdown(self) -> str:
        """Render the matrix (speed/throughput + latency) as markdown."""
        base = self.baseline
        lat = base.latency if base is not None else LatencySummary()
        lines: list[str] = []
        lines.append("## Speed and throughput")
        lines.append("")
        lines.append("| metric | value |")
        lines.append("| --- | --- |")
        lines.append(f"| Throughput fps (model-only) | {_fmt(self.model_fps, 2)} |")
        lines.append(f"| End-to-end fps (encode, composite, stream) | {_fmt(self.e2e_fps, 2)} |")
        lines.append(f"| Real-time factor (fps / {_fmt(self.target_fps, 0)} target) | {_fmt(self.rtf, 2)}x |")
        lines.append(f"| Concurrent streams per GPU at real-time | {self.streams_at_realtime} |")
        per_stream = ", ".join(f"{p.concurrency}: {_fmt(p.per_stream_fps, 1)}" for p in self.points)
        lines.append(f"| Per-stream fps under load | {per_stream or 'n/a'} |")
        if base is not None:
            lines.append(
                f"| Model / encode / other split (s) | {_fmt(base.model_seconds, 2)} / "
                f"{_fmt(base.encode_seconds, 2)} / {_fmt(base.other_seconds, 2)} |"
            )
            lines.append(f"| GPU utilisation (model share of wall time) | {_fmt(base.gpu_utilisation * 100, 1)}% |")
        lines.append("")

        lines.append("## Latency")
        lines.append("")
        lines.append("| metric | value |")
        lines.append("| --- | --- |")
        lines.append(f"| Time to first frame (TTFF) | {_fmt(base.ttff_ms if base else None, 1, ' ms')} |")
        lines.append(f"| Audio-in to video-out | {_fmt(base.audio_in_to_video_out_ms if base else None, 1, ' ms')} |")
        lines.append(
            f"| Conversational latency (stop -> avatar) | "
            f"{_fmt(base.conversational_latency_ms if base else None, 1, ' ms')} |"
        )
        lines.append(
            f"| p50 / p95 / p99 latency | {_fmt(lat.p50_ms, 1)} / {_fmt(lat.p95_ms, 1)} / {_fmt(lat.p99_ms, 1)} ms |"
        )
        lines.append(f"| Frame-time jitter (stddev) | {_fmt(base.jitter_ms if base else 0.0, 2, ' ms')} |")
        dropped = base.dropped if base is not None else 0
        frames = base.frames if base is not None else 0
        ratio = base.drop_ratio if base is not None else 0.0
        lines.append(f"| Dropped frames | {dropped} / {frames + dropped} ({_fmt(ratio * 100, 2)}%) |")
        lines.append("")

        if self.points:
            lines.append("### Concurrency sweep")
            lines.append("")
            lines.append(
                "| streams | aggregate fps | per-stream fps | worst stream | rtf | model time share | dropped |"
            )
            lines.append("| --- | --- | --- | --- | --- | --- | --- |")
            lines.extend(
                f"| {p.concurrency} | {_fmt(p.aggregate_fps, 2)} | {_fmt(p.per_stream_fps, 2)} | "
                f"{_fmt(p.worst_stream_fps, 2)} | {_fmt(p.rtf, 2)}x | {_fmt(p.gpu_saturation * 100, 1)}% | "
                f"{p.dropped} |"
                for p in self.points
            )
            lines.append("")

        if self.notes:
            lines.append("### Notes")
            lines.append("")
            lines.extend(f"- {note}" for note in self.notes)
            lines.append("")

        return "\n".join(lines)


def fold_stage_seconds(totals: dict[str, dict[str, float]], stages: Iterable[str]) -> float:
    """Sum the total wall time of ``stages`` in a ``TimingCollector.totals()`` map."""
    return float(sum(float(totals.get(stage, {}).get("total_s", 0.0)) for stage in stages))


__all__ = [
    "ENCODE_STAGES",
    "MODEL_STAGES",
    "ConcurrencyPoint",
    "FrameClock",
    "LatencySummary",
    "PerfMatrix",
    "StreamMetrics",
    "fold_stage_seconds",
    "percentile",
]
