"""Perf matrix endpoints: measure the speed/throughput + latency matrix.

``POST /perf/benchmark`` runs the benchmark against the *running* agent. It
takes the inference lock, so live requests queue behind it: measuring "how
many streams still run at real time" is only meaningful when nothing else is
using the GPU. The run happens in a worker thread; the endpoint returns
immediately and the frontend polls ``GET /perf/matrix``.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from fast_float.core.perf_bench import BenchmarkConfig, cleanup_outputs, run_benchmark
from fast_float.core.state import app_state
from fast_float.utils.logger import logger

router = APIRouter(tags=["performance"])


class BenchmarkRequest(BaseModel):
    """What to measure. Defaults mirror the deployment settings."""

    concurrency: list[int] = Field(default_factory=lambda: [1, 4, 8, 16])
    nfe: int = 10
    target_fps: float = 25.0
    repeats: int = 1
    emotion: str = "S2E"
    a_cfg: float = 2.0
    r_cfg: float = 1.0
    e_cfg: float = 1.0
    seed: int = 25
    ref: str | None = None
    audio: str | None = None
    keep_files: bool = False


class PerfMatrixResponse(BaseModel):
    """The state of the benchmark plus its last result."""

    status: str = "idle"  # idle | running | done | error
    started_at: float | None = None
    finished_at: float | None = None
    elapsed_s: float | None = None
    error: str | None = None
    matrix: dict[str, Any] | None = None


# Process-local state of the last benchmark run.
_state: dict[str, Any] = {"status": "idle", "started_at": None, "finished_at": None, "error": None, "matrix": None}
_task: asyncio.Task | None = None


def _snapshot() -> PerfMatrixResponse:
    matrix = _state.get("matrix")
    started = _state.get("started_at")
    finished = _state.get("finished_at")
    return PerfMatrixResponse(
        status=str(_state.get("status", "idle")),
        started_at=started,
        finished_at=finished,
        elapsed_s=(finished - started) if (started and finished) else None,
        error=_state.get("error"),
        matrix=matrix,
    )


async def _run(config: BenchmarkConfig) -> None:
    """Run the benchmark with the inference lock held."""
    agent = app_state.agent
    if agent is None:
        _state.update(status="error", error="model is not loaded")
        return
    lock = app_state.inference_lock
    try:
        if lock is not None:
            # Exclusive: live requests wait for the benchmark to finish.
            async with lock:
                matrix = await asyncio.to_thread(run_benchmark, agent, config)
        else:
            matrix = await asyncio.to_thread(run_benchmark, agent, config)
    except Exception as exc:
        logger.exception("[perf] benchmark failed")
        _state.update(status="error", error=f"{type(exc).__name__}: {exc}")
        return
    finally:
        cleanup_outputs(config)

    _state.update(
        status="done",
        finished_at=time.time(),
        error=None,
        matrix=matrix.to_dict(),
    )
    logger.info("[perf] benchmark finished: %d stream(s) at real time", matrix.streams_at_realtime)


@router.get("/perf/matrix", response_model=PerfMatrixResponse)
async def perf_matrix() -> PerfMatrixResponse:
    """The last benchmark result (and whether one is running)."""
    return _snapshot()


@router.get("/perf/requests")
async def perf_requests() -> dict[str, Any]:
    """The rolling per-request view: distributions across the recent requests.

    Fed by every finished request (model fps, e2e fps, RTF, TTFF, latency
    percentiles, jitter, drops). Unlike the benchmark sweep, this needs no
    synthetic load - it is what the box actually did.
    """
    return app_state.perf_requests.summary()


@router.post("/perf/benchmark", response_model=PerfMatrixResponse)
async def start_benchmark(request: BenchmarkRequest) -> PerfMatrixResponse:
    """Start a benchmark run. Returns immediately; poll /perf/matrix."""
    global _task

    if _task is not None and not _task.done():
        raise HTTPException(status_code=409, detail="a benchmark is already running")
    if app_state.agent is None:
        raise HTTPException(status_code=503, detail="model is not loaded yet")

    levels = [int(level) for level in request.concurrency if int(level) > 0]
    if not levels:
        raise HTTPException(status_code=400, detail="no concurrency levels given")
    if max(levels) > 64:
        raise HTTPException(status_code=400, detail="concurrency is capped at 64 streams per level")

    out_dir = os.path.abspath(os.path.join(getattr(app_state.opt, "res_dir", "./results"), "bench"))
    config = BenchmarkConfig(
        ref=request.ref or BenchmarkConfig.ref,
        audio=request.audio or BenchmarkConfig.audio,
        target_fps=request.target_fps,
        concurrency=tuple(levels),
        repeats=max(1, request.repeats),
        nfe=request.nfe,
        seed=request.seed,
        emotion=request.emotion,
        a_cfg=request.a_cfg,
        r_cfg=request.r_cfg,
        e_cfg=request.e_cfg,
        out_dir=out_dir,
        keep_files=request.keep_files,
    )
    for path in (config.ref, config.audio):
        if not os.path.exists(path):
            raise HTTPException(status_code=400, detail=f"missing benchmark input: {path}")

    _state.update(status="running", started_at=time.time(), finished_at=None, error=None, matrix=None)
    _task = asyncio.create_task(_run(config))
    logger.info("[perf] benchmark started: levels=%s nfe=%s", levels, config.nfe)
    return _snapshot()


__all__ = ["BenchmarkRequest", "PerfMatrixResponse", "router"]
