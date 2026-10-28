import asyncio
import os
import re
import time
import uuid

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from fast_float.core.service import generate_video
from fast_float.core.state import app_state
from fast_float.schemas.generation import (
    GenerationProgress,
    GenerationRequest,
    GenerationResponse,
    PerformanceTimings,
)
from fast_float.streaming.live import LiveStreamDispatcher
from fast_float.utils.logger import logger
from fast_float.utils.media import TARGET_IMAGE_SIZE

router = APIRouter(tags=["generation"])

# How long a request waits for a free generation slot before getting a 429.
# Inference takes minutes; holding the connection open for all of it would
# exhaust the server's connection pool, so we fail fast instead of queueing.
ACQUIRE_TIMEOUT_S = 5.0

# How long a request waits for the *boot warmup* to finish. Unlike the slot
# queue above this is intentional: a client that fires right after a restart
# used to get an immediate 503 "Model is warming up, try again shortly" for the
# whole warmup window (tens of seconds, minutes on a cold TensorRT cache), so a
# single request could never succeed. Waiting rides it out instead.
WARMUP_WAIT_TIMEOUT_S = float(os.getenv("WARMUP_WAIT_TIMEOUT_S", "600"))


def _output_resolution(state: object) -> str:
    """``<input_size>x<input_size>`` — the size the frames are actually decoded at.

    Reported as ``TARGET_IMAGE_SIZE`` before, which is the placeholder size, so
    every response announced 256x256 while the MP4 was 512x512.
    """
    opt = getattr(state, "opt", None)
    size = getattr(opt, "input_size", None) or TARGET_IMAGE_SIZE
    try:
        size = int(size)
    except (TypeError, ValueError):
        size = TARGET_IMAGE_SIZE
    return f"{size}x{size}"


def _safe_request_id(client_id: str | None) -> str:
    """Build a pollable request id, sanitizing any client-supplied value."""
    if client_id:
        cleaned = re.sub(r"[^A-Za-z0-9_.-]", "", client_id)[:48]
        if cleaned:
            return cleaned
    app_state.queue_counter += 1
    return f"req_{app_state.queue_counter}_{uuid.uuid4().hex[:8]}"


def _close_live(dispatcher, request_id: str, message: str) -> None:
    """Retire a live stream so watchers stop waiting for frames."""
    if dispatcher is None:
        return
    stream = dispatcher.stream
    if app_state.live_streams.get(request_id) is stream:
        app_state.live_streams.pop(request_id, None)
    # Keep the delivery metrics readable after the stream is retired: a client
    # that polls /progress once the request is done should still see how many
    # frames were buffered on disk or dropped.
    metrics = stream.metrics()
    if metrics:
        app_state.set_progress(request_id, delivery_metrics=metrics)
    stream.finish(error=message)


@router.post("/generate", response_model=GenerationResponse)
async def generate(request: GenerationRequest):
    """Generate talking-face video from reference image and audio."""
    if app_state.agent is None:
        raise HTTPException(status_code=503, detail="FLOAT model is not yet loaded")

    if app_state.warming_up:
        elapsed_at_start = app_state.warmup_elapsed()
        logger.info(f"[warmup] request held until boot warmup finishes (stage={app_state.warmup_stage})")
        if not await app_state.wait_for_warmup(WARMUP_WAIT_TIMEOUT_S):
            elapsed = app_state.warmup_elapsed()
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Model warmup is still running after {elapsed:.0f}s "
                    f"(stage: {app_state.warmup_stage}, waited {elapsed - elapsed_at_start:.0f}s); "
                    "retry shortly"
                ),
                headers={"Retry-After": "10"},
            )

    if app_state.request_semaphore is None:
        app_state.initialize_runtime()

    request_id = _safe_request_id(request.client_request_id)
    app_state.set_progress(
        request_id,
        status="queued",
        stage="queued",
        percent=0.0,
        message="Waiting for a free GPU slot",
        started_at=time.time(),
        elapsed_s=0.0,
    )

    live_dispatcher = None
    if request.live:
        loop = asyncio.get_running_loop()
        live_dispatcher = LiveStreamDispatcher(request_id, loop)
        app_state.live_streams[request_id] = live_dispatcher.stream
        # Batched sampling decodes several chunks at once and cannot feed a
        # per-frame transport.
        if int(getattr(request, "chunk_batch_size", 1) or 1) > 1:
            logger.info(f"[{request_id}] live stream: forcing chunk_batch_size=1")

        try:
            from fast_float.livekit.settings import LiveKitSettings

            lk_settings = LiveKitSettings.from_env()
            if lk_settings.configured:
                from fast_float.livekit.worker import spawn_live_publish_task

                _t = spawn_live_publish_task(live_dispatcher.stream, room_name=request_id, settings=lk_settings)
                if _t is None:
                    logger.info(f"[{request_id}] livekit publisher already running for room {request_id}")
                else:
                    logger.info(f"[{request_id}] spawned LiveKit avatar publisher for room {request_id}")
        except Exception as exc:
            logger.debug(f"[{request_id}] livekit publisher auto-spawn skipped: {exc}")

    try:
        await asyncio.wait_for(app_state.request_semaphore.acquire(), timeout=ACQUIRE_TIMEOUT_S)
    except asyncio.TimeoutError:
        app_state.set_progress(
            request_id,
            status="error",
            stage="error",
            message="Server busy, try again shortly",
        )
        _close_live(live_dispatcher, request_id, "Server busy, try again shortly")
        raise HTTPException(status_code=429, detail="Server busy, try again shortly") from None
    except BaseException:
        _close_live(live_dispatcher, request_id, "Request cancelled")
        raise

    try:
        try:
            req_dict = request.model_dump() if hasattr(request, "model_dump") else request.dict()
            if live_dispatcher is not None:
                req_dict["chunk_batch_size"] = 1
            video_path, timings = await generate_video(
                req_dict,
                request_id,
                app_state,
                dispatcher=live_dispatcher,
            )
            app_state.active_streams[request_id] = video_path

            pub_url = app_state.public_url or "http://localhost:8000"
            pub_backup = app_state.public_backup_url or pub_url

            encoder_name = app_state.video_optimizer.encoder_type if app_state.video_optimizer else "default"

            file_size_mb = round(os.path.getsize(video_path) / (1024 * 1024), 2) if os.path.exists(video_path) else 0.0

            response = GenerationResponse(
                status="success",
                request_id=request_id,
                download_url=f"{pub_url}/download/{request_id}",
                download_backup=f"{pub_backup}/download/{request_id}" if app_state.public_backup_url else None,
                stream_url=f"{pub_url}/stream/{request_id}",
                video_path=video_path,
                file_size_mb=file_size_mb,
                resolution=_output_resolution(app_state),
                encoder=encoder_name,
                performance=PerformanceTimings(
                    preprocess_time=round(timings["preprocess"], 3),
                    inference_time=round(timings["inference"], 3),
                    total_time=round(timings["total"], 3),
                    fps=round(timings["fps"], 2),
                ),
                chunks=int(timings.get("chunks", 1)),
            )
            if live_dispatcher is not None:
                live_dispatcher.stream.finish(result=response.model_dump())
            return response

        except HTTPException as e:
            if live_dispatcher is not None:
                live_dispatcher.stream.finish(error=str(e.detail))
            raise
        except ValueError as e:
            logger.warning(f"[{request_id}] Bad request: {e}")
            app_state.set_progress(request_id, status="error", stage="error", message=str(e))
            if live_dispatcher is not None:
                live_dispatcher.stream.finish(error=str(e))
            raise HTTPException(status_code=400, detail=str(e)) from e
        except FileNotFoundError as e:
            logger.error(f"[{request_id}] Generation output missing: {e}")
            app_state.set_progress(
                request_id,
                status="error",
                stage="error",
                message="Video generation failed",
            )
            if live_dispatcher is not None:
                live_dispatcher.stream.finish(error="Generation output missing")
            raise HTTPException(status_code=502, detail="Video generation failed") from e
        except Exception as e:
            logger.error(f"[{request_id}] Generation failed: {e!s}")
            app_state.set_progress(
                request_id,
                status="error",
                stage="error",
                message="Video generation failed",
            )
            if live_dispatcher is not None:
                live_dispatcher.stream.finish(error=str(e))
            raise HTTPException(status_code=500, detail="Video generation failed") from e
    finally:
        app_state.request_semaphore.release()
        _close_live(live_dispatcher, request_id, "generation ended")


@router.get("/progress/{request_id}", response_model=GenerationProgress)
async def generation_progress(request_id: str):
    """Poll the progress of a generation request (chunk-by-chunk)."""
    record = app_state.get_progress(request_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Unknown request id")
    allowed = set(GenerationProgress.model_fields)
    payload = {k: v for k, v in record.items() if k in allowed}
    # Live streams also report their delivery metrics (TTFF, latency
    # percentiles, jitter, drops) so the latency half of the perf matrix is
    # observable in production, not only in the benchmark.
    stream = app_state.live_streams.get(request_id)
    if stream is not None:
        metrics = stream.metrics()
        if metrics:
            payload["delivery_metrics"] = metrics
    return GenerationProgress(**payload)


@router.get("/download/{request_id}")
async def download_video(request_id: str):
    """Download the generated MP4 video file."""
    if request_id not in app_state.active_streams:
        raise HTTPException(status_code=404, detail="Requested video ID not found")

    video_path = app_state.active_streams[request_id]
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video file on disk not found")

    return FileResponse(
        video_path,
        media_type="video/mp4",
        filename=f"{request_id}.mp4",
    )


@router.get("/stream/{request_id}")
async def stream_video(request_id: str):
    """Stream generated MP4 video with HTTP byte-range support."""
    if request_id not in app_state.active_streams:
        raise HTTPException(status_code=404, detail="Requested stream ID not found")

    video_path = app_state.active_streams[request_id]
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video file on disk not found")

    return FileResponse(
        video_path,
        media_type="video/mp4",
        headers={"Accept-Ranges": "bytes"},
    )
