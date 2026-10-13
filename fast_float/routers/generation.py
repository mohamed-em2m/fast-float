import asyncio
import os
import uuid
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from fast_float.utils.logger import logger
from fast_float.utils.media import TARGET_IMAGE_SIZE
from fast_float.schemas.generation import GenerationRequest, GenerationResponse, PerformanceTimings
from fast_float.core.state import app_state
from fast_float.core.service import generate_video

router = APIRouter(tags=["generation"])

# How long a request waits for a free generation slot before getting a 429.
# Inference takes minutes; holding the connection open for all of it would
# exhaust the server's connection pool, so we fail fast instead of queueing.
ACQUIRE_TIMEOUT_S = 5.0


@router.post("/generate", response_model=GenerationResponse)
async def generate(request: GenerationRequest):
    """Generate talking-face video from reference image and audio."""
    if app_state.agent is None:
        raise HTTPException(status_code=503, detail="FLOAT model is not yet loaded")

    if app_state.request_semaphore is None:
        app_state.initialize_runtime()

    try:
        await asyncio.wait_for(
            app_state.request_semaphore.acquire(), timeout=ACQUIRE_TIMEOUT_S
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=429, detail="Server busy, try again shortly"
        ) from None

    try:
        app_state.queue_counter += 1
        request_id = f"req_{app_state.queue_counter}_{uuid.uuid4().hex[:8]}"

        try:
            req_dict = request.model_dump() if hasattr(request, "model_dump") else request.dict()
            video_path, timings = await generate_video(req_dict, request_id, app_state)
            app_state.active_streams[request_id] = video_path

            pub_url = app_state.public_url or "http://localhost:8000"
            pub_backup = app_state.public_backup_url or pub_url

            encoder_name = (
                app_state.video_optimizer.encoder_type
                if app_state.video_optimizer
                else "default"
            )

            file_size_mb = (
                round(os.path.getsize(video_path) / (1024 * 1024), 2)
                if os.path.exists(video_path)
                else 0.0
            )

            return GenerationResponse(
                status="success",
                request_id=request_id,
                download_url=f"{pub_url}/download/{request_id}",
                download_backup=f"{pub_backup}/download/{request_id}" if app_state.public_backup_url else None,
                stream_url=f"{pub_url}/stream/{request_id}",
                video_path=video_path,
                file_size_mb=file_size_mb,
                resolution=f"{TARGET_IMAGE_SIZE}x{TARGET_IMAGE_SIZE}",
                encoder=encoder_name,
                performance=PerformanceTimings(
                    preprocess_time=round(timings["preprocess"], 3),
                    inference_time=round(timings["inference"], 3),
                    total_time=round(timings["total"], 3),
                    fps=round(timings["fps"], 2),
                ),
            )

        except HTTPException:
            raise
        except ValueError as e:
            logger.warning(f"[{request_id}] Bad request: {e}")
            raise HTTPException(status_code=400, detail=str(e))
        except FileNotFoundError as e:
            logger.error(f"[{request_id}] Generation output missing: {e}")
            raise HTTPException(status_code=502, detail="Video generation failed")
        except Exception as e:
            logger.error(f"[{request_id}] Generation failed: {e!s}")
            raise HTTPException(status_code=500, detail="Video generation failed")
    finally:
        app_state.request_semaphore.release()


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
