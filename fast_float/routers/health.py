import torch
from fastapi import APIRouter

from fast_float.core.state import app_state
from fast_float.schemas.generation import HealthResponse
from fast_float.utils.gpu import (
    clear_gpu_memory,
    estimate_stream_capacity,
    get_current_gpu_memory,
    get_gpu_info,
)

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint with current model and performance status."""
    total_req = app_state.performance_stats["total_requests"]
    total_inf_time = app_state.performance_stats["total_inference_time"]
    avg_inf_time = round(total_inf_time / max(total_req, 1), 3)

    return HealthResponse(
        status="healthy",
        model_loaded=app_state.agent is not None,
        warming_up=app_state.warming_up,
        warmup_stage=app_state.warmup_stage or None,
        warmup_elapsed_s=round(app_state.warmup_elapsed(), 2) if app_state.warmup_started_at else None,
        encoder=(app_state.video_optimizer.encoder_type if app_state.video_optimizer else "unknown"),
        active_streams=len(app_state.active_streams),
        performance_stats={
            "total_requests": total_req,
            "avg_inference_time": avg_inf_time,
            "average_fps": round(app_state.performance_stats["average_fps"], 2),
        },
        gpu_info=get_gpu_info(),
    )


@router.get("/gpu")
async def gpu_status():
    """Get detailed GPU metrics, memory utilization and stream capacity."""
    rtf = app_state.live_rtf
    if rtf is None:
        # No live avatar has rendered yet; fall back to the rolling window of
        # finished /generate requests (p50 ignores warmup outliers).
        sample = app_state.perf_requests.summary().get("rtf", {})
        p50 = float(sample.get("p50") or 0.0)
        rtf = p50 if p50 > 0 else None
    capacity = estimate_stream_capacity(
        config_val=_configured_max_streams(), observed_rtf=rtf
    )
    return {
        "info": get_gpu_info(),
        "memory": get_current_gpu_memory(),
        "capacity": capacity.as_dict(),
    }


def _configured_max_streams() -> int:
    """``livekit_max_streams`` from the shared config (0 = auto)."""
    try:
        from fast_float.livekit.settings import LiveKitSettings

        return int(getattr(LiveKitSettings.from_env(), "livekit_max_streams", 0) or 0)
    except Exception:
        return 0


@router.post("/gpu/clear")
async def clear_gpu():
    """Manually clear CUDA memory cache and garbage collector."""
    if not torch.cuda.is_available():
        return {
            "status": "no_gpu",
            "message": "No CUDA GPU available, nothing to clear",
            "current_memory": None,
        }
    clear_gpu_memory()
    return {
        "status": "success",
        "message": "GPU memory cleared",
        "current_memory": get_current_gpu_memory(),
    }
