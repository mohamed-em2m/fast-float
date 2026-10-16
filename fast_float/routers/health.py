import torch
from fastapi import APIRouter

from fast_float.core.state import app_state
from fast_float.schemas.generation import HealthResponse
from fast_float.utils.gpu import (
    clear_gpu_memory,
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
    """Get detailed GPU metrics and memory utilization."""
    return {
        "info": get_gpu_info(),
        "memory": get_current_gpu_memory(),
    }


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
