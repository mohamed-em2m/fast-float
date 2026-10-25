import gc
import os
from typing import Any

import torch

from fast_float.utils.logger import logger


def get_gpu_info() -> dict[str, Any]:
    """Get GPU information safely without crashing if CUDA is unavailable."""
    if not torch.cuda.is_available():
        return {"status": "no_gpu"}

    try:
        gpu_name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        total_memory_gb = props.total_memory / (1024**3)
        memory_fraction = float(os.environ.get("PYTORCH_CUDA_MEMORY_FRACTION", "1.0"))

        return {
            "status": "available",
            "name": gpu_name,
            "total_memory_gb": round(total_memory_gb, 2),
            "memory_limit_gb": round(total_memory_gb * memory_fraction, 2),
            "memory_fraction": memory_fraction,
        }
    except Exception as e:
        logger.warning(f"Failed to query GPU properties: {e}")
        return {"status": "error", "error": str(e)}


def clear_gpu_memory() -> None:
    """Aggressively clear PyTorch CUDA cache and invoke garbage collection."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()


def get_current_gpu_memory() -> dict[str, float] | None:
    """Get current GPU memory allocation and usage statistics in GB."""
    if not torch.cuda.is_available():
        return None

    try:
        allocated_gb = torch.cuda.memory_allocated(0) / (1024**3)
        reserved_gb = torch.cuda.memory_reserved(0) / (1024**3)
        total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

        return {
            "allocated_gb": round(allocated_gb, 2),
            "reserved_gb": round(reserved_gb, 2),
            "total_gb": round(total_gb, 2),
            "usage_percent": round((allocated_gb / total_gb) * 100, 1) if total_gb > 0 else 0.0,
        }
    except Exception as e:
        logger.debug(f"Failed to get current GPU memory: {e}")
        return None


def print_gpu_status() -> None:
    """Log formatted GPU status."""
    info = get_gpu_info()
    mem = get_current_gpu_memory()

    if info.get("status") == "no_gpu":
        logger.warning("⚠️ No GPU available")
        return

    logger.info(f"🎮 GPU: {info.get('name', 'Unknown')}")
    logger.info(f"   Total: {info.get('total_memory_gb', 0)}GB | Limit: {info.get('memory_limit_gb', 0)}GB")

    if mem:
        logger.info(f"   Current: {mem['allocated_gb']}GB ({mem['usage_percent']}%) | Reserved: {mem['reserved_gb']}GB")


def auto_detect_max_streams(
    config_val: int = 0,
    per_stream_gib: float = 2.0,
    min_streams: int = 1,
    max_streams_cap: int = 8,
) -> int:
    """Determine the optimal number of parallel live avatar streams for the GPU.

    If ``config_val > 0``, honors the user configuration directly.
    Otherwise (when ``config_val <= 0`` or auto), inspects total GPU VRAM:
      * Allocates ~2.5 GB baseline for model weights + runtime.
      * Each active stream needs ~1.5 - 2.0 GB transient VRAM during render bursts.
      * Clamps between ``min_streams`` (1) and ``max_streams_cap``.
    """
    if config_val > 0:
        return max(1, int(config_val))

    if not torch.cuda.is_available():
        return 1

    try:
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        # Model baseline: ~2.5 GB. Each concurrent stream takes ~2.0 GB.
        available_for_streams = max(0.0, total_vram_gb - 2.5)
        streams = max(min_streams, min(max_streams_cap, int(available_for_streams // per_stream_gib)))
        return streams
    except Exception as e:
        logger.warning(f"Error auto-detecting max streams, falling back to 1: {e}")
        return 1
