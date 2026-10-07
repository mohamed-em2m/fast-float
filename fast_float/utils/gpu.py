import os
import gc
from typing import Optional, Dict, Any
import torch

from fast_float.utils.logger import logger


def get_gpu_info() -> Dict[str, Any]:
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


def get_current_gpu_memory() -> Optional[Dict[str, float]]:
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
        logger.info(
            f"   Current: {mem['allocated_gb']}GB ({mem['usage_percent']}%) | Reserved: {mem['reserved_gb']}GB"
        )
