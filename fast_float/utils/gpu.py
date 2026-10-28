import gc
import json
import os
from dataclasses import dataclass
from typing import Any

import torch

from fast_float.utils.logger import logger

_GIB = 1024**3

# Capacity planning knobs. Live avatar renders are bursty (a 2 s window is
# rendered in a few hundred ms), so both the VRAM footprint and the GPU time
# per stream matter; the usable number is the smaller of the two.
SAFETY_RESERVE_GIB = 1.0  # driver, CUDA context, allocator fragmentation
MODEL_BASELINE_GIB = 2.5  # weights + runtime when this process has not loaded yet
PER_STREAM_OVERHEAD_GIB = 0.5  # encoder scratch + per-stream buffers on top of activations
DEFAULT_PER_STREAM_GIB = 2.0  # fallback when no decode profile has been measured
DEFAULT_TARGET_UTILIZATION = 0.85  # keep 15% headroom for spikes / other work
MAX_STREAMS_CAP = 8  # hard ceiling; more than this is not worth the GPU time
MIN_STREAMS = 1


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


@dataclass(frozen=True)
class StreamCapacity:
    """Why a particular concurrent-stream count was chosen.

    ``memory_bound`` is what the remaining VRAM allows, ``compute_bound`` is
    what the measured render speed allows (``None`` until a real-time factor
    has been observed), and ``recommended`` is the smaller of the two, capped.
    """

    recommended: int
    memory_bound: int
    compute_bound: int | None
    free_vram_gib: float
    per_stream_gib: float
    observed_rtf: float | None
    target_utilization: float
    max_cap: int
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "recommended": self.recommended,
            "memory_bound": self.memory_bound,
            "compute_bound": self.compute_bound,
            "free_vram_gib": self.free_vram_gib,
            "per_stream_gib": self.per_stream_gib,
            "observed_rtf": self.observed_rtf,
            "target_utilization": self.target_utilization,
            "max_cap": self.max_cap,
            "reason": self.reason,
            "summary": self.summary(),
        }

    def summary(self) -> str:
        """One-line, log/API friendly description of the decision."""
        compute = "?" if self.compute_bound is None else str(self.compute_bound)
        rtf = "?" if self.observed_rtf is None else f"{self.observed_rtf:.2f}"
        return (
            f"streams={self.recommended} (memory={self.memory_bound}, compute={compute}, "
            f"rtf={rtf}, free={self.free_vram_gib:.1f}GiB, "
            f"per_stream={self.per_stream_gib:.2f}GiB) - {self.reason}"
        )


def measured_per_stream_gib() -> float | None:
    """Peak transient VRAM of *one* stream from the persisted decode profile.

    The decode auto-tuner measures the largest activation batch that fits and
    stores its peak bytes in ``decode_tune.json``. That is a measured floor for
    a single stream, far better than a hard-coded guess. Returns ``None`` when
    no profile has been written yet.
    """
    try:
        from fast_float.utils.decode_tuner import cache_path

        path = cache_path()
        if not path or not os.path.exists(path):
            return None
        with open(path) as fh:
            profiles = json.load(fh)
        if not isinstance(profiles, dict):
            return None
        transients = [
            float(entry.get("transient_bytes") or 0)
            for entry in profiles.values()
            if isinstance(entry, dict)
        ]
        transients = [t for t in transients if t > 0]
        if not transients:
            return None
        return max(transients) / _GIB
    except Exception as exc:  # a corrupt cache must not break scheduling
        logger.debug(f"Could not read the decode tuning profile: {exc}")
        return None


def estimate_stream_capacity(
    *,
    config_val: int = 0,
    observed_rtf: float | None = None,
    per_stream_gib: float | None = None,
    target_utilization: float = DEFAULT_TARGET_UTILIZATION,
    min_streams: int = MIN_STREAMS,
    max_streams_cap: int = MAX_STREAMS_CAP,
) -> StreamCapacity:
    """Estimate how many concurrent avatar streams this GPU can serve.

    The answer is ``min(memory_bound, compute_bound)``, clamped to
    ``[min_streams, max_streams_cap]``:

    * **memory** — VRAM that is still usable *now* (driver-free memory plus
      cached-but-unallocated blocks this process can reuse), divided by one
      stream's peak footprint. Using free memory instead of total VRAM is what
      keeps the estimate honest on a box that also runs other processes.
    * **compute** — a stream needs ``rtf`` GPU-seconds per second of audio, so
      ``target_utilization / rtf`` streams saturate the target utilisation.
      ``observed_rtf`` is the wall-clock render time over audio duration from a
      real request; without it the compute bound is unknown and only memory is
      enforced.

    ``config_val > 0`` is an operator override: it wins over the estimate, but
    the measured bounds are still reported so callers can see both the
    hardware estimate and the configured choice.
    """
    if per_stream_gib is None:
        measured = measured_per_stream_gib()
        per_stream_gib = max(
            DEFAULT_PER_STREAM_GIB,
            (measured + PER_STREAM_OVERHEAD_GIB) if measured else 0.0,
        )
    per_stream_gib = max(0.1, float(per_stream_gib))

    free_gib = 0.0
    usable_gib = 0.0
    if torch.cuda.is_available():
        try:
            free_bytes, _total_bytes = torch.cuda.mem_get_info()
            free_gib = free_bytes / _GIB
            allocated_gib = torch.cuda.memory_allocated() / _GIB
            reserved_gib = torch.cuda.memory_reserved() / _GIB
            # Blocks this process cached are reusable without new allocation.
            reusable_gib = max(0.0, reserved_gib - allocated_gib)
            usable_gib = free_gib + reusable_gib - SAFETY_RESERVE_GIB
            if allocated_gib < 1.0:
                # Nothing is loaded in this process yet: the model still needs
                # its baseline before any stream can render.
                usable_gib -= MODEL_BASELINE_GIB
        except Exception as exc:
            logger.warning(f"Error reading GPU memory for stream capacity: {exc}")
    usable_gib = max(0.0, usable_gib)
    memory_bound = int(usable_gib // per_stream_gib)

    compute_bound: int | None = None
    if observed_rtf is not None and observed_rtf > 0:
        compute_bound = max(0, int(target_utilization / float(observed_rtf)))

    limits = [memory_bound]
    if compute_bound is not None:
        limits.append(compute_bound)
    estimate = max(min_streams, min(max_streams_cap, *limits))
    recommended = estimate

    if not torch.cuda.is_available():
        reason = "no CUDA device; serial fallback"
    elif compute_bound is not None and compute_bound <= memory_bound:
        reason = (
            f"compute-bound: {float(observed_rtf or 0.0):.2f}s render per 1s of audio "
            f"-> {compute_bound} at {target_utilization:.0%} target "
            f"(memory allows {memory_bound})"
        )
    elif memory_bound < (compute_bound or max_streams_cap):
        reason = (
            f"memory-bound: {usable_gib:.1f}GiB usable / "
            f"{per_stream_gib:.2f}GiB per stream -> {memory_bound} "
            f"(compute allows {compute_bound if compute_bound is not None else '?'})"
        )
    else:
        reason = (
            f"capped at {max_streams_cap}: memory allows {memory_bound}, "
            f"compute allows {compute_bound if compute_bound is not None else '?'}"
        )

    if config_val > 0:
        recommended = max(min_streams, int(config_val))
        reason = (
            f"configured livekit_max_streams={config_val}; estimate would be "
            f"{estimate} (memory={memory_bound}, "
            f"compute={compute_bound if compute_bound is not None else '?'})"
        )

    return StreamCapacity(
        recommended=recommended,
        memory_bound=memory_bound,
        compute_bound=compute_bound,
        free_vram_gib=round(free_gib, 2),
        per_stream_gib=round(per_stream_gib, 2),
        observed_rtf=None if observed_rtf is None else round(float(observed_rtf), 3),
        target_utilization=target_utilization,
        max_cap=max_streams_cap,
        reason=reason,
    )


def auto_detect_max_streams(
    config_val: int = 0,
    per_stream_gib: float | None = None,
    min_streams: int = MIN_STREAMS,
    max_streams_cap: int = MAX_STREAMS_CAP,
) -> int:
    """Backwards-compatible wrapper around :func:`estimate_stream_capacity`."""
    return estimate_stream_capacity(
        config_val=config_val,
        per_stream_gib=per_stream_gib,
        min_streams=min_streams,
        max_streams_cap=max_streams_cap,
    ).recommended
