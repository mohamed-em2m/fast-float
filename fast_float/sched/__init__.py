from __future__ import annotations

# Native FLOAT Omni Scheduling & Streaming Pipeline
from fast_float.sched.types import (
    RequestStatus,
    ClipCallback,
    FloatOmniRequest,
    LatentChunkTask,
)
from fast_float.sched.omni_scheduler import FloatOmniScheduler
from fast_float.sched.pipeline_engine import FloatOmniPipelineEngine

# Optional vLLM-Omni reference classes (if vllm is installed in environment)
try:
    from fast_float.sched.base_scheduler import BaseScheduler, SchedulerInterface
    from fast_float.sched.interface import (
        DiffusionRequestStatus,
        SchedulerRequestState,
        StepBatchSamplingParamsKey,
        DiffusionSchedulerOutput,
    )
    from fast_float.sched.request_scheduler import RequestScheduler
    from fast_float.sched.step_scheduler import StepScheduler
except ImportError:
    BaseScheduler = None
    SchedulerInterface = None
    DiffusionRequestStatus = None
    SchedulerRequestState = None
    StepBatchSamplingParamsKey = None
    DiffusionSchedulerOutput = None
    RequestScheduler = None
    StepScheduler = None

__all__ = [
    "RequestStatus",
    "ClipCallback",
    "FloatOmniRequest",
    "LatentChunkTask",
    "FloatOmniScheduler",
    "FloatOmniPipelineEngine",
]
