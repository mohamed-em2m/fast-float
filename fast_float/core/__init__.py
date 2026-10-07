from fast_float.core.state import AppState, app_state
from fast_float.core.service import generate_video, run_inference_blocking

try:
    from fast_float.core.inference_optimized import OptimizedInferenceAgent, InferenceOptions
except Exception:
    OptimizedInferenceAgent = None
    InferenceOptions = None

__all__ = [
    "AppState",
    "app_state",
    "generate_video",
    "run_inference_blocking",
    "OptimizedInferenceAgent",
    "InferenceOptions",
]
