from fast_float.core.state import AppState, app_state
from fast_float.core.service import generate_video, run_inference_blocking

try:
    from fast_float.core.inference_optimized import OptimizedInferenceAgent
    from fast_float.core.data_processor import DataProcessor
    from fast_float.core.options import InferenceOptions
    from fast_float.core.simple_encoder import SimpleVideoEncoder
    from fast_float.core.nvenc_encoder import encode_video_nvenc
except Exception:
    OptimizedInferenceAgent = None
    InferenceOptions = None
    DataProcessor = None
    SimpleVideoEncoder = None
    encode_video_nvenc = None

__all__ = [
    "AppState",
    "app_state",
    "generate_video",
    "run_inference_blocking",
    "OptimizedInferenceAgent",
    "InferenceOptions",
    "DataProcessor",
    "SimpleVideoEncoder",
    "encode_video_nvenc",
]
