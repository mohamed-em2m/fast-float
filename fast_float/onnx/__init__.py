"""Torch -> ONNX conversion toolkit (one converter per sub-model).

Each converter lives in its own module (``convert_emotion``, ``convert_...``)
and shares :mod:`fast_float.onnx.export` (dynamo-first export) and
:mod:`fast_float.onnx.parity` (torch-vs-ORT checks).
"""

from fast_float.onnx.backends import (
    BackendConfig,
    ComponentConfig,
    cuda_ep_usable,
    default_config,
    env_fingerprint,
    load_backends,
    save_backends,
)
from fast_float.onnx.export import (
    add_ort_args,
    export_module,
    fp16_tol,
    make_ort_session,
    make_session_from_args,
    trt_optimization_profiles,
)
from fast_float.onnx.parity import check_parity
from fast_float.onnx.runtime import (
    EmotionOnnxBackend,
    FMTStepBackend,
    Wav2vecOnnxBackend,
)

__all__ = [
    "BackendConfig",
    "ComponentConfig",
    "EmotionOnnxBackend",
    "FMTStepBackend",
    "Wav2vecOnnxBackend",
    "add_ort_args",
    "check_parity",
    "cuda_ep_usable",
    "default_config",
    "env_fingerprint",
    "export_module",
    "fp16_tol",
    "load_backends",
    "make_ort_session",
    "make_session_from_args",
    "save_backends",
    "trt_optimization_profiles",
]
