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
    make_runtime_session,
    make_session_from_args,
    trt_optimization_profiles,
)
from fast_float.onnx.parity import check_parity

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
    "make_runtime_session",
    "make_session_from_args",
    "save_backends",
    "trt_optimization_profiles",
]

# ``fast_float.onnx.runtime`` re-exports the ctor-wired ORT backends, which live
# under ``fast_float.runtimes.onnx`` and import ``fast_float.onnx.export``.
# Importing them eagerly here created an import cycle when a caller touched
# ``fast_float.runtimes.onnx.*`` first (backends -> onnx.export -> onnx/__init__
# -> onnx.runtime -> runtimes.onnx.backends, still initialising -> ImportError).
# Re-export lazily so the runner/backends import cleanly in either order.
_RUNTIME_BACKENDS = ("EmotionOnnxBackend", "FMTStepBackend", "Wav2vecOnnxBackend")


def __getattr__(name):
    if name in _RUNTIME_BACKENDS:
        from fast_float.onnx.runtime import (
            EmotionOnnxBackend,
            FMTStepBackend,
            Wav2vecOnnxBackend,
        )

        return {
            "EmotionOnnxBackend": EmotionOnnxBackend,
            "FMTStepBackend": FMTStepBackend,
            "Wav2vecOnnxBackend": Wav2vecOnnxBackend,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
