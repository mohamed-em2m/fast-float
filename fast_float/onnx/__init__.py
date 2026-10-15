"""Torch -> ONNX conversion toolkit (one converter per sub-model).

Each converter lives in its own module (``convert_emotion``, ``convert_...``)
and shares :mod:`fast_float.onnx.export` (dynamo-first export) and
:mod:`fast_float.onnx.parity` (torch-vs-ORT checks).
"""

from fast_float.onnx.export import export_module, make_ort_session
from fast_float.onnx.parity import check_parity
from fast_float.onnx.runtime import (
    EmotionOnnxBackend,
    FMTStepBackend,
    Wav2vecOnnxBackend,
)

__all__ = [
    "export_module",
    "make_ort_session",
    "check_parity",
    "EmotionOnnxBackend",
    "Wav2vecOnnxBackend",
    "FMTStepBackend",
]
