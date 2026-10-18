"""Compatibility shim: the ONNX runtime backends moved.

The emotion/audio/fmt ORT sessions now live in
``fast_float.runtimes.onnx.backends`` (Phase 2 of
``docs/BUILD_RUNTIME_SESSIONS.md``); this module re-exports them so existing
imports keep working.
"""

from __future__ import annotations

from fast_float.runtimes.onnx.backends import (
    LABELS,
    EmotionOnnxBackend,
    FMTStepBackend,
    Wav2vecOnnxBackend,
)

__all__ = ["LABELS", "EmotionOnnxBackend", "FMTStepBackend", "Wav2vecOnnxBackend"]
