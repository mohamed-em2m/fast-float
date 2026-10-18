"""Inspect: live session discovery and the resolved-plan warnings."""

import torch.nn as nn

from fast_float.runtimes.base_runtime import RuntimeSession
from fast_float.runtimes.inspect import inspect_model, plan_warnings


class _Sess(RuntimeSession):
    component = "fmt"
    backend = "onnx"

    def forward(self, x):
        return x


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = _Sess()  # nn.Module submodule
        self._plain = _Sess()
        self._plain.component = "emotion"  # plain attribute


def test_inspect_model_finds_module_and_plain_attributes():
    model = _Model()
    model._plain.degrade("artifact missing")

    rows = {r["attr"]: r for r in inspect_model(model)}
    assert rows["enc"]["component"] == "fmt"
    assert rows["enc"]["degraded"] is False
    assert rows["_plain"]["component"] == "emotion"
    assert rows["_plain"]["degraded"] is True
    assert rows["_plain"]["reason"] == "artifact missing"


def test_inspect_model_is_empty_without_sessions():
    assert inspect_model(nn.Linear(2, 2)) == []


def _row(component, mode, provider=None, path=None, exists=None):
    return {
        "component": component,
        "mode": mode,
        "provider": provider,
        "onnx_path": path,
        "onnx_exists": exists,
    }


def test_plan_warnings_flag_missing_artifacts_and_providers():
    rows = [
        _row("fmt", "onnx", "trt", "fmt.onnx", exists=False),
        _row("decoder", "onnx", "cuda", "decoder.onnx", exists=True),
    ]
    warnings = plan_warnings(rows, {"trt": False, "cuda_ep": False}, compile_available=True)
    assert any("fmt" in w and "not found" in w for w in warnings)
    assert any("fmt" in w and "TensorRT" in w for w in warnings)
    assert any("decoder" in w and "CUDA EP" in w for w in warnings)


def test_plan_warnings_compile_unavailable():
    rows = [_row("direction", "compile")]
    warnings = plan_warnings(rows, {}, compile_available=False)
    assert warnings == ["direction: compile selected but torch.compile is unavailable; would stay eager"]


def test_plan_warnings_all_eager_note():
    rows = [_row("fmt", "eager"), _row("audio", "eager")]
    assert plan_warnings(rows, {}, compile_available=True) == ["all components eager (no acceleration configured)"]


def test_plan_warnings_quiet_when_honoured():
    rows = [_row("fmt", "onnx", "cuda", "fmt.onnx", exists=True)]
    assert plan_warnings(rows, {"cuda_ep": True}, compile_available=True) == []
