"""The provider recorded by the selector must reach session construction.

Regression: ``resolve_provider`` was exported but never called, so a component
the selector benchmarked and parity-checked on CUDA/CPU was still deployed on
TensorRT -- a different engine (and different numerics) than the one validated.
"""

import numpy as np
import pytest
import torch

from fast_float.onnx import backends as backends_mod
from fast_float.onnx.backends import (
    BackendConfig,
    ComponentConfig,
    resolve_use_trt,
    save_backends,
)
from fast_float.runtimes.base_runtime import RuntimeStats
from fast_float.runtimes.onnx import backends as rt_backends
from fast_float.runtimes.onnx import runner as runner_mod


@pytest.fixture
def use_config(tmp_path, monkeypatch):
    def _apply(mapping):
        cfg = BackendConfig(
            source="<test>",
            components={
                name: ComponentConfig(name=name, mode="onnx", provider=provider) for name, provider in mapping.items()
            },
        )
        path = tmp_path / "backends.yaml"
        save_backends(cfg, path)
        monkeypatch.setenv(backends_mod.ENV_CONFIG, str(path))
        monkeypatch.setattr(backends_mod, "_CACHED_CONFIG", None)
        return cfg

    return _apply


@pytest.mark.parametrize(
    ("provider", "expected"),
    [("trt", None), ("cuda", False), ("cpu", False), (None, None)],
)
def test_use_trt_follows_the_recorded_provider(use_config, provider, expected):
    use_config({"decoder": provider})
    assert resolve_use_trt("decoder") is expected


def test_use_trt_is_unconstrained_for_unknown_components(use_config):
    use_config({"decoder": "cuda"})
    assert resolve_use_trt("not-a-component") is None


def test_runtime_session_forwards_the_trt_veto(monkeypatch, tmp_path):
    from fast_float.onnx import export

    seen = {}

    def fake_make_ort_session(path, **kwargs):
        seen.update(kwargs)
        return _StubSession()

    monkeypatch.setattr(export, "make_ort_session", fake_make_ort_session)
    monkeypatch.setattr(export.torch.cuda, "is_available", lambda: True)

    export.make_runtime_session(str(tmp_path / "x.onnx"), use_trt=False)
    assert seen["use_trt"] is False

    export.make_runtime_session(str(tmp_path / "x.onnx"))
    assert seen["use_trt"] is True


def test_builder_runtime_honours_the_trt_veto(monkeypatch, tmp_path):
    """The selector must benchmark the provider serving will actually use.

    Regression: ``OnnxBuilder.runtime`` called ``make_ort_session`` directly, so
    a ``.no_trt``-vetoed decoder was benchmarked on TensorRT while
    ``make_runtime_session`` deployed it on the CUDA EP -- the selector chose
    from a candidate config the runtime would never load.
    """
    from fast_float.builders.onnx.builder import OnnxBuilder
    from fast_float.builders.spec import BuildSpec
    from fast_float.onnx import export

    seen = {}

    def fake_make_ort_session(path, **kwargs):
        seen.clear()
        seen.update(kwargs)
        return _StubSession()

    monkeypatch.setattr(export, "make_ort_session", fake_make_ort_session)
    monkeypatch.setattr(export.torch.cuda, "is_available", lambda: True)

    onnx_path = tmp_path / "decoder_v1.onnx"
    onnx_path.write_bytes(b"stub")
    spec = BuildSpec(
        component="decoder",
        backend="onnx",
        input_names=("latent",),
        precision="fp16",
        extra={"out_path": str(onnx_path)},
    )
    builder = OnnxBuilder()

    builder.runtime(spec, device_id=0)
    assert seen["use_trt"] is True

    (tmp_path / f"decoder_v1.onnx{export.TRT_DISABLED_MARKER}").write_text("veto\n", encoding="utf-8")
    builder.runtime(spec, device_id=0)
    assert seen["use_trt"] is False


class _StubSession:
    def get_providers(self):
        return ["CPUExecutionProvider"]

    def get_inputs(self):
        return [type("I", (), {"name": "latent", "shape": [1, 512]})()]

    def get_outputs(self):
        return [type("O", (), {"name": "img", "shape": [1, 3, 8, 8]})()]


def test_runners_pass_the_veto_to_session_construction(monkeypatch, tmp_path):
    seen = {}
    session = _StubSession()

    def fake_make_runtime_session(path, **kwargs):
        seen.update(kwargs)
        return session, False

    monkeypatch.setattr(runner_mod, "make_runtime_session", fake_make_runtime_session)
    path = tmp_path / "x.onnx"
    path.write_bytes(b"stub")
    runner_mod.DirectionOnnx(str(path), use_trt=False)
    assert seen["use_trt"] is False


def test_emotion_backend_sends_c_contiguous_input(monkeypatch):
    backend = object.__new__(rt_backends.EmotionOnnxBackend)
    backend.device_id = None
    backend.stats = RuntimeStats()
    backend.session = _StubSession()
    backend.input_name = "input_values"
    backend.output_name = "probs"
    backend.output_dim = 7
    backend._io_failed = True
    seen = {}

    def fake_run(wav_np):
        seen["arr"] = wav_np
        return np.zeros((wav_np.shape[0], 7), dtype=np.float32)

    backend._run = fake_run
    expanded = torch.zeros(1, 8)[0:1].expand(3, 8)

    out = backend.predict_probs(expanded)

    assert seen["arr"].flags["C_CONTIGUOUS"]
    assert seen["arr"].strides == (8 * 4, 4)
    assert out.shape == (3, 7)
