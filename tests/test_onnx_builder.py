"""OnnxBuilder: path resolution, fingerprinting, cache, and the CPU runtime path."""

import pytest
import torch

import onnx
from fast_float.builders.onnx import OnnxBuilder
from fast_float.builders.spec import BuildArtifact, BuildSpec
from fast_float.onnx.parity import compare_outputs
from onnx import TensorProto, helper


def _tiny_onnx(path):
    """A minimal ``y = x + x`` graph so the ORT path runs on CPU."""
    node = helper.make_node("Add", ["x", "x"], ["y"])
    graph = helper.make_graph(
        [node],
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, 3])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [None, 3])],
    )
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), str(path))


def test_out_path_prefers_spec_extra(tmp_path):
    builder = OnnxBuilder(tmp_path / "m.json")
    spec = BuildSpec(component="direction", backend="onnx", extra={"out_path": "custom.onnx"})
    assert builder.out_path(spec) == "custom.onnx"


def test_out_path_falls_back_to_registry_default(tmp_path):
    builder = OnnxBuilder(tmp_path / "m.json")
    spec = BuildSpec(component="direction", backend="onnx")
    assert builder.out_path(spec) == "./onnx/direction_v1.onnx"


def test_fingerprint_uses_module_class_from_extra(tmp_path):
    builder = OnnxBuilder(tmp_path / "m.json")
    a = BuildSpec(component="direction", backend="onnx", extra={"module_class": "A"})
    b = BuildSpec(component="direction", backend="onnx", extra={"module_class": "B"})
    assert builder.fingerprint(a) != builder.fingerprint(b)


def test_lookup_miss_then_record_roundtrip(tmp_path):
    builder = OnnxBuilder(tmp_path / "m.json")
    spec = BuildSpec(component="direction", backend="onnx", input_names=("x",))
    assert builder.lookup(spec) is None

    fingerprint = builder.fingerprint(spec)
    builder.record(
        BuildArtifact(
            component="direction",
            kind="onnx",
            path="d.onnx",
            provider="cpu",
            fingerprint=fingerprint,
            parity={"ok": True},
        )
    )
    got = builder.lookup(spec)
    assert got is not None
    assert got.path == "d.onnx"
    assert got.provider == "cpu"
    assert got.parity == {"ok": True}


def test_build_requires_module_and_sample_inputs(tmp_path):
    builder = OnnxBuilder(tmp_path / "m.json")
    spec = BuildSpec(component="direction", backend="onnx", input_names=("x",), output_names=("y",))
    with pytest.raises(ValueError, match="module"):
        builder.build(spec)


def test_runtime_cpu_forward_parity_and_stats(tmp_path):
    onnx_path = tmp_path / "tiny.onnx"
    _tiny_onnx(onnx_path)
    builder = OnnxBuilder(tmp_path / "m.json")
    spec = BuildSpec(
        component="direction",
        backend="onnx",
        input_names=("x",),
        opt_shapes={"x": (1, 3)},
        precision="fp32",
        provider_chain=("cpu",),
        extra={"out_path": str(onnx_path)},
    )

    runtime = builder.runtime(spec)
    x = torch.randn(2, 3)
    outs = runtime(x)

    assert isinstance(outs, list)
    assert tuple(outs[0].shape) == (2, 3)
    assert runtime.providers == ["CPUExecutionProvider"]
    assert runtime.stats.calls == 1
    assert compare_outputs(x + x, outs[0])["ok"]
