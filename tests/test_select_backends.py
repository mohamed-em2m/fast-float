"""The selector must reach the ONNX runtime through the OnnxBuilder.

Regression: ``main`` used to rebind ``builder`` to the per-component case
factory inside the loop, so ``_select_component`` received a plain function and
``builder.runtime(...)`` raised, silently disabling ONNX for every component.
"""

import torch

import onnx
from fast_float.onnx import select_backends
from fast_float.onnx.backends import load_backends
from onnx import TensorProto, helper


def _identity_onnx(path):
    node = helper.make_node("Identity", ["x"], ["y"])
    graph = helper.make_graph(
        [node],
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, 3])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [None, 3])],
    )
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), str(path))


def test_main_passes_the_builder_not_the_case_factory(tmp_path, monkeypatch):
    _identity_onnx(tmp_path / "direction_v1.onnx")

    class Identity(torch.nn.Module):
        def forward(self, x):
            return x

    def make_case(args, device):
        x = torch.randn(2, 3, device=device)
        return select_backends.Case("direction", Identity().to(device), (x,), {"x": x.detach().cpu().numpy()})

    monkeypatch.setattr(select_backends, "BUILDERS", {"direction": make_case})
    out = tmp_path / "backends.yaml"
    rc = select_backends.main(
        [
            "--components",
            "direction",
            "--onnx-dir",
            str(tmp_path),
            "--out",
            str(out),
            "--manifest",
            str(tmp_path / "m.json"),
            "--cpu-ort",
            "--no-compile",
            "--no-global",
            "--iters",
            "2",
            "--warmup",
            "1",
        ]
    )
    assert rc == 0
    metrics = load_backends(str(out)).components["direction"].metrics
    assert "onnx_error" not in metrics, metrics
    assert "onnx_ms" in metrics
