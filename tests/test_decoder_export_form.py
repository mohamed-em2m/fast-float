"""The decoder artifact must be the model, not a grouped-conv substitute.

Regression: the exporter used to replace every ``ModulatedConv2d`` with a
grouped variant (``groups=batch`` baked into the ONNX attribute). The ORT CUDA
EP can only serve grouped convs through its *fallback* kernel, which drifts
~90 uint8 LSB from torch's and is not even run-to-run reproducible -- the
artifact then could not reproduce the frames the eager path produces, and the
uint8 parity gate failed on a box where the provider is the CUDA EP.

The graph is now traced from the model itself (dense modulated convs), still at
a fixed batch, so the artifact set and the runtime's pad/trim are unchanged.
"""

from __future__ import annotations

import pytest
import torch

from fast_float.models.float.styledecoder import ModulatedConv2d


def test_the_deployed_decoder_has_no_grouped_convolutions():
    """The model must be dense: no conv bakes ``groups=batch``."""
    opt = pytest.importorskip("fast_float.models.float.styledecoder")
    src = pytest.importorskip("inspect").getsource(opt.ModulatedConv2d.forward)
    assert "groups=" not in src, "ModulatedConv2d must stay a dense conv (no groups=batch)"


def test_the_export_module_keeps_the_deployed_conv():
    """The export module must not substitute the conv implementation."""
    from fast_float.onnx.convert_decoder import DecoderExport

    class _Conv(torch.nn.Module):
        def forward(self, x, style):
            return x

    class _Dec(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = ModulatedConv2d(8, 8, 3, 16)

        def forward(self, latent, alpha=None, feats=None):
            return self.conv(torch.zeros(1, 8, 4, 4), torch.zeros(1, 16))

    dec = _Dec()
    module = DecoderExport(dec, 0)
    assert type(module.dec.conv) is ModulatedConv2d, (
        "the exported module must contain the deployed ModulatedConv2d, not a substitute"
    )
    assert _Conv is not type(module.dec.conv)


def test_the_decoder_graph_has_no_grouped_conv_nodes(tmp_path):
    """The exported ONNX graph must not contain a grouped convolution."""
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxscript")
    pytest.importorskip("onnxruntime")

    from fast_float.onnx.convert_decoder import DecoderExport, _fold_static_weights
    from fast_float.onnx.export import export_module

    class _Dec(torch.nn.Module):
        """A tiny stand-in with the same conv shape as the real decoder."""

        def __init__(self):
            super().__init__()
            self.conv = ModulatedConv2d(8, 8, 3, 16, upsample=False)

        def forward(self, latent, alpha=None, feats=None):
            x = self.conv(torch.zeros(1, 8, 8, 8), torch.zeros(1, 16))
            return x, None

    dec = _Dec()
    _fold_static_weights(dec)
    module = DecoderExport(dec, 0).eval()
    out = str(tmp_path / "decoder.onnx")
    export_module(
        module,
        (torch.zeros(1, 512),),
        out,
        input_names=["latent"],
        output_names=["img"],
        dynamic_axes={},
        opset=17,
        fixed_batch=True,
    )
    model = onnx.load(out)
    groups = [
        n.attribute[0].i
        for n in model.graph.node
        if n.op_type == "Conv" and n.attribute and n.attribute[0].name == "group"
    ]
    assert all(g == 1 for g in groups), f"the graph bakes grouped convs (groups={sorted(set(groups))})"
