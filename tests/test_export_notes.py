"""``export_module`` reports graph dynamics honestly.

Regression guard for the old unconditional ``"dynamo export with symbolic
batch/length"`` note, which contradicted the ``fixed batch: no dynamic marking``
note on the decoder (static-batch grouped conv) and over-claimed for units that
mark only some dims.
"""

import torch

from fast_float.onnx.export import export_module


class _Doubler(torch.nn.Module):
    def forward(self, x):
        return x + x


def _export(tmp_path, name, **kw):
    x = torch.randn(1, 3)
    return export_module(
        _Doubler().eval(),
        x,
        str(tmp_path / name),
        input_names=["x"],
        output_names=["y"],
        **kw,
    )


def test_fixed_batch_note_is_static(tmp_path):
    info = _export(tmp_path, "fixed.onnx", fixed_batch=True)
    notes = " | ".join(info["notes"])
    assert "static batch/length" in notes
    assert "symbolic" not in notes  # the old unconditional claim is gone


def test_all_dims_marked_note(tmp_path):
    info = _export(tmp_path, "dyn.onnx")  # mark_dims defaults to (0, 1)
    assert any("symbolic dims=(0, 1)" in n for n in info["notes"])


def test_partial_mark_note(tmp_path):
    info = _export(tmp_path, "partial.onnx", mark_dims=(0,))
    assert any("symbolic dims=(0,)" in n for n in info["notes"])


def test_no_dims_marked_note(tmp_path):
    info = _export(tmp_path, "static.onnx", mark_dims=())
    assert any("static shapes" in n for n in info["notes"])
