"""DecoderOnnx batch-set discovery + per-call selection (CPU, tiny graphs).

``convert_decoder`` writes a set of static-batch artifacts (the primary
``--out`` plus ``<stem>_b<N>.onnx`` siblings); these tests build 2-element
identity graphs instead of the real decoder so the runner's discovery, lazy
session loading, and smallest-covering-batch selection stay pure-CPU.
"""

import pytest
import torch

import onnx
from fast_float.runtimes.onnx.runner import DecoderOnnx, _discover_decoder_batches
from onnx import TensorProto, helper


def _static_decoder(path, batch) -> None:
    """Minimal static-batch graph: ``latent [B, 3, 2, 2] -> img [B, 3, 2, 2]``.

    The output must be 4D NCHW frames: ``DecoderOnnx`` validates that format at
    session build, because the decode path quantizes the output as NCHW.
    """
    node = helper.make_node("Identity", ["latent"], ["img"])
    graph = helper.make_graph(
        [node],
        "dec",
        [helper.make_tensor_value_info("latent", TensorProto.FLOAT, [batch, 3, 2, 2])],
        [helper.make_tensor_value_info("img", TensorProto.FLOAT, [batch, 3, 2, 2])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    # Pin IR 10 like the exported graphs: ORT 1.23 caps model IR at 11, while a
    # recent onnx defaults make_model() to a newer IR the runtime rejects.
    model.ir_version = 10
    onnx.save(model, str(path))


@pytest.fixture
def decoder_set(tmp_path):
    primary = tmp_path / "decoder_v1.onnx"
    _static_decoder(primary, 8)
    for b in (1, 2, 4, 16):
        _static_decoder(tmp_path / f"decoder_v1_b{b}.onnx", b)
    return primary


def test_discovers_sibling_batches(decoder_set):
    assert sorted(_discover_decoder_batches(str(decoder_set))) == [1, 2, 4, 16]


def test_largest_batch_and_lazy_sessions(decoder_set):
    runner = DecoderOnnx(str(decoder_set), use_gpu=False)
    assert runner.batches == [1, 2, 4, 8, 16]
    assert runner.batch == 16
    # Only the primary is loaded up front; the rest are lazy.
    assert set(runner._sessions) == {8}
    # m=1 -> batch-1 session; m=3 -> smallest covering is 4.
    runner.forward(torch.randn(1, 3, 2, 2), [])
    assert 1 in runner._sessions
    runner.forward(torch.randn(3, 3, 2, 2), [])
    assert set(runner._sessions) == {1, 4, 8}
    # m=8 stays on the primary; m=16 uses the largest.
    runner.forward(torch.randn(8, 3, 2, 2), [])
    runner.forward(torch.randn(16, 3, 2, 2), [])
    assert set(runner._sessions) == {1, 4, 8, 16}
    assert 2 not in runner._sessions


def test_pad_and_trim_returns_m(decoder_set):
    runner = DecoderOnnx(str(decoder_set), use_gpu=False)
    out = runner.forward(torch.randn(3, 3, 2, 2), [])
    assert tuple(out.shape) == (3, 3, 2, 2)


def test_single_artifact_still_works(tmp_path):
    path = tmp_path / "decoder_v1.onnx"
    _static_decoder(path, 8)
    runner = DecoderOnnx(str(path), use_gpu=False)
    assert _discover_decoder_batches(str(path)) == {}
    assert runner.batches == [8]
    assert runner.batch == 8
    out = runner.forward(torch.randn(4, 3, 2, 2), [])
    assert tuple(out.shape) == (4, 3, 2, 2)


def test_rejects_batch_larger_than_max(decoder_set):
    runner = DecoderOnnx(str(decoder_set), use_gpu=False)
    with pytest.raises(ValueError, match="exceeds static ONNX batch"):
        runner.forward(torch.randn(32, 3, 2, 2), [])
