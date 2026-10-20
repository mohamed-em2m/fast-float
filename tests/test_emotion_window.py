"""Emotion must not depend on which backend serves the session.

The emotion graph is exported with symbolic batch/length. Only TensorRT bakes
rigid profile bounds, so ``predict_probs`` must clamp the waveform *only* under
TRT -- on the CUDA/CPU EPs the whole clip has to reach the graph exactly like
the eager SER path, or the same request reads a different emotion purely
because the backend selection changed.
"""

import pytest
import torch

import onnx
from fast_float.runtimes.onnx import backends as backends_mod
from fast_float.runtimes.onnx.backends import EmotionOnnxBackend
from onnx import TensorProto, helper


def _length_emotion_graph(path) -> None:
    """``input_values [B, L]`` -> ``probs [B, 7]``, every entry the clip's length.

    An exact integer probe: any pad/truncate the backend applies before the call
    is reflected in the returned value, with no float tolerance involved.
    """
    nodes = [
        helper.make_node("Shape", ["input_values"], ["shape"]),
        helper.make_node("Gather", ["shape", "one"], ["length"], axis=0),
        helper.make_node("Cast", ["length"], ["length_f"], to=TensorProto.FLOAT),
        helper.make_node("Gather", ["shape", "zero"], ["batch"], axis=0),
        helper.make_node("Concat", ["batch", "seven"], ["out_shape"], axis=0),
        helper.make_node("Expand", ["length_f", "out_shape"], ["probs"]),
    ]
    graph = helper.make_graph(
        nodes,
        "emotion",
        [helper.make_tensor_value_info("input_values", TensorProto.FLOAT, ["B", "L"])],
        [helper.make_tensor_value_info("probs", TensorProto.FLOAT, ["B", 7])],
        initializer=[
            helper.make_tensor("zero", TensorProto.INT64, [1], [0]),
            helper.make_tensor("one", TensorProto.INT64, [1], [1]),
            helper.make_tensor("seven", TensorProto.INT64, [1], [7]),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    # Pin IR 10: ORT 1.23 rejects models above IR 11 (see test_decoder_runner).
    model.ir_version = 10
    onnx.save(model, str(path))


@pytest.fixture
def emotion_backend(tmp_path):
    path = tmp_path / "emotion_v1.onnx"
    _length_emotion_graph(path)
    return EmotionOnnxBackend(str(path))


def test_cpu_session_passes_the_whole_clip(emotion_backend):
    """No clamp on a CPU session: the short clip's own length reaches the graph."""
    out = emotion_backend.predict_probs(torch.zeros(1, 1000))
    assert out.shape == (1, 7)
    assert out[0, 0].item() == 1000.0


def test_cpu_session_does_not_truncate_a_long_clip(emotion_backend):
    """Above TRT's 160000 maximum the full clip still reaches the graph."""
    out = emotion_backend.predict_probs(torch.zeros(1, 200_000))
    assert out[0, 0].item() == 200_000.0


def test_trt_session_still_clamps_to_the_profile(emotion_backend, monkeypatch):
    """Under TRT the profile bounds are honoured (pad up to 16000 / truncate)."""
    monkeypatch.setattr(backends_mod, "_trt_active", lambda session: True)

    assert emotion_backend.predict_probs(torch.zeros(1, 1000))[0, 0].item() == 16000.0
    assert emotion_backend.predict_probs(torch.zeros(1, 200_000))[0, 0].item() == 160000.0
