"""Decoder parity is gated in the uint8 frame domain, not in float space.

Regression: the decoder gate relaxed ``atol`` by 40x for TensorRT FP16 and then
fell back to a "functional" branch, so a graph that was off by 4.375 with
cosine 0.983 shipped as PASS. On a 512x512 frame that is a visibly broken face.
"""

import numpy as np
import pytest
import torch

from fast_float.onnx import convert_decoder, export


def _rep(max_lsb: int, mean_lsb: float) -> dict:
    return {"max_lsb": max_lsb, "mean_lsb": mean_lsb, "cosine": 1.0}


def test_u8_lsb_matches_the_runtime_quantization():
    ref = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    out = ref + np.float32(1.0 / 127.5)  # one LSB (only the 0.0 bucket moves)
    max_lsb, mean_lsb = convert_decoder._u8_lsb(ref, out)
    assert max_lsb == 1
    assert mean_lsb == pytest.approx(0.2)


def test_u8_lsb_is_zero_for_sub_lsb_drift():
    # Values park between two truncation boundaries: a 1e-6 drift stays inside
    # one uint8 level, i.e. the frame is bit-identical on screen.
    ref = torch.zeros(2, 3, 8, 8)
    assert convert_decoder._u8_lsb(ref, ref + 1e-6) == (0, 0.0)


def test_u8_lsb_clamps_like_the_writer():
    ref = np.array([[-5.0, 5.0]], dtype=np.float32)
    out = np.array([[-4.0, 4.0]], dtype=np.float32)
    # both clamp to 0 / 255, so the huge float drift is invisible on screen
    assert convert_decoder._u8_lsb(ref, out) == (0, 0.0)


@pytest.mark.parametrize(
    ("max_lsb", "mean_lsb", "ok"),
    [(0, 0.0, True), (2, 0.5, True), (3, 0.0, False), (1, 0.6, False)],
)
def test_gate_rejects_lsb_beyond_the_limit(max_lsb, mean_lsb, ok):
    args = type("A", (), {"max_lsb": 2, "max_mean_lsb": 0.5})()
    assert convert_decoder._gate(_rep(max_lsb, mean_lsb), args) is ok


def test_runtime_session_honours_the_no_trt_marker(tmp_path, monkeypatch):
    model = tmp_path / "decoder_v1.onnx"
    model.write_bytes(b"")
    seen = {}

    def fake_session(path, **kwargs):
        seen.update(kwargs)

        class Session:
            @staticmethod
            def get_providers():
                return ["CPUExecutionProvider"]

        return Session()

    monkeypatch.setattr(export, "make_ort_session", fake_session)
    monkeypatch.setattr(export.torch.cuda, "is_available", lambda: True)

    export.make_runtime_session(str(model))
    assert seen["use_trt"] is True

    (tmp_path / f"decoder_v1.onnx{export.TRT_DISABLED_MARKER}").write_text("veto", encoding="utf-8")
    export.make_runtime_session(str(model))
    assert seen["use_trt"] is False
