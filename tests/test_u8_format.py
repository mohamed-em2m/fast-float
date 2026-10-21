"""Formatting contract of the deployed decode path.

The decode path reads the decoder output as NCHW float frames in [-1, 1] and
writes uint8 HWC. Every downstream consumer (frame pool, NVENC, ffmpeg) assumes
exactly that, so the conversion must be channel-correct, layout-independent and
identical to the formula the converter gates parity against.
"""

import numpy as np
import pytest
import torch

from fast_float.models.float.decoder import FloatDecoder
from fast_float.runtimes.onnx.runner import DecoderOnnx


def _rgb_pattern():
    x = torch.zeros(2, 3, 3, 4)
    x[:, 0] = -1.0
    x[:, 1] = 0.0
    x[:, 2] = 1.0
    return x


def test_channel_order_is_c_to_last():
    out = FloatDecoder._quantize_u8_hwc(_rgb_pattern())
    assert tuple(out.shape) == (2, 3, 4, 3)
    assert (out[..., 0] == 0).all()
    assert (out[..., 1] == 127).all()
    assert (out[..., 2] == 255).all()


def test_layout_and_dtype_do_not_change_the_bytes():
    x = _rgb_pattern()
    ref = FloatDecoder._quantize_u8_hwc(x)
    assert torch.equal(FloatDecoder._quantize_u8_hwc(x.to(memory_format=torch.channels_last)), ref)
    assert torch.equal(FloatDecoder._quantize_u8_hwc(x.permute(0, 2, 3, 1).permute(0, 3, 1, 2)), ref)
    assert torch.equal(FloatDecoder._quantize_u8_hwc(x.half()), ref)


def test_matches_the_converter_gate_formula():
    x = torch.linspace(-1.5, 1.5, 2001).view(1, 1, 1, 2001)
    got = FloatDecoder._quantize_u8_hwc(x).reshape(-1)
    arr = x.reshape(-1).numpy().astype(np.float32)
    want = np.clip((arr + 1.0) * 127.5, 0.0, 255.0).astype(np.uint8)
    assert torch.equal(got, torch.from_numpy(want))


def test_out_of_range_saturates_instead_of_wrapping():
    z = torch.tensor([[-3.0, 3.0, -1.0, 1.0]]).view(1, 1, 1, 4)
    assert FloatDecoder._quantize_u8_hwc(z).reshape(-1).tolist() == [0, 255, 0, 255]


def test_input_is_left_untouched():
    x = _rgb_pattern()
    before = x.clone()
    FloatDecoder._quantize_u8_hwc(x)
    assert torch.equal(x, before)


class _Session:
    def __init__(self, out_shape, providers=("CPUExecutionProvider",)):
        self._out_shape = out_shape
        self._providers = list(providers)

    def get_inputs(self):
        return [type("I", (), {"name": "latent", "shape": [4, 512]})()]

    def get_outputs(self):
        return [type("O", (), {"name": "img", "shape": self._out_shape})()]

    def get_providers(self):
        return self._providers


@pytest.mark.parametrize("shape", [(4, 3, 64, 64), (4, 4, 64, 64)])
def test_read_io_accepts_nchw_frames(shape):
    meta = DecoderOnnx._read_io(_Session(shape), "decoder_b4.onnx")
    assert meta["output_shape"] == shape


@pytest.mark.parametrize("shape", [(4, 64, 64, 3), (4, 512), (4, 2, 64, 64)])
def test_read_io_rejects_non_nchw_output(shape):
    with pytest.raises(ValueError, match="NCHW"):
        DecoderOnnx._read_io(_Session(shape), "decoder_b4.onnx")
