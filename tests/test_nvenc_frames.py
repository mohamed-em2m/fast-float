"""CPU-only tests for the tensor -> NVENC frame conversion."""

import numpy as np
import pytest
import torch

from fast_float.core.nvenc_encoder import frames_to_uint8_hwc


def test_nchw_float_maps_to_uint8_hwc():
    vid = torch.zeros(2, 3, 4, 5)
    vid[:, :, 0, 0] = -1.0
    vid[:, :, 1, 1] = 1.0
    out = frames_to_uint8_hwc(vid)
    assert out.shape == (2, 4, 5, 3)
    assert out.dtype == np.uint8
    assert out.flags["C_CONTIGUOUS"]
    assert out[0, 0, 0].tolist() == [0, 0, 0]
    assert out[0, 1, 1].tolist() == [255, 255, 255]
    assert out[0, 0, 1].tolist() == [127, 127, 127]


def test_hwc_uint8_is_not_repermuted():
    vid = torch.randint(0, 255, (2, 4, 5, 3), dtype=torch.uint8)
    out = frames_to_uint8_hwc(vid)
    assert out.shape == (2, 4, 5, 3)
    assert np.array_equal(out, vid.numpy())


def test_rejects_non_4d():
    with pytest.raises(ValueError, match="4D"):
        frames_to_uint8_hwc(torch.zeros(3, 4, 5))
