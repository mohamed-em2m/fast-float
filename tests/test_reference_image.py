"""Reference-image handling: keep the upload's pixels until the face crop.

The request path used to normalize every upload to 256x256, so a 512x512 model
rendered an upscaled 256x256 source and non-square portraits were squashed
before face detection. These tests pin the contract that normalization only
fixes the container (RGB JPEG), and that the response reports the resolution the
frames are really decoded at.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from PIL import Image

from fast_float.routers.generation import _output_resolution
from fast_float.utils.media import TARGET_IMAGE_SIZE, normalize_image


@pytest.mark.parametrize("size", [(512, 512), (480, 600), (1024, 768)])
def test_normalize_image_keeps_the_upload_resolution(tmp_path, size):
    src = tmp_path / "in.png"
    Image.new("RGB", size, (120, 60, 30)).save(src)

    out = normalize_image(str(src), str(tmp_path / "out.jpg"))

    with Image.open(out) as img:
        assert img.format == "JPEG"
        assert img.size == size


def test_normalize_image_flattens_alpha_onto_white(tmp_path):
    src = tmp_path / "alpha.png"
    Image.new("RGBA", (64, 48), (0, 0, 0, 0)).save(src)

    out = normalize_image(str(src), str(tmp_path / "out.jpg"))

    with Image.open(out) as img:
        assert img.mode == "RGB"
        assert img.size == (64, 48)
        # Fully transparent pixels must land on the white background, not black.
        assert img.getpixel((10, 10)) == pytest.approx((255, 255, 255), abs=2)


def test_output_resolution_follows_the_model_input_size():
    assert _output_resolution(SimpleNamespace(opt=SimpleNamespace(input_size=512))) == "512x512"
    # A box without a loaded model keeps the documented default.
    assert _output_resolution(SimpleNamespace(opt=None)) == f"{TARGET_IMAGE_SIZE}x{TARGET_IMAGE_SIZE}"
    assert _output_resolution(SimpleNamespace()) == f"{TARGET_IMAGE_SIZE}x{TARGET_IMAGE_SIZE}"


def test_default_target_size_matches_the_released_checkpoint():
    assert TARGET_IMAGE_SIZE == 512
