"""Tests for SynthesisInfer: inference-only drop-in replacement for Synthesis."""

import copy

import torch

from fast_float.models.float.generator import Generator
from fast_float.models.float.styledecoder import (
    Blur,
    Direction,
    StyledConv,
    Synthesis,
    Upsample,
)
from fast_float.models.float.synthesis_infer import (
    DirectionInfer,
    FIRBlur,
    FIRUp2,
    StyledConvInfer,
    SynthesisInfer,
)
from fast_float.onnx import dispatch


def test_direction_infer_parity():
    torch.manual_seed(42)
    dir_orig = Direction(20).eval()
    dir_infer = DirectionInfer(dir_orig).eval()

    # Test None
    assert torch.allclose(dir_orig(None), dir_infer(None))

    # Test 1D
    x1 = torch.randn(20)
    out1_orig = dir_orig(x1)
    out1_infer = dir_infer(x1)
    assert out1_orig.shape == out1_infer.shape == (512,)
    assert torch.allclose(out1_orig, out1_infer, atol=1e-5)

    # Test 2D (batched)
    x2 = torch.randn(4, 20)
    out2_orig = dir_orig(x2)
    out2_infer = dir_infer(x2)
    assert out2_orig.shape == out2_infer.shape == (4, 512)
    assert torch.allclose(out2_orig, out2_infer, atol=1e-5)

    # Test 3D
    x3 = torch.randn(2, 5, 20)
    out3_infer = dir_infer(x3)
    assert out3_infer.shape == (2, 5, 512)


def test_fir_blur_and_up2_parity():
    torch.manual_seed(42)
    blur_kernel = [1, 3, 3, 1]
    up = Upsample(blur_kernel, factor=2).eval()
    up_fir = FIRUp2(up).eval()

    x = torch.randn(2, 16, 8, 8)
    out_orig = up(x)
    out_fir = up_fir(x)
    assert out_orig.shape == out_fir.shape == (2, 16, 16, 16)
    assert torch.allclose(out_orig, out_fir, atol=1e-6)

    blur = Blur(blur_kernel, pad=(1, 1), upsample_factor=2).eval()
    blur_fir = FIRBlur(blur.kernel, blur.pad).eval()
    out_blur_orig = blur(out_orig)
    out_blur_fir = blur_fir(out_orig)
    assert out_blur_orig.shape == out_blur_fir.shape
    assert torch.allclose(out_blur_orig, out_blur_fir, atol=1e-6)


def test_styled_conv_infer_parity():
    torch.manual_seed(42)
    # Upsample conv
    sc = StyledConv(64, 32, 3, 128, upsample=True).eval()
    sci = StyledConvInfer(sc, fold_blur=False).eval()
    sci_fold = StyledConvInfer(sc, fold_blur=True).eval()

    x = torch.randn(2, 64, 8, 8)
    s = torch.randn(2, 128)
    s_mod = sc.conv.modulation(s)

    out_orig = sc(x, s)
    out_sci = sci(x, s_mod)
    out_fold = sci_fold(x, s_mod)

    assert out_orig.shape == out_sci.shape == out_fold.shape == (2, 32, 16, 16)
    assert torch.allclose(out_orig, out_sci, atol=1e-5)
    assert torch.allclose(out_orig, out_fold, atol=1e-5)

    # Non-upsample conv
    sc_plain = StyledConv(32, 32, 3, 128, upsample=False).eval()
    sci_plain = StyledConvInfer(sc_plain, fold_blur=False).eval()
    s_mod_plain = sc_plain.conv.modulation(s)

    out_plain_orig = sc_plain(out_orig, s)
    out_plain_sci = sci_plain(out_orig, s_mod_plain)
    assert out_plain_orig.shape == out_plain_sci.shape == (2, 32, 16, 16)
    assert torch.allclose(out_plain_orig, out_plain_sci, atol=1e-5)


def test_synthesis_infer_signatures_and_precompute():
    torch.manual_seed(42)
    syn = Synthesis(size=64, style_dim=512, motion_dim=10).eval()
    dec = SynthesisInfer.from_synthesis(syn).eval()

    # Check properties
    assert dec.size == syn.size
    assert dec.style_dim == syn.style_dim
    assert dec.motion_dim == syn.motion_dim
    assert dec.n_levels == len(syn.to_flows)

    # Prepare inputs
    B = 2
    wa = torch.randn(B, 512)
    alpha = [torch.randn(B, 10)]
    feats = []
    for k in range(dec.n_levels):
        res = 2 ** (k + 3)
        c = syn.channels[res]
        feats.append(torch.randn(1, c, res, res))

    # Precompute
    z_last = dec.precompute(feats)
    assert z_last is not None
    assert z_last.shape == (1, 3, 64, 64)

    # Test signature 1: dec(wa, alpha, feats)
    img1, _ = dec(wa, alpha, feats)
    assert img1.shape == (B, 3, 64, 64)

    # Test signature 2: dec(wa, None, feats, z_last=z_last)
    img2, _ = dec(wa, None, feats, z_last=z_last)
    assert img2.shape == (B, 3, 64, 64)

    # Test signature 3: dec(wa, feats=feats)
    img3, _ = dec(wa, feats=feats)
    assert img3.shape == (B, 3, 64, 64)

    # Test signature 4: dec(wa, feats)
    img4, _ = dec(wa, feats)
    assert img4.shape == (B, 3, 64, 64)

    # Test signature 5: 1D single input
    img_single, flw_single = dec(wa[0], feats=feats)
    assert img_single.shape == (3, 64, 64)
    if flw_single is not None:
        assert flw_single.dim() == 3

    # Test alias synthesis()
    img_syn, _ = dec.synthesis(wa, None, feats)
    assert img_syn.shape == (B, 3, 64, 64)


def test_synthesis_and_generator_to_infer():
    torch.manual_seed(42)
    syn = Synthesis(size=64, style_dim=64, motion_dim=10).eval()
    infer = syn.to_infer()
    assert isinstance(infer, SynthesisInfer)

    gen = Generator(size=64, style_dim=64, motion_dim=10).eval()
    gen.to_infer()
    assert isinstance(gen.dec, SynthesisInfer)


def test_channels_last_and_deepcopy():
    torch.manual_seed(42)
    syn = Synthesis(size=64, style_dim=64, motion_dim=10).eval()
    dec = SynthesisInfer.from_synthesis(syn, channels_last=True).eval()
    assert dec._ff_channels_last is True

    # Test deepcopy for multi-gpu replica creation
    dec_copy = copy.deepcopy(dec)
    assert isinstance(dec_copy, SynthesisInfer)
    assert dec_copy._ff_channels_last is True


def test_dispatch_applies_synthesis_infer(monkeypatch):
    monkeypatch.setattr(dispatch, "resolve_mode", lambda name: "eager")
    gen = Generator(size=64, style_dim=64, motion_dim=10).eval()

    class _MockModel:
        motion_autoencoder = gen

    model = _MockModel()
    dispatch._apply_decoder(model, model.motion_autoencoder, None)
    assert isinstance(model.motion_autoencoder.dec, SynthesisInfer)
