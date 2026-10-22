"""DecoderMixin chunking, u8 postprocess, layout and OOM policy (CPU, fake decoder)."""

import types

import pytest
import torch

import fast_float.models.float.decoder as decoder_mod
from fast_float.models.float.decoder import FloatDecoder


class _FakeDec(torch.nn.Module):
    """``latent [m, dim] -> (img [m, 3, 2, 2] in [-1, 1], None)``; records chunk sizes."""

    def __init__(self):
        super().__init__()
        self.batches = []
        self.feat_batches = []

    def forward(self, latent, alpha=None, feats=None):
        self.batches.append(int(latent.shape[0]))
        self.feat_batches.append([int(f.shape[0]) for f in feats])
        x = torch.tanh(latent[:, :3])
        return x.view(-1, 3, 1, 1).expand(-1, 3, 2, 2).contiguous(), None


class _FakeAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dec = _FakeDec()


def _model(**overrides):
    base = {"dim_w": 8, "decode_batch_size": None}
    base.update(overrides)
    opt = types.SimpleNamespace(**base)
    ae = _FakeAE()
    return FloatDecoder(opt, motion_autoencoder=ae), ae


def _expected_u8(latent):
    x = torch.tanh(latent[:, :3]).view(-1, 3, 1, 1).expand(-1, 3, 2, 2)
    return ((x + 1.0) * 127.5).clamp(0.0, 255.0).to(torch.uint8).permute(0, 2, 3, 1).contiguous()


@pytest.fixture(autouse=True)
def _no_tuning_env(monkeypatch):
    monkeypatch.delenv("FLOAT_DECODE_CHANNELS_LAST", raising=False)
    monkeypatch.delenv("FLOAT_DECODE_AUTOCAST", raising=False)


def test_decode_u8_returns_uint8_hwc():
    model, ae = _model()
    lat = torch.randn(5, 8)
    out = model._decode_u8(ae, [torch.randn(1, 8)], lat)
    assert out.dtype is torch.uint8
    assert tuple(out.shape) == (5, 2, 2, 3)
    assert torch.equal(out, _expected_u8(lat))
    assert ae.dec.feat_batches == [[5]]


def test_static_batch_tiling_preallocates_output(monkeypatch):
    model, ae = _model()
    monkeypatch.setattr(model, "_decoder_static_batch", lambda: 2)
    lat = torch.randn(5, 8)
    out = model._decode_u8(ae, [torch.randn(1, 8)], lat)
    assert ae.dec.batches == [2, 2, 1]
    assert torch.equal(out, _expected_u8(lat))
    assert out.is_contiguous()


def test_batched_decode_uses_tuned_profile_batch():
    model, ae = _model()
    model._decode_profiles = {0: types.SimpleNamespace(batch_size=4)}
    out = model._decode_latent_batched(ae, torch.randn(10, 8), [torch.randn(1, 8)], device=torch.device("cuda", 0))
    assert ae.dec.batches == [4, 4, 2]
    # Feats are expanded once per request, so every chunk gets exactly ``m`` rows.
    assert ae.dec.feat_batches == [[4], [4], [2]]
    assert tuple(out.shape) == (10, 3, 2, 2)


def test_explicit_batch_size_beats_profile():
    model, ae = _model(decode_batch_size=3)
    model._decode_profiles = {0: types.SimpleNamespace(batch_size=8)}
    model._decode_latent_batched(ae, torch.randn(7, 8), [torch.randn(1, 8)], device=torch.device("cuda", 0))
    assert ae.dec.batches == [3, 3, 1]


def test_batched_decode_flattens_3d_latents():
    model, ae = _model()
    out = model._decode_latent_batched(ae, torch.randn(2, 3, 8), [torch.randn(1, 8)], batch_size=4)
    assert ae.dec.batches == [4, 2]
    assert tuple(out.shape) == (2, 3, 3, 2, 2)


def test_oom_split_defers_empty_cache(monkeypatch):
    model, ae = _model()
    evictions = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda *a, **k: evictions.append(1))
    real = model._decode_call

    def flaky(ae_arg, latent, feats, **kwargs):
        if latent.shape[0] > 2:
            raise torch.cuda.OutOfMemoryError("boom")
        return real(ae_arg, latent, feats, **kwargs)

    monkeypatch.setattr(model, "_decode_call", flaky)
    lat = torch.randn(4, 8)
    out = model._decode_u8(ae, [torch.randn(1, 8)], lat)
    assert torch.equal(out, _expected_u8(lat))
    assert evictions == []


def test_oom_evicts_cache_after_configured_streak(monkeypatch):
    model, ae = _model(decode_oom_evict_after=1)
    evictions = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda *a, **k: evictions.append(1))
    real = model._decode_call

    def flaky(ae_arg, latent, feats, **kwargs):
        if latent.shape[0] > 2:
            raise torch.cuda.OutOfMemoryError("boom")
        return real(ae_arg, latent, feats, **kwargs)

    monkeypatch.setattr(model, "_decode_call", flaky)
    model._decode_u8(ae, [torch.randn(1, 8)], torch.randn(4, 8))
    assert evictions == [1]


def test_quantize_is_a_view_for_channels_last():
    x = torch.randn(2, 3, 4, 5)
    plain = FloatDecoder._quantize_u8_hwc(x.clone())
    cl = FloatDecoder._quantize_u8_hwc(x.clone().to(memory_format=torch.channels_last))
    assert tuple(cl.shape) == (2, 4, 5, 3)
    assert plain.is_contiguous()
    assert cl.is_contiguous()
    assert torch.equal(plain, cl)

    stored = x.clone().to(memory_format=torch.channels_last)
    stored = ((stored + 1.0) * 127.5).clamp_(0.0, 255.0).to(torch.uint8)
    permuted = stored.permute(0, 2, 3, 1)
    assert permuted.is_contiguous()
    assert permuted.contiguous() is permuted


def test_quantize_does_not_mutate_its_input():
    x = torch.randn(2, 3, 4, 5)
    before = x.clone()
    FloatDecoder._quantize_u8_hwc(x)
    assert torch.equal(x, before)


def test_untileable_feature_batch_warns_instead_of_failing_silently(monkeypatch):
    model, _ = _model()
    feats = [torch.randn(2, 8)]
    # The app config logs to stderr through its own handler, so caplog sees nothing.
    calls = []
    monkeypatch.setattr(decoder_mod.logger, "warning", lambda msg: calls.append(msg))

    aligned = model._align_feats_2d(feats, 3)

    assert tuple(aligned[0].shape) == (3, 8)
    assert len(calls) == 1
    assert "does not tile onto 3 frames" in calls[0]
    # Only once per model: a per-chunk warning would flood the log.
    model._align_feats_2d(feats, 3)
    assert len(calls) == 1


def test_prepare_decoder_applies_channels_last_once(monkeypatch):
    monkeypatch.setenv("FLOAT_DECODE_CHANNELS_LAST", "1")
    model, ae = _model()
    ae.dec = torch.nn.Conv2d(3, 4, 3).to(memory_format=torch.channels_last)
    model._prepare_decoder(ae)
    model._prepare_decoder(ae)
    assert ae.dec._ff_channels_last is True
    assert ae.dec.weight.is_contiguous(memory_format=torch.channels_last)


def test_autocast_is_off_by_default_and_opt_in(monkeypatch):
    model, _ = _model()
    assert model._decode_autocast_dtype() is None
    monkeypatch.setenv("FLOAT_DECODE_AUTOCAST", "fp16")
    assert model._decode_autocast_dtype() is torch.float16
    monkeypatch.setenv("FLOAT_DECODE_AUTOCAST", "bf16")
    assert model._decode_autocast_dtype() is torch.bfloat16
    monkeypatch.setenv("FLOAT_DECODE_AUTOCAST", "0")
    assert model._decode_autocast_dtype() is None


def test_decode_u8_binds_decoder_once_per_call(monkeypatch):
    model, ae = _model()
    monkeypatch.setattr(model, "_decoder_static_batch", lambda: 2)
    binds = []
    real = model._bind_decoder

    def counting(ae_arg, device):
        binds.append(device)
        return real(ae_arg, device)

    monkeypatch.setattr(model, "_bind_decoder", counting)
    model._decode_u8(ae, [torch.randn(1, 8)], torch.randn(5, 8))
    assert len(binds) == 1
    assert not hasattr(model, "_active_bound_dec")


def test_replica_gets_its_own_compiled_decoder(monkeypatch):
    model, _ = _model()
    replica = _FakeAE()
    monkeypatch.setattr(torch, "compile", lambda fn, *a, **k: fn)

    def primary_only(*a, **k):
        raise AssertionError("primary compiled decoder used for a replica")

    model._decoder_compiled = primary_only
    out = model._decode_call(replica, torch.randn(2, 8), [torch.randn(2, 8)])
    assert replica._ff_decoder_compiled is replica.dec
    assert tuple(out.shape) == (2, 3, 2, 2)
