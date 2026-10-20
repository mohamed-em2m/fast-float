"""Installing an ONNX backend must not trust the artifact blindly.

Regression: the decoder graph was validated at export time with a tolerance
that accepted ``max_abs=4.375`` / ``cosine=0.983``, then installed at runtime
without any check and produced a visibly wrong avatar. The graph is only as
good as the execution provider that loads it, so each install self-tests once.
"""

import pytest
import torch

from fast_float.onnx import dispatch


class _FakeDec(torch.nn.Module):
    def forward(self, latent, alpha=None, feats=None):
        return torch.zeros(latent.shape[0], 3, 2, 2)


class _FakeAE:
    dec = _FakeDec()
    enc = None


class _FakeModel:
    motion_autoencoder = _FakeAE()


def _install(monkeypatch, *, ok: bool):
    installed = {}

    def fake_runner(*args, **kwargs):
        return object()

    monkeypatch.setattr(dispatch, "DecoderOnnx", fake_runner)
    monkeypatch.setattr(dispatch, "resolve_mode", lambda name: "onnx")
    monkeypatch.setattr(dispatch, "_selftest_decoder", lambda *a, **k: ok)
    model = _FakeModel()
    dispatch._apply_decoder(model, model.motion_autoencoder, None)
    installed["runner"] = getattr(model, "_decoder_onnx", None)
    return installed


def test_broken_decoder_artifact_stays_eager(monkeypatch):
    assert _install(monkeypatch, ok=False)["runner"] is None


def test_passing_decoder_artifact_is_installed(monkeypatch):
    assert _install(monkeypatch, ok=True)["runner"] is not None


def test_self_test_can_be_skipped_for_benchmarks(monkeypatch):
    monkeypatch.setenv("FLOAT_SKIP_BACKEND_SELFTEST", "1")
    calls = []

    def boom(*args, **kwargs):
        calls.append(args)
        return False

    monkeypatch.setattr(dispatch, "DecoderOnnx", lambda *a, **k: object())
    monkeypatch.setattr(dispatch, "resolve_mode", lambda name: "onnx")
    monkeypatch.setattr(dispatch, "_selftest_decoder", boom)
    model = _FakeModel()
    dispatch._apply_decoder(model, model.motion_autoencoder, None)
    assert calls == []
    assert model._decoder_onnx is not None


@pytest.mark.parametrize(("value", "enabled"), [("", True), ("0", True), ("1", False), ("ON", False)])
def test_selftest_env_parsing(monkeypatch, value, enabled):
    monkeypatch.setenv("FLOAT_SKIP_BACKEND_SELFTEST", value)
    assert dispatch._selftests_enabled() is enabled
