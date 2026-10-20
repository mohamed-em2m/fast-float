"""Guards the legacy weight_norm -> parametrization restore for wav2vec2.

transformers builds ``Wav2Vec2PositionalConvEmbedding.conv`` with
``nn.utils.parametrizations.weight_norm`` on torch>=2.1, but checkpoints store
the legacy ``weight_g``/``weight_v`` pair. The stock loader then reports those
keys as unused and leaves the parametrized position conv at its random init,
which silently degrades the audio/emotion encoders. These tests pin the shim's
behaviour on both the parametrized and the legacy path.
"""

import pytest
import torch
import torch.nn as nn

from fast_float.models import wav2vec2 as w2v2

G_SHAPE = (8, 1, 1)
V_SHAPE = (8, 4, 3)
POSCONV = "encoder.pos_conv_embed.conv"


class _FakeParametrizations(nn.Module):
    """Stands in for torch's ``ParametrizationList`` (two original tensors)."""

    def __init__(self, first_shape, second_shape):
        super().__init__()
        self.original0 = nn.Parameter(torch.zeros(first_shape))
        self.original1 = nn.Parameter(torch.zeros(second_shape))


def _module_at(conv):
    """Wrap ``conv`` so ``named_modules()`` reports ``encoder.pos_conv_embed.conv``."""
    encoder = nn.Module()
    pos_conv_embed = nn.Module()
    pos_conv_embed.conv = conv
    encoder.pos_conv_embed = pos_conv_embed
    root = nn.Module()
    root.encoder = encoder
    return root


def _fake_posconv(*, swapped=False):
    conv = nn.Module()
    shapes = (V_SHAPE, G_SHAPE) if swapped else (G_SHAPE, V_SHAPE)
    conv.parametrizations = nn.ModuleDict({"weight": _FakeParametrizations(*shapes)})
    return conv


def _pair(conv):
    weight = conv.parametrizations.weight
    first, second = weight.original0, weight.original1
    if first.numel() > second.numel():
        return second, first
    return first, second


def _state_dict():
    return {
        f"wav2vec2.{POSCONV}.weight_g": torch.arange(8.0).reshape(G_SHAPE),
        f"wav2vec2.{POSCONV}.weight_v": torch.arange(float(8 * 4 * 3)).reshape(V_SHAPE),
    }


@pytest.mark.parametrize("swapped", [False, True])
def test_restores_by_shape_not_slot_order(monkeypatch, swapped):
    """torch may put the magnitude in either slot; both must end up correct."""
    monkeypatch.setattr(w2v2, "PARAMETRIZED_WEIGHT_NORM", True)
    conv = _fake_posconv(swapped=swapped)
    sd = _state_dict()

    assert w2v2.restore_legacy_weight_norm(_module_at(conv), sd) == 1

    magnitude, direction = _pair(conv)
    assert torch.equal(magnitude, sd[f"wav2vec2.{POSCONV}.weight_g"])
    assert torch.equal(direction, sd[f"wav2vec2.{POSCONV}.weight_v"])


def test_matches_prefixed_and_unprefixed_keys(monkeypatch):
    monkeypatch.setattr(w2v2, "PARAMETRIZED_WEIGHT_NORM", True)
    conv = _fake_posconv()
    sd = {
        f"{POSCONV}.weight_g": torch.ones(G_SHAPE),
        f"{POSCONV}.weight_v": torch.ones(V_SHAPE),
    }
    assert w2v2.restore_legacy_weight_norm(_module_at(conv), sd) == 1


def test_noop_on_legacy_weight_norm(monkeypatch):
    """torch<2.1 keeps ``weight_g``/``weight_v`` as the real keys — do not touch."""
    monkeypatch.setattr(w2v2, "PARAMETRIZED_WEIGHT_NORM", False)
    conv = _fake_posconv()
    assert w2v2.restore_legacy_weight_norm(_module_at(conv), _state_dict()) == 0
    assert torch.count_nonzero(_pair(conv)[0]) == 0


@pytest.mark.parametrize("state_dict", [{}, {f"w2v2.{POSCONV}.weight_g": torch.ones(G_SHAPE)}])
def test_noop_without_a_complete_pair(monkeypatch, state_dict):
    monkeypatch.setattr(w2v2, "PARAMETRIZED_WEIGHT_NORM", True)
    conv = _fake_posconv()
    assert w2v2.restore_legacy_weight_norm(_module_at(conv), state_dict) == 0


def test_shape_mismatch_is_skipped_not_crashed(monkeypatch):
    monkeypatch.setattr(w2v2, "PARAMETRIZED_WEIGHT_NORM", True)
    conv = _fake_posconv()
    sd = {
        f"wav2vec2.{POSCONV}.weight_g": torch.ones(3, 1, 1),
        f"wav2vec2.{POSCONV}.weight_v": torch.ones(3, 4, 3),
    }
    assert w2v2.restore_legacy_weight_norm(_module_at(conv), sd) == 0
    assert torch.count_nonzero(_pair(conv)[0]) == 0


def test_read_checkpoint_state_dict_empty_without_weights(tmp_path):
    assert w2v2.read_checkpoint_state_dict(tmp_path) == {}


def test_read_checkpoint_state_dict_reads_safetensors(tmp_path):
    from safetensors.torch import save_file

    save_file({"a": torch.zeros(2)}, str(tmp_path / "model.safetensors"))
    assert set(w2v2.read_checkpoint_state_dict(tmp_path)) == {"a"}


@pytest.mark.skipif(not w2v2.PARAMETRIZED_WEIGHT_NORM, reason="needs torch>=2.1 parametrizations.weight_norm")
def test_real_parametrized_conv_roundtrip():
    """End-to-end on the real torch class, using it exactly as transformers does."""
    from torch.nn.utils import parametrizations

    conv = parametrizations.weight_norm(nn.Conv1d(4, 8, 3), name="weight", dim=2)
    reference = conv.weight.detach().clone()
    legacy = {
        f"wav2vec2.{POSCONV}.weight_g": conv.parametrizations.weight.original0.detach().clone(),
        f"wav2vec2.{POSCONV}.weight_v": conv.parametrizations.weight.original1.detach().clone(),
    }
    with torch.no_grad():
        conv.parametrizations.weight.original0.zero_()
        conv.parametrizations.weight.original1.zero_()

    assert w2v2.restore_legacy_weight_norm(_module_at(conv), legacy) == 1
    assert torch.allclose(conv.weight.detach(), reference)
