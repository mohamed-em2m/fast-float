"""settings.toml [float]: model/inference options, precedence and request defaults."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from fast_float.config.float_settings import (
    _flatten,
    float_table,
    load_float_options,
    request_default,
)
from fast_float.schemas.generation import GenerationRequest


@pytest.fixture
def config_dir(monkeypatch, tmp_path):
    """Point the shared config loader at a temporary settings.toml."""
    monkeypatch.setenv("FLOAT_CONFIG_DIR", str(tmp_path))
    return tmp_path


def _write(config_dir: Path, text: str) -> None:
    (config_dir / "settings.toml").write_text(text, encoding="utf-8")


def test_every_argparse_option_has_a_settings_key(monkeypatch):
    """The committed file must document the complete option set."""
    monkeypatch.delenv("FLOAT_CONFIG_DIR", raising=False)

    from fast_float.core.options import InferenceOptions

    parser = argparse.ArgumentParser(add_help=False)
    InferenceOptions().initialize(parser)
    dests = {action.dest for action in parser._actions}

    keys = {
        path[-1]
        for path, _ in _flatten(float_table())
        if path[0] not in {"livekit", "request"} and path[-1] != "livekit_nfe"
    }
    assert dests - keys == set()


def test_settings_values_are_typed_and_applied(config_dir):
    _write(
        config_dir,
        """
[float]
livekit_nfe = 5

[float.checkpoint]
ckpt_path = "./custom/float.pth"
res_dir = "./out"

[float.runtime]
ngpus = 2
optimization_mode = "channels_last"
use_fp16 = false
decode_batch_size = 8
decode_channels_last = true
decode_autocast = "bf16"

[float.video]
fps = 30.0
input_size = 256

[float.audio]
wav2vec_sec = 1.0
only_last_features = true

[float.model]
dim_w = 256
fmt_depth = 4
no_learned_pe = true

[float.ode]
nfe = 12
ode_atol = 1e-6
torchdiffeq_ode_method = "rk4"

[float.guidance]
a_cfg_scale = 1.5

[float.diffusion]
diff_schedule = "linear"

[float.paths]
emo = ""
ref_path = ""
""",
    )

    opt = load_float_options()

    assert opt.ckpt_path == "./custom/float.pth"
    assert opt.res_dir == "./out"
    assert opt.ngpus == 2
    assert opt.optimization_mode == "channels_last"
    assert opt.use_fp16 is False
    assert opt.decode_batch_size == 8
    assert opt.decode_channels_last is True
    assert opt.decode_autocast == "bf16"
    assert opt.fps == 30.0
    assert opt.input_size == 256
    assert opt.wav2vec_sec == 1.0
    assert opt.only_last_features is True
    assert opt.dim_w == 256
    assert opt.fmt_depth == 4
    assert opt.no_learned_pe is True
    assert opt.nfe == 12
    assert opt.ode_atol == 1e-6
    assert opt.torchdiffeq_ode_method == "rk4"
    assert opt.a_cfg_scale == 1.5
    assert opt.diff_schedule == "linear"
    assert opt.emo is None
    assert opt.ref_path is None


def test_settings_win_over_legacy_env(config_dir, monkeypatch):
    _write(config_dir, "[float.runtime]\nngpus = 2\nuse_fp16 = false\n")
    monkeypatch.setenv("NGPUS", "8")
    monkeypatch.setenv("USE_FP16", "true")

    opt = load_float_options()

    assert opt.ngpus == 2
    assert opt.use_fp16 is False


def test_env_is_a_fallback_when_the_file_omits_a_key(config_dir, monkeypatch):
    _write(config_dir, "[float.checkpoint]\nckpt_path = './float.pth'\n")
    monkeypatch.setenv("NGPUS", "3")
    monkeypatch.setenv("OPTIMIZATION_MODE", "amp")
    monkeypatch.setenv("RESULTS_DIR", "./env-results")
    monkeypatch.setenv("FLOAT_DECODE_CHANNELS_LAST", "1")
    monkeypatch.delenv("USE_FP16", raising=False)

    opt = load_float_options()

    assert opt.ngpus == 3
    assert opt.optimization_mode == "amp"
    assert opt.res_dir == "./env-results"
    assert opt.decode_channels_last is True
    # Existing API behaviour: FP16 is on unless the file/env says otherwise.
    assert opt.use_fp16 is True


def test_invalid_values_keep_the_parser_default(config_dir):
    _write(
        config_dir,
        """
[float.runtime]
ngpus = "many"
optimization_mode = "turbo"

[float.ode]
nfe = "fast"
""",
    )

    opt = load_float_options()

    assert opt.ngpus == 1
    assert opt.optimization_mode == "all"
    assert opt.nfe == 10


def test_unknown_keys_are_ignored(config_dir):
    _write(config_dir, "[float.runtime]\nnot_a_real_option = 1\nngpus = 2\n")

    opt = load_float_options()

    assert opt.ngpus == 2
    assert not hasattr(opt, "not_a_real_option")


def test_options_do_not_read_sys_argv(config_dir, monkeypatch):
    _write(config_dir, "[float.ode]\nnfe = 7\n")
    monkeypatch.setattr("sys.argv", ["api", "--nfe", "25"])

    assert load_float_options().nfe == 7


def test_request_defaults_come_from_settings_and_are_clamped(config_dir):
    _write(
        config_dir,
        """
[float.request]
nfe = 99
seed = -5
emotion = "grumpy"
a_cfg_scale = 0.1
decode_batch_size = 100
""",
    )

    assert request_default("nfe") == 25
    assert request_default("seed") == 0
    assert request_default("emotion") == "S2E"
    assert request_default("a_cfg_scale") == 0.5
    assert request_default("decode_batch_size") == 64


def test_generation_request_uses_the_configured_defaults(config_dir):
    _write(
        config_dir,
        """
[float.request]
nfe = 7
seed = 42
emotion = "happy"
no_crop = true
""",
    )

    request = GenerationRequest(audio_base64="")

    assert request.nfe == 7
    assert request.seed == 42
    assert request.emotion == "happy"
    assert request.no_crop is True

    # An explicit request value still wins.
    explicit = GenerationRequest(audio_base64="", nfe=9, seed=1)
    assert explicit.nfe == 9
    assert explicit.seed == 1
