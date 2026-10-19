"""The decode auto-tuner must not re-measure the same GPU on every boot."""

import json

import pytest

from fast_float.utils.decode_tuner import DecodeAutoTuner, DecodeProfile


@pytest.fixture
def tuner(tmp_path, monkeypatch):
    monkeypatch.setenv("FLOAT_DECODE_TUNE_CACHE", str(tmp_path / "decode_tune.json"))
    return DecodeAutoTuner(device=None, max_batch=16)


def _profile(batch=4):
    return DecodeProfile(
        batch_size=batch,
        transient_bytes=512 * 1024**2,
        height=512,
        width=512,
        channels=3,
        transient_by_batch={1: 100 * 1024**2, 2: 200 * 1024**2, 4: 512 * 1024**2, 8: 900 * 1024**2},
        fps_by_batch={1: 0.5, 2: 170.6, 4: 174.9, 8: 184.4},
    )


def test_cache_roundtrip(tuner):
    assert tuner.load_cached("gpu-a") is None
    tuner.save_cached("gpu-a", _profile())

    loaded = tuner.load_cached("gpu-a")
    assert loaded is not None
    assert loaded.batch_size == 4
    assert loaded.transient_bytes == 512 * 1024**2
    assert loaded.transient_by_batch[8] == 900 * 1024**2
    assert loaded.fps_by_batch[2] == pytest.approx(170.6)
    # Different key (driver/torch/shape change) is a miss -> re-measure.
    assert tuner.load_cached("gpu-b") is None


def test_cache_is_not_written_when_disabled(tmp_path, monkeypatch):
    path = tmp_path / "decode_tune.json"
    monkeypatch.setenv("FLOAT_DECODE_TUNE_CACHE", "0")
    tuner = DecodeAutoTuner(device=None, max_batch=16)
    tuner.save_cached("gpu-a", _profile())
    assert not path.exists()
    assert tuner.load_cached("gpu-a") is None


def test_corrupt_cache_is_ignored(tuner, tmp_path):
    (tmp_path / "decode_tune.json").write_text("{not json")
    assert tuner.load_cached("gpu-a") is None
    tuner.save_cached("gpu-a", _profile())
    assert tuner.load_cached("gpu-a").batch_size == 4


def test_fit_cached_keeps_measured_batch(tuner, monkeypatch):
    monkeypatch.setattr(tuner, "_budget", lambda: 4 * 1024**3)
    fitted = tuner.fit_cached(_profile())
    assert fitted is not None
    assert fitted.batch_size == 4


def test_fit_cached_shrinks_when_vram_is_tighter(tuner, monkeypatch):
    # Something else (training job) now holds most of the GPU. With safety=1.3
    # a 200 MiB batch needs 260 MiB, so 300 MiB admits batch 2 but not 4.
    monkeypatch.setattr(tuner, "_budget", lambda: 300 * 1024**2)
    fitted = tuner.fit_cached(_profile())
    assert fitted is not None
    assert fitted.batch_size == 2
    assert fitted.transient_bytes == 200 * 1024**2


def test_fit_cached_returns_none_when_nothing_fits(tuner, monkeypatch):
    monkeypatch.setattr(tuner, "_budget", lambda: 10 * 1024**2)
    assert tuner.fit_cached(_profile()) is None


def test_cache_file_is_valid_json_on_disk(tuner, tmp_path):
    tuner.save_cached("gpu-a", _profile())
    data = json.loads((tmp_path / "decode_tune.json").read_text())
    assert data["gpu-a"]["batch_size"] == 4
    assert data["gpu-a"]["transient_by_batch"]["8"] == 900 * 1024**2
