"""Stream-capacity estimation: ``min(memory, compute)`` instead of raw VRAM.

The old heuristic divided *total* VRAM by a hard-coded 2 GiB per stream, which
over-commits a box that already runs other processes and ignores the fact that
a stream that renders slower than real time cannot be served at all.
"""

from __future__ import annotations

import json

import pytest

from fast_float.utils import gpu as gpu_utils

GIB = 1024**3


def _mock_memory(monkeypatch, *, free: float, allocated: float, reserved: float, total: float = 48.0):
    monkeypatch.setattr(gpu_utils.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        gpu_utils.torch.cuda, "mem_get_info", lambda *a, **k: (int(free * GIB), int(total * GIB))
    )
    monkeypatch.setattr(gpu_utils.torch.cuda, "memory_allocated", lambda *a, **k: int(allocated * GIB))
    monkeypatch.setattr(gpu_utils.torch.cuda, "memory_reserved", lambda *a, **k: int(reserved * GIB))


def test_configured_value_overrides_but_keeps_the_estimate(monkeypatch):
    _mock_memory(monkeypatch, free=20.0, allocated=5.0, reserved=7.0)
    monkeypatch.setattr(gpu_utils, "measured_per_stream_gib", lambda: None)

    cap = gpu_utils.estimate_stream_capacity(
        config_val=3, observed_rtf=0.9, per_stream_gib=2.0
    )

    assert cap.recommended == 3
    # The measured bounds are still reported next to the operator override.
    assert cap.memory_bound == 10
    assert cap.compute_bound == 0
    assert "configured livekit_max_streams=3" in cap.reason
    assert "estimate would be 1" in cap.reason


def test_memory_bound_uses_free_vram_not_total(monkeypatch):
    # 20 GiB driver-free + 2 GiB reusable cache - 1 GiB reserve = 21 usable.
    _mock_memory(monkeypatch, free=20.0, allocated=5.0, reserved=7.0)
    monkeypatch.setattr(gpu_utils, "measured_per_stream_gib", lambda: None)

    cap = gpu_utils.estimate_stream_capacity(per_stream_gib=2.0)

    assert cap.memory_bound == 10  # 21 // 2
    assert cap.recommended == 8  # capped
    assert "capped at" in cap.reason


def test_cached_blocks_are_reusable(monkeypatch):
    # Only 2 GiB driver-free, but 30 GiB cached by this process: still usable.
    _mock_memory(monkeypatch, free=2.0, allocated=5.0, reserved=35.0)
    monkeypatch.setattr(gpu_utils, "measured_per_stream_gib", lambda: None)

    cap = gpu_utils.estimate_stream_capacity(per_stream_gib=2.0)

    assert cap.memory_bound == 15  # (2 + 30 - 1) // 2
    assert cap.recommended == 8


def test_compute_bound_can_be_lower_than_memory(monkeypatch):
    _mock_memory(monkeypatch, free=40.0, allocated=5.0, reserved=5.0)
    monkeypatch.setattr(gpu_utils, "measured_per_stream_gib", lambda: None)

    cap = gpu_utils.estimate_stream_capacity(observed_rtf=0.20, per_stream_gib=2.0)

    assert cap.compute_bound == 4  # 0.85 / 0.20
    assert cap.recommended == 4
    assert cap.reason.startswith("compute-bound")


def test_unknown_rtf_leaves_only_the_memory_bound(monkeypatch):
    _mock_memory(monkeypatch, free=10.0, allocated=5.0, reserved=5.0)
    monkeypatch.setattr(gpu_utils, "measured_per_stream_gib", lambda: None)

    cap = gpu_utils.estimate_stream_capacity(per_stream_gib=2.0)

    assert cap.compute_bound is None
    assert cap.recommended == 4  # (10 + 0 - 1) // 2


def test_measured_decode_profile_sets_the_footprint(monkeypatch, tmp_path):
    from fast_float.utils import decode_tuner

    cache = tmp_path / "decode_tune.json"
    cache.write_text(json.dumps({"key": {"transient_bytes": int(3.0 * GIB)}}))
    monkeypatch.setattr(decode_tuner, "cache_path", lambda: str(cache))
    assert gpu_utils.measured_per_stream_gib() == pytest.approx(3.0)

    # 3.0 GiB measured activation + 0.5 GiB overhead = 3.5 GiB per stream.
    _mock_memory(monkeypatch, free=8.0, allocated=5.0, reserved=5.0)
    cap = gpu_utils.estimate_stream_capacity()
    assert cap.per_stream_gib == pytest.approx(3.5)
    assert cap.memory_bound == 2  # (8 + 0 - 1) // 3.5


def test_no_cuda_falls_back_to_one(monkeypatch):
    monkeypatch.setattr(gpu_utils.torch.cuda, "is_available", lambda: False)

    cap = gpu_utils.estimate_stream_capacity()

    assert cap.recommended == 1
    assert "no CUDA" in cap.reason


def test_capacity_is_serialisable_and_readable(monkeypatch):
    _mock_memory(monkeypatch, free=20.0, allocated=5.0, reserved=5.0)
    monkeypatch.setattr(gpu_utils, "measured_per_stream_gib", lambda: None)

    cap = gpu_utils.estimate_stream_capacity(observed_rtf=0.19, per_stream_gib=2.0)
    data = cap.as_dict()

    assert data["recommended"] == cap.recommended
    assert data["observed_rtf"] == pytest.approx(0.19)
    summary = data["summary"]
    assert summary.startswith(f"streams={cap.recommended} (memory=")
    for field in ("compute=", "rtf=", "free=", "per_stream="):
        assert field in summary


def test_auto_detect_wrapper_keeps_its_contract(monkeypatch):
    monkeypatch.setattr(gpu_utils, "measured_per_stream_gib", lambda: None)

    assert gpu_utils.auto_detect_max_streams(config_val=4) == 4
    assert gpu_utils.auto_detect_max_streams(config_val=1) == 1
    assert gpu_utils.auto_detect_max_streams(config_val=0) >= 1


def test_gpu_endpoint_reports_capacity(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from fast_float.core.state import app_state
    from fast_float.routers import health

    monkeypatch.setattr(app_state, "live_rtf", 0.2)
    monkeypatch.setattr(health, "_configured_max_streams", lambda: 0)

    app = FastAPI()
    app.include_router(health.router)
    with TestClient(app) as client:
        body = client.get("/gpu").json()

    capacity = body["capacity"]
    assert capacity["observed_rtf"] == pytest.approx(0.2)
    assert capacity["recommended"] >= 1
    assert "streams=" in capacity["summary"]


def test_zmq_generator_records_the_measured_rtf(monkeypatch):
    from fast_float.core.state import app_state
    from fast_float.livekit.zmq_generator import ZmqFloatVideoGenerator

    monkeypatch.setattr(app_state, "live_rtf", None)
    gen = ZmqFloatVideoGenerator(shm_name="test_capacity")

    gen._record_rtf(0.25)
    assert gen.last_rtf == pytest.approx(0.25)
    assert app_state.live_rtf == pytest.approx(0.25)
    assert gen._capacity_logged is True

    # Later chunks update the measurement but do not spam the log.
    gen._record_rtf(0.32)
    assert gen.last_rtf == pytest.approx(0.32)
    assert app_state.live_rtf == pytest.approx(0.32)
