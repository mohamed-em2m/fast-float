"""Tests for the perf matrix endpoints (/perf/matrix, /perf/benchmark)."""

from __future__ import annotations

import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fast_float.routers import perf


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(perf.router)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def reset_state(monkeypatch):
    """The router keeps its last result in a module global; reset it per test."""
    monkeypatch.setattr(
        perf,
        "_state",
        {"status": "idle", "started_at": None, "finished_at": None, "error": None, "matrix": None},
    )
    monkeypatch.setattr(perf, "_task", None)


def test_matrix_is_idle_before_any_run(client):
    res = client.get("/perf/matrix")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "idle"
    assert body["matrix"] is None


def test_benchmark_needs_a_loaded_model(client, monkeypatch):
    monkeypatch.setattr(perf.app_state, "agent", None)
    res = client.post("/perf/benchmark", json={"concurrency": [1]})
    assert res.status_code == 503


def test_benchmark_rejects_an_empty_sweep(client, monkeypatch):
    monkeypatch.setattr(perf.app_state, "agent", object())
    res = client.post("/perf/benchmark", json={"concurrency": []})
    assert res.status_code == 400


def test_benchmark_rejects_an_absurd_sweep(client, monkeypatch):
    monkeypatch.setattr(perf.app_state, "agent", object())
    res = client.post("/perf/benchmark", json={"concurrency": [128]})
    assert res.status_code == 400


def test_benchmark_rejects_a_missing_input(client, monkeypatch):
    monkeypatch.setattr(perf.app_state, "agent", object())
    res = client.post("/perf/benchmark", json={"concurrency": [1], "ref": "/nope/missing.jpg"})
    assert res.status_code == 400


def test_benchmark_start_and_result(client, monkeypatch):
    async def fake_run(config):
        perf._state.update(status="done", finished_at=time.time(), error=None, matrix={"model_fps": 12.5})

    monkeypatch.setattr(perf, "_run", fake_run)
    monkeypatch.setattr(perf.app_state, "agent", object())

    res = client.post("/perf/benchmark", json={"concurrency": [1, 2]})
    assert res.status_code == 200
    assert res.json()["status"] == "running"

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if client.get("/perf/matrix").json()["status"] == "done":
            break
        time.sleep(0.02)

    body = client.get("/perf/matrix").json()
    assert body["status"] == "done"
    assert body["matrix"]["model_fps"] == 12.5
    assert body["elapsed_s"] is not None
