"""Self-test checks for the LiveKit + Gemini pipeline (no live network)."""

from __future__ import annotations

import asyncio

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fast_float.livekit import selftest
from fast_float.livekit.settings import LiveKitSettings
from fast_float.routers.livekit import router as livekit_router


def _settings(monkeypatch, *, gemini: bool = True, livekit: bool = True) -> LiveKitSettings:
    def _secrets() -> dict:
        data: dict = {}
        if livekit:
            data["livekit"] = {"url": "ws://127.0.0.1:7880", "api_key": "devkey", "api_secret": "devsecret" * 5}
        if gemini:
            data["gemini"] = {"api_key": "AI-probe-key"}
        return data

    monkeypatch.setattr("fast_float.livekit.settings.load_secrets", _secrets)
    monkeypatch.setattr("fast_float.livekit.settings.load_settings", lambda: {})
    for name in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    return LiveKitSettings.from_env()


@pytest.fixture(autouse=True)
def _fresh_probe_cache(monkeypatch):
    """Isolate the module-level probe cache between tests (no network)."""
    monkeypatch.setattr(selftest, "_connectivity", None)
    monkeypatch.setattr(selftest, "_connectivity_started", False)


def test_run_checks_reports_ok_when_everything_is_configured(monkeypatch):
    settings = _settings(monkeypatch)
    checks = selftest.run_checks(settings)
    by_name = {check.name: check for check in checks}
    assert by_name["livekit_credentials"].status == "ok"
    assert by_name["gemini_api_key"].status == "ok"
    assert by_name["agent_name"].status == "ok"
    assert by_name["autodispatch"].status == "ok"
    # Connectivity is the probe placeholder until verified.
    assert by_name["livekit_connectivity"].status == "skip"
    assert not selftest.warning_messages(checks)


def test_run_checks_flags_missing_gemini_and_livekit(monkeypatch):
    settings = _settings(monkeypatch, gemini=False, livekit=False)
    checks = selftest.run_checks(settings)
    by_name = {check.name: check for check in checks}
    assert by_name["livekit_credentials"].status == "fail"
    assert by_name["gemini_api_key"].status == "fail"
    assert selftest.warning_messages(checks), "failures must surface as warnings"


def test_probe_without_credentials_fails_without_network(monkeypatch):
    settings = _settings(monkeypatch, gemini=False, livekit=False)
    check = asyncio.run(selftest.probe_livekit(settings))
    assert check.name == "livekit_connectivity"
    assert check.status == "fail"
    assert "not configured" in check.message


def test_run_verify_caches_a_probe_result(monkeypatch):
    settings = _settings(monkeypatch)

    async def _fake_probe(_settings):
        return selftest.Check("livekit_connectivity", "ok", "connected")

    monkeypatch.setattr("fast_float.livekit.selftest.probe_livekit", _fake_probe)

    result = asyncio.run(selftest.run_verify(settings))
    assert result.status == "ok"
    # Cached afterwards, so a later cached_check() sees it without probing.
    assert asyncio.run(selftest.cached_check()) == result


def test_status_endpoint_exposes_checks_and_warnings(monkeypatch):
    _settings(monkeypatch, gemini=True, livekit=True)
    app = FastAPI()
    app.include_router(livekit_router)
    with TestClient(app) as client:
        body = client.get("/livekit/status").json()
    names = {check["name"] for check in body["checks"]}
    assert names >= {"livekit_credentials", "livekit_connectivity", "gemini_api_key", "agent_name"}
    assert isinstance(body["warnings"], list)
    # Server key must never appear anywhere in the payload.
    assert "AI-probe-key" not in str(body)


def test_status_verify_runs_a_fresh_probe_into_checks(monkeypatch):
    _settings(monkeypatch, gemini=True, livekit=True)
    canned = selftest.Check("livekit_connectivity", "ok", "connected to the SFU")

    async def _fake_probe(_settings):
        return canned

    monkeypatch.setattr("fast_float.livekit.selftest.probe_livekit", _fake_probe)

    app = FastAPI()
    app.include_router(livekit_router)
    with TestClient(app) as client:
        body = client.get("/livekit/status?verify=1").json()
    connectivity = next(check for check in body["checks"] if check["name"] == "livekit_connectivity")
    assert connectivity == {"name": "livekit_connectivity", "status": "ok", "message": "connected to the SFU"}
    # The fresh result is cached for later (non-verify) polls.
    assert asyncio.run(selftest.cached_check()) == canned
