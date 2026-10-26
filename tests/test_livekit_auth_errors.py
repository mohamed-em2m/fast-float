"""LiveKit authentication failures must be diagnosable, not ASGI tracebacks.

LiveKit reports every rejected JWT as ``401 ... invalid API key`` without saying
whether the key id or the signature was wrong. The API now maps that to a 502
with an actionable message (the Studio shows ``detail`` verbatim) and warns at
config-load time when key/secret come from different layers.
"""

from __future__ import annotations

import logging

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fast_float.livekit.settings import LiveKitSettings
from fast_float.routers import livekit as livekit_router


class _FakeConnectError(Exception):
    """Stand-in for livekit.rtc.ConnectError (the SDK may be absent)."""


class _ListHandler(logging.Handler):
    """The project logger does not propagate, so caplog cannot see its records."""

    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def _settings(**overrides) -> LiveKitSettings:
    values = {"url": "wss://demo.livekit.cloud", "api_key": "APIDEMO123", "api_secret": "s" * 44}
    values.update(overrides)
    return LiveKitSettings(**values)


def test_auth_error_detail_names_the_project_and_the_fix():
    detail = livekit_router._connect_error_detail(
        _settings(),
        _FakeConnectError("engine: signal failure: client error: 401 Unauthorized - invalid API key"),
    )

    assert "invalid API key" in detail
    assert "wss://demo.livekit.cloud" in detail
    assert "Settings -> Keys" in detail
    assert "secerts.toml" not in detail  # default source is "none"
    assert "revoked" in detail


def test_non_auth_connect_errors_are_passed_through():
    detail = livekit_router._connect_error_detail(_settings(), _FakeConnectError("connection refused"))
    assert detail == "LiveKit connection to wss://demo.livekit.cloud failed: connection refused"


def test_conversation_start_returns_502_with_the_reason(monkeypatch):
    from fast_float.core.state import app_state

    monkeypatch.setenv("LIVEKIT_URL", "wss://demo.livekit.cloud")
    monkeypatch.setenv("LIVEKIT_API_KEY", "APIDEMO123")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "s" * 44)
    monkeypatch.setattr(livekit_router, "RTC_AVAILABLE", True)
    monkeypatch.setattr(livekit_router, "_RTC_CONNECT_ERRORS", (_FakeConnectError,))
    monkeypatch.setattr(app_state, "agent", object())

    async def reject(*args, **kwargs):
        raise _FakeConnectError("engine: signal failure: client error: 401 Unauthorized - invalid API key")

    monkeypatch.setattr(livekit_router, "start_session", reject)

    app = FastAPI()
    app.include_router(livekit_router.router)
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/livekit/conversation/start", json={})

    assert response.status_code == 502
    detail = response.json()["detail"]
    assert "invalid API key" in detail
    assert "Settings -> Keys" in detail


def test_mixed_key_and_secret_layers_warn(monkeypatch):
    monkeypatch.setattr(
        "fast_float.livekit.settings.load_settings",
        lambda: {"livekit": {"api_secret": "file-secret"}},
    )
    monkeypatch.setattr("fast_float.livekit.settings.load_secrets", lambda: {})
    monkeypatch.setenv("LIVEKIT_URL", "ws://localhost:7880")
    monkeypatch.setenv("LIVEKIT_API_KEY", "env-key")
    monkeypatch.delenv("LIVEKIT_API_SECRET", raising=False)

    target = logging.getLogger("fast_float.livekit.settings")
    handler = _ListHandler()
    target.addHandler(handler)
    try:
        settings = LiveKitSettings.from_env()
    finally:
        target.removeHandler(handler)

    assert (settings.api_key, settings.api_secret) == ("env-key", "file-secret")
    assert any("different layers" in message for message in handler.messages)


def test_swapped_credentials_on_a_cloud_url_warn(monkeypatch):
    monkeypatch.setattr("fast_float.livekit.settings.load_secrets", lambda: {})
    monkeypatch.setattr("fast_float.livekit.settings.load_settings", lambda: {})
    monkeypatch.setenv("LIVEKIT_URL", "wss://demo.livekit.cloud")
    monkeypatch.setenv("LIVEKIT_API_KEY", "looks-like-the-secret")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "s" * 44)

    target = logging.getLogger("fast_float.livekit.settings")
    handler = _ListHandler()
    target.addHandler(handler)
    try:
        LiveKitSettings.from_env()
    finally:
        target.removeHandler(handler)

    assert any("does not start with 'API'" in message for message in handler.messages)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
