"""Tests for the LiveKit token endpoint and settings (no network, no server)."""

from __future__ import annotations

import time

import jwt
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fast_float.livekit.settings import LiveKitSettings, valid_room_name
from fast_float.livekit.tokens import issue_token
from fast_float.routers import livekit as livekit_router

API_KEY = "devkey"
# A 32+ byte secret: HMAC-SHA256 keys shorter than that are flagged by pyjwt.
API_SECRET = "devsecretdevsecretdevsecretdevsecret"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("LIVEKIT_URL", "ws://127.0.0.1:7880")
    monkeypatch.setenv("LIVEKIT_API_KEY", API_KEY)
    monkeypatch.setenv("LIVEKIT_API_SECRET", API_SECRET)
    app = FastAPI()
    app.include_router(livekit_router.router)
    with TestClient(app) as test_client:
        yield test_client


def test_settings_are_read_from_the_env(monkeypatch):
    monkeypatch.setenv("LIVEKIT_URL", "ws://localhost:7880")
    monkeypatch.setenv("LIVEKIT_API_KEY", "k")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "s")
    monkeypatch.setenv("LIVEKIT_TOKEN_TTL_S", "900")
    monkeypatch.setenv("LIVEKIT_ROOM_PREFIX", "studio")

    settings = LiveKitSettings.from_env()
    assert settings.configured is True
    assert settings.url == "ws://localhost:7880"
    assert settings.token_ttl_s == 900
    assert settings.room_prefix == "studio"
    assert settings.room_for("my-room") == "my-room"
    assert settings.room_for().startswith("studio-")


def test_unconfigured_settings_say_so(monkeypatch):
    for name in ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET"):
        monkeypatch.delenv(name, raising=False)
    # Also neutralize the config files (an operator checkout may carry a
    # populated secerts.toml); here we test the layered "nothing set" case.
    monkeypatch.setattr("fast_float.livekit.settings.load_secrets", lambda: {})
    monkeypatch.setattr("fast_float.livekit.settings.load_settings", lambda: {})
    settings = LiveKitSettings.from_env()
    assert settings.configured is False
    assert settings.source == "none"


def test_room_names_are_validated():
    assert valid_room_name("float-abc123") is True
    assert valid_room_name("has space") is False
    assert valid_room_name("") is False
    assert valid_room_name("a" * 65) is False


def test_settings_layering_secrets_beat_files_env_beats_all(monkeypatch):
    # secerts.toml beats settings.toml; the env beats everything.
    monkeypatch.setattr(
        "fast_float.livekit.settings.load_settings",
        lambda: {"livekit": {"url": "ws://settings-layer:7880", "room_prefix": "set"}},
    )
    monkeypatch.setattr(
        "fast_float.livekit.settings.load_secrets",
        lambda: {"livekit": {"url": "wss://secrets-layer.livekit.cloud", "api_key": "k", "api_secret": "s"}},
    )
    for name in ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET", "LIVEKIT_ROOM_PREFIX"):
        monkeypatch.delenv(name, raising=False)
    s = LiveKitSettings.from_env()
    assert s.url == "wss://secrets-layer.livekit.cloud"
    assert s.api_key == "k"
    assert s.configured is True
    assert s.source == "secerts.toml"
    assert s.room_prefix == "set"

    monkeypatch.setenv("LIVEKIT_URL", "ws://env-layer:7880")
    s = LiveKitSettings.from_env()
    assert s.url == "ws://env-layer:7880"
    assert s.source == "env"


def test_settings_stun_urls_from_config(monkeypatch):
    monkeypatch.setattr(
        "fast_float.livekit.settings.load_settings",
        lambda: {"webrtc": {"stun_urls": ["stun:127.0.0.1:3478", "stun:stun.example:3478"]}},
    )
    s = LiveKitSettings.from_env()
    assert s.stun_urls == ("stun:127.0.0.1:3478", "stun:stun.example:3478")


def test_issued_token_carries_the_room_grant():
    issued = issue_token(
        url="ws://127.0.0.1:7880",
        api_key=API_KEY,
        api_secret=API_SECRET,
        identity="viewer-1",
        name="Viewer",
        room="float-test",
        ttl_s=600,
    )
    claims = jwt.decode(issued.token, API_SECRET, algorithms=["HS256"])

    assert claims["iss"] == API_KEY
    assert claims["sub"] == "viewer-1"
    assert claims["name"] == "Viewer"
    # livekit-api mints the grants flat (the server parses them the same way)
    assert claims["video"]["room"] == "float-test"
    assert claims["video"]["roomJoin"] is True
    assert claims["video"]["canSubscribe"] is True
    # a viewer token must not be able to publish
    assert claims["video"]["canPublish"] is False
    assert 0 < claims["exp"] - claims["nbf"] <= 600


def test_worker_token_may_publish():
    issued = issue_token(
        url="ws://127.0.0.1:7880",
        api_key=API_KEY,
        api_secret=API_SECRET,
        identity="float-avatar",
        room="float-test",
        can_publish=True,
    )
    claims = jwt.decode(issued.token, API_SECRET, algorithms=["HS256"])
    assert claims["video"]["canPublish"] is True


def test_token_endpoint_mints_a_token(client):
    res = client.post("/livekit/token", json={"room": "float-abc", "name": "Studio"})
    assert res.status_code == 200
    body = res.json()
    assert body["room"] == "float-abc"
    assert body["url"] == "ws://127.0.0.1:7880"
    assert body["identity"].startswith("viewer-")
    assert body["expires_at"] > time.time()

    claims = jwt.decode(body["token"], API_SECRET, algorithms=["HS256"])
    assert claims["video"]["room"] == "float-abc"
    assert claims["video"]["canPublish"] is False


def test_token_endpoint_ignores_a_client_publish_request(client):
    # The browser is only ever a viewer: even when the client body asks for
    # publishing rights, the endpoint mints a subscriber-only token. (The
    # request field was removed; extra JSON keys are ignored.)
    body = client.post(
        "/livekit/token",
        json={"room": "float-abc", "name": "Studio", "can_publish": True},
    ).json()
    claims = jwt.decode(body["token"], API_SECRET, algorithms=["HS256"])
    assert claims["video"]["canPublish"] is False


def test_token_endpoint_generates_a_room_when_omitted(client):
    body = client.post("/livekit/token", json={}).json()
    assert body["room"].startswith("float-")


def test_token_endpoint_rejects_a_bad_room(client):
    res = client.post("/livekit/token", json={"room": "not a room"})
    assert res.status_code == 400


def test_token_endpoint_reports_an_unconfigured_server(monkeypatch):
    for name in ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET"):
        monkeypatch.delenv(name, raising=False)
    app = FastAPI()
    app.include_router(livekit_router.router)
    with TestClient(app) as test_client:
        res = test_client.post("/livekit/token", json={})
    assert res.status_code == 503


def test_status_endpoint_reports_the_configuration(client):
    body = client.get("/livekit/status").json()
    assert body["configured"] is True
    assert body["url"] == "ws://127.0.0.1:7880"
    assert body["room_prefix"] == "float"
