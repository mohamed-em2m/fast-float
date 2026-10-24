"""Gemini API key config: shared secerts.toml / settings.toml + env precedence."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agents"))

from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient
from float_agent.config import AgentConfig, dispatch_config

from fast_float.core.state import app_state
from fast_float.livekit.session import _dispatch_payload, dispatch_agent
from fast_float.livekit.settings import LiveKitSettings
from fast_float.routers.livekit import router as livekit_router

try:
    from livekit import rtc  # noqa: F401

    RTC_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the environment
    RTC_AVAILABLE = False


@pytest.fixture
def api_client(monkeypatch):
    """Minimal app exposing the livekit router (same shape as test_livekit_worker)."""
    monkeypatch.setenv("LIVEKIT_URL", "ws://127.0.0.1:7880")
    monkeypatch.setenv("LIVEKIT_API_KEY", "devkey")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "devsecretdevsecretdevsecretdevsecret")
    if not RTC_AVAILABLE:
        import fast_float.livekit.worker as worker

        monkeypatch.setattr(worker, "RTC_AVAILABLE", True)
    app = FastAPI()
    app.include_router(livekit_router)
    with TestClient(app) as client:
        yield client


def _write_configs(tmp_path: Path, *, gemini_key: str = "", livekit_url: str = "ws://127.0.0.1:7880") -> Path:
    (tmp_path / "secerts.toml").write_text(
        "[livekit]\n"
        f'url = "{livekit_url}"\n'
        'api_key = "devkey"\n'
        'api_secret = "devsecretdevsecretdevsecretdevsecret"\n'
        "[gemini]\n"
        f'api_key = "{gemini_key}"\n',
        encoding="utf-8",
    )
    (tmp_path / "settings.toml").write_text(
        '[gemini]\nmodel = "gemini-3.8-live"\nvoice = "Zephyr"\nlanguage = "en-US"\n',
        encoding="utf-8",
    )
    return tmp_path


def test_agent_reads_gemini_key_from_shared_secerts(monkeypatch, tmp_path):
    _write_configs(tmp_path, gemini_key="AI-secret-from-secerts")
    monkeypatch.setenv("FLOAT_CONFIG_DIR", str(tmp_path))
    for name in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LIVEKIT_URL", "ws://override:7880")  # env still wins

    config = AgentConfig.from_env()
    assert config.google_api_key == "AI-secret-from-secerts"
    # Env beats the shared file for LiveKit; shared file is a fallback.
    assert config.livekit_url == "ws://override:7880"


def test_agent_shared_falls_back_to_livekit_and_non_secret_defaults(monkeypatch, tmp_path):
    _write_configs(tmp_path)
    monkeypatch.setenv("FLOAT_CONFIG_DIR", str(tmp_path))
    for name in (
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "LIVEKIT_URL",
        "LIVEKIT_API_KEY",
        "LIVEKIT_API_SECRET",
        "GEMINI_MODEL",
        "GEMINI_VOICE",
        "GEMINI_LANGUAGE",
    ):
        monkeypatch.delenv(name, raising=False)

    config = AgentConfig.from_env()
    # [livekit] secrets + [gemini] non-secret defaults come from the shared files.
    assert config.livekit_api_key == "devkey"
    assert config.livekit_url == "ws://127.0.0.1:7880"
    assert config.model == "gemini-3.8-live"
    assert config.voice == "Zephyr"
    assert config.language == "en-US"
    assert "GOOGLE_API_KEY" in config.missing()[0]


def test_agent_env_beats_shared_secret(monkeypatch, tmp_path):
    _write_configs(tmp_path, gemini_key="from-file")
    monkeypatch.setenv("FLOAT_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("GOOGLE_API_KEY", "from-env")

    config = AgentConfig.from_env()
    assert config.google_api_key == "from-env"


def test_livekit_settings_gemini_from_shared_files(monkeypatch):
    monkeypatch.setattr(
        "fast_float.livekit.settings.load_secrets",
        lambda: {
            "livekit": {"url": "ws://s:7880", "api_key": "k", "api_secret": "s" * 32},
            "gemini": {"api_key": "AI-file-key"},
        },
    )
    monkeypatch.setattr(
        "fast_float.livekit.settings.load_settings",
        lambda: {"gemini": {"model": "gemini-3.8-live", "voice": "Zephyr", "language": "en-US"}},
    )
    for name in ("GOOGLE_API_KEY", "GEMINI_API_KEY", "GEMINI_MODEL", "GEMINI_VOICE", "GEMINI_LANGUAGE"):
        monkeypatch.delenv(name, raising=False)

    settings = LiveKitSettings.from_env()
    assert settings.gemini_configured is True
    assert settings.gemini.api_key == "AI-file-key"
    assert settings.gemini.model == "gemini-3.8-live"
    assert settings.gemini.voice == "Zephyr"
    assert settings.gemini.language == "en-US"
    # Non-secret identities keep working from [livekit] as before.
    assert settings.agent_name == "float-gemini"
    assert settings.avatar_identity == "float-avatar"


def test_livekit_settings_gemini_env_beats_files(monkeypatch):
    monkeypatch.setattr(
        "fast_float.livekit.settings.load_secrets",
        lambda: {"gemini": {"api_key": "from-file"}},
    )
    monkeypatch.setattr("fast_float.livekit.settings.load_settings", lambda: {})
    monkeypatch.setenv("GOOGLE_API_KEY", "from-env")
    monkeypatch.setenv("GEMINI_VOICE", "Aster")

    settings = LiveKitSettings.from_env()
    assert settings.gemini.api_key == "from-env"
    assert settings.gemini.voice == "Aster"


def test_livekit_status_reports_gemini_configured_without_the_key(api_client, monkeypatch):
    monkeypatch.setattr(
        "fast_float.livekit.settings.load_secrets",
        lambda: {
            "livekit": {"url": "ws://127.0.0.1:7880", "api_key": "devkey", "api_secret": "devsecret" * 5},
            "gemini": {"api_key": "AI-secret"},
        },
    )
    monkeypatch.setattr("fast_float.livekit.settings.load_settings", lambda: {})
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    res = api_client.get("/livekit/status")
    assert res.status_code == 200
    body = res.json()
    assert body["gemini_configured"] is True
    # The key must never be serialized.
    assert "AI-secret" not in res.text


# ---------------------------------------------------------------------------
# Per-session key: browser -> start request -> dispatch metadata -> worker.
# ---------------------------------------------------------------------------


def test_conversation_start_forwards_browser_key_to_session(api_client, monkeypatch):
    captured = {}

    async def fake_start(room_name, **_kwargs):
        captured.update(_kwargs)
        return type(
            "Session",
            (),
            {
                "agent_backend": "gemini",
                "room_name": room_name,
                "resolved_agent_identity": "float-gemini",
                "avatar_identity": "float-avatar",
                "agent_dispatched": True,
                "tts_fn": None,
            },
        )()

    monkeypatch.setattr("fast_float.routers.livekit.start_session", fake_start)
    monkeypatch.setattr("fast_float.routers.livekit.get_session", lambda room: None)
    monkeypatch.setattr("fast_float.routers.livekit.stop_session", lambda room: None)
    monkeypatch.setattr(app_state, "agent", MagicMock())
    # A fresh room so the in-memory registry does not interfere.
    res = api_client.post(
        "/livekit/conversation/start",
        json={
            "room": "gemini-keysession",
            "agent": "gemini",
            "gemini_api_key": "AI-browser-key",
            "gemini_model": "gemini-3.8-live",
        },
    )
    assert res.status_code == 200
    assert captured["agent_backend"] == "gemini"
    assert captured["gemini_api_key"] == "AI-browser-key"
    assert captured["gemini_model"] == "gemini-3.8-live"


def test_dispatch_payload_includes_metadata_only_when_needed():
    assert _dispatch_payload("float-r", "float-gemini") == {"room": "float-r", "agent_name": "float-gemini"}
    payload = _dispatch_payload("float-r", "float-gemini", "  AI-browser-key  ", "  gemini-3.8-live  ")
    assert payload["agent_name"] == "float-gemini"
    assert json.loads(payload["metadata"]) == {
        "gemini_api_key": "AI-browser-key",
        "gemini_model": "gemini-3.8-live",
    }
    # A model alone still produces metadata; no value means no metadata key at all.
    payload_model = _dispatch_payload("float-r", "float-gemini", gemini_model="gemini-3.8-live")
    assert json.loads(payload_model["metadata"]) == {"gemini_model": "gemini-3.8-live"}


def test_dispatch_agent_forwards_overrides_through_create_dispatch(monkeypatch):
    captured = {}

    class _FakeAgentDispatch:
        async def create_dispatch(self, request):
            captured["request"] = request
            return None

    class _FakeLiveKitAPI:
        def __init__(self, **_kwargs):
            self.agent_dispatch = _FakeAgentDispatch()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

    class _FakeCreateRequest:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr("livekit.api.LiveKitAPI", _FakeLiveKitAPI)
    monkeypatch.setattr("livekit.api.CreateAgentDispatchRequest", _FakeCreateRequest)

    async def _run():
        settings = LiveKitSettings(url="ws://127.0.0.1:7880", api_key="k", api_secret="s" * 32)
        ok = await dispatch_agent(
            settings,
            "float-r",
            agent_name="float-gemini",
            gemini_api_key="AI-browser",
            gemini_model="gemini-3.8-live",
        )
        return ok

    assert asyncio.run(_run()) is True
    assert captured["request"].kwargs["agent_name"] == "float-gemini"
    assert json.loads(captured["request"].kwargs["metadata"]) == {
        "gemini_api_key": "AI-browser",
        "gemini_model": "gemini-3.8-live",
    }


def test_worker_parses_dispatch_metadata_overrides():
    assert dispatch_config("") == {}
    assert dispatch_config("not-json") == {}
    assert dispatch_config('{"foo": 1}') == {}
    assert dispatch_config(json.dumps({"gemini_api_key": "AI-dispatch-key"})) == {"gemini_api_key": "AI-dispatch-key"}
    full = dispatch_config(json.dumps({"gemini_api_key": "  AI-padded  ", "gemini_model": "  m-audio  "}))
    assert full == {"gemini_api_key": "AI-padded", "gemini_model": "m-audio"}
    # Unknown keys and empty values are ignored.
    assert dispatch_config(json.dumps({"gemini_api_key": "", "other": "x"})) == {}


class _FakeJob:
    def __init__(self, status: int, error: str = "", job_id: str = "JOB_12345678") -> None:
        self.id = job_id
        self.state = type("State", (), {"status": status, "error": error})()


class _FakeDispatch:
    def __init__(self, agent_name: str, jobs) -> None:
        self.agent_name = agent_name
        self.state = type("State", (), {"jobs": jobs})()


def test_fetch_room_dispatch_reports_pending_dispatch_when_worker_is_missing(monkeypatch):
    """No job on the dispatch == no worker registered under the agent name."""
    from fast_float.livekit.session import fetch_room_dispatch

    class _FakeAgentDispatch:
        async def list_dispatch(self, room_name):
            return [_FakeDispatch("float-gemini", [])]

    class _FakeLiveKitAPI:
        def __init__(self, **_kwargs):
            self.agent_dispatch = _FakeAgentDispatch()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

    monkeypatch.setattr("livekit.api.LiveKitAPI", _FakeLiveKitAPI)

    async def _run():
        settings = LiveKitSettings(url="ws://127.0.0.1:7880", api_key="k", api_secret="s" * 32)
        return await fetch_room_dispatch(settings, "float-r")

    message = asyncio.run(_run())
    assert "NO job" in message
    assert "no worker is registered/available" in message


def test_fetch_room_dispatch_surfaces_a_failed_job_error(monkeypatch):
    from fast_float.livekit.session import fetch_room_dispatch

    class _FakeAgentDispatch:
        async def list_dispatch(self, room_name):
            return [_FakeDispatch("float-gemini", [_FakeJob(3, error="invalid api key (gemini)")])]

    class _FakeLiveKitAPI:
        def __init__(self, **_kwargs):
            self.agent_dispatch = _FakeAgentDispatch()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

    monkeypatch.setattr("livekit.api.LiveKitAPI", _FakeLiveKitAPI)

    async def _run():
        settings = LiveKitSettings(url="ws://127.0.0.1:7880", api_key="k", api_secret="s" * 32)
        return await fetch_room_dispatch(settings, "float-r")

    message = asyncio.run(_run())
    assert "status=failed" in message
    assert "invalid api key (gemini)" in message


def test_session_wires_dispatch_with_keyword_only_args(monkeypatch):
    """Regression: _start_external_agent used to pass agent_name positionally,
    which dispatch_agent() rejects (keyword-only after settings, room)."""
    from types import SimpleNamespace

    from fast_float.livekit.session import ConversationSession

    captured = {}

    async def fake_dispatch(settings, room, **kwargs):
        captured["settings"] = settings
        captured["room"] = room
        captured.update(kwargs)
        return True

    monkeypatch.setattr("fast_float.livekit.session.dispatch_agent", fake_dispatch)
    session = ConversationSession.__new__(ConversationSession)
    session.settings = SimpleNamespace(autodispatch=True)
    session.room_name = "float-wiring"
    session.agent_name = "float-gemini"
    session.gemini_api_key = "AI-wire"
    session.gemini_model = "gemini-3.8-live"

    assert asyncio.run(session._dispatch_external_agent()) is True
    assert captured["room"] == "float-wiring"
    assert captured == {
        "settings": session.settings,
        "room": "float-wiring",
        "agent_name": "float-gemini",
        "gemini_api_key": "AI-wire",
        "gemini_model": "gemini-3.8-live",
    }

    # Auto-dispatch off -> no dispatch, returns None.
    session.settings.autodispatch = False
    assert asyncio.run(session._dispatch_external_agent()) is None
    # Unset agent name -> no dispatch either.
    session.settings.autodispatch = True
    session.agent_name = ""
    assert asyncio.run(session._dispatch_external_agent()) is None
