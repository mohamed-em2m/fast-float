"""Gemini API key config: shared secerts.toml / settings.toml + env precedence."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fast_float.core.state import app_state
from fast_float.livekit.gemini.config import AgentConfig, dispatch_config
from fast_float.livekit.session import _dispatch_payload, dispatch_agent
from fast_float.livekit.settings import LiveKitSettings
from fast_float.routers.livekit import router as livekit_router

try:
    from livekit import rtc  # noqa: F401

    RTC_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the environment
    RTC_AVAILABLE = False


@pytest.fixture
def visible_fast_float_logs(monkeypatch):
    """Let ``fast_float.*`` records reach pytest's caplog.

    fast_float.utils.logger installs its own handler on the ``fast_float``
    logger with ``propagate = False``, so caplog (a root-logger handler)
    never sees records from module loggers. Re-enable propagation for the
    tests that assert on drift warnings.
    """
    monkeypatch.setattr(logging.getLogger("fast_float"), "propagate", True)


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


def test_dynamic_config_dir_honors_float_config_dir(monkeypatch, tmp_path):
    """The API and the worker must be able to share one config directory."""
    from fast_float.config import dynamic

    monkeypatch.setenv("FLOAT_CONFIG_DIR", str(tmp_path))
    assert dynamic.config_dir() == tmp_path
    assert dynamic.settings_path() == tmp_path / "settings.toml"
    assert dynamic.secrets_path() == tmp_path / "secerts.toml"


def _write_configs(
    tmp_path: Path,
    *,
    gemini_key: str = "",
    livekit_url: str = "ws://127.0.0.1:7880",
    model: str = "gemini-3.8-live",
) -> Path:
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
        f'[gemini]\nmodel = "{model}"\nvoice = "Zephyr"\nlanguage = "en-US"\n',
        encoding="utf-8",
    )
    return tmp_path


def test_agent_reads_gemini_key_from_shared_secerts(monkeypatch, tmp_path):
    _write_configs(tmp_path, gemini_key="AI-secret-from-secerts")
    monkeypatch.setenv("FLOAT_CONFIG_DIR", str(tmp_path))
    for name in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LIVEKIT_URL", "ws://override:7880")  # stale env copy

    config = AgentConfig.from_env()
    assert config.google_api_key == "AI-secret-from-secerts"
    # The shared file is the single source of truth: a disagreeing env copy
    # loses (and is logged as stale).
    assert config.livekit_url == "ws://127.0.0.1:7880"


def test_agent_warns_when_env_copy_disagrees_with_shared_file(monkeypatch, tmp_path, caplog, visible_fast_float_logs):
    """The stale-env drift that used to register the worker on the wrong SFU."""
    _write_configs(tmp_path, gemini_key="AI-file-key")
    monkeypatch.setenv("FLOAT_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("LIVEKIT_URL", "ws://other-project.livekit.cloud")
    monkeypatch.setenv("GOOGLE_API_KEY", "AI-env-stale")

    with caplog.at_level(logging.WARNING, logger="fast_float.livekit"):
        config = AgentConfig.from_env()
    # File wins, in the config and for the google-genai SDK (env re-export).
    assert config.livekit_url == "ws://127.0.0.1:7880"
    assert config.google_api_key == "AI-file-key"
    assert os.environ["GOOGLE_API_KEY"] == "AI-file-key"
    drift = [r for r in caplog.records if "shared FLOAT config" in r.getMessage()]
    assert len(drift) >= 2  # LIVEKIT_URL and GOOGLE_API_KEY copies are stale


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


def test_agent_exports_livekit_triple_to_env_for_the_sdk(monkeypatch, tmp_path):
    """livekit-agents reads LIVEKIT_* from os.environ, not from our config.

    Regression: file-first resolution made the config correct while the
    worker still died with "ws_url is required, or set LIVEKIT_URL
    environment variable" (AgentServer never saw the file's values).
    """
    _write_configs(tmp_path)
    monkeypatch.setenv("FLOAT_CONFIG_DIR", str(tmp_path))
    for name in ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET"):
        monkeypatch.delenv(name, raising=False)

    config = AgentConfig.from_env()
    assert os.environ["LIVEKIT_URL"] == config.livekit_url == "ws://127.0.0.1:7880"
    assert os.environ["LIVEKIT_API_KEY"] == config.livekit_api_key == "devkey"
    assert os.environ["LIVEKIT_API_SECRET"] == config.livekit_api_secret


def test_agent_exported_livekit_env_follows_the_file_not_a_stale_copy(monkeypatch, tmp_path):
    """The SDK must see the same winner the config resolves (file > env)."""
    _write_configs(tmp_path)
    monkeypatch.setenv("FLOAT_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("LIVEKIT_URL", "ws://stale-project:7880")

    config = AgentConfig.from_env()
    assert config.livekit_url == "ws://127.0.0.1:7880"
    assert os.environ["LIVEKIT_URL"] == "ws://127.0.0.1:7880"


def test_agent_reports_the_shared_config_dir(monkeypatch, tmp_path):
    """Startup errors name the directory actually read (config-dir drift)."""
    _write_configs(tmp_path)
    monkeypatch.setenv("FLOAT_CONFIG_DIR", str(tmp_path))
    config = AgentConfig.from_env()
    assert config.config_dir == tmp_path


def test_agent_shared_file_wins_over_env_secret(monkeypatch, tmp_path):
    """secerts.toml is the single source of truth; env is only a fallback."""
    _write_configs(tmp_path, gemini_key="from-file")
    monkeypatch.setenv("FLOAT_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("GOOGLE_API_KEY", "from-env")

    config = AgentConfig.from_env()
    assert config.google_api_key == "from-file"

    # Without the shared config the env keeps working (fallback). The call
    # above re-exported the winning file key into the real environment
    # (that is the google-genai contract), so restore the env copy.
    monkeypatch.setenv("GOOGLE_API_KEY", "from-env")
    (tmp_path / "secerts.toml").write_text("", encoding="utf-8")
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
    monkeypatch.setenv("GEMINI_VOICE", "Aster")

    settings = LiveKitSettings.from_env()
    # The shared file is the source of truth for the key...
    assert settings.gemini.api_key == "from-file"
    # ...and env is the fallback where the file does not define the key.
    assert settings.gemini.voice == "Aster"


def test_livekit_settings_file_wins_over_env_copy_and_warns(monkeypatch, caplog, visible_fast_float_logs):
    """A disagreeing env copy must not silently fight the shared file."""
    monkeypatch.setattr(
        "fast_float.livekit.settings.load_secrets",
        lambda: {
            "livekit": {"url": "wss://demo.livekit.cloud", "api_key": "APIfile", "api_secret": "s" * 44},
        },
    )
    monkeypatch.setattr("fast_float.livekit.settings.load_settings", lambda: {})
    monkeypatch.setenv("LIVEKIT_URL", "wss://stale-env.livekit.cloud")
    monkeypatch.setenv("LIVEKIT_API_KEY", "APIstale")
    # config.from_env exports winners into the real environment (that is the
    # SDK contract), so a secret may be left over from an earlier test.
    monkeypatch.delenv("LIVEKIT_API_SECRET", raising=False)

    with caplog.at_level(logging.WARNING, logger="fast_float.livekit.settings"):
        settings = LiveKitSettings.from_env()
    assert settings.api_key == "APIfile"
    assert settings.source == "secerts.toml"
    drift = [r for r in caplog.records if "different deployments" in r.getMessage()]
    warned = {r.args[0] for r in drift}
    assert {"LIVEKIT_URL", "LIVEKIT_API_KEY"} <= warned  # stale env copies flagged


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
    dispatch_obj = SimpleNamespace(id="AD_dispatch1")

    class _FakeAgentDispatch:
        async def create_dispatch(self, request):
            captured["request"] = request
            return dispatch_obj

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

    # The created dispatch object is returned, so the caller can delete it
    # again on teardown (dispatch-leak fix).
    assert asyncio.run(_run()) is dispatch_obj
    assert captured["request"].kwargs["agent_name"] == "float-gemini"
    assert json.loads(captured["request"].kwargs["metadata"]) == {
        "gemini_api_key": "AI-browser",
        "gemini_model": "gemini-3.8-live",
    }


def test_session_deletes_its_dispatch_on_teardown(monkeypatch):
    """Dispatch-leak regression: a torn-down session must not leave its SFU
    dispatch behind — each leaked dispatch shows up as one more 'NO job' line
    in every later failure report."""
    from fast_float.livekit.session import ConversationSession

    captured = {}

    class _FakeAgentDispatch:
        async def delete_dispatch(self, dispatch_id, room_name):
            captured["dispatch_id"] = dispatch_id
            captured["room_name"] = room_name
            return None

    class _FakeLiveKitAPI:
        def __init__(self, **_kwargs):
            self.agent_dispatch = _FakeAgentDispatch()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

    monkeypatch.setattr("livekit.api.LiveKitAPI", _FakeLiveKitAPI)

    session = ConversationSession.__new__(ConversationSession)
    session.settings = SimpleNamespace(url="ws://127.0.0.1:7880", api_key="k", api_secret="s" * 32)
    session.room_name = "float-r"
    session.dispatch_id = "AD-1234567890abcdef"

    # Happy path first: the right dispatch of the right room is deleted.
    asyncio.run(session._delete_dispatch())
    assert captured == {"dispatch_id": "AD-1234567890abcdef", "room_name": "float-r"}

    # Best-effort: even when the SFU rejects the delete, the call never raises.
    class _RaisingDispatch(_FakeAgentDispatch):
        async def delete_dispatch(self, dispatch_id, room_name):
            raise RuntimeError("sfu says no")

    monkeypatch.setattr(
        "livekit.api.LiveKitAPI",
        type(
            "_FakeRaisingAPI",
            (),
            {
                "__init__": _FakeLiveKitAPI.__init__,
                "__aenter__": _FakeLiveKitAPI.__aenter__,
                "__aexit__": _FakeLiveKitAPI.__aexit__,
                "agent_dispatch": property(lambda self: _RaisingDispatch()),
            },
        ),
    )
    asyncio.run(session._delete_dispatch())  # must not raise

    # No dispatch -> no API call at all.
    session.dispatch_id = None
    asyncio.run(session._delete_dispatch())
    assert captured["dispatch_id"] == "AD-1234567890abcdef"


def test_worker_parses_dispatch_metadata_overrides():
    assert dispatch_config("") == {}
    assert dispatch_config("not-json") == {}
    assert dispatch_config('{"foo": 1}') == {}
    assert dispatch_config(json.dumps({"gemini_api_key": "AI-dispatch-key"})) == {"gemini_api_key": "AI-dispatch-key"}
    full = dispatch_config(json.dumps({"gemini_api_key": "  AI-padded  ", "gemini_model": "  m-audio  "}))
    assert full == {"gemini_api_key": "AI-padded", "gemini_model": "m-audio"}
    # Unknown keys and empty values are ignored.
    assert dispatch_config(json.dumps({"gemini_api_key": "", "other": "x"})) == {}


# ---------------------------------------------------------------------------
# Model-name validation: free-text Studio values must not silently break the
# Gemini session (the google plugin only fails on the websocket, late).
# ---------------------------------------------------------------------------


def test_sanitize_gemini_model_passes_known_and_wellformed_ids(caplog):
    from fast_float.livekit.gemini.config import known_live_models, sanitize_gemini_model

    known = known_live_models()
    assert known  # a fallback snapshot is always available
    assert sanitize_gemini_model(next(iter(known))) in known
    # Whitespace, surrounding quotes and case are normalised; ids are lowercase.
    assert sanitize_gemini_model('  "Gemini-3.8-LIVE"  ') == "gemini-3.8-live"
    # Unknown but well-formedpreview ids are forwarded (with a warning), not
    # dropped: Google ships ids the plugin's list may lag behind.
    assert sanitize_gemini_model("gemini-9.9-flash-live-future-preview").startswith("gemini-9.9-flash-live")
    # models/ prefix is stripped: the plugin adds it itself.
    assert sanitize_gemini_model("models/gemini-3.8-live") == "gemini-3.8-live"
    # Provider-qualified names from a Studio/gateway paste are normalized.
    # "google/gemini-3.8-live" reached the Live API as
    # "models/google/gemini-3.8-live" and the session died with 1008
    # (policy violation) as if the model did not exist.
    assert sanitize_gemini_model("google/gemini-3.8-live") == "gemini-3.8-live"
    assert sanitize_gemini_model(' "Google/gemini-3.8-live" ') == "gemini-3.8-live"


def test_configured_model_is_normalized_too(monkeypatch, tmp_path):
    """The shared settings.toml value must pass the same validation as Studio.

    Regression: sanitize_gemini_model() only guarded the dispatch override;
    _build_realtime_model() then fell back to the raw config.model, so a
    provider-prefixed settings.toml value bypassed validation and broke every
    session with a 1008 from the Gemini websocket.
    """
    _write_configs(tmp_path, model="google/gemini-3.8-live")
    monkeypatch.setenv("FLOAT_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv("GEMINI_MODEL", raising=False)

    config = AgentConfig.from_env()
    assert config.model == "gemini-3.8-live"


def test_sanitize_gemini_model_drops_free_text():
    from fast_float.livekit.gemini.config import sanitize_gemini_model

    for bad in (
        None,
        "",
        "   ",
        "gemini flash live please",
        "explain the GPU status",
        "gemini-3.8-live!!",
        "gemini-3.8-live; rm -rf",
        "x" * 200,
    ):
        assert sanitize_gemini_model(bad) == ""


def test_dispatch_payload_drops_free_text_model_names():
    payload = _dispatch_payload("float-r", "float-gemini", gemini_model="explain the GPU status")
    assert "metadata" not in payload
    # Model-id-shaped values still pass through untouched (authoritative
    # validation happens in the worker).
    payload = _dispatch_payload("float-r", "float-gemini", gemini_model="gemini-9.9-flash-live-future")
    assert json.loads(payload["metadata"]) == {"gemini_model": "gemini-9.9-flash-live-future"}


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

    fake_created = SimpleNamespace(id="AD_wire1")

    async def fake_dispatch(settings, room, **kwargs):
        captured["settings"] = settings
        captured["room"] = room
        captured.update(kwargs)
        return fake_created

    monkeypatch.setattr("fast_float.livekit.session.dispatch_agent", fake_dispatch)
    session = ConversationSession.__new__(ConversationSession)
    session.settings = SimpleNamespace(autodispatch=True)
    session.room_name = "float-wiring"
    session.agent_name = "float-gemini"
    session.gemini_api_key = "AI-wire"
    session.gemini_model = "gemini-3.8-live"

    assert asyncio.run(session._dispatch_external_agent()) is fake_created
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
