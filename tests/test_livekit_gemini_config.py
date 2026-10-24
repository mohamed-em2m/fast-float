"""Gemini API key config: shared secerts.toml / settings.toml + env precedence."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agents"))

from fastapi import FastAPI
from fastapi.testclient import TestClient
from float_agent.config import AgentConfig

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
