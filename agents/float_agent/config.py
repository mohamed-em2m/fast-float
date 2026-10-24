"""Environment-driven configuration for the Gemini Live worker.

The worker prefers environment variables (including ``agents/.env``), but the
Gemini API key and model defaults can also live in the FLOAT repo's shared
config files, so one key serves both processes:

* ``fast_float/config/secerts.toml`` → ``[gemini].api_key`` (secret)
* ``fast_float/config/settings.toml`` → ``[gemini].model/voice/language``

Finally, the FLOAT API can hand a per-session key through the dispatch
``metadata`` (``{"gemini_api_key": "..."}``) when the Studio supplied one —
see :func:`dispatch_api_key`.

Override the config directory with ``FLOAT_CONFIG_DIR`` (defaults to the
sibling ``fast_float/config`` of this repository).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_INSTRUCTIONS = """\
You are the voice interface of FLOAT, a real-time talking-avatar system.
Speak naturally, concisely and warmly. Your words are rendered by a talking
portrait, so keep turns short unless the user asks for detail.

You can inspect the FLOAT system through tools:
- float_health / float_gpu for service and GPU status,
- generation_progress for a running render,
- conversation_status for this room,
- end_conversation to stop the session.

Use a tool when the user asks about status or wants to end the call. Never
invent status values: call the tool. Do not mention tool names, JSON, or
internal implementation details out loud.
"""


def _flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw not in {"0", "false", "no", "off"}


def _repo_config_dir() -> Path:
    """Directory holding settings.toml / secerts.toml for the main repo."""
    override = os.getenv("FLOAT_CONFIG_DIR", "").strip()
    if override:
        return Path(override).expanduser()
    # agents/float_agent/config.py -> agents -> repository root.
    return Path(__file__).resolve().parents[2] / "fast_float" / "config"


def _read_toml(path: Path) -> dict[str, object]:
    """Parse one TOML file; returns {} when absent, unreadable or invalid."""
    try:
        import tomllib  # Python 3.11+
    except ImportError:  # pragma: no cover - 3.10 fallback
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ImportError:
            return {}
    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _shared_sections(*names: str) -> dict[str, dict[str, object]]:
    """Read the requested sections of settings.toml / secerts.toml.

    Returns one mapping per section; secerts.toml (per-machine) wins over
    settings.toml (defaults) per key.
    """
    out: dict[str, dict[str, object]] = {name: {} for name in names}
    config_dir = _repo_config_dir()
    for filename in ("secerts.toml", "settings.toml"):
        data = _read_toml(config_dir / filename)
        for name in names:
            section = data.get(name, {})
            if not isinstance(section, dict):
                continue
            for key, value in section.items():
                if key not in out[name] and value not in (None, ""):
                    out[name][key] = value
    return out


def _first_set(*candidates: object) -> str:
    """First candidate that is a non-empty string, stripped."""
    for value in candidates:
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def dispatch_api_key(metadata: str) -> str:
    """Extract the per-dispatch Gemini key from job metadata (``""`` when absent).

    The FLOAT API puts ``{"gemini_api_key": "..."}`` into the dispatch
    metadata when the Studio supplied a key; the worker reads it from
    ``job.metadata`` and prefers it over its own config for that session.
    """
    raw = (metadata or "").strip()
    if not raw:
        return ""
    try:
        data = json.loads(raw)
    except ValueError:
        return ""
    if not isinstance(data, dict):
        return ""
    return str(data.get("gemini_api_key") or "").strip()


@dataclass(frozen=True)
class AgentConfig:
    """Everything the worker reads from the environment at startup.

    Precedence: environment variable → shared ``secerts.toml`` /
    ``settings.toml`` → built-in default.
    """

    livekit_url: str = ""
    livekit_api_key: str = ""
    livekit_api_secret: str = ""
    google_api_key: str = ""
    vertexai: bool = False
    google_cloud_project: str = ""
    google_cloud_location: str = ""
    model: str = ""
    voice: str = "Puck"
    language: str = ""
    avatar_identity: str = "float-avatar"
    agent_name: str = "float-gemini"
    float_api_url: str = "http://127.0.0.1:8000"
    float_api_timeout_s: float = 8.0
    instructions: str = field(default=DEFAULT_INSTRUCTIONS)

    @classmethod
    def from_env(cls) -> AgentConfig:
        sections = _shared_sections("gemini", "livekit")
        gemini_cfg = sections["gemini"]
        livekit_cfg = sections["livekit"]

        api_key = os.getenv("GOOGLE_API_KEY", "").strip() or os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            api_key = str(gemini_cfg.get("api_key") or "").strip()
        # google-genai reads GOOGLE_API_KEY; accept GEMINI_* / shared config.
        if api_key and not os.getenv("GOOGLE_API_KEY"):
            os.environ["GOOGLE_API_KEY"] = api_key

        try:
            timeout = float(os.getenv("FLOAT_API_TIMEOUT_S", "8") or "8")
        except ValueError:
            timeout = 8.0

        return cls(
            livekit_url=_first_set(os.getenv("LIVEKIT_URL"), livekit_cfg.get("url")),
            livekit_api_key=_first_set(os.getenv("LIVEKIT_API_KEY"), livekit_cfg.get("api_key")),
            livekit_api_secret=_first_set(os.getenv("LIVEKIT_API_SECRET"), livekit_cfg.get("api_secret")),
            google_api_key=api_key,
            vertexai=_flag("GOOGLE_GENAI_USE_VERTEXAI") or _as_bool(gemini_cfg.get("vertexai")),
            google_cloud_project=_first_set(os.getenv("GOOGLE_CLOUD_PROJECT"), gemini_cfg.get("project")),
            google_cloud_location=_first_set(os.getenv("GOOGLE_CLOUD_LOCATION"), gemini_cfg.get("location")),
            model=_first_set(os.getenv("GEMINI_MODEL"), gemini_cfg.get("model")),
            voice=_first_set(os.getenv("GEMINI_VOICE"), gemini_cfg.get("voice")) or "Puck",
            language=_first_set(os.getenv("GEMINI_LANGUAGE"), gemini_cfg.get("language")),
            avatar_identity=_first_set(
                os.getenv("FLOAT_AVATAR_IDENTITY"),
                gemini_cfg.get("avatar_identity"),
                livekit_cfg.get("avatar_identity"),
            )
            or "float-avatar",
            agent_name=_first_set(
                os.getenv("FLOAT_GEMINI_AGENT_NAME"), gemini_cfg.get("agent_name"), livekit_cfg.get("agent_name")
            )
            or "float-gemini",
            float_api_url=(
                os.getenv("FLOAT_API_URL", "http://127.0.0.1:8000").strip() or "http://127.0.0.1:8000"
            ).rstrip("/"),
            float_api_timeout_s=max(1.0, min(timeout, 60.0)),
            instructions=os.getenv("FLOAT_AGENT_INSTRUCTIONS", "").strip() or DEFAULT_INSTRUCTIONS,
        )

    def missing(self) -> list[str]:
        """Required settings that are absent (for a readable startup error)."""
        missing = [
            name
            for name, value in (
                ("LIVEKIT_URL", self.livekit_url),
                ("LIVEKIT_API_KEY", self.livekit_api_key),
                ("LIVEKIT_API_SECRET", self.livekit_api_secret),
            )
            if not value
        ]
        if not self.vertexai and not self.google_api_key:
            missing.append("GOOGLE_API_KEY (or [gemini].api_key in fast_float/config/secerts.toml)")
        return missing


def _as_bool(value: object, default: bool = False) -> bool:
    """Coerce a shared-config value to bool (strings included)."""
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    if isinstance(value, bool):
        return value
    return bool(value) if value is not None else default


__all__ = ["DEFAULT_INSTRUCTIONS", "AgentConfig", "dispatch_api_key"]
