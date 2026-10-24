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
import logging
import os
import re
import typing
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

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


def _shared_first(env_name: str, file_raw: object) -> str:
    """Deployment value where the shared config file wins, env only falls back.

    ``fast_float/config/secerts.toml`` (and ``settings.toml``) is read by both
    the FLOAT API and this worker, so its copy is guaranteed to agree on both
    sides. An env copy in ``agents/.env`` is only a fallback for boxes without
    the shared config: when both exist and differ, the file wins and the stale
    env copy is flagged — a silent mismatch here meant registering the worker
    on a different LiveKit project than the API dispatches into (the "NO job"
    dispatch symptom).
    """
    env_value = os.getenv(env_name, "").strip()
    file_value = _first_set(file_raw)
    if file_value:
        if env_value and env_value != file_value:
            logger.warning(
                "%s is set both in agents/.env and in the shared FLOAT config (secerts.toml/"
                "settings.toml), with different values. The shared file wins — remove or align "
                "the .env copy, or this worker and the FLOAT API will use different deployments.",
                env_name,
            )
        return file_value
    return env_value


def dispatch_config(metadata: str) -> dict[str, str]:
    """Per-session overrides sent by the FLOAT API in the dispatch metadata.

    The Studio's Config panel forwards a Gemini key and/or model name with the
    session start; the API puts them into the dispatch ``metadata``
    (``{"gemini_api_key": "...", "gemini_model": "..."}``) and the worker reads
    them from ``job.metadata``, preferring them over its own config for that
    session. Unknown keys are ignored; only non-empty values are returned.
    """
    raw = (metadata or "").strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except ValueError:
        return {}
    if not isinstance(data, dict):
        return {}
    overrides: dict[str, str] = {}
    for key in ("gemini_api_key", "gemini_model"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            overrides[key] = value.strip()
    return overrides


# --------------------------------------------------------------- model name
# ``gemini_model`` arrives from a free-text Studio field and reaches
# ``RealtimeModel(model=...)`` unfiltered: the google plugin accepts *any*
# string and defers rejection to the session's Gemini websocket — i.e. after
# the dispatch and the avatar session are already up, where the operator sees
# the opaque "worker did not join / no audio" failure. The name also silently
# changes plugin heuristics keyed on substrings (reply placeholder, tool
# behavior). Validate the format at this boundary so the worst case is
# "fall back to the box default", never a broken session.

# Live-API model ids known to current plugin snapshots; the *installed*
# plugin's list is preferred at runtime (see :func:`known_live_models`).
_KNOWN_LIVE_MODELS_FALLBACK = frozenset(
    {
        "gemini-3.8-live",
        "gemini-live-2.5-flash-native-audio",
        "gemini-3.8-live-extended-thinking",
        "gemini-3.1-flash-live-preview",
        "gemini-2.5-flash-native-audio-preview-12-2025",
        "gemini-2.0-flash-live-001",
    }
)

# Google's model-id charset: lowercase alnum with `.`/`-`/`_`, optionally
# prefixed with ``models/``. Longer than any real id, so it cannot hide
# free-text explanations in the metadata field.
_MODEL_ID_RE = re.compile(r"^(?:models/)?[a-z0-9][a-z0-9._-]{0,98}$")


def known_live_models() -> frozenset[str]:
    """Live-API model ids the *installed* google plugin knows.

    Falls back to a bundled snapshot when ``livekit-agents`` is not
    importable (e.g. unit tests of the shared config in the main repo).
    """
    try:
        from livekit.plugins.google.realtime.api_proto import LiveAPIModels
    except Exception:
        return _KNOWN_LIVE_MODELS_FALLBACK
    names = typing.get_args(LiveAPIModels)
    return frozenset(names) if names else _KNOWN_LIVE_MODELS_FALLBACK


def sanitize_gemini_model(value: str | None) -> str:
    """Validate a browser/Studio-supplied ``gemini_model`` for ``RealtimeModel``.

    The installed plugin never rejects unknown ids locally, so:

    * a known Live-API model id is returned as-is;
    * an unknown but well-formed id (``models/`` prefix allowed) is returned
      with a warning — Google ships preview ids the plugin's list lags
      behind, so names must not be hard-restricted;
    * anything else (typos, prose, wrong charset) yields ``""`` so the caller
      keeps its configured default.

    Returned ids are lowercase; Google model ids are.
    """
    cleaned = (value or "").strip().strip("\"'").strip()
    if not cleaned:
        return ""
    cleaned = cleaned.lower()
    if cleaned in known_live_models():
        return cleaned
    if _MODEL_ID_RE.match(cleaned):
        logger.warning(
            "[config] gemini_model %r is not a known Live-API model for the installed "
            "google plugin; forwarding it unchecked (the API and not the format will "
            "be the judge)",
            cleaned,
        )
        return cleaned
    logger.warning(
        f"[config] gemini_model {value!r} does not look like a model id; ignoring it "
        "and keeping the configured default"
    )
    return ""


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

        api_key = _shared_first("GOOGLE_API_KEY", gemini_cfg.get("api_key"))
        if not api_key:
            api_key = os.getenv("GEMINI_API_KEY", "").strip()
        # google-genai reads GOOGLE_API_KEY from the environment; keep the
        # SDK in sync with whatever layer won above, so a stale env copy
        # (already warned about in _shared_first) cannot win for the SDK
        # while the config reports the file's key.
        if api_key and os.getenv("GOOGLE_API_KEY") != api_key:
            os.environ["GOOGLE_API_KEY"] = api_key

        try:
            timeout = float(os.getenv("FLOAT_API_TIMEOUT_S", "8") or "8")
        except ValueError:
            timeout = 8.0

        return cls(
            livekit_url=_shared_first("LIVEKIT_URL", livekit_cfg.get("url")),
            livekit_api_key=_shared_first("LIVEKIT_API_KEY", livekit_cfg.get("api_key")),
            livekit_api_secret=_shared_first("LIVEKIT_API_SECRET", livekit_cfg.get("api_secret")),
            google_api_key=api_key,
            # `vertexai = false` is the file's placeholder default, so only a
            # *set* file value wins; the env flag stays the enabling fallback.
            vertexai=(
                True
                if _as_bool(gemini_cfg.get("vertexai"))
                else _flag("GOOGLE_GENAI_USE_VERTEXAI")
            ),
            google_cloud_project=_shared_first("GOOGLE_CLOUD_PROJECT", gemini_cfg.get("project")),
            google_cloud_location=_shared_first("GOOGLE_CLOUD_LOCATION", gemini_cfg.get("location")),
            model=_shared_first("GEMINI_MODEL", gemini_cfg.get("model")),
            voice=_shared_first("GEMINI_VOICE", gemini_cfg.get("voice")) or "Puck",
            language=_shared_first("GEMINI_LANGUAGE", gemini_cfg.get("language")),
            avatar_identity=_shared_first("FLOAT_AVATAR_IDENTITY", gemini_cfg.get("avatar_identity"))
            or "float-avatar",
            agent_name=_shared_first("FLOAT_GEMINI_AGENT_NAME", gemini_cfg.get("agent_name"))
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


__all__ = [
    "DEFAULT_INSTRUCTIONS",
    "AgentConfig",
    "dispatch_config",
    "known_live_models",
    "sanitize_gemini_model",
]
