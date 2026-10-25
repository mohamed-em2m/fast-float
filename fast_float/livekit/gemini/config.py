"""Configuration for the Gemini Live worker.

The worker shares FLOAT's config layer: :class:`fast_float.livekit.settings.LiveKitSettings`
is the single source of truth for the LiveKit deployment and the Gemini
defaults, so the API and the worker cannot drift apart. This module only adds
the worker-specific values (FLOAT API URL/timeout, instructions) and the
process-level exports the SDKs require.

Precedence (highest first), inherited from ``LiveKitSettings``:

1. the shared ``secerts.toml`` / ``settings.toml`` files (single source of truth)
2. environment variables (``LIVEKIT_*``, ``GOOGLE_API_KEY``, ``GEMINI_*``)
3. built-in defaults

Override the config directory with ``FLOAT_CONFIG_DIR``. Both processes must
read the same directory; the worker surfaces the effective one in its startup
log and errors.
"""

from __future__ import annotations

import json
import logging
import os
import re
import typing
from dataclasses import dataclass, field
from pathlib import Path

from fast_float.config.dynamic import config_dir as resolve_config_dir
from fast_float.livekit.settings import LiveKitSettings

logger = logging.getLogger(__name__)

DEFAULT_INSTRUCTIONS = """\
Your goal is to make every interaction feel natural, warm, and human.

### Speaking Style
- Speak naturally, warmly, and conversationally.
- Keep responses concise, usually 1–3 short sentences.
- Use simple, clear language that sounds natural when spoken aloud.
- Answer directly without unnecessary introductions, repetition, or filler.
- Adapt your tone and level of detail to the user's needs.
- Expand only when the user asks for more detail.
- Ask one brief follow-up question when clarification is needed.

### Real-Time Conversation
- Respond directly to what the user says.
- Maintain context across turns and avoid repeating information.
- Handle interruptions, corrections, and topic changes naturally.
- Never sound overly scripted, robotic, or excessively enthusiastic.
- Avoid long lists, dense explanations, and complicated phrasing.

Be helpful, present, and human-sounding. Say only what adds value to the conversation.
"""

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _read_instructions_file(value: str, config_dir: Path) -> str:
    """Load the configured Markdown prompt, resolving common relative bases.

    ``settings.toml`` lives in ``fast_float/config`` but the prompt folder is
    at the repository root, so a relative path is tried against the config
    directory, the current working directory and the repo root. The first
    existing file wins; a missing file falls back to the inline/env/default
    prompt instead of taking the worker down.
    """
    raw = (value or "").strip()
    if not raw:
        return ""
    path = Path(raw).expanduser()
    candidates = [path] if path.is_absolute() else [config_dir / path, Path.cwd() / path, _REPO_ROOT / path]
    for candidate in candidates:
        try:
            if candidate.is_file():
                return candidate.read_text(encoding="utf-8").strip()
        except OSError as exc:
            logger.warning("[config] could not read instructions file %s: %s", candidate, exc)
            return ""
    logger.warning("[config] instructions_file not found: %s (checked %s)", raw, ", ".join(map(str, candidates)))
    return ""


def _export_env(name: str, value: str) -> None:
    """Expose a resolved value to libraries that read ``os.environ`` directly.

    ``livekit-agents`` reads ``LIVEKIT_URL`` / ``LIVEKIT_API_KEY`` /
    ``LIVEKIT_API_SECRET`` when the worker registers, and google-genai reads
    ``GOOGLE_API_KEY``. They have no notion of the file-first precedence, so
    the winner must be exported here. Without this the worker exits with
    "ws_url is required, or set LIVEKIT_URL environment variable" even though
    secerts.toml defines it.
    """
    if value and os.environ.get(name) != value:
        os.environ[name] = value


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
    # Studios and gateways often hand over a provider-qualified name
    # ("google/gemini-3.8-live", "models/gemini-..."). The google plugin
    # prefixes "models/" itself, so such a value reaches the Live API as
    # "models/google/gemini-3.8-live" and is rejected with
    # "1008 ... not found ... for bidiGenerateContent" — a connect failure
    # that looks like a bad key. Strip the prefix before validating.
    bare = cleaned
    for prefix in ("models/", "google/", "google-ai/", "gemini/"):
        while bare.startswith(prefix):
            bare = bare[len(prefix) :]
    if bare != cleaned:
        logger.info(
            "[config] normalized gemini_model %r -> %r (provider prefix removed)",
            value,
            bare,
        )
        cleaned = bare
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
        f"[config] gemini_model {value!r} does not look like a model id; ignoring it and keeping the configured default"
    )
    return ""


@dataclass(frozen=True)
class AgentConfig:
    """Everything the worker reads from the environment at startup.

    Deployment values come from :class:`LiveKitSettings` (shared files first,
    env fallback); only worker-specific knobs are read here.
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
    config_dir: Path = field(default_factory=resolve_config_dir)

    @classmethod
    def from_env(cls) -> AgentConfig:
        settings = LiveKitSettings.from_env()

        api_key = settings.gemini.api_key
        # LiveKitSettings warns about LIVEKIT_* env copies that disagree with
        # the shared file. Do the same for the Gemini key: google-genai reads
        # GOOGLE_API_KEY from the environment, so a stale copy is otherwise
        # silently ignored by our config but still used by the SDK.
        for env_name in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
            env_value = os.getenv(env_name, "").strip()
            if env_value and api_key and env_value != api_key:
                logger.warning(
                    "%s is defined in both the shared FLOAT config and the environment "
                    "(differing). The shared file wins — remove or align the env copy, "
                    "or the Gemini worker and the FLOAT API will use different keys.",
                    env_name,
                )
        _export_env("GOOGLE_API_KEY", api_key)

        try:
            timeout = float(os.getenv("FLOAT_API_TIMEOUT_S", "8") or "8")
        except ValueError:
            timeout = 8.0

        config = cls(
            livekit_url=settings.url,
            livekit_api_key=settings.api_key,
            livekit_api_secret=settings.api_secret,
            google_api_key=api_key,
            vertexai=settings.gemini.vertexai,
            google_cloud_project=settings.gemini.project,
            google_cloud_location=settings.gemini.location,
            model=sanitize_gemini_model(settings.gemini.model),
            voice=settings.gemini.voice or "Puck",
            language=settings.gemini.language,
            avatar_identity=settings.avatar_identity or "float-avatar",
            agent_name=settings.agent_name or "float-gemini",
            float_api_url=(
                os.getenv("FLOAT_API_URL", "http://127.0.0.1:8000").strip() or "http://127.0.0.1:8000"
            ).rstrip("/"),
            float_api_timeout_s=max(1.0, min(timeout, 60.0)),
            instructions=(
                _read_instructions_file(settings.gemini.instructions_file, resolve_config_dir())
                or settings.gemini.instructions.strip()
                or os.getenv("FLOAT_AGENT_INSTRUCTIONS", "").strip()
                or DEFAULT_INSTRUCTIONS
            ),
            config_dir=resolve_config_dir(),
        )
        # livekit-agents (AgentServer / run) and google-genai read the
        # environment, not our config object; export the winners so the
        # file-first values actually reach the SDKs.
        _export_env("LIVEKIT_URL", config.livekit_url)
        _export_env("LIVEKIT_API_KEY", config.livekit_api_key)
        _export_env("LIVEKIT_API_SECRET", config.livekit_api_secret)
        return config

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


__all__ = [
    "DEFAULT_INSTRUCTIONS",
    "AgentConfig",
    "dispatch_config",
    "known_live_models",
    "sanitize_gemini_model",
]
