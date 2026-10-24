"""Environment-driven configuration for the Gemini Live worker."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

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


@dataclass(frozen=True)
class AgentConfig:
    """Everything the worker reads from the environment at startup."""

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
        api_key = os.getenv("GOOGLE_API_KEY", "").strip() or os.getenv("GEMINI_API_KEY", "").strip()
        # google-genai reads GOOGLE_API_KEY; accept the GEMINI_* spelling too.
        if api_key and not os.getenv("GOOGLE_API_KEY"):
            os.environ["GOOGLE_API_KEY"] = api_key

        try:
            timeout = float(os.getenv("FLOAT_API_TIMEOUT_S", "8") or "8")
        except ValueError:
            timeout = 8.0

        return cls(
            livekit_url=os.getenv("LIVEKIT_URL", "").strip(),
            livekit_api_key=os.getenv("LIVEKIT_API_KEY", "").strip(),
            livekit_api_secret=os.getenv("LIVEKIT_API_SECRET", "").strip(),
            google_api_key=api_key,
            vertexai=_flag("GOOGLE_GENAI_USE_VERTEXAI"),
            google_cloud_project=os.getenv("GOOGLE_CLOUD_PROJECT", "").strip(),
            google_cloud_location=os.getenv("GOOGLE_CLOUD_LOCATION", "").strip(),
            model=os.getenv("GEMINI_MODEL", "").strip(),
            voice=os.getenv("GEMINI_VOICE", "Puck").strip() or "Puck",
            language=os.getenv("GEMINI_LANGUAGE", "").strip(),
            avatar_identity=(os.getenv("FLOAT_AVATAR_IDENTITY", "float-avatar").strip() or "float-avatar"),
            agent_name=(os.getenv("FLOAT_GEMINI_AGENT_NAME", "float-gemini").strip() or "float-gemini"),
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
            missing.append("GOOGLE_API_KEY (or GEMINI_API_KEY)")
        return missing


__all__ = ["DEFAULT_INSTRUCTIONS", "AgentConfig"]
