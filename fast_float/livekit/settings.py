"""LiveKit settings, read from the environment.

The API secret never leaves the server: the browser only ever receives the
server URL and a short-lived, room-scoped token.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from uuid import uuid4

DEFAULT_TTL_S = 3600
DEFAULT_ROOM_PREFIX = "float"

# LiveKit room names: letters, digits, dash and underscore.
_ROOM_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def valid_room_name(name: str) -> bool:
    """True when ``name`` is a usable LiveKit room name."""
    return bool(_ROOM_RE.match(name or ""))


@dataclass(frozen=True)
class LiveKitSettings:
    """Everything the server needs to talk to the LiveKit deployment."""

    url: str = ""
    api_key: str = ""
    api_secret: str = ""
    token_ttl_s: int = DEFAULT_TTL_S
    room_prefix: str = DEFAULT_ROOM_PREFIX

    @property
    def configured(self) -> bool:
        """True when a token can be minted (url + key + secret are all set)."""
        return bool(self.url and self.api_key and self.api_secret)

    @classmethod
    def from_env(cls) -> LiveKitSettings:
        try:
            ttl = int(os.getenv("LIVEKIT_TOKEN_TTL_S", "") or DEFAULT_TTL_S)
        except ValueError:
            ttl = DEFAULT_TTL_S
        return cls(
            url=os.getenv("LIVEKIT_URL", "").strip(),
            api_key=os.getenv("LIVEKIT_API_KEY", "").strip(),
            api_secret=os.getenv("LIVEKIT_API_SECRET", "").strip(),
            token_ttl_s=max(60, min(ttl, 24 * 3600)),
            room_prefix=(os.getenv("LIVEKIT_ROOM_PREFIX", "") or DEFAULT_ROOM_PREFIX).strip(),
        )

    def room_for(self, name: str | None = None) -> str:
        """The room to use: the requested one, or a prefixed generated name."""
        if name:
            return name
        return f"{self.room_prefix}-{uuid4().hex[:12]}"


__all__ = [
    "DEFAULT_ROOM_PREFIX",
    "DEFAULT_TTL_S",
    "LiveKitSettings",
    "valid_room_name",
]
