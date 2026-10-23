"""LiveKit settings.

Effective configuration, highest precedence first:

1. environment variables (``LIVEKIT_URL``, ``LIVEKIT_API_KEY``, ...)
2. ``fast_float/config/secerts.toml``  — per-machine secrets
3. ``fast_float/config/settings.toml`` — deployment defaults
4. built-in defaults

The API secret never leaves the server: the browser only ever receives the
server URL and a short-lived, room-scoped token.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from fast_float.config.dynamic import first_value, load_secrets, load_settings

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
    stun_urls: tuple[str, ...] = ()
    source: str = "none"

    @property
    def configured(self) -> bool:
        """True when a token can be minted (url + key + secret are all set)."""
        return bool(self.url and self.api_key and self.api_secret)

    @classmethod
    def from_env(cls) -> LiveKitSettings:
        secrets = load_secrets().get("livekit", {})
        config = load_settings().get("livekit", {})
        web = load_settings().get("webrtc", {})

        def _s(value: Any) -> str:
            return str(value or "").strip()

        url, url_src = first_value(
            (os.getenv("LIVEKIT_URL", "").strip(), "env"),
            (_s(secrets.get("url")), "secerts.toml"),
            (_s(config.get("url")), "settings.toml"),
        )
        api_key, key_src = first_value(
            (os.getenv("LIVEKIT_API_KEY", "").strip(), "env"),
            (_s(secrets.get("api_key")), "secerts.toml"),
            (_s(config.get("api_key")), "settings.toml"),
        )
        api_secret, sec_src = first_value(
            (os.getenv("LIVEKIT_API_SECRET", "").strip(), "env"),
            (_s(secrets.get("api_secret")), "secerts.toml"),
            (_s(config.get("api_secret")), "settings.toml"),
        )

        ttl_raw, _ttl_src = first_value(
            (os.getenv("LIVEKIT_TOKEN_TTL_S", "").strip(), "env"),
            (_s(config.get("token_ttl_s")), "settings.toml"),
        )
        try:
            ttl = int(ttl_raw or "") or DEFAULT_TTL_S
        except ValueError:
            ttl = DEFAULT_TTL_S

        prefix, _ = first_value(
            (os.getenv("LIVEKIT_ROOM_PREFIX", "").strip(), "env"),
            (_s(config.get("room_prefix")), "settings.toml"),
        )

        raw_stun: Any = web.get("stun_urls", [])
        if isinstance(raw_stun, str):
            raw_stun = [raw_stun]
        stun_urls = tuple(url.strip() for url in (raw_stun or []) if str(url).strip())

        # Which layer decided the effective configuration (reported by
        # /livekit/status): the highest layer that contributes url/key/secret.
        sources = (url_src, key_src, sec_src)
        source = (
            "env"
            if "env" in sources
            else "secerts.toml"
            if "secerts.toml" in sources
            else "settings.toml"
            if "settings.toml" in sources
            else "none"
        )

        return cls(
            url=url,
            api_key=api_key,
            api_secret=api_secret,
            token_ttl_s=max(60, min(ttl, 24 * 3600)),
            room_prefix=prefix or DEFAULT_ROOM_PREFIX,
            stun_urls=stun_urls,
            source=source,
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
