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

import base64
import hashlib
import hmac
import ipaddress
import os
import re
import time
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from fast_float.config.dynamic import first_value, load_secrets, load_settings

DEFAULT_TTL_S = 3600
DEFAULT_ROOM_PREFIX = "float"
DEFAULT_STUN_PORT = 3478
DEFAULT_TURN_TTL_S = 3600

# LiveKit room names: letters, digits, dash and underscore.
_ROOM_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

_FALSE_STRINGS = {"0", "false", "no", "off"}


def valid_room_name(name: str) -> bool:
    """True when ``name`` is a usable LiveKit room name."""
    return bool(_ROOM_RE.match(name or ""))


def stun_url(host: str, port: int = DEFAULT_STUN_PORT) -> str:
    """Build a ``stun:`` URL, bracketing IPv6 literals correctly."""
    host = str(host or "").strip().strip("[]")
    if not host:
        return ""
    try:
        if ipaddress.ip_address(host).version == 6:
            host = f"[{host}]"
    except ValueError:
        # A DNS name (or a host already carrying a port) is passed through.
        pass
    return f"stun:{host}:{int(port)}"


def host_only(value: str | None) -> str:
    """Extract the hostname from ``Host``/``X-Forwarded-Host`` header values.

    Accepts ``example.com:8000``, ``[2001:db8::1]:8000`` and comma-separated
    proxy chains; returns ``""`` when no usable host is present.
    """
    first = str(value or "").split(",")[0].strip()
    if not first:
        return ""
    if first.startswith("["):
        end = first.find("]")
        return first[1:end] if end > 0 else first
    # A single colon is host:port; multiple colons mean a bare IPv6 literal.
    if first.count(":") == 1:
        return first.rsplit(":", 1)[0]
    return first


@dataclass(frozen=True)
class LiveKitSettings:
    """Everything the server needs to talk to the LiveKit deployment."""

    url: str = ""
    api_key: str = ""
    api_secret: str = ""
    token_ttl_s: int = DEFAULT_TTL_S
    room_prefix: str = DEFAULT_ROOM_PREFIX
    # ICE/STUN for the raw WebRTC transport. When ``stun_urls`` is empty the
    # API itself runs a STUN responder on ``stun_port`` and advertises
    # ``stun:<public host>:<port>`` (see fast_float/utils/stun.py).
    stun_urls: tuple[str, ...] = ()
    stun_host: str = ""
    stun_port: int = DEFAULT_STUN_PORT
    stun_enabled: bool = True
    # Optional TURN relay for the raw WebRTC transport. STUN alone cannot
    # traverse a TCP-only proxy (RunPod, Cloudflare, strict corporate NAT);
    # a TURN server reachable over UDP/TCP/TLS is required there.
    turn_urls: tuple[str, ...] = ()
    turn_username: str = ""
    turn_credential: str = ""
    # coturn `use-auth-secret` shared secret: when set, short-lived credentials
    # are minted instead of exposing a static username/password.
    turn_secret: str = ""
    turn_ttl_s: int = DEFAULT_TURN_TTL_S
    source: str = "none"

    @property
    def configured(self) -> bool:
        """True when a token can be minted (url + key + secret are all set)."""
        return bool(self.url and self.api_key and self.api_secret)

    @classmethod
    def from_env(cls) -> LiveKitSettings:
        all_secrets = load_secrets()
        all_config = load_settings()
        secrets = all_secrets.get("livekit", {})
        config = all_config.get("livekit", {})
        web = all_config.get("webrtc", {})
        web_secrets = all_secrets.get("webrtc", {})

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
        env_stun = os.getenv("WEBRTC_STUN_URLS", "").strip()
        if env_stun:
            raw_stun = env_stun.split(",")
        stun_urls = tuple(url.strip() for url in (raw_stun or []) if str(url).strip())

        # Built-in STUN responder (fast_float/utils/stun.py). It is only used
        # when no explicit stun_urls are configured: an operator-provided
        # coturn/LiveKit STUN always wins.
        stun_enabled_raw = os.getenv("WEBRTC_STUN", "").strip()
        if stun_enabled_raw:
            stun_enabled = stun_enabled_raw.lower() not in _FALSE_STRINGS
        else:
            configured_enabled = web.get("stun_enabled", True)
            if isinstance(configured_enabled, str):
                stun_enabled = configured_enabled.strip().lower() not in _FALSE_STRINGS
            else:
                stun_enabled = bool(configured_enabled)
        stun_port_raw = os.getenv("WEBRTC_STUN_PORT", "").strip() or web.get("stun_port", DEFAULT_STUN_PORT)
        try:
            stun_port = int(stun_port_raw)
        except (TypeError, ValueError):
            stun_port = DEFAULT_STUN_PORT
        stun_host, _ = first_value(
            (os.getenv("WEBRTC_STUN_HOST", "").strip(), "env"),
            (str(web.get("stun_host", "") or "").strip(), "settings.toml"),
        )

        # TURN relay for browsers/aiortc behind a proxy or strict NAT. The
        # credential may be a static password, or a coturn shared secret
        # (`use-auth-secret`); the latter mints short-lived credentials.
        turn_env = os.getenv("WEBRTC_TURN_URLS", "").strip()
        if turn_env:
            raw_turn: Any = turn_env.split(",")
        else:
            raw_turn = web_secrets.get("turn_urls", web.get("turn_urls", []))
        if isinstance(raw_turn, str):
            raw_turn = [raw_turn]
        turn_urls = tuple(url.strip() for url in (raw_turn or []) if str(url).strip())

        def _layered_web(name: str, env_name: str) -> str:
            value, _ = first_value(
                (os.getenv(env_name, "").strip(), "env"),
                (str(web_secrets.get(name, "") or "").strip(), "secerts.toml"),
                (str(web.get(name, "") or "").strip(), "settings.toml"),
            )
            return str(value or "")

        turn_username = _layered_web("turn_username", "WEBRTC_TURN_USERNAME")
        turn_credential = _layered_web("turn_credential", "WEBRTC_TURN_CREDENTIAL")
        turn_secret = _layered_web("turn_secret", "WEBRTC_TURN_SECRET")
        try:
            turn_ttl = int(
                os.getenv("WEBRTC_TURN_TTL_S", "").strip()
                or web_secrets.get("turn_ttl_s", web.get("turn_ttl_s", DEFAULT_TURN_TTL_S))
            )
        except (TypeError, ValueError):
            turn_ttl = DEFAULT_TURN_TTL_S

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
            stun_host=str(stun_host or ""),
            stun_port=max(1, min(stun_port, 65535)),
            stun_enabled=stun_enabled,
            turn_urls=turn_urls,
            turn_username=turn_username,
            turn_credential=turn_credential,
            turn_secret=turn_secret,
            turn_ttl_s=max(60, min(turn_ttl, 24 * 3600)),
            source=source,
        )

    def resolved_stun_urls(self, request_host: str | None = None) -> tuple[str, ...]:
        """STUN URLs to hand to browsers/aiortc for this deployment.

        An explicit ``[webrtc].stun_urls`` list (or ``WEBRTC_STUN_URLS``) is
        authoritative. Otherwise, when the built-in responder is enabled, the
        API advertises itself at ``stun_host`` — or the host the client used
        to reach this API — so ICE gathering does not depend on LiveKit or
        Google.
        """
        if self.stun_urls:
            return self.stun_urls
        if not self.stun_enabled:
            return ()
        host = self.stun_host or (request_host or "")
        url = stun_url(host_only(host), self.stun_port)
        return (url,) if url else ()

    def turn_credentials(self, ttl_s: int | None = None) -> tuple[str, str]:
        """Mint coturn REST credentials from ``turn_secret``.

        coturn's ``use-auth-secret`` mode expects::

            username   = "<unix-expiry>:<opaque>"
            credential = base64(HMAC-SHA1(secret, username))

        Static username/password credentials are returned unchanged when no
        secret is configured.
        """
        if not self.turn_secret:
            return self.turn_username, self.turn_credential
        ttl = max(60, min(int(ttl_s or self.turn_ttl_s), 24 * 3600))
        username = f"{int(time.time()) + ttl}:{uuid4().hex[:12]}"
        digest = hmac.new(self.turn_secret.encode(), username.encode(), hashlib.sha1).digest()
        return username, base64.b64encode(digest).decode("ascii")

    def resolved_ice_servers(
        self,
        request_host: str | None = None,
        *,
        stun_available: bool = True,
    ) -> tuple[dict[str, str], ...]:
        """STUN + TURN entries ready for the browser's ``iceServers`` list."""
        stun_urls = self.resolved_stun_urls(request_host) if stun_available else ()
        servers: list[dict[str, str]] = [{"urls": url} for url in stun_urls]
        if not self.turn_urls:
            return tuple(servers)

        username, credential = self.turn_credentials()
        for url in self.turn_urls:
            entry = {"urls": url}
            if username and credential:
                entry["username"] = username
                entry["credential"] = credential
            servers.append(entry)
        return tuple(servers)

    def room_for(self, name: str | None = None) -> str:
        """The room to use: the requested one, or a prefixed generated name."""
        if name:
            return name
        return f"{self.room_prefix}-{uuid4().hex[:12]}"


__all__ = [
    "DEFAULT_ROOM_PREFIX",
    "DEFAULT_STUN_PORT",
    "DEFAULT_TTL_S",
    "DEFAULT_TURN_TTL_S",
    "LiveKitSettings",
    "host_only",
    "stun_url",
    "valid_room_name",
]
