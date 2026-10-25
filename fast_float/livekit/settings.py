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
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from fast_float.config.dynamic import first_value, load_secrets, load_settings

DEFAULT_TTL_S = 3600
DEFAULT_ROOM_PREFIX = "float"
DEFAULT_STUN_PORT = 3478
DEFAULT_TURN_TTL_S = 3600
DEFAULT_AVATAR_IDENTITY = "float-avatar"
DEFAULT_AGENT_NAME = "float-gemini"
# FLOAT's live renderer: NFE (ODE sampling steps) trades quality for latency.
# 5 is the fastest value the public /generate API accepts, 25 the slowest.
DEFAULT_LIVE_NFE = 6
DEFAULT_LIVE_FIRST_CHUNK_SEC = 0.4
DEFAULT_LIVE_SILENCE_SEC = 2.0
DEFAULT_LIVE_MAX_STREAMS = 0  # 0 = auto-detect based on GPU VRAM
MIN_LIVE_NFE = 5
MAX_LIVE_NFE = 25

logger = logging.getLogger(__name__)

# LiveKit room names: letters, digits, dash and underscore.
_ROOM_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

_FALSE_STRINGS = {"0", "false", "no", "off"}


def _as_bool(value: Any, default: bool = False) -> bool:
    """Coerce a TOML/env value to bool (``"false"``/``"0"`` are falsy)."""
    if isinstance(value, str):
        return value.strip().lower() not in _FALSE_STRINGS
    if isinstance(value, bool):
        return value
    return bool(value) if value is not None else default


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
class GeminiSettings:
    """Configuration for the Gemini Live voice worker.

    Read with the usual precedence: the shared ``secerts.toml`` /
    ``settings.toml`` files first, then environment variables, then built-in
    defaults. The key is consumed by
    :mod:`fast_float.livekit.gemini.agent`; the API only reports whether it is
    configured and never exposes the secret over HTTP.
    """

    #: Google Gemini API key: env ``GOOGLE_API_KEY``/``GEMINI_API_KEY``, else
    #: ``[gemini].api_key`` in secerts.toml.
    api_key: str = ""
    #: ``GEMINI_MODEL`` / ``[gemini].model`` — empty means the plugin default.
    model: str = ""
    #: ``GEMINI_VOICE`` / ``[gemini].voice`` — a Gemini prebuilt voice.
    voice: str = "Puck"
    #: ``GEMINI_LANGUAGE`` / ``[gemini].language`` — optional BCP-47 hint.
    language: str = ""
    #: ``GOOGLE_GENAI_USE_VERTEXAI`` / ``[gemini].vertexai`` — Vertex AI mode.
    vertexai: bool = False
    #: ``GOOGLE_CLOUD_PROJECT`` / ``[gemini].project`` (Vertex AI).
    project: str = ""
    #: ``GOOGLE_CLOUD_LOCATION`` / ``[gemini].location`` (Vertex AI).
    location: str = ""
    #: ``[gemini].instructions`` — inline spoken persona, or
    #: ``[gemini].instructions_file`` — a Markdown/text file with the prompt.
    instructions: str = ""
    instructions_file: str = ""


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
    # External conversational agent (livekit-agents + Gemini Live).
    #   avatar_identity: room identity the agent streams TTS audio to.
    #   agent_name: LiveKit agent name to dispatch when a Gemini session starts.
    #   autodispatch: ask the SFU to dispatch that agent automatically.
    avatar_identity: str = DEFAULT_AVATAR_IDENTITY
    agent_name: str = DEFAULT_AGENT_NAME
    autodispatch: bool = True
    #: FLOAT live renderer ODE steps (``[float].livekit_nfe``): 5 fastest,
    #: 25 best quality. The file /generate path keeps its own per-request NFE.
    livekit_nfe: int = DEFAULT_LIVE_NFE
    livekit_first_chunk_sec: float = DEFAULT_LIVE_FIRST_CHUNK_SEC
    livekit_silence_sec: float = DEFAULT_LIVE_SILENCE_SEC
    livekit_max_streams: int = DEFAULT_LIVE_MAX_STREAMS
    #: Gemini Live settings (API key + model/voice defaults, see GeminiSettings).
    gemini: GeminiSettings = field(default_factory=GeminiSettings)
    source: str = "none"

    @property
    def configured(self) -> bool:
        """True when a token can be minted (url + key + secret are all set)."""
        return bool(self.url and self.api_key and self.api_secret)

    @property
    def gemini_configured(self) -> bool:
        """True when the Gemini Live API key is available (env or config)."""
        return bool(self.gemini.api_key)

    @classmethod
    def from_env(cls) -> LiveKitSettings:
        all_secrets = load_secrets()
        all_config = load_settings()
        secrets = all_secrets.get("livekit", {})
        config = all_config.get("livekit", {})
        web = all_config.get("webrtc", {})
        web_secrets = all_secrets.get("webrtc", {})
        float_cfg = all_config.get("float", {})

        def _s(value: Any) -> str:
            return str(value or "").strip()

        # The shared FLOAT config (secerts.toml / settings.toml) is the single
        # source of truth for the deployment values that must agree between
        # this API and the Gemini worker (same LiveKit project, same agent
        # name): the file wins, and the environment is only a *fallback* for
        # boxes without the shared config. A disagreeing env copy is stale —
        # say so loudly instead of letting the copies drift apart silently.
        def _shared(name: str, env_name: str, *layers: tuple[str, str]) -> tuple[str, str]:
            env_value = os.getenv(env_name, "").strip()
            for raw_value, src in layers:
                value = str(raw_value or "").strip()
                if value:
                    if env_value and env_value != value:
                        logger.warning(
                            "%s is defined in both the shared FLOAT config and the environment "
                            "(%s, differing). The shared file (%s) wins — remove or align the env "
                            "copy, or the Gemini worker and this API will use different deployments.",
                            name,
                            env_name,
                            src,
                        )
                    return value, src
            if env_value:
                return env_value, "env"
            return "", "none"

        url, url_src = _shared(
            "LIVEKIT_URL",
            "LIVEKIT_URL",
            (_s(secrets.get("url")), "secerts.toml"),
            (_s(config.get("url")), "settings.toml"),
        )
        api_key, key_src = _shared(
            "LIVEKIT_API_KEY",
            "LIVEKIT_API_KEY",
            (_s(secrets.get("api_key")), "secerts.toml"),
            (_s(config.get("api_key")), "settings.toml"),
        )
        api_secret, sec_src = _shared(
            "LIVEKIT_API_SECRET",
            "LIVEKIT_API_SECRET",
            (_s(secrets.get("api_secret")), "secerts.toml"),
            (_s(config.get("api_secret")), "settings.toml"),
        )

        # Gemini Live voice worker (fast_float/livekit/gemini). The worker
        # consumes the key; the API only stores it for the status checks.
        gemini_secrets = all_secrets.get("gemini", {})
        gemini_cfg = all_config.get("gemini", {})
        gemini_key, _ = first_value(
            (_s(gemini_secrets.get("api_key")), "secerts.toml"),
            (_s(gemini_cfg.get("api_key")), "settings.toml"),
            (
                os.getenv("GOOGLE_API_KEY", "").strip() or os.getenv("GEMINI_API_KEY", "").strip(),
                "env",
            ),
        )
        gemini_model, _ = first_value(
            (_s(gemini_cfg.get("model")), "settings.toml"),
            (os.getenv("GEMINI_MODEL", "").strip(), "env"),
        )
        gemini_voice, _ = first_value(
            (_s(gemini_cfg.get("voice")), "settings.toml"),
            (os.getenv("GEMINI_VOICE", "").strip(), "env"),
        )
        gemini_lang, _ = first_value(
            (_s(gemini_cfg.get("language")), "settings.toml"),
            (os.getenv("GEMINI_LANGUAGE", "").strip(), "env"),
        )
        gemini_vertex_raw = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").strip()
        gemini_vertex = (
            gemini_vertex_raw.lower() not in _FALSE_STRINGS
            if gemini_vertex_raw
            else _as_bool(gemini_cfg.get("vertexai", False))
        )
        gemini_project, _ = first_value(
            (_s(gemini_cfg.get("project")), "settings.toml"),
            (os.getenv("GOOGLE_CLOUD_PROJECT", "").strip(), "env"),
        )
        gemini_location, _ = first_value(
            (_s(gemini_cfg.get("location")), "settings.toml"),
            (os.getenv("GOOGLE_CLOUD_LOCATION", "").strip(), "env"),
        )
        # The spoken persona lives in settings.toml (inline or a prompt file)
        # so an operator can edit it without touching code. The env fallback
        # is kept for existing boxes.
        gemini_instructions = _s(gemini_cfg.get("instructions"))
        gemini_instructions_file = _s(gemini_cfg.get("instructions_file"))

        # FLOAT live renderer speed/quality. This is intentionally file-first:
        # the API and the avatar worker must render every chunk identically.
        try:
            livekit_nfe = int(float_cfg.get("livekit_nfe", DEFAULT_LIVE_NFE))
        except (TypeError, ValueError):
            livekit_nfe = DEFAULT_LIVE_NFE
        livekit_nfe = max(MIN_LIVE_NFE, min(MAX_LIVE_NFE, livekit_nfe))
        try:
            livekit_first_chunk_sec = float(float_cfg.get("livekit_first_chunk_sec", DEFAULT_LIVE_FIRST_CHUNK_SEC))
        except (TypeError, ValueError):
            livekit_first_chunk_sec = DEFAULT_LIVE_FIRST_CHUNK_SEC
        livekit_first_chunk_sec = max(0.2, min(2.0, livekit_first_chunk_sec))
        try:
            livekit_silence_sec = float(float_cfg.get("livekit_silence_sec", DEFAULT_LIVE_SILENCE_SEC))
        except (TypeError, ValueError):
            livekit_silence_sec = DEFAULT_LIVE_SILENCE_SEC
        livekit_silence_sec = max(0.5, min(10.0, livekit_silence_sec))
        try:
            livekit_max_streams = int(float_cfg.get("livekit_max_streams", DEFAULT_LIVE_MAX_STREAMS))
        except (TypeError, ValueError):
            livekit_max_streams = DEFAULT_LIVE_MAX_STREAMS
        livekit_max_streams = max(0, min(32, livekit_max_streams))

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

        # Mixing layers is the classic way to end up with a key that does not
        # match its secret (LiveKit answers every such token with
        # "401 invalid API key"). Say so once instead of debugging it live.
        if api_key and api_secret and key_src != sec_src:
            logger.warning(
                "LiveKit API key comes from %s but the API secret from %s. "
                "A pair taken from different layers is rejected as 'invalid API key'; "
                "set both from the same LiveKit project.",
                key_src,
                sec_src,
            )
        url_host = str(url or "").split("://", 1)[-1].split("/", 1)[0]
        if api_key and ".livekit.cloud" in url_host and not str(api_key).startswith("API"):
            logger.warning(
                "LiveKit Cloud URL %s is paired with an API key that does not start with 'API' "
                "(got %r...). Check that LIVEKIT_API_KEY and LIVEKIT_API_SECRET are not swapped.",
                url,
                str(api_key)[:4],
            )

        avatar_identity, _ = first_value(
            (_s(gemini_cfg.get("avatar_identity")) or _s(config.get("avatar_identity")), "settings.toml"),
            (os.getenv("FLOAT_AVATAR_IDENTITY", "").strip(), "env"),
        )
        agent_name, _ = first_value(
            (_s(gemini_cfg.get("agent_name")) or _s(config.get("agent_name")), "settings.toml"),
            (os.getenv("FLOAT_GEMINI_AGENT_NAME", "").strip(), "env"),
        )
        autodispatch_raw = os.getenv("FLOAT_GEMINI_AUTODISPATCH", "").strip()
        autodispatch = (
            autodispatch_raw.lower() not in _FALSE_STRINGS
            if autodispatch_raw
            else _as_bool(gemini_cfg.get("autodispatch", config.get("autodispatch", True)))
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
            avatar_identity=avatar_identity or DEFAULT_AVATAR_IDENTITY,
            agent_name=agent_name or DEFAULT_AGENT_NAME,
            autodispatch=autodispatch,
            livekit_nfe=livekit_nfe,
            livekit_first_chunk_sec=livekit_first_chunk_sec,
            livekit_silence_sec=livekit_silence_sec,
            livekit_max_streams=livekit_max_streams,
            gemini=GeminiSettings(
                api_key=gemini_key,
                model=gemini_model,
                voice=gemini_voice or "Puck",
                language=gemini_lang,
                vertexai=gemini_vertex,
                project=gemini_project,
                location=gemini_location,
                instructions=gemini_instructions,
                instructions_file=gemini_instructions_file,
            ),
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

        coturn's ``use-auth-secret`` mode expects:

        .. code-block:: text

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
        """The room to use: the requested one, or a prefixed generated name.

        Generated names always satisfy :func:`valid_room_name`; if the
        configured ``room_prefix`` would break the charset (or the 64-char
        cap), the default prefix is used instead so a bad operator setting can
        never mint an unusable room.
        """
        if name:
            return name
        candidate = f"{self.room_prefix}-{uuid4().hex[:12]}"
        if valid_room_name(candidate):
            return candidate
        return f"{DEFAULT_ROOM_PREFIX}-{uuid4().hex[:12]}"


__all__ = [
    "DEFAULT_AGENT_NAME",
    "DEFAULT_AVATAR_IDENTITY",
    "DEFAULT_LIVE_NFE",
    "DEFAULT_ROOM_PREFIX",
    "DEFAULT_STUN_PORT",
    "DEFAULT_TTL_S",
    "DEFAULT_TURN_TTL_S",
    "MAX_LIVE_NFE",
    "MIN_LIVE_NFE",
    "GeminiSettings",
    "LiveKitSettings",
    "host_only",
    "stun_url",
    "valid_room_name",
]
