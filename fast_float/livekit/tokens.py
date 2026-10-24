"""Mint LiveKit access tokens (room grants) for the browser.

The token is a JWT signed with the deployment's API secret; the server hands
it to the browser, which joins the room with it. The secret itself is never
sent to a client.

``livekit-api`` is an optional dependency (it is numpy-free, so it lives in
the main environment); the import is lazy so that importing this module never
fails on a box without it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from uuid import uuid4


class LiveKitUnavailable(RuntimeError):
    """The LiveKit token API is not installed in this environment."""


@dataclass(frozen=True)
class IssuedToken:
    """A minted token and everything the browser needs to use it."""

    token: str
    identity: str
    name: str
    room: str
    url: str
    expires_at: float

    def to_dict(self) -> dict[str, object]:
        return {
            "url": self.url,
            "token": self.token,
            "room": self.room,
            "identity": self.identity,
            "name": self.name,
            "expires_at": self.expires_at,
        }


def issue_token(
    *,
    url: str,
    api_key: str,
    api_secret: str,
    identity: str | None = None,
    name: str | None = None,
    room: str,
    ttl_s: int = 3600,
    can_publish: bool = False,
    agent: bool = False,
) -> IssuedToken:
    """Mint a room-scoped access token.

    ``can_publish`` stays off by default: a browser that only *watches* must
    not be able to publish into the room. The avatar worker (a server-side
    participant) gets its own token with publishing enabled.

    ``agent`` marks the participant as kind ``AGENT``. The FLOAT avatar worker
    uses that in external-agent mode so ``livekit-agents``' RoomIO does not
    mistake the avatar for the human participant and subscribe to its own
    synthesized voice.
    """
    try:
        from livekit.api import AccessToken, VideoGrants
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise LiveKitUnavailable("livekit-api is not installed; install the 'livekit' dependency group") from exc

    identity = identity or f"viewer-{uuid4().hex[:10]}"
    name = name or "FLOAT viewer"
    ttl_s = max(60, min(int(ttl_s), 24 * 3600))

    grants = VideoGrants(
        room=room,
        room_join=True,
        can_publish=can_publish,
        can_subscribe=True,
        can_publish_data=can_publish,
        agent=agent or None,
    )
    token = (
        AccessToken(api_key, api_secret)
        .with_identity(identity)
        .with_name(name)
        .with_ttl(timedelta(seconds=ttl_s))
        .with_grants(grants)
        .to_jwt()
    )
    expires_at = datetime.now(timezone.utc).timestamp() + ttl_s
    return IssuedToken(
        token=token,
        identity=identity,
        name=name,
        room=room,
        url=url,
        expires_at=expires_at,
    )


__all__ = ["IssuedToken", "LiveKitUnavailable", "issue_token"]
