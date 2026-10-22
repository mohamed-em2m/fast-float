"""LiveKit endpoints: hand the browser a room token, and report the setup state.

The browser never sees the API secret: it asks this API for a short-lived,
room-scoped token and joins the room with it.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from fast_float.livekit.settings import LiveKitSettings, valid_room_name
from fast_float.livekit.tokens import LiveKitUnavailable, issue_token

router = APIRouter(tags=["livekit"])


class LiveKitTokenRequest(BaseModel):
    """What the browser asks for. The room may be reused across a session."""

    room: str | None = Field(default=None, description="Room to join (generated when omitted)")
    identity: str | None = Field(default=None, description="Participant identity (generated when omitted)")
    name: str | None = Field(default=None, description="Display name")
    ttl_s: int | None = Field(default=None, ge=60, le=86400, description="Token lifetime")
    can_publish: bool = Field(default=False, description="Allow publishing (workers only)")


@router.get("/livekit/status")
async def livekit_status() -> dict:
    """Whether LiveKit is configured, and with which settings.

    The frontend uses this to show or hide the LiveKit transport.
    """
    settings = LiveKitSettings.from_env()
    return {
        "configured": settings.configured,
        "url": settings.url or None,
        "room_prefix": settings.room_prefix,
        "token_ttl_s": settings.token_ttl_s,
    }


@router.post("/livekit/token")
async def livekit_token(request: LiveKitTokenRequest) -> dict:
    """Mint a room token for the browser."""
    settings = LiveKitSettings.from_env()
    if not settings.configured:
        raise HTTPException(
            status_code=503,
            detail=(
                "LiveKit is not configured. Set LIVEKIT_URL, LIVEKIT_API_KEY and "
                "LIVEKIT_API_SECRET (see docker-compose.livekit.yml)."
            ),
        )

    room = settings.room_for(request.room)
    if not valid_room_name(room):
        raise HTTPException(status_code=400, detail="Invalid room name")

    try:
        issued = issue_token(
            url=settings.url,
            api_key=settings.api_key,
            api_secret=settings.api_secret,
            identity=request.identity,
            name=request.name,
            room=room,
            ttl_s=request.ttl_s or settings.token_ttl_s,
            can_publish=request.can_publish,
        )
    except LiveKitUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return issued.to_dict()


__all__ = ["LiveKitTokenRequest", "router"]
