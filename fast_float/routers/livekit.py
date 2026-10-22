"""LiveKit endpoints: hand the browser a room token, report setup, and trigger publishing.

The browser never sees the API secret: it asks this API for a short-lived,
room-scoped token and joins the room with it.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from fast_float.core.state import app_state
from fast_float.livekit.settings import LiveKitSettings, valid_room_name
from fast_float.livekit.tokens import LiveKitUnavailable, issue_token

router = APIRouter(tags=["livekit"])

_livekit_tasks: set[asyncio.Task[Any]] = set()


class LiveKitTokenRequest(BaseModel):
    """What the browser asks for. The room may be reused across a session."""

    room: str | None = Field(default=None, description="Room to join (generated when omitted)")
    identity: str | None = Field(default=None, description="Participant identity (generated when omitted)")
    name: str | None = Field(default=None, description="Display name")
    ttl_s: int | None = Field(default=None, ge=60, le=86400, description="Token lifetime")
    can_publish: bool = Field(default=False, description="Allow publishing (workers only)")


class LiveKitPublishRequest(BaseModel):
    """Request to publish a live or finished video stream to a LiveKit room."""

    room: str | None = Field(default=None, description="Target room (defaults to request_id)")


@router.get("/livekit/status")
async def livekit_status() -> dict[str, Any]:
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
async def livekit_token(request: LiveKitTokenRequest) -> dict[str, Any]:
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


@router.post("/livekit/publish/{request_id}")
async def livekit_publish(request_id: str, request: LiveKitPublishRequest | None = None) -> dict[str, Any]:
    """Publish an active LiveStream or completed MP4 file to a LiveKit room."""
    settings = LiveKitSettings.from_env()
    if not settings.configured:
        raise HTTPException(status_code=503, detail="LiveKit is not configured")

    target_room = (request.room if request else None) or request_id
    if not valid_room_name(target_room):
        raise HTTPException(status_code=400, detail="Invalid target room name")

    # 1. In-flight LiveStream
    stream = app_state.live_streams.get(request_id)
    if stream is not None:
        from fast_float.livekit.worker import publish_live_stream_task

        task = asyncio.create_task(publish_live_stream_task(stream, room_name=target_room, settings=settings))
        _livekit_tasks.add(task)
        task.add_done_callback(_livekit_tasks.discard)
        return {
            "status": "publishing",
            "type": "live",
            "room": target_room,
            "request_id": request_id,
        }

    # 2. Completed MP4 file in active_streams
    video_path = app_state.active_streams.get(request_id)
    if video_path and os.path.exists(video_path):
        from fast_float.livekit.worker import publish_media_file_task

        task = asyncio.create_task(publish_media_file_task(video_path, room_name=target_room, settings=settings))
        _livekit_tasks.add(task)
        task.add_done_callback(_livekit_tasks.discard)
        return {
            "status": "publishing",
            "type": "file",
            "room": target_room,
            "request_id": request_id,
        }

    # 3. Wait up to 2 seconds for stream registration if request is currently spinning up
    stream = await app_state.wait_for_live_stream(request_id, timeout=2.0)
    if stream is not None:
        from fast_float.livekit.worker import publish_live_stream_task

        task = asyncio.create_task(publish_live_stream_task(stream, room_name=target_room, settings=settings))
        _livekit_tasks.add(task)
        task.add_done_callback(_livekit_tasks.discard)
        return {
            "status": "publishing",
            "type": "live",
            "room": target_room,
            "request_id": request_id,
        }

    raise HTTPException(status_code=404, detail=f"Request {request_id} not found in live or active streams")


__all__ = ["LiveKitPublishRequest", "LiveKitTokenRequest", "router"]
