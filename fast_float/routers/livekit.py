"""LiveKit endpoints: hand the browser a room token, report setup, and trigger publishing.

The browser never sees the API secret: it asks this API for a short-lived,
room-scoped token and joins the room with it.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from fast_float.core.state import app_state
from fast_float.livekit.session import get_session, start_session, stop_session
from fast_float.livekit.settings import LiveKitSettings, host_only, valid_room_name
from fast_float.livekit.tokens import LiveKitUnavailable, issue_token
from fast_float.livekit.worker import (
    RTC_AVAILABLE,
    rtc_unavailable_message,
    spawn_live_publish_task,
    spawn_media_publish_task,
)

router = APIRouter(tags=["livekit"])


def _request_host(request: Request) -> str | None:
    """Public host the client used, preferring proxy headers over the socket."""
    return (
        host_only(request.headers.get("x-forwarded-host"))
        or host_only(request.headers.get("host"))
        or request.url.hostname
    )


class LiveKitTokenRequest(BaseModel):
    """What the browser asks for. The room may be reused across a session."""

    room: str | None = Field(default=None, description="Room to join (generated when omitted)")
    identity: str | None = Field(default=None, description="Participant identity (generated when omitted)")
    name: str | None = Field(default=None, description="Display name")
    ttl_s: int | None = Field(default=None, ge=60, le=86400, description="Token lifetime")


class LiveKitPublishRequest(BaseModel):
    """Request to publish a live or finished video stream to a LiveKit room."""

    room: str | None = Field(default=None, description="Target room (defaults to request_id)")


class ConversationStartRequest(BaseModel):
    """Start an in-process conversational avatar session (agent + renderer + worker)."""

    room: str | None = Field(default=None, description="Room to run the conversation in (generated when omitted)")
    image_path: str | None = Field(default=None, description="Reference face image (default placeholder when omitted)")


class ConversationSpeakRequest(BaseModel):
    """Say something through the session's TTS engine (renders and publishes it)."""

    room: str = Field(description="The conversation room")
    text: str = Field(description="Utterance for the avatar to say")


class ConversationStopRequest(BaseModel):
    """Stop the conversation session in a room."""

    room: str = Field(description="The conversation room")


@router.get("/livekit/status")
async def livekit_status(request: Request) -> dict[str, Any]:
    """Whether LiveKit is configured, and with which settings (and from where).

    The frontend uses this to show or hide the LiveKit transport, to feed the
    browser STUN servers, and to know when to fall back to the raw WebRTC
    transport (``POST /offer``), which is signalled by this same API.

    ``stun_urls`` prefers an explicit ``[webrtc].stun_urls`` list; otherwise it
    advertises the API's own built-in STUN responder so ICE gathering does not
    depend on LiveKit (or a public Google server). The URL host is the one the
    client used to reach us, which is also the host its ICE candidates must
    target.
    """
    settings = LiveKitSettings.from_env()
    stun_urls = list(settings.resolved_stun_urls(_request_host(request)))
    # Only advertise the built-in responder once its UDP socket actually bound.
    if stun_urls and not settings.stun_urls and app_state.stun_port is None:
        stun_urls = []
    # Complete ICE list for the raw WebRTC transport: STUN plus optional TURN.
    # Secrets are never returned; coturn REST credentials are short-lived.
    ice_servers = [
        dict(server)
        for server in settings.resolved_ice_servers(
            _request_host(request),
            stun_available=bool(stun_urls),
        )
    ]
    return {
        "configured": settings.configured,
        "url": settings.url or None,
        "source": settings.source,
        "room_prefix": settings.room_prefix,
        "token_ttl_s": settings.token_ttl_s,
        "stun_urls": stun_urls,
        "ice_servers": ice_servers,
        "turn_urls": list(settings.turn_urls),
        "stun_port": app_state.stun_port,
        # The WebSocket live transport still exists, but it is a last resort:
        # the frontend falls back from LiveKit to raw WebRTC first.
        "ws_fallback": True,
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
            # A browser only ever watches: publish-capable tokens are minted
            # internally for the avatar worker, never handed to a client.
            can_publish=False,
        )
    except LiveKitUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return issued.to_dict()


def _publishing_response(type_: str, target_room: str, request_id: str, task: asyncio.Task | None) -> dict[str, Any]:
    return {
        "status": "already_publishing" if task is None else "publishing",
        "type": type_,
        "room": target_room,
        "request_id": request_id,
    }


@router.post("/livekit/publish/{request_id}")
async def livekit_publish(request_id: str, request: LiveKitPublishRequest | None = None) -> dict[str, Any]:
    """Publish an active LiveStream or completed MP4 file to a LiveKit room."""
    settings = LiveKitSettings.from_env()
    if not settings.configured:
        raise HTTPException(status_code=503, detail="LiveKit is not configured")

    target_room = (request.room if request else None) or request_id
    if not valid_room_name(target_room):
        raise HTTPException(status_code=400, detail="Invalid target room name")

    # The publish task runs the RTC SDK in this process; fail loudly here
    # instead of returning 200 "publishing" and failing silently in the task.
    if not RTC_AVAILABLE:
        raise HTTPException(status_code=503, detail=rtc_unavailable_message())

    # 1. In-flight LiveStream
    stream = app_state.live_streams.get(request_id)
    if stream is not None:
        task = spawn_live_publish_task(stream, room_name=target_room, settings=settings)
        return _publishing_response("live", target_room, request_id, task)

    # 2. Completed MP4 file in active_streams
    video_path = app_state.active_streams.get(request_id)
    if video_path and os.path.exists(video_path):
        task = spawn_media_publish_task(video_path, room_name=target_room, settings=settings)
        return _publishing_response("file", target_room, request_id, task)

    # 3. Wait up to 2 seconds for stream registration if request is currently spinning up
    stream = await app_state.wait_for_live_stream(request_id, timeout=2.0)
    if stream is not None:
        task = spawn_live_publish_task(stream, room_name=target_room, settings=settings)
        return _publishing_response("live", target_room, request_id, task)

    raise HTTPException(status_code=404, detail=f"Request {request_id} not found in live or active streams")


def _conversation_guards() -> LiveKitSettings:
    settings = LiveKitSettings.from_env()
    if not settings.configured:
        raise HTTPException(status_code=503, detail="LiveKit is not configured")
    if not RTC_AVAILABLE:
        raise HTTPException(status_code=503, detail=rtc_unavailable_message())
    if app_state.agent is None:
        raise HTTPException(
            status_code=503,
            detail="The FLOAT model is still warming up — retry the conversation shortly.",
        )
    return settings


@router.post("/livekit/conversation/start")
async def conversation_start(request: ConversationStartRequest) -> dict[str, Any]:
    """Spin up an in-process conversational session: agent + renderer + worker."""
    settings = _conversation_guards()
    room = settings.room_for(request.room)
    if not valid_room_name(room):
        raise HTTPException(status_code=400, detail="Invalid room name")

    try:
        session = await start_session(room, settings=settings, image_path=request.image_path)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {
        "status": "ready",
        "room": session.room_name,
        "identity": session.identity,
        "avatar_identity": session.avatar_identity,
        "tts": session.tts_fn is not None,
    }


@router.post("/livekit/conversation/speak")
async def conversation_speak(request: ConversationSpeakRequest) -> dict[str, Any]:
    """Push an utterance through the session's TTS into the LiveKit room."""
    session = get_session(request.room)
    if session is None:
        raise HTTPException(
            status_code=404,
            detail=f"No conversation session in room {request.room} — start one at /livekit/conversation/start",
        )
    if session.tts_fn is None:
        raise HTTPException(status_code=400, detail="This session has no TTS engine attached")
    try:
        await session.speak(request.text)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"room": request.room, "status": "speaking"}


@router.post("/livekit/conversation/stop")
async def conversation_stop(request: ConversationStopRequest) -> dict[str, Any]:
    """Tear the conversation session down (streams end, worker leaves the room)."""
    had_session = get_session(request.room) is not None
    await stop_session(request.room)
    return {"room": request.room, "status": "stopped", "had_session": had_session}


__all__ = [
    "ConversationSpeakRequest",
    "ConversationStartRequest",
    "ConversationStopRequest",
    "LiveKitPublishRequest",
    "LiveKitTokenRequest",
    "router",
]
