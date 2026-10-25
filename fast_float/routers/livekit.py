"""LiveKit endpoints: hand the browser a room token, report setup, and trigger publishing.

The browser never sees the API secret: it asks this API for a short-lived,
room-scoped token and joins the room with it.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import logging
import os
from dataclasses import asdict
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from fast_float.core.state import app_state
from fast_float.livekit import selftest
from fast_float.livekit.session import get_session, start_session, stop_session
from fast_float.livekit.settings import LiveKitSettings, host_only, valid_room_name
from fast_float.livekit.tokens import LiveKitUnavailable, issue_token
from fast_float.livekit.worker import (
    RTC_AVAILABLE,
    live_publish_active,
    publish_error,
    rtc_unavailable_message,
    spawn_live_publish_task,
    spawn_media_publish_task,
)

try:  # the RTC SDK is optional; importing this module must not require it
    from livekit.rtc import ConnectError as _RtcConnectError
except Exception:  # pragma: no cover - depends on the installed SDK
    _RtcConnectError = None  # type: ignore[assignment,misc]

# Empty tuple = "catch nothing" when the SDK is absent, so the handler below
# can stay a plain `except` clause.
_RTC_CONNECT_ERRORS: tuple[type[BaseException], ...] = (_RtcConnectError,) if _RtcConnectError else ()

logger = logging.getLogger(__name__)

router = APIRouter(tags=["livekit"])


def _connect_error_detail(settings: LiveKitSettings, exc: BaseException) -> str:
    """Turn an RTC ``ConnectError`` into an actionable message for the operator.

    The LiveKit server deliberately reports *every* JWT rejection as
    ``401 ... invalid API key`` (it does not reveal whether the key id or the
    signature was wrong), so the message always points at the credential pair.
    """
    message = str(exc) or type(exc).__name__
    lowered = message.lower()
    if "invalid api key" not in lowered and "unauthorized" not in lowered:
        return f"LiveKit connection to {settings.url or '(unconfigured)'} failed: {message}"

    host = str(settings.url or "").split("://", 1)[-1].split("/", 1)[0]
    where = f" ({host})" if host else ""
    return (
        f"LiveKit rejected the API key for {settings.url or '(unconfigured)'}{where}: {message}. "
        "The key/secret pair is not valid for this project — they were revoked, copied from a "
        "different project, or the URL points at another deployment. Copy the WebSocket URL, "
        "API key and secret together from LiveKit Cloud -> your project -> Settings -> Keys "
        f"(current values come from {settings.source}), then retry."
    )


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
    publish_mic: bool = Field(
        default=False,
        description="Allow publishing (needed by Gemini Live, which listens to the room microphone)",
    )


class LiveKitPublishRequest(BaseModel):
    """Request to publish a live or finished video stream to a LiveKit room."""

    room: str | None = Field(default=None, description="Target room (defaults to request_id)")


class ConversationStartRequest(BaseModel):
    """Start a conversational avatar session (renderer + avatar worker)."""

    room: str | None = Field(default=None, description="Room to run the conversation in (generated when omitted)")
    image_path: str | None = Field(default=None, description="Reference face image (default placeholder when omitted)")
    image_base64: str | None = Field(
        default=None,
        max_length=24_000_000,
        description="Reference face image as base64 (browser upload); image_path wins when both are set",
    )
    image_format: str = Field(default="jpg", description="Image container hint: jpg, png, webp or bmp")
    agent: Literal["builtin", "gemini"] = Field(
        default="builtin",
        description="'builtin' keeps the in-process agent; 'gemini' expects the livekit-agents worker",
    )
    agent_name: str | None = Field(
        default=None,
        description="LiveKit agent name to dispatch for the Gemini backend (defaults to FLOAT_GEMINI_AGENT_NAME)",
    )
    gemini_api_key: str | None = Field(
        default=None,
        max_length=256,
        description="Per-session Gemini API key from the Studio config; forwarded to the worker via dispatch",
    )
    gemini_model: str | None = Field(
        default=None,
        max_length=128,
        description="Per-session Gemini Live model name from the Studio config; forwarded to the worker via dispatch",
    )


class ConversationAudioRequest(BaseModel):
    """One live-microphone chunk for a running conversation session.

    The browser captures mono int16 PCM at 16 kHz and posts ~200 ms batches.
    ``flush`` renders whatever is still buffered (end of an utterance).
    """

    room: str = Field(description="The conversation room")
    audio_base64: str = Field(default="", max_length=4_000_000, description="Base64 mono int16 PCM")
    sample_rate: int = Field(default=16000, ge=8000, le=48000)
    flush: bool = Field(default=False, description="Render the buffered tail immediately")


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

    ``checks``/``warnings`` report the LiveKit + Gemini readiness that the
    Studio shows in the Config panel and Live Audio tab; ``?verify=1`` dials
    the SFU once (cached afterwards) to prove the credentials actually connect.
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

    checks = selftest.run_checks(settings)
    verify = str(request.query_params.get("verify", "")).lower() in {"1", "true", "yes", "on"}
    for index, check in enumerate(checks):
        if check.name != "livekit_connectivity":
            continue
        checks[index] = await selftest.run_verify(settings) if verify else await selftest.cached_check()
        break

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
        "avatar_identity": settings.avatar_identity,
        "gemini_agent_name": settings.agent_name,
        "gemini_autodispatch": settings.autodispatch,
        # Presence only — the key itself never leaves the server.
        "gemini_configured": settings.gemini_configured,
        # Readiness checklist the Studio surfaces to the user.
        "checks": [asdict(check) for check in checks],
        "warnings": selftest.warning_messages(checks),
        # The WebSocket live transport still exists, but it is a last resort:
        # the frontend falls back from LiveKit to raw WebRTC first.
        "ws_fallback": True,
        "max_streams": settings.livekit_max_streams,
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
            # Browsers only watch by default. Gemini Live needs the microphone
            # published into the room, so that mode opts in explicitly and the
            # grant is scoped to this room and token lifetime.
            can_publish=request.publish_mic,
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


@router.get("/livekit/publish/{request_id}")
async def livekit_publish_status(request_id: str) -> dict[str, Any]:
    """Whether a publisher is running for this request, and its last error.

    Publishing is asynchronous: ``POST /livekit/publish/{id}`` returns as soon
    as the task is spawned, so the browser polls this to find out *why* the
    room never received tracks (bad credentials, SFU unreachable, ...) instead
    of waiting forever and silently falling back.
    """
    return {
        "request_id": request_id,
        "publishing": live_publish_active(request_id),
        "error": publish_error(request_id),
    }


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
    """Start a conversational avatar session: renderer + avatar worker.

    ``agent="builtin"`` keeps the original in-process agent (browser posts PCM).
    ``agent="gemini"`` connects the avatar, dispatches the external
    ``livekit-agents`` Gemini Live worker, and waits for its first audio stream.
    """
    settings = _conversation_guards()
    room = settings.room_for(request.room)
    if not valid_room_name(room):
        raise HTTPException(status_code=400, detail="Invalid room name")

    try:
        session = await start_session(
            room,
            settings=settings,
            image_path=request.image_path,
            image_base64=request.image_base64,
            image_format=request.image_format,
            agent_backend=request.agent,
            agent_name=request.agent_name,
            gemini_api_key=request.gemini_api_key,
            gemini_model=request.gemini_model,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except _RTC_CONNECT_ERRORS as exc:
        # A rejected API key is the most common failure here. Returning a
        # readable 502 lets the Studio show the real reason instead of an ASGI
        # traceback and a bare "Internal Server Error".
        logger.warning(f"[livekit] conversation start rejected by {settings.url}: {exc}")
        raise HTTPException(status_code=502, detail=_connect_error_detail(settings, exc)) from exc

    return {
        "status": "ready",
        "room": session.room_name,
        "agent": session.agent_backend,
        "identity": session.resolved_agent_identity,
        "avatar_identity": session.avatar_identity,
        "agent_dispatched": session.agent_dispatched,
        "tts": session.tts_fn is not None,
    }


@router.get("/livekit/conversation/status")
async def conversation_status(room: str) -> dict[str, Any]:
    """Report the live session for ``room`` (used by the Studio and agent tools)."""
    session = get_session(room)
    if session is None:
        return {"room": room, "active": False}
    return {
        "room": room,
        "active": True,
        "agent": session.agent_backend,
        "identity": session.resolved_agent_identity,
        "avatar_identity": session.avatar_identity,
        "agent_dispatched": session.agent_dispatched,
        "agent_connected": session.receiver is not None and bool(session.receiver.agent_identity),
    }


@router.post("/livekit/conversation/audio")
async def conversation_audio(request: ConversationAudioRequest) -> dict[str, Any]:
    """Stream a live microphone chunk into the avatar renderer.

    This is the browser counterpart of ``ConversationalAgent.send_speech``:
    it feeds PCM directly to ``FloatVideoGenerator``, whose 2 s window and
    idle-flush timer decide when to launch a render. Audio must be mono int16
    at 16 kHz; the browser downsamples before posting.
    """
    session = get_session(request.room)
    if session is None:
        raise HTTPException(
            status_code=404,
            detail=f"No live session in room {request.room} — start one at /livekit/conversation/start",
        )
    if session.agent_backend == "gemini":
        raise HTTPException(
            status_code=409,
            detail="Gemini Live takes the microphone through LiveKit; publish the browser mic track instead.",
        )
    if request.sample_rate != 16000:
        raise HTTPException(
            status_code=400,
            detail="Live audio must be 16 kHz mono int16; downsample in the browser.",
        )

    accepted = 0
    if request.audio_base64:
        try:
            pcm = base64.b64decode(request.audio_base64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise HTTPException(status_code=400, detail="audio_base64 is not valid base64") from exc
        # Whole int16 samples only; a dangling byte would shift every later sample.
        if len(pcm) % 2:
            pcm = pcm[:-1]
        if pcm:
            session.push_audio(pcm)
            accepted = len(pcm)

    if request.flush:
        session.flush_audio()

    return {
        "room": request.room,
        "status": "ok",
        "accepted_bytes": accepted,
        "flushed": request.flush,
    }


@router.post("/livekit/conversation/interrupt")
async def conversation_interrupt(request: ConversationStopRequest) -> dict[str, Any]:
    """Barge-in: drop buffered and queued audio so the avatar stops immediately."""
    session = get_session(request.room)
    if session is None:
        raise HTTPException(status_code=404, detail=f"No live session in room {request.room}")
    session.interrupt()
    return {"room": request.room, "status": "interrupted"}


@router.post("/livekit/conversation/speak")
async def conversation_speak(request: ConversationSpeakRequest) -> dict[str, Any]:
    """Push an utterance through the session's TTS into the LiveKit room."""
    session = get_session(request.room)
    if session is None:
        raise HTTPException(
            status_code=404,
            detail=f"No conversation session in room {request.room} — start one at /livekit/conversation/start",
        )
    if session.agent_backend == "gemini":
        raise HTTPException(
            status_code=409,
            detail="Gemini Live owns the voice; speak into the browser microphone instead.",
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
    "ConversationAudioRequest",
    "ConversationSpeakRequest",
    "ConversationStartRequest",
    "ConversationStopRequest",
    "LiveKitPublishRequest",
    "LiveKitTokenRequest",
    "router",
]
