"""Conversation session: wires the agent, the FLOAT renderer and the avatar worker.

Two backends are supported:

* ``builtin`` — :class:`ConversationalAgent` (mic PCM posted by the browser) +
  :class:`FloatVideoGenerator` + :class:`AvatarRunner`, all in this process.
* ``gemini`` — a separate ``livekit-agents`` worker (Gemini Live
  speech-to-speech) streams TTS audio over LiveKit DataStream;
  :class:`AgentAvatarRunner` renders it and publishes synchronized A/V.

The browser joins the same room with a viewer token (POST /livekit/token) and
attaches the tracks published by the avatar worker. One session per room, kept
in :data:`_sessions`; manage through the ``/livekit/conversation/*`` endpoints.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import os
import tempfile
from collections.abc import Callable
from typing import Any

try:
    from livekit import rtc

    RTC_AVAILABLE = True
except ImportError:  # pragma: no cover
    rtc = None  # type: ignore[assignment]
    RTC_AVAILABLE = False

from fast_float.core.state import app_state
from fast_float.livekit.agent import DEFAULT_AGENT_IDENTITY, ConversationalAgent
from fast_float.livekit.agent_runner import AgentAvatarRunner
from fast_float.livekit.generator import FloatVideoGenerator
from fast_float.livekit.receiver import DataStreamAudioReceiver
from fast_float.livekit.settings import LiveKitSettings
from fast_float.livekit.tokens import issue_token
from fast_float.livekit.worker import AvatarOptions, AvatarRunner
from fast_float.utils.media import ensure_default_image

logger = logging.getLogger(__name__)

# One GPU, one inference at a time: a session render and a /generate request
# must never run concurrently on the same weights.
_render_lock = asyncio.Semaphore(1)

# room -> running session
_sessions: dict[str, ConversationSession] = {}

# Only raster formats the loader understands; anything else falls back to jpg.
_ALLOWED_IMAGE_FORMATS = {"jpg", "jpeg", "png", "webp", "bmp"}


def _safe_image_format(value: str | None) -> str:
    """Normalise an upload's container hint to a whitelisted extension."""
    ext = str(value or "jpg").strip().lower().lstrip(".")
    if ext == "jpeg":
        return "jpg"
    return ext if ext in _ALLOWED_IMAGE_FORMATS else "jpg"


def _materialize_image(image_base64: str, image_format: str) -> str:
    """Decode a browser-uploaded portrait into a temp file for the renderer."""
    try:
        raw = base64.b64decode(image_base64, validate=True)
    except Exception as exc:
        raise ValueError("image_base64 is not valid base64") from exc
    if not raw:
        raise ValueError("image_base64 is empty")
    fd, path = tempfile.mkstemp(prefix="float-live-", suffix=f".{_safe_image_format(image_format)}")
    with os.fdopen(fd, "wb") as handle:
        handle.write(raw)
    return path


async def dispatch_agent(
    settings: LiveKitSettings,
    room: str,
    *,
    agent_name: str | None = None,
    gemini_api_key: str | None = None,
) -> bool:
    """Ask the SFU to dispatch the Gemini worker into ``room``.

    ``gemini_api_key`` (when a browser supplied one) is forwarded to the
    worker inside the dispatch ``metadata``; the worker reads it from
    ``job.metadata``. The value is never written to any log.

    Returns ``False`` (and logs) when the LiveKit API is unavailable or the
    request is rejected; the avatar session stays up so an operator can start
    the worker manually.
    """
    name = (agent_name or settings.agent_name or "").strip()
    if not name:
        return False
    try:
        from livekit.api import CreateAgentDispatchRequest, LiveKitAPI

        request = CreateAgentDispatchRequest(**_dispatch_payload(room, name, gemini_api_key))
        async with LiveKitAPI(
            url=settings.url,
            api_key=settings.api_key,
            api_secret=settings.api_secret,
        ) as lk:
            await lk.agent_dispatch.create_dispatch(request)
    except Exception as exc:
        logger.warning(f"[session] could not dispatch agent {name!r} into {room}: {exc}")
        return False
    logger.info(f"[session] dispatched agent {name!r} into {room}")
    return True


def _dispatch_payload(room: str, agent_name: str, gemini_api_key: str | None = None) -> dict[str, Any]:
    """``CreateAgentDispatchRequest`` kwargs, adding metadata only when needed."""
    payload: dict[str, Any] = {"room": room, "agent_name": agent_name}
    key = (gemini_api_key or "").strip()
    if key:
        payload["metadata"] = json.dumps({"gemini_api_key": key})
    return payload


class ConversationSession:
    """One conversational avatar session in a LiveKit room."""

    def __init__(
        self,
        room_name: str,
        *,
        settings: LiveKitSettings | None = None,
        identity: str | None = None,
        image_path: str | None = None,
        image_base64: str | None = None,
        image_format: str = "jpg",
        tts_fn: Callable[[str], bytes] | None = None,
        agent_backend: str = "builtin",
        agent_name: str | None = None,
        gemini_api_key: str | None = None,
    ) -> None:
        self.room_name = room_name
        self.settings = settings or LiveKitSettings.from_env()
        self.agent_backend = (agent_backend or "builtin").strip().lower()
        if self.agent_backend not in {"builtin", "gemini"}:
            raise ValueError("agent_backend must be 'builtin' or 'gemini'")
        self.identity = identity or DEFAULT_AGENT_IDENTITY
        self.avatar_identity = (
            self.settings.avatar_identity if self.agent_backend == "gemini" else f"avatar-{room_name[:24]}"
        )
        self.agent_name = (agent_name or self.settings.agent_name or "").strip()
        # Optional per-session API key supplied by the Studio UI. Forwarded to
        # the Gemini worker in the dispatch metadata; never logged.
        self.gemini_api_key = (gemini_api_key or "").strip()
        self.image_path = image_path
        self.image_base64 = image_base64
        self.image_format = _safe_image_format(image_format)
        self.tts_fn = tts_fn
        # Written from ``image_base64`` on start and removed on stop, so a
        # browser-uploaded portrait can drive a live session without exposing
        # a server path.
        self._temp_image: str | None = None

        self.agent: ConversationalAgent | None = None
        self.generator: FloatVideoGenerator | None = None
        self.worker_room: Any = None
        self.runner: AvatarRunner | AgentAvatarRunner | None = None
        self.receiver: DataStreamAudioReceiver | None = None
        self.agent_dispatched: bool | None = None
        self._stopped = False

    @property
    def is_stopped(self) -> bool:
        return self._stopped

    async def start(self) -> None:
        """Connect the agent, the avatar worker room and start rendering."""
        if not RTC_AVAILABLE:
            raise RuntimeError("LiveKit RTC SDK is not available in this environment")
        if not self.settings.configured:
            raise RuntimeError("LiveKit settings not configured (LIVEKIT_URL / API_KEY / API_SECRET)")

        engine = app_state.agent
        if engine is None:
            raise RuntimeError("The FLOAT model is not loaded yet (still warming up)")

        if self.image_path is None and self.image_base64:
            self._temp_image = _materialize_image(self.image_base64, self.image_format)
            self.image_path = self._temp_image
            self.image_base64 = None

        if self.image_path is None:
            self.image_path = ensure_default_image()

        self.generator = FloatVideoGenerator(
            agent=engine,
            image_path=self.image_path,
            semaphore=_render_lock,
        )

        if self.agent_backend == "gemini":
            await self._start_external_agent()
            return

        self.agent = ConversationalAgent(
            room_name=self.room_name,
            settings=self.settings,
            identity=self.identity,
            avatar_identity=self.avatar_identity,
            video_gen=self.generator,
            tts_fn=self.tts_fn,
        )
        await self.agent.connect()

        token_result = issue_token(
            url=self.settings.url,
            api_key=self.settings.api_key,
            api_secret=self.settings.api_secret,
            identity=self.avatar_identity,
            name="FLOAT Avatar (worker)",
            room=self.room_name,
            can_publish=True,
        )
        self.worker_room = rtc.Room()
        await self.worker_room.connect(self.settings.url, token_result.token)

        self.runner = AvatarRunner(
            self.worker_room,
            self.generator,
            options=AvatarOptions(),
            agent_identity=self.agent.identity,
        )
        await self.runner.start()
        logger.info(f"[session] conversation ready in {self.room_name} ({self.identity} + {self.avatar_identity})")

    async def _start_external_agent(self) -> None:
        """Connect the avatar worker and wait for the Gemini Live agent."""
        if self.generator is None:
            raise RuntimeError("Generator was not created")

        token_result = issue_token(
            url=self.settings.url,
            api_key=self.settings.api_key,
            api_secret=self.settings.api_secret,
            identity=self.avatar_identity,
            name="FLOAT Avatar (Gemini Live)",
            room=self.room_name,
            can_publish=True,
            # Kind=AGENT keeps the avatar out of livekit-agents' RoomIO
            # participant selection; its synthesized voice must not loop back.
            agent=True,
        )
        self.worker_room = rtc.Room()
        await self.worker_room.connect(self.settings.url, token_result.token)

        if self.settings.autodispatch and self.agent_name:
            self.agent_dispatched = await dispatch_agent(
                self.settings,
                self.room_name,
                self.agent_name,
                gemini_api_key=self.gemini_api_key or None,
            )

        self.receiver = DataStreamAudioReceiver(self.worker_room)
        self.runner = AgentAvatarRunner(
            self.worker_room,
            video_gen=self.generator,
            receiver=self.receiver,
            options=AvatarOptions(),
        )
        try:
            await self.runner.start()
        except TimeoutError as exc:
            raise RuntimeError(str(exc)) from exc
        self.identity = self.receiver.agent_identity or self.identity
        logger.info(
            f"[session] Gemini Live conversation ready in {self.room_name} ({self.identity} -> {self.avatar_identity})"
        )

    @property
    def resolved_agent_identity(self) -> str:
        if self.receiver is not None and self.receiver.agent_identity:
            return self.receiver.agent_identity
        return self.identity

    async def speak(self, text: str) -> None:
        """Turn a user utterance reply into audio and push it to the renderer."""
        if self.agent_backend == "gemini":
            raise RuntimeError("Gemini Live owns the voice; use the room microphone instead")
        if self.agent is None:
            raise RuntimeError("Session not started")
        if self.tts_fn is None:
            raise RuntimeError("No TTS engine attached to this session")
        await self.agent.send_speech(text)

    def push_audio(self, pcm16_mono: bytes) -> None:
        """Feed live microphone PCM (16 kHz, mono, int16) to the avatar.

        The browser posts short chunks; :class:`FloatVideoGenerator` buffers
        them and starts a 2 s render window as soon as enough audio arrives.
        Call this from the event-loop thread. Gemini sessions take the
        microphone through LiveKit instead, so this path is disabled there.
        """
        if self.agent_backend == "gemini":
            raise RuntimeError("Gemini sessions publish the microphone track to the room")
        if self.generator is None:
            raise RuntimeError("Session not started")
        self.generator.push_audio(pcm16_mono)

    def flush_audio(self) -> None:
        """Render the tail of the current utterance immediately."""
        if self.agent_backend == "gemini" or self.generator is None:
            return
        self.generator.flush()

    def interrupt(self) -> None:
        """Barge-in: drop buffered audio and any queued/in-flight render."""
        if self.generator is None:
            return
        if isinstance(self.runner, AgentAvatarRunner):
            self.runner.interrupt()
            return
        self.generator.clear_buffer()

    async def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        if self.generator is not None:
            self.generator.stop()
        if self.runner is not None:
            with contextlib.suppress(Exception):
                await self.runner.stop()
            self.runner = None
        self.receiver = None
        if self.worker_room is not None:
            with contextlib.suppress(Exception):
                await self.worker_room.disconnect()
            self.worker_room = None
        if self.agent is not None:
            with contextlib.suppress(Exception):
                await self.agent.disconnect()
            self.agent = None
        if self._temp_image:
            with contextlib.suppress(OSError):
                os.unlink(self._temp_image)
            self._temp_image = None
        logger.info(f"[session] conversation stopped in {self.room_name}")


def get_session(room_name: str | None) -> ConversationSession | None:
    """The running session for ``room_name``, if any."""
    if room_name is None:
        return None
    session = _sessions.get(room_name)
    if session is not None and session.is_stopped:
        return None
    return session


async def start_session(
    room_name: str,
    *,
    settings: LiveKitSettings | None = None,
    identity: str | None = None,
    image_path: str | None = None,
    image_base64: str | None = None,
    image_format: str = "jpg",
    tts_fn: Callable[[str], bytes] | None = None,
    agent_backend: str = "builtin",
    agent_name: str | None = None,
    gemini_api_key: str | None = None,
) -> ConversationSession:
    """Start (or reuse) the conversation session for ``room_name``."""
    existing = get_session(room_name)
    if existing is not None:
        logger.info(f"[session] reusing existing conversation session in {room_name}")
        return existing
    session = ConversationSession(
        room_name,
        settings=settings,
        identity=identity,
        image_path=image_path,
        image_base64=image_base64,
        image_format=image_format,
        tts_fn=tts_fn,
        agent_backend=agent_backend,
        agent_name=agent_name,
        gemini_api_key=gemini_api_key,
    )
    try:
        await session.start()
    except Exception:
        # A failed start must not leak a temp portrait or a half-open room.
        await session.stop()
        raise
    _sessions[room_name] = session
    return session


async def stop_session(room_name: str) -> None:
    """Stop and forget the conversation session for ``room_name``."""
    session = _sessions.pop(room_name, None)
    if session is not None:
        await session.stop()


__all__ = [
    "ConversationSession",
    "dispatch_agent",
    "get_session",
    "start_session",
    "stop_session",
]
