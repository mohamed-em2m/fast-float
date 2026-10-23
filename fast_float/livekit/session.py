"""Conversation session: wires the agent, the FLOAT renderer and the avatar worker.

Brings the Phase-3 pieces together in one process so a conversation actually
runs end to end:

    ConversationalAgent (float-agent)  -- talk/speak + playback RPCs
    FloatVideoGenerator                -- audio in -> lip-synced VideoFrames
    AvatarRunner (avatar-<room>)       -- publishes the synced A/V tracks

The browser joins the same room with a viewer token (POST /livekit/token) and
attaches the tracks published by the avatar worker. One session per room, kept
in :data:`_sessions`; manage through the ``/livekit/conversation/*`` endpoints.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
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
from fast_float.livekit.generator import FloatVideoGenerator
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


class ConversationSession:
    """One conversational avatar session in a LiveKit room."""

    def __init__(
        self,
        room_name: str,
        *,
        settings: LiveKitSettings | None = None,
        identity: str | None = None,
        image_path: str | None = None,
        tts_fn: Callable[[str], bytes] | None = None,
    ) -> None:
        self.room_name = room_name
        self.settings = settings or LiveKitSettings.from_env()
        self.identity = identity or DEFAULT_AGENT_IDENTITY
        self.avatar_identity = f"avatar-{room_name[:24]}"
        self.image_path = image_path
        self.tts_fn = tts_fn

        self.agent: ConversationalAgent | None = None
        self.generator: FloatVideoGenerator | None = None
        self.worker_room: Any = None
        self.runner: AvatarRunner | None = None
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

        if self.image_path is None:
            self.image_path = ensure_default_image()

        self.generator = FloatVideoGenerator(
            agent=engine,
            image_path=self.image_path,
            semaphore=_render_lock,
        )

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

    async def speak(self, text: str) -> None:
        """Turn a user utterance reply into audio and push it to the renderer."""
        if self.agent is None:
            raise RuntimeError("Session not started")
        if self.tts_fn is None:
            raise RuntimeError("No TTS engine attached to this session")
        await self.agent.send_speech(text)

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
        if self.worker_room is not None:
            with contextlib.suppress(Exception):
                await self.worker_room.disconnect()
            self.worker_room = None
        if self.agent is not None:
            with contextlib.suppress(Exception):
                await self.agent.disconnect()
            self.agent = None
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
    tts_fn: Callable[[str], bytes] | None = None,
) -> ConversationSession:
    """Start (or reuse) the conversation session for ``room_name``."""
    existing = get_session(room_name)
    if existing is not None:
        logger.info(f"[session] reusing existing conversation session in {room_name}")
        return existing
    session = ConversationSession(room_name, settings=settings, identity=identity, image_path=image_path, tts_fn=tts_fn)
    await session.start()
    _sessions[room_name] = session
    return session


async def stop_session(room_name: str) -> None:
    """Stop and forget the conversation session for ``room_name``."""
    session = _sessions.pop(room_name, None)
    if session is not None:
        await session.stop()


__all__ = [
    "ConversationSession",
    "get_session",
    "start_session",
    "stop_session",
]
