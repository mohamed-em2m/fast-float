"""LiveKit Conversational Agent (Phase 3).

The conversational agent owns the STT -> LLM -> TTS pipeline.
Its audio output is streamed directly to the FLOAT avatar worker (either over a
LiveKit data stream or via direct generator push), rather than being published
to the room. The avatar worker then renders lip-synced video and publishes the
synchronized result.

The agent coordinates playback state with the worker through LiveKit RPCs:
- `lk.playback_started`
- `lk.playback_finished`
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

from fast_float.livekit.generator import FloatVideoGenerator
from fast_float.livekit.settings import LiveKitSettings
from fast_float.livekit.tokens import issue_token

logger = logging.getLogger(__name__)

DEFAULT_AGENT_IDENTITY = "float-agent"
DEFAULT_AVATAR_IDENTITY = "float-avatar"


class ConversationalAgent:
    """LiveKit Voice Agent managing conversation turns and streaming audio to avatar."""

    def __init__(
        self,
        *,
        room_name: str,
        settings: LiveKitSettings | None = None,
        identity: str = DEFAULT_AGENT_IDENTITY,
        avatar_identity: str = DEFAULT_AVATAR_IDENTITY,
        video_gen: FloatVideoGenerator | None = None,
        tts_fn: Callable[[str], bytes] | None = None,
    ) -> None:
        self.room_name = room_name
        self.settings = settings or LiveKitSettings.from_env()
        self.identity = identity
        self.avatar_identity = avatar_identity
        self.video_gen = video_gen
        self.tts_fn = tts_fn

        self.room: Any = None
        self._stopped = False
        self.is_speaking = False
        self._playback_started_event = asyncio.Event()
        self._playback_finished_event = asyncio.Event()

    async def connect(self) -> None:
        """Connect to LiveKit room as agent participant."""
        if not RTC_AVAILABLE:
            raise RuntimeError("LiveKit RTC SDK is not available")

        token_result = issue_token(
            url=self.settings.url,
            api_key=self.settings.api_key,
            api_secret=self.settings.api_secret,
            identity=self.identity,
            name="FLOAT Agent",
            room=self.room_name,
            can_publish=True,
        )

        self.room = rtc.Room()

        await self.room.connect(self.settings.url, token_result.token)
        # `local_participant` only exists once the room is connected, so RPC
        # handlers must be registered after connect (before the avatar worker
        # can send its first playback callback).
        self._register_rpc_handlers()
        logger.info(f"[agent] connected to {self.room_name} as {self.identity}")

    def _register_rpc_handlers(self) -> None:
        """Register lk.playback_started and lk.playback_finished RPC callbacks.

        Must be called after ``self.room`` has connected: the RTC SDK does not
        expose ``local_participant`` until then.
        """
        if self.room is None:
            raise RuntimeError("_register_rpc_handlers called before room was created")
        if not hasattr(self.room.local_participant, "register_rpc_method"):
            return

        def on_playback_started(data: Any) -> str:
            logger.info("[agent] RPC lk.playback_started received from avatar worker")
            self.is_speaking = True
            self._playback_started_event.set()
            return "ok"

        def on_playback_finished(data: Any) -> str:
            logger.info("[agent] RPC lk.playback_finished received from avatar worker")
            self.is_speaking = False
            self._playback_finished_event.set()
            return "ok"

        self.room.local_participant.register_rpc_method("lk.playback_started", on_playback_started)
        self.room.local_participant.register_rpc_method("lk.playback_finished", on_playback_finished)

    async def send_speech(self, text: str, pcm_audio: bytes | None = None) -> None:
        """Generate audio (via TTS or direct PCM) and stream it to the avatar worker."""
        if pcm_audio is None and self.tts_fn:
            pcm_audio = self.tts_fn(text)

        if not pcm_audio:
            logger.warning("[agent] no audio data to send for speech")
            return

        self._playback_started_event.clear()
        self._playback_finished_event.clear()

        # If we have a direct video generator handle, push directly
        if self.video_gen:
            self.video_gen.push_audio(pcm_audio)
            self.video_gen.flush()

        # Also send over LiveKit DataStream if connected
        if self.room and self.room.local_participant:
            try:
                # Send data packet with audio
                await self.room.local_participant.publish_data(
                    pcm_audio,
                    destination_identities=[self.avatar_identity],
                    reliable=True,
                )
            except Exception as exc:
                logger.debug(f"[agent] publish_data fallback: {exc}")

    async def disconnect(self) -> None:
        """Disconnect agent from room."""
        if self._stopped:
            return
        self._stopped = True
        if self.room:
            with contextlib.suppress(Exception):
                await self.room.disconnect()
            self.room = None
        logger.info(f"[agent] disconnected from {self.room_name}")


__all__ = [
    "DEFAULT_AGENT_IDENTITY",
    "DEFAULT_AVATAR_IDENTITY",
    "ConversationalAgent",
]
