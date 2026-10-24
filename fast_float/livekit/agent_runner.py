"""Avatar runner for the Gemini Live worker.

The built-in :class:`fast_float.livekit.worker.AvatarRunner` is push-based: an
in-process agent calls ``generator.push_audio()``. This runner is pull-based:
it consumes :class:`DataStreamAudioReceiver` segments streamed by the
``livekit-agents`` worker in :mod:`fast_float.livekit.gemini`, renders them
with FLOAT, and publishes synchronized ``avatar_video`` + ``avatar_audio``
tracks.

It mirrors ``livekit.agents.voice.avatar.AvatarRunner`` but uses only the RTC
SDK, so the avatar side does not depend on the worker's agent stack.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any

try:
    from livekit import rtc

    RTC_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the environment
    rtc = None  # type: ignore[assignment]
    RTC_AVAILABLE = False

from fast_float.livekit.generator import FloatVideoGenerator
from fast_float.livekit.receiver import SEGMENT_END, DataStreamAudioReceiver
from fast_float.livekit.worker import AvatarOptions

logger = logging.getLogger(__name__)

VOICE_TRACK_NAME = "avatar_audio"
VIDEO_TRACK_NAME = "avatar_video"

# Upper bound on how long a segment may take to render after the agent's
# stream ends. FLOAT runs ~6x real time; a 2 s window plus drain is well under
# this, and a wedged inference must not block playback_finished forever.
SEGMENT_RENDER_TIMEOUT_S = 60.0
SEGMENT_PLAYOUT_TIMEOUT_S = 30.0


class AgentAvatarRunner:
    """Publish FLOAT video + audio rendered from a remote agent's TTS stream."""

    DRAIN_TIMEOUT_S = 2.0

    def __init__(
        self,
        room: Any,
        *,
        video_gen: FloatVideoGenerator,
        receiver: DataStreamAudioReceiver,
        options: AvatarOptions = AvatarOptions(),
    ) -> None:
        self.room = room
        self.video_gen = video_gen
        self.receiver = receiver
        self.options = options

        self._audio_source: Any = None
        self._video_source: Any = None
        self._av_sync: Any = None
        self._video_pub: Any = None
        self._audio_pub: Any = None

        self._running = False
        self._stopped = False
        self._forward_task: asyncio.Task[Any] | None = None
        self._read_task: asyncio.Task[Any] | None = None
        self._segment_task: asyncio.Task[Any] | None = None
        self._background: set[asyncio.Task[Any]] = set()

        fps = float(options.video_fps or 25.0)
        self._frame_interval = 1.0 / (fps if fps > 0 else 25.0)
        self._next_frame_at: float | None = None

        # Per-segment bookkeeping. ``_epoch`` invalidates a segment whose
        # completion notification races a barge-in.
        self._epoch = 0
        self._segment_queue: asyncio.Queue[int] = asyncio.Queue()
        self._segment_started = False
        self._segment_audio_s = 0.0
        self._playback_position = 0.0
        self._last_agent_identity: str | None = None

    # -------------------------------------------------------------- lifecycle

    async def start(self) -> None:
        if not RTC_AVAILABLE:
            raise RuntimeError("LiveKit RTC SDK is not available in this environment")
        if not self.room.isconnected():
            raise RuntimeError("AgentAvatarRunner requires a connected room")

        self._running = True

        # Wait for the remote agent before publishing: its identity becomes the
        # lk.publish_on_behalf value the browser uses to select the avatar.
        await self.receiver.start()
        self._last_agent_identity = self.receiver.agent_identity

        self._audio_source = rtc.AudioSource(
            sample_rate=self.options.audio_sample_rate,
            num_channels=self.options.audio_channels,
            queue_size_ms=self.options.queue_size_ms,
        )
        self._video_source = rtc.VideoSource(
            width=self.options.video_width,
            height=self.options.video_height,
        )
        self._av_sync = rtc.AVSynchronizer(
            audio_source=self._audio_source,
            video_source=self._video_source,
            video_fps=self.options.video_fps,
            video_queue_size_ms=self.options.queue_size_ms,
        )

        video_track = rtc.LocalVideoTrack.create_video_track(VIDEO_TRACK_NAME, self._video_source)
        audio_track = rtc.LocalAudioTrack.create_audio_track(VOICE_TRACK_NAME, self._audio_source)

        width = self.options.video_width
        height = self.options.video_height
        fps = float(self.options.video_fps)
        video_encoding = rtc.VideoEncoding(
            max_framerate=fps,
            max_bitrate=max(400_000, min(6_000_000, int(width * height * fps * 0.15))),
        )
        self._video_pub = await self.room.local_participant.publish_track(
            video_track,
            rtc.TrackPublishOptions(
                source=rtc.TrackSource.SOURCE_CAMERA,
                video_encoding=video_encoding,
            ),
        )
        # Deliberately NOT SOURCE_MICROPHONE: livekit-agents' RoomIO subscribes
        # to microphone tracks from the linked human participant, and the
        # avatar's synthesized voice must never be fed back into Gemini.
        self._audio_pub = await self.room.local_participant.publish_track(
            audio_track,
            rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_SCREENSHARE_AUDIO),
        )

        if self.receiver.agent_identity:
            with contextlib.suppress(Exception):
                await self.room.local_participant.set_attributes({"lk.publish_on_behalf": self.receiver.agent_identity})

        self.receiver.on_clear_buffer(self._on_clear_buffer)

        self._read_task = asyncio.create_task(self._read_audio_loop())
        self._forward_task = asyncio.create_task(self._forward_video_loop())
        self._segment_task = asyncio.create_task(self._segment_loop())
        logger.info(
            f"[agent-avatar] publishing {VIDEO_TRACK_NAME}+{VOICE_TRACK_NAME} "
            f"on behalf of {self.receiver.agent_identity}"
        )

    async def _pace(self) -> None:
        loop = asyncio.get_running_loop()
        now = loop.time()
        if self._next_frame_at is None:
            self._next_frame_at = now
        delay = self._next_frame_at - now
        if delay > 0:
            await asyncio.sleep(delay)
        self._next_frame_at = max(self._next_frame_at + self._frame_interval, loop.time())

    def _spawn(self, coro: Any) -> None:
        task = asyncio.create_task(coro)
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    # ---------------------------------------------------------------- reading

    async def _read_audio_loop(self) -> None:
        try:
            async for item in self.receiver:
                if not self._running:
                    break
                if item is SEGMENT_END:
                    # The agent flushed this TTS segment: render the tail, then
                    # let the segment watcher report playback_finished.
                    self.video_gen.flush()
                    await self._segment_queue.put(self._epoch)
                else:
                    self.video_gen.push_audio(item)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(f"[agent-avatar] audio receive loop stopped: {exc}")

    # ------------------------------------------------------------- forwarding

    async def _forward_video_loop(self) -> None:
        try:
            async for item in self.video_gen:
                await self._pace()
                audio_frame = None
                if isinstance(item, tuple):
                    video_frame, audio_frame = item
                else:
                    video_frame = item

                if self._av_sync is not None:
                    if isinstance(video_frame, rtc.VideoFrame):
                        await self._av_sync.push(video_frame)
                    if isinstance(audio_frame, rtc.AudioFrame):
                        await self._av_sync.push(audio_frame)

                if not self._segment_started:
                    self._segment_started = True
                    self._last_agent_identity = self.receiver.agent_identity
                    if self._last_agent_identity:
                        self._spawn(self.receiver.notify_playback_started())

                if isinstance(audio_frame, rtc.AudioFrame):
                    self._segment_audio_s += audio_frame.duration
                    self._playback_position += audio_frame.duration
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(f"[agent-avatar] video forward loop stopped: {exc}")

    async def _segment_loop(self) -> None:
        try:
            while True:
                epoch = await self._segment_queue.get()
                if epoch != self._epoch:
                    continue
                rendered = await self.video_gen.wait_until_idle(SEGMENT_RENDER_TIMEOUT_S)
                if not rendered:
                    logger.warning(
                        "[agent-avatar] segment render did not finish within %ss; reporting playback anyway",
                        SEGMENT_RENDER_TIMEOUT_S,
                    )
                if epoch != self._epoch:
                    continue
                if self._av_sync is not None:
                    try:
                        await asyncio.wait_for(
                            self._av_sync.wait_for_playout(),
                            timeout=SEGMENT_PLAYOUT_TIMEOUT_S,
                        )
                    except asyncio.TimeoutError:
                        logger.warning("[agent-avatar] playout drain timed out")
                if epoch != self._epoch:
                    continue
                position = self._segment_audio_s
                self._segment_started = False
                self._segment_audio_s = 0.0
                self._playback_position = 0.0
                if self._last_agent_identity:
                    await self.receiver.notify_playback_finished(playback_position=position, interrupted=False)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(f"[agent-avatar] segment watcher stopped: {exc}")

    # --------------------------------------------------------------- barge-in

    def _on_clear_buffer(self) -> None:
        """Agent interrupted the avatar (user barge-in)."""
        self.interrupt()

    def interrupt(self) -> None:
        """Drop buffered/queued audio and tell the agent playback was cut short."""
        self._epoch += 1
        position = self._playback_position
        self._segment_started = False
        self._segment_audio_s = 0.0
        self._playback_position = 0.0
        with contextlib.suppress(Exception):
            self.video_gen.clear_buffer()
        if self._av_sync is not None:
            with contextlib.suppress(Exception):
                self._av_sync.clear_queue()
        logger.info("[agent-avatar] playback interrupted; cleared buffered audio")
        if self._last_agent_identity:
            self._spawn(self.receiver.notify_playback_finished(playback_position=position, interrupted=True))

    # ------------------------------------------------------------------ stop

    async def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        self._running = False

        if hasattr(self.video_gen, "stop"):
            self.video_gen.stop()

        for task in (self._segment_task, self._read_task):
            if task is not None and not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        self._segment_task = None
        self._read_task = None

        task = self._forward_task
        self._forward_task = None
        if task is not None and not task.done():
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=self.DRAIN_TIMEOUT_S)
            except asyncio.TimeoutError:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            except asyncio.CancelledError:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
                raise

        for bg in list(self._background):
            bg.cancel()
        if self._background:
            await asyncio.gather(*self._background, return_exceptions=True)
        self._background.clear()

        with contextlib.suppress(Exception):
            await self.receiver.aclose()

        if self._av_sync:
            with contextlib.suppress(Exception):
                await self._av_sync.aclose()
            self._av_sync = None

        if self.room and self.room.local_participant:
            if self._video_pub:
                with contextlib.suppress(Exception):
                    await self.room.local_participant.unpublish_track(self._video_pub.sid)
            if self._audio_pub:
                with contextlib.suppress(Exception):
                    await self.room.local_participant.unpublish_track(self._audio_pub.sid)


__all__ = [
    "SEGMENT_PLAYOUT_TIMEOUT_S",
    "SEGMENT_RENDER_TIMEOUT_S",
    "VIDEO_TRACK_NAME",
    "VOICE_TRACK_NAME",
    "AgentAvatarRunner",
]
