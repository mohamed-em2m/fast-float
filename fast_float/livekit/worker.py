"""LiveKit Avatar Worker & Stream Publisher.

Implements:
1. `AvatarOptions` - video and audio formats.
2. `AvatarRunner` - the core avatar worker loop that publishes synchronized
   avatar_video and avatar_audio tracks using `rtc.AVSynchronizer`.
3. `LiveKitLiveStreamPublisher` - connects an in-flight `LiveStream` from a
   `/generate` request directly into a LiveKit room as an avatar worker.
4. `LiveKitMediaFilePublisher` - streams a finished MP4 file into a LiveKit room.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass
from typing import Any

RTC_IMPORT_ERROR = ""
try:
    from livekit import rtc

    RTC_AVAILABLE = True
except ImportError as exc:  # pragma: no cover - depends on the environment
    rtc = None  # type: ignore[assignment]
    RTC_AVAILABLE = False
    RTC_IMPORT_ERROR = str(exc)

from fast_float.livekit.generator import (
    FloatVideoGenerator,
    LiveStreamVideoGenerator,
    MediaFileVideoGenerator,
)
from fast_float.livekit.settings import LiveKitSettings
from fast_float.livekit.tokens import issue_token
from fast_float.streaming.live import LiveStream

logger = logging.getLogger(__name__)

if not RTC_AVAILABLE:
    logger.warning(f"LiveKit RTC SDK unavailable: {RTC_IMPORT_ERROR}")


def rtc_unavailable_message() -> str:
    """Human-readable reason the RTC SDK could not be imported."""
    base = "LiveKit RTC SDK is not installed; run the setup script to install the 'livekit' SDK."
    return f"{base} ({RTC_IMPORT_ERROR})" if RTC_IMPORT_ERROR else base


# Publisher registries. Double-publishing the same request into the same room
# joins two participants with the *same* worker identity (deriving from
# request_id), which LiveKit resolves by disconnecting one of them: both
# spawn sites (/generate auto-spawn and POST /livekit/publish) therefore go
# through `spawn_*_publish_task`, which refuses a second task for the same key.
_live_publish_jobs: dict[str, asyncio.Task[None]] = {}
_media_publish_jobs: dict[tuple[str, str], asyncio.Task[None]] = {}
# Last publisher error per request, so the browser can report *why* the SFU
# stream never appeared instead of silently waiting for tracks.
_publish_errors: dict[str, str] = {}


def _track_job(registry: dict, key: Any, task: asyncio.Task[None]) -> asyncio.Task[None]:
    registry[key] = task
    task.add_done_callback(lambda _t: registry.pop(key, None))
    return task


def publish_error(request_id: str) -> str | None:
    """The last error raised by a LiveKit publisher for ``request_id``."""
    return _publish_errors.get(request_id)


def live_publish_active(request_id: str) -> bool:
    """True when a LiveKit publisher is already running for this request."""
    task = _live_publish_jobs.get(request_id)
    return task is not None and not task.done()


def spawn_live_publish_task(
    stream: LiveStream,
    room_name: str,
    settings: LiveKitSettings | None = None,
) -> asyncio.Task[None] | None:
    """Spawn the background publisher for ``stream`` unless one is already active.

    Returns the running task, or ``None`` when a publisher for this
    ``request_id`` already exists (the caller should report "already publishing").
    """
    if live_publish_active(stream.request_id):
        logger.info(f"[livekit-pub] {stream.request_id} is already publishing; ignoring duplicate request")
        return None
    task = _track_job(
        _live_publish_jobs,
        stream.request_id,
        asyncio.get_running_loop().create_task(
            publish_live_stream_task(stream, room_name=room_name, settings=settings)
        ),
    )
    return task


def spawn_media_publish_task(
    file_path: str,
    room_name: str,
    settings: LiveKitSettings | None = None,
    loop: bool = False,
) -> asyncio.Task[None] | None:
    """Spawn the background publisher for a finished file, deduped per (file, room)."""
    key = (file_path, room_name)
    existing = _media_publish_jobs.get(key)
    if existing is not None and not existing.done():
        logger.info(f"[livekit-pub] {file_path} is already publishing into {room_name}; ignoring duplicate request")
        return None
    task = _track_job(
        _media_publish_jobs,
        key,
        asyncio.get_running_loop().create_task(
            publish_media_file_task(file_path, room_name=room_name, settings=settings, loop=loop)
        ),
    )
    return task


DEFAULT_AVATAR_IDENTITY = "float-avatar"
DEFAULT_AVATAR_NAME = "FLOAT Avatar"


@dataclass(frozen=True)
class AvatarOptions:
    """Options for video and audio format published by the avatar."""

    video_width: int = 512
    video_height: int = 512
    video_fps: float = 25.0
    audio_sample_rate: int = 16000
    audio_channels: int = 1
    queue_size_ms: int = 100


class AvatarRunner:
    """Worker that publishes synchronized avatar video + audio into a LiveKit room."""

    # How long ``stop()`` lets already-generated frames play out before it
    # tears the tracks down. Bounded so a wedged consumer cannot block shutdown.
    DRAIN_TIMEOUT_S = 2.0

    def __init__(
        self,
        room: Any,
        video_gen: FloatVideoGenerator | LiveStreamVideoGenerator | MediaFileVideoGenerator | Any,
        *,
        options: AvatarOptions = AvatarOptions(),
        agent_identity: str | None = None,
    ) -> None:
        self.room = room
        self.video_gen = video_gen
        self.options = options
        self.agent_identity = agent_identity

        self._audio_source: Any = None
        self._video_source: Any = None
        self._av_sync: Any = None
        self._video_pub: Any = None
        self._audio_pub: Any = None

        self._running = False
        self._forward_task: asyncio.Task[Any] | None = None
        self._stopped = False

    async def start(self) -> None:
        """Initialize sources, publish tracks, and start media forwarding."""
        if not RTC_AVAILABLE:
            raise RuntimeError("LiveKit RTC SDK is not available in this environment")

        self._running = True

        # 1. Sources & AV synchronizer
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

        # 2. Local tracks
        video_track = rtc.LocalVideoTrack.create_video_track("avatar_video", self._video_source)
        audio_track = rtc.LocalAudioTrack.create_audio_track("avatar_audio", self._audio_source)

        # 3. Publish to room
        # LiveKit's native default caps a published video track at 20 fps, so a
        # 25 fps avatar silently lost every fifth frame at the encoder (the
        # "stream skips/cuts mid-send" symptom). Ask for the source rate and a
        # bitrate that fits the frame size; ~0.15 bit/pixel/s keeps a moving
        # face sharp, and the floor/cap keep tiny and huge frames sane.
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
        self._audio_pub = await self.room.local_participant.publish_track(
            audio_track,
            rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE),
        )

        # 4. Attribute: lk.publish_on_behalf (links avatar worker to agent participant)
        if self.agent_identity:
            with contextlib.suppress(Exception):
                await self.room.local_participant.set_attributes({"lk.publish_on_behalf": self.agent_identity})

        # 5. Start frame forwarding loop
        self._forward_task = asyncio.create_task(self._forward_video_loop())

    async def push_audio_frame(self, frame: Any) -> None:
        """Push an audio frame through the AV synchronizer."""
        if self._av_sync and self._running:
            await self._av_sync.push(frame)

    async def _forward_video_loop(self) -> None:
        """Pull generated video & audio frames and forward them through AV synchronizer."""
        first_frame = True
        try:
            async for item in self.video_gen:
                # Deliberately do NOT abort on ``stop()`` here: the generator
                # queues the tail of the stream before ending, and dropping it
                # cut the last words off. The iterator ends by itself; stop()
                # bounds the drain with a timeout.
                if first_frame:
                    first_frame = False
                    if self.agent_identity:
                        try:
                            await self.room.local_participant.perform_rpc(
                                destination_identity=self.agent_identity,
                                method="lk.playback_started",
                                payload="{}",
                                response_timeout=2.0,
                            )
                        except Exception as exc:
                            logger.debug(f"[avatar-runner] playback_started rpc skipped: {exc}")

                if isinstance(item, tuple):
                    v_frame, a_frame = item
                    if isinstance(v_frame, rtc.VideoFrame):
                        await self._av_sync.push(v_frame)
                    if a_frame is not None and isinstance(a_frame, rtc.AudioFrame):
                        await self._av_sync.push(a_frame)
                elif isinstance(item, rtc.VideoFrame):
                    await self._av_sync.push(item)

            # Wait for playout to complete
            if self._av_sync:
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(self._av_sync.wait_for_playout(), timeout=10.0)

            if self.agent_identity:
                try:
                    await self.room.local_participant.perform_rpc(
                        destination_identity=self.agent_identity,
                        method="lk.playback_finished",
                        payload="{}",
                        response_timeout=2.0,
                    )
                except Exception as exc:
                    logger.debug(f"[avatar-runner] playback_finished rpc skipped: {exc}")

        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error(f"[avatar-runner] error in forward loop: {exc}")
        finally:
            self._running = False

    async def stop(self) -> None:
        """Stop forwarding, play out queued frames, unpublish tracks and clean up."""
        if self._stopped:
            return
        self._stopped = True
        self._running = False

        # End the generator first: it queues the frames it already rendered and
        # then signals end-of-stream, so the forward loop can play the tail out
        # instead of cutting mid-sentence.
        if hasattr(self.video_gen, "stop"):
            self.video_gen.stop()

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


class LiveKitLiveStreamPublisher:
    """Connects to a LiveKit room and publishes an in-flight LiveStream."""

    def __init__(
        self,
        stream: LiveStream,
        *,
        room_name: str,
        settings: LiveKitSettings | None = None,
        identity: str | None = None,
        name: str | None = None,
    ) -> None:
        self.stream = stream
        self.room_name = room_name
        self.settings = settings or LiveKitSettings.from_env()
        self.identity = identity or f"worker-{stream.request_id[:8]}"
        self.name = name or DEFAULT_AVATAR_NAME

        self.room: Any = None
        self.runner: AvatarRunner | None = None
        self._stopped = False

    async def run(self) -> None:
        """Join room, publish synchronized video + audio until stream completes, then disconnect."""
        if not RTC_AVAILABLE:
            raise RuntimeError("LiveKit RTC SDK is not available in this environment")

        if not self.settings.configured:
            raise RuntimeError("LiveKit settings not configured")

        token_result = issue_token(
            url=self.settings.url,
            api_key=self.settings.api_key,
            api_secret=self.settings.api_secret,
            identity=self.identity,
            name=self.name,
            room=self.room_name,
            can_publish=True,
        )

        self.room = rtc.Room()

        logger.info(f"[livekit-pub] connecting to {self.settings.url} room {self.room_name}")
        await self.room.connect(self.settings.url, token_result.token)

        try:
            meta = await self.stream.wait_meta(timeout=15.0)
            if meta is None:
                logger.warning(
                    f"[livekit-pub] {self.room_name}: no stream metadata after 15s; "
                    "publishing with default 25fps (audio may be missing)"
                )
            fps = float(meta.fps) if meta and meta.fps else 25.0
            # Match the source to the frames the pipeline actually renders: a
            # 512x512 VideoSource fed different-sized frames corrupts or scales
            # the preview, and the browser then sees a stuttering/misaligned
            # picture instead of the avatar.
            width = int(meta.width) if meta and meta.width else 512
            height = int(meta.height) if meta and meta.height else 512
            # H.264 needs even dimensions; round down instead of letting the
            # encoder reject the track.
            width -= width % 2
            height -= height % 2
            options = AvatarOptions(
                video_width=max(2, width),
                video_height=max(2, height),
                video_fps=fps,
                audio_sample_rate=16000,
                audio_channels=1,
            )

            # LiveStreamVideoGenerator handles frame delivery and lockstep audio slicing
            video_gen = LiveStreamVideoGenerator(self.stream)
            self.runner = AvatarRunner(self.room, video_gen, options=options)
            await self.runner.start()

            if self.runner._forward_task:
                await self.runner._forward_task

        finally:
            await self.close()

    async def close(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        if self.runner:
            await self.runner.stop()
            self.runner = None
        if self.room:
            with contextlib.suppress(Exception):
                await self.room.disconnect()
            self.room = None
        logger.info(f"[livekit-pub] disconnected from room {self.room_name}")


class LiveKitMediaFilePublisher:
    """Publishes a completed MP4 video file to a LiveKit room."""

    def __init__(
        self,
        file_path: str,
        *,
        room_name: str,
        settings: LiveKitSettings | None = None,
        identity: str | None = None,
        name: str | None = None,
        loop: bool = False,
    ) -> None:
        self.file_path = file_path
        self.room_name = room_name
        self.settings = settings or LiveKitSettings.from_env()
        self.identity = identity or "worker-media"
        self.name = name or DEFAULT_AVATAR_NAME
        self.loop = loop

        self.room: Any = None
        self.runner: AvatarRunner | None = None
        self._stopped = False

    async def run(self) -> None:
        if not RTC_AVAILABLE:
            raise RuntimeError("LiveKit RTC SDK is not available in this environment")
        if not self.settings.configured:
            raise RuntimeError("LiveKit settings not configured")

        token_result = issue_token(
            url=self.settings.url,
            api_key=self.settings.api_key,
            api_secret=self.settings.api_secret,
            identity=self.identity,
            name=self.name,
            room=self.room_name,
            can_publish=True,
        )

        self.room = rtc.Room()
        await self.room.connect(self.settings.url, token_result.token)

        try:
            video_gen = MediaFileVideoGenerator(self.file_path, loop=self.loop)
            # Publish at the file's own size/rate: the default 25 fps would
            # stretch a 30 fps render and drift against its audio.
            width, height, fps = video_gen.probe()
            self.runner = AvatarRunner(
                self.room,
                video_gen,
                options=AvatarOptions(video_width=width, video_height=height, video_fps=fps),
            )
            await self.runner.start()

            if self.runner._forward_task:
                await self.runner._forward_task
        finally:
            await self.close()

    async def close(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        if self.runner:
            await self.runner.stop()
            self.runner = None
        if self.room:
            with contextlib.suppress(Exception):
                await self.room.disconnect()
            self.room = None


async def publish_live_stream_task(
    stream: LiveStream,
    room_name: str,
    settings: LiveKitSettings | None = None,
) -> None:
    """Convenience background task to publish an active LiveStream to LiveKit."""
    publisher = LiveKitLiveStreamPublisher(stream, room_name=room_name, settings=settings)
    try:
        await publisher.run()
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        _publish_errors[stream.request_id] = str(exc)
        logger.warning(f"[livekit-pub] live publishing failed for {room_name}: {exc}")
    else:
        _publish_errors.pop(stream.request_id, None)
    finally:
        await publisher.close()


async def publish_media_file_task(
    file_path: str,
    room_name: str,
    settings: LiveKitSettings | None = None,
    loop: bool = False,
) -> None:
    """Convenience background task to publish an MP4 file to LiveKit."""
    publisher = LiveKitMediaFilePublisher(file_path, room_name=room_name, settings=settings, loop=loop)
    try:
        await publisher.run()
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        _publish_errors[room_name] = str(exc)
        logger.warning(f"[livekit-pub] file publishing failed for {room_name}: {exc}")
    else:
        _publish_errors.pop(room_name, None)
    finally:
        await publisher.close()


__all__ = [
    "DEFAULT_AVATAR_IDENTITY",
    "DEFAULT_AVATAR_NAME",
    "AvatarOptions",
    "AvatarRunner",
    "LiveKitLiveStreamPublisher",
    "LiveKitMediaFilePublisher",
    "live_publish_active",
    "publish_error",
    "publish_live_stream_task",
    "publish_media_file_task",
    "spawn_live_publish_task",
    "spawn_media_publish_task",
]
