"""Receive agent TTS audio over a LiveKit DataStream.

The conversational agent (``livekit-agents`` + Gemini Live) does not publish
its voice to the room: it streams the PCM to the FLOAT avatar worker over the
``lk.audio_stream`` byte-stream topic, using
``livekit.agents.voice.avatar.DataStreamAudioOutput``. This module is the
receiver side of that protocol, implemented with the RTC SDK only: the avatar
process does not need to import the full agents/STT/LLM stack to consume the
worker's PCM.

Wire protocol (mirrors livekit-agents 1.x):

* one byte stream per TTS segment on ``lk.audio_stream``, with attributes
  ``sample_rate`` and ``num_channels`` and raw little-endian int16 PCM;
* ``lk.clear_buffer`` RPC from the agent means "barge-in": drop the current
  segment and queued audio;
* the avatar calls ``lk.playback_started`` / ``lk.playback_finished`` on the
  agent once the rendered audio starts / ends.

The module emits :class:`SegmentEnd` after each stream trailer so the runner
knows when to flush FLOAT's tail renderer.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
from collections import deque
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    from livekit import rtc

    RTC_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the environment
    rtc = None  # type: ignore[assignment]
    RTC_AVAILABLE = False

logger = logging.getLogger(__name__)

AUDIO_STREAM_TOPIC = "lk.audio_stream"
RPC_CLEAR_BUFFER = "lk.clear_buffer"
RPC_PLAYBACK_STARTED = "lk.playback_started"
RPC_PLAYBACK_FINISHED = "lk.playback_finished"

DEFAULT_FRAME_SIZE_MS = 100
DEFAULT_IDLE_TIMEOUT_S = 10.0

# A "MissedChunk" stream error (a chunk lost/overwritten in transport, e.g.
# under packet loss) does not necessarily end the segment: the native reader
# can keep delivering the chunks after the gap. Keep pulling for a few
# attempts before declaring the segment dead, so one lost 16 KB chunk skips
# a fraction of a second instead of truncating the whole reply.
READER_ERROR_RETRIES = 3
READER_ERROR_RETRY_DELAY_S = 0.05


@dataclass(frozen=True)
class SegmentEnd:
    """A TTS segment's byte stream ended (the agent flushed its audio)."""


SEGMENT_END = SegmentEnd()


class _PcmFramer:
    """Split an arbitrarily chunked int16 PCM stream into fixed-size frames."""

    def __init__(self, sample_rate: int, num_channels: int, frame_size_ms: int) -> None:
        self.sample_rate = max(1, int(sample_rate))
        self.num_channels = max(1, int(num_channels))
        self.samples_per_channel = max(1, math.ceil(self.sample_rate * max(1, frame_size_ms) / 1000.0))
        self._bytes_per_frame = self.samples_per_channel * self.num_channels * 2
        self._buffer = bytearray()

    def clear(self) -> None:
        self._buffer.clear()

    def push(self, data: bytes) -> list[Any]:
        if not RTC_AVAILABLE:
            return []
        self._buffer.extend(data)
        frames: list[Any] = []
        while len(self._buffer) >= self._bytes_per_frame:
            chunk = bytes(self._buffer[: self._bytes_per_frame])
            del self._buffer[: self._bytes_per_frame]
            frames.append(
                rtc.AudioFrame(
                    chunk,
                    self.sample_rate,
                    self.num_channels,
                    self.samples_per_channel,
                )
            )
        return frames

    def flush(self) -> list[Any]:
        if not RTC_AVAILABLE or not self._buffer:
            return []
        # Whole samples only; a dangling byte (or a partial multi-channel
        # sample) would shift every later sample.
        sample_bytes = self.num_channels * 2
        samples = len(self._buffer) // sample_bytes
        if samples <= 0:
            self._buffer.clear()
            return []
        chunk = bytes(self._buffer[: samples * sample_bytes])
        self._buffer.clear()
        return [rtc.AudioFrame(chunk, self.sample_rate, self.num_channels, samples)]


class DataStreamAudioReceiver:
    """Async iterator of audio frames streamed by a LiveKit Agents participant.

    ``sender_identity`` pins the sender. When omitted, the first participant of
    kind ``AGENT`` is used (the Gemini worker joins with that kind).
    """

    def __init__(
        self,
        room: Any,
        *,
        sender_identity: str | None = None,
        sample_rate: int | None = None,
        num_channels: int = 1,
        frame_size_ms: int = DEFAULT_FRAME_SIZE_MS,
        idle_timeout_s: float = DEFAULT_IDLE_TIMEOUT_S,
        agent_wait_timeout_s: float = 30.0,
        rpc_max_retries: int = 3,
    ) -> None:
        if not RTC_AVAILABLE:
            raise RuntimeError("LiveKit RTC SDK is not available in this environment")
        self._room = room
        self._sender_identity = sender_identity
        # FLOAT resamples to 16 kHz internally; keep the source rate unless the
        # caller explicitly asks for a fixed one.
        self._sample_rate = sample_rate
        self._num_channels = num_channels
        self._frame_size_ms = frame_size_ms
        self._idle_timeout_s = idle_timeout_s
        self._agent_wait_timeout_s = agent_wait_timeout_s
        self._rpc_max_retries = max(1, rpc_max_retries)

        self.agent_identity = sender_identity
        self._queue: asyncio.Queue[Any] = asyncio.Queue()
        self._readers: deque[Any] = deque()
        self._reader_event = asyncio.Event()
        self._superseded = asyncio.Event()
        self._current_cleared = False
        self._clear_handlers: list[Callable[[], None]] = []
        self._main_task: asyncio.Task[None] | None = None
        self._closing = False
        self._started = False
        self._agent_ready = asyncio.Event()

    # ---------------------------------------------------------------- lifecycle

    async def start(self) -> None:
        """Wait for the agent, register the stream/RPC handlers and start reading."""
        if self._started:
            return
        self.agent_identity = await self._wait_for_agent()
        self._register_handlers()
        self._main_task = asyncio.create_task(self._read_loop())
        self._started = True
        logger.info(f"[receiver] listening for agent audio from {self.agent_identity}")

    def on_clear_buffer(self, callback: Callable[[], None]) -> None:
        """Register a barge-in callback (called on the RTC event-loop thread)."""
        self._clear_handlers.append(callback)

    async def aclose(self) -> None:
        if self._closing:
            return
        self._closing = True
        self._reader_event.set()
        if self._main_task is not None:
            self._main_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._main_task
            self._main_task = None
        if self._started:
            with contextlib.suppress(Exception):
                self._room.unregister_byte_stream_handler(AUDIO_STREAM_TOPIC)
            with contextlib.suppress(Exception):
                self._room.off("participant_connected", self._on_participant_connected)
        self._started = False

    # -------------------------------------------------------------- discovery

    def _matches_agent(self, participant: Any) -> bool:
        if self._sender_identity:
            return participant.identity == self._sender_identity
        return participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_AGENT

    def _on_participant_connected(self, participant: Any) -> None:
        if self.agent_identity or not self._matches_agent(participant):
            return
        self.agent_identity = participant.identity
        self._agent_ready.set()

    async def _wait_for_agent(self) -> str:
        # The agent may already be in the room when the avatar connects.
        for participant in self._room.remote_participants.values():
            if self._matches_agent(participant):
                return participant.identity

        self._agent_ready = asyncio.Event()
        self._room.on("participant_connected", self._on_participant_connected)
        try:
            await asyncio.wait_for(self._agent_ready.wait(), timeout=self._agent_wait_timeout_s)
        except asyncio.TimeoutError as exc:
            room = str(getattr(self._room, "name", "") or "")
            where = f" in room {room!r}" if room else ""
            raise TimeoutError(
                f"No LiveKit agent participant joined the room{where} within {self._agent_wait_timeout_s:g}s."
            ) from exc
        finally:
            if self.agent_identity:
                with contextlib.suppress(Exception):
                    self._room.off("participant_connected", self._on_participant_connected)
        assert self.agent_identity is not None
        return self.agent_identity

    # --------------------------------------------------------------- handlers

    def _register_handlers(self) -> None:
        def on_clear_buffer(data: Any) -> str:
            if data.caller_identity != self.agent_identity:
                logger.warning(
                    "[receiver] lk.clear_buffer from unexpected participant %s",
                    data.caller_identity,
                )
                return "reject"
            self._current_cleared = True
            self._superseded.set()
            while not self._queue.empty():
                with contextlib.suppress(asyncio.QueueEmpty):
                    self._queue.get_nowait()
            for callback in tuple(self._clear_handlers):
                try:
                    callback()
                except Exception as exc:
                    logger.warning(f"[receiver] clear-buffer callback failed: {exc}")
            return "ok"

        self._room.local_participant.register_rpc_method(RPC_CLEAR_BUFFER, on_clear_buffer)

        def on_stream(reader: Any, participant_id: str) -> None:
            if participant_id != self.agent_identity:
                logger.debug("[receiver] ignoring lk.audio_stream from %s", participant_id)
                return
            if self._readers:
                # A new segment supersedes an open one whose trailer was lost.
                self._superseded.set()
            self._readers.append(reader)
            self._reader_event.set()

        self._room.register_byte_stream_handler(AUDIO_STREAM_TOPIC, on_stream)

    # ---------------------------------------------------------------- reading

    async def _read_loop(self) -> None:
        try:
            while not self._closing:
                await self._reader_event.wait()
                # Clear before consuming: a stream that arrives while this
                # batch is being read must set the event again, not have its
                # wake-up wiped out after the loop.
                self._reader_event.clear()
                while self._readers and not self._closing:
                    reader = self._readers.popleft()
                    await self._consume_reader(reader)
        except asyncio.CancelledError:
            raise
        except Exception:
            # A broken reader must not take the whole receiver down: the room
            # keeps sending TTS segments after a corrupted one.
            logger.exception("[receiver] audio stream reader failed")

    async def _consume_reader(self, reader: Any) -> None:
        self._current_cleared = False
        self._superseded.clear()
        attrs = getattr(reader.info, "attributes", None) or {}
        try:
            sample_rate = int(attrs.get("sample_rate") or self._sample_rate or 24000)
            num_channels = int(attrs.get("num_channels") or self._num_channels)
        except (TypeError, ValueError):
            sample_rate = int(self._sample_rate or 24000)
            num_channels = self._num_channels
        if self._sample_rate:
            sample_rate = int(self._sample_rate)

        framer = _PcmFramer(sample_rate, num_channels, self._frame_size_ms)
        try:
            async for data in self._iter_reader(reader):
                if self._current_cleared:
                    framer.clear()
                    break
                for frame in framer.push(data):
                    self._queue.put_nowait(frame)
            if not self._current_cleared:
                for frame in framer.flush():
                    self._queue.put_nowait(frame)
                self._queue.put_nowait(SEGMENT_END)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if self._current_cleared:
                logger.debug("[receiver] segment ended after clear_buffer: %s", exc)
            else:
                # A native stream error ("expected chunk index ...", a
                # disconnect, a lost trailer) is scoped to this segment. The
                # audio already received is still usable, and ending the
                # segment lets the renderer flush its tail and report
                # playback_finished instead of leaving the agent waiting.
                logger.warning("[receiver] segment read failed: %s", exc)
                for frame in framer.flush():
                    self._queue.put_nowait(frame)
                self._queue.put_nowait(SEGMENT_END)
        finally:
            # Readers abandoned after a trailer error / supersede / barge-in
            # otherwise keep an FFI subscription for the life of the process.
            close = getattr(reader, "close", None)
            if callable(close):
                with contextlib.suppress(Exception):
                    close()

    async def _iter_reader(self, reader: Any) -> AsyncIterator[bytes]:
        consecutive_errors = 0
        while True:
            next_chunk = asyncio.ensure_future(reader.__anext__())
            superseded = asyncio.ensure_future(self._superseded.wait())
            try:
                done, _pending = await asyncio.wait(
                    {next_chunk, superseded},
                    timeout=self._idle_timeout_s,
                    return_when=asyncio.FIRST_COMPLETED,
                )
            except asyncio.CancelledError:
                next_chunk.cancel()
                superseded.cancel()
                raise
            if next_chunk in done:
                superseded.cancel()
                try:
                    data = next_chunk.result()
                except StopAsyncIteration:
                    return
                except Exception as exc:
                    # e.g. "expected chunk index to be exactly one more than
                    # the previous" (MissedChunk): one chunk was lost in
                    # transport. That does not have to end the segment — the
                    # reader may still deliver the chunks after the gap. Retry
                    # a few times; abort only on repeated failure.
                    consecutive_errors += 1
                    if consecutive_errors > READER_ERROR_RETRIES:
                        raise
                    logger.warning(
                        "[receiver] stream chunk lost, retrying (%d/%d): %s",
                        consecutive_errors,
                        READER_ERROR_RETRIES,
                        exc,
                    )
                    await asyncio.sleep(READER_ERROR_RETRY_DELAY_S)
                    continue
                consecutive_errors = 0
                yield data
                continue
            # Superseded, cleared, or idle: stop this segment. The reader's
            # trailer is gone, and waiting for it would wedge the receiver.
            next_chunk.cancel()
            superseded.cancel()
            if not self._current_cleared:
                logger.warning(
                    "[receiver] audio segment ended before its trailer (superseded or idle for %ss)",
                    self._idle_timeout_s,
                )
            return

    # ------------------------------------------------------------------- rpc

    async def notify_playback_started(self) -> None:
        await self._rpc(RPC_PLAYBACK_STARTED, "")

    async def notify_playback_finished(
        self,
        *,
        playback_position: float,
        interrupted: bool,
        synchronized_transcript: str | None = None,
    ) -> None:
        payload = json.dumps(
            {
                "playback_position": max(0.0, float(playback_position)),
                "interrupted": bool(interrupted),
                "synchronized_transcript": synchronized_transcript,
            }
        )
        await self._rpc(RPC_PLAYBACK_FINISHED, payload)

    async def _rpc(self, method: str, payload: str) -> None:
        if not self.agent_identity:
            return
        for attempt in range(self._rpc_max_retries):
            try:
                await self._room.local_participant.perform_rpc(
                    destination_identity=self.agent_identity,
                    method=method,
                    payload=payload,
                    response_timeout=2.0,
                )
                return
            except Exception as exc:
                if attempt == self._rpc_max_retries - 1:
                    logger.debug(f"[receiver] {method} rpc failed: {exc}")
                    return
                await asyncio.sleep(0.1)

    # -------------------------------------------------------------- iteration

    def __aiter__(self) -> AsyncIterator[Any]:
        return self

    async def __anext__(self) -> Any:
        if self._closing:
            raise StopAsyncIteration
        item = await self._queue.get()
        if isinstance(item, Exception):
            raise item
        return item


def pcm16_to_float32(frame: Any) -> np.ndarray:
    """Test/utility helper: convert a frame's interleaved PCM16 to float32."""
    samples = np.frombuffer(frame.data, dtype=np.int16)
    return samples.astype(np.float32) / 32768.0


__all__ = [
    "AUDIO_STREAM_TOPIC",
    "DEFAULT_FRAME_SIZE_MS",
    "RPC_CLEAR_BUFFER",
    "RPC_PLAYBACK_FINISHED",
    "RPC_PLAYBACK_STARTED",
    "SEGMENT_END",
    "DataStreamAudioReceiver",
    "SegmentEnd",
    "pcm16_to_float32",
]
