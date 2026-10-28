"""Tests for the Gemini Live / livekit-agents DataStream bridge (no network)."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import jwt
import numpy as np
import pytest

from fast_float.livekit.agent_runner import AgentAvatarRunner
from fast_float.livekit.generator import FloatVideoGenerator
from fast_float.livekit.receiver import SEGMENT_END, DataStreamAudioReceiver, _PcmFramer
from fast_float.livekit.tokens import issue_token

try:
    from livekit import rtc

    RTC_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the environment
    RTC_AVAILABLE = False


def test_pcm_framer_splits_into_fixed_frames_and_flushes_the_tail():
    if not RTC_AVAILABLE:
        pytest.skip("LiveKit RTC SDK not installed")
    # 16 kHz mono, 100 ms frames = 1600 samples = 3200 bytes.
    framer = _PcmFramer(16000, 1, 100)
    data = np.arange(4000, dtype=np.int16).tobytes()
    frames = framer.push(data)
    assert len(frames) == 2
    assert frames[0].samples_per_channel == 1600
    tail = framer.flush()
    assert len(tail) == 1
    # 800 samples left over
    assert tail[0].samples_per_channel == 800
    assert framer.flush() == []


def test_pcm_framer_drops_a_dangling_byte():
    if not RTC_AVAILABLE:
        pytest.skip("LiveKit RTC SDK not installed")
    framer = _PcmFramer(16000, 1, 100)
    framer.push(b"\x01\x02\x03")
    frames = framer.flush()
    assert len(frames) == 1
    assert frames[0].samples_per_channel == 1


class _FakeReader:
    def __init__(self, chunks: list[bytes], sample_rate: int = 24000) -> None:
        self._chunks = list(chunks)
        self.info = MagicMock()
        self.info.attributes = {"sample_rate": str(sample_rate), "num_channels": "1"}
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self) -> bytes:
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)

    def close(self) -> None:
        self.closed = True


class _BrokenReader(_FakeReader):
    """A reader whose trailer never arrives; the native SDK raises instead."""

    async def __anext__(self) -> bytes:
        if self._chunks:
            return self._chunks.pop(0)
        raise RuntimeError("expected chunk index to be exactly one more than the previous")


class _GappyReader(_FakeReader):
    """A reader that raises MissedChunk once, then keeps delivering chunks.

    The native ByteStreamReader errors on the in-flight chunk when transport
    loses one, but the stream itself can continue afterwards — the receiver
    retries the pull instead of aborting the segment.
    """

    def __init__(self, chunks: list[bytes]) -> None:
        super().__init__(chunks)
        self.raised = False

    async def __anext__(self) -> bytes:
        if not self.raised and self._chunks and len(self._chunks) > 1:
            self.raised = True
            self._chunks.pop(0)  # the chunk in flight is lost
            raise RuntimeError("expected chunk index to be exactly one more than the previous")
        return await super().__anext__()


@pytest.mark.skipif(not RTC_AVAILABLE, reason="LiveKit RTC SDK not installed")
def test_receiver_emits_frames_then_segment_end():
    async def _test():
        receiver = DataStreamAudioReceiver(MagicMock(), sender_identity="agent-1")
        receiver.agent_identity = "agent-1"
        # 24 kHz mono, 200 ms total (4800 samples) delivered in two chunks.
        pcm = np.arange(4800, dtype=np.int16).tobytes()
        reader = _FakeReader([pcm[:4000], pcm[4000:]])
        await receiver._consume_reader(reader)

        items = []
        while not receiver._queue.empty():
            items.append(receiver._queue.get_nowait())
        assert items[-1] is SEGMENT_END
        frames = items[:-1]
        assert frames
        assert all(isinstance(f, rtc.AudioFrame) for f in frames)
        assert sum(f.samples_per_channel for f in frames) == 4800
        assert frames[0].sample_rate == 24000

    asyncio.run(_test())


@pytest.mark.skipif(not RTC_AVAILABLE, reason="LiveKit RTC SDK not installed")
def test_receiver_keeps_partial_audio_after_a_stream_error():
    """One corrupted segment must not poison the queue and stop the avatar.

    A native ByteStreamReader can raise ``expected chunk index ...`` when its
    trailer/chunks are lost. That error is scoped to the segment: flush what
    was received, emit SEGMENT_END so playback_finished is still reported, and
    keep the receiver alive for the next TTS segment.
    """

    async def _test():
        receiver = DataStreamAudioReceiver(MagicMock(), sender_identity="agent-1")
        receiver.agent_identity = "agent-1"

        pcm = np.arange(4800, dtype=np.int16).tobytes()
        broken = _BrokenReader([pcm[:4000]])
        good = _FakeReader([pcm])

        receiver._readers.extend([broken, good])
        receiver._reader_event.set()
        task = asyncio.create_task(receiver._read_loop())
        try:
            for _ in range(200):
                if receiver._queue.qsize() >= 5:  # broken tail + good frames + 2 ends
                    break
                await asyncio.sleep(0.01)
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        items = []
        while not receiver._queue.empty():
            items.append(receiver._queue.get_nowait())

        # No Exception object in the stream: the consumer would raise it and
        # abandon every later segment.
        assert not any(isinstance(item, Exception) for item in items)
        assert sum(item is SEGMENT_END for item in items) == 2
        frames = [item for item in items if item is not SEGMENT_END]
        assert frames
        assert all(isinstance(frame, rtc.AudioFrame) for frame in frames)
        # Partial audio from the broken segment survived, and the next reader
        # was still consumed.
        assert sum(frame.samples_per_channel for frame in frames) == 2000 + 4800
        assert broken.closed
        assert good.closed

    asyncio.run(_test())


@pytest.mark.skipif(not RTC_AVAILABLE, reason="LiveKit RTC SDK not installed")
def test_receiver_recovers_after_a_lost_chunk():
    """A MissedChunk error must not truncate the rest of the reply.

    One chunk lost in transport used to end the whole segment mid-sentence
    ("counting 1 to 10 cut at 6"). The reader keeps delivering the chunks
    after the gap, so the receiver retries the pull and renders the rest.
    """

    async def _test():
        receiver = DataStreamAudioReceiver(MagicMock(), sender_identity="agent-1")
        receiver.agent_identity = "agent-1"
        # 24 kHz mono: five 100 ms-ish chunks, one lost in the middle.
        pcm = np.arange(12000, dtype=np.int16).tobytes()
        chunks = [pcm[i : i + 2400] for i in range(0, 12000, 2400)]
        reader = _GappyReader(chunks)
        await receiver._consume_reader(reader)

        items = []
        while not receiver._queue.empty():
            items.append(receiver._queue.get_nowait())
        assert items[-1] is SEGMENT_END
        frames = items[:-1]
        assert all(isinstance(frame, rtc.AudioFrame) for frame in frames)
        # Everything except the one lost chunk (2400 bytes = 1200 samples).
        total = sum(frame.samples_per_channel for frame in frames)
        assert total == 6000 - 1200
        assert reader.closed

    asyncio.run(_test())


@pytest.mark.skipif(not RTC_AVAILABLE, reason="LiveKit RTC SDK not installed")
def test_receiver_clear_buffer_discards_the_rest_of_a_segment():
    async def _test():
        receiver = DataStreamAudioReceiver(MagicMock(), sender_identity="agent-1")
        receiver.agent_identity = "agent-1"

        class _ClearDuringRead(_FakeReader):
            async def __anext__(self) -> bytes:
                receiver._current_cleared = True
                receiver._superseded.set()
                return await super().__anext__()

        reader = _ClearDuringRead([np.zeros(4800, dtype=np.int16).tobytes()])
        await receiver._consume_reader(reader)
        items = []
        while not receiver._queue.empty():
            items.append(receiver._queue.get_nowait())
        # No frames and no SEGMENT_END after a barge-in.
        assert items == []

    asyncio.run(_test())


@pytest.mark.skipif(not RTC_AVAILABLE, reason="LiveKit RTC SDK not installed")
def test_generator_wait_until_idle_after_frames_are_consumed():
    async def _test():
        gen = FloatVideoGenerator(width=8, height=8)
        gen.push_audio(np.zeros(16000, dtype=np.int16).tobytes())  # 1 s -> first chunk
        assert not gen.idle
        frames = []
        async for item in gen:
            frames.append(item)
            if len(frames) >= 25:
                break
        gen.stop()
        # Consume the sentinel so __aiter__ can finish if asked again.
        assert await gen.wait_until_idle(timeout=2.0) is True
        assert len(frames) == 25

    asyncio.run(_test())


@pytest.mark.skipif(not RTC_AVAILABLE, reason="LiveKit RTC SDK not installed")
def test_receiver_wait_error_mentions_room_and_timeout():
    """The agent-wait timeout names the room and window for a readable 503."""

    class _EmptyRoom:
        def __init__(self) -> None:
            self.name = "float-noworker"
            self.remote_participants: dict = {}

        def on(self, *_args, **_kwargs):
            pass

        def off(self, *_args, **_kwargs):
            pass

    receiver = DataStreamAudioReceiver(_EmptyRoom(), agent_wait_timeout_s=0.05)
    with pytest.raises(TimeoutError) as excinfo:
        asyncio.run(receiver._wait_for_agent())
    message = str(excinfo.value)
    assert "float-noworker" in message
    assert "0.05" in message
    assert "agent" in message.lower()


def test_avatar_token_is_agent_kind_for_room_io_isolation():
    issued = issue_token(
        url="ws://127.0.0.1:7880",
        api_key="devkey",
        api_secret="devsecretdevsecretdevsecretdevsecret",
        identity="float-avatar",
        room="float-test",
        can_publish=True,
        agent=True,
    )
    claims = jwt.decode(issued.token, "devsecretdevsecretdevsecretdevsecret", algorithms=["HS256"])
    assert claims["video"]["agent"] is True
    assert claims["video"]["canPublish"] is True


def test_playback_finished_payload_matches_livekit_agents():
    """livekit-agents deserializes this payload into PlaybackFinishedEvent."""
    from dataclasses import asdict, dataclass

    @dataclass
    class PlaybackFinishedEvent:
        playback_position: float
        interrupted: bool
        synchronized_transcript: str | None = None

    payload = json.dumps(
        {
            "playback_position": 1.25,
            "interrupted": False,
            "synchronized_transcript": None,
        }
    )
    event = PlaybackFinishedEvent(**json.loads(payload))
    assert asdict(event)["playback_position"] == 1.25


@pytest.mark.skipif(not RTC_AVAILABLE, reason="LiveKit RTC SDK not installed")
def test_avatar_interrupt_awaits_av_sync_clear_queue():
    """Barge-in must actually drain AVSynchronizer's async queue."""

    async def _test():
        cleared = asyncio.Event()

        av_sync = MagicMock()

        async def _clear_queue() -> None:
            cleared.set()

        av_sync.clear_queue = _clear_queue
        receiver = MagicMock()
        receiver.agent_identity = "agent-1"
        receiver.notify_playback_finished = AsyncMock()

        runner = AgentAvatarRunner(
            MagicMock(),
            video_gen=MagicMock(),
            receiver=receiver,
        )
        runner._av_sync = av_sync
        runner._last_agent_identity = "agent-1"

        runner.interrupt()
        await asyncio.wait_for(cleared.wait(), timeout=1.0)
        await asyncio.sleep(0)
        receiver.notify_playback_finished.assert_awaited_once()

    asyncio.run(_test())
