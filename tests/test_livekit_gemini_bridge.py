"""Tests for the Gemini Live / livekit-agents DataStream bridge (no network)."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock

import jwt
import numpy as np
import pytest

from fast_float.livekit.generator import FloatVideoGenerator
from fast_float.livekit.receiver import SEGMENT_END, DataStreamAudioReceiver, _PcmFramer
from fast_float.livekit.tokens import issue_token

try:
    from livekit import rtc

    RTC_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the environment
    RTC_AVAILABLE = False


def test_pcm_framer_splits_into_fixed_frames_and_flushes_the_tail():
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

    def __aiter__(self):
        return self

    async def __anext__(self) -> bytes:
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)


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
