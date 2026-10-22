"""Tests for LiveKit generator, worker, agent, and publish endpoint."""

from __future__ import annotations

import asyncio
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fast_float.core.state import app_state
from fast_float.livekit.agent import ConversationalAgent
from fast_float.livekit.generator import (
    FloatVideoGenerator,
    LiveStreamVideoGenerator,
    MediaFileVideoGenerator,
)
from fast_float.livekit.worker import (
    AvatarOptions,
    AvatarRunner,
)
from fast_float.routers.livekit import router as livekit_router
from fast_float.streaming.live import LiveStream
from fast_float.streaming.types import StreamMeta

try:
    from livekit import rtc

    RTC_AVAILABLE = True
except ImportError:
    RTC_AVAILABLE = False


@pytest.fixture
def api_client(monkeypatch):
    monkeypatch.setenv("LIVEKIT_URL", "ws://127.0.0.1:7880")
    monkeypatch.setenv("LIVEKIT_API_KEY", "devkey")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "devsecretdevsecretdevsecretdevsecret")
    app = FastAPI()
    app.include_router(livekit_router)
    with TestClient(app) as client:
        yield client


def test_float_video_generator_accumulates_and_yields():
    async def _test():
        loop = asyncio.get_running_loop()
        gen = FloatVideoGenerator(loop=loop)

        # Push 2.0s of audio (32,000 samples @ 16kHz int16 mono = 64,000 bytes)
        audio_pcm = np.zeros(32000, dtype=np.int16).tobytes()
        gen.push_audio(audio_pcm)

        # Wait for the chunk to process
        await asyncio.sleep(0.1)
        gen.stop()

        frames = [frame async for frame in gen]
        assert len(frames) == 50
        if RTC_AVAILABLE:
            first = frames[0]
            if isinstance(first, tuple):
                v_frame, a_frame = first
                assert isinstance(v_frame, rtc.VideoFrame)
                assert v_frame.width == 512
                assert v_frame.height == 512
                assert isinstance(a_frame, rtc.AudioFrame)
            else:
                assert isinstance(first, rtc.VideoFrame)
                assert first.width == 512

    asyncio.run(_test())


def test_livestream_video_generator_taps_stream():
    async def _test():
        loop = asyncio.get_running_loop()
        stream = LiveStream("req-123", loop)
        meta = StreamMeta(request_id="req-123", fps=25.0, first_idx=0)
        stream.start(meta)

        gen = LiveStreamVideoGenerator(stream)

        # Push mock frames
        q = next(iter(stream._subs))
        frame_arr = np.zeros((512, 512, 3), dtype=np.uint8)
        q.put_nowait(("frame", 0, 0.0, frame_arr))
        q.put_nowait(("frame", 1, 40.0, frame_arr))
        stream.finish()

        collected = [frame async for frame in gen]
        assert len(collected) == 2
        if RTC_AVAILABLE:
            first = collected[0]
            if isinstance(first, tuple):
                v_frame, _ = first
                assert isinstance(v_frame, rtc.VideoFrame)
            else:
                assert isinstance(first, rtc.VideoFrame)

    asyncio.run(_test())


def test_avatar_runner_lifecycle():
    if not RTC_AVAILABLE:
        pytest.skip("RTC SDK not available")

    async def _test():
        room = MagicMock()
        room.local_participant = MagicMock()
        room.local_participant.publish_track = AsyncMock()
        room.local_participant.set_attributes = AsyncMock()
        room.local_participant.perform_rpc = AsyncMock()
        room.local_participant.unpublish_track = AsyncMock()

        loop = asyncio.get_running_loop()
        gen = FloatVideoGenerator(loop=loop)

        options = AvatarOptions(video_width=512, video_height=512, video_fps=25.0)
        runner = AvatarRunner(room, gen, options=options, agent_identity="agent-1")

        await runner.start()

        assert runner._running is True
        assert room.local_participant.publish_track.call_count == 2
        room.local_participant.set_attributes.assert_called_with({"lk.publish_on_behalf": "agent-1"})

        audio_data = np.zeros(320, dtype=np.int16).tobytes()
        a_frame = rtc.AudioFrame(audio_data, 16000, 1, 320)
        await runner.push_audio_frame(a_frame)

        await runner.stop()
        assert runner._stopped is True

    asyncio.run(_test())


def test_conversational_agent_turn():
    if not RTC_AVAILABLE:
        pytest.skip("RTC SDK not available")

    async def _test():
        loop = asyncio.get_running_loop()
        gen = FloatVideoGenerator(loop=loop)

        agent = ConversationalAgent(
            room_name="room-test",
            video_gen=gen,
            tts_fn=lambda text: np.zeros(16000, dtype=np.int16).tobytes(),
        )

        await agent.send_speech("Hello world")
        await asyncio.sleep(0.05)
        assert gen._inference_running or gen._frame_queue.qsize() > 0 or gen._frame_count > 0

    asyncio.run(_test())


def test_media_file_video_generator_lifecycle():
    gen = MediaFileVideoGenerator("dummy.mp4")
    assert gen.file_path == "dummy.mp4"
    gen.stop()
    assert gen._stopped is True


def test_publish_endpoint_404_when_stream_not_found(api_client):
    res = api_client.post("/livekit/publish/nonexistent-id", json={})
    assert res.status_code == 404


def test_publish_endpoint_starts_live_publisher(api_client):
    loop = asyncio.new_event_loop()
    stream = LiveStream("req-active", loop)
    app_state.live_streams["req-active"] = stream

    with patch("fast_float.livekit.worker.publish_live_stream_task") as mock_task:
        res = api_client.post("/livekit/publish/req-active", json={"room": "custom-room"})
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "publishing"
        assert data["type"] == "live"
        assert data["room"] == "custom-room"
        assert data["request_id"] == "req-active"


def test_publish_endpoint_starts_file_publisher(api_client):
    with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
        app_state.active_streams["req-done"] = tmp.name

        with patch("fast_float.livekit.worker.publish_media_file_task") as mock_task:
            res = api_client.post("/livekit/publish/req-done", json={"room": "done-room"})
            assert res.status_code == 200
            data = res.json()
            assert data["status"] == "publishing"
            assert data["type"] == "file"
            assert data["room"] == "done-room"
            assert data["request_id"] == "req-done"
