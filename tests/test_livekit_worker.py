"""Tests for LiveKit generator, worker, agent, and publish endpoint."""

from __future__ import annotations

import asyncio
import tempfile
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fast_float.core.state import app_state
from fast_float.livekit.agent import ConversationalAgent
from fast_float.livekit.generator import (
    BYTES_PER_FRAME,
    FloatVideoGenerator,
    LiveStreamVideoGenerator,
    MediaFileVideoGenerator,
    _LiveKitDispatcherBridge,
)
from fast_float.livekit.worker import (
    AvatarOptions,
    AvatarRunner,
    live_publish_active,
    spawn_live_publish_task,
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
    # The publish endpoint refuses to spawn when the RTC SDK is missing (503);
    # force the flag so endpoint tests run on boxes without the SDK too.
    if not RTC_AVAILABLE:
        import fast_float.livekit.worker as worker

        monkeypatch.setattr(worker, "RTC_AVAILABLE", True)
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


def test_generator_bridge_keeps_dispatcher_cursor_and_audio_position():
    async def _test():
        loop = asyncio.get_running_loop()
        pcm = np.arange(32000, dtype=np.int16).tobytes()
        pushed = []
        bridge = _LiveKitDispatcherBridge(loop, lambda gen, item: pushed.append(item), pcm, 32000, 0, lambda gen: True)
        bridge.bind(StreamMeta(fps=25.0, width=8, height=8, channels=3))

        frames = np.zeros((10, 8, 8, 3), dtype=np.uint8)
        bridge.submit(frames)
        await asyncio.sleep(0.1)
        # idx=None must use the dispatcher's frame cursor (10), not the audio
        # position; otherwise the next batch is submitted at a huge index and
        # stalls until shutdown.
        bridge.submit(frames)
        await asyncio.to_thread(bridge.shutdown, 1.0)
        await asyncio.sleep(0.05)

        assert [item[0] for item in pushed] == list(range(20))
        for item in pushed:
            idx = item[0]
            expected = pcm[idx * BYTES_PER_FRAME : (idx + 1) * BYTES_PER_FRAME]
            assert item[3] == expected

    asyncio.run(_test())


def test_generator_bridge_pairs_audio_after_dispatcher_gap():
    async def _test():
        loop = asyncio.get_running_loop()
        pcm = np.arange(32000, dtype=np.int16).tobytes()
        pushed = []
        bridge = _LiveKitDispatcherBridge(loop, lambda gen, item: pushed.append(item), pcm, 32000, 0, lambda gen: True)
        bridge.bind(StreamMeta(fps=25.0, width=8, height=8, channels=3))

        bridge.submit(np.zeros((2, 8, 8, 3), dtype=np.uint8))  # frames 0, 1
        bridge.submit(np.zeros((2, 8, 8, 3), dtype=np.uint8), idx=10)  # frames 2-9 lost
        await asyncio.to_thread(bridge.shutdown, 1.0)
        await asyncio.sleep(0.05)

        assert [item[0] for item in pushed] == [0, 1, 10, 11]
        for item in pushed:
            idx = item[0]
            expected = pcm[idx * BYTES_PER_FRAME : (idx + 1) * BYTES_PER_FRAME]
            assert item[3] == expected

    asyncio.run(_test())


def test_float_generator_shuts_down_bridge_per_chunk(monkeypatch):
    async def _test():
        loop = asyncio.get_running_loop()
        captured = {}

        def fake_inference(agent, video, image, wav_path, request, bridge):
            captured["bridge"] = bridge
            bridge.bind(StreamMeta(fps=25.0, width=8, height=8, channels=3))
            bridge.submit(np.zeros((25, 8, 8, 3), dtype=np.uint8))
            bridge.submit(np.zeros((25, 8, 8, 3), dtype=np.uint8))
            return "ok"

        monkeypatch.setattr("fast_float.core.service.run_inference_blocking", fake_inference)
        gen = FloatVideoGenerator(agent=object(), image_path="fake.jpg", loop=loop, width=8, height=8)
        gen.push_audio(np.zeros(32000, dtype=np.int16).tobytes())

        for _ in range(200):
            if captured.get("bridge") is not None and not gen._inference_running:
                break
            await asyncio.sleep(0.01)

        bridge = captured["bridge"]
        assert bridge._cursor == 50  # dispatcher frame cursor, not an audio cursor
        assert bridge._thread is not None
        assert not bridge._thread.is_alive()

        gen.stop()
        frames = [frame async for frame in gen]
        assert len(frames) == 50

    asyncio.run(_test())


def test_generator_bridge_cancelled_aborts_dispatcher():
    async def _test():
        loop = asyncio.get_running_loop()
        stop = threading.Event()
        pushed = []
        bridge = _LiveKitDispatcherBridge(
            loop,
            lambda gen, item: pushed.append(item),
            np.zeros(32000, dtype=np.int16).tobytes(),
            32000,
            0,
            lambda gen: False,
        )
        bridge.bind(StreamMeta(fps=25.0, width=8, height=8, channels=3), stop_event=stop)
        bridge.submit(np.zeros((4, 8, 8, 3), dtype=np.uint8))

        for _ in range(100):
            if stop.is_set():
                break
            await asyncio.sleep(0.01)

        assert stop.is_set()
        assert pushed == []
        await asyncio.to_thread(bridge.shutdown, 1.0)

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


def test_agent_registers_rpc_handlers_after_connect():
    class FakeLocalParticipant:
        def __init__(self):
            self.methods = []

        def register_rpc_method(self, name, handler):
            self.methods.append(name)

    class FakeRoom:
        def __init__(self):
            self.connected = False
            self.local = FakeLocalParticipant()

        @property
        def local_participant(self):
            # The RTC SDK raises this until the room is connected.
            if not self.connected:
                raise Exception("cannot access local participant before connecting")
            return self.local

        async def connect(self, url, token):
            self.connected = True

    room = FakeRoom()
    agent = ConversationalAgent(room_name="room-test")
    with (
        patch("fast_float.livekit.agent.rtc.Room", return_value=room),
        patch("fast_float.livekit.agent.issue_token", return_value=MagicMock(token="tok")),
    ):
        asyncio.run(agent.connect())

    assert room.connected is True
    assert room.local.methods == ["lk.playback_started", "lk.playback_finished"]


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

    spawn_calls = {"count": 0}

    async def _never_finish(*args, **kwargs):
        spawn_calls["count"] += 1
        await asyncio.sleep(3600)

    with patch("fast_float.livekit.worker.publish_live_stream_task", _never_finish):
        res = api_client.post("/livekit/publish/req-active", json={"room": "custom-room"})
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "publishing"
        assert data["type"] == "live"
        assert data["room"] == "custom-room"
        assert data["request_id"] == "req-active"
        assert spawn_calls["count"] == 1

        # A second publish trigger for the same request must not spawn another
        # publisher: both would join the room with the same worker identity and
        # LiveKit would drop one of them.
        res = api_client.post("/livekit/publish/req-active", json={"room": "custom-room"})
        assert res.status_code == 200
        assert res.json()["status"] == "already_publishing"
        assert spawn_calls["count"] == 1

    # task cleanup between tests: the blocking task from this test must not
    # leak into the next one through the registry
    import fast_float.livekit.worker as worker

    for job in list(worker._live_publish_jobs.values()):
        job.cancel()


async def _never_finish_media(*args, **kwargs):
    await asyncio.sleep(3600)


def test_publish_endpoint_starts_file_publisher(api_client):
    with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
        app_state.active_streams["req-done"] = tmp.name

        with patch("fast_float.livekit.worker.publish_media_file_task", _never_finish_media):
            res = api_client.post("/livekit/publish/req-done", json={"room": "done-room"})
            assert res.status_code == 200
            data = res.json()
            assert data["status"] == "publishing"
            assert data["type"] == "file"
            assert data["room"] == "done-room"
            assert data["request_id"] == "req-done"

    import fast_float.livekit.worker as worker

    for job in list(worker._media_publish_jobs.values()):
        job.cancel()


def test_spawn_live_publish_task_dedupes():
    async def _test():
        stream = LiveStream("req-dup", asyncio.get_running_loop())
        stream.request_id = "req-dup"
        with patch("fast_float.livekit.worker.publish_live_stream_task") as mock_task:
            first = spawn_live_publish_task(stream, room_name="r1")
            assert first is not None
            second = spawn_live_publish_task(stream, room_name="r1")
            assert second is None
            assert mock_task.call_count == 1
            assert live_publish_active("req-dup") is True
            first.cancel()

    asyncio.run(_test())


def test_conversation_session_wiring():
    if not RTC_AVAILABLE:
        pytest.skip("RTC SDK not available")

    async def _test():
        import time

        from fast_float.livekit import session as sess
        from fast_float.livekit.settings import LiveKitSettings

        def _slow_inference(agent, *args, **kwargs):
            # Real inference takes seconds; keep the chunk busy so the flag
            # under test is observable deterministically.
            time.sleep(0.5)
            return "ok"

        settings = LiveKitSettings(
            url="ws://127.0.0.1:7880",
            api_key="devkey",
            api_secret="devsecretdevsecretdevsecretdevsecret",
        )
        try:
            app_state.agent = MagicMock()
            fake_room = MagicMock()
            fake_room.connect = AsyncMock()
            fake_room.local_participant = MagicMock()
            fake_room.local_participant.publish_track = AsyncMock()
            fake_room.local_participant.set_attributes = AsyncMock()
            fake_room.disconnect = AsyncMock()

            with (
                patch.object(ConversationalAgent, "connect", AsyncMock()) as agent_connect,
                patch("fast_float.livekit.session.rtc") as rtc_mod,
                patch("fast_float.core.service.run_inference_blocking", _slow_inference),
            ):
                rtc_mod.Room.return_value = fake_room
                session = sess.ConversationSession(
                    "room-conv-test",
                    settings=settings,
                    image_path="./images/img.jpg",
                    tts_fn=lambda text: np.zeros(1600, dtype=np.int16).tobytes(),
                )
                await session.start()

                assert session.agent is not None
                assert session.runner is not None
                assert session.runner.agent_identity == session.agent.identity
                assert fake_room.connect.call_count == 1
                # avatar video + audio published into the worker room
                assert fake_room.local_participant.publish_track.call_count == 2

                await session.speak("Hello there")
                await asyncio.sleep(0.05)
                assert session.generator._inference_running

                await session.stop()
                assert session.is_stopped
                agent_connect.assert_awaited_once()
        finally:
            app_state.agent = None

    asyncio.run(_test())


def test_conversation_start_reports_warmup(api_client):
    res = api_client.post("/livekit/conversation/start", json={})
    assert res.status_code == 503
    assert "warming up" in res.json()["detail"]


def test_conversation_endpoints_flow(api_client, monkeypatch):
    import types

    import fast_float.routers.livekit as lk_router_module

    fake_session = types.SimpleNamespace(
        room_name="room-conv",
        identity="float-agent",
        avatar_identity="avatar-room-conv",
        tts_fn=lambda text: b"",
        is_stopped=False,
        speak=AsyncMock(),
    )

    async def fake_start(room_name, **kwargs):
        return fake_session

    async def fake_stop(room_name):
        fake_session.is_stopped = True

    monkeypatch.setattr(lk_router_module, "start_session", fake_start)
    monkeypatch.setattr(lk_router_module, "stop_session", fake_stop)
    monkeypatch.setattr(lk_router_module, "get_session", lambda room: fake_session)
    monkeypatch.setattr(app_state, "agent", MagicMock())

    res = api_client.post("/livekit/conversation/start", json={"room": "room-conv"})
    assert res.status_code == 200
    assert res.json() == {
        "status": "ready",
        "room": "room-conv",
        "identity": "float-agent",
        "avatar_identity": "avatar-room-conv",
        "tts": True,
    }

    res = api_client.post("/livekit/conversation/speak", json={"room": "room-conv", "text": "hi"})
    assert res.status_code == 200
    assert res.json()["status"] == "speaking"

    res = api_client.post("/livekit/conversation/stop", json={"room": "room-conv"})
    assert res.status_code == 200
    assert res.json() == {"room": "room-conv", "status": "stopped", "had_session": True}


def test_float_generator_paces_production_without_dropping_frames():
    """A long utterance must not be cut in the middle by queue overflow.

    Production is faster than playout; without backpressure the frame queue
    fills and drops the oldest frames, so the listener hears a jump cut. The
    generator must wait for the consumer instead.
    """

    async def _test():
        loop = asyncio.get_running_loop()
        gen = FloatVideoGenerator(loop=loop, width=8, height=8)

        # 10 s of audio = 250 frames at 25 fps, twice the queue capacity.
        gen.push_audio(np.zeros(16000 * 10, dtype=np.int16).tobytes())
        gen.flush()

        seen = []
        for _ in range(10000):
            try:
                item = await asyncio.wait_for(gen._frame_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                break
            seen.append(item[0])
            await asyncio.sleep(0.001)  # consumer slower than inference
            if len(seen) == 250:
                break

        gen.stop()
        # Every frame of the utterance arrives (none was dropped to catch up)
        # and the counters agree; frame indices restart per chunk by design.
        assert len(seen) == 250
        assert gen._dropped_frames == 0

    asyncio.run(_test())


def test_float_generator_stop_plays_out_the_queued_tail():
    async def _test():
        loop = asyncio.get_running_loop()
        gen = FloatVideoGenerator(loop=loop, width=8, height=8)
        gen.push_audio(np.zeros(16000 * 2, dtype=np.int16).tobytes())

        for _ in range(300):
            if gen._frame_queue.qsize() >= 50:
                break
            await asyncio.sleep(0.01)
        assert gen._frame_queue.qsize() == 50

        # stop() must not throw the already-generated frames away.
        gen.stop()
        frames = [frame async for frame in gen]
        assert len(frames) == 50

    asyncio.run(_test())


def test_livestream_generator_stop_plays_out_the_queued_tail():
    async def _test():
        loop = asyncio.get_running_loop()
        stream = LiveStream("req-tail", loop)
        stream.start(StreamMeta(fps=25.0, width=8, height=8, channels=3, request_id="req-tail"))
        gen = LiveStreamVideoGenerator(stream)

        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        for idx in range(5):
            gen._queue.put_nowait(("frame", idx, idx * 40.0, frame))

        gen.stop()
        collected = [item async for item in gen]
        assert len(collected) == 5

    asyncio.run(_test())


def test_media_file_probe_falls_back_to_defaults_for_unreadable_file():
    from fast_float.livekit.generator import TARGET_FPS, TARGET_HEIGHT, TARGET_WIDTH

    gen = MediaFileVideoGenerator("does-not-exist.mp4")
    assert gen.probe() == (TARGET_WIDTH, TARGET_HEIGHT, TARGET_FPS)


def test_publish_status_endpoint_reports_publisher_error(api_client, monkeypatch):
    import fast_float.livekit.worker as worker

    monkeypatch.setattr(worker, "_publish_errors", {"req-status": "SFU unreachable"})

    res = api_client.get("/livekit/publish/req-status")
    assert res.status_code == 200
    assert res.json() == {
        "request_id": "req-status",
        "publishing": False,
        "error": "SFU unreachable",
    }

    ok = api_client.get("/livekit/publish/req-ok")
    assert ok.status_code == 200
    assert ok.json() == {"request_id": "req-ok", "publishing": False, "error": None}
