"""Unit tests for the live fan-out hub and its dispatcher."""

from __future__ import annotations

import asyncio

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fast_float.core.state import app_state
from fast_float.routers.streaming import HEADER
from fast_float.routers.streaming import router as streaming_router
from fast_float.streaming.live import CLOSE, LiveStream, LiveStreamDispatcher
from fast_float.streaming.types import StreamMeta


def test_live_stream_fanout_and_finish():
    async def scenario():
        loop = asyncio.get_running_loop()
        stream = LiveStream("req-fanout", loop)
        stream.start(StreamMeta(fps=25.0, width=2, height=2, channels=3, request_id="req-fanout"))

        a = stream.subscribe()
        b = stream.subscribe()
        stream.publish(("gap", 0, 2, "dropped"))
        assert (await asyncio.wait_for(a.get(), 1.0)) == ("gap", 0, 2, "dropped")
        assert (await asyncio.wait_for(b.get(), 1.0)) == ("gap", 0, 2, "dropped")

        stream.unsubscribe(b)
        stream.finish(result={"status": "success"})

        assert (await asyncio.wait_for(a.get(), 1.0)) is CLOSE
        assert stream.done
        assert stream.result == {"status": "success"}

        late = stream.subscribe()
        assert (await asyncio.wait_for(late.get(), 1.0)) is CLOSE
        assert b.empty()

    asyncio.run(scenario())


def test_finish_queues_close_after_pending_frames():
    async def scenario():
        loop = asyncio.get_running_loop()
        stream = LiveStream("req-done", loop)
        q = stream.subscribe()
        # A frame published (from the delivery thread) just before finish()
        # must still reach the client ahead of the CLOSE sentinel.
        loop.call_soon_threadsafe(stream.publish, ("frame", 0, 0.0, np.zeros((2, 2, 3), np.uint8)))
        stream.finish(result={"status": "success"})

        first = await asyncio.wait_for(q.get(), 1.0)
        assert first[0] == "frame"
        assert (await asyncio.wait_for(q.get(), 1.0)) is CLOSE
        assert q.empty()
        assert stream.done
        assert stream.result == {"status": "success"}

    asyncio.run(scenario())


def test_live_dispatcher_publishes_frames_with_pts():
    async def scenario():
        loop = asyncio.get_running_loop()
        dispatcher = LiveStreamDispatcher("req-frames", loop)
        dispatcher.set_audio_path("/tmp/request.wav")
        dispatcher.bind(StreamMeta(fps=25.0, width=2, height=2, channels=3))

        meta = await dispatcher.stream.wait_meta(1.0)
        assert meta is not None
        assert meta.audio_path == "/tmp/request.wav"

        q = dispatcher.stream.subscribe()
        dispatcher.submit(np.zeros((3, 2, 2, 3), dtype=np.uint8))

        items = [await asyncio.wait_for(q.get(), 2.0) for _ in range(3)]
        assert [item[1] for item in items] == [0, 1, 2]
        assert [item[2] for item in items] == [0.0, 40.0, 80.0]
        assert items[0][3].shape == (2, 2, 3)

        dispatcher.shutdown()
        dispatcher.stream.finish()
        assert (await asyncio.wait_for(q.get(), 2.0)) is CLOSE

    asyncio.run(scenario())


def test_live_dispatcher_trims_overlapping_chunk_head():
    async def scenario():
        loop = asyncio.get_running_loop()
        dispatcher = LiveStreamDispatcher("req-trim", loop)
        dispatcher.bind(StreamMeta(fps=25.0, width=2, height=2, channels=3))
        q = dispatcher.stream.subscribe()

        dispatcher.begin_chunk(0)
        dispatcher.submit(np.zeros((4, 2, 2, 3), dtype=np.uint8))
        # The next chunk overlaps two frames with the previous one; those
        # frames are already on the wire and must not be published twice.
        dispatcher.begin_chunk(2)
        dispatcher.submit(np.zeros((5, 2, 2, 3), dtype=np.uint8))

        items = [await asyncio.wait_for(q.get(), 2.0) for _ in range(7)]
        assert [item[1] for item in items] == [0, 1, 2, 3, 4, 5, 6]
        assert dispatcher.stats()["submitted"] == 7

        dispatcher.shutdown()

    asyncio.run(scenario())


def test_live_dispatcher_configuration():
    async def scenario():
        loop = asyncio.get_running_loop()
        dispatcher = LiveStreamDispatcher("req-mode", loop)
        assert dispatcher.dispatch_granularity == "frame"
        assert dispatcher.mode == "live"
        assert dispatcher.max_queue_frames > 0

    asyncio.run(scenario())


def test_frame_header_layout():
    assert HEADER.size == 12
    assert HEADER.pack(7, 1, 1234) == b"\x00\x00\x00\x07\x00\x00\x00\x01\x00\x00\x04\xd2"
    idx, count, pts_ms = HEADER.unpack(HEADER.pack(123456, 3, 999))
    assert (idx, count, pts_ms) == (123456, 3, 999)


def test_wait_for_live_stream_times_out():
    async def scenario():
        assert await app_state.wait_for_live_stream("missing-request", timeout=0.2) is None

    asyncio.run(scenario())


def test_wait_for_live_stream_returns_registered_stream():
    async def scenario():
        loop = asyncio.get_running_loop()
        stream = LiveStream("req-registered", loop)
        app_state.live_streams["req-registered"] = stream
        try:
            found = await app_state.wait_for_live_stream("req-registered", timeout=0.2)
            assert found is stream
        finally:
            app_state.live_streams.pop("req-registered", None)

    asyncio.run(scenario())


class _FakeStream:
    """Minimal LiveStream stand-in that produces one frame on the app loop."""

    def __init__(self, meta, frame):
        self.meta = meta
        self._frame = frame
        self.result = {"status": "success"}
        self.error = None
        self.unsubscribed = False

    async def wait_meta(self, timeout: float = 30.0):
        return self.meta

    def subscribe(self, maxsize: int = 64):
        queue = asyncio.Queue(maxsize=maxsize)
        frame = self._frame

        async def produce():
            await asyncio.sleep(0.02)
            queue.put_nowait(("frame", 0, 0.0, frame))
            queue.put_nowait(CLOSE)

        asyncio.get_running_loop().create_task(produce())
        return queue

    def unsubscribe(self, queue):
        self.unsubscribed = True


def test_ws_stream_frames_done_and_unsubscribe():
    app = FastAPI()
    app.include_router(streaming_router)
    meta = StreamMeta(fps=25.0, width=2, height=2, channels=3, request_id="ws-test")
    frame = np.full((2, 2, 3), 7, dtype=np.uint8)
    stream = _FakeStream(meta, frame)
    app_state.live_streams["ws-test"] = stream
    try:
        with TestClient(app) as client, client.websocket_connect("/ws/stream/ws-test") as ws:
            payload = ws.receive_json()
            assert payload["type"] == "meta"
            assert (payload["width"], payload["height"], payload["channels"]) == (2, 2, 3)

            data = ws.receive_bytes()
            idx, count, pts_ms = HEADER.unpack(data[:12])
            assert (idx, count, pts_ms) == (0, 1, 0)
            assert data[12:] == frame.tobytes()

            done = ws.receive_json()
            assert done == {"type": "done", "result": {"status": "success"}}
        assert stream.unsubscribed
    finally:
        app_state.live_streams.pop("ws-test", None)


def test_ws_stream_unknown_request_reports_error(monkeypatch):
    import fast_float.routers.streaming as streaming

    monkeypatch.setattr(streaming, "WAIT_TIMEOUT_S", 0.1)
    app = FastAPI()
    app.include_router(streaming_router)
    with TestClient(app) as client, client.websocket_connect("/ws/stream/does-not-exist") as ws:
        payload = ws.receive_json()
        assert payload["type"] == "error"


def test_ws_stream_end_to_end_with_dispatcher():
    """The real dispatcher (delivery thread) must wake a waiting socket.

    Regression guard for publishing metadata/frames across threads.
    """
    app = FastAPI()
    app.include_router(streaming_router)
    state = {"dispatcher": None}

    @app.post("/create")
    async def create():
        dispatcher = LiveStreamDispatcher("e2e", asyncio.get_running_loop())
        state["dispatcher"] = dispatcher
        app_state.live_streams["e2e"] = dispatcher.stream
        return {"ok": True}

    @app.post("/send")
    async def send():
        dispatcher = state["dispatcher"]
        stream = app_state.live_streams["e2e"]
        for _ in range(200):
            if stream.subscriber_count:
                break
            await asyncio.sleep(0.01)
        dispatcher.bind(StreamMeta(fps=25.0, width=2, height=2, channels=3))
        dispatcher.submit(np.full((3, 2, 2, 3), 9, dtype=np.uint8))
        await asyncio.to_thread(dispatcher.shutdown)
        stream.finish(result={"status": "success"})
        return {"ok": True}

    try:
        with TestClient(app) as client:
            assert client.post("/create").status_code == 200
            with client.websocket_connect("/ws/stream/e2e") as ws:
                assert client.post("/send").status_code == 200

                meta = ws.receive_json()
                assert (meta["type"], meta["width"], meta["height"]) == ("meta", 2, 2)

                received = [ws.receive_bytes() for _ in range(3)]
                assert [HEADER.unpack(data[:12])[0] for data in received] == [0, 1, 2]
                assert [HEADER.unpack(data[:12])[2] for data in received] == [0, 40, 80]
                assert all(data[12:] == bytes([9] * 12) for data in received)

                done = ws.receive_json()
                assert done["type"] == "done"
    finally:
        app_state.live_streams.pop("e2e", None)
