"""WebRTC transport tests: offer negotiation, live media, ICE and cleanup.

The trickle-ICE endpoint has its own focused file (``test_webrtc_trickle.py``);
this file covers the rest of the raw WebRTC path.  Most tests use a fake peer
connection so they stay fast; the two integration tests negotiate a real
aiortc answer against a real aiortc offer in the same event loop and pull
frames through the server-side tracks, i.e. they prove the transport actually
works without needing UDP connectivity.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fast_float.core.state import app_state
from fast_float.routers import webrtc as webrtc_router
from fast_float.streaming.live import CLOSE, LiveQueue
from fast_float.streaming.types import StreamMeta

try:
    import httpx
    from aiortc import RTCPeerConnection
    from aiortc.mediastreams import MediaStreamError
except ImportError:  # pragma: no cover - aiortc is a hard dependency of the transport
    httpx = None
    RTCPeerConnection = None
    MediaStreamError = None


needs_aiortc = pytest.mark.skipif(webrtc_router.AIORTC_AVAILABLE is False, reason="aiortc not installed")


@pytest.fixture(autouse=True)
def _clean_webrtc_state():
    """Every test starts and ends with no peers, players or live subscriptions."""
    state = (
        webrtc_router._pending_peers,
        webrtc_router._peer_ids,
        webrtc_router._players,
        webrtc_router._live_subs,
        webrtc_router._live_end_tasks,
    )
    saved_streams = dict(app_state.active_streams)
    saved_live = dict(app_state.live_streams)
    saved_pcs = set(app_state.pcs)
    for mapping in state:
        mapping.clear()
    app_state.active_streams.clear()
    app_state.live_streams.clear()
    app_state.pcs.clear()
    yield
    for mapping in state:
        mapping.clear()
    app_state.active_streams.clear()
    app_state.active_streams.update(saved_streams)
    app_state.live_streams.clear()
    app_state.live_streams.update(saved_live)
    app_state.pcs.clear()
    app_state.pcs.update(saved_pcs)


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(webrtc_router.router)
    return TestClient(app)


def _api() -> FastAPI:
    app = FastAPI()
    app.include_router(webrtc_router.router)
    return app


async def _post_offer(payload: dict):
    transport = httpx.ASGITransport(app=_api())
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as http:
        return await http.post("/offer", json=payload)


# --------------------------------------------------------------------------- #
# ICE configuration


class _SettingsStub:
    host: str | None = None

    @classmethod
    def from_env(cls):
        return cls

    @classmethod
    def resolved_ice_servers(cls, host):
        _SettingsStub.host = host
        return ({"urls": "stun:stun.example:3478"}, {"urls": "turn:turn.example:3478"})


def test_peer_configuration_prefers_forwarded_host(monkeypatch):
    monkeypatch.setattr(webrtc_router.LiveKitSettings, "from_env", _SettingsStub.from_env)
    request = type(
        "Req",
        (),
        {
            "headers": {"x-forwarded-host": "float.example.com:8000, proxy.internal", "host": "internal:8000"},
            "url": type("URL", (), {"hostname": "127.0.0.1"})(),
        },
    )()

    configuration = webrtc_router._peer_configuration(request)

    assert _SettingsStub.host == "float.example.com"
    assert [server.urls for server in configuration.iceServers] == [
        "stun:stun.example:3478",
        "turn:turn.example:3478",
    ]


def test_peer_configuration_falls_back_to_request_hostname(monkeypatch):
    monkeypatch.setattr(webrtc_router.LiveKitSettings, "from_env", _SettingsStub.from_env)
    request = type(
        "Req",
        (),
        {
            "headers": {},
            "url": type("URL", (), {"hostname": "192.0.2.10"})(),
        },
    )()

    webrtc_router._peer_configuration(request)

    assert _SettingsStub.host == "192.0.2.10"


def test_peer_configuration_without_http_request(monkeypatch):
    monkeypatch.setattr(webrtc_router.LiveKitSettings, "from_env", _SettingsStub.from_env)

    webrtc_router._peer_configuration(None)

    assert _SettingsStub.host is None


def test_peer_configuration_keeps_an_empty_ice_server_list(monkeypatch):
    """An empty list must not make aiortc fall back to public STUN servers."""

    class NoServers(_SettingsStub):
        @classmethod
        def resolved_ice_servers(cls, host):
            return ()

    monkeypatch.setattr(webrtc_router.LiveKitSettings, "from_env", NoServers.from_env)

    configuration = webrtc_router._peer_configuration(None)

    assert configuration.iceServers == []


def test_peer_configuration_is_none_without_aiortc(monkeypatch):
    monkeypatch.setattr(webrtc_router, "AIORTC_AVAILABLE", False)

    assert webrtc_router._peer_configuration(None) is None


# --------------------------------------------------------------------------- #
# Error paths


def test_offer_and_candidate_report_missing_aiortc(client, monkeypatch):
    monkeypatch.setattr(webrtc_router, "AIORTC_AVAILABLE", False)

    offer = client.post("/offer", json={"sdp": "v=0", "type": "offer"})
    candidate = client.post("/offer/candidate", json={"peer_id": "x", "candidate": None})

    assert offer.status_code == 501
    assert candidate.status_code == 501
    assert "aiortc" in offer.json()["detail"]


def test_offer_rejects_unknown_active_stream(client):
    response = client.post("/offer", json={"sdp": "v=0", "type": "offer", "stream_id": "missing"})

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_offer_rejects_missing_file_for_active_stream(client, tmp_path):
    missing = tmp_path / "gone.mp4"
    app_state.active_streams["req"] = str(missing)

    response = client.post("/offer", json={"sdp": "v=0", "type": "offer", "stream_id": "req"})

    assert response.status_code == 404
    assert "on disk" in response.json()["detail"]


def test_live_offer_requires_stream_id(client):
    response = client.post("/offer", json={"sdp": "v=0", "type": "offer", "live": True})

    assert response.status_code == 400
    assert "stream_id" in response.json()["detail"]


class _FakeLiveStream:
    def __init__(self, meta: StreamMeta | None = None, *, error: str | None = None, audio_path=None):
        self.meta = meta
        self.error = error
        self.audio_path = audio_path
        self.subscribed: list[LiveQueue] = []
        self.unsubscribed: list[LiveQueue] = []
        self.queues: list[LiveQueue] = []

    def subscribe(self, maxsize: int = 64) -> LiveQueue:
        queue = LiveQueue(maxsize=maxsize, request_id="fake")
        self.subscribed.append(queue)
        self.queues.append(queue)
        return queue

    def unsubscribe(self, queue: LiveQueue) -> None:
        # Mirrors LiveStream.unsubscribe: leaving twice is a no-op.
        if queue in self.subscribed:
            self.subscribed.remove(queue)
            self.unsubscribed.append(queue)

    async def wait_meta(self, timeout: float = 30.0):
        return self.meta


def test_live_offer_returns_404_when_stream_never_registers(monkeypatch):
    async def no_stream(request_id: str, timeout: float = 15.0):
        return None

    monkeypatch.setattr(app_state, "wait_for_live_stream", no_stream)

    response = asyncio.run(
        _post_offer({"sdp": "v=0", "type": "offer", "stream_id": "late", "live": True})
    )

    assert response.status_code == 404


def test_live_offer_returns_504_when_metadata_times_out():
    stream = _FakeLiveStream(meta=None)
    app_state.live_streams["req"] = stream

    response = asyncio.run(
        _post_offer({"sdp": "v=0", "type": "offer", "stream_id": "req", "live": True})
    )

    assert response.status_code == 504
    assert stream.unsubscribed == stream.queues
    assert stream.subscribed == []
    assert webrtc_router._pending_peers == {}


def test_live_offer_returns_502_when_the_stream_failed():
    stream = _FakeLiveStream(meta=None, error="render exploded")
    app_state.live_streams["req"] = stream

    response = asyncio.run(
        _post_offer({"sdp": "v=0", "type": "offer", "stream_id": "req", "live": True})
    )

    assert response.status_code == 502
    assert response.json()["detail"] == "render exploded"
    assert stream.unsubscribed == stream.queues
    assert stream.subscribed == []


class _FakePC:
    def __init__(self) -> None:
        self.tracks: list = []
        self.handlers: dict[str, object] = {}
        self.remote = None
        self.localDescription = None
        self.iceConnectionState = "new"
        self.connectionState = "new"
        self.closed = False
        self.candidates: list = []

    def on(self, event: str):
        def register(handler):
            self.handlers[event] = handler
            return handler

        return register

    def addTrack(self, track) -> None:
        self.tracks.append(track)

    async def setRemoteDescription(self, offer) -> None:
        self.remote = offer

    async def createAnswer(self):
        return type("Answer", (), {"sdp": "v=0\r\nm=video 9 UDP/TLS/RTP/SAVPF 96", "type": "answer"})()

    async def setLocalDescription(self, answer) -> None:
        self.localDescription = answer

    async def addIceCandidate(self, candidate) -> None:
        self.candidates.append(candidate)

    async def close(self) -> None:
        self.closed = True


def test_live_offer_cleans_up_when_negotiation_fails(monkeypatch):
    stream = _FakeLiveStream(meta=StreamMeta(fps=25.0, width=2, height=2, channels=3))
    app_state.live_streams["req"] = stream
    pc = _FakePC()

    async def boom(offer):
        raise RuntimeError("bad sdp")

    pc.setRemoteDescription = boom
    monkeypatch.setattr(webrtc_router, "_new_peer_connection", lambda request: pc)

    response = asyncio.run(
        _post_offer({"sdp": "v=0", "type": "offer", "stream_id": "req", "live": True})
    )

    assert response.status_code == 500
    assert pc.closed
    assert stream.unsubscribed == stream.queues
    assert stream.subscribed == []
    assert webrtc_router._live_subs == {}
    assert app_state.pcs == set()


def test_candidate_endpoint_strips_sdp_and_candidate_prefixes(client):
    pc = _FakePC()
    webrtc_router._pending_peers["peer"] = pc

    response = client.post(
        "/offer/candidate",
        json={
            "peer_id": "peer",
            "candidate": "a=candidate:1 1 UDP 2130706431 192.0.2.7 5000 typ host",
            "sdp_mid": "0",
            "sdp_mline_index": 0,
        },
    )

    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert len(pc.candidates) == 1
    candidate = pc.candidates[0]
    assert (candidate.ip, candidate.port, candidate.sdpMid) == ("192.0.2.7", 5000, "0")


def test_candidate_endpoint_reports_malformed_candidate_without_raising(client):
    pc = _FakePC()
    webrtc_router._pending_peers["peer"] = pc

    response = client.post(
        "/offer/candidate",
        json={"peer_id": "peer", "candidate": "not-a-candidate", "sdp_mid": "0"},
    )

    assert response.status_code == 200
    assert response.json() == {"ok": False}
    assert pc.candidates == []


# --------------------------------------------------------------------------- #
# Finished-file offers (fake peer, fake media player)


class _FakeTrack:
    kind = "video"

    def __init__(self) -> None:
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True


class _FakePlayer:
    instances: list[_FakePlayer] = []

    def __init__(self, path: str, loop: bool = False) -> None:
        self.path = path
        self.loop = loop
        self.video = _FakeTrack()
        self.audio = _FakeTrack()
        _FakePlayer.instances.append(self)


def test_file_offer_attaches_media_player_tracks(monkeypatch, tmp_path):
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"not really an mp4")
    app_state.active_streams["req"] = str(media)
    _FakePlayer.instances.clear()
    pc = _FakePC()
    monkeypatch.setattr(webrtc_router, "MediaPlayer", _FakePlayer)
    monkeypatch.setattr(webrtc_router, "_new_peer_connection", lambda request: pc)

    response = asyncio.run(
        _post_offer({"sdp": "v=0", "type": "offer", "stream_id": "req"})
    )

    assert response.status_code == 200
    body = response.json()
    assert body["type"] == "answer"
    assert body["peer_id"]
    assert len(pc.tracks) == 2
    player = _FakePlayer.instances[-1]
    assert player.path == str(media)
    assert player.loop is True
    assert webrtc_router._players[id(pc)] is player
    # Trickle candidates from the browser reach the same peer.
    candidate_pc = webrtc_router._pending_peers[body["peer_id"]]
    assert candidate_pc is pc

    webrtc_router._cleanup_peer(pc)
    assert player.video.stopped and player.audio.stopped
    assert webrtc_router._pending_peers == {}


def test_file_offer_rejects_media_without_playable_tracks(monkeypatch, tmp_path):
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"")
    app_state.active_streams["req"] = str(media)
    pc = _FakePC()

    class VideoOnlylessPlayer(_FakePlayer):
        def __init__(self, path: str, loop: bool = False) -> None:
            super().__init__(path, loop)
            self.video = None
            self.audio = None

    monkeypatch.setattr(webrtc_router, "MediaPlayer", VideoOnlylessPlayer)
    monkeypatch.setattr(webrtc_router, "_new_peer_connection", lambda request: pc)

    response = asyncio.run(
        _post_offer({"sdp": "v=0", "type": "offer", "stream_id": "req"})
    )

    assert response.status_code == 500
    assert "playable" in response.json()["detail"]
    assert pc.closed
    assert app_state.pcs == set()


def test_connection_state_failure_closes_and_unregisters_peer(monkeypatch, tmp_path):
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"x")
    app_state.active_streams["req"] = str(media)
    _FakePlayer.instances.clear()
    pc = _FakePC()
    monkeypatch.setattr(webrtc_router, "MediaPlayer", _FakePlayer)
    monkeypatch.setattr(webrtc_router, "_new_peer_connection", lambda request: pc)

    async def scenario():
        response = await _post_offer({"sdp": "v=0", "type": "offer", "stream_id": "req"})
        assert response.status_code == 200
        pc.connectionState = "failed"
        await pc.handlers["connectionstatechange"]()
        return response

    response = asyncio.run(scenario())
    body = response.json()
    assert pc.closed
    assert body["peer_id"] not in webrtc_router._pending_peers
    assert app_state.pcs == set()
    assert webrtc_router._players == {}


# --------------------------------------------------------------------------- #
# Live audio feed + tracks


def _pcm(*, samples: int = 16000, sample_rate: int = 16000) -> np.ndarray:
    return np.arange(samples, dtype=np.int16)


def test_live_audio_feed_slices_samples_in_video_order(monkeypatch):
    monkeypatch.setattr(webrtc_router, "load_pcm16_mono", lambda path, rate: _pcm())

    async def scenario():
        feed = webrtc_router.LiveAudioFeed("/tmp/request.wav", fps=25.0, sample_rate=16000)
        assert feed.available

        feed.publish(0)
        data, count = await feed.next_frame()
        assert count == 640
        assert np.frombuffer(data, dtype=np.int16).tolist() == list(range(640))

        # Frame 25 starts at sample 16000, i.e. past the end: padded silence.
        feed.publish(25)
        data, count = await feed.next_frame()
        assert count == 640
        assert data == bytes(640 * 2)

        feed.close()
        assert await feed.next_frame() is None
        # Publishing after close must not resurrect the queue.
        feed.publish(26)
        assert feed._queue.qsize() == 0

    asyncio.run(scenario())


def test_live_audio_feed_drops_the_oldest_mapping_on_overflow(monkeypatch):
    monkeypatch.setattr(webrtc_router, "load_pcm16_mono", lambda path, rate: _pcm(samples=16000 * 4))

    async def scenario():
        feed = webrtc_router.LiveAudioFeed("/tmp/request.wav", fps=25.0, sample_rate=16000)
        for idx in range(70):
            feed.publish(idx)

        # Queue maxsize is 64: the 6 oldest indices were dropped, the newest
        # 64 remain and are still delivered in order.
        got = []
        for _ in range(64):
            item = await feed.next_frame()
            assert item is not None
            assert len(item[0]) == 640 * 2
            got.append(item)
        assert feed._queue.empty()
        assert len(got) == 64

        feed.close()
        assert await feed.next_frame() is None

    asyncio.run(scenario())


def test_live_audio_feed_without_decodable_audio(monkeypatch):
    monkeypatch.setattr(webrtc_router, "load_pcm16_mono", lambda path, rate: None)

    async def scenario():
        feed = webrtc_router.LiveAudioFeed("/tmp/missing.wav", fps=25.0)
        assert not feed.available
        feed.publish(3)
        feed.close()
        assert await feed.next_frame() is None

    asyncio.run(scenario())


class _AudioFeedStub:
    def __init__(self) -> None:
        self.published: list[int] = []
        self.closed = False

    def publish(self, idx: int) -> None:
        self.published.append(idx)

    def close(self) -> None:
        self.closed = True


def test_live_video_track_converts_frames_and_publishes_audio_indices():
    async def scenario():
        queue = LiveQueue(maxsize=8)
        feed = _AudioFeedStub()
        meta = StreamMeta(fps=1000.0, width=2, height=2, channels=3, first_idx=0)
        track = webrtc_router.LiveVideoTrack(queue, meta, feed)

        frame_a = np.zeros((2, 2, 3), dtype=np.uint8)
        frame_a[:, :, 0] = 255
        queue.put_nowait(("frame", 0, 0.0, frame_a))
        queue.put_nowait(("gap", 1, 2, "dropped"))
        queue.put_nowait(("frame", 2, 2.0, np.zeros((2, 2, 4), dtype=np.uint8)))

        first = await asyncio.wait_for(track.recv(), 1.0)
        assert (first.width, first.height) == (2, 2)
        assert first.pts == 0
        assert first.time_base.denominator == 90000
        assert feed.published == [0]

        second = await asyncio.wait_for(track.recv(), 1.0)
        assert second.format.name == "rgba"
        assert second.pts == 180
        assert feed.published == [0, 2]

        metrics = track.metrics()
        assert metrics["delivered"] == 2
        assert metrics["dropped"] == 1

    asyncio.run(scenario())


def test_live_video_track_signals_end_and_closes_the_audio_feed():
    async def scenario():
        queue = LiveQueue(maxsize=8)
        feed = _AudioFeedStub()
        track = webrtc_router.LiveVideoTrack(queue, StreamMeta(fps=25.0, width=2, height=2), feed)
        queue.put_nowait(CLOSE)

        with pytest.raises(MediaStreamError):
            await asyncio.wait_for(track.recv(), 1.0)

        assert track.ended.is_set()
        assert feed.closed
        with pytest.raises(MediaStreamError):
            await track.recv()

    asyncio.run(scenario())


def test_live_video_track_stop_unblocks_recv():
    async def scenario():
        queue = LiveQueue(maxsize=8)
        feed = _AudioFeedStub()
        track = webrtc_router.LiveVideoTrack(queue, StreamMeta(fps=25.0, width=2, height=2), feed)

        track.stop()

        assert feed.closed
        with pytest.raises(MediaStreamError):
            await asyncio.wait_for(track.recv(), 1.0)

    asyncio.run(scenario())


# --------------------------------------------------------------------------- #
# Peer registry + cleanup


def test_register_peer_is_unique_and_maps_both_ways():
    first = _FakePC()
    second = _FakePC()

    first_id = webrtc_router._register_peer(first)
    second_id = webrtc_router._register_peer(second)

    assert first_id != second_id
    assert webrtc_router._pending_peers[first_id] is first
    assert webrtc_router._peer_ids[id(first)] == first_id
    assert webrtc_router._pending_peers[second_id] is second


def test_cleanup_peer_stops_everything_it_owns():
    async def scenario():
        pc = _FakePC()
        player = _FakePlayer("/tmp/clip.mp4")
        queue = LiveQueue(maxsize=4)
        queue.put_nowait(("gap", 0, 1, "x"))
        track = webrtc_router.LiveVideoTrack(queue, StreamMeta(fps=25.0, width=2, height=2), _AudioFeedStub())

        class Stream:
            def __init__(self) -> None:
                self.unsubscribed: list[LiveQueue] = []

            def unsubscribe(self, target: LiveQueue) -> None:
                self.unsubscribed.append(target)

        stream = Stream()
        feed = _AudioFeedStub()
        end_task = asyncio.create_task(asyncio.sleep(60))

        app_state.pcs.add(pc)
        webrtc_router._pending_peers["peer"] = pc
        webrtc_router._peer_ids[id(pc)] = "peer"
        webrtc_router._players[id(pc)] = player
        webrtc_router._live_subs[id(pc)] = (stream, queue, track, feed)
        webrtc_router._live_end_tasks[id(pc)] = end_task

        webrtc_router._cleanup_peer(pc)
        await asyncio.sleep(0)

        assert webrtc_router._pending_peers == {}
        assert webrtc_router._peer_ids == {}
        assert webrtc_router._players == {}
        assert webrtc_router._live_subs == {}
        assert webrtc_router._live_end_tasks == {}
        assert app_state.pcs == set()
        assert player.video.stopped and player.audio.stopped
        assert stream.unsubscribed == [queue]
        assert feed.closed
        assert track._stopped
        assert end_task.cancelled()

    asyncio.run(scenario())


def test_cleanup_peer_is_idempotent():
    pc = _FakePC()
    app_state.pcs.add(pc)
    webrtc_router._register_peer(pc)

    webrtc_router._cleanup_peer(pc)
    webrtc_router._cleanup_peer(pc)

    assert webrtc_router._pending_peers == {}
    assert app_state.pcs == set()


# --------------------------------------------------------------------------- #
# Real aiortc negotiation (offer -> answer + server-side media drain)


async def _client_offer_sdp() -> str:
    pc = RTCPeerConnection()
    pc.addTransceiver("video", direction="recvonly")
    pc.addTransceiver("audio", direction="recvonly")
    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)
    sdp = pc.localDescription.sdp
    await pc.close()
    return sdp


@needs_aiortc
def test_real_aiortc_live_offer_answers_and_delivers_frames(monkeypatch, tmp_path):
    """Negotiate a real answer and pull one video + one audio frame through it."""
    audio = tmp_path / "request.wav"
    audio.write_bytes(b"placeholder")
    monkeypatch.setattr(
        webrtc_router,
        "load_pcm16_mono",
        lambda path, rate: _pcm(samples=16000),
    )
    stream = _FakeLiveStream(
        meta=StreamMeta(fps=25.0, width=4, height=3, channels=3, request_id="live-1"),
        audio_path=str(audio),
    )
    app_state.live_streams["live-1"] = stream

    async def scenario():
        offer_sdp = await _client_offer_sdp()
        response = await _post_offer(
            {"sdp": offer_sdp, "type": "offer", "stream_id": "live-1", "live": True}
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["type"] == "answer"
        assert body["sdp"].count("m=video") == 1
        assert body["sdp"].count("m=audio") == 1
        assert body["peer_id"] in webrtc_router._pending_peers

        pc = webrtc_router._pending_peers[body["peer_id"]]
        subscribed_stream, queue, video_track, audio_feed = webrtc_router._live_subs[id(pc)]
        assert subscribed_stream is stream
        assert audio_feed is not None
        assert [sender.kind for sender in pc.getSenders()] == ["video", "audio"]

        frame = np.full((3, 4, 3), 200, dtype=np.uint8)
        queue.put_nowait(("frame", 0, 0.0, frame))
        delivered = await asyncio.wait_for(video_track.recv(), 1.0)
        assert (delivered.width, delivered.height) == (4, 3)
        assert delivered.pts == 0

        audio_sender = next(sender for sender in pc.getSenders() if sender.kind == "audio")
        audio_frame = await asyncio.wait_for(audio_sender.track.recv(), 1.0)
        assert audio_frame.samples == 640
        assert audio_frame.sample_rate == 16000

        # End of stream: the track raises, the watcher closes and unregisters
        # the peer so the browser's <video> sees a clean end.
        queue.put_nowait(CLOSE)
        with pytest.raises(MediaStreamError):
            await asyncio.wait_for(video_track.recv(), 1.0)
        await asyncio.sleep(0)
        for _ in range(20):
            if body["peer_id"] not in webrtc_router._pending_peers:
                break
            await asyncio.sleep(0.01)

        assert body["peer_id"] not in webrtc_router._pending_peers
        assert webrtc_router._live_subs == {}
        assert app_state.pcs == set()

    asyncio.run(scenario())


@needs_aiortc
def test_real_aiortc_offer_without_stream_answers_without_tracks():
    async def scenario():
        offer_sdp = await _client_offer_sdp()
        response = await _post_offer({"sdp": offer_sdp, "type": "offer"})
        assert response.status_code == 200, response.text
        body = response.json()

        pc = webrtc_router._pending_peers[body["peer_id"]]
        # The answer still mirrors both recvonly m-lines, but sends no media.
        assert all(sender.track is None for sender in pc.getSenders())
        assert body["sdp"].count("m=video") == 1
        assert body["sdp"].count("m=audio") == 1

        await pc.close()
        webrtc_router._cleanup_peer(pc)

    asyncio.run(scenario())


@needs_aiortc
def test_real_aiortc_peer_closes_cleanly_when_the_stream_ends():
    """A finished stream ends the track and unregisters the peer."""
    stream = _FakeLiveStream(meta=StreamMeta(fps=25.0, width=2, height=2, channels=3))
    app_state.live_streams["done"] = stream

    async def scenario():
        offer_sdp = await _client_offer_sdp()
        response = await _post_offer(
            {"sdp": offer_sdp, "type": "offer", "stream_id": "done", "live": True}
        )
        assert response.status_code == 200, response.text
        body = response.json()
        pc = webrtc_router._pending_peers[body["peer_id"]]
        _, queue, video_track, audio_feed = webrtc_router._live_subs[id(pc)]
        # No audio path on this stream: video only, no silent audio track.
        assert audio_feed is None
        assert [sender.kind for sender in pc.getSenders() if sender.track is not None] == ["video"]

        queue.put_nowait(CLOSE)
        with pytest.raises(MediaStreamError):
            await asyncio.wait_for(video_track.recv(), 1.0)

        # The end watcher closes the peer shortly after the track ends.
        for _ in range(20):
            if body["peer_id"] not in webrtc_router._pending_peers:
                break
            await asyncio.sleep(0.01)

        assert body["peer_id"] not in webrtc_router._pending_peers
        assert webrtc_router._live_subs == {}
        assert app_state.pcs == set()

    asyncio.run(scenario())
