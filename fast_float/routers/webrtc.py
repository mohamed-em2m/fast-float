import asyncio
import os
import time
from contextlib import suppress
from fractions import Fraction
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request

from fast_float.core.perf_metrics import FrameClock
from fast_float.core.state import app_state
from fast_float.livekit.settings import LiveKitSettings, host_only
from fast_float.schemas.generation import (
    WebRTCIceCandidateRequest,
    WebRTCOfferRequest,
    WebRTCOfferResponse,
)
from fast_float.streaming.live import CLOSE, live_queue_size
from fast_float.utils.audio import load_pcm16_mono, slice_audio_for_frame
from fast_float.utils.logger import logger

try:
    import av
    import numpy as np
    from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCSessionDescription
    from aiortc.contrib.media import MediaPlayer
    from aiortc.mediastreams import MediaStreamError, MediaStreamTrack
    from aiortc.sdp import candidate_from_sdp

    AIORTC_AVAILABLE = True
except ImportError:
    AIORTC_AVAILABLE = False

router = APIRouter(tags=["webrtc"])

# MediaPlayer instances keyed by id(pc). A player spawns an ffmpeg subprocess,
# so it must be stopped when its peer connection goes away.
_players: dict[int, Any] = {}
# Live subscriptions keyed by id(pc): (stream, queue).
_live_subs: dict[int, Any] = {}
# Watcher that closes a live peer when its stream finishes, keyed by id(pc).
_live_end_tasks: dict[int, asyncio.Task] = {}
# Trickle-ICE signalling: the browser gets an opaque peer id with the answer
# and POSTs candidates discovered after the offer to /offer/candidate.
_pending_peers: dict[str, Any] = {}
_peer_ids: dict[int, str] = {}

LIVE_META_TIMEOUT_S = 30.0


def _peer_configuration(http_request: Request | None) -> "RTCConfiguration | None":
    """ICE config for aiortc: the API host's own STUN responder.

    aiortc only gathers host candidates by default, which a browser on another
    network cannot reach. Pointing it at the deployment's STUN endpoint lets it
    discover the public reflexive candidate as well — no LiveKit/TURN needed.
    An explicit ``[webrtc].stun_urls`` list overrides the built-in responder.
    """
    if not AIORTC_AVAILABLE:
        return None
    settings = LiveKitSettings.from_env()
    if http_request is not None:
        host = (
            host_only(http_request.headers.get("x-forwarded-host"))
            or host_only(http_request.headers.get("host"))
            or http_request.url.hostname
        )
    else:
        host = None
    servers = settings.resolved_ice_servers(host)
    # An empty list is deliberate: it stops aiortc from silently falling back
    # to its built-in Google/Twilio STUN servers when this deployment disabled
    # STUN. Host candidates still work on a LAN; a TURN entry is required when
    # the browser reaches this API through a TCP-only proxy.
    return RTCConfiguration(
        iceServers=[
            RTCIceServer(
                urls=server["urls"],
                username=server.get("username"),
                credential=server.get("credential"),
            )
            for server in servers
        ]
    )


def _new_peer_connection(http_request: Request | None) -> "RTCPeerConnection":
    configuration = _peer_configuration(http_request)
    if configuration is None:
        return RTCPeerConnection()
    logger.debug(
        "WebRTC ICE servers: "
        f"{[(server.urls, bool(server.username)) for server in configuration.iceServers or []]}"
    )
    return RTCPeerConnection(configuration)


if AIORTC_AVAILABLE:

    class LiveAudioFeed:
        """Audio slices paired one-to-one with the video frames actually sent.

        A ``MediaPlayer`` starts at t=0 when the peer connects, so a viewer
        joining a render already in progress heard the beginning of the request
        while watching the middle of the video.  The video track hands each
        delivered frame's index to this feed and the audio track slices the
        request audio at exactly that index, so neither side can run ahead.
        """

        def __init__(self, audio_path: str | None, fps: float, sample_rate: int = 16000) -> None:
            self.sample_rate = int(sample_rate)
            self._fps = max(1.0, float(fps or 25.0))
            self._audio = load_pcm16_mono(audio_path, self.sample_rate)
            self._queue: asyncio.Queue = asyncio.Queue(maxsize=64)
            self._closed = False

        @property
        def available(self) -> bool:
            return self._audio is not None

        def publish(self, idx: int) -> None:
            """Record the index of a video frame that was just handed to RTP."""
            if self._closed or self._audio is None:
                return
            try:
                self._queue.put_nowait(int(idx))
            except asyncio.QueueFull:
                # The audio sender fell behind: keep the newest mapping instead
                # of letting audio drift further behind the video.
                with suppress(asyncio.QueueEmpty):
                    self._queue.get_nowait()
                with suppress(asyncio.QueueFull):
                    self._queue.put_nowait(int(idx))

        def close(self) -> None:
            if self._closed:
                return
            self._closed = True
            try:
                self._queue.put_nowait(None)
            except asyncio.QueueFull:
                # Make room for the sentinel: the stream is over, so dropping
                # one queued frame is better than leaving the audio track
                # blocked on a queue that will never be filled again.
                with suppress(asyncio.QueueEmpty):
                    self._queue.get_nowait()
                with suppress(asyncio.QueueFull):
                    self._queue.put_nowait(None)

        async def next_frame(self) -> tuple[bytes, int] | None:
            idx = await self._queue.get()
            if idx is None:
                return None
            return slice_audio_for_frame(self._audio, idx, self._fps, self.sample_rate)

    class LiveAudioTrack(MediaStreamTrack):
        """Audio track fed by :class:`LiveAudioFeed` in lockstep with the video."""

        kind = "audio"

        def __init__(self, feed: LiveAudioFeed) -> None:
            super().__init__()
            self._feed = feed
            self._pts = 0

        async def recv(self) -> "av.AudioFrame":
            item = await self._feed.next_frame()
            if item is None:
                raise MediaStreamError("stream finished")
            data, samples = item
            frame = av.AudioFrame(format="s16", layout="mono", samples=samples)
            frame.planes[0].update(data)
            frame.sample_rate = self._feed.sample_rate
            frame.pts = self._pts
            frame.time_base = Fraction(1, self._feed.sample_rate)
            # A continuous sample clock: audio content follows the video's
            # index, but RTP timestamps never jump when a frame was dropped.
            self._pts += samples
            return frame

    class LiveVideoTrack(MediaStreamTrack):
        """Video track fed by a LiveStream subscription queue."""

        kind = "video"

        def __init__(self, queue: asyncio.Queue, meta: Any, audio_feed: LiveAudioFeed | None = None) -> None:
            super().__init__()
            self._queue = queue
            self._audio_feed = audio_feed
            self._fps = max(1.0, float(meta.fps or 25.0))
            self._next_at: float | None = None
            self._stopped = False
            # Set when the publisher's stream finished (not when the peer is
            # torn down) so the offer can close the peer connection and the
            # browser sees a clean end instead of a frozen last frame.
            self.ended = asyncio.Event()
            # Consumer-side clock: how late each frame reached the WebRTC
            # encoder (TTFF, latency percentiles, pacing jitter, drops).
            started = getattr(meta, "started_at_s", None)
            self.clock = FrameClock(
                fps=self._fps,
                start_s=float(started) if started is not None else time.monotonic(),
                first_idx=int(getattr(meta, "first_idx", 0) or 0),
            )

        def metrics(self) -> dict:
            """Delivery metrics of this track (TTFF, p50/p95/p99, jitter, drops)."""
            return self.clock.to_dict()

        def stop(self) -> None:
            self._stopped = True
            if self._audio_feed is not None:
                self._audio_feed.close()
            super().stop()

        async def recv(self) -> "av.VideoFrame":
            while True:
                if self._stopped:
                    raise MediaStreamError("track stopped")
                item = await self._queue.get()
                if item is CLOSE:
                    self.ended.set()
                    self.stop()
                    raise MediaStreamError("stream finished")
                if item[0] != "frame":
                    continue
                _, idx, pts_ms, frame_array = item
                now = time.monotonic()
                if self._next_at is None:
                    self._next_at = now
                delay = self._next_at - now
                if delay > 0:
                    await asyncio.sleep(delay)
                self._next_at = max(self._next_at + 1.0 / self._fps, time.monotonic())
                self.clock.on_frame(idx, pts_ms, time.monotonic())
                if self._audio_feed is not None:
                    self._audio_feed.publish(idx)

                frame = av.VideoFrame.from_ndarray(
                    np.ascontiguousarray(frame_array),
                    format="rgba" if frame_array.shape[-1] == 4 else "rgb24",
                )
                frame.pts = round(float(pts_ms) * 90)
                frame.time_base = Fraction(1, 90000)
                return frame


def _register_peer(pc) -> str:
    """Expose a peer to the trickle-ICE endpoint and return its opaque id."""
    peer_id = uuid4().hex
    _pending_peers[peer_id] = pc
    _peer_ids[id(pc)] = peer_id
    return peer_id


def _cleanup_peer(pc) -> None:
    """Stop the attached media player, unblock live tracks, drop the peer."""
    peer_id = _peer_ids.pop(id(pc), None)
    if peer_id:
        _pending_peers.pop(peer_id, None)
    player = _players.pop(id(pc), None)
    if player is not None:
        try:
            if player.video is not None:
                player.video.stop()
        except Exception:
            pass
        try:
            if player.audio is not None:
                player.audio.stop()
        except Exception:
            pass
    sub = _live_subs.pop(id(pc), None)
    if sub is not None:
        stream, queue, track, audio_feed = sub
        stream.unsubscribe(queue)
        with suppress(RuntimeError):
            track.stop()
        if audio_feed is not None:
            audio_feed.close()
        # Wake a recv() blocked on the queue and release its overflow spool.
        queue.dispose()
    end_task = _live_end_tasks.pop(id(pc), None)
    if end_task is not None and end_task is not asyncio.current_task():
        end_task.cancel()
    app_state.pcs.discard(pc)


async def _live_offer(request: WebRTCOfferRequest, http_request: Request) -> WebRTCOfferResponse:
    """Attach the in-flight live stream (video + request audio) to a new peer."""
    if not request.stream_id:
        raise HTTPException(status_code=400, detail="live offers require a stream_id")

    stream = await app_state.wait_for_live_stream(request.stream_id)
    if stream is None:
        raise HTTPException(status_code=404, detail="Requested live stream not found")

    # Subscribe before waiting for meta so the first decoded frames are queued.
    # A fast render produces far more frames than real-time playout consumes;
    # buffer them (bounded by ``live_queue_size``) so the preview plays the
    # whole clip instead of freezing after the first ~8 frames.
    queue = stream.subscribe(maxsize=live_queue_size(stream.meta))
    try:
        meta = await stream.wait_meta(LIVE_META_TIMEOUT_S)
        if meta is None:
            raise HTTPException(
                status_code=502 if stream.error else 504,
                detail=stream.error or "Live stream did not start in time",
            )
        # Meta arrived after the provisional subscription: resize now that the
        # frame dimensions are known.
        queue.resize(live_queue_size(meta))

        offer = RTCSessionDescription(sdp=request.sdp, type=request.type)
        pc = _new_peer_connection(http_request)
        app_state.pcs.add(pc)
        # The request audio is sliced in lockstep with the video track below.
        # A MediaPlayer would start at t=0 even when the viewer joined in the
        # middle of a render, putting audio and video seconds apart.
        audio_feed = None
        if stream.audio_path and os.path.exists(stream.audio_path):
            feed = LiveAudioFeed(stream.audio_path, meta.fps)
            if feed.available:
                audio_feed = feed
        # Register the subscription before addTrack so any failure below is
        # cleaned up by _cleanup_peer (which unsubscribes it).
        track = LiveVideoTrack(queue, meta, audio_feed)
        _live_subs[id(pc)] = (stream, queue, track, audio_feed)

        @pc.on("iceconnectionstatechange")
        async def on_iceconnectionstatechange():
            logger.info(f"WebRTC live ICE state: {pc.iceConnectionState}")

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            logger.info(f"WebRTC live connection state: {pc.connectionState}")
            if pc.connectionState in ("failed", "closed", "disconnected"):
                try:
                    await pc.close()
                finally:
                    _cleanup_peer(pc)

        try:
            pc.addTrack(track)

            # When the render finishes, the track raises MediaStreamError and
            # ends; close the peer so the viewer's <video> reports the end
            # instead of freezing on the last frame forever.
            async def _close_when_stream_ends() -> None:
                await track.ended.wait()
                logger.info(f"Live WebRTC stream {request.stream_id} finished; closing peer")
                with suppress(Exception):
                    await pc.close()
                _cleanup_peer(pc)

            _live_end_tasks[id(pc)] = asyncio.create_task(_close_when_stream_ends())

            if audio_feed is not None:
                pc.addTrack(LiveAudioTrack(audio_feed))

            await pc.setRemoteDescription(offer)
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)
        except Exception as exc:
            logger.warning(f"Live WebRTC negotiation failed: {exc}")
            try:
                await pc.close()
            finally:
                _cleanup_peer(pc)
            raise HTTPException(status_code=500, detail="Live WebRTC negotiation failed") from exc

        logger.info(f"Live WebRTC peer attached for {request.stream_id}")
        return WebRTCOfferResponse(
            sdp=pc.localDescription.sdp,
            type=pc.localDescription.type,
            peer_id=_register_peer(pc),
        )
    except BaseException:
        # Cancelled/timed out before a peer owned the subscription; _cleanup_peer
        # has already unsubscribed if a peer was created.  discard() is idempotent.
        stream.unsubscribe(queue)
        raise


@router.post("/offer", response_model=WebRTCOfferResponse)
async def webrtc_offer(request: WebRTCOfferRequest, http_request: Request):
    """Handle WebRTC SDP offer and establish peer connection for stream playback."""
    if not AIORTC_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="aiortc is not installed in the current environment",
        )

    if request.live:
        return await _live_offer(request, http_request)

    if request.stream_id is not None and request.stream_id not in app_state.active_streams:
        raise HTTPException(
            status_code=404,
            detail="Requested stream ID not found",
        )

    offer = RTCSessionDescription(sdp=request.sdp, type=request.type)
    pc = _new_peer_connection(http_request)
    app_state.pcs.add(pc)

    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        logger.info(f"WebRTC ICE state: {pc.iceConnectionState}")

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info(f"WebRTC connection state: {pc.connectionState}")
        if pc.connectionState in ("failed", "closed", "disconnected"):
            try:
                await pc.close()
            finally:
                _cleanup_peer(pc)

    # If stream_id exists in active streams, attach its media player tracks
    if request.stream_id and request.stream_id in app_state.active_streams:
        video_path = app_state.active_streams[request.stream_id]
        if os.path.exists(video_path):
            try:
                player = MediaPlayer(video_path, loop=True)
                if player.video:
                    pc.addTrack(player.video)
                if player.audio:
                    pc.addTrack(player.audio)
                _players[id(pc)] = player
                if player.video is None and player.audio is None:
                    logger.warning(f"Media has no playable tracks for {request.stream_id}")
                    try:
                        await pc.close()
                    finally:
                        _cleanup_peer(pc)
                    raise HTTPException(
                        status_code=500,
                        detail="Media file has no playable video/audio tracks",
                    )
                logger.info(f"Attached MediaPlayer tracks for {request.stream_id}")
            except HTTPException:
                # A player with no tracks was already cleaned up and raised a
                # specific 500; do not replace it with the generic message.
                raise
            except Exception as e:
                logger.warning(f"Failed to attach media player for {request.stream_id}: {e}")
                try:
                    await pc.close()
                finally:
                    _cleanup_peer(pc)
                raise HTTPException(
                    status_code=500,
                    detail="Failed to prepare media for WebRTC stream",
                ) from e
        else:
            await pc.close()
            app_state.pcs.discard(pc)
            raise HTTPException(
                status_code=404,
                detail="Video file on disk not found",
            )

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return WebRTCOfferResponse(
        sdp=pc.localDescription.sdp,
        type=pc.localDescription.type,
        peer_id=_register_peer(pc),
    )


@router.post("/offer/candidate")
async def webrtc_offer_candidate(request: WebRTCIceCandidateRequest) -> dict[str, bool]:
    """Accept a late ICE candidate from the browser (trickle ICE).

    ``/offer`` answers with server-side candidates, but browsers gather their
    own candidates asynchronously. Without this endpoint a slow TURN allocation
    or a missed host candidate would leave ICE stuck in ``checking``.
    """
    if not AIORTC_AVAILABLE:
        raise HTTPException(status_code=501, detail="aiortc is not installed")

    pc = _pending_peers.get(request.peer_id)
    if pc is None:
        raise HTTPException(status_code=404, detail="Peer connection not found or already closed")

    if request.candidate:
        try:
            sdp = request.candidate.strip()
            if sdp.startswith("a="):
                sdp = sdp[2:]
            if sdp.startswith("candidate:"):
                sdp = sdp[len("candidate:") :]
            candidate = candidate_from_sdp(sdp)
            candidate.sdpMid = request.sdp_mid
            candidate.sdpMLineIndex = request.sdp_mline_index
            await pc.addIceCandidate(candidate)
        except Exception as exc:
            # A candidate for a closed/renamed m-line is normal during teardown;
            # do not make the browser retry a whole negotiation for it.
            logger.debug(f"WebRTC ignored ICE candidate for {request.peer_id}: {exc}")
            return {"ok": False}
    else:
        # End-of-candidates: lets aioice prune incomplete component pairs.
        try:
            await pc.addIceCandidate(None)
        except Exception as exc:
            logger.debug(f"WebRTC ignored end-of-candidates for {request.peer_id}: {exc}")
            return {"ok": False}
    return {"ok": True}
