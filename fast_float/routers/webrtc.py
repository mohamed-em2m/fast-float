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
from fast_float.streaming.live import CLOSE
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

    class LiveVideoTrack(MediaStreamTrack):
        """Video track fed by a LiveStream subscription queue."""

        kind = "video"

        def __init__(self, queue: asyncio.Queue, meta: Any) -> None:
            super().__init__()
            self._queue = queue
            self._fps = max(1.0, float(meta.fps or 25.0))
            self._next_at: float | None = None
            self._stopped = False
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
            super().stop()

        async def recv(self) -> "av.VideoFrame":
            while True:
                if self._stopped:
                    raise MediaStreamError("track stopped")
                item = await self._queue.get()
                if item is CLOSE:
                    self.stop()
                    raise MediaStreamError("stream finished")
                if item[0] != "frame":
                    continue
                _, _idx, pts_ms, frame_array = item
                now = time.monotonic()
                if self._next_at is None:
                    self._next_at = now
                delay = self._next_at - now
                if delay > 0:
                    await asyncio.sleep(delay)
                self._next_at = max(self._next_at + 1.0 / self._fps, time.monotonic())
                self.clock.on_frame(_idx, pts_ms, time.monotonic())

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
        stream, queue, track = sub
        stream.unsubscribe(queue)
        with suppress(RuntimeError):
            track.stop()
        with suppress(asyncio.QueueFull):
            queue.put_nowait(CLOSE)
    app_state.pcs.discard(pc)


async def _live_offer(request: WebRTCOfferRequest, http_request: Request) -> WebRTCOfferResponse:
    """Attach the in-flight live stream (video + request audio) to a new peer."""
    if not request.stream_id:
        raise HTTPException(status_code=400, detail="live offers require a stream_id")

    stream = await app_state.wait_for_live_stream(request.stream_id)
    if stream is None:
        raise HTTPException(status_code=404, detail="Requested live stream not found")

    # Subscribe before waiting for meta so the first decoded frames are queued.
    queue = stream.subscribe(maxsize=8)
    try:
        meta = await stream.wait_meta(LIVE_META_TIMEOUT_S)
        if meta is None:
            raise HTTPException(
                status_code=502 if stream.error else 504,
                detail=stream.error or "Live stream did not start in time",
            )

        offer = RTCSessionDescription(sdp=request.sdp, type=request.type)
        pc = _new_peer_connection(http_request)
        app_state.pcs.add(pc)
        # Register the subscription before addTrack so any failure below is
        # cleaned up by _cleanup_peer (which unsubscribes it).
        track = LiveVideoTrack(queue, meta)
        _live_subs[id(pc)] = (stream, queue, track)

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

            if stream.audio_path and os.path.exists(stream.audio_path):
                try:
                    player = MediaPlayer(stream.audio_path)
                    if player.audio:
                        pc.addTrack(player.audio)
                    _players[id(pc)] = player
                except Exception as exc:
                    logger.warning(f"Live audio attach failed: {exc}")

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
