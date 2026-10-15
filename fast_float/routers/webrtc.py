import os
from typing import Dict, Any
from fastapi import APIRouter, HTTPException
from fast_float.schemas.generation import WebRTCOfferRequest, WebRTCOfferResponse
from fast_float.core.state import app_state
from fast_float.utils.logger import logger

try:
    from aiortc import RTCPeerConnection, RTCSessionDescription
    from aiortc.contrib.media import MediaPlayer

    AIORTC_AVAILABLE = True
except ImportError:
    AIORTC_AVAILABLE = False

router = APIRouter(tags=["webrtc"])

# MediaPlayer instances keyed by id(pc). A player spawns an ffmpeg subprocess,
# so it must be stopped when its peer connection goes away.
_players: Dict[int, Any] = {}


def _cleanup_peer(pc) -> None:
    """Stop the attached media player and drop the peer connection."""
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
    app_state.pcs.discard(pc)


@router.post("/offer", response_model=WebRTCOfferResponse)
async def webrtc_offer(request: WebRTCOfferRequest):
    """Handle WebRTC SDP offer and establish peer connection for stream playback."""
    if not AIORTC_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="aiortc is not installed in the current environment",
        )

    if request.stream_id is not None and request.stream_id not in app_state.active_streams:
        raise HTTPException(
            status_code=404,
            detail="Requested stream ID not found",
        )

    offer = RTCSessionDescription(sdp=request.sdp, type=request.type)
    pc = RTCPeerConnection()
    app_state.pcs.add(pc)

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
                    logger.warning(
                        f"Media has no playable tracks for {request.stream_id}"
                    )
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
                logger.warning(
                    f"Failed to attach media player for {request.stream_id}: {e}"
                )
                try:
                    await pc.close()
                finally:
                    _cleanup_peer(pc)
                raise HTTPException(
                    status_code=500,
                    detail="Failed to prepare media for WebRTC stream",
                )
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
    )
