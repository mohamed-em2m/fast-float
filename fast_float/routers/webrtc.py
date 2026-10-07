import os
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


@router.post("/offer", response_model=WebRTCOfferResponse)
async def webrtc_offer(request: WebRTCOfferRequest):
    """Handle WebRTC SDP offer and establish peer connection for stream playback."""
    if not AIORTC_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="aiortc is not installed in the current environment",
        )

    offer = RTCSessionDescription(sdp=request.sdp, type=request.type)
    pc = RTCPeerConnection()
    app_state.pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info(f"WebRTC connection state: {pc.connectionState}")
        if pc.connectionState in ["failed", "closed"]:
            await pc.close()
            app_state.pcs.discard(pc)

    # If stream_id exists in active streams, attach its media player tracks
    if request.stream_id and request.stream_id in app_state.active_streams:
        video_path = app_state.active_streams[request.stream_id]
        if os.path.exists(video_path):
            try:
                player = MediaPlayer(video_path)
                if player.video:
                    pc.addTrack(player.video)
                if player.audio:
                    pc.addTrack(player.audio)
                logger.info(f"Attached MediaPlayer tracks for {request.stream_id}")
            except Exception as e:
                logger.warning(
                    f"Failed to attach media player for {request.stream_id}: {e}"
                )

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return WebRTCOfferResponse(
        sdp=pc.localDescription.sdp,
        type=pc.localDescription.type,
    )
