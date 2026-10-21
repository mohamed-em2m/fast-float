import asyncio
import os
import time
from contextlib import suppress
from fractions import Fraction
from typing import Any

from fastapi import APIRouter, HTTPException

from fast_float.core.perf_metrics import FrameClock
from fast_float.core.state import app_state
from fast_float.schemas.generation import WebRTCOfferRequest, WebRTCOfferResponse
from fast_float.streaming.live import CLOSE
from fast_float.utils.logger import logger

try:
    import av
    import numpy as np
    from aiortc import RTCPeerConnection, RTCSessionDescription
    from aiortc.contrib.media import MediaPlayer
    from aiortc.mediastreams import MediaStreamError, MediaStreamTrack

    AIORTC_AVAILABLE = True
except ImportError:
    AIORTC_AVAILABLE = False

router = APIRouter(tags=["webrtc"])

# MediaPlayer instances keyed by id(pc). A player spawns an ffmpeg subprocess,
# so it must be stopped when its peer connection goes away.
_players: dict[int, Any] = {}
# Live subscriptions keyed by id(pc): (stream, queue).
_live_subs: dict[int, Any] = {}

LIVE_META_TIMEOUT_S = 30.0


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


def _cleanup_peer(pc) -> None:
    """Stop the attached media player, unblock live tracks, drop the peer."""
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


async def _live_offer(request: WebRTCOfferRequest) -> WebRTCOfferResponse:
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
        pc = RTCPeerConnection()
        app_state.pcs.add(pc)
        # Register the subscription before addTrack so any failure below is
        # cleaned up by _cleanup_peer (which unsubscribes it).
        track = LiveVideoTrack(queue, meta)
        _live_subs[id(pc)] = (stream, queue, track)

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
        )
    except BaseException:
        # Cancelled/timed out before a peer owned the subscription; _cleanup_peer
        # has already unsubscribed if a peer was created.  discard() is idempotent.
        stream.unsubscribe(queue)
        raise


@router.post("/offer", response_model=WebRTCOfferResponse)
async def webrtc_offer(request: WebRTCOfferRequest):
    """Handle WebRTC SDP offer and establish peer connection for stream playback."""
    if not AIORTC_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="aiortc is not installed in the current environment",
        )

    if request.live:
        return await _live_offer(request)

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
    )
