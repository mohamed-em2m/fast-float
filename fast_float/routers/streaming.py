"""Live frame transport over WebSocket.

Protocol (per connection to ``/ws/stream/{request_id}``)::

    server -> {"type": "meta", "fps", "width", "height", "channels", ...}
    server -> binary  [u32 frame_idx][u32 frame_count][u32 pts_ms][rgb bytes]
    server -> {"type": "gap", "start", "end", "reason"}
    server -> {"type": "done", "result": {...}} | {"type": "error", "message"}

Frames are raw ``width * height * channels`` uint8 RGB(A) payloads, one frame
per message.  A slow client loses frames (and sees a gap/idx jump), never
backpressures the GPU.
"""

from __future__ import annotations

import asyncio
import struct

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from fast_float.core.state import app_state
from fast_float.streaming.live import CLOSE, live_queue_size
from fast_float.utils.logger import logger

router = APIRouter(tags=["streaming"])

HEADER = struct.Struct(">III")  # frame_idx, frame_count, pts_ms
# The browser POSTs /generate and opens the socket immediately; allow for the
# request to be queued behind the semaphore before giving up.
WAIT_TIMEOUT_S = 15.0
META_TIMEOUT_S = 30.0


def _meta_payload(meta) -> dict:
    return {
        "type": "meta",
        "fps": meta.fps,
        "width": meta.width,
        "height": meta.height,
        "channels": meta.channels,
        "first_idx": meta.first_idx,
        "total_frames": meta.total_frames,
        "has_audio": bool(meta.audio_path),
    }


@router.websocket("/ws/stream/{request_id}")
async def live_frames(ws: WebSocket, request_id: str) -> None:
    """Stream generated frames as they are decoded."""
    await ws.accept()

    stream = await app_state.wait_for_live_stream(request_id, timeout=WAIT_TIMEOUT_S)
    if stream is None:
        await ws.send_json({"type": "error", "message": "unknown or finished request"})
        await ws.close(code=4404)
        return

    # Subscribe before waiting for meta so the first decoded frames are queued
    # for this client instead of being published to nobody. A fast render can
    # outrun a slow socket; buffer it (bounded by ``live_queue_size``) rather
    # than dropping most of the clip on the floor.
    q = stream.subscribe(maxsize=live_queue_size(stream.meta))
    try:
        meta = await stream.wait_meta(META_TIMEOUT_S)
        if meta is None:
            await ws.send_json({"type": "error", "message": stream.error or "stream did not start"})
            await ws.close(code=4408)
            return
        # Meta arrived after the provisional subscription: size the queue from
        # the real frame dimensions now that they are known.
        q.resize(live_queue_size(meta))

        await ws.send_json(_meta_payload(meta))
        # Play the preview at the stream's own rate.  A fast render is buffered
        # in ``q`` (spilling to disk if needed), so without this the sender
        # would push the whole clip as fast as the socket accepts it.  Those
        # bursts are synchronous (768 KiB message + permessage-deflate), which
        # starved the event loop for hundreds of ms and made a LiveKit
        # publisher on the same loop fall behind by tens of seconds.
        fps = float(meta.fps or 25.0)
        interval = 1.0 / fps if fps > 0 else 0.04
        loop = asyncio.get_running_loop()
        next_frame_at = loop.time()
        while True:
            item = await q.get()
            if item is CLOSE:
                break
            kind = item[0]
            if kind == "frame":
                _, idx, pts_ms, frame = item
                await ws.send_bytes(HEADER.pack(int(idx), 1, int(pts_ms)) + frame.tobytes())
                next_frame_at = max(next_frame_at + interval, loop.time())
                delay = next_frame_at - loop.time()
                if delay > 0:
                    await asyncio.sleep(delay)
            elif kind == "gap":
                _, start, end, reason = item
                await ws.send_json({"type": "gap", "start": start, "end": end, "reason": reason})
    except (WebSocketDisconnect, RuntimeError):
        pass
    except Exception as exc:  # pragma: no cover - transport failures are client-specific
        logger.debug("[ws %s] stream closed: %s", request_id, exc)
    finally:
        stream.unsubscribe(q)
        q.dispose()

    try:
        if stream.error:
            await ws.send_json({"type": "error", "message": stream.error})
        else:
            await ws.send_json({"type": "done", "result": stream.result})
        await ws.close()
    except Exception:
        pass
