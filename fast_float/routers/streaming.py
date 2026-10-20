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

import struct

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from fast_float.core.state import app_state
from fast_float.streaming.live import CLOSE
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
    # for this client instead of being published to nobody.
    q = stream.subscribe(maxsize=64)
    try:
        meta = await stream.wait_meta(META_TIMEOUT_S)
        if meta is None:
            await ws.send_json({"type": "error", "message": stream.error or "stream did not start"})
            await ws.close(code=4408)
            return

        await ws.send_json(_meta_payload(meta))
        while True:
            item = await q.get()
            if item is CLOSE:
                break
            kind = item[0]
            if kind == "frame":
                _, idx, pts_ms, frame = item
                await ws.send_bytes(HEADER.pack(int(idx), 1, int(pts_ms)) + frame.tobytes())
            elif kind == "gap":
                _, start, end, reason = item
                await ws.send_json({"type": "gap", "start": start, "end": end, "reason": reason})
    except (WebSocketDisconnect, RuntimeError):
        pass
    except Exception as exc:  # pragma: no cover - transport failures are client-specific
        logger.debug("[ws %s] stream closed: %s", request_id, exc)
    finally:
        stream.unsubscribe(q)

    try:
        if stream.error:
            await ws.send_json({"type": "error", "message": stream.error})
        else:
            await ws.send_json({"type": "done", "result": stream.result})
        await ws.close()
    except Exception:
        pass
