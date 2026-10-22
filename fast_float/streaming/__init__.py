"""Live frame delivery for FLOAT inference (WebSocket, WebRTC, custom sinks)."""

from fast_float.streaming.dispatcher import FrameDispatcher, FrameQueue
from fast_float.streaming.live import CLOSE, LiveStream, LiveStreamDispatcher
from fast_float.streaming.types import StreamMeta

__all__ = ["CLOSE", "FrameDispatcher", "FrameQueue", "LiveStream", "LiveStreamDispatcher", "StreamMeta"]
