from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StreamMeta:
    """Everything a dispatcher can know before the first frame is generated."""

    fps: float = 25.0
    width: int = 0
    height: int = 0
    channels: int = 3
    request_id: str = ""
    first_idx: int = 0
    total_frames: int | None = None
    audio_path: str | None = None

    def pts_ms(self, idx: int) -> float:
        """Timestamp of ``idx`` in milliseconds, relative to the stream start."""
        return (int(idx) - int(self.first_idx)) * 1000.0 / float(self.fps or 1.0)
