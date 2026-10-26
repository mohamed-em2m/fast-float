"""IPC message protocols and serialization for the dual-process architecture."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass


@dataclass
class IPCRequest:
    """Request sent from Streaming Backend (Process 1) to FLOAT Worker (Process 2)."""

    cmd: str  # "infer", "prepare_silence", "interrupt", "status"
    session_id: str = ""
    req_id: str = ""
    audio_pcm: str = ""  # Base64-encoded PCM16 mono bytes (or path)
    valid_samples: int = 0
    sample_rate: int = 16000
    nfe: int = 5
    ref_image_path: str = ""
    gen: int = 0

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, payload: str | bytes) -> IPCRequest:
        data = json.loads(payload)
        return cls(**data)


@dataclass
class IPCResponse:
    """Response / Status sent back from FLOAT Worker (Process 2) to Backend."""

    status: str  # "ok", "error", "interrupted"
    req_id: str = ""
    session_id: str = ""
    error_msg: str = ""
    total_frames: int = 0
    rtf: float = 0.0

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, payload: str | bytes) -> IPCResponse:
        data = json.loads(payload)
        return cls(**data)


@dataclass
class IPCFrameNotification:
    """Lightweight notification sent over PUB/SUB or PUSH/PULL when a frame is ready in SHM."""

    session_id: str
    gen: int
    slot_idx: int
    frame_idx: int
    pts_ms: float
    width: int
    height: int
    audio_bytes_len: int = 0
    is_last: bool = False

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, payload: str | bytes) -> IPCFrameNotification:
        data = json.loads(payload)
        return cls(**data)
