from typing import Literal, Optional, Dict, Any
from pydantic import BaseModel, Field


class GenerationRequest(BaseModel):
    audio_base64: str = Field(..., description="Base64 encoded audio bytes")
    audio_format: str = Field("pcm", description="Audio container format (pcm, wav, mp3, etc.)")
    sample_rate: int = Field(16000, ge=8000, le=48000, description="Audio sample rate in Hz")
    channels: int = Field(1, ge=1, le=2, description="Audio channel count")
    sample_width: int = Field(2, description="Audio sample width in bytes (1=8bit, 2=16bit, 4=32bit)")
    image_base64: Optional[str] = Field(None, description="Base64 encoded portrait image. If omitted, default image is used.")
    image_format: Optional[str] = Field("jpg", description="Format of image_base64 (jpg, png, etc.)")
    emotion: Literal["S2E", "neutral", "happy", "sad", "angry", "surprise", "disgust"] = Field(
        "S2E", description="Emotion conditioning mode"
    )
    nfe: int = Field(10, ge=5, le=25, description="Number of ODE solver function evaluations (steps)")
    seed: int = Field(25, ge=0, description="Random seed")
    a_cfg_scale: float = Field(2.0, ge=0.5, le=5.0, description="Audio classifier-free guidance scale")
    r_cfg_scale: float = Field(1.0, ge=0.5, le=3.0, description="Reference classifier-free guidance scale")
    e_cfg_scale: float = Field(1.0, ge=0.5, le=3.0, description="Emotion classifier-free guidance scale")
    no_crop: bool = Field(False, description="Disable automatic face crop")
    compress: bool = Field(True, description="Compress output video using optimized encoder")
    quality: Literal["fast", "balanced", "high"] = Field("balanced", description="Encoding quality (fast, balanced, high)")
    chunk_seconds: float = Field(
        0.0,
        ge=0.0,
        le=600.0,
        description="Split the audio into windows of this many seconds and generate "
        "each independently, then merge. 0 disables chunking (single pass).",
    )
    chunk_overlap_seconds: float = Field(
        0.5,
        ge=0.0,
        le=5.0,
        description="Overlap between consecutive chunks, trimmed from the head of "
        "each later chunk when merging to avoid duplicated frames.",
    )
    decode_batch_size: int = Field(
        0,
        ge=0,
        le=64,
        description="Frames decoded per micro-batch on the GPU. 0 lets the VRAM "
        "auto-tuner pick the largest size that fits.",
    )
    client_request_id: Optional[str] = Field(
        None,
        max_length=64,
        description="Client-supplied id so the browser can poll /progress/{id} "
        "before the long-running /generate call returns.",
    )


class PerformanceTimings(BaseModel):
    preprocess_time: float
    inference_time: float
    total_time: float
    fps: float


class GenerationResponse(BaseModel):
    status: str = "success"
    request_id: str
    download_url: str
    download_backup: Optional[str] = None
    stream_url: str
    video_path: str
    file_size_mb: float
    resolution: str
    encoder: str
    performance: PerformanceTimings
    chunks: int = Field(1, description="Number of audio chunks generated (1 = no chunking)")


class GenerationProgress(BaseModel):
    request_id: str
    status: Literal["queued", "processing", "merging", "done", "error"]
    stage: str = ""
    total_chunks: int = 0
    completed_chunks: int = 0
    percent: float = 0.0
    message: str = ""


class HealthResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    status: str
    model_loaded: bool
    encoder: str
    active_streams: int
    performance_stats: Dict[str, Any]
    gpu_info: Optional[Dict[str, Any]] = None


class WebRTCOfferRequest(BaseModel):
    sdp: str = Field(..., min_length=1)
    type: str = Field(..., min_length=1)
    stream_id: Optional[str] = None


class WebRTCOfferResponse(BaseModel):
    sdp: str
    type: str
