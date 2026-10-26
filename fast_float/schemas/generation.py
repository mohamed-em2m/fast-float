from functools import partial
from typing import Any, Literal

from pydantic import BaseModel, Field

from fast_float.config.float_settings import request_default


class GenerationRequest(BaseModel):
    audio_base64: str = Field(..., description="Base64 encoded audio bytes")
    audio_format: str = Field("pcm", description="Audio container format (pcm, wav, mp3, etc.)")
    sample_rate: int = Field(16000, ge=8000, le=48000, description="Audio sample rate in Hz")
    channels: int = Field(1, ge=1, le=2, description="Audio channel count")
    sample_width: int = Field(2, description="Audio sample width in bytes (1=8bit, 2=16bit, 4=32bit)")
    image_base64: str | None = Field(
        None,
        description="Base64 encoded portrait image. If omitted, default image is used.",
    )
    image_format: str | None = Field("jpg", description="Format of image_base64 (jpg, png, etc.)")
    emotion: Literal["S2E", "neutral", "happy", "sad", "angry", "surprise", "disgust"] = Field(
        default_factory=partial(request_default, "emotion"),
        description="Emotion conditioning mode",
    )
    nfe: int = Field(
        default_factory=partial(request_default, "nfe"),
        ge=5,
        le=25,
        description="Number of ODE solver function evaluations (steps)",
    )
    seed: int = Field(default_factory=partial(request_default, "seed"), ge=0, description="Random seed")
    a_cfg_scale: float = Field(
        default_factory=partial(request_default, "a_cfg_scale"),
        ge=0.5,
        le=5.0,
        description="Audio classifier-free guidance scale",
    )
    r_cfg_scale: float = Field(
        default_factory=partial(request_default, "r_cfg_scale"),
        ge=0.5,
        le=3.0,
        description="Reference classifier-free guidance scale",
    )
    e_cfg_scale: float = Field(
        default_factory=partial(request_default, "e_cfg_scale"),
        ge=0.5,
        le=3.0,
        description="Emotion classifier-free guidance scale",
    )
    no_crop: bool = Field(
        default_factory=partial(request_default, "no_crop"),
        description="Disable automatic face crop",
    )
    compress: bool = Field(True, description="Compress output video using optimized encoder")
    quality: Literal["fast", "balanced", "high"] = Field(
        "balanced", description="Encoding quality (fast, balanced, high)"
    )
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
        default_factory=partial(request_default, "decode_batch_size"),
        ge=0,
        le=64,
        description="Frames decoded per micro-batch on the GPU. 0 lets the VRAM "
        "auto-tuner pick the largest size that fits.",
    )
    chunk_batch_size: int = Field(
        1,
        ge=1,
        le=8,
        description="How many audio chunks to sample in one model batch. 1 keeps "
        "today's sequential chunk loop; >1 pads short chunks with silence "
        "(capped by chunk_pad_seconds) and samples them together.",
    )
    chunk_pad_seconds: float = Field(
        2.0,
        ge=0.0,
        le=10.0,
        description="Max silence padding when equalizing chunk lengths for a "
        "sample batch. Chunks needing more padding run solo.",
    )
    client_request_id: str | None = Field(
        None,
        max_length=64,
        description="Client-supplied id so the browser can poll /progress/{id} "
        "before the long-running /generate call returns.",
    )
    live: bool = Field(
        False,
        description="Also publish frames live to /ws/stream/{id} and live /offer peers while the video is generated.",
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
    download_backup: str | None = None
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
    # Delivery-side metrics of a live stream (TTFF, p50/p95/p99 latency,
    # jitter, dropped frames). None when the request is not streaming.
    delivery_metrics: dict[str, Any] | None = None


class HealthResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    status: str
    model_loaded: bool
    warming_up: bool = False
    # "starting" / "single-inference" / "batched-inference" / "done" / "failed",
    # so a UI can show what the server is doing instead of a bare bool.
    warmup_stage: str | None = None
    warmup_elapsed_s: float | None = None
    encoder: str
    active_streams: int
    performance_stats: dict[str, Any]
    gpu_info: dict[str, Any] | None = None


class WebRTCOfferRequest(BaseModel):
    sdp: str = Field(..., min_length=1)
    type: str = Field(..., min_length=1)
    stream_id: str | None = None
    live: bool = Field(False, description="Attach the in-flight live stream instead of a finished file")


class WebRTCIceCandidateRequest(BaseModel):
    """A browser ICE candidate to add to an existing peer connection."""

    peer_id: str = Field(..., min_length=1)
    candidate: str | None = Field(
        None,
        description="Candidate attribute string (omit or null for end-of-candidates)",
    )
    sdp_mid: str | None = Field(None, description="Media section id from the offer")
    sdp_mline_index: int | None = Field(None, ge=0, description="Media section index from the offer")


class WebRTCOfferResponse(BaseModel):
    sdp: str
    type: str
    peer_id: str | None = Field(
        None,
        description="Opaque id used to POST later ICE candidates to /offer/candidate",
    )
