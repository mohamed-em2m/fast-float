from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class GenerationRequest(BaseModel):
    audio_base64: str = Field(..., description="Base64 encoded audio bytes")
    audio_format: str = Field("pcm", description="Audio container format (pcm, wav, mp3, etc.)")
    sample_rate: int = Field(16000, description="Audio sample rate in Hz")
    channels: int = Field(1, description="Audio channel count")
    sample_width: int = Field(2, description="Audio sample width in bytes (1=8bit, 2=16bit, 4=32bit)")
    image_base64: Optional[str] = Field(None, description="Base64 encoded portrait image. If omitted, default image is used.")
    image_format: Optional[str] = Field("jpg", description="Format of image_base64 (jpg, png, etc.)")
    emotion: str = Field("S2E", description="Emotion conditioning mode")
    nfe: int = Field(10, description="Number of ODE solver function evaluations (steps)")
    seed: int = Field(25, description="Random seed")
    a_cfg_scale: float = Field(2.0, description="Audio classifier-free guidance scale")
    r_cfg_scale: float = Field(1.0, description="Reference classifier-free guidance scale")
    e_cfg_scale: float = Field(1.0, description="Emotion classifier-free guidance scale")
    no_crop: bool = Field(False, description="Disable automatic face crop")
    compress: bool = Field(True, description="Compress output video using optimized encoder")
    quality: str = Field("balanced", description="Encoding quality (fast, balanced, high)")


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


class HealthResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    status: str
    model_loaded: bool
    encoder: str
    active_streams: int
    performance_stats: Dict[str, Any]
    gpu_info: Optional[Dict[str, Any]] = None


class WebRTCOfferRequest(BaseModel):
    sdp: str
    type: str
    stream_id: Optional[str] = None


class WebRTCOfferResponse(BaseModel):
    sdp: str
    type: str
