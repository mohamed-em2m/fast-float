import json
import base64
import tempfile
import datetime
import asyncio
import io
import logging
import time
from pathlib import Path
from typing import Optional, Dict, Tuple
import uuid
import subprocess
import gc
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, StreamingResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field

# WebRTC
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer
from aiortc.contrib.media import MediaPlayer
from av import VideoFrame

# Standard libraries
import cv2
import numpy as np
from PIL import Image
from pydub import AudioSegment
import soundfile as sf
from scipy import signal

# PyTorch (after environment setup)
import torch

# Your inference module
from inference_optimized import OptimizedInferenceAgent, InferenceOptions
from initialize_env import initialize_environment
initialize_environment()
# ============================================================================
# GPU UTILITY FUNCTIONS (Safe - no CUDA initialization issues)
# ============================================================================

def get_gpu_info():
    """Get GPU information safely"""
    if not torch.cuda.is_available():
        return {"status": "no_gpu"}
    
    gpu_name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    total_memory_gb = props.total_memory / 1024**3
    memory_fraction = float(os.environ.get('PYTORCH_CUDA_MEMORY_FRACTION', '1.0'))
    
    return {
        "name": gpu_name,
        "total_memory_gb": round(total_memory_gb, 2),
        "memory_limit_gb": round(total_memory_gb * memory_fraction, 2),
        "memory_fraction": memory_fraction
    }

def clear_gpu_memory():
    """Aggressively clear GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()

def get_current_gpu_memory():
    """Get current GPU memory usage"""
    if not torch.cuda.is_available():
        return None
    
    allocated_gb = torch.cuda.memory_allocated(0) / 1024**3
    reserved_gb = torch.cuda.memory_reserved(0) / 1024**3
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    return {
        "allocated_gb": round(allocated_gb, 2),
        "reserved_gb": round(reserved_gb, 2),
        "total_gb": round(total_gb, 2),
        "usage_percent": round((allocated_gb / total_gb) * 100, 1)
    }

def print_gpu_status():
    """Print GPU status"""
    info = get_gpu_info()
    mem = get_current_gpu_memory()
    
    if info.get("status") == "no_gpu":
        logger.warning("⚠️ No GPU available")
        return
    
    logger.info(f"🎮 GPU: {info['name']}")
    logger.info(f"   Total: {info['total_memory_gb']}GB | Limit: {info['memory_limit_gb']}GB")
    
    if mem:
        logger.info(f"   Current: {mem['allocated_gb']}GB ({mem['usage_percent']}%) | Reserved: {mem['reserved_gb']}GB")

import os
import json
import base64
import tempfile
import datetime
import asyncio
import io
import logging
import time
from pathlib import Path
from typing import Optional, Dict, Tuple
import uuid
import subprocess
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, StreamingResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field

# WebRTC
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer
from aiortc.contrib.media import MediaPlayer
from av import VideoFrame
import cv2
import numpy as np
from PIL import Image
import torch
# Audio processing
from pydub import AudioSegment
import soundfile as sf
from scipy import signal

# Ngrok

# Import OPTIMIZED inference components
from inference_optimized import OptimizedInferenceAgent, InferenceOptions

# --- Logging Configuration ---
class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors and better structure"""
    
    grey = "\x1b[38;21m"
    blue = "\x1b[38;5;39m"
    yellow = "\x1b[38;5;226m"
    red = "\x1b[38;5;196m"
    bold_red = "\x1b[31;1m"
    green = "\x1b[38;5;46m"
    reset = "\x1b[0m"
    
    FORMATS = {
        logging.DEBUG: grey + "%(asctime)s | %(levelname)-8s | %(message)s" + reset,
        logging.INFO: green + "%(asctime)s | %(levelname)-8s | %(message)s" + reset,
        logging.WARNING: yellow + "%(asctime)s | %(levelname)-8s | %(message)s" + reset,
        logging.ERROR: red + "%(asctime)s | %(levelname)-8s | %(message)s" + reset,
        logging.CRITICAL: bold_red + "%(asctime)s | %(levelname)-8s | %(message)s" + reset,
    }
    
    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt='%Y-%m-%d %H:%M:%S')
        return formatter.format(record)

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setFormatter(ColoredFormatter())
logger.addHandler(console_handler)

# Suppress noisy loggers
logging.getLogger('libav').setLevel(logging.ERROR)
logging.getLogger('ffmpeg').setLevel(logging.ERROR)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('asyncio').setLevel(logging.WARNING)

# --- Configuration ---
DEFAULT_IMAGE_PATH = "./images/img.jpg"
RESULTS_DIR = "./results"
MAX_CONCURRENT_REQUESTS = 10
PORT = 8000
TARGET_IMAGE_SIZE = 256
STUN_SERVERS = [
    "stun:stun.l.google.com:19302",
    "stun:stun1.l.google.com:19302",
    "stun:stun2.l.google.com:19302"
]

# Model optimization settings
OPTIMIZATION_MODE = "all"
USE_FP16 = True
COMPRESS_RESPONSES = True

app = FastAPI(title="FLOAT WebRTC API - NVENC Optimized", version="5.1.0")
# Standard libraries
import cv2
import numpy as np
from PIL import Image
from pydub import AudioSegment
import soundfile as sf
from scipy import signal

# PyTorch (after environment setup)
import torch

# Your inference module
from inference_optimized import OptimizedInferenceAgent, InferenceOptions

# ... [Keep all GPU utility functions and other code] ...

# Global variables
agent = None
opt = None
public_url = None
pcs = set()
active_streams = {}
request_semaphore = None
queue_counter = 0
executor = ThreadPoolExecutor(max_workers=15)  # Single worker for sequential execution
video_optimizer = None
inference_lock = None  # Will be initialized in startup
if COMPRESS_RESPONSES:
    app.add_middleware(GZipMiddleware, minimum_size=1000)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.environ["NGROK_AUTHTOKEN"] = "2yjRvpvFWHwJS0JC95ghnpBeZf0_5aLxWWF5pWK6LtVafzXyN"

# Global variables
agent = None
opt = None
public_url = None
public_backup_url = None
pcs = set()
active_streams = {}
request_semaphore = None
queue_counter = 0
executor = ThreadPoolExecutor(max_workers=14)
video_optimizer = None

# Performance tracking
performance_stats = {
    "total_requests": 0,
    "total_inference_time": 0.0,
    "total_preprocess_time": 0.0,
    "total_postprocess_time": 0.0,
    "average_fps": 0.0
}

# --- NVENC Video Optimizer ---
class VideoSaveOptimizer:
    """Smart video encoder with GPU acceleration and fallback"""
    
    def __init__(self):
        self.encoder_type = None
        self.selected_encoder = None
        self._check_nvidia_gpu()
        self.encoder_available = self._detect_best_encoder()
        logger.info(f"🎬 Video Encoder: {self.encoder_type}")
    
    def _check_nvidia_gpu(self):
        """Check if NVIDIA GPU is available"""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=3
            )
            if result.returncode == 0 and result.stdout.strip():
                gpu_name = result.stdout.strip()
                logger.info(f"🎮 NVIDIA GPU detected: {gpu_name}")
                
                # Check NVENC capability
                result2 = subprocess.run(
                    ["nvidia-smi", "--query-gpu=encoder.h264", "--format=csv,noheader"],
                    capture_output=True,
                    text=True,
                    timeout=3
                )
                if result2.returncode == 0:
                    logger.info(f"   NVENC support: {result2.stdout.strip()}")
                return True
            else:
                logger.warning("⚠️ No NVIDIA GPU detected (nvidia-smi returned no output)")
                return False
        except FileNotFoundError:
            logger.warning("⚠️ nvidia-smi not found - NVIDIA drivers may not be installed")
            return False
        except Exception as e:
            logger.warning(f"⚠️ GPU check failed: {e}")
            return False
    
    def _detect_best_encoder(self) -> bool:
        """Auto-detect best available encoder (NVENC > QSV > VideoToolbox > CPU)"""
        
        gpu_encoders = [
            ("h264_nvenc", "NVENC (NVIDIA GPU)"),
            ("hevc_nvenc", "NVENC HEVC (NVIDIA GPU)"),
            ("h264_qsv", "QuickSync (Intel GPU)"),
            ("h264_videotoolbox", "VideoToolbox (Apple)"),
            ("h264_amf", "AMF (AMD GPU)")
        ]
        
        try:
            # First check if ffmpeg is available
            result = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                text=True,
                timeout=3
            )
            
            if result.returncode != 0:
                logger.error("❌ FFmpeg not found!")
                self.encoder_type = "libx264 (CPU - fallback)"
                self.selected_encoder = "libx264"
                return False
            
            logger.info(f"📹 FFmpeg version: {result.stdout.split('version')[1].split()[0] if 'version' in result.stdout else 'unknown'}")
            
            # Get available encoders
            result = subprocess.run(
                ["ffmpeg", "-hide_banner", "-encoders"],
                capture_output=True,
                text=True,
                timeout=3
            )
            
            encoders_output = result.stdout
            logger.info("🔍 Checking available GPU encoders...")
            
            # Try GPU encoders first
            for encoder, name in gpu_encoders:
                if encoder in encoders_output:
                    logger.info(f"   ✓ Found {encoder} in encoders list")
                    if self._test_encoder(encoder):
                        logger.info(f"   ✓ {encoder} test PASSED")
                        self.encoder_type = name
                        self.selected_encoder = encoder
                        return True
                    else:
                        logger.warning(f"   ✗ {encoder} test FAILED (encoder exists but can't encode)")
                else:
                    logger.debug(f"   ✗ {encoder} not available")
            
            # Fallback to CPU
            logger.warning("⚠️ No working GPU encoder found, falling back to CPU")
            if "libx264" in encoders_output:
                self.encoder_type = "libx264 (CPU - ultrafast)"
                self.selected_encoder = "libx264"
                return True
            
            logger.error("❌ No suitable video encoder found!")
            return False
            
        except FileNotFoundError:
            logger.error("❌ FFmpeg not installed or not in PATH!")
            self.encoder_type = "libx264 (CPU - fallback)"
            self.selected_encoder = "libx264"
            return False
        except Exception as e:
            logger.error(f"❌ Encoder detection failed: {e}")
            self.encoder_type = "libx264 (CPU - fallback)"
            self.selected_encoder = "libx264"
            return False
    
    def _test_encoder(self, encoder: str) -> bool:
        """Test if encoder actually works"""
        try:
            logger.info(f"      🧪 Testing {encoder}...")
            
            test_cmd = [
                "ffmpeg", "-y",
                "-f", "lavfi",
                "-i", "color=c=black:s=256x256:d=0.04",  # 1 frame at 25fps
                "-c:v", encoder,
                "-frames:v", "1",
                "-f", "null",
                "-"
            ]
            
            # Use capture_output=True OR stdout/stderr separately, not both
            result = subprocess.run(
                test_cmd,
                capture_output=True,  # This captures both stdout and stderr
                timeout=10
            )
            
            if result.returncode == 0:
                logger.info(f"      ✅ {encoder} working!")
                return True
            else:
                error_output = result.stderr.decode() if result.stderr else "Unknown error"
                # Show first meaningful error line
                error_lines = [line for line in error_output.split('\n') if 'error' in line.lower() or 'failed' in line.lower()]
                error_msg = error_lines[0] if error_lines else error_output[:200]
                logger.warning(f"      ❌ {encoder} failed: {error_msg}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.warning(f"      ⏱️ {encoder} test timeout")
            return False
        except Exception as e:
            logger.warning(f"      ❌ {encoder} test error: {e}")
            return False
    
    def save_video_optimized(
        self,
        input_path: str,
        output_path: str,
        fps: int = 25,
        quality: str = "fast",
        resolution: Tuple[int, int] = (256, 256)
    ) -> Tuple[bool, float, str]:
        start_time = time.time()
        width, height = resolution
    
        # tmp location (prefer tmpfs to avoid disk I/O)
        tmp_dir = "/dev/shm" if os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK) else tempfile.gettempdir()
        tmp_out = os.path.join(tmp_dir, f"tmp_opt_{uuid.uuid4().hex}.mp4")
    
        # Decide NVENC usage
        use_gpu = bool(self.selected_encoder and "nvenc" in (self.selected_encoder or ""))
    
        if use_gpu:
            # GPU-accelerated pipeline: decode -> hwupload -> scale_cuda -> encode (NVENC)
            # -hwaccel cuda & -hwaccel_output_format cuda request hardware decode path, if available.
            cmd = [
                "ffmpeg", "-y",
                "-hide_banner", "-loglevel", "error",
                "-hwaccel", "cuda",
                "-hwaccel_output_format", "cuda",
                "-i", input_path,
                "-vf", f"hwupload_cuda,scale_cuda={width}:{height}:interp_algo=linear,fps={fps}",
                "-c:v", "h264_nvenc",
                "-preset", "p1",                # p1 = fastest
                "-rc", "constqp",              # minimal rate control overhead
                "-qp", "28",                   # quality/size tradeoff (lower = better quality, slower)
                "-profile:v", "baseline",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "64k",
                "-movflags", "+faststart",
                tmp_out
            ]
            encoder_label = "h264_nvenc_p1"
        else:
            # CPU fallback: fastest libx264 settings
            cmd = [
                "ffmpeg", "-y",
                "-hide_banner", "-loglevel", "error",
                "-i", input_path,
                "-vf", f"scale={width}:{height}:flags=bilinear,fps={fps}",
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-tune", "zerolatency",
                "-crf", "30",
                "-threads", "0",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "64k",
                "-movflags", "+faststart",
                tmp_out
            ]
            encoder_label = "libx264_ultrafast"
    
        try:
            proc = subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, timeout=120)
            elapsed = time.time() - start_time
    
            # Move tmp_out to final output path (atomic)
            try:
                os.replace(tmp_out, output_path)
            except Exception:
                # fallback copy
                import shutil
                shutil.copy(tmp_out, output_path)
                os.remove(tmp_out)
    
            return True, elapsed, encoder_label
    
        except subprocess.CalledProcessError as e:
            err = e.stderr or ""
            # If NVENC failed, try CPU fallback once
            if use_gpu:
                logger.warning(f"⚠️ NVENC encode failed: {err.splitlines()[:5]} -- retrying with CPU fallback")
                self.selected_encoder = "libx264"
                return self.save_video_optimized(input_path, output_path, fps, quality, resolution)
            logger.error(f"❌ Encoding failed: {err}")
            return False, time.time() - start_time, "failed"
    
        except subprocess.TimeoutExpired:
            logger.error("❌ Encoding timeout")
            return False, time.time() - start_time, "timeout"
    
        except Exception as e:
            logger.error(f"❌ Unexpected error during encoding: {e}")
            return False, time.time() - start_time, "error"    
    
# --- Data Models ---
class GenerationRequest(BaseModel):
    audio_base64: str
    audio_format: str = "pcm"
    sample_rate: int = 16000
    channels: int = 1
    sample_width: int = 2
    image_base64: Optional[str] = None
    image_format: Optional[str] = "jpg"
    emotion: str = "S2E"
    nfe: int = 100
    seed: int = 25
    a_cfg_scale: float = 1.0
    r_cfg_scale: float = 3.0
    e_cfg_scale: float = 1.0
    no_crop: bool = False
    compress: bool = True
    quality: str = "balanced"


# --- Utility Functions ---
def ensure_default_image():
    """Create default 256x256 image if not exists"""
    if not os.path.exists(DEFAULT_IMAGE_PATH):
        logger.warning(f"Default image not found, creating placeholder")
        img = np.zeros((TARGET_IMAGE_SIZE, TARGET_IMAGE_SIZE, 3), dtype=np.uint8)
        cv2.putText(img, "Default", (TARGET_IMAGE_SIZE//4, TARGET_IMAGE_SIZE//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imwrite(DEFAULT_IMAGE_PATH, img)
        logger.info(f"Created default {TARGET_IMAGE_SIZE}x{TARGET_IMAGE_SIZE} image")


def resize_image_to_256(image_path: str, output_path: str = None) -> str:
    """
    Keep the original image size (no resizing).
    Converts to RGB if needed and saves as JPEG.
    """
    try:
        if output_path is None:
            output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg").name

        img = Image.open(image_path)

        # Fix alpha transparency by pasting on white background
        if img.mode == 'RGBA':
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])
            img = background

        # Convert other modes to RGB
        elif img.mode != 'RGB':
            img = img.convert('RGB')

        # Save WITHOUT resizing
        img.save(output_path, 'JPEG', quality=95)

        return output_path

    except Exception as e:
        logger.error(f"Failed to process image: {e}")
        raise


def convert_pcm_to_wav(pcm_data: bytes, sample_rate: int = 16000, 
                       channels: int = 1, sample_width: int = 2) -> bytes:
    """Convert PCM to WAV with proper validation"""
    try:
        # Determine dtype based on sample width
        if sample_width == 1:
            dtype = np.uint8
            max_val = 128.0
            offset = 128
        elif sample_width == 2:
            dtype = np.int16
            max_val = 32768.0
            offset = 0
        elif sample_width == 4:
            dtype = np.int32
            max_val = 2147483648.0
            offset = 0
        else:
            dtype = np.int16
            max_val = 32768.0
            offset = 0
        
        # Convert PCM bytes to numpy array
        audio_array = np.frombuffer(pcm_data, dtype=dtype)
        
        if len(audio_array) == 0:
            raise ValueError("Empty audio data")
        
        # Reshape for channels if needed
        if channels > 1:
            if len(audio_array) % channels != 0:
                # Trim to nearest multiple
                audio_array = audio_array[:len(audio_array) - (len(audio_array) % channels)]
            audio_array = audio_array.reshape(-1, channels)
        
        # Convert to float32 normalized to [-1, 1]
        if dtype == np.uint8:
            audio_float = (audio_array.astype(np.float32) - offset) / max_val
        else:
            audio_float = audio_array.astype(np.float32) / max_val
        
        # Ensure valid range
        audio_float = np.clip(audio_float, -1.0, 1.0)
        
        # Write to WAV using soundfile
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, audio_float, sample_rate, format='WAV', subtype='PCM_16')
        wav_buffer.seek(0)
        wav_bytes = wav_buffer.read()
        
        # Validate WAV file
        if len(wav_bytes) < 44:
            raise ValueError(f"Generated WAV too small: {len(wav_bytes)} bytes")
        
        # Verify it can be read back
        test_buffer = io.BytesIO(wav_bytes)
        test_data, test_sr = sf.read(test_buffer)
        logger.debug(f"WAV validation: {len(test_data)} samples at {test_sr}Hz")
        
        return wav_bytes
        
    except Exception as e:
        logger.error(f"PCM to WAV conversion failed: {e}")
        raise


def process_audio_format(audio_content: str, ext: str, 
                        sample_rate: int = 16000,
                        channels: int = 1, 
                        sample_width: int = 2) -> bytes:
    """Process audio and convert to WAV with robust validation"""
    try:
        audio_bytes = base64.b64decode(audio_content)
        ext = ext.lower().lstrip('.')
        
        logger.debug(f"Processing audio: format={ext}, size={len(audio_bytes)} bytes, sr={sample_rate}, ch={channels}")
        
        # Always treat input as PCM/raw and convert to proper WAV
        if ext in ['pcm', 'raw', 'wav']:
            # Treat all as raw PCM data and convert
            logger.info(f"Converting {ext} to WAV format")
            wav_bytes = convert_pcm_to_wav(audio_bytes, sample_rate, channels, sample_width)
            
        else:
            # Use pydub for other formats
            try:
                logger.debug(f"Using pydub for {ext} format")
                audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes), format=ext)
                
                # Convert to mono
                if audio_segment.channels > 1:
                    audio_segment = audio_segment.set_channels(1)
                
                # Set sample rate
                if audio_segment.frame_rate != sample_rate:
                    audio_segment = audio_segment.set_frame_rate(sample_rate)
                
                # Export as WAV
                wav_buffer = io.BytesIO()
                audio_segment.export(wav_buffer, format='wav')
                wav_buffer.seek(0)
                wav_bytes = wav_buffer.read()
                
            except Exception as e:
                logger.error(f"Pydub conversion failed: {e}, trying PCM fallback")
                wav_bytes = convert_pcm_to_wav(audio_bytes, sample_rate, channels, sample_width)
        
        # Final validation
        if len(wav_bytes) < 44:
            raise ValueError(f"Generated WAV file too small: {len(wav_bytes)} bytes")
        
        # Verify WAV header
        if wav_bytes[:4] != b'RIFF' or wav_bytes[8:12] != b'WAVE':
            raise ValueError("Invalid WAV file header")
        
        # Test read with soundfile
        try:
            test_buffer = io.BytesIO(wav_bytes)
            test_data, test_sr = sf.read(test_buffer)
            logger.info(f"✓ Audio processed: {len(test_data)} samples at {test_sr}Hz ({len(wav_bytes)} bytes)")
        except Exception as e:
            raise ValueError(f"Generated WAV validation failed: {e}")
        
        return wav_bytes
        
    except Exception as e:
        logger.error(f"Audio processing failed: {e}", exc_info=True)
        raise
def _run_inference_blocking(video_path_raw, image_path, audio_path, request_data):
    """Synchronous wrapper for optimized inference - uses global agent"""
    return agent.run_inference(
        res_video_path=video_path_raw,
        ref_path=image_path,
        audio_path=audio_path,
        a_cfg_scale=request_data.get("a_cfg_scale", 2.0),
        r_cfg_scale=request_data.get("r_cfg_scale", 1.0),
        e_cfg_scale=request_data.get("e_cfg_scale", 1.0),
        emo=request_data.get("emotion", "S2E"),
        nfe=request_data.get("nfe", 10),
        no_crop=request_data.get("no_crop", False),
        seed=request_data.get("seed", 25),
        verbose=True
    )
async def generate_video(request_data: Dict, request_id: str) -> Tuple[str, Dict]:
    """Generate video"""
    global inference_lock
    
    temp_files = []
    total_start_time = time.time()
    timings = {}
    
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.get_event_loop()
    
    try:
        logger.info(f"[{request_id}] 🚀 Starting video generation")
        
        # Audio processing
        preprocess_start = time.time()
        
        wav_bytes = process_audio_format(
            request_data["audio_base64"],
            request_data.get("audio_format", "pcm"),
            request_data.get("sample_rate", 16000),
            request_data.get("channels", 1),
            request_data.get("sample_width", 2)
        )
        
        audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        with open(audio_path, 'wb') as f:
            f.write(wav_bytes)
        temp_files.append(audio_path)
        
        # Image processing
        if request_data.get("image_base64"):
            image_bytes = base64.b64decode(request_data["image_base64"])
            image_path_temp = tempfile.NamedTemporaryFile(
                delete=False, 
                suffix=f".{request_data.get('image_format', 'jpg')}"
            ).name
            with open(image_path_temp, 'wb') as f:
                f.write(image_bytes)
            temp_files.append(image_path_temp)
            
            image_path = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg").name
            resize_image_to_256(image_path_temp, image_path)
            temp_files.append(image_path)
        else:
            if not os.path.exists(DEFAULT_IMAGE_PATH):
                ensure_default_image()
            image_path = DEFAULT_IMAGE_PATH
        
        timings['preprocess'] = time.time() - preprocess_start
        
        # Inference
        video_path_final = os.path.join(
            opt.res_dir,
            f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{request_id}.mp4"
        )
        
        inference_start = time.time()
        
        # Acquire lock to ensure only one inference runs at a time
        async with inference_lock:
            output_path = await loop.run_in_executor(
                executor,
                _run_inference_blocking,
                video_path_final,
                image_path,
                audio_path,
                request_data
            )
        
        timings['inference'] = time.time() - inference_start
        timings['total'] = time.time() - total_start_time
        timings['fps'] = 1.0 / timings['inference'] if timings['inference'] > 0 else 0
        
        # Update stats
        performance_stats['total_requests'] += 1
        performance_stats['total_inference_time'] += timings['inference']
        performance_stats['total_preprocess_time'] += timings['preprocess']
        
        if performance_stats['total_inference_time'] > 0:
            performance_stats['average_fps'] = performance_stats['total_requests'] / performance_stats['total_inference_time']
        
        # Cleanup temp files
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as cleanup_error:
                logger.debug(f"[{request_id}] Cleanup warning: {cleanup_error}")
        
        # Verify final output exists
        if not os.path.exists(output_path):
            raise FileNotFoundError(f"Final video not found at {output_path}")
        
        file_size = os.path.getsize(output_path) / 1024 / 1024
        
        logger.info(
            f"[{request_id}] ✅ Complete: {timings['total']:.2f}s "
            f"(prep: {timings['preprocess']:.2f}s, inf: {timings['inference']:.2f}s) | "
            f"{file_size:.2f}MB"
        )
        
        return output_path, timings
        
    except Exception as e:
        logger.error(f"[{request_id}] ❌ Generation failed: {str(e)}", exc_info=True)
        
        # Cleanup all temp files on error
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except:
                pass
        
        raise
# --- FastAPI Event Handlers ---
@app.on_event("startup")
async def startup_event():
    """Initialize model, NVENC encoder, and ngrok"""
    global agent, opt, public_url, request_semaphore, video_optimizer, executor, inference_lock,public_backup_url
    
    request_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    inference_lock = asyncio.Lock()
    
    logger.info("="*70)
    logger.info("🚀 Starting FLOAT WebRTC API - NVENC Optimized v5.1.0")
    logger.info(f"⚡ Optimization: {OPTIMIZATION_MODE} | FP16: {USE_FP16}")
    logger.info(f"📐 Target resolution: {TARGET_IMAGE_SIZE}x{TARGET_IMAGE_SIZE}")
    
    # Initialize NVENC optimizer
    video_optimizer = VideoSaveOptimizer()
    logger.info(f"🎬 Video Encoder: {video_optimizer.encoder_type}")
    logger.info("="*70)
    
    ensure_default_image()
    
    import sys
    original_argv = sys.argv.copy()
    sys.argv = [sys.argv[0]]
    
    opt = InferenceOptions().parse()
    opt.rank = 0
    opt.ngpus = 1
    opt.res_dir = RESULTS_DIR
    opt.no_crop = False
    opt.optimization_mode = OPTIMIZATION_MODE
    opt.use_fp16 = USE_FP16
    opt.n_diff_steps=100
    opt.diff_schedule="cosine"
    opt.diffusion_mode="sample"
    sys.argv = original_argv
    os.makedirs(opt.res_dir, exist_ok=True)
    
    logger.info(f"🤖 Loading FLOAT model with {OPTIMIZATION_MODE} optimization...")
    agent = OptimizedInferenceAgent(opt)
    logger.info("✅ Model loaded successfully")
    
    # ===== MODEL WARMUP WITH REAL INFERENCE =====
    logger.info("🔥 Warming up model with real inference (empty audio)...")
    warmup_start = time.time()
    
    try:
        # Create minimal test files
        warmup_audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        warmup_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        
        # Generate 0.5 seconds of silent audio (minimal but valid)
        silent_audio = np.zeros(8000, dtype=np.float32)  # 0.5 sec at 16kHz
        sf.write(warmup_audio_path, silent_audio, 16000)
        
        # Use default image
        warmup_image_path = DEFAULT_IMAGE_PATH
        
        logger.info("   Running warmup inference...")
        
        # Run actual model inference
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            executor,
            _run_inference_blocking,
            warmup_video_path,
            warmup_image_path,
            warmup_audio_path,
            {
                "a_cfg_scale": 2.0,
                "r_cfg_scale": 1.0,
                "e_cfg_scale": 1.0,
                "emotion": "S2E",
                "nfe": 10,
                "no_crop": False,
                "seed": 25
            }
        )
        
        warmup_time = time.time() - warmup_start
        logger.info(f"   ✓ Warmup inference complete ({warmup_time:.2f}s)")
        
        # Verify output was created
        if os.path.exists(warmup_video_path):
            file_size = os.path.getsize(warmup_video_path) / 1024
            logger.info(f"   ✓ Warmup video generated: {file_size:.1f}KB")
        
        # Cleanup warmup files
        for f in [warmup_audio_path, warmup_video_path]:
            try:
                if os.path.exists(f):
                    os.remove(f)
            except:
                pass
        
        logger.info("✅ Model warmup complete - ready for production traffic")
        
    except Exception as e:
        logger.warning(f"⚠️ Model warmup failed (non-critical): {e}")
        logger.warning("   First request may be slower than usual")
    
    # ===== NVENC ENCODER WARMUP (Lightweight) =====
    logger.info("🎬 Testing NVENC encoder...")
    try:
        warmup_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        
        # Create single test frame
        test_frame = np.zeros((TARGET_IMAGE_SIZE, TARGET_IMAGE_SIZE, 3), dtype=np.uint8)
        
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-pix_fmt", "rgb24", "-s", f"{TARGET_IMAGE_SIZE}x{TARGET_IMAGE_SIZE}",
            "-r", "25", "-i", "-",
            "-c:v", video_optimizer.selected_encoder,
            "-preset", "p1", "-frames:v", "1",
            "-f", "null", "-"  # Don't save, just test encoding
        ]
        
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, 
                               stderr=subprocess.PIPE)
        proc.stdin.write(test_frame.tobytes())
        proc.stdin.close()
        proc.wait(timeout=5)
        
        logger.info("   ✓ NVENC encoder ready")
        
    except Exception as e:
        logger.warning(f"⚠️ NVENC warmup failed (non-critical): {e}")
    
    # ===== END WARMUP =====
    
    logger.info("🌐 Starting ngrok tunnel...")
    try:
        public_url = os.environ.get("AVATER_PUBLIC_URL") 
        public_backup_url = os.environ.get("BACKUP_AVATER_PUBLIC_URL")   
  
        logger.info("="*70)
        logger.info("📺 NVENC-Optimized API Ready!")
        logger.info(f"🌐 Public URL: {public_url}")
        logger.info(f"🎬 Encoder: {video_optimizer.encoder_type}")
        logger.info(f"⚡ Warmup time: {warmup_time:.2f}s")
        logger.info("="*70)
        
    except Exception as e:
        logger.error(f"⚠️ Ngrok Error: {e}")
        public_url = f"http://localhost:{PORT}"

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup"""
    logger.info("🛑 Shutting down...")
    
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros, return_exceptions=True)
    pcs.clear()
    
    executor.shutdown(wait=False)
    


# --- API Endpoints ---
@app.post("/generate")
async def generate(request: GenerationRequest):
    """Generate video with NVENC optimization"""
    if agent is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    global request_semaphore, queue_counter
    
    async with request_semaphore:
        queue_counter += 1
        request_id = f"req_{queue_counter}_{uuid.uuid4().hex[:8]}"
        
        try:
            video_path, timings = await generate_video(request.dict(), request_id)
            active_streams[request_id] = video_path
            
            return {
                "status": "success",
                "request_id": request_id,
                "download_url": f"{public_url}/download/{request_id}",
                "download_backup":f"{public_backup_url}/download/{request_id}",
                "stream_url": f"{public_url}/stream/{request_id}",
                "video_path": video_path,
                "file_size_mb": round(os.path.getsize(video_path) / 1024 / 1024, 2),
                "resolution": f"{TARGET_IMAGE_SIZE}x{TARGET_IMAGE_SIZE}",
                "encoder": video_optimizer.encoder_type,
                "performance": {
                    "preprocess_time": round(timings['preprocess'], 3),
                    "inference_time": round(timings['inference'], 3),
                    "total_time": round(timings['total'], 3),
                    "fps": round(timings['fps'], 2)
                }
            }
        except Exception as e:
            logger.error(f"[{request_id}] Failed: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))


@app.get("/download/{request_id}")
async def download_video(request_id: str):
    """Download generated video"""
    if request_id not in active_streams:
        raise HTTPException(status_code=404, detail="Video not found")
    
    video_path = active_streams[request_id]
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video file not found")
    
    return FileResponse(
        video_path,
        media_type="video/mp4",
        filename=f"{request_id}.mp4"
    )


@app.get("/health")
async def health():
    """Health check with performance stats"""
    return {
        "status": "healthy",
        "model_loaded": agent is not None,
        "encoder": video_optimizer.encoder_type if video_optimizer else "unknown",
        "active_streams": len(active_streams),
        "performance_stats": {
            "total_requests": performance_stats['total_requests'],
            "avg_inference_time": round(
                performance_stats['total_inference_time'] / max(performance_stats['total_requests'], 1), 3
            ),
            "average_fps": round(performance_stats['average_fps'], 2)
        }
    }
if __name__ == "__main__":

    import asyncio
    from hypercorn.asyncio import serve
    from hypercorn.config import Config

    config = Config()
    config.bind = [f"0.0.0.0:{PORT}"]
    config.loglevel = "info"
    config.startup_timeout = 60  # increase if your app is slow to start

    asyncio.run(serve(app, config))

