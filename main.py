from initialize_env import initialize_environment
initialize_environment()
import os
import json
import base64
import tempfile
import datetime
import asyncio
import io
import logging
import time
from typing import Optional, Dict, Tuple, List, Union
import uuid
import subprocess
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, StreamingResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field
import cv2
import numpy as np
from PIL import Image
# Audio processing
from pydub import AudioSegment
import soundfile as sf
from inference_optimized import OptimizedInferenceAgent, InferenceOptions

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

# Model optimization settings
OPTIMIZATION_MODE = "all"
USE_FP16 = True
COMPRESS_RESPONSES = True

app = FastAPI(title="FLOAT WebRTC API - NVENC Optimized", version="5.2.0")

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
                return True
            return False
        except:
            return False
    
    def _detect_best_encoder(self) -> bool:
        gpu_encoders = [
            ("h264_nvenc", "NVENC (NVIDIA GPU)"),
            ("hevc_nvenc", "NVENC HEVC (NVIDIA GPU)"),
            ("h264_qsv", "QuickSync (Intel GPU)"),
            ("h264_videotoolbox", "VideoToolbox (Apple)"),
            ("h264_amf", "AMF (AMD GPU)")
        ]
        
        try:
            result = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=3)
            encoders_output = result.stdout
            
            for encoder, name in gpu_encoders:
                if encoder in encoders_output:
                    self.encoder_type = name
                    self.selected_encoder = encoder
                    return True
            
            self.encoder_type = "libx264 (CPU - ultrafast)"
            self.selected_encoder = "libx264"
            return True
            
        except:
            self.encoder_type = "libx264 (CPU - fallback)"
            self.selected_encoder = "libx264"
            return False

# --- Data Models ---

# Single Request Model
class GenerationRequest(BaseModel):
    audio_base64: str
    audio_format: str = "pcm"
    sample_rate: int = 16000
    channels: int = 1
    sample_width: int = 2
    image_base64: Optional[str] = None
    image_format: Optional[str] = "jpg"
    emotion: str = "S2E"
    nfe: int = 10
    seed: int = 25
    a_cfg_scale: float = 2.0
    r_cfg_scale: float = 1.0
    e_cfg_scale: float = 1.0
    no_crop: bool = False
    compress: bool = True
    quality: str = "balanced"

# Batch Request Models
class BatchItem(BaseModel):
    id: str  # Unique ID for the client to track this specific item
    audio_base64: str
    image_base64: Optional[str] = None

class BatchGenerationRequest(BaseModel):
    items: List[BatchItem]
    # Shared settings for the whole batch
    audio_format: str = "pcm"
    sample_rate: int = 16000
    channels: int = 1
    sample_width: int = 2
    image_format: str = "jpg"
    emotion: str = "S2E"
    nfe: int = 10
    seed: int = 25
    a_cfg_scale: float = 2.0
    r_cfg_scale: float = 1.0
    e_cfg_scale: float = 1.0
    no_crop: bool = False
    batch_size: Optional[int] = None # Helper for the inference engine


# --- Utility Functions ---
def ensure_default_image():
    if not os.path.exists(DEFAULT_IMAGE_PATH):
        img = np.zeros((TARGET_IMAGE_SIZE, TARGET_IMAGE_SIZE, 3), dtype=np.uint8)
        cv2.putText(img, "Default", (TARGET_IMAGE_SIZE//4, TARGET_IMAGE_SIZE//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imwrite(DEFAULT_IMAGE_PATH, img)

def resize_image_to_256(image_path: str, output_path: str = None) -> str:
    try:
        if output_path is None:
            output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg").name
        img = Image.open(image_path)
        if img.mode == 'RGBA':
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        img.save(output_path, 'JPEG', quality=95)
        return output_path
    except Exception as e:
        logger.error(f"Failed to process image: {e}")
        raise

def convert_pcm_to_wav(pcm_data: bytes, sample_rate: int = 16000, 
                       channels: int = 1, sample_width: int = 2) -> bytes:
    try:
        if sample_width == 1:
            dtype = np.uint8
            max_val = 128.0
            offset = 128
        elif sample_width == 2:
            dtype = np.int16
            max_val = 32768.0
            offset = 0
        else:
            dtype = np.int16
            max_val = 32768.0
            offset = 0
        
        audio_array = np.frombuffer(pcm_data, dtype=dtype)
        if len(audio_array) == 0: raise ValueError("Empty audio data")
        
        if channels > 1 and len(audio_array) % channels != 0:
            audio_array = audio_array[:len(audio_array) - (len(audio_array) % channels)]
            audio_array = audio_array.reshape(-1, channels)
        
        if dtype == np.uint8:
            audio_float = (audio_array.astype(np.float32) - offset) / max_val
        else:
            audio_float = audio_array.astype(np.float32) / max_val
        
        audio_float = np.clip(audio_float, -1.0, 1.0)
        
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, audio_float, sample_rate, format='WAV', subtype='PCM_16')
        wav_buffer.seek(0)
        return wav_buffer.read()
    except Exception as e:
        logger.error(f"PCM to WAV conversion failed: {e}")
        raise

def process_audio_format(audio_content: str, ext: str, 
                        sample_rate: int = 16000,
                        channels: int = 1, 
                        sample_width: int = 2) -> bytes:
    try:
        audio_bytes = base64.b64decode(audio_content)
        ext = ext.lower().lstrip('.')
        
        if ext in ['pcm', 'raw', 'wav']:
            wav_bytes = convert_pcm_to_wav(audio_bytes, sample_rate, channels, sample_width)
        else:
            try:
                audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes), format=ext)
                if audio_segment.channels > 1: audio_segment = audio_segment.set_channels(1)
                if audio_segment.frame_rate != sample_rate: audio_segment = audio_segment.set_frame_rate(sample_rate)
                wav_buffer = io.BytesIO()
                audio_segment.export(wav_buffer, format='wav')
                wav_buffer.seek(0)
                wav_bytes = wav_buffer.read()
            except:
                wav_bytes = convert_pcm_to_wav(audio_bytes, sample_rate, channels, sample_width)
        return wav_bytes
    except Exception as e:
        logger.error(f"Audio processing failed: {e}")
        raise

# --- Inference Wrappers ---

def _run_inference_blocking(video_path_raw, image_path, audio_path, request_data):
    """Synchronous wrapper for single inference"""
    return agent.run_inference(
        res_video_paths=[video_path_raw],
        ref_paths=[image_path],
        audio_paths=[audio_path],
        a_cfg_scale=request_data.get("a_cfg_scale", 2.0),
        r_cfg_scale=request_data.get("r_cfg_scale", 1.0),
        e_cfg_scale=request_data.get("e_cfg_scale", 1.0),
        emo=request_data.get("emotion", "S2E"),
        nfe=request_data.get("nfe", 10),
        no_crop=request_data.get("no_crop", False),
        seed=request_data.get("seed", 25),
        verbose=True
    )[0]

def _run_batch_inference_blocking(res_paths, ref_paths, aud_paths, request_data: BatchGenerationRequest):
    """Synchronous wrapper for BATCH inference"""
    return agent.run_inference(
        res_video_paths=res_paths,
        ref_paths=ref_paths,
        audio_paths=aud_paths,
        a_cfg_scale=request_data.a_cfg_scale,
        r_cfg_scale=request_data.r_cfg_scale,
        e_cfg_scale=request_data.e_cfg_scale,
        emo=request_data.emotion,
        nfe=request_data.nfe,
        no_crop=request_data.no_crop,
        seed=request_data.seed,
        verbose=True,
        batch_size=request_data.batch_size
    )

async def generate_video(request_data: Dict, request_id: str) -> Tuple[str, Dict]:
    """Single video generation"""
    global inference_lock
    temp_files = []
    total_start = time.time()
    timings = {}
    
    try:
        loop = asyncio.get_running_loop()
        
        # Audio
        preprocess_start = time.time()
        wav_bytes = process_audio_format(
            request_data["audio_base64"], request_data.get("audio_format", "pcm"),
            request_data.get("sample_rate", 16000), request_data.get("channels", 1), request_data.get("sample_width", 2)
        )
        audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        with open(audio_path, 'wb') as f: f.write(wav_bytes)
        temp_files.append(audio_path)
        
        # Image
        if request_data.get("image_base64"):
            image_bytes = base64.b64decode(request_data["image_base64"])
            image_path_temp = tempfile.NamedTemporaryFile(delete=False, suffix=f".{request_data.get('image_format', 'jpg')}").name
            with open(image_path_temp, 'wb') as f: f.write(image_bytes)
            temp_files.append(image_path_temp)
            image_path = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg").name
            resize_image_to_256(image_path_temp, image_path)
            temp_files.append(image_path)
        else:
            if not os.path.exists(DEFAULT_IMAGE_PATH): ensure_default_image()
            image_path = DEFAULT_IMAGE_PATH
            
        timings['preprocess'] = time.time() - preprocess_start
        
        # Inference
        video_path_final = os.path.join(opt.res_dir, f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{request_id}.mp4")
        inference_start = time.time()
        
        async with inference_lock:
            output_path = await loop.run_in_executor(
                executor, _run_inference_blocking, video_path_final, image_path, audio_path, request_data
            )
        
        timings['inference'] = time.time() - inference_start
        timings['total'] = time.time() - total_start
        timings['fps'] = 1.0 / timings['inference'] if timings['inference'] > 0 else 0
        
        performance_stats['total_requests'] += 1
        performance_stats['total_inference_time'] += timings['inference']
        if performance_stats['total_inference_time'] > 0:
            performance_stats['average_fps'] = performance_stats['total_requests'] / performance_stats['total_inference_time']
            
        return output_path, timings
        
    except Exception as e:
        raise e
    finally:
        for f in temp_files:
            if os.path.exists(f): os.remove(f)

async def generate_video_batch(request_data: BatchGenerationRequest, batch_id: str) -> Tuple[List[Dict], Dict]:
    """Batch video generation"""
    global inference_lock
    temp_files = []
    total_start = time.time()
    timings = {}
    
    try:
        loop = asyncio.get_running_loop()
        preprocess_start = time.time()
        
        ref_paths = []
        audio_paths = []
        res_paths = []
        item_ids = []
        
        # Preprocess all items in the batch
        for i, item in enumerate(request_data.items):
            item_id = item.id
            
            # Audio
            wav_bytes = process_audio_format(
                item.audio_base64, request_data.audio_format,
                request_data.sample_rate, request_data.channels, request_data.sample_width
            )
            aud_p = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{i}.wav").name
            with open(aud_p, 'wb') as f: f.write(wav_bytes)
            audio_paths.append(aud_p)
            temp_files.append(aud_p)
            
            # Image
            if item.image_base64:
                img_bytes = base64.b64decode(item.image_base64)
                img_t = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{i}.{request_data.image_format}").name
                with open(img_t, 'wb') as f: f.write(img_bytes)
                temp_files.append(img_t)
                
                img_p = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{i}.jpg").name
                resize_image_to_256(img_t, img_p)
                ref_paths.append(img_p)
                temp_files.append(img_p)
            else:
                if not os.path.exists(DEFAULT_IMAGE_PATH): ensure_default_image()
                ref_paths.append(DEFAULT_IMAGE_PATH)
            
            # Output path
            safe_id = "".join([c for c in item_id if c.isalnum() or c in ['_','-']])
            out_p = os.path.join(opt.res_dir, f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{batch_id}_{safe_id}.mp4")
            res_paths.append(out_p)
            item_ids.append(item_id)
            
        timings['preprocess'] = time.time() - preprocess_start
        
        # Run Batch Inference
        inference_start = time.time()
        async with inference_lock:
            completed_paths = await loop.run_in_executor(
                executor, _run_batch_inference_blocking, res_paths, ref_paths, audio_paths, request_data
            )
            
        timings['inference'] = time.time() - inference_start
        timings['total'] = time.time() - total_start
        timings['fps'] = len(completed_paths) / timings['inference'] if timings['inference'] > 0 else 0
        
        # Structure results
        results = []
        for idx, path in enumerate(completed_paths):
            results.append({
                "id": item_ids[idx],
                "path": path,
                "file_size_mb": round(os.path.getsize(path) / 1024 / 1024, 2) if os.path.exists(path) else 0
            })
            
        return results, timings
        
    except Exception as e:
        logger.error(f"Batch failed: {e}")
        raise
    finally:
        for f in temp_files:
            try:
                if os.path.exists(f): os.remove(f)
            except: pass

# --- FastAPI Event Handlers ---
@app.on_event("startup")
async def startup_event():
    global agent, opt, public_url, request_semaphore, video_optimizer, executor, inference_lock, public_backup_url
    
    request_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    inference_lock = asyncio.Lock()
    video_optimizer = VideoSaveOptimizer()
    ensure_default_image()
    
    import sys
    original_argv = sys.argv.copy()
    sys.argv = [sys.argv[0]]
    opt = InferenceOptions().parse()
    opt.rank = 0
    opt.ngpus = 1
    opt.res_dir = RESULTS_DIR
    opt.optimization_mode = OPTIMIZATION_MODE
    opt.use_fp16 = USE_FP16
    sys.argv = original_argv
    os.makedirs(opt.res_dir, exist_ok=True)
    
    logger.info("🤖 Loading FLOAT model...")
    agent = OptimizedInferenceAgent(opt)
    
    # Warmup
    logger.info("🔥 Warming up model...")
    try:
        warmup_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        warmup_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        sf.write(warmup_audio, np.zeros(8000), 16000)
        
        await asyncio.get_event_loop().run_in_executor(
            executor, _run_inference_blocking, warmup_video, DEFAULT_IMAGE_PATH, warmup_audio, {}
        )
        logger.info("✅ Warmup complete")
    except Exception as e:
        logger.warning(f"⚠️ Warmup error: {e}")

    public_url = os.environ.get("AVATER_PUBLIC_URL", f"http://localhost:{PORT}") 
    public_backup_url = os.environ.get("BACKUP_AVATER_PUBLIC_URL", public_url)

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("🛑 Shutting down...")
    executor.shutdown(wait=False)

# --- API Endpoints ---
@app.post("/generate")
async def generate(request: GenerationRequest):
    """Generate single video"""
    if agent is None: raise HTTPException(status_code=503, detail="Model not loaded")
    global queue_counter
    
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
                "download_backup": f"{public_backup_url}/download/{request_id}",
                "video_path": video_path,
                "performance": timings
            }
        except Exception as e:
            logger.error(f"[{request_id}] Failed: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate/batch")
async def generate_batch(request: BatchGenerationRequest):
    """Generate multiple videos in a batch"""
    if agent is None: raise HTTPException(status_code=503, detail="Model not loaded")
    global queue_counter
    
    async with request_semaphore:
        queue_counter += 1
        batch_id = f"batch_{queue_counter}_{uuid.uuid4().hex[:8]}"
        
        logger.info(f"[{batch_id}] 🚀 Starting batch generation for {len(request.items)} items")
        
        try:
            results, timings = await generate_video_batch(request, batch_id)
            
            response_items = []
            for item in results:
                # Use item ID from client or internal ID
                res_id = item['id']
                # We can store the path using the ID as key, or generate a specific download key
                # Here we use the ID provided by the client combined with batch ID to ensure uniqueness in stream map
                stream_key = f"{batch_id}_{res_id}"
                active_streams[stream_key] = item['path']
                
                response_items.append({
                    "id": res_id,
                    "status": "success",
                    "download_url": f"{public_url}/download/{stream_key}",
                    "download_backup": f"{public_backup_url}/download/{stream_key}",
                    "file_size_mb": item['file_size_mb']
                })

            return {
                "status": "success",
                "batch_id": batch_id,
                "results": response_items,
                "performance": {
                    "total_time": round(timings['total'], 3),
                    "preprocess_time": round(timings['preprocess'], 3),
                    "inference_time": round(timings['inference'], 3),
                    "fps_effective": round(timings['fps'], 2)
                }
            }
        except Exception as e:
            logger.error(f"[{batch_id}] Batch Failed: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/{request_id}")
async def download_video(request_id: str):
    if request_id not in active_streams: raise HTTPException(status_code=404, detail="Video not found")
    path = active_streams[request_id]
    if not os.path.exists(path): raise HTTPException(status_code=404, detail="File lost")
    return FileResponse(path, media_type="video/mp4", filename=f"{request_id}.mp4")

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": agent is not None, "performance": performance_stats}

if __name__ == "__main__":
    import asyncio
    from hypercorn.asyncio import serve
    from hypercorn.config import Config
    config = Config()
    config.bind = [f"0.0.0.0:{PORT}"]
    asyncio.run(serve(app, config))