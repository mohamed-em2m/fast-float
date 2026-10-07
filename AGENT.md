# FLOAT Fast — Agent Development Guide (`agent.md`)

This document provides system architecture, coding patterns, directory layouts, and execution workflows for AI coding agents and developers working on this repository.

---

## 1. Project Overview

**FLOAT (Flow Matching for Audio-Driven Talking Portrait Video Generation - ICCV 2025)** is a deep learning system that produces temporally consistent, audio-driven talking portraits using learned orthogonal motion latent flow matching.

This repository (**`float-fast`**) provides:
1. **High-Performance FastAPI Server**: Hardware-accelerated inference with NVENC video encoding, HTTP byte-range streaming, and WebRTC signaling.
2. **Gradio Interfaces**: Batch generator (`gradio_interface.py`) and live audio streaming (`float_live_app.py`).
3. **Modern Python Architecture**: Modular subpackages managed by `uv` with pinned PyTorch CUDA 11.8 wheels.

---

## 2. Directory Structure

```
float-fast/
├── agent.md                        # This development guide
├── pyproject.toml                  # Project metadata, dependencies, uv indexes
├── uv.lock                         # Pinned dependency lockfile
├── Dockerfile                      # Container build definition (CUDA 12.1 base)
├── config/
│   └── api.json                    # Endpoint and public URL configurations
├── scripts/
│   ├── run_api.sh                  # Launch FastAPI server via hypercorn
│   ├── run_gradio.sh               # Launch Gradio batch web app
│   ├── run_live.sh                 # Launch Gradio live audio streaming app
│   ├── setup.sh                    # Setup environment and checkpoints
│   ├── download_checkpoints.sh     # Fetch model weights
│   └── environments.sh             # Run `uv sync --frozen`
├── checkpoints/                    # Model weights directory
│   ├── float.pth
│   ├── wav2vec2-base-960h/
│   └── wav2vec-english-speech-emotion-recognition/
└── fast_float/                     # Main Python application package
    ├── __init__.py
    ├── main.py                     # FastAPI application entrypoint with lifespan
    ├── initialize_env.py           # Multi-path environment initialization
    ├── core/                       # Inference pipeline & shared state
    │   ├── __init__.py
    │   ├── state.py                # AppState singleton (locks, executor, active streams)
    │   ├── service.py              # Video generation orchestrator (generate_video)
    │   ├── generate.py             # Inference pipeline & data processing
    │   └── inference_optimized.py  # Optimized inference agent & custom encoders
    ├── routers/                    # Modular FastAPI endpoint routers
    │   ├── __init__.py             # Aggregate APIRouter
    │   ├── generation.py           # POST /generate, GET /download/{id}, GET /stream/{id}
    │   ├── health.py               # GET /health, GET /gpu, POST /gpu/clear
    │   └── webrtc.py               # POST /offer (WebRTC SDP signaling & media player)
    ├── schemas/                    # Pydantic V2 data models & options
    │   ├── __init__.py
    │   ├── base_options.py         # BaseOptions argparse configuration parser
    │   └── generation.py           # GenerationRequest, GenerationResponse, HealthResponse, WebRTC schemas
    ├── utils/                      # Utilities & hardware accelerators
    │   ├── __init__.py
    │   ├── gpu.py                  # Safe CUDA query, VRAM usage tracking, cache clearing
    │   ├── logger.py               # Structured colored console logger
    │   ├── media.py                # PCM to WAV conversions, image resizing, default placeholders
    │   └── video_optimizer.py      # VideoSaveOptimizer (NVENC / QSV / AMF / CPU fallback)
    ├── models/                     # PyTorch neural network architectures
    │   ├── __init__.py
    │   ├── float/                  # FLOAT generator, FMT, style decoder, encoders
    │   │   ├── FLOAT.py
    │   │   ├── FMT.py
    │   │   ├── encoder.py
    │   │   ├── generator.py
    │   │   └── styledecoder.py
    │   ├── wav2vec2.py             # Wav2Vec2 feature extraction wrapper
    │   └── wav2vec2_ser.py         # Speech emotion recognition network
    ├── interfaces/                 # Web interfaces
    │   ├── gradio_interface.py     # Batch UI for image + audio upload
    │   ├── float_live_app.py       # Live mic streaming UI
    │   └── test_gradio.py          # Testing UI with WebRTC player integration
    └── options/                    # Backward-compatibility shims for legacy imports
        ├── __init__.py
        └── base_options.py
```

---

## 3. Environment & Dependencies

### Package Manager
We use **`uv`**. Never use bare `pip` directly inside the virtual environment unless specified.

```bash
# Sync all dependencies from uv.lock
uv sync --frozen

# Run commands within the environment
uv run <command>
```

### PyTorch Wheels
PyTorch CUDA 11.8 wheels are pinned in `pyproject.toml`:
```toml
[[tool.uv.index]]
name = "pytorch-cu118"
url = "https://download.pytorch.org/whl/cu118"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu118" }
torchvision = { index = "pytorch-cu118" }
torchaudio = { index = "pytorch-cu118" }
```

---

## 4. How to Run

### 1. Launch FastAPI Server
```bash
# Direct run with uv and hypercorn
uv run hypercorn fast_float.main:app --bind 0.0.0.0:8000 --log-level info

# Or using the shell script
./scripts/run_api.sh
```

### 2. Launch React Studio Frontend
```bash
# Using npm directly inside frontend/
cd frontend && npm run dev

# Or using the root launcher script
./scripts/run_frontend.sh
# Open http://localhost:3000 in browser
```

### 3. Launch Gradio Batch App
```bash
uv run python fast_float/interfaces/gradio_interface.py
```

### 4. Launch Gradio Live App
```bash
uv run python fast_float/interfaces/float_live_app.py
```

---

## 5. API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Service health status and links |
| `POST` | `/generate` | Generate talking portrait video from audio + image |
| `GET` | `/download/{request_id}` | Download generated MP4 video |
| `GET` | `/stream/{request_id}` | Stream generated MP4 with HTTP `Range` header support |
| `GET` | `/health` | System health, model status, and FPS statistics |
| `GET` | `/gpu` | Real-time GPU VRAM allocated and reserved in GB |
| `POST` | `/gpu/clear` | Clear CUDA cache and run garbage collection |
| `POST` | `/offer` | WebRTC SDP negotiation for real-time video playback |

### Example `/generate` Request Body
```json
{
  "audio_base64": "<base64-pcm-or-wav>",
  "audio_format": "pcm",
  "sample_rate": 16000,
  "channels": 1,
  "sample_width": 2,
  "image_base64": null,
  "emotion": "S2E",
  "nfe": 10,
  "seed": 25,
  "a_cfg_scale": 2.0,
  "r_cfg_scale": 1.0,
  "e_cfg_scale": 1.0,
  "no_crop": false,
  "compress": true,
  "quality": "balanced"
}
```

---

## 6. Architecture & Concurrency Rules for Agents

When editing or extending the codebase, follow these design patterns:

### 1. Concurrency Management
- Model inference is GPU-bound and must **not** run concurrently on the same device.
- Always execute inference through `app_state.executor` (ThreadPoolExecutor) while holding `app_state.inference_lock`.
- Use `app_state.request_semaphore` to limit the queue depth of pending requests.

### 2. Imports Convention
- Always use absolute imports rooted at `fast_float`:
  ```python
  from fast_float.schemas.generation import GenerationRequest
  from fast_float.core.state import app_state
  from fast_float.utils.gpu import get_gpu_info
  ```
- Keep fallback imports if the script might be executed directly inside `fast_float/`.

### 3. Pydantic V2 Usage
- Use `.model_dump()` rather than deprecated `.dict()`.
- If a schema has a field beginning with `model_` (e.g., `model_loaded`), set:
  ```python
  model_config = {"protected_namespaces": ()}
  ```

### 4. Audio Processing
- Always route audio conversions through `fast_float.utils.media.process_audio_format`.
- The neural network expects single-channel 16,000 Hz float32 audio.

### 5. Video Compression & Encoders
- Video encoding uses `VideoSaveOptimizer` (`fast_float.utils.video_optimizer`).
- Prefer NVENC hardware encoding (`h264_nvenc`) with fallback to `libx264 ultrafast`.
