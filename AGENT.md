# FLOAT Fast — Agent Development Guide (`agent.md`)

This document provides system architecture, coding patterns, directory layouts, and execution workflows for AI coding agents and developers working on this repository.

---

## 1. Project Overview

**FLOAT (Flow Matching for Audio-Driven Talking Portrait Video Generation - ICCV 2025)** is a deep learning system that produces temporally consistent, audio-driven talking portraits using learned orthogonal motion latent flow matching.

This repository (**`float-fast`**) provides:
1. **High-Performance FastAPI Server**: Hardware-accelerated inference with NVENC video encoding, HTTP byte-range streaming, and WebRTC signaling.
2. **Gradio Interfaces**: Batch generator (`gradio_interface.py`) and near-real-time live streaming (`float_live_app.py`) with seamless temporal continuity across chunks.
3. **Decoupled Omni Streaming Architecture**: vLLM-Omni and SGLang-inspired disaggregated pipeline engine (`FloatOmniPipelineEngine` + `FloatOmniScheduler`) supporting fair-share continuous chunk scheduling and per-request streaming callbacks (`ClipCallback`).
4. **Modular Model Architecture**: Decoupled `fast_float.models.float` package divided into focused submodules (audio encoding, decode auto-tuning, ODE sampling, and Omni streaming).
5. **Modern Python Architecture**: Managed by `uv` with pinned PyTorch CUDA wheels.

---

## 2. Directory Structure

```
float-fast/
├── AGENT.md                        # This development guide
├── README.md                       # Repository overview and quickstart
├── pyproject.toml                  # Project metadata, dependencies, uv indexes
├── uv.lock                         # Pinned dependency lockfile
├── Dockerfile                      # Container build definition (CUDA 12.1 base)
├── config/
│   └── api.json                    # Endpoint and public URL configurations
├── plans/                          # Architectural plans and technical blueprints
│   ├── original_gemini_cloud_architecture_plan.md # Original Gemini Omni Cloud architecture plan
│   └── global_omni_cloud_architecture_plan.md   # Merged Omni Cloud architecture & audit report
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
    │   ├── state.py                # AppState singleton (locks, executor, pipeline_engine)
    │   ├── service.py              # Video generation orchestrator (generate_video)
    │   ├── generate.py             # Inference pipeline & data processing
    │   ├── data_processor.py       # Face alignment, cropping, and audio feature extraction
    │   ├── nvenc_encoder.py        # Hardware FFmpeg NVENC encoder
    │   └── inference_optimized.py  # Optimized inference agent & custom encoders
    ├── sched/                      # Omni Decoupled Streaming Pipeline & Scheduler
    │   ├── __init__.py             # Exports FloatOmniPipelineEngine, FloatOmniScheduler, types
    │   ├── types.py                # FloatOmniRequest, LatentChunkTask, ClipCallback, RequestStatus
    │   ├── omni_scheduler.py       # Continuous round-robin fair-share chunk scheduler
    │   ├── pipeline_engine.py      # Decoupled Sampler worker, Latent Queue, and Decoder pool
    │   ├── base_scheduler.py       # Reference vLLM-Omni scheduler base
    │   ├── request_scheduler.py    # Request-level scheduler
    │   └── step_scheduler.py       # Step-level scheduler
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
    │   ├── pyav_writer.py          # In-process CPU PyAV and Torchcodec video writers
    │   └── video_optimizer.py      # VideoSaveOptimizer (NVENC / PyAV / Torchcodec / CPU fallback)
    ├── models/                     # PyTorch neural network architectures
    │   ├── __init__.py
    │   ├── float/                  # Modular FLOAT model architecture
    │   │   ├── __init__.py         # Package exports (FLOAT, FMT, Generator, Encoders, Tuners)
    │   │   ├── FLOAT.py            # Central FLOAT model coordinating submodules
    │   │   ├── audio_encoder.py    # AudioEncoder, Audio2Emotion, and FrameData
    │   │   ├── video_saver.py      # DecodeProfile, DecodeAutoTuner, and StreamingVideoSaver
    │   │   ├── omni_ops.py         # OmniStreamingMixin (prepare_omni_conditions, chunk sampler/decoder)
    │   │   ├── inference_ops.py    # FullInferenceMixin (sample, batched decoding, inference_stream)
    │   │   ├── FMT.py              # Flow Matching Transformer (ODE hotspot)
    │   │   ├── encoder.py          # Reference portrait appearance encoder
    │   │   ├── generator.py        # Motion Autoencoder Generator
    │   │   └── styledecoder.py     # Convolutional StyleDecoder (synthesis hotspot)
    │   ├── wav2vec2.py             # Wav2Vec2 feature extraction wrapper
    │   └── wav2vec2_ser.py         # Speech emotion recognition network
    ├── interfaces/                 # Web interfaces
    │   ├── gradio_interface.py     # Batch UI for image + audio upload
    │   ├── float_live_app.py       # Live mic streaming UI with seamless temporal continuity
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

## 6. Omni Decoupled Streaming Architecture

The repository incorporates the disaggregated pipeline paradigm inspired by **vLLM-Omni** and **SGLang**:

```
Client Request (Audio + Reference Portrait + Callback)
         │
         ▼
FloatOmniPipelineEngine.submit_request()
         │  (one-time invariant precomputation)
         ▼
FLOAT.prepare_omni_conditions()  ← Wav2Vec2 + image identity encoding
         │
         ▼
FloatOmniScheduler.add_request()  ← Fair-share round-robin queue
         │
         ▼
┌── [_sampler_loop] ──────────────────────────────────────────┐
│   req = scheduler.schedule_next_chunk()                     │ Fair-share
│   task = await model.sample_omni_chunk(req)                 │ chunk interleaving
│   await latent_queue.put(task)                              │ across all active requests
└─────────────────────────────────────────────────────────────┘
                               │
                    asyncio.Queue[LatentChunkTask]
                               │
┌── [_decoder_loop × N] ───────┘
│   task = await latent_queue.get()
│   clip = await model.decode_omni_chunk(task)
│   await task.callback(request_id, chunk_idx, clip, is_last)
└──► Output Dispatcher (WebRTC / WebSocket / SSE / MP4)
```

### Key Advantages:
1. **Low Time-To-First-Frame (TTFF)**: Drops from ~15–20 seconds down to **< 500ms** by dispatching fine-grained chunks (50 frames / ~2 seconds) instead of waiting for the full video.
2. **Hardware Concurrency**: The ODE sampler and StyleDecoder run concurrently on separate CUDA streams or devices.
3. **Seamless Temporal Continuity**: By carrying motion latents (`prev_x`, `prev_wa`) between chunks, consecutive video clips flow without visible boundary seams or mouth glitches.

---

## 7. Modular `fast_float.models.float` Subpackages

[`fast_float/models/float/`](file:///c:/Users/emam2/tech_projects/float-fast/fast_float/models/float) is organized into distinct, modular components:

1. **[`audio_encoder.py`](file:///c:/Users/emam2/tech_projects/float-fast/fast_float/models/float/audio_encoder.py)**:
   - `AudioEncoder`: Wav2Vec2 feature extraction and linear projection to motion space.
   - `Audio2Emotion`: 7-class speech emotion classification.
   - `FrameData`: Container for frame chunks.
2. **[`video_saver.py`](file:///c:/Users/emam2/tech_projects/float-fast/fast_float/models/float/video_saver.py)**:
   - `DecodeProfile` & `DecodeAutoTuner`: Dynamic GPU VRAM profiling and auto-tuning.
   - `StreamingVideoSaver`: Mixin coordinating multi-GPU decode workers and hardware encoders (FFmpeg NVENC, CPU PyAV, Torchcodec).
3. **[`omni_ops.py`](file:///c:/Users/emam2/tech_projects/float-fast/fast_float/models/float/omni_ops.py)**:
   - `OmniStreamingMixin`: Implements `prepare_omni_conditions`, `sample_omni_chunk`, and `decode_omni_chunk`.
4. **[`inference_ops.py`](file:///c:/Users/emam2/tech_projects/float-fast/fast_float/models/float/inference_ops.py)**:
   - `FullInferenceMixin`: Implements `sample`, `_decode_latent_batched`, `decode_latent_into_image`, `inference_stream`, and `inference`.
5. **[`FLOAT.py`](file:///c:/Users/emam2/tech_projects/float-fast/fast_float/models/float/FLOAT.py)**:
   - Core `FLOAT` class inheriting from `BaseModel`, `StreamingVideoSaver`, `OmniStreamingMixin`, and `FullInferenceMixin`.
   - Re-exports all components for 100% backward compatibility.

---

## 8. Coding Patterns & Concurrency Rules for Agents

When editing or extending the codebase, adhere to these guidelines:

### 1. Concurrency Management
- Model inference is GPU-bound and must not run concurrently on the same device without scheduling.
- In FastAPI: Always execute through `app_state.executor` (or `app_state.pipeline_engine`) while respecting queue limits.
- In Gradio Live App: Pre-encode reference face identity once, then stream audio chunks with temporal continuity (`prev_x`, `prev_wa`).

### 2. Imports Convention
- Always use absolute imports rooted at `fast_float`:
  ```python
  from fast_float.schemas.generation import GenerationRequest
  from fast_float.core.state import app_state
  from fast_float.models.float.FLOAT import FLOAT
  from fast_float.sched.pipeline_engine import FloatOmniPipelineEngine
  ```
- Keep fallback imports if a script might be executed directly within subdirectories.

### 3. Pydantic V2 Usage
- Use `.model_dump()` rather than deprecated `.dict()`.
- If a schema has a field beginning with `model_` (e.g., `model_loaded`), set:
  ```python
  model_config = {"protected_namespaces": ()}
  ```

### 4. Audio Processing
- Route audio conversions through `fast_float.utils.media.process_audio_format`.
- The neural network expects single-channel 16,000 Hz float32 audio.

### 5. Video Compression & Encoders
- Video encoding uses `VideoSaveOptimizer` (`fast_float.utils.video_optimizer`) or `fast_float.utils.pyav_writer`.
- Supported backends:
  - **FFmpeg NVENC**: NVIDIA hardware encoder (`h264_nvenc`).
  - **Torchcodec**: PyTorch CUDA / CPU video encoder.
  - **CPU PyAV**: Zero-subprocess in-process libx264 software encoder.
