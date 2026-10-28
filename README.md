<div align="center">

<p align="center">
  <img src="assets/logo.png" alt="FLOAT Fast Logo" width="140" style="border-radius: 24px; margin-bottom: 14px;" />
</p>

# FLOAT Fast

High-performance, hardware-accelerated serving and inference stack for audio-driven talking portrait generation (FLOAT, ICCV 2025).

[![Build Status](https://img.shields.io/github/actions/workflow/status/mohamed-em2m/float-fast/ci.yml?branch=main)](https://github.com/mohamed-em2m/float-fast/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python: 3.10](https://img.shields.io/badge/Python-3.10-3776AB.svg?logo=python&logoColor=white)](pyproject.toml)
[![PyTorch: 2.7.1+cu126](https://img.shields.io/badge/PyTorch-2.7.1%2Bcu126-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.122-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React: 19](https://img.shields.io/badge/React-19-61DAFB.svg?logo=react&logoColor=black)](frontend/)
[![Version](https://img.shields.io/badge/version-v5.2.0-blue.svg)](https://github.com/mohamed-em2m/float-fast/releases)

[Demo](http://localhost:3000) · [Documentation](docs/USAGE.md) · [Report Bug](https://github.com/mohamed-em2m/float-fast/issues) · [Request Feature](https://github.com/mohamed-em2m/float-fast/issues)

</div>

---

## Table of Contents

- [About](#about)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [One-Command Automated Setup (`scripts/setup.sh`)](#one-command-automated-setup-scriptssetupsh)
  - [Manual Installation](#manual-installation)
- [Running the System](#running-the-system)
  - [1. High-Performance Dual-Process Architecture (`scripts/run_dual_process.sh`)](#1-high-performance-dual-process-architecture-scriptsrun_dual_processsh)
  - [2. All-in-One Dev Runner (`scripts/run_all.sh`)](#2-all-in-one-dev-runner-scriptsrun_allsh)
  - [3. CLI Supervisor (`fast-float`)](#3-cli-supervisor-fast-float)
  - [4. Running Individual Components](#4-running-individual-components)
- [API Usage](#api-usage)
  - [REST Endpoint Example](#rest-endpoint-example)
  - [Python Client Example](#python-client-example)
- [Configuration](#configuration)
- [Hardware Acceleration & ONNX/TensorRT](#hardware-acceleration--onnxtensorrt)
- [Testing & Quality Checks](#testing--quality-checks)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## About

**FLOAT Fast** is a solo open-source project providing a production-grade inference engine and real-time interactive avatar platform built around **FLOAT (Flow Matching for Audio-Driven Talking Portrait Video Generation, ICCV 2025)**.

Standard talking-head generation models are computationally intensive, slow to render, and traditionally restricted to offline batch video synthesis. **FLOAT Fast** solves these challenges by combining:

1. **Dual-Process Architecture**: Decouples GPU inference execution from async I/O handling via ZeroMQ and POSIX Shared Memory (`shm_buffer`), eliminating event-loop blocking and memory bottlenecks.
2. **Low-Latency Streaming Pipeline**: Stream video frames in real time over WebSockets, WebRTC, and LiveKit while inference is still in-flight.
3. **Hardware Acceleration & Batched Decoding**: Micro-batched latent decoding, NVENC hardware encoding, and per-component ONNX/TensorRT execution providers.
4. **Conversational Live Voice Agent**: Real-time bidirectional voice conversations driven by Gemini Live and LiveKit Agents with instant lip-sync.
5. **Modern Studio Interface**: A unified React 19 + Vite web studio supporting both batch generation and low-latency interactive live audio streaming.

> [!IMPORTANT]
> **Upstream Research & License Notice**:
> Users are advised to review the upstream [FLOAT repository](https://github.com/deepbrainai-research/float) and read the original [FLOAT license](https://github.com/deepbrainai-research/float/blob/main/LICENSE) regarding the underlying model weights and architecture.

---

## Key Features

- ✅ **Dual-Process IPC Architecture**: Dedicated GPU worker process linked to the async FastAPI/LiveKit streaming server via ZeroMQ control sockets and zero-copy shared memory frame ring buffers.
- ✅ **Ultra-Fast Video Generation**: Hardware-accelerated inference with auto-tuned batched latent decoding and NVENC/QSV/AMF encoding.
- ✅ **Real-Time Live Transports**: Push frames straight off the decoder over WebSocket, WebRTC, or LiveKit SFU before the MP4 file is finished.
- ✅ **Conversational Voice Avatar**: Live speech-to-speech interaction powered by Gemini Live and LiveKit Agents (`fast_float/livekit/gemini/`).
- ✅ **Per-Component Backend Optimization**: Run each model unit in PyTorch Eager, `torch.compile`, or ONNX/TensorRT with automated parity verification.
- ✅ **Unified React 19 Studio**: Synchronized layouts and design language between Studio generation and interactive Live Audio modes.
- ✅ **Comprehensive Setup & CLI**: Complete single-command installation (`scripts/setup.sh`) and unified CLI supervisor (`fast-float`).

---

## Architecture

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        Fast Float Architecture                         │
└────────────────────────────────────────────────────────────────────────┘

 [Browser / React 19 Studio] ─── WebSocket / WebRTC / HTTP Range
             │
             ▼
 ┌──────────────────────────────────────────────────────────────────────┐
 │ Process 2: Streaming Backend (FastAPI / Hypercorn / LiveKit)        │
 │  - /generate, /stream/{id}, /ws/stream/{id}, WebRTC signaling       │
 │  - LiveKit SFU Session management & Gemini Live Agent coordination  │
 └───────────────────────────────────┬──────────────────────────────────┘
                                     │ ZeroMQ (Control: 5555, Frames: 5556)
                                     │ + POSIX Shared Memory (float_frame_ring)
                                     ▼
 ┌──────────────────────────────────────────────────────────────────────┐
 │ Process 1: Dedicated GPU Inference Worker                           │
 │  - Zero-copy shared memory frame exchange                           │
 │  - FLOAT Flow Matching ODE Solver + Style Latent Generator           │
 │  - Batched Latent Decoding (VRAM Auto-Tuned / TensorRT / CUDA)      │
 └──────────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology | Details |
|---|---|---|
| **Runtime & Language** | Python 3.10, JavaScript (ESNext / JSX) | Pinned via `.python-version` and `pyproject.toml` |
| **Backend & Serving** | FastAPI, Hypercorn, Uvicorn, ZeroMQ | Async REST, WebSockets, ZeroMQ IPC control |
| **Deep Learning** | PyTorch 2.7.1 (CUDA 12.6), ONNX Runtime GPU 1.23, TensorRT 10.9 | Hardware-accelerated flow matching and decoding |
| **Real-time Media** | LiveKit SFU, LiveKit Agents, aiortc (WebRTC), PyAV | WebRTC tracks, live microphone streaming |
| **Frontend Studio** | React 19, Vite 8, Lucide Icons | Responsive glassmorphism interface |
| **Package Management** | `uv`, `pnpm` | Fast, deterministic dependency management |

---

## Project Structure

```text
float-fast/
├── .gitattributes          # Line-ending and binary tracking configuration
├── .github/                # GitHub Actions CI/CD workflows
├── assets/                 # Branding assets, logo, sample media
├── config/
│   ├── api.json            # API endpoint URLs and settings
│   ├── backends.yaml       # Hardware backend dispatch configuration
│   └── logging.json        # Structured multi-process logging configuration
├── docs/                   # Architectural specs, guides, and documentation
├── fast_float/             # Main Python application package
│   ├── cli.py              # CLI entry point (`fast-float` supervisor)
│   ├── core/               # Video generation orchestrator and pipeline logic
│   ├── ipc/                # ZeroMQ client/server and shared memory ring buffers
│   ├── livekit/            # LiveKit integration and Gemini Live agent worker
│   ├── models/             # PyTorch neural network architectures (FLOAT, FMT, Wav2Vec2)
│   ├── onnx/               # ONNX export, TensorRT runtimes & backend dispatch
│   ├── routers/            # FastAPI endpoint routers (generation, livekit, webrtc, health)
│   ├── streaming/          # Frame dispatcher, tap queues & live transports
│   ├── utils/              # GPU tracking, video optimizers & logger
│   ├── workers/            # Dedicated GPU inference worker
│   └── main.py             # FastAPI entrypoint with lifespan management
├── frontend/               # React 19 + Vite studio application
│   ├── src/
│   │   ├── components/     # Studio & Live mode UI components
│   │   ├── services/       # API clients, audio streaming & telemetry
│   │   ├── App.jsx         # App root & mode switcher
│   │   └── App.css         # Glassmorphism design system & layouts
│   └── package.json        # Frontend dependencies
├── scripts/                # Automation, setup, and runner scripts
│   ├── setup.sh            # Automated system setup (uv, dependencies, weights)
│   ├── run_dual_process.sh # Production dual-process runner (Worker + API + UI)
│   ├── run_all.sh          # All-in-one dev launcher (Backend + Frontend)
│   ├── run_api.sh          # FastAPI backend launcher
│   ├── run_frontend.sh     # React Vite frontend launcher
│   ├── run_gemini_agent.sh # Gemini Live worker launcher
│   └── download_checkpoints.sh # Weight downloader
├── tests/                  # Pytest unit and integration test suites
├── Dockerfile              # Multi-stage CUDA 12.6 container
├── LICENSE                 # Project MIT license
├── pyproject.toml          # Python project metadata and uv configuration
├── README.md               # Project documentation
└── uv.lock                 # Pinned dependencies lockfile
```

---

## Getting Started

### Prerequisites

- **OS**: Linux or Windows (Linux recommended for CUDA NVENC, TensorRT, and POSIX SHM)
- **GPU**: NVIDIA GPU (CUDA 12.6 compatible driver)
- **Python**: 3.10 managed by [`uv`](https://docs.astral.sh/uv/)
- **Node.js**: Node 20.19+ or Node 22.12+ with [`pnpm`](https://pnpm.io/)
- **System Tools**: `ffmpeg`, `git`, `curl`

---

### One-Command Automated Setup (`scripts/setup.sh`)

The easiest way to bootstrap the full system (system dependencies, `uv`, Python 3.10 virtual environment, PyTorch CUDA 12.6 wheels, frontend dependencies with `pnpm`, ONNX/TensorRT tools, and checkpoints) is:

```bash
chmod +x scripts/setup.sh
./scripts/setup.sh
```

What `scripts/setup.sh` handles automatically:

1. Installs system packages (`ffmpeg`, `git`, `wget`, `curl`, `build-essential`).
2. Installs `uv` and provisions Python 3.10 matching `.python-version`.
3. Verifies and installs Node 22 (via NodeSource) and activates `pnpm`.
4. Installs frontend dependencies in `frontend/`.
5. Syncs pinned PyTorch CUDA 12.6 dependencies into `.venv` via `uv sync --inexact`.
6. Sets up the ONNX Runtime GPU and TensorRT execution providers.
7. Starts the local LiveKit SFU container via Docker Compose (if Docker is present).
8. Downloads model weights (`float.pth`, Wav2Vec2, speech emotion model) via `scripts/download_checkpoints.sh`.

---

### Manual Installation

If you prefer step-by-step setup:

```bash
# 1. Clone repository
git clone https://github.com/mohamed-em2m/float-fast.git
cd float-fast

# 2. Sync Python virtual environment with pinned CUDA 12.6 wheels
uv sync --frozen

# 3. Download model checkpoints
chmod +x scripts/download_checkpoints.sh
./scripts/download_checkpoints.sh

# 4. Install frontend dependencies
cd frontend
pnpm install
cd ..
```

---

## Running the System

### 1. High-Performance Dual-Process Architecture (`scripts/run_dual_process.sh`)

For optimal performance and maximum responsiveness, run the dual-process architecture. This launches the dedicated GPU worker in an isolated CUDA process and connects it to the FastAPI backend and React frontend:

```bash
chmod +x scripts/run_dual_process.sh
./scripts/run_dual_process.sh
```

**Environment overrides:**

```bash
# Custom ports and GPU configuration
PORT=8000 FRONTEND_PORT=3000 NGPUS=1 ./scripts/run_dual_process.sh

# Custom ZeroMQ IPC ports
FLOAT_ZMQ_CONTROL_PORT=5555 FLOAT_ZMQ_FRAMES_PORT=5556 ./scripts/run_dual_process.sh
```

- **Backend API**: `http://localhost:8000` (Swagger UI: `http://localhost:8000/docs`)
- **React Studio**: `http://localhost:3000`
- **Logs**: `logs/worker.log`, `logs/api.log`, `logs/combined.log`

---

### 2. All-in-One Dev Runner (`scripts/run_all.sh`)

To launch the backend API and frontend dev server together in a single process setup:

```bash
chmod +x scripts/run_all.sh
./scripts/run_all.sh
```

---

### 3. CLI Supervisor (`fast-float`)

The repository installs the `fast-float` CLI script defined in `pyproject.toml` (`[project.scripts]`):

```bash
# Launch the full supervised stack (Worker + Backend + Frontend)
uv run --no-sync fast-float serve --frontend

# Run only the GPU inference worker
uv run --no-sync fast-float worker --control-port 5555 --frames-port 5556

# Run only the Studio frontend UI
uv run --no-sync fast-float ui --port 3000

# Run only the Gemini Live agent
uv run --no-sync fast-float agent
```

---

### 4. Running Individual Components

**FastAPI Backend Server:**

```bash
uv run --frozen --no-sync hypercorn fast_float.main:app --bind 0.0.0.0:8000 --log-level info
# or
./scripts/run_api.sh
```

**React Studio Frontend:**

```bash
cd frontend && pnpm dev
# or
./scripts/run_frontend.sh
```

**Gemini Live Voice Worker:**

```bash
./scripts/run_gemini_agent.sh
```

**Gradio Batch Web Interface:**

```bash
./scripts/run_gradio.sh
```

**Gradio Live Audio Interface:**

```bash
./scripts/run_live.sh
```

---

## API Usage

### REST Endpoint Example

Submit audio and reference image to `/generate`:

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "audio_base64": "<BASE64_PCM_OR_WAV>",
    "image_base64": "<BASE64_IMAGE_DATA>",
    "emotion": "S2E",
    "nfe": 10,
    "quality": "balanced",
    "live": true
  }'
```

### Python Client Example

```python
import base64
import requests

# Load reference portrait and speech audio
with open("portrait.jpg", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode("utf-8")

with open("voice.wav", "rb") as f:
    aud_b64 = base64.b64encode(f.read()).decode("utf-8")

# Request video synthesis
response = requests.post(
    "http://localhost:8000/generate",
    json={
        "image_base64": img_b64,
        "audio_base64": aud_b64,
        "emotion": "S2E",
        "nfe": 10,
        "quality": "balanced",
    },
)

result = response.json()
print("Generation Time:", result["performance"]["total_time"], "s")
print("Stream URL:     ", result["stream_url"])
print("Download URL:   ", result["download_url"])
```

---

## Configuration

Core runtime settings are managed via environment variables or `fast_float/config/settings.toml`:

| Variable | Description | Default |
|---|---|---|
| `PORT` | FastAPI server listening port | `8000` |
| `FRONTEND_PORT` | React Studio Vite development port | `3000` |
| `FLOAT_IPC_ENABLED` | Enable dual-process ZeroMQ/SHM worker mode | `true` |
| `FLOAT_ZMQ_CONTROL_PORT` | ZeroMQ control port between API and Worker | `5555` |
| `FLOAT_ZMQ_FRAMES_PORT` | ZeroMQ frame publishing port | `5556` |
| `FLOAT_SHM_NAME` | POSIX shared memory ring buffer name | `float_frame_ring` |
| `FLOAT_SHM_SLOTS` | Ring buffer frame capacity slots | `64` |
| `WARMUP_WAIT_TIMEOUT_S` | Maximum seconds `/generate` waits for boot warmup | `600` |
| `LIVEKIT_URL` | LiveKit SFU WebSocket endpoint | `ws://localhost:7880` |
| `LIVEKIT_API_KEY` | LiveKit API Key | `devkey` |
| `LIVEKIT_API_SECRET` | LiveKit API Secret | `secret` |
| `GEMINI_API_KEY` | Google Gemini API key for real-time speech agent | — |

---

## Hardware Acceleration & ONNX/TensorRT

Convert individual units (FMT, Wav2Vec2, Style Decoder) to ONNX or TensorRT:

```bash
# Automatically convert all supported units and run the backend selector
./scripts/convert_to_onnx.sh

# Convert individual modules
./scripts/convert_fmt_to_onnx.sh
./scripts/convert_wav2vec_to_onnx.sh
```

Backends can be configured in `config/backends.yaml` to specify `eager`, `compile`, or `onnx` (TensorRT → CUDA → CPU) per component.

---

## Testing & Quality Checks

Run the verification test suites and code linters:

```bash
# Python lint and style format checks
uv run ruff check .
uv run ruff format --check .

# Python test suite
uv run pytest tests/

# Frontend test suite
cd frontend && pnpm test

# Pre-commit hook checks across all files
uv run pre-commit run --all-files
```

---

## Roadmap

- [x] High-performance FastAPI server with NVENC hardware acceleration
- [x] Dual-process ZeroMQ + Shared Memory architecture for GPU isolation
- [x] Real-time live frame taps via WebSocket, WebRTC, and LiveKit
- [x] LiveKit conversational avatar integration with Gemini Live voice agent
- [x] Per-component ONNX & TensorRT automated backend selector
- [x] Synchronized React 19 Studio with dual Studio and Live Audio modes
- [x] One-command automated environment setup script (`scripts/setup.sh`)
- [ ] Multi-speaker audio separation and diarization support
- [ ] Multi-face portrait animation and scene background stabilization

---

## Contributing

Contributions are welcome!

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m "feat(scope): add amazing feature"`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

Please ensure code complies with the project style guidelines (`uv run ruff check .`) and pre-commit checks pass (`uv run pre-commit run --all-files`).

---

## License

This project is licensed under the [MIT License](LICENSE).

> [!NOTE]
> Please refer to the upstream [FLOAT repository](https://github.com/deepbrainai-research/float) to review its license terms regarding original checkpoints and research materials.

---

## Contact

Mohamed Emam · [GitHub Profile](https://github.com/mohamed-em2m)

Project Link: [https://github.com/mohamed-em2m/float-fast](https://github.com/mohamed-em2m/float-fast)
