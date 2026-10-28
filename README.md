<div align="center">

# FLOAT Fast

High-performance, hardware-accelerated serving and inference stack for audio-driven talking portrait generation (FLOAT, ICCV 2025).



[![Build Status](https://img.shields.io/github/actions/workflow/status/mohamed-em2m/float-fast/ci.yml?branch=main)](https://github.com/mohamed-em2m/float-fast/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-v5.2.0-blue.svg)](https://github.com/mohamed-em2m/float-fast/releases)

[Demo](http://localhost:3000) · [Documentation](docs/USAGE.md) · [Report Bug](https://github.com/mohamed-em2m/float-fast/issues) · [Request Feature](https://github.com/mohamed-em2m/float-fast/issues)
<img src="assets/logo.png" alt="FLOAT Fast Logo" width="120" style="border-radius: 20px; margin-bottom: 12px;" />

</div>

---

## Table of Contents

- [About](#about)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Configuration](#configuration)
- [Testing](#testing)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## About

**FLOAT Fast** is a solo project providing a production-grade inference engine and real-time interactive avatar platform built around **FLOAT (Flow Matching for Audio-Driven Talking Portrait Video Generation, ICCV 2025)**.

Standard talking-head generation models are computationally intensive, slow to render, and traditionally restricted to offline batch video synthesis. **FLOAT Fast** solves these challenges by combining:

1. **Low-Latency Streaming Pipeline**: Stream video frames in real time over WebSockets, WebRTC, and LiveKit while inference is still in-flight.
2. **Hardware Acceleration & Batched Decoding**: Micro-batched latent decoding, NVENC hardware encoding, and per-component ONNX/TensorRT execution providers.
3. **Conversational Live Voice Agent**: Real-time bidirectional voice conversations driven by Gemini Live and LiveKit Agents with instant lip-sync.
4. **Modern Studio Interface**: A unified React 19 + Vite web studio supporting both batch generation and low-latency interactive live audio streaming.

> [!IMPORTANT]
> **Upstream Research & License Notice**:
> Users are advised to review the upstream [FLOAT repository](https://github.com/deepbrainai-research/float) and read the original [FLOAT license](https://github.com/deepbrainai-research/float/blob/main/LICENSE) regarding the underlying model weights and architecture.

## Features

- ✅ **Ultra-Fast Video Generation**: Hardware-accelerated inference with batched latent decoding and NVENC/QSV/AMF encoding.
- ✅ **Real-Time Live Transports**: Push frames straight off the decoder over WebSocket, WebRTC, or LiveKit SFU before the MP4 file is finished.
- ✅ **Conversational Voice Avatar**: Live speech-to-speech interaction powered by Gemini Live and LiveKit Agents (`fast_float/livekit/gemini/`).
- ✅ **Per-Component Backend Optimization**: Run each model unit in PyTorch Eager, `torch.compile`, or ONNX/TensorRT with automated parity verification.
- ✅ **Dual-Mode React Studio**: High-fidelity dark mode interface with dedicated Studio and Live Audio modes sharing unified dimensions and card structures.
- ✅ **HTTP Byte-Range Video Streaming**: Efficient video playback and instant seeking for rendered MP4 portraits.
- ✅ **Telemetry & GPU Resource Management**: Real-time VRAM monitoring, CUDA cache purging, and stage-by-stage inference progress tracking.

## Tech Stack

| Layer     | Technology            |
|-----------|-----------------------|
| Language  | Python 3.10, JavaScript (ESNext / JSX) |
| Framework | FastAPI, Hypercorn, React 19, Vite 8 |
| Deep Learning | PyTorch 2.5.1+cu126, ONNX Runtime GPU 1.23, TensorRT 10.9 |
| Real-time Audio/Video | LiveKit, WebRTC (aiortc), WebSockets |
| Tooling   | `uv`, `pnpm`, Docker, FFmpeg, Ruff |

## Project Structure

```text
float-fast/
├── .github/
│   └── workflows/          # CI pipelines and automated checks
├── assets/                 # Brand assets and graphics (logo.png)
├── config/
│   ├── api.json            # API endpoint URLs and settings
│   └── backends.yaml       # Hardware backend dispatch configuration
├── docs/                   # Architectural specs, guides, and documentation
├── fast_float/             # Main Python backend application
│   ├── core/               # Video orchestrator, pipeline & state management
│   ├── models/             # PyTorch neural network architectures (FLOAT, FMT, Wav2Vec2)
│   ├── onnx/               # ONNX export, TensorRT runtimes & backend dispatch
│   ├── routers/            # FastAPI endpoint routers (generation, livekit, webrtc, health)
│   ├── streaming/          # Frame dispatcher, tap queues & live transports
│   ├── livekit/            # LiveKit integration & Gemini Live agent worker
│   ├── utils/              # GPU tracking, video optimizers & logger
│   └── main.py             # FastAPI entrypoint
├── frontend/               # React 19 + Vite studio application
│   ├── src/
│   │   ├── components/     # Studio & Live mode UI components
│   │   ├── services/       # API clients, audio streaming & telemetry
│   │   ├── App.jsx         # App root & mode switcher
│   │   └── App.css         # Glassmorphism design system & layouts
│   └── package.json        # Frontend dependencies
├── scripts/                # Environment setup, model conversion, and runners
├── tests/                  # Unit and integration test suites
├── Dockerfile              # Multi-stage CUDA 12.6 container
├── LICENSE                 # Project MIT license
├── pyproject.toml          # Python project metadata and uv configuration
├── README.md               # Project documentation
└── uv.lock                 # Pinned dependencies lockfile
```

## Getting Started

### Prerequisites

- **OS**: Linux or Windows (Linux recommended for CUDA NVENC/TensorRT)
- **GPU**: NVIDIA GPU (CUDA 12.6 compatible driver)
- **Python**: 3.10 with [`uv`](https://docs.astral.sh/uv/) installed
- **Node.js**: Node 20.19+ or Node 22.12+ with [`pnpm`](https://pnpm.io/)
- **System Tools**: `ffmpeg`, `git`, `curl`

### Installation

```bash
# Clone the repository
git clone https://github.com/mohamed-em2m/float-fast.git
cd float-fast

# Sync Python dependencies using uv (CUDA 12.6 wheels pinned)
uv sync --frozen

# Download required checkpoints (FLOAT, Wav2Vec2)
./scripts/download_checkpoints.sh

# Install frontend dependencies
cd frontend
pnpm install
cd ..
```

## Usage

### Run Backend + Frontend Together

Launch both the FastAPI backend and Vite frontend studio in one command:

```bash
./scripts/run_all.sh
```

- **Backend API**: `http://localhost:8000` (Docs: `http://localhost:8000/docs`)
- **Studio Frontend**: `http://localhost:3000`

### Run Services Separately

**FastAPI Backend Server:**

```bash
uv run --frozen --no-sync hypercorn fast_float.main:app --bind 0.0.0.0:8000 --log-level info
# or
./scripts/run_api.sh
```

**React Studio Frontend:**

```bash
cd frontend
pnpm dev
# or
./scripts/run_frontend.sh
```

**Gemini Live Voice Agent (Optional):**

```bash
./scripts/run_gemini_agent.sh
```

### Python API Example

```python
import base64
import requests

with open("portrait.jpg", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode("utf-8")

with open("voice.wav", "rb") as f:
    aud_b64 = base64.b64encode(f.read()).decode("utf-8")

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

data = response.json()
print("Download URL:", data["download_url"])
```

## Configuration

Environment variables and configuration files:

| Variable | Description | Default |
|---|---|---|
| `PORT` | FastAPI server port | `8000` |
| `FRONTEND_PORT` | Vite development server port | `3000` |
| `WARMUP_WAIT_TIMEOUT_S` | Maximum wait time for boot warmup on `/generate` | `600` |
| `LIVEKIT_URL` | LiveKit SFU server WebSocket URL | `ws://localhost:7880` |
| `LIVEKIT_API_KEY` | LiveKit API Key | `devkey` |
| `LIVEKIT_API_SECRET` | LiveKit API Secret | `secret` |
| `GEMINI_API_KEY` | Google Gemini API key for real-time speech agent | — |

Hardware backends can be configured in `config/backends.yaml` or generated via `./scripts/convert_to_onnx.sh`.

## Testing

Run unit tests and lint checks:

```bash
# Frontend test suite
cd frontend && pnpm test

# Python tests & style checks
uv run ruff check .
uv run pytest tests/
```

## Roadmap

- [x] High-performance FastAPI server with NVENC hardware acceleration
- [x] Real-time live frame taps via WebSocket & WebRTC
- [x] LiveKit conversational avatar integration with Gemini Live
- [x] Per-component ONNX & TensorRT automated backend selector
- [x] Unified React 19 Studio with synchronized Studio & Live mode layouts
- [ ] Multi-speaker audio separation and diarization support
- [ ] Multi-face portrait animation and scene background stabilization

See the [open issues](https://github.com/mohamed-em2m/float-fast/issues) for the full list of proposed features.

## Contributing

Contributions are welcome!

1. Fork the repository
2. Create a branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m "feat: add amazing feature"`
4. Push the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

Please ensure code conforms to project standards: `uv run ruff check .` and `uv run pre-commit run --all-files`.

## License

This project is licensed under the [MIT License](LICENSE).

> [!NOTE]
> Please refer to the upstream [FLOAT repository](https://github.com/deepbrainai-research/float) to review its license terms regarding original checkpoints and research materials.

## Contact

Mohamed Emam · [GitHub Profile](https://github.com/mohamed-em2m) · [LinkedIn](https://linkedin.com)

Project link: [https://github.com/mohamed-em2m/float-fast](https://github.com/mohamed-em2m/float-fast)
