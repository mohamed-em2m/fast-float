<div align="center">

<img src="assets/logo.png" alt="FLOAT Fast logo" width="55%" />

# FLOAT Fast

**Real-time, GPU-accelerated serving stack for audio-driven talking portraits.**

Built on [FLOAT](https://github.com/deepbrainai-research/float) (ICCV 2025) · Stream video while it is still being generated.

[![CI](https://img.shields.io/github/actions/workflow/status/mohamed-em2m/float-fast/ci.yml?branch=main&label=CI)](https://github.com/mohamed-em2m/float-fast/actions)
[![Release](https://img.shields.io/github/v/release/mohamed-em2m/float-fast?label=release)](https://github.com/mohamed-em2m/float-fast/releases)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)
[![Python 3.10](https://img.shields.io/badge/python-3.10-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![PyTorch 2.7.1](https://img.shields.io/badge/PyTorch-2.7.1%20%2B%20cu126-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.122-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React 19](https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=black)](frontend/)

[Quick Start](#quick-start) · [Documentation](docs/USAGE.md) · [API](#api) · [Report a Bug](https://github.com/mohamed-em2m/float-fast/issues) · [Request a Feature](https://github.com/mohamed-em2m/float-fast/issues)

</div>

<!-- TODO: add a demo GIF or screenshot of the Studio here -->
<!-- <p align="center"><img src="assets/demo.gif" alt="FLOAT Fast Studio demo" width="800" /></p> -->

---

## Overview

Talking-head models such as FLOAT produce impressive results, but they are heavy and are usually run as offline batch jobs. **FLOAT Fast** turns FLOAT into an interactive service:

- **Dual-process design** – a dedicated GPU worker is isolated from the async API server, linked by ZeroMQ and POSIX shared memory, so inference never blocks the event loop.
- **Streaming first** – frames are pushed over WebSocket, WebRTC, or LiveKit while inference is still running, before the MP4 is finished.
- **Hardware acceleration** – micro-batched latent decoding, NVENC/QSV/AMF encoding, and per-component PyTorch / `torch.compile` / ONNX / TensorRT backends.
- **Conversational avatar** – speech-to-speech interaction through Gemini Live and LiveKit Agents, with live lip-sync.
- **Studio UI** – a React 19 + Vite interface for both batch generation and live audio.

> [!IMPORTANT]
> This project builds on the upstream [FLOAT repository](https://github.com/deepbrainai-research/float). Review its [license](https://github.com/deepbrainai-research/float/blob/main/LICENSE) before using the model weights or architecture, especially for commercial use.

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Secrets & Configuration](#secrets--configuration)
- [Running the System](#running-the-system)
- [API](#api)
- [Hardware Acceleration (ONNX / TensorRT)](#hardware-acceleration-onnx--tensorrt)
- [Remote & Colab Usage](#remote--colab-usage)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Citation & Acknowledgments](#citation--acknowledgments)
- [Contact](#contact)

---

## Features

| | Feature | Description |
|---|---|---|
| ⚡ | **Dual-process IPC** | GPU worker and API server communicate over ZeroMQ control sockets and a zero-copy shared-memory frame ring. |
| 🎞️ | **Fast generation** | Auto-tuned batched latent decoding with NVENC / QSV / AMF encoding. |
| 📡 | **Live transports** | WebSocket, WebRTC, and LiveKit SFU streaming straight off the decoder. |
| 🗣️ | **Voice avatar** | Real-time speech-to-speech with Gemini Live and LiveKit Agents (`fast_float/livekit/gemini/`). |
| 🧩 | **Per-component backends** | Run each model unit on Eager, `torch.compile`, or ONNX/TensorRT, with automated parity checks. |
| 🖥️ | **Unified Studio** | One React 19 interface for batch generation and interactive live audio. |
| 🛠️ | **One-command setup** | `scripts/setup.sh` bootstraps everything; the `fast-float` CLI supervises all processes. |

## Architecture

```mermaid
flowchart TB
    Browser["Browser · React 19 Studio"] <-->|"WebSocket / WebRTC / HTTP Range"| API

    subgraph P2["Process 2 · Streaming backend"]
        API["FastAPI + Hypercorn<br/>/generate · /stream · /ws/stream · WebRTC signaling"]
        LK["LiveKit session manager<br/>+ Gemini Live agent"]
    end

    subgraph P1["Process 1 · GPU inference worker"]
        W["FLOAT flow-matching ODE solver<br/>+ style latent generator"]
        D["Batched latent decoding<br/>VRAM auto-tuned · TensorRT / CUDA"]
        W --> D
    end

    API <-->|"ZeroMQ control :5555 / frames :5556<br/>+ POSIX shared memory (float_frame_ring)"| W
    LK <--> API
```

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10, JavaScript (ESNext / JSX) |
| Serving | FastAPI, Hypercorn, Uvicorn, ZeroMQ |
| Deep learning | PyTorch 2.7.1 (CUDA 12.6), ONNX Runtime GPU 1.23, TensorRT 10.9 |
| Real-time media | LiveKit SFU & Agents, aiortc (WebRTC), PyAV |
| Frontend | React 19, Vite 8, Lucide Icons |
| Tooling | `uv`, `pnpm`, Ruff, pytest, pre-commit |

---

## Quick Start

```bash
git clone https://github.com/mohamed-em2m/float-fast.git
cd float-fast

./scripts/setup.sh        # one-time: dependencies, ONNX/TensorRT, checkpoints
./scripts/run_dual_process.sh
```

Then open:

| Service | URL |
|---|---|
| Studio | <http://localhost:3000> |
| API + Swagger UI | <http://localhost:8000/docs> |

The first boot warms up for roughly 1–5 minutes (weights, ONNX/TensorRT sessions, decode auto-tuning). Results are cached in `~/.cache/fast_float/`.

## Installation

### Prerequisites

| Requirement | Details |
|---|---|
| OS | Linux recommended (NVENC, TensorRT, POSIX shared memory). Windows is supported with limitations. |
| GPU | NVIDIA GPU with a CUDA 12.6-compatible driver |
| Python | 3.10, managed by [`uv`](https://docs.astral.sh/uv/) |
| Node.js | 20.19+ or 22.12+ with [`pnpm`](https://pnpm.io/) |
| System tools | `ffmpeg`, `git`, `curl` |

### Option A — Automated (recommended)

```bash
chmod +x scripts/setup.sh
./scripts/setup.sh
```

The script installs system packages, `uv` and Python 3.10, Node 22 and `pnpm`, frontend dependencies, the pinned CUDA 12.6 PyTorch stack (`uv sync --inexact`), ONNX Runtime GPU and TensorRT, optionally starts a local LiveKit SFU via Docker Compose, and downloads the checkpoints (`float.pth`, Wav2Vec2, speech-emotion model).

### Option B — Manual

```bash
git clone https://github.com/mohamed-em2m/float-fast.git
cd float-fast

uv sync --frozen                                   # Python environment
chmod +x scripts/download_checkpoints.sh
./scripts/download_checkpoints.sh                  # model weights

cd frontend && pnpm install && cd ..              # Studio dependencies
```

> [!NOTE]
> On a CUDA machine, prefer `./scripts/setup.sh` over a plain `uv sync`. Setup layers `onnxruntime-gpu` and TensorRT 10.9 on top of the lockfile; a plain `uv sync` / `uv run` reverts to the CPU `onnxruntime`, and the server boots with only `AzureExecutionProvider` and `CPUExecutionProvider`. All launchers therefore use `uv run --frozen --no-sync`.

### Docker

A multi-stage CUDA 12.6 `Dockerfile` is included at the repository root.

---

## Secrets & Configuration

Settings are resolved in this order (highest priority first):

1. Environment variables
2. `fast_float/config/secerts.toml` — git-ignored, machine-local secrets *(the filename's spelling is intentional)*
3. `fast_float/config/settings.toml` — committed, non-secret defaults
4. `.env` at the repo root — optional fallback (see `.env.example`)

```toml
# fast_float/config/secerts.toml  (never commit this file)
[livekit]
url        = "wss://<your-project>.livekit.cloud"  # or "ws://127.0.0.1:7880" for local Docker
api_key    = "<LIVEKIT_API_KEY>"
api_secret = "<LIVEKIT_API_SECRET>"

[gemini]
api_key = "<GOOGLE_API_KEY>"   # GOOGLE_API_KEY / GEMINI_API_KEY env vars also work
```

The same file is read by the API and the Gemini worker, so a single key serves both. If you have no LiveKit project and no local Docker, leave `[livekit].url` empty: live transports fall back to WebRTC, then WebSocket/MP4.

### Environment variables

| Variable | Description | Default |
|---|---|---|
| `PORT` | API server port | `8000` |
| `FRONTEND_PORT` | Studio dev-server port | `3000` |
| `FLOAT_IPC_ENABLED` | Enable dual-process ZeroMQ/SHM mode | `true` |
| `FLOAT_ZMQ_CONTROL_PORT` | ZeroMQ control port (API ↔ worker) | `5555` |
| `FLOAT_ZMQ_FRAMES_PORT` | ZeroMQ frame-publishing port | `5556` |
| `FLOAT_SHM_NAME` | Shared-memory ring buffer name | `float_frame_ring` |
| `FLOAT_SHM_SLOTS` | Ring buffer capacity (frame slots) | `64` |
| `WARMUP_WAIT_TIMEOUT_S` | Max seconds `/generate` waits for warmup | `600` |
| `LIVEKIT_URL` | LiveKit SFU WebSocket endpoint | set in `settings.toml` |
| `LIVEKIT_API_KEY` | LiveKit API key | `devkey` (local dev only) |
| `LIVEKIT_API_SECRET` | LiveKit API secret | `secret` (local dev only) |
| `GEMINI_API_KEY` | Gemini key for the voice agent | — |

> [!WARNING]
> `devkey` / `secret` are development defaults for a local LiveKit server. Always set your own credentials before exposing the service publicly.

---

## Running the System

### Dual-process mode (production)

```bash
./scripts/run_dual_process.sh

# Overrides
PORT=8000 FRONTEND_PORT=3000 NGPUS=1 ./scripts/run_dual_process.sh
FLOAT_ZMQ_CONTROL_PORT=5555 FLOAT_ZMQ_FRAMES_PORT=5556 ./scripts/run_dual_process.sh
```

Logs are written to `logs/worker.log`, `logs/api.log`, and `logs/combined.log`.

### All-in-one dev runner

```bash
./scripts/run_all.sh      # single-process API + Studio (no ZeroMQ worker)
```

### `fast-float` CLI

```bash
# Full supervised stack: worker + backend + frontend
uv run --frozen --no-sync fast-float serve --port 8000 --dual-process --frontend

# Single-process API + Studio
uv run --frozen --no-sync fast-float serve --port 8000 --no-dual-process --frontend

# Individual components
uv run --frozen --no-sync fast-float worker --control-port 5555 --frames-port 5556
uv run --frozen --no-sync fast-float ui --port 3000
uv run --frozen --no-sync fast-float agent
```

### Individual scripts

| Component | Command |
|---|---|
| FastAPI backend | `./scripts/run_api.sh` or `uv run --frozen --no-sync hypercorn fast_float.main:app --bind 0.0.0.0:8000` |
| React Studio | `./scripts/run_frontend.sh` or `cd frontend && pnpm dev` |
| Gemini Live worker | `./scripts/run_gemini_agent.sh` |
| Gradio batch UI | `./scripts/run_gradio.sh` |
| Gradio live audio UI | `./scripts/run_live.sh` |

### Verify the boot

```bash
curl http://localhost:8000/health
# {"model_loaded": true, ..., "warming_up": false}  → ready
```

---

## API

Interactive docs are served at `http://localhost:8000/docs`.

### `POST /generate`

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "audio_base64": "<BASE64_PCM_OR_WAV>",
    "image_base64": "<BASE64_IMAGE>",
    "emotion": "S2E",
    "nfe": 10,
    "quality": "balanced",
    "live": true
  }'
```

| Field | Description |
|---|---|
| `audio_base64` | Base64-encoded PCM or WAV speech |
| `image_base64` | Base64-encoded reference portrait |
| `emotion` | Emotion mode (e.g. `S2E`, speech-to-emotion) |
| `nfe` | Number of function evaluations for the ODE solver |
| `quality` | Quality preset (e.g. `balanced`) |
| `live` | Enable live frame streaming while generating |

### Python client

```python
import base64
import requests


def b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


response = requests.post(
    "http://localhost:8000/generate",
    json={
        "image_base64": b64("portrait.jpg"),
        "audio_base64": b64("voice.wav"),
        "emotion": "S2E",
        "nfe": 10,
        "quality": "balanced",
    },
    timeout=600,
)
response.raise_for_status()
result = response.json()

print("Generation time:", result["performance"]["total_time"], "s")
print("Stream URL:     ", result["stream_url"])
print("Download URL:   ", result["download_url"])
```

---

## Hardware Acceleration (ONNX / TensorRT)

Convert individual units (FMT, Wav2Vec2, Style Decoder) to ONNX/TensorRT:

```bash
./scripts/convert_to_onnx.sh             # convert all units and run the backend selector
./scripts/convert_fmt_to_onnx.sh         # FMT only
./scripts/convert_wav2vec_to_onnx.sh     # Wav2Vec2 only
```

Choose a backend per component in `config/backends.yaml`:

| Backend | Behavior |
|---|---|
| `eager` | Standard PyTorch |
| `compile` | `torch.compile` |
| `onnx` | ONNX Runtime with TensorRT → CUDA → CPU fallback |

## Remote & Colab Usage

`examples/fast_float.ipynb` is the tested end-to-end path for a fresh GPU machine (verified on Colab T4). On a remote box with no public URL, expose the **Studio** (port 3000, not the API) over HTTPS; browsers require a secure context for microphone access.

```bash
uv run --frozen --no-sync fast-float serve --port 8000 --dual-process --frontend

# second shell
./scripts/run_tunnel.sh          # prints https://<id>.trycloudflare.com
./scripts/run_tunnel.sh url      # re-print the URL
./scripts/run_tunnel.sh stop     # stop the tunnel
```

If `cloudflared` is missing:

```bash
curl -fsSL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb -o /tmp/cloudflared.deb
sudo dpkg -i /tmp/cloudflared.deb
```

---

## Project Structure

```text
float-fast/
├── assets/                  # Logo and sample media
├── config/                  # api.json, backends.yaml, logging.json
├── docs/                    # Guides and specifications
├── fast_float/              # Python package
│   ├── cli.py               # `fast-float` supervisor CLI
│   ├── core/                # Generation orchestrator and pipeline
│   ├── ipc/                 # ZeroMQ client/server, shared-memory ring buffers
│   ├── livekit/             # LiveKit integration, Gemini Live agent
│   ├── models/              # FLOAT, FMT, Wav2Vec2 architectures
│   ├── onnx/                # ONNX export, TensorRT runtimes, backend dispatch
│   ├── routers/             # generation, livekit, webrtc, health
│   ├── streaming/           # Frame dispatcher, tap queues, live transports
│   ├── utils/               # GPU tracking, video optimizers, logging
│   ├── workers/             # Dedicated GPU inference worker
│   └── main.py              # FastAPI entrypoint
├── frontend/                # React 19 + Vite Studio
├── scripts/                 # setup, run_*, convert_*, download_checkpoints
├── tests/                   # pytest suites
├── Dockerfile               # Multi-stage CUDA 12.6 image
├── pyproject.toml
└── uv.lock
```

## Testing

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest tests/
(cd frontend && pnpm test)
uv run pre-commit run --all-files
```

## Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| Server lists only `AzureExecutionProvider` / `CPUExecutionProvider` | A plain `uv sync` replaced `onnxruntime-gpu`. Re-run `./scripts/setup.sh` and launch with `uv run --frozen --no-sync`. |
| `/generate` hangs right after startup | The model is still warming up. Check `/health` for `"warming_up": false` (timeout: `WARMUP_WAIT_TIMEOUT_S`). |
| Microphone does not work remotely | `getUserMedia` needs HTTPS. Tunnel the Studio on port 3000, not the API. |
| No live LiveKit stream | Leave `[livekit].url` empty to fall back to WebRTC/WebSocket, or verify your keys in `secerts.toml`. |
| Shared-memory errors | Another instance may hold `float_frame_ring`. Stop it or change `FLOAT_SHM_NAME`. |

## Roadmap

- [x] FastAPI server with NVENC hardware acceleration
- [x] Dual-process ZeroMQ + shared-memory GPU isolation
- [x] Real-time frame taps over WebSocket, WebRTC, and LiveKit
- [x] LiveKit conversational avatar with Gemini Live
- [x] Per-component ONNX / TensorRT backend selector
- [x] React 19 Studio with Studio and Live Audio modes
- [x] One-command setup (`scripts/setup.sh`)
- [ ] Multi-speaker audio separation and diarization
- [ ] Multi-face portrait animation and background stabilization

## Contributing

Contributions are welcome.

1. Fork the repository
2. Create a branch: `git checkout -b feature/amazing-feature`
3. Commit using [Conventional Commits](https://www.conventionalcommits.org/): `git commit -m "feat(scope): add amazing feature"`
4. Push: `git push origin feature/amazing-feature`
5. Open a Pull Request

Before submitting, make sure `uv run ruff check .` and `uv run pre-commit run --all-files` pass.

## License

Released under the [MIT License](LICENSE). The upstream FLOAT checkpoints and research materials are governed by their own [license](https://github.com/deepbrainai-research/float/blob/main/LICENSE).

## Citation & Acknowledgments

This project builds on the work of the FLOAT authors. If you use it in research, please cite the original paper:

```bibtex
@inproceedings{ki2025float,
  title     = {FLOAT: Generative Motion Latent Flow Matching for Audio-driven Talking Portrait},
  author    = {Ki, Taekyung and Min, Dongchan and Chae, Gyeongsu},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2025}
}
```

Thanks to the teams behind [FLOAT](https://github.com/deepbrainai-research/float), [LiveKit](https://livekit.io/), [FastAPI](https://fastapi.tiangolo.com/), and [PyTorch](https://pytorch.org/).

## Contact

**Mohamed Emam** · [@mohamed-em2m](https://github.com/mohamed-em2m)
Project: <https://github.com/mohamed-em2m/float-fast>
