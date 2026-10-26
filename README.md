# FLOAT Fast

**A serving and inference stack for FLOAT — audio-driven talking-portrait generation (ICCV 2025).**

[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/) #
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch 2.7.1](https://img.shields.io/badge/pytorch-2.7.1-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![ICCV 2025](https://img.shields.io/badge/ICCV-2025-4b44ce.svg)](https://iccv.thecvf.com)

> **Attribution — please read.** FLOAT Fast is a community engineering project built on top of the official **FLOAT** research implementation by [Taekyung Ki](https://taekyungki.github.io), [Dongchan Min](https://kevinmin95.github.io), and [Gyeongsu Chae](https://www.aistudios.com/ko) (DeepBrain AI), presented at **ICCV 2025**. The model architecture, trained checkpoints, method, and original inference code are entirely their work, released under a [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/) license. This repository adds a serving/inference layer around it and **inherits the same non-commercial license**. If you use it, please cite the original paper — see [Citation](#citation).
>
> Upstream project: [deepbrainai-research/float](https://github.com/deepbrainai-research/float) · Project page: <https://deepbrainai-research.github.io/float/>

---

## Contents

- [What this repository adds](#what-this-repository-adds)
- [Requirements](#requirements)
- [Quick start](#quick-start)
- [API](#api)
- [Live streaming](#live-streaming)
- [Per-component backends (ONNX / TensorRT)](#per-component-backends-onnx--tensorrt)
- [Configuration](#configuration)
- [Development](#development)
- [Repository layout](#repository-layout)
- [Known limitations](#known-limitations)
- [License](#license)
- [About FLOAT (upstream)](#about-float-upstream)

---

## What this repository adds

FLOAT Fast wraps the FLOAT generator in production-friendly interfaces and performance work:

- **FastAPI server** — hardware-accelerated inference with **NVENC** video encoding, HTTP **byte-range streaming**, and **WebRTC** signaling for real-time playback.
- **Live frame streaming** — frames are published to a **WebSocket** (`/ws/stream/{id}`) and to live WebRTC peers *while* the video is being generated, so the browser can start playing before the MP4 is finished.
- **React + Vite "Studio" frontend** — a browser UI for driving generation against the API, including a live-preview surface.
- **Gradio apps** — a batch UI (`fast_float/legacy/gradio_interface.py`) and a live microphone streaming UI (`fast_float/interfaces/float_live_app.py`).
- **Batched latent decoding + VRAM auto-sizing** — decodes frames in micro-batches sized to free GPU memory, with a tuner cache to skip re-probing on repeat runs. See [`docs/BATCH_DECODING.md`](docs/BATCH_DECODING.md).
- **Asynchronous streaming pipeline** — a multi-threaded producer/consumer over multiple CUDA streams and GPUs, overlapping encode, transfer, and disk I/O. See [`docs/STREAMING_PIPELINE.md`](docs/STREAMING_PIPELINE.md).
- **LiveKit avatar transport** — FLOAT publishes synchronized avatar video + audio into LiveKit rooms for conversational sessions, with raw WebRTC and WebSocket fallbacks. See [`docs/LIVEKIT_INTEGRATION.md`](docs/LIVEKIT_INTEGRATION.md) and [`docs/LIVEKIT_GENERATOR.md`](docs/LIVEKIT_GENERATOR.md).
- **Gemini Live voice agent** — a [LiveKit Agents](https://docs.livekit.io/agents/) worker (`fast_float/livekit/gemini/`) drives the live avatar conversation with Gemini speech-to-speech, exposes FLOAT voice tools, and streams its voice to the avatar for lip-sync. It runs as its own process but shares the project venv, lockfile and config. See [`docs/LIVEKIT_GEMINI_AGENT.md`](docs/LIVEKIT_GEMINI_AGENT.md).
- **Per-component ONNX / TensorRT acceleration** — each module can run `eager`, `compile`, or `onnx` (TensorRT → CUDA → CPU), selected from a config generated **on the target box** after parity + benchmark checks. See [Backends](#per-component-backends-onnx--tensorrt).

---

## Requirements

- Linux with an **NVIDIA GPU** (upstream tested on A100 and V100; the stack also runs on T4 / RTX-class cards).
- **Python 3.10** (from `.python-version`), managed by [uv](https://docs.astral.sh/uv/getting-started/installation/).
- `ffmpeg`, `git`, `wget`, `curl`.
- **Node 20.19+ / 22.12+** for the React frontend.
- Optional: **TensorRT 10.9 (CUDA 12)** for the TRT execution provider. ONNX Runtime 1.23 links against `libnvinfer.so.10`; a TensorRT 11 install cannot be loaded.

> The upstream `generate.py` CLI still works from [`fast_float/legacy/`](fast_float/legacy), but this fork drives inference through the FastAPI server and Gradio apps.

## Quick start

### Setup

```bash
git clone https://github.com/mohamed-em2m/float-fast.git
cd float-fast
./scripts/setup.sh          # apt deps + Node 22 + uv + uv sync + checkpoints + ONNX toolchain
```

Or manually:

```bash
uv sync --frozen            # Python from .python-version, CUDA torch pins from pyproject.toml
./scripts/download_checkpoints.sh
```

### Checkpoints

Download automatically:

```bash
./scripts/download_checkpoints.sh
```

…or manually from this [Google Drive](https://drive.google.com/file/d/1rvWuM12cyvNvBQNCLmG4Fr2L1rpjQBF0/view?usp=sharing). The checkpoints should be organized as:

```text
./checkpoints
|-- float.pth                                       # main model
|-- wav2vec2-base-960h/                             # audio encoder
|   |-- .gitattributes
|   |-- config.json
|   |-- feature_extractor_config.json
|   |-- model.safetensors
|   |-- preprocessor_config.json
|   |-- pytorch_model.bin
|   |-- README.md
|   |-- special_tokens_map.json
|   |-- tf_model.h5
|   |-- tokenizer_config.json
|   '-- vocab.json
'-- wav2vec-english-speech-emotion-recognition/     # emotion encoder
    |-- .gitattributes
    |-- config.json
    |-- preprocessor_config.json
    |-- pytorch_model.bin
    |-- README.md
    '-- training_args.bin
```

The Wav2Vec2-based models are [wav2vec2-base-960h](https://huggingface.co/facebook/wav2vec2-base-960h) and [wav2vec-english-speech-emotion-recognition](https://huggingface.co/r-f/wav2vec-english-speech-emotion-recognition).

### Running

| Goal | Command | Default URL |
|---|---|---|
| Backend + frontend together | `./scripts/run_all.sh` | API `:8000`, Studio `:3000` |
| FastAPI / WebRTC / WebSocket server only | `./scripts/run_api.sh` | `:8000` |
| React Studio only | `./scripts/run_frontend.sh` | `:3000` |
| Gradio batch app | `./scripts/run_gradio.sh` | `:7860` |
| Gradio live (mic) app | `./scripts/run_live.sh` | `:7860` |

The server starts warming up the model on boot; `GET /health` reports `warmup_stage` until it is ready.

---

## API

Interactive docs: `http://localhost:8000/docs`.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Service metadata (`service`, `version`, `status`, `docs_url`, `public_url`) |
| `GET` | `/health` | Health, warmup stage, encoder, active streams, FPS statistics |
| `GET` | `/gpu` | GPU VRAM allocated / reserved (GB) |
| `POST` | `/gpu/clear` | Clear the CUDA cache and run garbage collection |
| `POST` | `/generate` | Generate a talking-portrait video from audio + image |
| `GET` | `/progress/{request_id}` | Progress for a queued/running generation |
| `GET` | `/download/{request_id}` | Download the generated MP4 |
| `GET` | `/stream/{request_id}` | Stream the MP4 with HTTP `Range` support |
| `WS` | `/ws/stream/{request_id}` | Live frames while the video is generated (see [Live streaming](#live-streaming)) |
| `POST` | `/offer` | WebRTC SDP negotiation (finished file, or `"live": true`) |

### `POST /generate`

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
  "quality": "balanced",
  "decode_batch_size": 0,
  "client_request_id": "optional-browser-id",
  "live": false
}
```

| Field | Default | Notes |
|---|---|---|
| `audio_format` | `pcm` | `pcm`, `wav`, `mp3`, … |
| `sample_rate` | `16000` | 8 000–48 000 Hz |
| `image_base64` | `null` | Omit to use the built-in default portrait |
| `emotion` | `S2E` | `S2E` infers emotion from audio; or `neutral`/`happy`/`sad`/`angry`/`surprise`/`disgust` |
| `nfe` | `10` | ODE solver steps, 5–25 |
| `chunk_seconds` / `chunk_overlap_seconds` | `0` / `0.5` | Split long audio into independently generated windows and merge |
| `chunk_batch_size` / `chunk_pad_seconds` | `1` / `2.0` | Sample several chunks in one model batch (pads short chunks with silence) |
| `decode_batch_size` | `0` | Frames per decode micro-batch; `0` lets the VRAM tuner choose |
| `client_request_id` | `null` | Lets the browser poll `/progress/{id}` before the long-running call returns |
| `live` | `false` | Also publish frames to `/ws/stream/{id}` and to live `/offer` peers |

The response contains `request_id`, `download_url`, `stream_url`, `performance` timings, `encoder`, `resolution`, and the number of generated `chunks`.

> A generation is long-running. To show progress immediately, send `client_request_id` and poll `GET /progress/{client_request_id}`; to watch frames, open the WebSocket described below.

---

## Live streaming

Three transports share the same generated frames:

| Transport | Endpoint | Latency | Use when |
|---|---|---|---|
| MP4 file | `GET /stream/{id}` (Range) + `GET /download/{id}` | after completion | You need a complete, seekable file |
| WebSocket frames | `WS /ws/stream/{id}` | per decoded frame | You need low-latency pixels and can draw them yourself |
| WebRTC | `POST /offer` with `"live": true` | per decoded frame | You want the browser's native video pipeline |

### Flow

1. `POST /generate` with `"live": true` (and a `client_request_id` so the id is known up-front).
2. Immediately open `WS /ws/stream/{client_request_id}` — the socket waits up to 15 s for the request to be registered.
3. The server sends a `meta` message, then one binary message per decoded frame, then `done` (or `error`).

### WebSocket wire format

```text
server → {"type":"meta","fps":25,"width":512,"height":512,"channels":3,
          "first_idx":0,"total_frames":120,"has_audio":true}
server → binary: [u32 frame_idx][u32 frame_count][u32 pts_ms][raw RGB(A) bytes]
server → {"type":"gap","start":40,"end":47,"reason":"..."}
server → {"type":"done","result":{...}}     # same object /generate returns
server → {"type":"error","message":"..."}
```

- The 12-byte header is big-endian (`>III`); `count` is currently always `1`.
- Pixel payload is `width * height * channels` bytes of **uint8, RGB(A), row-major** — no per-message image encoding.
- Frames are **lossy by design**: each client has a bounded queue (64). A slow client skips frames and observes an index jump (`gap` / `frame_idx` discontinuity) rather than slowing the GPU.
- Close codes: **4404** unknown or already-finished request, **4408** the stream never produced metadata.

```js
// Minimal browser client (after POST /generate returns the request id)
const ws = new WebSocket(`ws://localhost:8000/ws/stream/${requestId}`);
ws.binaryType = "arraybuffer";
let meta;

ws.onmessage = (event) => {
  if (typeof event.data === "string") {
    const msg = JSON.parse(event.data);
    if (msg.type === "meta") meta = msg;
    if (msg.type === "done") console.log("finished", msg.result);
    if (msg.type === "error") console.error(msg.message);
    return;
  }
  const view = new DataView(event.data);
  const idx = view.getUint32(0);
  const ptsMs = view.getUint32(8);
  const pixels = new Uint8ClampedArray(event.data, 12);
  drawFrame(pixels, meta.width, meta.height, idx, ptsMs); // putImageData / WebGL upload
};
```

The React Studio uses the same protocol through `frontend/src/services/live.js` (routed via the Vite `/api` proxy in development).

---

## Per-component backends (ONNX / TensorRT)

The pipeline is split into named components, each of which can run in a different mode:

| Component | Unit | Call frequency |
|---|---|---|
| `emotion` | S2E emotion classifier | once / request |
| `audio` | wav2vec2 + projection | once / chunk |
| `direction` | motion direction (QR) | once / request |
| `fmt` | FMT vector field | NFE × chunks |
| `decoder` | StyleDecoder synthesis | per micro-batch |
| `refenc` | reference encoder | once / request |

| Mode | What it is |
|---|---|
| `eager` | Plain PyTorch — the reference and the universal fallback |
| `compile` | `torch.compile` (inductor) around the torch module |
| `onnx` | The exported ONNX graph on an execution provider (`trt` → `cuda` → `cpu`) |

The config is **hardware-specific** and generated on the target box — never hand-tuned from another machine:

```bash
./scripts/convert_to_onnx.sh
```

For each unit the script exports (`torch.export` → ONNX), runs a **parity check** against the PyTorch reference, **benchmarks** warmed latency, and records only the units that actually win. The generated file is `onnx/backends.generated.yaml`.

> **Pre-converted artifacts.** The six graphs are also published on Hugging Face at
> [`mohamedemam/float`](https://huggingface.co/mohamedemam/float) under `onnx/` (2.5 GB), so a
> target box can skip the conversion:
>
> ```bash
> huggingface-cli download mohamedemam/float --include "onnx/*" --local-dir ./hf
> cp -r hf/onnx ./onnx
> # the TensorRT engines are box-specific -> they are built on first use
> uv run --frozen --no-sync python -m fast_float.onnx.select_backends \
>     --onnx-dir ./onnx --out ./onnx/backends.generated.yaml --ckpt ./checkpoints/float.pth
> ```
>
> The graphs are portable (the checkpoint weights are baked in); the **engine cache is not** — it is
> tied to the GPU and the TensorRT/CUDA versions, and the backend plan is per-GPU, so the selector
> still has to run on the target box.

The `provider` recorded per unit is honoured at load time: a unit the selector benchmarked and parity-checked on `cuda`/`cpu` is never silently upgraded to TensorRT later (TRT's partitioned engine changes the numerics, which is how the broken decoder frames got deployed).

Resolution order at runtime:

1. `$FLOAT_BACKENDS_CONFIG` (explicit path)
2. `onnx/backends.generated.yaml` (produced on this box)
3. `config/backends.yaml` (committed default: everything `eager`)
4. `eager` fallback for anything missing or failing to load

Per-component debugging override: `FLOAT_MODE_fmt=eager`, and `FLOAT_DISABLE_BACKENDS=1` forces eager everywhere.

TensorRT is preferred when available (FP16 with an on-disk engine cache), then the CUDA EP, then CPU. Override via `ORT_ARGS`:

```bash
ORT_ARGS="--no-trt" ./scripts/convert_to_onnx.sh        # force the CUDA EP
ORT_ARGS="--trt-fp32" ./scripts/convert_to_onnx.sh      # strict parity (no FP16)
ORT_ARGS="--trt-max-batch 16" ./scripts/convert_to_onnx.sh
```

See [`docs/BACKEND_SELECTION.md`](docs/BACKEND_SELECTION.md) for the measurement methodology and [`docs/BUILD_RUNTIME_SESSIONS.md`](docs/BUILD_RUNTIME_SESSIONS.md) for session construction. Partial acceleration is always safe: any component that is missing, fails parity, or fails to load falls back to PyTorch.

> The decoder is frequently *not* worth exporting: the ONNX decoder runs at **0.16× eager** on an
> A6000 (the CUDA EP has no fast kernel for the modulated-conv stack, so its convolutions fall back),
> while `fmt` gained 2.6× on TensorRT FP16. Always trust the generated config over intuition.
>
> TensorRT is also opt-in for this converter (`--trt`) and off by default. On a T4 the TRT engine for the decoder ran at **0.09× eager** with `max_abs=4.375` / `cosine=0.983` versus eager: ORT partitions the graph across TRT-FP16 and CUDA-FP32, which is both slow and visibly wrong. The converter gates parity in uint8 frames (≤2 LSB max, ≤0.5 LSB mean) and writes `<model>.onnx.no_trt` so the runtime uses the same provider it validated. The install-time self-test applies the same gate, so a wrong artifact on a box without the marker also falls back to eager.
>
> Both gates compare **the same computation on both sides**: the exporter pins torch and ORT to fp32
> (TF32 alone moves this 22-conv stack by ~40 uint8 levels, which would drown the graph's own error),
> and the runtime self-test — where the provider is TF32 and cannot be pinned — judges the *provider*
> by the mean (≤2 LSB) plus a cosine floor (≥0.999) and reports the max as information. A
> structurally wrong graph (cosine 0.983) still fails both.
>
> The exported graph must be the *deployed* model: the converter traces the model as it runs and
> gates the artifact against a snapshot of that model, never against the module it traced. The
> decoder's graph is the dense modulated conv the model actually uses — an earlier export
> substituted a grouped variant, which the ORT CUDA EP serves only through a fallback kernel that
> drifts ~90 uint8 LSB and is not reproducible.
>
> The float parity gate treats `min_cosine`/`max_mean` as **hard floors**: a loose `atol` (TRT FP16 relaxes it 40×) can no longer skip the structural check. That OR was exactly how the 0.983-cosine graph was reported as PASS.
>
> The decode path reads the artifact as NCHW float frames, so the runtime rejects an artifact whose output is not 4D NCHW with 3-4 channels at session build; a wrongly formatted graph falls back to eager instead of rendering scrambled frames.

---

## Configuration

### Server

| Variable | Default | Purpose |
|---|---|---|
| `PORT` | `8000` | API port |
| `FRONTEND_PORT` | `3000` | Studio dev port (`run_all.sh`) |
| `VITE_API_URL` | `http://localhost:${PORT}` | Backend URL used by the frontend |
| `CORS_ORIGINS` | `http://localhost:3000,http://127.0.0.1:3000` | Comma-separated allowed origins |
| `NGPUS` | `1` | Fallback for `[float.runtime].ngpus` (GPUs used for parallel decoding) |
| `MAX_CONCURRENT_REQUESTS` | `10` | Request semaphore |
| `RESULTS_DIR` | `./results` | Fallback for `[float.checkpoint].res_dir` |
| `WARMUP_TIMEOUT_S` | `900` | Boot warmup budget before serving anyway |
| `WARMUP_WAIT_TIMEOUT_S` | `600` | How long `/generate` waits for warmup |
| `SKIP_WARMUP` | `0` | `1` skips boot warmup (debug) |
| `AVATER_PUBLIC_URL` | `http://localhost:${PORT}` | Public base URL returned in responses |
| `BACKUP_AVATER_PUBLIC_URL` | unset | Secondary public URL |
| `OPTIMIZATION_MODE` | `all` | Fallback for `[float.runtime].optimization_mode` |
| `USE_FP16` | `true` | Fallback for `[float.runtime].use_fp16` |
| `HYPERCORN_STARTUP_TIMEOUT` | `1200` | Hypercorn startup timeout (seconds) |

### FLOAT model & inference (`settings.toml [float]`)

All FLOAT model/inference options live in `fast_float/config/settings.toml`
under `[float]`; the API and the LiveKit avatar read the same file, so both
render identically. Edit the file instead of exporting env vars. The old
`NGPUS` / `OPTIMIZATION_MODE` / `USE_FP16` / `RESULTS_DIR` variables remain
fallbacks only for keys the file omits.

| Table | Keys |
|---|---|
| `[float]` | `livekit_nfe` — conversational avatar sampling steps (5–25) |
| `[float.checkpoint]` | `pretrained_dir`, `ckpt_path`, `res_dir` |
| `[float.runtime]` | `seed`, `fix_noise_seed`, `ngpus`, `optimization_mode`, `use_fp16`, `benchmark`, `video_encoder`, `no_crop`, `decode_batch_size`, `decode_channels_last`, `decode_autocast` |
| `[float.video]` | `input_size`, `input_nc`, `fps` |
| `[float.audio]` | `sampling_rate`, `audio_marcing`, `wav2vec_sec`, `attention_window`, `only_last_features`, `average_emotion` |
| `[float.paths]` | `wav2vec_model_path`, `audio2emotion_path`, `emotion_onnx_path`, `wav2vec_onnx_path`, `fmt_onnx_path`, `ref_path`, `aud_path`, `res_video_path`, `emo` |
| `[float.model]` | `style_dim`, `dim_a`, `dim_w`, `dim_h`, `dim_m`, `dim_e`, `fmt_depth`, `num_heads`, `mlp_ratio`, `no_learned_pe`, `num_prev_frames`, `max_grad_norm`, `*_dropout_prob` |
| `[float.ode]` | `nfe`, `ode_atol`, `ode_rtol`, `torchdiffeq_ode_method` |
| `[float.guidance]` | `a_cfg_scale`, `e_cfg_scale`, `r_cfg_scale` |
| `[float.diffusion]` | `n_diff_steps`, `diff_schedule`, `diffusion_mode` |
| `[float.request]` | `/generate` defaults: `nfe`, `seed`, `emotion`, `a_cfg_scale`, `e_cfg_scale`, `r_cfg_scale`, `no_crop`, `decode_batch_size` |

> Architecture values (`dim_*`, `fmt_depth`, `input_size`, `fps`,
> `wav2vec_sec`, …) are part of the checkpoint's shape and must match
> `checkpoints/float.pth`. Tune `[float.runtime]`, `[float.ode]`,
> `[float.guidance]` or `[float.request]` for speed/quality instead.

### Backends

| Variable | Default | Purpose |
|---|---|---|
| `FLOAT_BACKENDS_CONFIG` | unset | Explicit per-component config path |
| `FLOAT_MODE_<COMPONENT>` | unset | Override one component (e.g. `FLOAT_MODE_fmt=eager`) |
| `FLOAT_DISABLE_BACKENDS` | unset | Force `eager` everywhere |
| `FLOAT_TRT_MAX_BATCH` | `16` | TensorRT engine max batch |

### Decoding & performance

| Variable | Default | Purpose |
|---|---|---|
| `FLOAT_DECODE_TUNE_CACHE` | `~/.cache/fast_float/decode_tune.json` | Decode autotuner cache (`0` disables) |
| `FLOAT_CACHE_DIR` | `~/.cache/fast_float` | Base cache directory |
| `FLOAT_DECODE_CHANNELS_LAST` | `0` | Run the eager/compiled decoder in channels_last (A/B per device) |
| `FLOAT_DECODE_AUTOCAST` | unset | `fp16` or `bf16` autocast for the eager/compiled decoder; off by default |
| `FLOAT_SEAM_BLEND` | `3` | Frames cross-faded at chunk seams (0 disables) |
| `FLOAT_CUDNN_BENCHMARK` | `0` | Enable cuDNN autotuning (adds startup/heap churn; off by default) |
| `FLOAT_SKIP_BACKEND_SELFTEST` | unset | Skip the one-shot ONNX install-time self-tests (`1`; benchmarking only) |

> Decode knobs are opt-in on purpose: channels_last and autocast can change numerics or be a pessimization depending on the GPU and kernel mix. Measure with the tuner enabled (`FLOAT_DECODE_TUNE_CACHE`) before baking them into a deployment.

### Logging

| Variable | Default | Purpose |
|---|---|---|
| `FLOAT_LOG_LEVEL` | unset | Explicit log level |
| `FLOAT_DEBUG` / `FLOAT_TIMING` | unset | Per-call timing and debug lines |

---

## Development

```bash
# Python
uv run --frozen --no-sync python -m pytest -q     # tests live in tests/
uv run --frozen --no-sync ruff check .
uv run --frozen --no-sync ruff format .

# Frontend
cd frontend
npm ci
npm run lint        # oxlint
npm test            # Vitest + Testing Library (jsdom)
npm run build
npm run dev         # :3000, proxies /api -> :8000 (including WebSockets)
```

The frontend tests cover the API-base resolution, safe storage, request
timeouts, the shared telemetry poller, progress rendering, the live-preview
transport lifecycle and the generation error path; they need no backend.
`npm run preview` serves the production build through the same `/api` proxy
and honours `PORT` / `VITE_API_PROXY_TARGET` like the dev server.

The ONNX export tests need the conversion stack (`onnxscript`, `onnx>=1.17`), which is **not** part of the portable runtime lockfile — install it with `./scripts/convert_to_onnx.sh` or skip those tests on a runtime-only box.

Git hooks (ruff, gitleaks, commitizen, typos, markdownlint, shellcheck) are configured in `.pre-commit-config.yaml`:

```bash
pre-commit install --install-hooks
```

See [`AGENT.md`](AGENT.md) for the architecture guide and [`docs/UI_UX_BRIEF.md`](docs/UI_UX_BRIEF.md) for the frontend design notes.

---

## Repository layout

```text
float-fast/
├── pyproject.toml              # metadata, deps, uv indexes, ruff/pytest config
├── uv.lock                     # pinned lockfile
├── Dockerfile                  # container build (CUDA 12.1 base)
├── config/
│   ├── api.json                # optional public URL overrides
│   └── backends.yaml           # committed backend default (all eager)
├── scripts/                    # setup, run, ONNX conversion, TensorRT install
├── checkpoints/                # model weights (downloaded)
├── docs/                       # backend selection, batch decoding, streaming, UI notes
├── frontend/                   # React + Vite Studio (components, services/live.js)
├── tests/                      # pytest suite (no GPU required for most tests)
└── fast_float/
    ├── main.py                 # FastAPI entrypoint (routers, CORS, lifespan)
    ├── core/                   # request pipeline, service, shared app state
    ├── routers/                # generation, health, streaming (WS), webrtc
    ├── schemas/                # Pydantic v2 request/response models
    ├── streaming/              # frame dispatcher, live fan-out, stream types
    ├── runtimes/               # runtime backends (ONNX session selection)
    ├── onnx/                   # export, parity, dispatch, backend config
    ├── utils/                  # GPU, media, logging, timing, video optimizer, tuner
    ├── models/float/           # FLOAT generator, FMT, style decoder, encoders
    ├── interfaces/             # Gradio live app
    └── legacy/                 # upstream CLI (generate.py) + batch Gradio app
```

---

## Known limitations

- **Linux + NVIDIA only.** NVENC, CUDA streams, and the TensorRT path assume it; the portable ONNX runtime can run CPU-only, but the server targets GPU boxes.
- **Single API process for live streams.** The live-stream registry lives in process memory, so run one API worker. Horizontal scaling would need an external broker.
- **Live transport is lossy by design.** WebSocket and live WebRTC drop frames for slow consumers; use the finished MP4 (`/download`, `/stream`) when every frame matters.
- **Backend config is per machine.** `onnx/backends.generated.yaml` is an artifact of the box that produced it; regenerate after changing GPUs, drivers, or the ONNX/TensorRT stack.
- **TensorRT/ORT ABI pins matter.** ONNX Runtime 1.23 expects TensorRT 10.9 (`libnvinfer.so.10`); mismatched majors fail to load.
- **Non-commercial license** inherited from upstream (see below).

---

## License

This work is licensed under a [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License](https://creativecommons.org/licenses/by-nc-nd/4.0/), inherited from the upstream FLOAT release. **You may not use this work for commercial purposes and may use it only for research purposes.** For any commercial inquiries or collaboration opportunities, please contact <daniel@deepbrain.io>.

## Development note (upstream)

This repository builds on a research demonstration implementation provided as a one-time code drop. For research-related inquiries about the original method, please contact the first author [Taekyung Ki](https://github.com/TaekyungKi). The original work was done during the first author's South Korean Alternative Military Service at DeepBrain AI. Training code is not released.

---

## About FLOAT (upstream)

The remainder of this document describes the original FLOAT method, results, and credits.

![preview](./assets/float-abstract.png)

**FLOAT: Generative Motion Latent Flow Matching for Audio-driven Talking Portrait**<br>
[Taekyung Ki](https://taekyungki.github.io), [Dongchan Min](https://kevinmin95.github.io), [Gyeongsu Chae](https://www.aistudios.com/ko)

Project Page: <https://deepbrainai-research.github.io/float/>

**Abstract**: *With the rapid advancement of diffusion-based generative models, portrait image animation has achieved remarkable results. However, it still faces challenges in temporally consistent video generation and fast sampling due to its iterative sampling nature. This paper presents FLOAT, an audio-driven talking portrait video generation method based on flow matching generative model. Instead of a pixel-based latent space, we take advantage of a learned orthogonal motion latent space, enabling efficient generation and editing of temporally consistent motion. To achieve this, we introduce a transformer-based vector field predictor with an effective frame-wise conditioning mechanism. Additionally, our method supports speech-driven emotion enhancement, enabling a natural incorporation of expressive motions. Extensive experiments demonstrate that our method outperforms state-of-the-art audio-driven talking portrait methods in terms of visual quality, motion fidelity, and efficiency.*

**TL;DR: FLOAT is a flow matching based audio-driven talking portrait video generation method, which can enhance the speech-driven emotional motion.**

## Generation results

| Result 1 | Result 2 |
|---------------|---------|
| <video src="https://github.com/user-attachments/assets/8c00274d-795d-4ee9-870f-84a859f3e23f"> </video> | <video src="https://github.com/user-attachments/assets/c6e142b3-519b-4cda-b26d-e088414b478d"> </video> |

| Result 3 | Result 4 |
|--------|-----------|
| <video src="https://github.com/user-attachments/assets/7b201a5f-a293-46cd-974f-0612062d8d94"> </video> |  <video src="https://github.com/user-attachments/assets/dd4b74dd-40b4-4864-b87d-7cbf4f0d66da"> </video> |

<br>
The method runs faster than current diffusion-based methods with fewer sampling steps and lower memory cost. For more details, please refer to the paper.
<div align='center'>
    <image width= 80% src="./assets/fps.png"> </image>
</div>

## Updates

- [2025.06.26] FLOAT is accepted to **[ICCV 2025](https://iccv.thecvf.com)**
- [2025.02.17] The inference code and checkpoints are released under a **[Non-commercial License](https://creativecommons.org/licenses/by-nc-nd/4.0/)**
- [2024.12.03] Selected as a [HuggingFace Daily Papers](https://huggingface.co/papers?date=2024-12-03) on December 3, 2024
- [2024.12.02] The paper is publicly available on [ArXiv](https://arxiv.org/abs/2412.01064)

## Original usage (from upstream)

### Generating talking-portrait video from a single image and audio

**Pre-processing** — for better quality, please note:

- FLOAT is trained on frontal head-pose distributions. Non-frontal images may lead to suboptimal results.
- The inference code includes an automatic face-cropping function, which may involve black **padding** regions. You can manually disable cropping, though it may lead to suboptimal performance.
- If your audio contains heavy background music, use [ClearVoice](https://github.com/modelscope/ClearerVoice-Studio) to extract the vocals first.

> In this fork the CLI lives at `fast_float/legacy/generate.py` and expects the checkpoints described above.

**1. Emotion from audio** — generate with an emotion inferred from the audio (omit `--emo`). Adjust intensity with `--e_cfg_scale` (default 1; try 5–10 for more intensity).

```bash
CUDA_VISIBLE_DEVICES=0 python fast_float/legacy/generate.py \
    --ref_path path/to/reference/image \
    --aud_path path/to/audio \
    --seed 15 \
    --a_cfg_scale 2 \
    --e_cfg_scale 1 \
    --ckpt_path ./checkpoints/float.pth \
    --no_crop                    # [optional] skip cropping
```

**2. Redirecting emotion** — specify `--emo` with one of seven emotions: `['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']`.

```bash
CUDA_VISIBLE_DEVICES=0 python fast_float/legacy/generate.py \
    --ref_path path/to/reference/image \
    --aud_path path/to/audio \
    --emo 'happy' \
    --seed 15 \
    --a_cfg_scale 2 \
    --e_cfg_scale 1 \
    --ckpt_path ./checkpoints/float.pth \
    --no_crop                    # [optional] skip cropping
```

<video src="https://github.com/user-attachments/assets/fb3826cd-231b-46f2-809b-11adebe9a1cf"> </video>

**3. Running example and results**

```bash
CUDA_VISIBLE_DEVICES=0 python fast_float/legacy/generate.py \
    --ref_path assets/sam_altman.webp \
    --aud_path assets/aud-sample-vs-1.wav \
    --seed 15 \
    --a_cfg_scale 2 \
    --e_cfg_scale 1 \
    --ckpt_path ./checkpoints/float.pth
```

| Before crop | After crop | Result |
|---------------|---------|--------|
| ![Before crop](assets/sam_altman.webp) | ![After crop](assets/sam_altman_512x512.jpg) | <video src='https://github.com/user-attachments/assets/3353e4e0-00f5-429b-bc66-5db9a72186b8'> </video> |

## Citation

If you use FLOAT, please cite the original authors:

```bibtex
@inproceedings{ki2025float,
  title={Float: Generative motion latent flow matching for audio-driven talking portrait},
  author={Ki, Taekyung and Min, Dongchan and Chae, Gyeongsu},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={14699--14710},
  year={2025}
}
```

## Related works

- [StyleLipSync: Style-based Personalized Lip-sync Video Generation](https://arxiv.org/abs/2305.00521)<br>
- [StyleTalker: One-shot Style-based Audio-driven Talking Head Video Generation](https://arxiv.org/abs/2208.10922)<br>
- [Export3D: Learning to Generate Conditional Tri-plane for 3D-aware Expression Controllable Portrait Animation](https://arxiv.org/abs/2404.00636)<br>
- [Latent Image Animator: Learning to Animate Images via Latent Space Navigation](https://arxiv.org/abs/2203.09043)<br>

## Acknowledgements

The source images and audio are collected from the internet and other baselines, such as LIA, SadTalker, EMO, VASA-1, Hallo, LivePortrait, Loopy, and others. We appreciate their valuable contributions to this field. We employ the Wav2Vec2.0-based speech emotion recognizer by [Rob Field](https://huggingface.co/r-f/wav2vec-english-speech-emotion-recognition). We appreciate this good work.
