# FLOAT Fast

**An unofficial serving and streaming layer for [FLOAT](https://github.com/deepbrainai-research/float) (audio-driven talking-portrait generation, ICCV 2025).**

> ## Read this first
>
> - **Unofficial.** This project is independent. It is not affiliated with, endorsed by, or sponsored by the FLOAT authors or DeepBrain AI.
---

## What this repository is

FLOAT is a research model that animates a single portrait image from an audio clip. This repository is a separate piece of software that sits **around** an upstream FLOAT installation and adds:

- a **FastAPI** server (HTTP byte-range streaming, NVENC encoding, WebRTC signaling),
- **live frame streaming** over WebSocket and WebRTC while a video is still being generated,
- a **React + Vite "Studio"** frontend,
- **Gradio** apps (batch and live microphone),
- **VRAM-aware batched decoding** and an asynchronous multi-stream pipeline,
- scripts to **convert and benchmark components locally** with ONNX / TensorRT.

The model architecture, method, trained checkpoints, and original inference code are entirely the work of Taekyung Ki, Dongchan Min, and Gyeongsu Chae (DeepBrain AI). Please cite their paper (see [Citation](#citation)).

### What is and is not in this repository

| In this repository (original work) | **Not** in this repository (you obtain it from the authors) |
|---|---|
| FastAPI server, routers, schemas | FLOAT source code |
| WebSocket / WebRTC streaming code | FLOAT checkpoints (`float.pth`) |
| React Studio frontend | Converted or fine-tuned FLOAT weights (including ONNX / TensorRT files) |
| Gradio apps, scripts, Dockerfile | wav2vec2 models used by FLOAT |
| Benchmark and conversion *scripts* | Example images and audio from upstream |

Converted artifacts (for example ONNX graphs) contain the model weights and are therefore derivatives of the upstream checkpoint. **They are generated on your own machine and are never published from this project.** Please do not upload them anywhere either.

---

## Requirements

- Linux with an **NVIDIA GPU** (upstream is tested on A100 and V100; this stack also runs on T4 / RTX-class cards).
- **Python 3.10**, managed by [uv](https://docs.astral.sh/uv/getting-started/installation/).
- `ffmpeg`, `git`, `wget`, `curl`.
- **Node 20.19+ / 22.12+** for the frontend.
- Optional: **TensorRT 10.9 (CUDA 12)**. ONNX Runtime 1.23 links against `libnvinfer.so.10`; TensorRT 11 cannot be loaded.

## Setup

```bash
git clone https://github.com/mohamed-em2m/fast-float.git
cd fast-float
./scripts/setup.sh
```

`setup.sh` installs dependencies and then fetches FLOAT **from the authors' own repository and download links**, into a local folder that is git-ignored. You are responsible for accepting the upstream license when you download it.

Manual steps, if you prefer:

```bash
# 1. Python environment
uv sync --frozen

# 2. Get FLOAT from the authors (not included here)
git clone https://github.com/deepbrainai-research/float.git third_party/float

# 3. Get the checkpoints from the authors' instructions
#    https://github.com/deepbrainai-research/float#preparing-checkpoints
#    and place them under ./checkpoints as described there.
```

Expected layout of `./checkpoints` (as defined by upstream):

```text
./checkpoints
|-- float.pth
|-- wav2vec2-base-960h/
'-- wav2vec-english-speech-emotion-recognition/
```

The two wav2vec2 models are third-party and have their own licenses, see [Third-party components](#third-party-components).

### Running

| Goal | Command | Default URL |
|---|---|---|
| Backend + frontend together | `./scripts/run_all.sh` | API `:8000`, Studio `:3000` |
| API server only | `./scripts/run_api.sh` | `:8000` |
| React Studio only | `./scripts/run_frontend.sh` | `:3000` |
| Gradio batch app | `./scripts/run_gradio.sh` | `:7860` |
| Gradio live (mic) app | `./scripts/run_live.sh` | `:7860` |

The server warms up the model on boot; `GET /health` reports `warmup_stage` until it is ready.

---

## API

Interactive docs: `http://localhost:8000/docs`.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Service metadata |
| `GET` | `/health` | Health, warmup stage, encoder, active streams, FPS statistics |
| `GET` | `/gpu` | GPU VRAM allocated / reserved (GB) |
| `POST` | `/gpu/clear` | Clear the CUDA cache and run garbage collection |
| `POST` | `/generate` | Generate a talking-portrait video from audio + image |
| `GET` | `/progress/{request_id}` | Progress of a queued or running generation |
| `GET` | `/download/{request_id}` | Download the generated MP4 |
| `GET` | `/stream/{request_id}` | Stream the MP4 with HTTP `Range` support |
| `WS` | `/ws/stream/{request_id}` | Live frames while the video is generated |
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
| `audio_format` | `pcm` | `pcm`, `wav`, `mp3`, ... |
| `sample_rate` | `16000` | 8 000 to 48 000 Hz |
| `image_base64` | `null` | Omit to use the default portrait you configured locally |
| `emotion` | `S2E` | `S2E` infers emotion from audio, or one of `neutral` / `happy` / `sad` / `angry` / `surprise` / `disgust` |
| `nfe` | `10` | ODE solver steps, 5 to 25 |
| `chunk_seconds` / `chunk_overlap_seconds` | `0` / `0.5` | Split long audio into windows and merge |
| `chunk_batch_size` / `chunk_pad_seconds` | `1` / `2.0` | Sample several chunks in one batch |
| `decode_batch_size` | `0` | Frames per decode micro-batch; `0` lets the VRAM tuner choose |
| `client_request_id` | `null` | Lets the client poll `/progress/{id}` before the call returns |
| `live` | `false` | Also publish frames to `/ws/stream/{id}` and live `/offer` peers |

The response contains `request_id`, `download_url`, `stream_url`, `performance` timings, `encoder`, `resolution`, and `chunks`.

---

## Live streaming

| Transport | Endpoint | Latency | Use when |
|---|---|---|---|
| MP4 file | `GET /stream/{id}` + `GET /download/{id}` | after completion | You need a complete, seekable file |
| WebSocket frames | `WS /ws/stream/{id}` | per decoded frame | You want raw low-latency pixels |
| WebRTC | `POST /offer` with `"live": true` | per decoded frame | You want the browser's native video pipeline |

**Flow:** `POST /generate` with `"live": true` and a `client_request_id`, then immediately open `WS /ws/stream/{client_request_id}`. The server sends a `meta` message, one binary message per decoded frame, then `done` (or `error`).

```text
server -> {"type":"meta","fps":25,"width":512,"height":512,"channels":3,
           "first_idx":0,"total_frames":120,"has_audio":true}
server -> binary: [u32 frame_idx][u32 frame_count][u32 pts_ms][raw RGB(A) bytes]
server -> {"type":"gap","start":40,"end":47,"reason":"..."}
server -> {"type":"done","result":{...}}
server -> {"type":"error","message":"..."}
```

- The 12-byte header is big-endian (`>III`); `count` is currently always `1`.
- Pixels are uint8 RGB(A), row-major, with no per-message image encoding.
- Streaming is **lossy by design**: each client has a bounded queue (64), and slow clients skip frames rather than slowing the GPU.
- Close codes: `4404` unknown or finished request, `4408` stream never produced metadata.

```js
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
  drawFrame(pixels, meta.width, meta.height, idx, ptsMs);
};
```

---

## Optional: local ONNX / TensorRT acceleration

Each pipeline component can run in `eager` (plain PyTorch), `compile` (`torch.compile`), or `onnx` (TensorRT, then CUDA, then CPU). The choice is made **per machine** from a config you generate locally:

```bash
./scripts/convert_to_onnx.sh
```

This script exports each component from **your own local copy** of the checkpoint, checks numerical parity against PyTorch, benchmarks it, and keeps only the components that are actually faster. Output goes to `onnx/` (git-ignored). **Do not commit, upload, or redistribute these files**, because they contain the model weights.

| Component | Unit | Call frequency |
|---|---|---|
| `emotion` | S2E emotion classifier | once / request |
| `audio` | wav2vec2 + projection | once / chunk |
| `direction` | motion direction | once / request |
| `fmt` | FMT vector field | NFE x chunks |
| `decoder` | StyleDecoder synthesis | per micro-batch |
| `refenc` | reference encoder | once / request |

Resolution order at runtime: `$FLOAT_BACKENDS_CONFIG`, then `onnx/backends.generated.yaml`, then `config/backends.yaml` (everything `eager`), then an `eager` fallback for anything that fails to load. Partial acceleration is always safe.

Measured results are hardware-specific. For example, the ONNX decoder ran much slower than eager on one GPU while `fmt` sped up with TensorRT FP16. Always trust the config generated on your own machine. See [`docs/BACKEND_SELECTION.md`](docs/BACKEND_SELECTION.md) for methodology.

Useful switches: `ORT_ARGS="--no-trt"`, `ORT_ARGS="--trt-fp32"`, `FLOAT_MODE_<component>=eager`, `FLOAT_DISABLE_BACKENDS=1`.

---

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `PORT` | `8000` | API port |
| `FRONTEND_PORT` | `3000` | Studio dev port |
| `VITE_API_URL` | `http://localhost:${PORT}` | Backend URL used by the frontend |
| `CORS_ORIGINS` | `http://localhost:3000,http://127.0.0.1:3000` | Allowed origins |
| `NGPUS` | `1` | GPUs used for parallel decoding |
| `MAX_CONCURRENT_REQUESTS` | `10` | Request semaphore |
| `RESULTS_DIR` | `./results` | Output directory |
| `WARMUP_TIMEOUT_S` / `WARMUP_WAIT_TIMEOUT_S` | `900` / `600` | Warmup budgets |
| `SKIP_WARMUP` | `0` | `1` skips boot warmup (debug) |
| `USE_FP16` | `true` | FP16 model weights |
| `FLOAT_BACKENDS_CONFIG` | unset | Explicit per-component backend config |
| `FLOAT_DECODE_TUNE_CACHE` | `~/.cache/fast_float/decode_tune.json` | Decode autotuner cache (`0` disables) |
| `FLOAT_SEAM_BLEND` | `3` | Frames cross-faded at chunk seams |
| `FLOAT_LOG_LEVEL` / `FLOAT_DEBUG` / `FLOAT_TIMING` | unset | Logging and timing |

Decode options such as channels_last and autocast are opt-in. Measure on your GPU before enabling them.

---

## Development

```bash
# Python
uv run --frozen --no-sync python -m pytest -q
uv run --frozen --no-sync ruff check .
uv run --frozen --no-sync ruff format .

# Frontend
cd frontend && npm ci && npm run lint && npm run build
npm run dev     # :3000, proxies /api -> :8000 (including WebSockets)
```

See [`AGENT.md`](AGENT.md) for the architecture guide.

```text
fast-float/
|-- fast_float/          # original code: server, routers, streaming, utils, backends
|-- frontend/            # React + Vite Studio
|-- scripts/             # setup, run, local ONNX conversion
|-- config/ docs/ tests/
|-- third_party/         # git-ignored: upstream FLOAT is fetched here by setup
|-- checkpoints/         # git-ignored: downloaded from the authors
`-- onnx/                # git-ignored: generated locally, never redistributed
```

## Known limitations

- Linux and NVIDIA GPU only.
- One API process for live streams (the stream registry lives in memory).
- Live transports are lossy by design; use the MP4 when every frame matters.
- Backend config is per machine and must be regenerated after hardware or driver changes.
- TensorRT / ONNX Runtime versions must match (see Requirements).

---

## Responsible use

This software can generate realistic videos of a person speaking. Only use portraits and voices you have the **right and consent** to use. Do not use it to impersonate people, to deceive, or to harass. You are responsible for complying with the laws that apply to you, including rules on synthetic media and personal likeness.

---

## Third-party components

| Component | Source | License |
|---|---|---|
| FLOAT code and checkpoints | [deepbrainai-research/float](https://github.com/deepbrainai-research/float) | CC BY-NC-ND 4.0, non-commercial / research only |
| wav2vec2-base-960h | [facebook/wav2vec2-base-960h](https://huggingface.co/facebook/wav2vec2-base-960h) | See the model page |
| wav2vec-english-speech-emotion-recognition | [r-f/wav2vec-english-speech-emotion-recognition](https://huggingface.co/r-f/wav2vec-english-speech-emotion-recognition) | See the model page |
| Other Python / Node dependencies | `pyproject.toml`, `frontend/package.json` | Each under its own license |

Please check each license yourself before use.

## License

- **Original code in this repository** (server, streaming, frontend, scripts): licensed under **[CHOOSE: e.g. MIT]**, see `LICENSE`.
- **FLOAT itself** (code and checkpoints) is **not** covered by that license. It stays under the authors' [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/) license and non-commercial research terms, and you must comply with it when you download and use it.
- Because this project needs FLOAT to run, **using the combined setup is limited to non-commercial research** unless the FLOAT rights holders grant you permission.

## Takedown and contact

If you are a rights holder and believe something here should change, please open an issue or email **[YOUR EMAIL]**. I will respond promptly and fix problems quickly.

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

## Acknowledgements

Thanks to Taekyung Ki, Dongchan Min, and Gyeongsu Chae for FLOAT, and to Rob Field for the speech emotion recognizer it uses.
