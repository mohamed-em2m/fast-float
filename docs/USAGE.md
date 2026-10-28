# Using FLOAT Fast

End-to-end guide: install, pick a run mode, call the API, watch a live stream,
run a conversational avatar, read the logs, and get out of trouble.

Everything here is a real command or a field the code accepts. Deeper design
notes live in the per-topic docs listed at the end.

---

## 1. What runs

`float-fast` is a set of cooperating processes. A run mode simply chooses which
of them start, and which one owns the GPU.

| Process | Module | Role (log file) | GPU | Purpose |
| --- | --- | --- | --- | --- |
| API server | `fast_float.main:app` | `api` (`logs/api.log`) | yes, single-process mode | HTTP/WebSocket/WebRTC, inference |
| GPU worker | `fast_float.workers.inference_worker` | `worker` (`logs/worker.log`) | yes, dual-process mode | inference only, over ZeroMQ + shared memory |
| Studio frontend | `frontend/` (Vite) | -- | no | browser UI on `:3000` |
| Gemini worker | `fast_float.livekit.gemini.agent` | `agent` (`logs/agent.log`) | no | LiveKit Agents speech-to-speech conversation |
| CLI supervisor | `fast_float.cli` | `supervisor` (`logs/supervisor.log`) | no | starts/stops the stack, tees `logs/combined.log` |
| Gradio apps | `fast_float.interfaces.*` | `gradio` (`logs/gradio.log`) | yes | batch upload / live microphone UI |

In **single-process** mode the API owns CUDA and runs inference in its thread
pool. In **dual-process** mode (the default) a separate worker owns CUDA and the
API stays a lightweight streaming front end -- see
[`DUAL_PROCESS_IPC.md`](DUAL_PROCESS_IPC.md).

---

## 2. Prerequisites

- Linux with an **NVIDIA GPU** and a driver that supports CUDA 12.
- **Python 3.10** (pinned by `.python-version`), managed by
  [`uv`](https://docs.astral.sh/uv/).
- `ffmpeg`, `git`, `wget`, `curl`.
- **Node 20.19+ / 22.12+** for the Studio.
- Optional: **TensorRT 10.9 (CUDA 12)** for the TRT execution provider.

Check the driver before anything else:

```bash
nvidia-smi
ffmpeg -version | head -n 1
```

---

## 3. Install

```bash
git clone https://github.com/mohamed-em2m/fast-float.git
cd fast-float
./scripts/setup.sh
```

`setup.sh` installs the system packages, Node, and `uv`; syncs the lockfile;
downloads the checkpoints; layers the ONNX export stack; and probes TensorRT.
It reuses an existing `.venv`, so a second run is cheap.

Manual equivalent:

```bash
uv sync --frozen
./scripts/download_checkpoints.sh
```

Checkpoints land in `checkpoints/` (`float.pth`, `wav2vec2-base-960h/`,
`wav2vec-english-speech-emotion-recognition/`). Confirm:

```bash
ls -lh checkpoints/float.pth
```

Optional, on the target box: export and benchmark ONNX/TensorRT graphs.

```bash
./scripts/convert_to_onnx.sh
```

Or download the pre-converted graphs (the TensorRT engine cache is still built
per box):

```bash
hf download mohamedemam/float --include "onnx/*" --local-dir ./hf   # older: huggingface-cli
cp -r hf/onnx ./onnx
uv run --frozen --no-sync python -m fast_float.onnx.select_backends \
    --onnx-dir ./onnx --out ./onnx/backends.generated.yaml --ckpt ./checkpoints/float.pth
```

Without a generated config the runtime stays on `eager` PyTorch: correct, just
slower. See [`BACKEND_SELECTION.md`](BACKEND_SELECTION.md).

---

## 4. Run modes

| Goal | Command | Starts | URLs |
| --- | --- | --- | --- |
| Backend + Studio | `./scripts/run_all.sh` | API, frontend | API `:8000`, Studio `:3000` |
| API only | `./scripts/run_api.sh` | API (single process) | `:8000` |
| Dual-process stack | `./scripts/run_dual_process.sh` | worker + API + frontend (+ agent) | `:8000`, `:3000` |
| Studio only | `./scripts/run_frontend.sh` | Vite | `:3000` |
| Gradio batch | `./scripts/run_gradio.sh` | Gradio | `:7860` |
| Gradio live (+batch) | `./scripts/run_live.sh` | Gradio | `:7860` |
| Gemini worker only | `./scripts/run_gemini_agent.sh` | LiveKit Agents worker | -- |
| CLI (any of the above) | `uv run --frozen --no-sync fast-float ...` | see below | -- |

All launchers use `uv run --frozen --no-sync`. Do not drop `--no-sync`: a plain
`uv run` re-syncs the venv to the lock, which pins the **CPU** `onnxruntime`,
and the server boots without a CUDA/TensorRT execution provider.

### Backend + Studio

```bash
./scripts/run_all.sh
# override ports
PORT=9000 FRONTEND_PORT=3100 ./scripts/run_all.sh
```

Overrides: `PORT`, `FRONTEND_PORT`, `VITE_API_URL`, `CORS_ORIGINS`, `NGPUS`,
`LIVEKIT_START` (`auto`/`yes`/`no`), `LIVEKIT_URL` / `LIVEKIT_API_KEY` /
`LIVEKIT_API_SECRET`, `FLOAT_GEMINI_WORKER` (`auto`/`yes`/`no`). `Ctrl+C` stops
everything.

The Studio is served by Vite and calls the backend through a same-origin proxy:
`/api/*` is rewritten to `/*` and forwarded to `http://localhost:${PORT}`. That
is why browser code uses `/api/generate` while a direct client uses
`/generate`.

### API only

```bash
./scripts/run_api.sh
PORT=9000 ./scripts/run_api.sh
```

Single process: the API loads the model itself. Use this when you do not need
the Studio or the worker split.

Neither `run_api.sh` nor `run_all.sh` starts the GPU worker, while
`settings.toml` keeps `[ipc] enabled = true`. LiveKit conversations started by
those launchers therefore expect a worker on `:5555`: either run
`run_dual_process.sh` for conversational sessions, start `fast-float worker`
alongside, or set `[ipc] enabled = false` in an alternate config directory
(`FLOAT_CONFIG_DIR=...`). Plain `/generate` requests do not use IPC.

### Dual-process stack

```bash
./scripts/run_dual_process.sh
```

Exports the IPC settings for both halves:

```bash
FLOAT_IPC_ENABLED=true
FLOAT_ZMQ_CONTROL_PORT=5555
FLOAT_ZMQ_FRAMES_PORT=5556
FLOAT_SHM_NAME=float_frame_ring
FLOAT_SHM_SLOTS=64
```

The worker reads the port/ring variables; the API decides whether to use IPC
from `[ipc] enabled` in `settings.toml` (already `true`), and connects to the
endpoints listed there. So the env vars and `[ipc]` must agree -- change both,
or leave the defaults. Bump `FLOAT_SHM_SLOTS` for high-fps or multiple
consumers.

The worker must bind the sockets before the API connects, so start it first
(the launcher sleeps 2 s for exactly that reason). Override the ports only
together with `[ipc]`:

```bash
FLOAT_ZMQ_CONTROL_PORT=6555 FLOAT_ZMQ_FRAMES_PORT=6556 ./scripts/run_dual_process.sh
```

### CLI

`fast-float` is the installed entry point (`pyproject.toml` ->
`fast_float.cli:main`).

```bash
uv run --frozen --no-sync fast-float serve                     # worker + API (dual-process default)
uv run --frozen --no-sync fast-float serve --no-dual-process   # API only, in-process inference
uv run --frozen --no-sync fast-float serve --frontend          # + Studio on :3000
uv run --frozen --no-sync fast-float serve -p 9000 --frontend-port 3100
uv run --frozen --no-sync fast-float serve --server-type uvicorn
uv run --frozen --no-sync fast-float serve --ngpus 2 --shm-slots 128
uv run --frozen --no-sync fast-float worker                    # GPU worker only
uv run --frozen --no-sync fast-float ui -p 3000                # Studio only (npm)
uv run --frozen --no-sync fast-float agent                     # Gemini Live worker
```

| Flag | Default | Applies to |
| --- | --- | --- |
| `--host` | `0.0.0.0` | `serve`, `ui` |
| `-p`, `--port` | `8000` | `serve` |
| `--server-type` | `hypercorn` | `serve` |
| `--dual-process` / `--no-dual-process` | enabled | `serve` |
| `--frontend`, `--frontend-port` | off, `3000` | `serve` |
| `--ngpus` | from config | `serve` |
| `--control-port` | `5555` | `serve`, `worker` |
| `--frames-port` | `5556` | `serve`, `worker` |
| `--shm-name` | `float_frame_ring` | `serve`, `worker` |
| `--shm-slots` | `64` | `serve`, `worker` |

### Gradio

```bash
./scripts/run_gradio.sh          # batch: upload image + audio -> video, :7860
./scripts/run_live.sh            # live microphone -> chunked near-real-time video
./scripts/run_gradio.sh --port 7861 --share
```

The live app and the API must not both own the GPU at full tilt; stop one when
you switch.

### Gemini Live worker

```bash
./scripts/run_gemini_agent.sh
# equivalent
uv run --frozen --no-sync python -m fast_float.livekit.gemini.agent start
```

It needs `[livekit]` and `[gemini]` configured (env, `secerts.toml`, or
`settings.toml`). See [`LIVEKIT_GEMINI_AGENT.md`](LIVEKIT_GEMINI_AGENT.md).

---

## 5. Startup, warmup and health

`/health` answers before the model is ready, which is what you want for
container probes:

```bash
curl -s localhost:8000/health | jq '{status, model_loaded, warming_up, warmup_stage, warmup_elapsed_s}'
```

```json
{
  "status": "healthy",
  "model_loaded": true,
  "warming_up": true,
  "warmup_stage": "decode",
  "warmup_elapsed_s": 12.4
}
```

- Weights plus ONNX/TRT sessions take roughly **25 s**; the background warmup
  costs ~60 s on a warm TensorRT engine cache, 2-5 min on a cold one.
- `POST /generate` **waits** for warmup (up to `WARMUP_WAIT_TIMEOUT_S`, default
  600 s) instead of returning 503. `WARMUP_TIMEOUT_S` (default 900) bounds each
  warmup phase.
- `SKIP_WARMUP=1` skips warmup; the first request then pays the compile cost.
- The decode auto-tuner caches its sweep in `~/.cache/fast_float/decode_tune.json`
  (override `FLOAT_DECODE_TUNE_CACHE`, disable with `0`).

GPU state and the escape hatch:

```bash
curl -s localhost:8000/gpu | jq
curl -s -X POST localhost:8000/gpu/clear | jq
```

---

## 6. Generate a video

### 6.1 Prepare the request

`audio_base64` is required; `image_base64` is optional (omit it to use the
built-in portrait). A WAV file can be sent as-is with `audio_format: "wav"`.

```bash
# Linux
AUDIO=$(base64 -w0 speech.wav)
IMAGE=$(base64 -w0 portrait.jpg)

# macOS
AUDIO=$(base64 -i speech.wav)
```

```powershell
# Windows PowerShell
$AUDIO = [Convert]::ToBase64String([IO.File]::ReadAllBytes("speech.wav"))
$IMAGE = [Convert]::ToBase64String([IO.File]::ReadAllBytes("portrait.jpg"))
```

### 6.2 Send it

`curl` with `jq` keeps the payload readable:

```bash
jq -n --arg audio "$AUDIO" --arg image "$IMAGE" '{
  audio_base64: $audio,
  audio_format: "wav",
  sample_rate: 16000,
  channels: 1,
  image_base64: $image,
  image_format: "jpg",
  emotion: "S2E",
  nfe: 10,
  seed: 25,
  quality: "balanced",
  client_request_id: "demo-001"
}' > request.json

curl -s -X POST localhost:8000/generate \
  -H 'Content-Type: application/json' \
  --data @request.json | jq
```

The same thing in Python (useful for long audio, where the POST blocks until
generation finishes):

```python
import base64, json, pathlib, urllib.request

payload = {
    "audio_base64": base64.b64encode(pathlib.Path("speech.wav").read_bytes()).decode(),
    "audio_format": "wav",
    "sample_rate": 16000,
    "channels": 1,
    "image_base64": base64.b64encode(pathlib.Path("portrait.jpg").read_bytes()).decode(),
    "image_format": "jpg",
    "emotion": "S2E",
    "nfe": 10,
    "seed": 25,
    "client_request_id": "demo-001",
}

req = urllib.request.Request(
    "http://localhost:8000/generate",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json", "X-Session-ID": "demo"},
)
with urllib.request.urlopen(req, timeout=1800) as resp:
    print(json.dumps(json.load(resp), indent=2))
```

Response (abridged):

```json
{
  "status": "success",
  "request_id": "demo-001",
  "download_url": "http://localhost:8000/download/demo-001",
  "stream_url": "http://localhost:8000/stream/demo-001",
  "file_size_mb": 1.8,
  "resolution": "512x512",
  "encoder": "h264_nvenc",
  "chunks": 1,
  "performance": {
    "preprocess_time": 0.4,
    "inference_time": 6.1,
    "total_time": 7.2,
    "fps": 42.0
  }
}
```

`X-Request-ID` / `X-Session-ID` are optional, but sending them stamps the same
correlation ids on every API log line for that request, and the worker reuses
them over IPC.

### 6.3 Poll progress while it runs

Set `client_request_id` and the id is known before the call returns (progress
is registered under it up front):

```bash
curl -s localhost:8000/progress/demo-001 | jq
```

```json
{
  "request_id": "demo-001",
  "status": "processing",
  "stage": "decode",
  "total_chunks": 3,
  "completed_chunks": 1,
  "percent": 33.3,
  "elapsed_s": 8.7,
  "audio_seconds": 9.4
}
```

`status` moves `queued` -> `processing` -> `merging` -> `done` (`error` on
failure). `404` means the id was never registered.

### 6.4 Fetch the result

```bash
curl -o out.mp4 localhost:8000/download/demo-001

# Play it while it is still being written (HTTP Range)
curl -s -H 'Range: bytes=0-1023' -D - -o /dev/null localhost:8000/stream/demo-001
```

---

## 7. Long audio: chunking

By default the whole file is one model call. For anything longer than ~10 s,
split it into windows: memory stays flat and the first frames appear much
sooner.

```json
{
  "audio_base64": "...",
  "audio_format": "wav",
  "chunk_seconds": 5,
  "chunk_overlap_seconds": 0.5,
  "chunk_batch_size": 2,
  "chunk_pad_seconds": 2.0,
  "decode_batch_size": 0
}
```

| Field | Default | Meaning |
| --- | --- | --- |
| `chunk_seconds` | `0` | Window length; `0` = no chunking |
| `chunk_overlap_seconds` | `0.5` | Shared audio between windows (seam continuity) |
| `chunk_batch_size` | `1` | Windows sampled in one batch (VRAM vs throughput) |
| `chunk_pad_seconds` | `2.0` | Silence padding for short windows |
| `decode_batch_size` | `0` | Frames per decode micro-batch; `0` = VRAM tuner |

Chunks are reproducible: `seed` is authoritative and chunk `k` uses
`seed + k`. Seams are cross-faded (`FLOAT_SEAM_BLEND`, default 3 frames).

`live: true` forces `chunk_batch_size=1` -- batched sampling cannot feed a
per-frame transport.

---

## 8. Live preview

There are two ways to watch frames before the MP4 exists, plus the LiveKit
transport for conversational rooms.

### 8.1 WebSocket (raw frames)

1. `POST /generate` with `"live": true` and a `client_request_id`.
2. Immediately open `WS /ws/stream/{client_request_id}`.

```python
import asyncio, json, websockets  # `websockets` ships with the project deps


async def preview(request_id: str) -> None:
    uri = f"ws://localhost:8000/ws/stream/{request_id}"
    async with websockets.connect(uri, max_size=None) as ws:
        async for message in ws:
            if isinstance(message, str):
                event = json.loads(message)
                print(event["type"], event.get("fps") or event.get("result") or event.get("message"))
                if event["type"] in {"done", "error"}:
                    return
                continue
            idx = int.from_bytes(message[0:4], "big")
            pts_ms = int.from_bytes(message[8:12], "big")
            pixels = message[12:]  # width * height * channels, uint8 RGB(A)
            print(f"frame {idx} @ {pts_ms} ms ({len(pixels)} bytes)")


asyncio.run(preview("demo-001"))
```

Wire format and close codes (4404 unknown/finished, 4408 no metadata) are in
[`STREAMING_PIPELINE.md`](STREAMING_PIPELINE.md) and the README. Frames are
lossy by design: a slow client skips frames rather than slowing the GPU.

### 8.2 WebRTC (`POST /offer`)

Add `"live": true` plus `"stream_id"` to attach an in-flight generation; omit
them to play a finished file:

```bash
curl -s -X POST localhost:8000/offer \
  -H 'Content-Type: application/json' \
  -d '{"sdp":"<offer-sdp>","type":"offer","live":true,"stream_id":"demo-001"}' | jq
```

The answer carries `sdp`, `type` and `peer_id`; trickle candidates go to
`POST /offer/candidate`. The browser must declare `recvonly` audio+video
transceivers before `createOffer`.

### 8.3 LiveKit (SFU + conversational avatar)

```bash
# Is the SFU configured? (and from which source)
curl -s localhost:8000/livekit/status | jq

# Mint a join token for the browser
curl -s -X POST localhost:8000/livekit/token \
  -H 'Content-Type: application/json' \
  -d '{"room":"studio","identity":"alice","name":"Alice","publish_mic":true}' | jq

# Publish an active live stream / finished MP4 as an avatar participant
curl -s -X POST localhost:8000/livekit/publish/demo-001 \
  -H 'Content-Type: application/json' -d '{"room":"studio"}' | jq
```

Start a conversational session (the built-in agent echoes the browser mic;
`"agent": "gemini"` dispatches the LiveKit Agents worker instead):

```bash
curl -s -X POST localhost:8000/livekit/conversation/start \
  -H 'Content-Type: application/json' \
  -d '{"room":"studio","image_path":"checkpoints/portrait.jpg","agent":"builtin"}' | jq
```

Then, per room:

```bash
curl -s 'localhost:8000/livekit/conversation/status?room=studio' | jq

# Push ~200 ms of mono int16 16 kHz PCM (mic), optionally flushing the tail
curl -s -X POST localhost:8000/livekit/conversation/audio \
  -H 'Content-Type: application/json' \
  -d '{"room":"studio","audio_base64":"<pcm>","sample_rate":16000,"flush":true}' | jq

# Barge-in: drop queued frames
curl -s -X POST localhost:8000/livekit/conversation/interrupt \
  -H 'Content-Type: application/json' -d '{"room":"studio"}' | jq

# Make the avatar say something (TTS -> render)
curl -s -X POST localhost:8000/livekit/conversation/speak \
  -H 'Content-Type: application/json' \
  -d '{"room":"studio","text":"Hello from FLOAT."}' | jq

curl -s -X POST localhost:8000/livekit/conversation/stop \
  -H 'Content-Type: application/json' -d '{"room":"studio"}' | jq
```

Docs: [`LIVEKIT_INTEGRATION.md`](LIVEKIT_INTEGRATION.md),
[`LIVEKIT_GENERATOR.md`](LIVEKIT_GENERATOR.md),
[`LIVEKIT_GEMINI_AGENT.md`](LIVEKIT_GEMINI_AGENT.md).

---

## 9. Docker

```bash
docker build -t float-fast .
docker run --gpus all -p 8080:8080 float-fast
```

The image runs `python fast_float/main.py`, reads `PORT` (default `8080`), and
raises hypercorn's startup timeout to 600 s. Common overrides:

```bash
docker run --gpus all -p 8080:8080 \
  -e NGPUS=1 \
  -e WARMUP_TIMEOUT_S=900 \
  -e FLOAT_LOG_LEVEL=INFO \
  -e LIVEKIT_URL=wss://your-sfu.example \
  float-fast
```

`.env` is **not** baked into the image. Mount config or pass secrets at run
time.

---

## 10. Logs

Each process writes its own rotating JSONL file plus a console line; the
supervisor (or shell `tee`) writes `logs/combined.log`.

```bash
tail -f logs/combined.log                 # one timeline, all processes
tail -f logs/api.log | jq -r '"\(.ts) \(.level) \(.logger) \(.msg)"'
jq -c 'select(.level == "ERROR")' logs/*.log
jq -c 'select(.req_id == "demo-001")' logs/*.log
```

Windows PowerShell:

```powershell
Get-Content logs\api.log -Wait
Get-Content logs\api.log | Select-Object -Last 50 | ConvertFrom-Json
```

Useful knobs (full table in [`LOGGING.md`](LOGGING.md)):

```bash
FLOAT_DEBUG=1 ./scripts/run_api.sh           # debug level + per-call timing
FLOAT_LOG_LEVEL=WARNING ./scripts/run_api.sh
FLOAT_LOG_DIR=/var/log/float ./scripts/run_api.sh
FLOAT_LOG_CONSOLE=0 ./scripts/run_api.sh     # file only
```

---

## 11. Per-component backends (optional)

```bash
# What the runtime would do on this config
uv run --frozen --no-sync python -m fast_float.runtimes inspect
uv run --frozen --no-sync python -m fast_float.runtimes inspect --json

# Force one component back to torch (debug)
FLOAT_MODE_fmt=eager ./scripts/run_api.sh

# Ignore the generated config entirely
FLOAT_DISABLE_BACKENDS=1 ./scripts/run_api.sh
```

Resolution order: `$FLOAT_BACKENDS_CONFIG` -> `onnx/backends.generated.yaml` ->
`config/backends.yaml` -> all `eager`. A component whose graph fails the
install-time parity self-test is dropped back to eager; the server still boots.
See [`BACKEND_SELECTION.md`](BACKEND_SELECTION.md) and
[`BUILD_RUNTIME_SESSIONS.md`](BUILD_RUNTIME_SESSIONS.md).

---

## 12. Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `/generate` returns 503 "warmup is still running" | warmup exceeded `WARMUP_WAIT_TIMEOUT_S` | wait, raise the timeout, check `warmup_stage` |
| `/generate` returns 429 | `MAX_CONCURRENT_REQUESTS` semaphore full | wait or raise the limit |
| No CUDA/TensorRT EP in the log | plain `uv run` re-synced the CPU `onnxruntime` | use `uv run --frozen --no-sync` / re-run `convert_to_onnx.sh` |
| `libnvinfer is not loadable` | TensorRT major does not match ORT 1.23 | install `tensorrt-cu12==10.9.0.34` |
| TensorRT "Set dimension ... does not satisfy any optimization profiles" | batch above the TRT profile | lower `chunk_batch_size`, raise `FLOAT_TRT_MAX_BATCH`, or let `fmt` split |
| First frame takes seconds | decode tuner re-sweeping | keep `FLOAT_DECODE_TUNE_CACHE` (default cache path) |
| Studio says "Disconnected" | proxy target mismatch | set `PORT` for both backend and `run_all.sh`, or `VITE_API_PROXY_TARGET` |
| WebSocket closes 4404 | request id unknown or already finished | subscribe before `/generate` returns; use `client_request_id` |
| WebSocket closes 4408 | stream produced no metadata | check `logs/api.log` for the stream error |
| OOM during decode | batch too large for free VRAM | set `decode_batch_size` (e.g. `4`), lower `chunk_batch_size` |
| Empty/odd output video | broken ONNX artifact passed through | compare with `FLOAT_MODE_<c>=eager`, re-run the selector |
| No logs in a file | file sink off / unwritable dir | check `FLOAT_LOG_TO_FILE`, `FLOAT_LOG_DIR` |

---

## 13. Where to read next

| Topic | Doc |
| --- | --- |
| Dual-process worker, ZeroMQ, shared memory | [`DUAL_PROCESS_IPC.md`](DUAL_PROCESS_IPC.md), [`run_dual_process.md`](run_dual_process.md) |
| Logging, correlation, redaction, rotation | [`LOGGING.md`](LOGGING.md) |
| Frame dispatcher, live fan-out, WS protocol | [`STREAMING_PIPELINE.md`](STREAMING_PIPELINE.md) |
| Batched decode and VRAM auto-sizing | [`BATCH_DECODING.md`](BATCH_DECODING.md) |
| ONNX/TensorRT selection and parity gates | [`BACKEND_SELECTION.md`](BACKEND_SELECTION.md) |
| Builder/runtime session layout | [`BUILD_RUNTIME_SESSIONS.md`](BUILD_RUNTIME_SESSIONS.md) |
| LiveKit transport, avatar worker | [`LIVEKIT_INTEGRATION.md`](LIVEKIT_INTEGRATION.md), [`LIVEKIT_GENERATOR.md`](LIVEKIT_GENERATOR.md) |
| Gemini Live voice agent | [`LIVEKIT_GEMINI_AGENT.md`](LIVEKIT_GEMINI_AGENT.md) |
| Idle/silence avatar loop | [`silence_mechanism.md`](silence_mechanism.md) |
| Studio UX specification | [`UI_UX_BRIEF.md`](UI_UX_BRIEF.md) |
| Architecture and agent rules | [`../AGENT.md`](../AGENT.md) |
