# FLOAT Fast — Agent Development Guide (`AGENT.md`)

This document provides system architecture, coding patterns, directory layouts, and execution workflows for AI coding agents and developers working on this repository.

---

## 1. Project Overview

**FLOAT (Flow Matching for Audio-Driven Talking Portrait Video Generation - ICCV 2025)** is a deep learning system that produces temporally consistent, audio-driven talking portraits using learned orthogonal motion latent flow matching.

This repository (**`float-fast`**) provides:

1. **High-Performance FastAPI Server**: Hardware-accelerated inference with NVENC video encoding, HTTP byte-range streaming, and WebRTC signaling.
2. **Live Preview Transports**: frames tapped straight off the decoder and pushed to browsers while the request is still generating — raw frames over WebSocket, or an aiortc video track over WebRTC (`live: true`).
3. **Gradio Interfaces**: Batch generator (`gradio_interface.py`) and live audio streaming (`float_live_app.py`).
4. **Modern Python Architecture**: Modular subpackages managed by `uv` with pinned PyTorch CUDA 12.6 wheels.

---

## 2. Directory Structure

```
float-fast/
├── AGENT.md                        # This development guide
├── .pre-commit-config.yaml         # ruff/format/gitleaks/commitizen hooks
├── pyproject.toml                  # Project metadata, deps, uv indexes, ruff config
├── uv.lock                         # Pinned dependency lockfile (do not hand-edit)
├── .python-version                 # Pinned interpreter (3.10)
├── Dockerfile                      # Container build definition (CUDA 12.1 base)
├── config/
│   ├── api.json                    # Endpoint and public URL configurations
│   └── backends.yaml               # Committed all-eager backend default
├── docs/                           # BACKEND_SELECTION, BATCH_DECODING,
│                                   # STREAMING_PIPELINE, UI_UX_BRIEF
├── scripts/
│   ├── setup.sh                    # System deps, env, checkpoints
│   ├── environments.sh             # uv sync + CUDA/NVIDIA lib paths
│   ├── setup_cuda_libs.sh          # Export nvidia site-packages lib dirs
│   ├── convert_to_onnx.sh          # Export all ONNX units + run selector
│   ├── convert_fmt_to_onnx.sh      # Per-unit converters
│   ├── convert_wav2vec_to_onnx.sh
│   ├── download_checkpoints.sh     # Fetch model weights
│   ├── run_all.sh                  # Backend + frontend together
│   ├── run_api.sh                  # Launch FastAPI server via hypercorn
│   ├── run_frontend.sh             # Launch Vite dev server
│   ├── run_gradio.sh               # Launch Gradio batch web app
│   └── run_live.sh                 # Launch Gradio live audio streaming app
├── checkpoints/                    # Model weights directory
│   ├── float.pth
│   ├── wav2vec2-base-960h/
│   └── wav2vec-english-speech-emotion-recognition/
├── fast_float/                     # Main Python application package
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
    │   ├── streaming.py            # WS /ws/stream/{request_id} (live raw frames)
    │   ├── livekit.py              # /livekit/token, /status, conversation start/stop
    │   └── webrtc.py               # POST /offer (WebRTC SDP signaling & media player)
    ├── streaming/                  # Per-request frame tap feeding live transports
    │   ├── types.py                # StreamMeta (fps/size/channels/first_idx/pts_ms)
    │   ├── dispatcher.py           # FrameDispatcher + FrameQueue (live | lossless)
    │   └── live.py                 # LiveStream fan-out + LiveStreamDispatcher
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
    ├── onnx/                       # ONNX export + per-component backend selection
    │   ├── backends.py             # config schema, resolve_mode, maybe_compile
    │   ├── dispatch.py             # apply_backends (per-component routing)
    │   ├── export.py               # export_module, make_ort_session, IOBinding
    │   ├── parity.py               # parity / benchmark helpers
    │   ├── runtime.py              # emotion / wav2vec / fmt ORT backends
    │   ├── select_backends.py      # measure-and-pick selector -> YAML
    │   └── convert_*.py            # per-unit ONNX converters
    ├── interfaces/                 # Web interfaces
    │   ├── gradio_interface.py     # Batch UI for image + audio upload
    │   ├── float_live_app.py       # Live mic streaming UI
    │   └── test_gradio.py          # Testing UI with WebRTC player integration
    ├── livekit/                    # LiveKit transport + Gemini Live worker
    │   ├── settings.py             # LiveKit/Gemini config (shared files first)
    │   ├── tokens.py               # room JWT grants
    │   ├── generator.py            # audio -> rtc.VideoFrame renderer
    │   ├── worker.py               # avatar worker (in-process agent)
    │   ├── receiver.py             # DataStream TTS -> PCM segments
    │   ├── agent_runner.py         # avatar runner for the Gemini worker
    │   └── gemini/                 # livekit-agents worker (separate process)
    │       ├── agent.py            # entrypoint + voice tools
    │       ├── config.py           # AgentConfig over LiveKitSettings
    │       └── tools.py            # FloatApi HTTP client
    └── options/                    # Backward-compatibility shims for legacy imports
        ├── __init__.py
        └── base_options.py
├── tests/                          # pytest suite (dispatcher, live stream, backends)
└── frontend/                       # React 19 + Vite studio UI
    └── src/                        # App.jsx, StudioStage.jsx, services/api.js, ...
        ├── services/live.js        # WS protocol client
        └── components/LivePreview.jsx  # WS canvas / WebRTC <video> live preview
```

---

## 3. Environment & Dependencies

### Package Manager (uv — mandatory)

We use **`uv`** exclusively. The Python version is pinned by `.python-version`
(`3.10`); all dependencies live in `pyproject.toml` + `uv.lock`.

```bash
# Canonical install: honor the lockfile exactly (never re-resolves)
uv sync --frozen

# Run anything in the project environment (scripts, python, pre-commit, ...)
uv run <command>          # e.g. uv run python fast_float/interfaces/gradio_interface.py
uv run --frozen <command> # when in doubt, force lockfile-honoring

# Change dependencies (updates BOTH pyproject.toml and uv.lock)
uv add <pkg>              # runtime dep
uv add --dev <pkg>        # dev-only dep
uv lock                   # re-lock (only when you intend to change the lock)
```

**Rules:**

- **Never** use bare `pip`/`pip install` inside the project env. Use `uv add`
  (or `uv sync`) so `pyproject.toml` and `uv.lock` stay in sync.
- **Never hand-edit `uv.lock`** — it is generated; let uv own it.
- Prefer `uv run …` over activating `.venv` manually. In CI/scripts use
  `uv run --frozen` so a stale lock can never silently re-resolve.
- The PyTorch stack is pinned to explicit **CUDA 12.6** wheels via
  `[[tool.uv.index]]` + `[tool.uv.sources]` (see below). Do not add torch/
  torchvision/torchaudio without their CUDA source mapping.

### PyTorch Wheels

PyTorch CUDA 12.6 wheels are pinned in `pyproject.toml`:

```toml
[[tool.uv.index]]
name = "pytorch-cu126"
url = "https://download.pytorch.org/whl/cu126"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu126" }
torchvision = { index = "pytorch-cu126" }
torchaudio = { index = "pytorch-cu126" }
```

Two constraints decide the version and the CUDA major:

- **`torch >= 2.5.1`.** `fast_float/onnx/export.py` uses the dynamo exporter
  (`torch.export.export` + `torch.onnx.export(ExportedProgram, …)` with
  `external_data=`), and that call signature only exists from 2.5.1 on. The
  classic `torch.onnx.export` fallback is deliberately retired (commit
  `397cb2e`). On 2.0.1 every converter dies with
  `torch.export.export is required (torch>=2.1)`.
- **CUDA 12 (cu126), not cu118.** `onnxruntime-gpu` 1.23 dlopens CUDA **12**
  sonames (`libcublas.so.12`, `libcudnn.so.9`, `libcufft.so.11`, …). A cu126
  torch installs `nvidia-*-cu12` wheels that provide exactly those, so torch
  and ORT share one CUDA major. A cu118 torch instead drags in
  `nvidia-*-cu11` wheels whose `libcudnn.so.9` / `libcurand.so.10` /
  `libcusolver.so.11` / `libcusparse.so.12` **collide** with the cu12 files
  (same filenames, same `nvidia/<pkg>/lib` dir), so whichever installs last
  wins and one of the two stacks breaks.

### LiveKit / Gemini version window

The LiveKit transport and the Gemini Live worker share this environment, which
makes a few pins load-bearing. Do not bump them casually:

- **`numpy==1.26.4`** — LiveKit's floor (`>=1.26`) while staying on numpy 1.x
  (`onnx<1.17`, opencv 4.8 wheels).
- **`gradio==6.10.0`** — the only window that allows `aiofiles>=24` (LiveKit),
  `pydantic>=2.12.5` + `websockets>=13` (google-genai) and `starlette 0.50`
  (FastAPI 0.122). Gradio 5.x pins `pydantic<2.12`; Gradio ≥6.20 needs
  `starlette≥1.0.1`.
- **`uvicorn==0.54.0`** — 0.24 imported `websockets.legacy`, removed in 14+.

The Gemini worker lives in `fast_float/livekit/gemini/` and runs as its own
process (`scripts/run_gemini_agent.sh`), sharing the venv, lockfile and config
with the API. See `docs/LIVEKIT_GEMINI_AGENT.md`.

The ONNX **export** layer is still installed on top of the lock by
`scripts/_onnx_env.sh` (onnx 1.18 + onnxruntime-gpu + onnxscript 0.7.2); a
plain `uv sync` reverts it, so re-run the script after syncing if you export
models.

---

## 4. Development Workflow (git, pre-commit, commits)

### Branch policy

- **Never commit or push to `main`.** Feature work happens on a branch
  (`feat/onnx-supporting` is the current one).
- Commit and push after each coherent change; keep commits small and scoped.

### Pre-commit (required)

Hooks are defined in `.pre-commit-config.yaml` and install **both**
`pre-commit` and `commit-msg` stages (see `default_install_hook_types`).

```bash
# One-time per clone (already installed in this repo)
uv run pre-commit install

# Run everything
uv run pre-commit run --all-files

# Run on specific paths (preferred — fast)
uv run pre-commit run --files fast_float/onnx/runtime.py

# Run a single hook
uv run pre-commit run ruff-format --files fast_float/onnx/export.py
```

Hooks that run: `ruff` (`--fix`), `ruff-format`, gitleaks, validate-pyproject,
shebang/permission checks, case + Windows-name + symlink checks, merge-conflict +
submodule + large-file (max 6 MB) guards, `end-of-file-fixer`,
`trailing-whitespace`, `check-yaml`/`check-json`, and `commitizen` (commit-msg).

**Rules:**

- **Never** use `git commit --no-verify` / `--no-gpg-sign` to bypass hooks. Fix
  the cause instead.
- **Stage explicit paths**, never `git add -A` / `git add .` — the working tree
  frequently holds concurrent WIP, and a blanket add has swept unrelated files
  into commits. Confirm with `git status` and `git diff --cached --stat` first.
- **Auto-fixing hooks are expected to fail once.** `ruff --fix`, `ruff-format`,
  `end-of-file-fixer` and `trailing-whitespace` rewrite the files, abort the
  commit, and leave the fixes in place. Just `git add` the touched files and
  commit again; the second run passes.
- On Windows the pre-commit stash can report *"Stashed changes conflicted with
  hook auto-fixes … Rolling back fixes"*. Nothing is lost — re-run the commit.
- Keep **LF in the index**; the `LF will be replaced by CRLF` warnings on
  Windows are benign.

### Commit messages (conventional commits)

Validated by commitizen with `cz-conventional-gitmoji`. Use:

```
type(scope): imperative summary
```

`type` ∈ `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`,
`ci`, `chore`, `revert`. Use a short lowercase `scope` (module name, e.g.
`onnx`, `pipeline`, `fmt`, `api`). Summary is imperative, lowercase, no trailing
period; wrap the body at ~72 columns and explain **why**, not **what**.

Examples from this repo: `feat(onnx): runtime per-component backend dispatch`,
`fix(pipeline): bugs + perf drains across the served path`.

---

## 5. Quality Gates & Verification

Run these before every commit; CI and the pre-commit hooks enforce most of them.

```bash
# Lint + format (ruff config lives in pyproject.toml; lint rules E/F/B/I/UP/RUF/SIM/...)
uv run ruff check .
uv run ruff format --check .

# Fast syntax gate when the full environment is unavailable
uv run python -m py_compile <changed files>

# Types (config present; not yet enforced in hooks)
uv run mypy fast_float

# Tests (pytest config: testpaths = ["tests"]; create tests/ as needed)
uv run pytest
pre-commit run -all-files
```

**Environment reality (do not forget):**

- The full inference stack is **GPU-box only**. ONNX export/parity/bench, ORT
  sessions, CUDA IOBinding and TensorRT cannot be exercised on a CPU-only dev
  box — only `ruff`, `py_compile`, and `select_backends --dry-run` run locally.
- Validate the local, hardware-independent parts here; flag anything that needs
  the GPU box in your commit/PR notes.

---

## 6. How to Run

### 0. Launch Backend + Frontend Together (Recommended for Development)

```bash
# Single command, Ctrl+C stops both. Env overrides: PORT, FRONTEND_PORT,
# VITE_API_URL, CORS_ORIGINS
./scripts/run_all.sh
# Backend → http://localhost:8000 | Frontend → http://localhost:3000
```

The frontend needs **Node 20.19+ / 22.12+** (Vite 8's rolldown imports
`node:util.styleText`); `scripts/setup.sh` installs Node 22 from NodeSource when
the system Node is older, and `run_all.sh` refuses to start on a too-old Node
instead of dying inside Vite.

The backend is launched with `uv run --frozen --no-sync` (see §6): a plain
`uv run` re-syncs the venv back to the lock, which pins the *portable* CPU
`onnxruntime`, and the server then boots with no CUDA/TensorRT EP.

**Startup and boot warmup.** `/health` answers after ~25 s (weights + ONNX/TRT
sessions); the background warmup then costs ~60 s on a warm engine cache
(~2–5 min on a cold TensorRT one) and is reported as
`warming_up` / `warmup_stage` / `warmup_elapsed_s`. Two behaviours matter:

- `/generate` **waits** for warmup (up to `WARMUP_WAIT_TIMEOUT_S`, default 600)
  instead of returning 503 immediately, so the first request after a restart is
  served rather than rejected. `WARMUP_TIMEOUT_S` (default 900) bounds each
  warmup phase so a wedged inference cannot leave `warming_up` set forever.
- The decode auto-tuner's sweep (~30 s of GPU benchmarking) is cached in
  `~/.cache/fast_float/decode_tune.json` (override with
  `FLOAT_DECODE_TUNE_CACHE=/path`, disable with `=0`), keyed by GPU, CUDA/torch
  version, frame shape and tuner settings. It is re-measured when the key
  changes or when VRAM is now too tight; a run where nothing fits is never
  cached. Without the cache a transient low-VRAM moment could pin the server to
  batch 1 (0.5 fps instead of ~180). `SKIP_WARMUP=1` skips the warmup entirely
  (the first real request pays the compile cost).

### 1. Launch FastAPI Server

```bash
# Direct run with uv and hypercorn. `--no-sync` keeps the ONNX export stack /
# onnxruntime-gpu that setup.sh + convert_to_onnx.sh layered over the lock.
uv run --frozen --no-sync hypercorn fast_float.main:app --bind 0.0.0.0:8000 --log-level info

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

### 5. Convert to ONNX (on the target box)

```bash
# One command: weights -> deps -> TensorRT probe -> all six units -> selector.
# The selector writes onnx/backends.generated.yaml (read at runtime).
./scripts/convert_to_onnx.sh

# Per unit (same converter, one artifact):
./scripts/convert_fmt_to_onnx.sh          # -> onnx/fmt_step_v1.onnx
./scripts/convert_wav2vec_to_onnx.sh      # -> onnx/wav2vec_v1.onnx

# Env overrides: OUT_DIR, FMT_OUT, W2V_OUT, DEC_OUT, REF_OUT, EMO_OUT, CKPT,
# EMO_DIR, W2V_DIR, DEC_BATCH, DEC_BATCHES, W2V_SECONDS,
# ORT_ARGS (e.g. "--no-trt" / "--trt-fp32" / "--trt-max-batch 16"), SKIP_SELECT=1
```

In order: download the weights (float.pth via gdown, the two HF models),
verify/repair the pinned ONNX stack, probe the TensorRT EP (installing
TensorRT 10.9 if it is missing), run the six converters **independently**
(one failure never stops the others), then run the per-component selector.
Exit code 1 means at least one unit failed; the selector is skipped then and
the runtime keeps the committed all-eager config. TensorRT engines are cached
under `onnx/trt_cache/` — that cache is tied to the GPU and the TRT/CUDA
versions, so delete it after either changes.

**Every unit is gated before it is shipped.** A converter exports, then
compares the artifact against the torch model and refuses to pass a graph that
does not reproduce it (§8.6): a unit that cannot prove its graph fails, it
does not ship a "probably fine" artifact. The two rules that decide whether a
gate means anything:

- compare against the **deployed** model, never against the module that was
  traced (that blindness once shipped an FMT graph that was a different
  model — see §8.6);
- compare **the same computation on both sides**: the exporter pins both to
  fp32, because TF32 alone moves a deep conv stack by tens of uint8 LSB.

Two CLI renames bite in the weights step and are handled by the shared
helpers in `scripts/_onnx_env.sh` (`gdown_download`, `hf_download`): gdown 6
dropped the `--id` flag (the file id is positional now) and huggingface_hub
1.0 renamed the CLI from `huggingface-cli` to `hf`. The helpers probe both
spellings and fall back to the Python API, so a fresh env cannot abort the
run.

---

## 7. API Reference

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/` | Service health status and links |
| `POST` | `/generate` | Generate talking portrait video from audio + image (waits for boot warmup, up to `WARMUP_WAIT_TIMEOUT_S`) |
| `GET` | `/download/{request_id}` | Download generated MP4 video |
| `GET` | `/stream/{request_id}` | Stream generated MP4 with HTTP `Range` header support |
| `GET` | `/health` | System health, model status, FPS stats, and warmup progress (`warming_up`, `warmup_stage`, `warmup_elapsed_s`) |
| `GET` | `/gpu` | Real-time GPU VRAM allocated and reserved in GB |
| `POST` | `/gpu/clear` | Clear CUDA cache and run garbage collection |
| `POST` | `/offer` | WebRTC SDP negotiation for real-time video playback (add `"live": true` + `"stream_id"` to attach an in-flight generation's live track) |
| `WS` | `/ws/stream/{request_id}` | Live raw RGB(A) frames while `/generate` is running (see §8.9) |

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

## 8. Architecture & Concurrency Rules for Agents

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

### 6. Per-component backends (eager / compile / onnx)

- Each inference component (`emotion`, `audio`, `direction`, `fmt`, `decoder`,
  `refenc`) runs in a configured mode; **`eager` is the reference and the safe
  fallback**. The mode is read from YAML, never hard-coded.
- **Config is a build artifact, not source.** Resolution order: explicit path →
  `$FLOAT_BACKENDS_CONFIG` → `onnx/backends.generated.yaml` (gitignored, made by
  the selector) → `config/backends.yaml` (committed all-`eager`) → built-in
  all-`eager`. Never hand-edit the generated file; regenerate it on the target box.
- Runtime wiring lives in `fast_float/onnx/backends.py` (`resolve_mode`,
  `maybe_compile`) and `fast_float/onnx/dispatch.py` (`apply_backends`, called by
  `OptimizedInferenceAgent` right after the checkpoint loads, before the global
  optimisation pass). `FLOAT_DISABLE_BACKENDS=1` disables the whole pass;
  `FLOAT_MODE_<COMPONENT>` force-overrides one component (debug).
- **Parity is validated by the converter/selector scripts** (`python -m fast_float.onnx.select_backends`),
  **plus a one-shot self-test when a component is installed** at runtime: the
  graph is only as good as the execution provider that loads it, and the
  provider is chosen at runtime, not at export time. `apply_backends` probes
  the components it substitutes (`direction`, `refenc`, `decoder`) against the
  torch model, and `fmt` — which wires itself in its constructor, before the
  checkpoint is loaded — in the same pass, once the weights are there. A graph
  that does not reproduce the torch model is dropped and the component stays
  `eager`. `FLOAT_SKIP_BACKEND_SELFTEST=1` skips the probes. ORT providers
  (`trt → cuda → cpu`) fall back *inside* session creation.
- **The `decoder` graph is static-batch, so it ships as a set.** Each artifact
  is valid for exactly one `B`.
  `convert_decoder.py --batches 1,2,4,8,16` writes the
  primary `--batch` (default 8, matching the selector's `--dec-batch`) to `--out`
  and the rest to `<stem>_b<N>.onnx` siblings. `DecoderOnnx`
  (`runtimes/onnx/runner.py`) discovers the siblings, loads them **lazily**, and
  per call picks the smallest session covering the micro-batch (replicate-pad,
  trim). `_decoder_static_batch()` returns the *largest* available batch, so the
  decode tuner can size above the primary instead of being capped at it.
- **The decoder graph is the model itself — do not substitute grouped convs.**
  The deployed `ModulatedConv2d` is a *dense* modulated convolution (the style
  is folded into the input channels; no `groups`). An older export substituted
  a grouped variant (`groups=batch` baked into the ONNX attribute) to make the
  batch static, and the ORT CUDA EP can only serve grouped convs through its
  **fallback** kernel: measured against torch it drifts ~90 uint8 LSB (fp32 on
  both sides!) and is not even run-to-run reproducible, so the artifact could
  not reproduce the frames the eager path produces. The export now traces the
  model as-is (still at a fixed batch, so the artifact set and the runtime's
  pad/trim are unchanged) and the parity gate measures 1 LSB.
- **A parity check must compare the same computation on both sides.** TF32
  rounds every accumulation to a 10-bit mantissa; through the decoder's 22-conv
  stack that alone moves the output by ~40 uint8 LSB — far more than the
  2 LSB the frame gate allows — so comparing a TF32 ORT session against a TF32
  torch model measures the *kernels*, not the graph. The converter therefore
  pins **both** sides to fp32 (`make_session_from_args(..., use_tf32=False)` and
  `cudnn.allow_tf32=False` inside `_deterministic_cudnn()`), which makes the
  residual the graph's own error again (1 LSB). The runtime install self-test
  cannot pin the provider (the deployed session is TF32 for speed), so it
  judges the *provider* by the mean (2 LSB — the provider's TF32 path alone
  moves this stack by ~0.5 LSB mean) plus a cosine floor
  (0.999) and reports the max as information; a structurally wrong graph
  (max_abs 4.375, cosine 0.983) still fails both floors.
- **An exported graph must be the model the application runs — and the parity
  gate has to prove it.** `fmt`'s artifact decides the *motion latents*, so a
  wrong graph does not degrade the picture a little, it produces different
  motion. `FMT.Attention` has two implementations of the same math: the fused
  SDPA branch (`use_fused_attn()` is True, so this is what production runs) and
  the manual branch. They disagreed about the **polarity of the alignment
  mask**: `enc_dec_mask` marks the frames *outside* the local ±`attention_window`
  band, SDPA's `attn_mask=~mask` therefore keeps the band, but the manual
  branch's `masked_fill(~mask, -inf)` masked the band out and attended to
  *everything else* — the exact inverse. The converter exported the manual
  branch (`fused_attn = False`) while the app ran the fused one, so
  `fmt_step_v1.onnx` was a **different model**: per-step cosine 0.94, and the
  final latent after a full CFG ODE scored cosine 0.81 against the torch path
  (garbage frames). The converter's parity gate could not see it, because it
  compared the graph against *the very module it had just traced*. Fixed by
  (a) making both branches use the same polarity, (b) exporting the deployed
  path instead of an export-only substitute, and (c) gating the artifact
  against a **snapshot of the deployed model**, never against the traced
  module. The same blindness applies to the selector: its eager reference must
  be the deployed model, or a bad artifact passes selection and gets shipped.
  `tests/test_fmt_attention_export.py` pins all of it.
- **A constructor-wired backend still needs an install-time self-test.**
  `emotion`/`audio`/`fmt` install their ONNX backend in their own constructor,
  i.e. *before* the checkpoint is loaded, so `apply_backends` could not judge
  them (the decoder/direction/refenc self-tests run there). A stale `fmt`
  artifact therefore shipped broken frames with no complaint. `fmt` is now
  verified in `apply_backends` — one CFG-shaped step, torch vs the graph,
  gated functionally (`cosine >= FP16_MIN_COSINE`, `mean_abs <=
  FP16_MAX_MEAN`, the vector field is small-magnitude so the drift is
  absolute) — and dropped (component stays eager) when it does not reproduce
  the torch model. The wrong artifact scores cosine 0.958 and is rejected.
- **A converter must not wire a runtime backend into the model it exports.**
  `FlowMatchingTransformer` self-wires an ONNX backend in its constructor when
  the config says `onnx` and the artifact exists — so the converter loaded
  (and built a TensorRT engine for) the very artifact it was about to replace,
  and the model could no longer be deep-copied for the parity snapshot. The
  converter's builder passes `fmt_onnx_path=None` for that reason.
- **A batch-shared scalar input must stay static.** `fmt` takes one scalar time
  value `t` (`torch.tensor([t_val])`, shape `[1]`, broadcast over the batch
  *inside* the graph) — it is not a batch-first tensor. `export_module` used to
  duplicate every dim-0==1 sample to batch 2 and then mark dim 0 dynamic, so `t`
  became `[s0]` and shared the batch dim with the rank-3 inputs. TensorRT's ONNX
  parser cannot tie a rank-1 tensor's `s0` to the rank-3 inputs' `s0`: it froze
  `t` at the profile's *opt* batch, so the plain call worked and every CFG call
  (batch 3B) died with "Dimensions with name s0 must be equal. Condition '=='
  violated: 1 != 3". Converters now pass `shared_inputs=(...)` to
  `export_module`, which skips those inputs in both `_ensure_batch2` and
  `_dynamo_mark`; the exported `t` stays static `[1]` and TRT builds one engine
  for the whole batch range. (Renaming only the `dim_param` is not enough —
  ORT's TRT engine cache key ignores dim names and reuses the stale engine; the
  static-shape change alters the graph, which does bust it.)
- **A TRT profile is a hard ceiling; batched CFG can exceed it.** `fmt`'s
  profile is `[1..16]` in batch (`FLOAT_TRT_MAX_BATCH`, default 16), but
  batched chunk sampling sends `chunk_batch_size` samples through CFG, i.e.
  `3 * chunk_batch_size` (24 for 8 chunks). TensorRT then fails every call with
  "Set dimension [24,50,512] for tensor x does not satisfy any optimization
  profiles", and the old handler disabled the ONNX backend for the rest of the
  process — one dispatch turned an entire 15 s render into eager (174 s).
  `FMTStepBackend.run_step` now splits any batch above the profile cap and
  concatenates the results (identical math: no cross-batch ops), and a single
  failure no longer disables the backend (3 consecutive failures do). Raise
  `FLOAT_TRT_MAX_BATCH` (rebuilds the engine) to avoid the split entirely.
- **cuDNN autotuning is per thread, and the decode pipeline spawns threads per
  request.** With `torch.backends.cudnn.benchmark = True` every fresh decode
  worker re-searched algorithms for ~40 conv shapes, costing ~1.25 s on the
  first decode of *every* request (measured: 2849 ms first call per new thread
  vs 50 ms with benchmark off). `StreamingVideoSaver` now sets
  `benchmark = False` (heuristics are ~10% slower per call; set
  `FLOAT_CUDNN_BENCHMARK=1` for long renders). Result: 13-frame requests went
  1.8 s → 0.6 s and batched 15 s/35 s renders 151–184 s → 5.6/12.6 s.
- **Never mutate a module inside a compiled `forward`.** `ToFlow` built its
  coordinate grid lazily and stored it with `register_buffer` mid-forward, so
  Dynamo's `hasattr(module, '_cached_grid')` guard failed after the first call
  and forced a full decoder recompile (the bulk of boot time). The grid is now
  built functionally (`_identity_grid`), which is cheap and trace-stable; boot
  warmup dropped 58 s → 24 s with no `recompile_limit` warnings.
- **ORT IOBinding must hand-shake with the torch stream.** `run_ort_iobinding`
  binds raw `data_ptr()`s, and ORT runs on its own CUDA stream while PyTorch's
  streams are non-blocking. Without an explicit sync ORT reads tensors torch has
  only *enqueued* (the ODE's Euler update is produced immediately before every
  step) and torch reads outputs before ORT writes them. Symptom: identical
  requests produced visibly different videos, sometimes with NaN, and the
  effect was invisible when calling a backend in a tight loop with a fixed
  input. Both sides now `current_stream(device).synchronize()`. Verified: the
  ODE became bit-identical across runs, and 3 API requests produced identical
  frames.
- **Every 2 s chunk is re-synthesised from fresh noise, so the seam is a real
  jump.** `sample()` runs one ODE per `wav2vec_sec*fps = 50` frames; chunk
  `k>0` starts from new noise and only sees `prev_x`/`prev_wa` as conditioning,
  so its first frames can land in a different pose. Measured on a 15 s clip:
  latent seam delta 3-5x a normal step (4.8-6.6 vs 1.3 median) and a visible
  jolt every 50 frames (video frame-delta `8.4` vs `1.3` typical). `sample`
  now blends the first `FLOAT_SEAM_BLEND` frames (default 3, 0 disables) toward
  a linear extrapolation of the previous chunk's last two frames: seams drop to
  1.5-3.3, in line with normal motion. This is a cosmetic continuation of the
  motion latent, not a change to the model; generation stays bit-reproducible.
- **`seed` is authoritative for generation.** The request seed used to be
  ignored unless `--fix_noise_seed` was set, so every run drew fresh noise.
  `sample()` seeds the initial latent from the request seed, and chunked
  requests pass `seed + chunk_index` per chunk (`_chunk_seed`) with one
  generator per batch row, so a chunk's noise does not depend on how many
  chunks share the batch. Same request → byte-identical frames; changing
  `chunk_batch_size` still shifts results by ~1/255 because TensorRT builds
  different engines/profiles per batch size and padded audio changes the tail
  conditioning — that is numerical, not random.
- **TensorRT must major-match ORT.** `onnxruntime-gpu` 1.23 is built against
  **TensorRT 10.9** and dlopens `libnvinfer.so.10` / `libnvonnxparser.so.10`. A
  TRT 11 install (`tensorrt_cu13*` → `libnvinfer.so.11`) can never satisfy that
  link, so the provider library fails to load and ORT silently drops the whole
  session to CUDA/CPU (the `[ort] NOTE: … libnvinfer is not loadable …` line). On
  a CUDA-12 box install the matching wheels — remove TRT 11 first if present:

  ```bash
  pip uninstall -y tensorrt tensorrt_cu13 tensorrt_cu13_bindings tensorrt_cu13_libs
  pip install "tensorrt-cu12==10.9.0.34"
  ```

  Bump the TRT pin in lockstep with the `onnxruntime` pin. Delete
  `onnx/trt_cache/` after any TRT/ORT change — engine caches are tied to the exact
  TRT version and GPU.
- **A correct TensorRT install still looks broken on a bad loader path.** Two
  traps, both handled by `scripts/install_tensorrt.sh`:
  `onnxruntime-gpu` 1.23's CUDA EP dlopens the **CUDA 12** sonames
  (`libcublas.so.12`, `libcudnn.so.9`, `libcufft.so.11`, …). A cu126 torch
  provides them through its `nvidia-*-cu12` deps, but any box whose torch is
  built for another CUDA major (cu118/cu13) does not — the installer adds the
  wheels when the CUDA EP is listed but inactive. And once
  `LD_LIBRARY_PATH` is exported for that, the loader stops honouring the
  interpreter's `DT_RPATH`, so `/usr/lib64-nvidia` (the driver's `libcuda.so.1`)
  must be on it too or every CUDA call fails with error 35 ("CUDA driver version
  is insufficient for CUDA runtime version"). `fast_float/initialize_env.py` and
  `scripts/setup_cuda_libs.sh` keep the same dir list for the runtime — and
  `scripts/convert_to_onnx.sh` must call `augment_ort_libs` **before** it probes
  TensorRT, or the probe sees an un-augmented loader path and reports a
  false "TensorRT EP not available" even when TRT works.
- **A truncated TensorRT install passes every probe but cannot build engines.**
  `libnvinfer_builder_resource.so.10.9.0` (~2 GiB, shipped by
  `tensorrt-cu12-libs` next to `libnvinfer.so.10`) is dlopened *lazily* — only
  when a non-trivial network is actually built. An install that lost that file,
  typically because the disk filled during extraction, therefore still passes
  the Identity-model probe, and every real unit later dies mid-conversion with
  `createInferBuilder: Error Code 6: API Usage Error (Unable to load library:
  libnvinfer_builder_resource.so.10.9.0)`. Both the installer and the `setup.sh`
  probe now verify the file is physically present (`has_builder_resource`, a
  glob for `tensorrt*/libnvinfer_builder_resource.so*`), and the installer now
  *refuses to evict* the existing tree when under ~7 GiB is free (the wheel is
  2.7 GiB compressed, ~4.4 GiB extracted) instead of half-installing it.
  Recovery: free space (start with `uv cache clean`), then re-run
  `scripts/install_tensorrt.sh`. TRT engine builds also need ~2 GiB in
  `onnx/trt_cache/`, so keep headroom on the box.
- **`setup.sh` must not prune the layered GPU stack.** `uv sync` removes packages
  the lock does not contain, and neither `onnxruntime-gpu` nor
  `tensorrt-cu12` is in the lock. Combined with the old
  `uv venv .venv --clear`, every `setup.sh` run rebuilt the entire environment
  (~7 GiB torch re-download) and then re-extracted ~4.4 GiB of TensorRT — which
  on a 50 GiB disk ends in ENOSPC, a half-installed `tensorrt_libs/` (only the
  `_win` builder resource survives) and units that quietly fall back to CUDA.
  `setup.sh` now reuses an existing venv and syncs with `uv sync --inexact`.
- **A plain `uv sync` silently disables the GPU execution providers.**
  `pyproject.toml` declares the *CPU* `onnxruntime` as the portable default and
  the CUDA box swaps in `onnxruntime-gpu`. Both distributions ship the **same**
  `onnxruntime` package files, so whichever was installed last owns them: after
  a `uv sync` (or any install of the CPU wheel) `get_available_providers()`
  drops to `['AzureExecutionProvider', 'CPUExecutionProvider']` and every ONNX
  unit falls back to CPU. `scripts/convert_to_onnx.sh` detects this and hands
  the files back to the GPU wheel (`install_onnx_stack`) — it has to do that
  *after* installing `onnxscript`, which itself depends on the CPU wheel and
  would otherwise reintroduce it.
- **`uv run` re-syncs the venv and kills the GPU EPs.** Same trap, runtime side:
  the lock (`uv.lock`) pins the portable CPU `onnxruntime`, while the export
  stack (`onnx` 1.17/1.18 + `onnxruntime-gpu` + TensorRT) is layered on top by
  `setup.sh` / `convert_to_onnx.sh`. Every launcher therefore runs
  `uv run --frozen **--no-sync**`; a plain `uv run --frozen` re-syncs the env to
  the lock and the backend boots with
  `Available Execution Providers: ['AzureExecutionProvider', 'CPUExecutionProvider']`
  (observed in `scripts/run_all.sh` logs). Affected: `run_all.sh`, `run_api.sh`,
  `run_gradio.sh`, `run_live.sh`.
- **A `uv` venv has no `pip`.** `uv sync`/`uv venv` ship neither a `pip` binary
  nor the `pip` module, so `source .venv/bin/activate` leaves bare `pip`
  resolving to the **system** pip (Colab especially) — every install lands in
  the system site-packages and the venv stays untouched, silently. The venv +
  install logic now lives in one sourced file, **`scripts/_onnx_env.sh`**
  (`use_project_venv` defines `pip_install`/`pip_uninstall` on
  `uv pip install --python "${PY}"`, plus `probe_*`/`ensure_onnx_stack`/
  `augment_ort_libs`/`probe_trt`), which `scripts/convert_to_onnx.sh` **and** the
  per-unit `scripts/convert_{fmt,wav2vec}_to_onnx.sh` all source; none of them
  use bare `pip` anymore.
- **`onnx` has to stay in the 1.17–1.18 window.** ORT 1.23 refuses any model
  whose IR version exceeds **11**, and a model built with `onnx.helper` defaults
  to the *installed* onnx's IR version (1.17→10, 1.18→11, 1.19→12,
  1.20/1.21→13, 1.23→14). `onnxscript` (the dynamo exporter's translator) needs
  onnx>=1.17, so 1.17/1.18 is the only safe pair. Conversion itself is immune —
  Torch hardcodes IR 10 — but anything else that builds ONNX (the tests use
  `helper.make_model`) fails to load with "Unsupported model IR version".
  `scripts/_onnx_env.sh` therefore pins `ONNX_PIN="onnx>=1.17,<1.19"`, not the
  old `onnx<1.22` (which still allows IR 12/13), and its `probe_onnx` checks the
  onnx **version** (not just importability) so `ensure_onnx_stack` actually
  repairs an out-of-window onnx (e.g. a stray 1.19+ pulled in by another tool).
- **TRT FP16 cannot meet an elementwise gate on bounded outputs.** TensorRT
  FP16 rounds every accumulation in a deep stack, so on outputs that are
  *small or bounded* the drift is absolute and no `fp16_tol` factor fixes it:
  `emotion` (a probability vector; measured worst cosine 0.9985, `mean_abs`
  8.9e-3, `max_abs` 3.1e-2), `refenc` (conv feature maps, |b| up to ~39;
  worst `max_abs` 9.8e-2 on their small elements), `audio` (projected
  wav2vec feats; worst cosine 0.999996, `mean_abs` 4.2e-4, `max_abs` 1.6e-2),
  and `fmt` (the flow-matching vector field; worst cosine 0.999996, `mean_abs`
  5.9e-3, `max_abs` 4.1e-2).
  All four therefore also pass a **functional-equivalence gate** —
  `cosine >= FP16_MIN_COSINE` (0.995) and `mean_abs <= FP16_MAX_MEAN` (0.02),
  see `fast_float/onnx/export.py` and `parity.compare_outputs` — exactly as the
  decoder has since the first TRT run. The elementwise gate stays the primary
  check; `check_parity` prints `PASS(functional)` when only the fallback saved
  it. The same bounds are set on the `Case`s in `select_backends.py` so
  selection agrees with conversion. Each was verified FP32-exact first (CUDA
  EP: `max_abs` ~1e-5, cosine 1.0), so this is FP16 rounding, not a graph bug.
  (`audio` only started hitting TRT FP16 once its converter passed explicit
  TRT profiles for the dynamic `conv_feats`; before that TRT could not build an
  engine and the EP fell back to CUDA, where the strict gate passed by luck.
  `fmt` is the same story: before the scalar-`t` export fix above, TRT built no
  engine and CUDA carried it.)
- Full details and rationale: **`docs/BACKEND_SELECTION.md`** — read it before
  touching the ONNX path.

### 7. Logging & timing (debug mode)

- **One logging entry point**: `fast_float/utils/logger.py`. Importing it installs a
  coloured stderr handler on the `fast_float` logger (`propagate=False`, so
  Gradio/Jupyter root rewiring can't swallow our records). Module loggers
  (`getLogger(__name__)` under `fast_float.*`) inherit its level.
- **Env-driven levels** — never hard-code `setLevel` in modules:
  `FLOAT_LOG_LEVEL` (name, e.g. `DEBUG`) → else `FLOAT_DEBUG`/`FLOAT_TIMING`
  (truthy) → else `INFO`. `set_log_level(level)` / `configure_logging(debug=True)`
  flip **every** managed logger at once (this is what `--debug` calls — the old
  code set the *root* logger, which did nothing because we `propagate=False`).
- **Per-stage timing**: `fast_float/utils/timing.py`. Open a run with
  `with timing_run("name"):` (re-entrant: the outermost scope owns the collector
  and prints one summary table) and wrap blocks with
  `with timed("stage", gpu=True):`. Stages aggregate by name (count/total/mean/
  max/p95). `gpu=True` records a CUDA-event pair and defers the read to the
  summary, so the pipeline never stalls mid-run. Everything is a no-op when no
  run scope is active.
- **Always-on summary** (INFO) at end of a run; **per-call lines** only in debug.
  Instrumented stages: `preprocess`, `encode_image`, `encode_identity`,
  `audio_encode`, `emotion_encode`, `ode_solve`, `decode`, `decode_tune`,
  `frame_gather`, `video_write`/`video_encode`/`video_finalize`, `save`, `merge`.
- Serve env `FLOAT_DEBUG=1` to get both the debug level and per-call timing.

### 8. Shell & platform

- This repo is developed on **Windows (PowerShell 5.1)**; `bash` commands are
  **WSL** (`/mnt/c/...`). Use PowerShell for tooling (`uv`, `git`, `ruff`).
- In PowerShell, chain with `cmd1; if ($?) { cmd2 }` (there is no `&&`).
- Validate `.sh` changes against an LF, BOM-free copy if `bash -n` complains —
  committed scripts that fail locally usually only have CRLF line endings.

### 9. Live streaming (frame dispatcher → WebSocket / WebRTC)

**Why a tap, not a second pipeline.** Generation is GPU-bound and writes MP4 as
fast as it can; live transports need the *same* frames as they leave the decoder.
`fast_float/streaming/` is a per-request hook fed by the writer thread — it never
re-runs inference.

**The dispatcher theorem (locked — do not "fix" one leg by breaking another).**
Pick any two of {contiguous `idx`, bounded memory, never block the producer}:

| mode | queue | drops | producer |
| --- | --- | --- | --- |
| `live` (default) | bounded by `max_queue_frames` | oldest frames, reported as gaps | never blocked |
| `lossless` | bounded by `max_queue_frames` | none — blocks up to `submit_timeout_ms`, then disables | throttled by design |

- `FrameDispatcher` owns `idx`: `submit(frames)` numbers frames from `_cursor` in
  submission order, so chunked generation is continuous with no `frame_base`
  plumbing. `pts_ms = (idx - first_idx) * 1000 / fps`.
- `submit()` **copies** before returning; the pipeline recycles pinned pool
  buffers immediately (`pool.put(buf)` right after `submit`). Never hand
  `submit()` a buffer you keep mutating.
- Hooks (`open` → `dispatch`/`gap` → `close`) run on the **delivery thread**, in
  order. Keep them fast; `max_dispatch_ms`/`stall_limits` disable a stalling tap.
- `max_lag_ms` is a bound in **media time**, not wall clock — a wall-clock bound
  starves any stream whose generation is slower than real time.
- Defaults: `max_queue_frames=50`, `max_lag_ms=1000`, `gap_timeout_ms=0`,
  `submit_timeout_ms=2000`, `close_timeout=2.0`, `max_dispatch_ms=250`,
  `stall_limits=3`, `dispatch_granularity="batch"`, `abort_on_error=False`,
  `fail_fast_on_open=False`. `LiveStreamDispatcher` overrides granularity to
  `"frame"` with `max_queue_frames=64`, `max_lag_ms=1500`.

**Pipeline wiring** (already applied — follow the same shape for new writers):
`inference_and_save(..., dispatcher=)` calls `dispatcher.bind(StreamMeta(...))`
once per run, `dispatcher.submit(buf[:n].numpy())` in **every** writer branch
(stream-only, torchcodec, PyAV, FFmpeg) *before* `pool.put(buf)`, and
`dispatcher.error(exc)` on writer failure.
`video_path=None` selects the stream-only branch: no file is muxed and no
encoder/`ffmpeg` is spawned, because live callers (LiveKit) consume frames
through the dispatcher only. A run with neither `video_path` nor a dispatcher
raises `ValueError` up front instead of failing inside `subprocess.Popen`.
`generate_video` owns the terminal `await asyncio.to_thread(dispatcher.shutdown)`
in its outermost `finally` (so preprocessing failures shut down too).
`run_inference` / `generate` forward the tap; the batched path warns and ignores
it.

**`LiveStream` fan-out.** `LiveStreamDispatcher.dispatch`/`gap` publish tuples
`("frame", idx, pts_ms, ndarray)` / `("gap", start, end, reason)`; `finish()` adds
the `CLOSE` sentinel. Subscribers are `asyncio.Queue`s that drop the oldest item
when full. **Every loop interaction from a non-loop thread must go through
`call_soon_threadsafe`** — `start()` (called by `bind` on the writer thread) and
`finish()`/`publish_threadsafe()` do. A direct `Event.set()` from the pipeline
thread hangs `wait_meta()` for its full timeout. `finish()` defers the sentinel
through the loop so already-scheduled publishes land first, and `publish()` has
no `done` guard for the same reason. `subscribe()` happens **before**
`wait_meta()` so the first decoded frames are not published to nobody (frames
published before a subscriber exists are lost by design — live, not replay).

**WS protocol** (`fast_float/routers/streaming.py`, `HEADER = struct.Struct(">III")`):

| direction | payload |
| --- | --- |
| server → | `{"type":"meta","fps","width","height","channels","first_idx","total_frames","has_audio"}` |
| server → | binary `[u32 idx][u32 count][u32 pts_ms][rgb(a) bytes]`, one frame per message |
| server → | `{"type":"gap","start","end","reason"}` · `{"type":"done","result"}` · `{"type":"error","message"}` |

Close codes: `4404` (unknown/finished request), `4408` (no metadata — the
stream's `error` is forwarded in the message). `WAIT_TIMEOUT_S=15` (POST/WS race)
and `META_TIMEOUT_S=30` are module constants so tests can monkeypatch them. A
slow reader loses frames; it never backpressures the GPU.

**WebRTC** (`routers/webrtc.py`): `"live": true` routes to `_live_offer`, which
subscribes **before** `wait_meta`, registers the subscription in `_live_subs`
**before** `addTrack`, and wraps subscribe→hand-off in
`except BaseException: stream.unsubscribe(queue)` — a cancelled offer must not
leave a queue in the stream for the stream's lifetime. `LiveVideoTrack` paces at
`meta.fps`, sets `pts = pts_ms * 90` with `time_base=1/90000`, and uses
`format="rgba"` for 4-channel frames. Request audio is attached as a
`MediaPlayer` (approximate A/V sync: video trails generation, audio starts at
the offer). `_cleanup_peer` stops the track, unsubscribes, pushes `CLOSE`, and
stops the audio player's tracks. The browser must declare `recvonly` video+audio
transceivers *before* `createOffer` and must merge remote tracks
(`attachRemoteTrack`) instead of assigning `event.streams[0]` blindly.

**`/generate` integration:** register the stream in `app_state.live_streams`
*before* the semaphore (so a fast WS attach finds it), force
`chunk_batch_size=1`, pass `dispatcher=`, `finish(result=...)` on success, and
`_close_live(...)` on 429 / `BaseException` / `finally`. `_close_live` only pops
the registry entry when that id still maps to *this* stream. Every failure path
must call `finish(...)` so watchers never hang.

**Known limits (by design — do not "fix" casually):**

- `app_state.live_streams` is **per process** → single worker only; multi-worker
  deployments need a broker/broadcast layer.
- WS ships raw RGB (~19.6 MB/s at 512²×25 fps) and the browser converts with a
  per-pixel JS loop. Prefer WebRTC for bandwidth-limited clients.
- No WS auto-reconnect; a vanished client is reaped by ASGI pings and by send
  failures.
- `dispatcher.error()` is still invoked from the writer thread (not the delivery
  thread) — keep it side-effect free / logging-only.

**Tests:** `tests/test_frame_dispatcher.py` (live/lossless, gaps, lag,
shutdown, stats) and `tests/test_live_stream.py` (fan-out, drop-oldest, WS
round-trip, and an end-to-end `TestClient` run through the real delivery thread —
the regression guard for cross-thread wake-ups). Patch `WAIT_TIMEOUT_S` down when
adding WS tests so failures are fast.
