# FLOAT Fast — Agent Development Guide (`AGENT.md`)

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
    └── options/                    # Backward-compatibility shims for legacy imports
        ├── __init__.py
        └── base_options.py
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
- The PyTorch stack is pinned to explicit CUDA 11.8 wheels via
  `[[tool.uv.index]]` + `[tool.uv.sources]` (see below). Do not add torch/
  torchvision/torchaudio without their CUDA source mapping.

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

## 7. API Reference

| Method | Endpoint | Description |
| --- | --- | --- |
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
- **No automatic runtime self-test or fallback** — parity/selection is validated
  by the converter/selector scripts only (`python -m fast_float.onnx.select_backends`).
  ORT providers (`trt → cuda → cpu`) fall back *inside* session creation.
- Full details and rationale: **`docs/BACKEND_SELECTION.md`** — read it before
  touching the ONNX path.

### 7. Shell & platform

- This repo is developed on **Windows (PowerShell 5.1)**; `bash` commands are
  **WSL** (`/mnt/c/...`). Use PowerShell for tooling (`uv`, `git`, `ruff`).
- In PowerShell, chain with `cmd1; if ($?) { cmd2 }` (there is no `&&`).
- Validate `.sh` changes against an LF, BOM-free copy if `bash -n` complains —
  committed scripts that fail locally usually only have CRLF line endings.
