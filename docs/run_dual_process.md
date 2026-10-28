### How `run_dual_process.sh` Works

`scripts/run_dual_process.sh` is an orchestrator script that starts and manages the complete dual-process stack under a single command, ensuring both processes start in the correct order and shut down cleanly together.

Here is a step-by-step breakdown of how it works:

---

### 1. Step-by-Step Execution Lifecycle

```
                     ./scripts/run_dual_process.sh
                                   │
      ┌────────────────────────────┼────────────────────────────┐
      ▼                            ▼                            ▼
[Process 1: GPU Worker]   [Process 2: API Backend]    [Process 3: Studio Web]
- Loads PyTorch & weights - FastAPI / Hypercorn       - Vite Dev Server (port 3000)
- Creates Shared Memory   - LiveKit WebRTC Session   - Browser UI
- Binds ZMQ:5555 & 5556   - ZmqFloatVideoGenerator
```

#### Step 1: Environment Configuration & IPC Flags

The script first exports environment variables that tell both processes to use the ZeroMQ + Shared Memory pipeline:

```bash
export FLOAT_IPC_ENABLED="true"
export FLOAT_ZMQ_CONTROL_PORT="5555"
export FLOAT_ZMQ_FRAMES_PORT="5556"
export FLOAT_SHM_NAME="float_frame_ring"
export FLOAT_SHM_SLOTS="64"
```

#### Step 2: LiveKit SFU & Gemini Worker Discovery

- Checks if a LiveKit SFU (local Docker container or LiveKit Cloud) is configured.
- Checks if a Gemini Live conversational worker should be launched in the background (`fast_float.livekit.gemini.agent`).

#### Step 3: Process 1 — FLOAT GPU Inference Worker

```bash
uv run python -m fast_float.workers.inference_worker &
```

- **What it does**:
  1. Initializes the CUDA context and loads the full FLOAT model weights (`OptimizedInferenceAgent`) into GPU memory.
  2. Creates the master shared-memory segment (`float_frame_ring`) with 64 circular slots (~50 MB total RAM).
  3. Binds two ZeroMQ TCP ports:
     - `tcp://127.0.0.1:5555` (REP socket) for commands like `infer`, `interrupt`, and `prepare_silence`.
     - `tcp://127.0.0.1:5556` (PUB socket) for broadcasting per-frame notifications.
- The script sleeps for **2 seconds** to guarantee the worker's shared-memory segment and sockets are ready before the backend attempts to connect.

#### Step 4: Process 2 — Streaming Backend (FastAPI / LiveKit)

```bash
uv run hypercorn fast_float.main:app --bind "0.0.0.0:8000" &
```

- **What it does**:
  1. Starts the HTTP/WebRTC server on port 8000.
  2. When a LiveKit conversation or video stream starts, [`session.py`](../fast_float/livekit/session.py) sees `[ipc] enabled = true` in `settings.toml` and instantiates `ZmqFloatVideoGenerator`.
  3. `ZmqFloatVideoGenerator` connects to the worker's shared-memory ring (as a consumer) and subscribes to the ZeroMQ sockets.
  4. The backend never imports or runs heavy PyTorch CUDA inference; its Python event loop remains fast and responsive for WebRTC streaming.

#### Step 5: Process 3 — React Studio Frontend

```bash
(cd frontend && npm run dev -- --host 0.0.0.0 --port 3000) &
```

- Starts the Vite development server serving the Studio UI at `http://localhost:3000`.

---

### 2. How Data Flows at Runtime

When speech audio arrives (either from the user's mic or Gemini Live):

```
Audio In (16 kHz PCM)
      │
      ▼
Process 2 (Backend) ──[ ZeroMQ RPC (cmd: "infer") ]──► Process 1 (GPU Worker)
                                                              │
                                                        FLOAT Model runs
                                                              │
Process 2 (Backend) ◄──[ ZeroMQ Notification: "Slot 3" ]──────┤ (Writes RGB frame
      │                                                       │  to SHM Slot 3)
      ├── Reads Slot 3 directly from RAM (zero-copy)          ▼
      ▼                                                Shared Memory Ring
WebRTC Video & Audio Packets sent to Browser
```

1. **Audio Buffering**: The backend collects audio (default: 5-second chunk) and sends an `IPCRequest(cmd="infer")` to the worker via ZeroMQ port 5555.
2. **GPU Generation**: The worker runs the flow-matching ODE sampler on the GPU.
3. **Zero-Copy Frame Write**: For each generated frame, the worker writes the raw $512 \times 512$ RGB24 image and paired 40 ms PCM16 audio slice directly into the next slot of the Shared Memory buffer.
4. **Lightweight Notification**: The worker publishes a ~60-byte JSON notification (`IPCFrameNotification(slot_idx=3, frame_idx=10)`) over ZeroMQ port 5556.
5. **Instant Playout**: The backend receives the notification, reads slot 3 directly from RAM without network serialization, packages it into a LiveKit `rtc.VideoFrame`, and streams it over WebRTC.

---

### 3. Graceful Teardown (`Ctrl+C`)

The script uses a bash `trap` to handle termination:

```bash
cleanup() {
    trap - INT TERM EXIT
    kill 0 2>/dev/null || true
}
trap cleanup INT TERM EXIT
```

When you press `Ctrl+C` (or the script receives a `SIGTERM` / `EXIT` signal), `kill 0` broadcasts the termination signal to the **entire process group**.

This shuts down:

- The GPU worker (which unlinks and frees the shared memory segment and closes ZMQ sockets).
- The FastAPI backend.
- The Vite frontend.
- The Gemini Live agent (if running).

No orphaned background processes or leaked shared-memory segments remain in RAM.

---

### 4. Examples

```bash
# Full stack: worker + API + Studio (+ Gemini worker when configured)
./scripts/run_dual_process.sh

# Custom backend / Studio ports
PORT=9000 FRONTEND_PORT=3100 ./scripts/run_dual_process.sh

# Custom IPC endpoints: the worker binds the env value, the API connects to
# [ipc] in settings.toml - change both, or leave the defaults
FLOAT_ZMQ_CONTROL_PORT=6555 FLOAT_ZMQ_FRAMES_PORT=6556 ./scripts/run_dual_process.sh

# Bigger ring buffer (higher fps, slower consumers)
FLOAT_SHM_SLOTS=128 ./scripts/run_dual_process.sh

# Skip SFU / Gemini discovery
LIVEKIT_START=no FLOAT_GEMINI_WORKER=no ./scripts/run_dual_process.sh
```

Verify the halves are talking:

```bash
curl -s localhost:8000/health | jq '{status, model_loaded, warming_up}'
curl -s localhost:8000/livekit/status | jq
grep -h 'ipc\|zmq\|shm' logs/worker.log logs/api.log | tail -n 20
```

The same stack through the CLI is `fast-float serve --frontend` (dual-process is
the default; `--no-dual-process` runs the legacy single process). Full run-mode
matrix: [USAGE.md](USAGE.md).
