# Dual-Process Architecture & IPC (Streaming Backend + FLOAT Worker)

This document explains the architecture, communication protocol, shared memory ring buffer, lifecycle management, and current scheduling model implemented in `float-fast`.

---

## 1. High-Level Architecture Overview

To eliminate event-loop latency, isolate PyTorch CUDA memory and context, and avoid blocking WebRTC connections during heavy GPU flow-matching sampling, the system is separated into two dedicated processes:

```
+-----------------------------------------------------------------------------------+
| Browser / Client                                                                  |
| (Microphone Audio, Camera / Screen, WebRTC Playback)                              |
+-----------------------------------------------------------------------------------+
                                    |
                             WebRTC / LiveKit
                                    v
+-----------------------------------------------------------------------------------+
| PROCESS 1: Streaming Backend (LiveKit / WebRTC)                                   |
| - LiveKit room & WebRTC lifecycle                                                 |
| - Audio reception from browser or Gemini Live Agent                               |
| - Audio chunk buffering (first_chunk_sec / chunk_sec)                             |
| - Frame pacing & WebRTC track publication                                         |
| - ZmqFloatVideoGenerator (IPC client)                                             |
+-----------------------------------------------------------------------------------+
             |                                              ^
             | ZeroMQ (TCP)                                 | Shared Memory
             | Control (REQ/REP) & Notifications (PUB/SUB)  | RGB24 + Audio Frames
             v                                              |
+-----------------------------------------------------------------------------------+
| PROCESS 2: FLOAT Inference Worker (Dedicated GPU Process)                         |
| - CUDA Context & Model Weights (OptimizedInferenceAgent)                          |
| - Owns FrameRingBuffer (producer)                                                 |
| - ZeroMQ Server: REP (Port 5555) & PUB (Port 5556)                                |
| - SHM Dispatcher: writes RGB24 frames + 40ms PCM16 slices directly to RAM         |
| - Interrupt handling, silence loop pre-rendering                                  |
+-----------------------------------------------------------------------------------+
```

---

## 2. Communication & Transport Layers

### ZeroMQ Sockets (Windows-Safe TCP)

- **Control Socket (`tcp://127.0.0.1:5555`)**:
  - Worker runs a `REP` socket; backend connects via `REQ`.
  - Commands: `infer`, `prepare_silence`, `interrupt`, `status`, `shutdown`.
- **Frame Notification Socket (`tcp://127.0.0.1:5556`)**:
  - Worker runs a `PUB` socket; backend connects via `SUB`.
  - Emits lightweight `IPCFrameNotification` events whenever a decoded frame is written to the shared memory ring buffer.

### Shared Memory Frame Ring Buffer (`FrameRingBuffer`)

- Built on `multiprocessing.shared_memory.SharedMemory`.
- **Slot Structure**:
  - `Header` (24 bytes): `frame_idx` (int64), `pts_ms` (float64), `width` (int32), `height` (int32), `audio_len` (int32)
  - `Audio Data` (2048 bytes): up to 1024 16-bit mono samples (64 ms @ 16 kHz)
  - `Image Data` (786,432 bytes): 512×512×3 RGB24 raw bytes
- **Total Slot Size**: ~788 KB.
- **Default Ring Size**: 64 slots (~50 MB total RAM, ~2.56 seconds of video/audio headroom at 25 fps).
- **Windows Detached Copy**: Readers copy memory slices before returning numpy arrays to avoid Windows `BufferError: cannot close exported pointers exist` when closing/unlinking.

---

## 3. Configuration (`settings.toml`)

Dual-process mode is opt-in and backward-compatible:

```toml
[ipc]
# false = single-process mode (default, backwards compatible)
# true  = dual-process mode: streaming backend + FLOAT worker
enabled = false
zmq_control_endpoint = "tcp://127.0.0.1:5555"
zmq_frames_endpoint  = "tcp://127.0.0.1:5556"
shm_name = "float_frame_ring"
shm_ring_size = 64
```

When `enabled = true`:

1. Start the worker:

   ```bash
   python -m fast_float.workers.inference_worker
   ```

2. Start the API / Streaming Backend:

   ```bash
   python -m fast_float.main
   ```

---

## 4. Current Inference Scheduling vs. Priority-Ordered Scheduler

### What is currently implemented

1. **Synchronous REQ/REP Request-Response**:
   - The worker handles requests sequentially as they arrive over the ZeroMQ `REP` socket.
2. **Immediate Interruption (`cmd="interrupt"`)**:
   - When a user speaks or an interruption event occurs, the backend issues an `interrupt` command.
   - An internal `threading.Event` is set in the worker, causing `_ShmFrameDispatcher` to raise `_Cancelled` and immediately abort the active inference pass on the GPU.
3. **Chunk Batching & Internal Windowing**:
   - Audio is buffered up to `chunk_sec` (e.g. 5.0s) and passed directly to FLOAT.

### Is a Priority-Ordered Scheduler implemented?

**No, not yet.** The current worker processes incoming inference jobs on a **FIFO (First-In, First-Out)** basis via ZeroMQ REP/REQ, with preemptive interruption.

### Proposed Priority-Ordered Scheduler Design

A priority scheduler becomes essential when:

1. Multiple live streams or users share one GPU worker.
2. Background jobs (e.g., pre-rendering silence loops, offline file rendering via `/generate`, warming up cache) compete with live conversation.

#### Priority Tiers

| Priority Level | Traffic Type | Description |
|---|---|---|
| **P0 (Critical)** | Preemption / Interruption | Stop current generation, discard stale frames. |
| **P1 (High)** | Live Conversational Audio | Active user speech chunks; lowest latency requirement. |
| **P2 (Normal)** | Silence / Idle Pre-rendering | Offline preparation of natural resting state expressions. |
| **P3 (Batch / Low)** | Offline MP4 Generation | `/generate` API requests that can tolerate slight queuing. |

---

## Examples

Dual-process mode is already enabled in `settings.toml` (`[ipc] enabled = true`);
the scripts also export `FLOAT_IPC_ENABLED`, but Python reads the settings file.

```bash
# Whole stack: the worker binds :5555/:5556 + shared memory first
./scripts/run_dual_process.sh

# The two halves in separate terminals, worker first
uv run --frozen --no-sync python -m fast_float.workers.inference_worker
uv run --frozen --no-sync hypercorn fast_float.main:app --bind 0.0.0.0:8000

# Through the CLI (serve is dual-process by default)
uv run --frozen --no-sync fast-float worker
uv run --frozen --no-sync fast-float serve --no-dual-process
```

A standalone API (`run_api.sh`, `fast-float serve --no-dual-process`) has no
worker to talk to, so either start one or set `[ipc] enabled = false` in
`settings.toml`.

Move the endpoints: the worker binds `FLOAT_ZMQ_*`/`FLOAT_SHM_*`, while the API
connects to `[ipc]` in `settings.toml`, so both sides must be changed together.

```bash
export FLOAT_ZMQ_CONTROL_PORT=6555 FLOAT_ZMQ_FRAMES_PORT=6556 FLOAT_SHM_SLOTS=128
./scripts/run_dual_process.sh
```

Check the wiring:

```bash
curl -s localhost:8000/health | jq '{model_loaded, warmup_stage}'
grep -h 'ipc\|zmq\|shm' logs/worker.log logs/api.log | tail -n 20
```

Run-mode matrix: [USAGE.md](USAGE.md). Launcher lifecycle:
[run_dual_process.md](run_dual_process.md).
