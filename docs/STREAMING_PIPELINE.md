# FLOAT Streaming Inference Pipeline

This document describes the end-to-end asynchronous pipeline introduced in `inference_and_save()`, including the `DecodeAutoTuner` algorithm and all CUDA stream synchronization guarantees.

---

## 1. Architecture Overview

The pipeline is built around an **asynchronous multi-threaded producer-consumer architecture** that overlaps GPU compute, PCIe transfers, and disk encoding across multiple CUDA streams and GPUs. No stage blocks another.

```
                      ┌──────────────────────────────────────────────┐
                      │         Stage 1: Condition Encoding           │
                      │   stream_encode (Wav2Vec2 + Audio2Emotion)    │
                      └─────────────────────┬────────────────────────┘
                                            │  Event sync (bidirectional)
                                            ▼
                      ┌──────────────────────────────────────────────┐
                      │    Stage 2: ODE Sampling (producer thread)   │
                      │  primary_stream / FMT Flow Matching ODE      │
                      │  Immediate dispatch — no carry-frame stall   │
                      └─────────────────────┬────────────────────────┘
                                            │
                                 work_q  (micro-batches + events)
                                 ┌──────────┴────────────┐
                                 ▼                       ▼
              ┌──────────────────────────┐  ┌──────────────────────────┐
              │  Worker 0 (cuda:0)       │  │  Worker 1 (cuda:1)       │
              │  stream_0                │  │  stream_1                │
              │  StyleDecoder Synthesis  │  │  StyleDecoder Synthesis  │
              └────────────┬─────────────┘  └────────────┬─────────────┘
                           │                             │
                      done_q (decoded buffers + CUDA Events)
                                            │
                                            ▼
                      ┌──────────────────────────────────────────────┐
                      │  Stage 4: Video Encoding (writer thread)     │
                      │  In-order reorder window → FFmpeg pipe       │
                      │  (NVENC / x264)   ← recycles pinned bufs    │
                      └──────────────────────────────────────────────┘
```

---

## 2. Stage-by-Stage Breakdown

### Stage 1 — Condition Encoding

**Files:** `fast_float/models/float/FLOAT.py` – `FLOAT.sample()`

The reference image is encoded **once** up-front (before threads are spawned):

- `encode_image_into_latent(s)` → `s_r` (reference motion latent) + `s_r_feats` (StyleGAN feature maps)
- `motion_autoencoder.dec.direction(r_s_lambda)` → `r_s` (identity motion direction)

On multi-GPU setups, `s_r_feats` is cloned to secondary GPU memory using `f.to(dev)`, followed by explicit synchronization on **both** GPUs to guarantee the copy is complete before any worker thread reads the features:

```python
feats = [f.to(dev) for f in s_r_feats]
torch.cuda.synchronize(dev)
torch.cuda.synchronize(primary)
```

Audio and emotion encoding run inside `sample()` on the dedicated `stream_encode` stream, with a **bidirectional barrier**:

```python
# 1. Ensure stream_encode sees any input tensors written on cur_stream
encode_stream.wait_stream(cur_stream)

with torch.cuda.stream(encode_stream):
    wa = self.audio_encoder.inference(a, seq_len=T)
    we = self.emotion_encoder.predict_emotion(a)

# 2. ODE solver waits for wa / we to be fully ready
cur_stream.wait_stream(encode_stream)
```

### Stage 2 — ODE Sampling (`sampler_producer` thread)

**Files:** `fast_float/models/float/FLOAT.py`, `fast_float/models/float/FMT.py`

The `torchdiffeq.odeint` ODE solver runs **chunk-by-chunk** (typically 25 frames ≈ 1 second of audio) on `primary_stream`. Each ODE solve requires `nfe` (Number of Function Evaluations, default 10) forward passes of the Flow Matching Transformer.

#### Dynamic-Batch Coalescing (15 ms deadline)

Frames flow out of the ODE solver one chunk at a time (~25 frames ≈ 1 second of audio per call). Rather than slicing each chunk into fixed batches of exactly `B` and padding any remainder, the producer accumulates frames into a **carry buffer** and flushes it under two conditions:

- **(a) Carry reaches `B` frames** → flush immediately. Mid-video this fires on every frame, so there is zero added latency relative to the old fixed-slice approach.
- **(b) 15 ms elapses since the first frame entered the carry** → flush whatever is accumulated. This fires at ODE chunk boundaries and at end-of-video, coalescing leftover frames from adjacent chunks instead of dispatching tiny padded batches.

```python
COALESCE_S = 0.015  # 15 ms, configurable via opt.decode_coalesce_ms

carry, deadline = [], 0.0
for chunk in self.sample(...):
    for row in chunk.reshape(-1, dim_w).unbind(0):
        if not carry:
            deadline = time.perf_counter() + COALESCE_S
        carry.append(row)
        if len(carry) >= B or time.perf_counter() >= deadline:
            seq, carry = flush_carry(carry, seq)  # pad to B inside flush

if carry:  # final remainder
    seq, carry = flush_carry(carry, seq)
```

**Why 15 ms?** The ODE solve for one 25-frame chunk takes ~300 ms on a T4. The decoder finishes a batch of 4 in ~90 ms. So the carry window between chunks is long enough for the 15 ms deadline to fire and coalesce without any perceptible delay. At 25 fps, one frame period is 40 ms — a 15 ms coalesce adds at most 37% of one frame period of extra latency to boundary batches, which is imperceptible.

### Stage 3 — Multi-GPU Decode Workers

**Files:** `fast_float/models/float/FLOAT.py` – `_decode_u8()`, `fast_float/models/float/styledecoder.py`

Each GPU has **one dedicated worker thread** with its own CUDA stream (`max_streams=1`). A single stream per GPU prevents cuDNN workspace collisions caused by multiple concurrent forward passes through the same model instance.

#### CUDA Stream Synchronization

**Primary GPU worker:**

```python
with torch.cuda.stream(stream):
    stream.wait_event(ev)  # GPU-side wait — no CPU stall
    lat_gpu.record_stream(stream)  # Prevent premature buffer deallocation
    u8 = self._decode_u8(ae, feats, lat_gpu)
    buf[:n_valid].copy_(u8[:n_valid], non_blocking=True)
    done = torch.cuda.Event()
    done.record(stream)

done_q.put((seq, buf, n_valid, done))  # No done.synchronize() here!
```

**Secondary GPU worker:**

```python
with torch.cuda.stream(stream):
    ev.synchronize()  # CPU wait for host copy completion
    lat = lat_host.to(dev, non_blocking=True)
    lat.record_stream(stream)  # Prevent premature deallocation
    u8 = self._decode_u8(ae, feats, lat)
    buf[:n_valid].copy_(u8[:n_valid], non_blocking=True)
    done = torch.cuda.Event()
    done.record(stream)

done_q.put((seq, buf, n_valid, done))
```

> **Key insight:** `done.synchronize()` was moved **out of the worker** and into the `writer` thread. Without this, each worker thread would block until the GPU finished decoding, causing the decode stream to drain between batches. With the new design, the worker immediately loops to enqueue the next batch, keeping the GPU kernel queue 100% full.

#### In-Place uint8 Conversion (`_decode_u8`)

The old conversion chain created 5 separate intermediate GPU tensors (~250 MB of VRAM churn per batch at B=16):

```python
# Old: 5 allocations
(out.float().clamp(-1, 1) * 127.5 + 127.5).to(torch.uint8).permute(0, 2, 3, 1).contiguous()
```

The new version uses in-place operations to avoid all intermediate allocations:

```python
# New: 2 allocations (float cast + uint8 cast)
out = out.float()
out = out.add_(1.0).mul_(127.5).clamp_(0.0, 255.0).to(torch.uint8)
return out.permute(0, 2, 3, 1).contiguous()
```

#### Memory Alignment Fix in `ModulatedConv2d`

At large batch sizes, `ModulatedConv2d.forward()` constructs grouped convolutions by reshaping and transposing weight tensors. The resulting strides may not satisfy cuDNN's 16-byte alignment requirement for vectorized memory loads, causing a `CUDA error: misaligned address` crash.

**Fix:** Explicit `.contiguous()` calls before every grouped convolution:

```python
# Upsample path
input = input.contiguous().view(1, batch * in_channel, height, width)
weight = weight.transpose(1, 2).contiguous().reshape(...)
out = F.conv_transpose2d(input, weight, ..., groups=batch)

# Standard path
input = input.contiguous().view(1, batch * in_channel, height, width)
weight = weight.contiguous()
out = F.conv2d(input, weight, ..., groups=batch)
```

### Stage 4 — Video Writing (`writer` thread)

The writer maintains a `pending` dictionary keyed by sequence number `seq`, consuming items from `done_q` in any order and writing to FFmpeg only when `nxt` (the next expected sequence) is present:

```python
pending[seq] = (buf, n, done)
while nxt in pending:
    buf, n, done = pending.pop(nxt)
    done.synchronize()  # Wait for GPU → host copy to finish
    proc.stdin.write(buf[:n].numpy().reshape(-1))
    pool.put(buf)  # Recycle pinned host buffer
    nxt += 1
```

This guarantees:

- Output frames are always in correct temporal order.
- The GPU decode stream is never blocked waiting for the FFmpeg pipe.
- Pinned host buffers cycle back to `pool`, applying natural backpressure to the producer.

`_drain_ordered()` implements this loop for the **stream-only** branch; the
FFmpeg/PyAV/torchcodec writers keep the same reorder pattern inline ahead of
their own sink. With `video_path=None` the pipeline takes the stream-only
branch: no encoder is started and `dispatcher.submit()` is the only sink (this
is how LiveKit consumes frames).

---

## 3. `DecodeAutoTuner` — How It Works

The auto-tuner (`DecodeAutoTuner`) runs once per GPU at the start of inference (results are memoized across calls). It discovers the optimal decode batch size for the current GPU's available VRAM without requiring any manual configuration.

### Algorithm

```
FOR b = 1, 2, 4, 8, 16:
  IF projected_mem(b) > budget:  STOP
  IF OOM:                        STOP
  Measure transient VRAM and FPS at batch size b

feasible = { b : transient(b) * safety <= budget }

best_fps  = max(FPS(b) for b in feasible)
plateau   = 0.93 * best_fps

chosen = min(b for b in feasible if FPS(b) >= plateau)
```

### Memory Budgeting

Before testing any batch size, the tuner computes the available VRAM budget:

```python
free, total = torch.cuda.mem_get_info(device)

# Primary GPU: sampler + NVENC + cuDNN workspace
reserve = max(800 MB, 8% of total)

# Secondary GPU: decode only
reserve = max(350 MB, 3% of total)

budget = free - reserve
```

A `safety` factor of **1.3×** is applied to transient measurements to account for PyTorch caching allocator overhead and cuDNN workspace growth.

### Measurement

For each candidate batch size `b`:

1. **Warmup pass** — lets cuDNN select and initialize kernel algorithms (auto-tune).
2. **Measurement pass:**
   - `torch.cuda.reset_peak_memory_stats()`
   - Time `decode_fn(b)` end-to-end including GPU sync.
   - Record `peak_mem - base_mem` as transient bytes.
   - Compute `fps = b / elapsed_time`.

### Plateau Selection

The tuner selects the **smallest** batch size that achieves ≥ 93% of peak throughput:

```python
chosen = min(k for k, (_, f) in feasible.items() if f >= 0.93 * best_fps)
```

**Example from a Tesla T4 (2× GPU) run:**

| Batch | FPS  | Transient VRAM |
|-------|------|----------------|
| 1     | 32.0 | ~0.15 GiB      |
| 2     | 39.5 | ~0.31 GiB      |
| 4     | 44.4 | ~0.61 GiB      |
| **8** | **44.5** | ~1.23 GiB  |
| 16    | 43.9 | ~2.46 GiB      |

- Best FPS = 44.5 → plateau threshold = 0.93 × 44.5 = **41.4 fps**
- All batches ≥ 4 exceed the threshold.
- **Chosen = 4** (smallest in plateau) — saves ~4 GiB of VRAM vs batch=32 at identical throughput.

> **Note:** Choosing the smallest batch in the plateau also reduces per-chunk decoding latency, meaning frames stream out faster and the producer thread has less opportunity to accumulate backpressure on `work_q`.

### Startup Time

| Setting         | Old Default | New Default |
|-----------------|-------------|-------------|
| `max_batch`     | 32          | 16          |
| `repeats`       | 2           | 1           |
| Warmup passes   | 2× per step | 1× per step |
| Estimated startup | ~10 s (2 GPUs) | ~1.5–2 s (2 GPUs) |

---

## 4. Performance Summary

| Bottleneck | Before | After |
|---|---|---|
| **Decoder starvation** | Remainder frames held in `carry`; decoder idle 500–1000 ms per chunk | All frames dispatched immediately every chunk |
| **Worker serialization** | Workers locked on `state["next_alloc"]`; 100 ms sleep stalls | Lock removed; writer handles reordering |
| **GPU pipeline gaps** | `done.synchronize()` in worker drained the GPU kernel queue | Moved to writer; GPU stream stays saturated |
| **Batch VRAM waste** | Tuner selected batch=32 (4.91 GiB) at same throughput as batch=4 | Tuner selects batch=4 (0.61 GiB); saves ~4.3 GiB per stream |
| **Tuner startup time** | ~10 seconds | ~1.5–2 seconds |
| **Multi-stream collisions** | Up to 3 streams/GPU on one model instance | 1 stream per GPU (no cuDNN workspace races) |
| **CUDA misaligned address** | Non-contiguous tensors in grouped convs | Explicit `.contiguous()` before all grouped `F.conv2d` calls |
| **Encoder stream race** | One-way wait (cur_stream → ODE only) | Bidirectional barrier (`encode_stream ↔ cur_stream`) |
| **uint8 conversion VRAM churn** | 5 intermediate tensor allocations (~250 MB/batch) | In-place ops; 2 allocations total |

---

## 5. Configuration Options

All tuning parameters can be overridden via `opt` attributes on the inference call:

| `opt` attribute       | Default        | Description |
|-----------------------|----------------|-------------|
| `decode_coalesce_ms` | `15.0`         | Max ms to wait for a full batch before flushing a partial one |
| `decode_batch_size`   | `None` (auto)  | Fix decode batch size, bypassing auto-tuner |
| `decode_max_batch`    | `16`           | Maximum batch size the tuner may test |
| `decode_max_streams`  | `1`            | Maximum parallel decode streams per GPU |
| `decode_reserve_gib`  | `None` (auto)  | VRAM (GiB) to keep free per GPU |
| `decode_streams`      | `0` (auto)     | Force a fixed number of decode streams |
| `ngpus`               | `1`            | Number of GPUs to use for decoding |
| `decode_on_primary`   | `True`         | Whether primary GPU also decodes |
| `use_nvenc`           | `True`         | Use hardware H.264 encoder if available |
