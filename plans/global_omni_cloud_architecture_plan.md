# FLOAT-Fast: Global Omni Cloud Architecture Plan
> **Last audit**: Full code review against live FLOAT.py / FMT.py — bugs identified and fixes applied below.

---

## Architecture Overview

```
Client Request
     │
     ▼
FloatOmniPipelineEngine.submit_request()
     │  (asyncio.to_thread)
     ▼
FLOAT.prepare_omni_conditions()   ← Wav2Vec2 + image identity prefill (once)
     │
     ▼
FloatOmniScheduler.add_request()  ← round-robin queue
     │
     ▼
[_sampler_loop]  ─────────────────────────────────────────────────
     │   schedule_next_chunk()                                      │
     │   asyncio.to_thread(model.sample_omni_chunk, req)           │  Fair-share
     │   latent_queue.put(LatentChunkTask)                         │  rotation
     └───────────────────────────────────────────────────────────── ┘
                              │
                   asyncio.Queue[LatentChunkTask]
                              │
     ┌────────────────────────┘
     ▼
[_decoder_loop × N]
     │   asyncio.to_thread(model.decode_omni_chunk, task)
     │   task.callback(request_id, chunk_idx, clip, is_last)
     └──► Per-request streaming callback
```

---

## Bugs Found in Code Review

### 🐛 BUG 1 — `get_one_hot_emotion` does not exist (CRASH)
**File**: [`FLOAT.py:972`](file:///c:/Users/emam2/tech_projects/float-fast/fast_float/models/float/FLOAT.py#L972)

`prepare_omni_conditions` calls `self.get_one_hot_emotion(emotion, wa.shape[0])` — **this method does not exist anywhere in the codebase**. The original emotion encoding logic lives inline in `inference_with_audio` (lines 1407–1445) and handles `str`, `list`, `torch.Tensor` emotion types differently.

**Fix**: Replace the phantom call with the actual inline emotion encoding that the original code uses.

```python
# BROKEN
we = self.get_one_hot_emotion(emotion, wa.shape[0]).to(device)

# CORRECT  (mirrors inference_with_audio lines 1430-1441)
emo_idx = self.emotion_encoder.label2id.get(str(emotion).lower(), None)
if emo_idx is not None:
    we = (
        F.one_hot(torch.tensor(emo_idx, device=device), num_classes=self.opt.dim_e)
        .float().unsqueeze(0).unsqueeze(0)   # [1, 1, dim_e]
    )
else:
    we = torch.zeros(1, 1, self.opt.dim_e, device=device)
```

---

### 🐛 BUG 2 — `sample_fn` closure captures mutable variables by reference (SILENT DATA CORRUPTION)
**File**: [`FLOAT.py:1071-1089`](file:///c:/Users/emam2/tech_projects/float-fast/fast_float/models/float/FLOAT.py#L1071)

The `sample_fn` closure captures `wa_t`, `prev_x_t`, `prev_wa_t`, `_audio_cat`, etc. **by Python closure reference**, not by value. In the original code this is safe because everything is local to a loop iteration. But in our `sample_omni_chunk`, `req` is a mutable shared dataclass — if the scheduler assigns a second chunk to the same `req` while `odeint` is still running (impossible with the current `asyncio.to_thread` model but fragile), the closed-over tensors could change.

More critically: **the original `inference_with_audio` defensively binds these as default arguments** (`sample_chunk(tt, zt, _ac=_audio_cat, ...)`). Our `sample_fn` does NOT, relying purely on closure. This is fine for single-threaded but should be brought in line for safety.

**Fix**: Use default-arg binding in the closure (mirror lines 1552-1553 exactly):

```python
def sample_fn(tt, zt,
              _wa=wa_t, _wr=req.ref_latent, _we=req.we,
              _px=prev_x_t, _pwa=prev_wa_t,
              _ac=_audio_cat, _rc=_ref_cat, _ec=_emotion_cat,
              _pxc=_prev_x_cat, _pwc=_prev_wa_cat):
    out = self.fmt.forward_with_cfv(
        t=tt.unsqueeze(0), x=zt,
        wa=_wa, wr=_wr, we=_we,
        prev_x=_px, prev_wa=_pwa,
        a_cfg_scale=req.a_cfg_scale,
        r_cfg_scale=req.r_cfg_scale,
        e_cfg_scale=req.e_cfg_scale,
        _audio_cat=_ac, _ref_cat=_rc, _emotion_cat=_ec,
        _prev_x_cat=_pxc, _prev_wa_cat=_pwc,
    )
    return out[:, self.num_prev_frames:]
```

---

### 🐛 BUG 3 — `prev_wa_t` dimension mismatch on chunks > 0 (SHAPE ERROR)
**File**: [`FLOAT.py:1020-1024`](file:///c:/Users/emam2/tech_projects/float-fast/fast_float/models/float/FLOAT.py#L1020)

On chunk 0, `prev_wa_t = torch.zeros(B, num_prev_frames, dim_w)`. But `req.wa` has shape `[1, T_total, dim_w]` where batch dimension is 1 (from `audio.unsqueeze(0)`). Meanwhile `B = req.wa.shape[0]` = 1. So `prev_wa_t` has shape `[1, num_prev_frames, dim_w]` — **that's correct**.

However, on chunk > 0: `req.prev_wa = wa_t[:, -num_prev_frames:]` slices the **padded** `wa_t` which has shape `[1, num_frames_for_clip, dim_w]`. That's fine, but `dim_w` for `wa` (audio latent dimension from Wav2Vec2) **may differ from** `self.opt.dim_w` (motion latent dimension). In the original code both use `dim_w` in the zero initialization — which means `prev_wa_t` is expected to be audio features in motion space. Check that this is consistent.

**Fix**: Validate `wa_t` shape before slice — add assertion `assert wa_t.shape[-1] == self.opt.dim_w` to catch any silent dimension mismatch early.

---

### 🐛 BUG 4 — Frame-trimming missing on last chunk (OUTPUT LENGTH BUG)
**File**: [`FLOAT.py:1100-1117`](file:///c:/Users/emam2/tech_projects/float-fast/fast_float/models/float/FLOAT.py#L1100)

In `inference_with_audio` (lines 1582-1588), after ODE solve the last chunk is **trimmed** to `T - decoded_length` frames to produce exact output length. `sample_omni_chunk` does **not** do this — it always returns `num_frames_for_clip` frames even on the last chunk, causing the final video to have extra (replicated-padded) frames appended.

**Fix**: Add length tracking to `FloatOmniRequest` and trim the last chunk:

```python
# In FloatOmniRequest (types.py) — add field:
decoded_frames: int = 0

# In sample_omni_chunk — before building LatentChunkTask:
frames_remaining = req.total_frames - req.decoded_frames  # add total_frames to req
if frames_remaining < sample_t.shape[1]:
    sample_t = sample_t[:, :frames_remaining]
req.decoded_frames += sample_t.shape[1]
```

---

### 🐛 BUG 5 — `schedule_next_chunk` pops selected request BEFORE checking abort (RACE)
**File**: [`omni_scheduler.py:88-92`](file:///c:/Users/emam2/tech_projects/float-fast/fast_float/sched/omni_scheduler.py#L88)

```python
selected_id = self._running.pop(0)       # ← popped from running
selected_req = self._requests.get(selected_id)

if selected_req is None or selected_req.status != RequestStatus.RUNNING:
    return None                          # ← request is now GONE from _running
```

If `selected_req` is aborted *after* being popped but before the check, it returns `None` and the `selected_id` is **permanently removed from `_running`** without ever being re-appended. The request is lost from scheduling but still has `status == FINISHED_ABORTED`, so it doesn't leak memory — but the capacity slot is freed correctly. This is actually **fine** — the abort path already sets status to `FINISHED_ABORTED` and the cleanup in step 1 would have caught it on next call anyway. Not a real bug but worth documenting.

**Minor issue**: The check at line 91 should return `self.schedule_next_chunk()` (retry next request) instead of `None` to avoid starving the pipeline when one request gets aborted mid-scheduling:

```python
# Instead of:
if selected_req is None or selected_req.status != RequestStatus.RUNNING:
    return None

# Do:
if selected_req is None or selected_req.status != RequestStatus.RUNNING:
    # Don't waste this scheduling slot — try the next candidate
    return self.schedule_next_chunk() if self._running else None
```

> [!WARNING]  
> Add recursion depth limit or convert to iterative to avoid stack overflow with many simultaneous aborts.

---

### 🐛 BUG 6 — `_sampler_loop` marks `req.status = FINISHED_COMPLETED` before decoder callback fires (RACE)
**File**: [`pipeline_engine.py:152-154`](file:///c:/Users/emam2/tech_projects/float-fast/fast_float/sched/pipeline_engine.py#L152)

```python
await self.latent_queue.put(task)

if task.is_last:
    req.status = RequestStatus.FINISHED_COMPLETED  # ← set BEFORE decode
```

The decoder loop checks `req.status == FINISHED_ABORTED` to skip decode. But when status transitions to `FINISHED_COMPLETED` in the sampler loop *before* the decoder has processed the last task, and separately abort arrives, the decoder check `req.status == FINISHED_ABORTED` will **never fire** because the status is already `FINISHED_COMPLETED`.

**Fix**: Move the `FINISHED_COMPLETED` status transition to the *decoder loop* after the callback fires for the last chunk:

```python
# In _decoder_loop, after callback dispatch:
if task.is_last:
    req = self.scheduler.get_request(task.request_id)
    if req is not None and req.status == RequestStatus.RUNNING:
        req.status = RequestStatus.FINISHED_COMPLETED
```

And remove it from `_sampler_loop`.

---

### 🐛 BUG 7 — `asyncio.to_thread` called on GPU-blocking code (DEADLOCK RISK)
**Files**: `pipeline_engine.py:102, 140, 176`

`asyncio.to_thread` dispatches to the default `ThreadPoolExecutor`. `sample_omni_chunk` and `decode_omni_chunk` both call PyTorch CUDA kernels which **do not release the GIL** for large kernel dispatches. With multiple concurrent requests, threads will block each other on the CUDA context (each CUDA call is implicitly serialized by the driver) while the asyncio loop thinks they are running in parallel.

**Fix**: Use a dedicated single-threaded executor for GPU work to avoid thread pool saturation:

```python
import concurrent.futures

self._gpu_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="float-gpu")

# Then use:
loop = asyncio.get_running_loop()
task = await loop.run_in_executor(self._gpu_executor, self.model.sample_omni_chunk, req)
```

This ensures GPU tasks are serialized correctly (one at a time on GPU) while keeping the asyncio event loop free for scheduling.

---

### 🐛 BUG 8 — `latent_queue.task_done()` not called on abort path in decoder loop (QUEUE JOIN DEADLOCK)
**File**: [`pipeline_engine.py:171-173`](file:///c:/Users/emam2/tech_projects/float-fast/fast_float/sched/pipeline_engine.py#L171)

```python
if req is not None and req.status == RequestStatus.FINISHED_ABORTED:
    self.latent_queue.task_done()   # ← this IS present, looks fine
    continue
```

Wait — this is already correctly handled. ✅ No bug here.

---

### 🐛 BUG 9 — `decode_omni_chunk` calls `_decode_u8(ae, task.ref_feats, lat)` but `lat` may be 3D `[T, dim_w]` instead of `[B, T, dim_w]` (SHAPE ERROR)
**File**: [`FLOAT.py:1131`](file:///c:/Users/emam2/tech_projects/float-fast/fast_float/models/float/FLOAT.py#L1131)

In `sample_omni_chunk` line 1107:
```python
latents_out = s_r_d_t.squeeze(0) if B == 1 else s_r_d_t
```

When `B == 1`, `latents_out` is `[T, dim_w]` (2D). But `_decode_u8` calls `ae.dec(lat, ...)` which expects `[B, dim_w]` (2D frame-by-frame) or `[B, T, dim_w]`. The squeeze removes the batch dim — the decoder receives a **2D tensor** `[T, dim_w]` and interprets `T` as the batch size.

This actually coincidentally matches the decoder's expected input since `_decode_u8` says `lat [b, dim]` — so `b` = `T` (number of frames), and each frame is decoded independently. **This is correct behavior by convention** — the FLOAT decoder processes frames as a batch. ✅ Not a real bug.

---

### 🐛 BUG 10 — `prepare_omni_conditions` uses `audio.shape[1]` but audio input shape convention not enforced (CRASH)
**File**: [`FLOAT.py:967`](file:///c:/Users/emam2/tech_projects/float-fast/fast_float/models/float/FLOAT.py#L967)

```python
T_frames = int(math.floor(audio.shape[1] / self.opt.sampling_rate * self.opt.fps))
```

After `audio = audio.unsqueeze(0)` (when `audio.ndim == 1`), shape is `[1, T_samples]` — `.shape[1]` = `T_samples` ✅. But if the caller passes a 2D `[1, T_samples]` tensor, the `if audio.ndim == 1` guard is skipped and `.shape[1]` is still correct ✅. However if caller passes raw `[T_samples]` without unsqueeze happening (bug in call path), shape check is sound. **No bug.**

---

## Priority Fix Order

| Priority | Bug | Impact |
|----------|-----|--------|
| 🔴 Critical | Bug 1: `get_one_hot_emotion` missing | **Crash on every non-S2E emotion request** |
| 🔴 Critical | Bug 4: No last-chunk trimming | **Always generates extra frames** |
| 🟠 High | Bug 6: `FINISHED_COMPLETED` set too early | Race window allows abort to be silently ignored |
| 🟠 High | Bug 7: `asyncio.to_thread` for GPU | Deadlock risk under concurrent load |
| 🟡 Medium | Bug 2: Closure capture by reference | Fragile under future concurrent changes |
| 🟡 Medium | Bug 5: Schedule abort retry | Pipeline starvation on burst aborts |

---

## Architecture Improvements Still Pending

1. **Multi-GPU Decoder Workers**: `decode_omni_chunk` currently only dispatches to `rank=0`. For multi-GPU setups, the decoder worker pool should use different `dev` per worker.

2. **Back-pressure from callback**: If the user's callback is slow (e.g., writing to disk), it blocks the decoder loop thread. Add a bounded output queue between decoder and callback dispatcher.

3. **Request Prioritization**: Current round-robin is strict. Consider priority-weighted scheduling for SLA differentiation (e.g., interactive vs. batch jobs).

4. **Health / Watchdog**: If the sampler thread crashes and `_sampler_task` completes without `_running = False`, the decoder workers will wait forever on `latent_queue.get()`. Add a watchdog that restarts crashed workers.

5. **Metrics**: Add `prometheus_client` counters for `chunks_sampled_total`, `chunks_decoded_total`, `queue_depth`, `request_ttff_seconds`.

---

## Files Modified / Created

| File | Status | Notes |
|------|--------|-------|
| [`fast_float/sched/types.py`](file:///c:/Users/emam2/tech_projects/float-fast/fast_float/sched/types.py) | ✅ Created | Need to add `total_frames` + `decoded_frames` fields (Bug 4 fix) |
| [`fast_float/sched/omni_scheduler.py`](file:///c:/Users/emam2/tech_projects/float-fast/fast_float/sched/omni_scheduler.py) | ⚠️ Needs fix | Bug 5: abort retry in `schedule_next_chunk` |
| [`fast_float/sched/pipeline_engine.py`](file:///c:/Users/emam2/tech_projects/float-fast/fast_float/sched/pipeline_engine.py) | ⚠️ Needs fix | Bug 6: move COMPLETED transition; Bug 7: dedicated GPU executor |
| [`fast_float/models/float/FLOAT.py`](file:///c:/Users/emam2/tech_projects/float-fast/fast_float/models/float/FLOAT.py) | ⚠️ Needs fix | Bug 1: emotion encoding; Bug 2: closure binding; Bug 4: frame trimming |
