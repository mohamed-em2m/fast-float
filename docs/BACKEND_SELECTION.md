# Per-component backend selection

FLOAT's inference pipeline is split into named **components**. Each component
can run in one of two accelerated modes, with plain torch as the always-available
baseline:

| mode | what it is |
| --- | --- |
| `eager` | plain PyTorch (today's default; the fallback for everything) |
| `compile` | `torch.compile` (inductor) around the torch module |
| `onnx` | the exported ONNX graph on an execution provider (`provider: trt \| cuda \| cpu`; TensorRT preferred) |

`eager` is **not** a "mode we pick" — it is the reference and the fallback. The
converter *compares* the accelerated candidates per component and writes a
**config YAML** recording which component runs in which mode. The runtime reads
that YAML and routes accordingly.

The config is **hardware-specific** (TensorRT/CUDA availability, GPU model,
VRAM), so it is a *build artifact* produced on the target box, not a source
file we hand-edit.

## Why the config is generated, not assumed

From a real converter run (`/content/float-fast`, Tesla T4, torch `2.11.0+cu130`,
`onnxruntime-gpu 1.23.2`):

- ORT's CUDA EP needs **CUDA 12** libs (`libcublasLt.so.12`, cuDNN 9); the box
  ships **CUDA 13** (torch `cu130`). The CUDA EP **never loaded**, so every ONNX
  unit silently ran on **CPU**:
  - `decoder` **0.04x**, `refenc` **0.03-0.06x**, `audio` **0.67-0.80x** — losses
  - `direction` **3.0x**, `fmt` **1.14x** — wins
- the **TensorRT** EP was unavailable too: the box carried `libnvinfer.so.11`
  (TensorRT 11) but ORT `1.23` dlopens `libnvinfer.so.10` (built against TensorRT
  10.9) — a major-version ABI mismatch no `LD_LIBRARY_PATH` can bridge. Installing
  `tensorrt-cu12==10.9.*` makes the TRT EP loadable (see `AGENT.md` §6).

So on that box the only viable fast path for the heavy components is
`compile`; the `trt` mode is not even a candidate. A static, committed choice
would be wrong on the next box. Hence: probe, measure, write.

**Since then** the repo pins `torch 2.7.1+cu126` (torch >= 2.5.1 is required by
the dynamo exporter) and `tensorrt-cu12==10.9.*` (see `AGENT.md` §6), so the
CUDA 12 sonames ORT needs come from torch's own `nvidia-*-cu12` deps. On the
same T4 that means **both** the CUDA EP and the TensorRT EP load, all six units
export and pass parity under **TRT FP16**, and `trt` is a real candidate again.
`emotion`, `refenc`, `audio` and `fmt` need parity's functional-equivalence branch
(cosine + bounded mean error as **hard floors**, so a loose `atol` cannot skip
them) because their outputs are bounded/small-magnitude,
where FP16 drift is absolute — see `AGENT.md` §6.

## Components

| id | unit | ONNX file | call frequency | hot? |
| --- | --- | --- | --- | --- |
| `emotion` | S2E emotion classifier | `emotion_v1.onnx` | once / request | no |
| `audio` | wav2vec2 + projection | `wav2vec_v1.onnx` | once / chunk | **yes** |
| `direction` | motion direction (QR) | `direction_v1.onnx` | once / request | no |
| `fmt` | FMT vector field | `fmt_step_v1.onnx` | **NFE x chunks** | **yes** |
| `decoder` | StyleDecoder synthesis | `decoder_v1.onnx` (+ `decoder_v1_b<N>.onnx` set) | per micro-batch | **yes** |
| `refenc` | reference encoder | `refenc_v1.onnx` | once / request | no |

Only the **hot** components are worth optimizing. A 3x win on `direction`
(cold) is noise end-to-end; a 0.9x on `fmt` (hot, inside the ODE loop) is fatal.

## Why per-component results do NOT compose

Two effects mean isolated micro-benchmarks are only a **pre-filter**, never the
final answer:

1. **Device boundaries.** The ONNX backends do `tensor.to("cpu").numpy()` ->
   ORT -> `.to(device)`: a GPU->CPU->GPU round-trip per call. `fmt` is called
   *inside* the ODE loop between torch ops, so its isolated 1.14x can be a net
   loss once the transfer/sync cost is included.
2. **Loss of fusion.** `torch.compile` on submodules cannot fuse across
   component boundaries; compiling `fmt` and `decoder` separately is not the
   same as compiling a larger graph.

Therefore the decision is **validated end-to-end** in addition to per-component.

## Pipeline (`scripts/convert_to_onnx.sh` -> `select_backends`)

**Step 0 — green gate.** All six ONNX artifacts export and pass parity. Abort
selection if any *hot* unit is red.

**Step 1 — environment fingerprint -> YAML header.** Probe once:
`torch` version, CUDA availability + device name + compute capability + VRAM,
`onnxruntime` version, whether the CUDA EP is listed, whether `libnvinfer` is
loadable, TRT version. Recorded so *why* a mode was rejected is explicit.

**Step 2 — candidate pre-filter.** Drop `trt` candidates when the CUDA EP / TRT
are unusable. Only build `compile` candidates for the hot set.

**Step 3 — per-component micro-bench (pre-filter only).** Parity + warmed
median at the **deployment dtype**, including the real I/O transfer path. Also
bench an ONNX variant with **CUDA IOBinding** (device-resident) where relevant.
Record metrics + rejection reason.

**Step 4 — end-to-end integration check (the decider).** Measure real generation
latency/FPS for: (a) all-`eager`, (b) the best independent per-component pick,
(c) a small greedy search over the hot set. The chosen combo must beat
all-`eager` by a threshold (default 3%) or everything falls back to `eager`.

**Step 5 — emit YAML + wire runtime.** The runtime loads the YAML, routes each
component, logs the choice, exposes it on `/health`, supports overrides, and
defaults to all-`eager` when the YAML is missing/unreadable.

## Running the selector

`scripts/convert_to_onnx.sh` runs it automatically after all units pass (set
`SKIP_SELECT=1` to skip). Directly:

```bash
python -m fast_float.onnx.select_backends \
    --onnx-dir ./onnx --out ./onnx/backends.generated.yaml \
    [--hot-only] [--iters 30] [--min-gain 1.10] [--compile-mode default] \
    [--no-compile] [--no-onnx] [--no-global] [--dry-run] ${ORT_ARGS}
```

- `--dry-run` writes the safe all-eager config + env fingerprint only (no model
  construction) — works on a CPU-only box.
- A candidate is chosen only if it parity-passes and beats eager by
  `--min-gain` (default 10%) *and* by more than three times the measured
  run-to-run spread (MAD); otherwise the component stays `eager`. A 3% gain
  is inside the jitter band and used to reselect `torch.compile` every run.
- The whole-`Generator` compile probe is recorded under `e2e` as advisory; a
  true end-to-end objective is still validated by the app's own bench (the
  isolated medians here do **not** compose — see above).

## Config schema

Committed safe default: `config/backends.yaml` (all `eager`).
Generated, target-specific: `onnx/backends.generated.yaml` (gitignored; preferred
when present). Resolution order: explicit path -> `$FLOAT_BACKENDS_CONFIG` ->
generated -> committed default -> built-in all-`eager`.

```yaml
version: 2
target:
  gpu: Tesla T4
  cc: "7.5"
  vram_gb: 14.6
  torch: 2.11.0+cu130
  ort: 1.23.2
  cuda_ep: false
  trt: false
  trt_version: null
objective: end_to_end_ms
baseline: { mode: eager, e2e_ms: 51200, fps: 0.4 }
chosen_combo: { fmt: compile, decoder: eager, audio: eager }
e2e: { e2e_ms: 47800, speedup: 1.07, parity_ok: true }
components:
  fmt:
    mode: compile
    onnx_path: onnx/fmt_step_v1.onnx
    metrics: { eager_ms: 21.3, chosen_ms: 17.1, speedup: 1.24, parity_ok: true, compile_s: 95 }
  decoder:
    mode: eager
    reason: "compile/trt did not beat eager end-to-end"
  direction:
    mode: eager
    reason: "cold: once per request"
```

## Runtime wiring

Implemented in `fast_float/onnx/backends.py` (`resolve_mode`, `maybe_compile`)
and `fast_float/onnx/dispatch.py` (`apply_backends`).

- `OptimizedInferenceAgent` calls `apply_backends(self.G, self.opt)` right after
  the checkpoint is loaded and the model is on its device, before the global
  optimisation pass.
- `emotion`, `audio`, `fmt` are config-gated in their own modules (the ONNX
  backend is only built when the mode is `onnx`; `compile` wraps the torch body).
- `direction`, `refenc`, `decoder` are substituted by `dispatch`:
  `dec.direction` and `enc` (exposing `.fc`) are replaced, and `decoder` is
  installed as `model._decoder_onnx` / `model._decoder_compiled` and consumed by
  `FLOAT._decode_call` (which `_decode_latent_batched` and the streaming
  `_decode_u8` path both route through; static-batch pad/trim for ONNX).
- **No automatic runtime self-test / fallback**: the config is the contract.
  Parity and selection are validated only by the converter/selector scripts.
  A missing artifact still can't break startup — construction raises and is
  caught, logging a warning and leaving the component `eager`. ORT provider
  fallback (`trt → cuda → cpu`) happens *inside* `make_ort_session`.
- **The parity reference is the *deployed* model, never the traced one.** A
  graph compared against the module it was traced from proves nothing: the
  `fmt` artifact used to be exported with an inverted attention mask (a
  different model — per-step cosine 0.94) and passed every gate that way. The
  converter now traces the deployed module and gates the artifact against a
  snapshot of it; the selector's eager reference is the deployed model too.
- **Install-time self-tests** (`apply_backends`): the components it
  substitutes (`direction`, `refenc`, `decoder`) are probed against the torch
  model, and `fmt` — which wires itself in its constructor, before the
  checkpoint is loaded — is probed in the same pass, once the weights are
  there. A graph that does not reproduce the torch model is dropped and the
  component stays `eager`. `FLOAT_SKIP_BACKEND_SELFTEST=1` skips the probes.
- **A parity check must compare the same computation on both sides.** TF32
  rounds every accumulation to a 10-bit mantissa; through the decoder's
  22-conv stack that alone moves the output by ~40 uint8 LSB — more than the
  2 LSB the frame gate allows. The converter therefore pins **both** sides to
  fp32 (`make_session_from_args(..., use_tf32=False)` plus
  `cudnn.allow_tf32=False`), which makes the residual the graph's own error
  again (1 LSB measured). The runtime self-test cannot pin the provider (the
  deployed session is TF32 for speed), so it judges the provider by the mean
  (0.5 LSB = invisible) plus a cosine floor (0.999) and reports the max as
  information.
- **The decoder graph is the model.** It used to substitute a grouped conv
  (`groups=batch`); the ORT CUDA EP serves grouped convs only through a
  fallback kernel that drifts ~90 uint8 LSB from torch's and is not
  run-to-run reproducible, so the artifact could not reproduce the frames the
  eager path produces. The graph is now traced from the model itself (dense
  modulated convs, still at a fixed batch, so the artifact set and the
  runtime's pad/trim are unchanged).
- **TensorRT profiles are built automatically at runtime**, not read from the
  config: every ONNX runner goes through `export.make_runtime_session`, which
  hands the dynamic-shape graph representative TRT profiles (derived from each
  component's deployment shapes) so the TensorRT EP can build an engine and
  engage without a per-component hand-tuned profile. Without profiles TRT
  silently drops to the CUDA EP. The config `provider` key stays **advisory**
  (what won, for logging/health) — it does not gate provider selection.
  `max_batch` defaults to `$FLOAT_TRT_MAX_BATCH` (16).
- `FLOAT_DISABLE_BACKENDS=1` disables the whole dispatch pass.

## Runtime overrides

- `FLOAT_BACKENDS_CONFIG=<path>` — use a specific config file.
- `FLOAT_MODE_<COMPONENT>=eager|compile|onnx` — force one component (debug).

## Open decisions

1. **CUDA IOBinding** — implemented in `export.run_ort_iobinding` and used by
   all six ONNX runners (device-resident I/O, zero host copies) with automatic
   fallback to the numpy host path. Keep the fallback logs under watch: if a
   hot component keeps falling back, its shapes need attention.
2. **Greedy combos vs a global `torch.compile`** — add whole-model compile as one
   extra candidate in step 4. Per-component control caps fusion; accept it?
3. **Min gain threshold** and the e2e bench budget (a full generation is ~50s).
