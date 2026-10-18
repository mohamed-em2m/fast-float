# Builder / Runtime Sessions

Design plan for formalizing how FLOAT-Fast builds and runs per-component
accelerated backends (`eager`, `compile`, `onnx`) behind one session model.

Status: **in progress**. Landed: specs + registry (Phase 0), registry-driven
`apply_backends` (Phase 1a), both ONNX runner families under
`fast_float/runtimes/onnx/` on `RuntimeSession` (Phase 2), the dynamo
runtime + builder (Phase 3), and `builders/onnx` + its fingerprint cache
(Phase 4 core). Remaining: ctor wiring unification (1b), pointing the selector
at builders (4 tail), native TRT + `inspect` CLI (5).

## 1. Why

Today three concerns are entangled and wired inconsistently:

- **Build** (on the target box, heavy): `fast_float/onnx/convert_*.py` produce
  ONNX graphs; `fast_float/onnx/export.py` builds ORT/TRT sessions;
  `fast_float/onnx/select_backends.py` benches and writes the config.
- **Wiring**: `emotion`/`audio`/`fmt` self-wire inside their constructors
  (`condition_encoders.py`, `FMT.py`); `direction`/`refenc`/`decoder` are wired
  post-load by `apply_backends` (`onnx/dispatch.py`). Two ONNX runner families
  exist (`_OnnxRunner` in `dispatch.py`, `*OnnxBackend` in `onnx/runtime.py`).
- **Run**: per-call forwarding scattered across those classes.

This produces duplicated logic, inconsistent device/fallback handling, and dead
code (e.g. `from fast_float.onnx.backends import get_component` in
`condition_encoders.py:60,181` — that symbol does not exist).

Goal: one **Spec → Builder → Runtime** model, per component, with the system's
six real profiles (below) expressed as data.

## 2. Layers

| Layer | Role | Today | Target |
|---|---|---|---|
| **Spec** | declarative what-to-build/run | scattered kwargs | `fast_float/builders/spec.py` |
| **Builder** | build-time, heavy, target box → artifact | `convert_*.py`, `export.py`, `select_backends.py` | `fast_float/builders/{onnx,dynamo,tensorrt}/` |
| **Runtime** | load-time, per-device session serving forward | `dispatch.py`, `onnx/runtime.py` | `fast_float/runtimes/{onnx,dynamo,eager}/` |

Builders and Runtimes are keyed by component, because the six components differ
too much to share one implementation.

## 3. Component matrix (source of truth)

| Component | Torch target | Call site | IO / dynamics | Batch | Hot | Replicate | Tolerance | Wired in |
|---|---|---|---|---|---|---|---|---|
| `emotion` | `Audio2Emotion._predict_body` | `predict_emotion` | `[1,N]` N∈[1.6e4,1.6e5] → `[1,7]` | single | no | primary | atol/rtol 2e-4 | ctor |
| `audio` | `AudioEncoder._inference_body` | `inference` | wav `[1,T]` / conv_feats `[1,F,512]` F∈[10,2000] → `[1,frame,512]` | single | yes | primary | 2e-4 | ctor |
| `direction` | `ae.dec.direction` | `encode_identity_into_motion` | `[B,20]` → `[B,512]` | dynamic | no | primary | 2e-4 | dispatch |
| `fmt` | `FlowMatchingTransformer` step | `_forward_step` / `forward_with_cfv` | 7 tensors: x/wa `[1,clip,512]`, wr `[1,1,512]`, we `[1,1,7]`, prev_* `[1,prev,512]`; CFG **b=3** | fixed | **yes (ODE loop)** | primary | 2e-4 | ctor |
| `decoder` | `ae.dec` (`Synthesis`) | `_decode_call` → `_decode_u8` | latent `[m,512]` + feats → `[m,3,H,W]`; **static B baked** | micro-batch 1..16 | **yes** | **per-GPU** | cosine ≥ 0.995, mean_abs ≤ 0.05 | dispatch |
| `refenc` | `ae.enc` (exposes `.fc`) | `encode_image_into_latent` | `[1,3,512,512]` → `(x_r, lambda, *feats)` | single | no | primary | 2e-4 | dispatch |

Consequences that shape the architecture:

- Only `decoder` replicates per GPU; everything else is primary-only. Encode the
  `replicable` flag as registry data (today it is implicit in
  `_get_ae`/`_decoder_for`).
- `fmt` exports **only the single vector-field step**; the ODE loop, CFG combine
  and chunk autoregression stay in Python (`FMT._forward_step`). CFG `b=3` is the
  hot shape.
- `decoder` bakes `groups=batch` into its convs → a **static-batch** artifact;
  build enumerates candidate batches and runtime pads/trims (`_decode_u8`).
- `emotion` and `fmt` have **hard** TRT profile bounds (min length 16000;
  fixed clip). Exceeding them is an ORT hard error; only `fmt` currently catches
  it (`_forward_step` sets `_onnx_backend=None` → **sticky** degradation).
- `refenc`/`direction` need a call site that reaches into the module (`.fc`) →
  the runtime wrapper must forward attributes (see `_CompiledLeaf`, `RefEncOnnx`).

## 4. Spec and artifact

```python
@dataclass(frozen=True)
class BuildSpec:
    component: str
    backend: str  # eager | compile | onnx
    source: str  # module/class locator
    sample_inputs: tuple
    input_names: tuple[str, ...]
    dynamic_axes: dict | None
    opt_shapes: dict | None  # same args the converters use today
    extra_dynamic: dict | None
    precision: str = "fp16"  # fp16 | fp32
    provider_chain: tuple[str, ...] = ("trt", "cuda", "cpu")
    trt: dict | None = None  # min_batch, max_batch, profiles, timing_cache
    fixed_batch: bool = False
    batch: int | None = None  # decoder static B
    opset: int = 17
    tolerance: float = 2e-4


@dataclass(frozen=True)
class RunSpec:
    component: str
    artifact: "BuildArtifact | None"
    device_id: int | None
    provider_chain: tuple[str, ...]
    fallback_mode: str = "eager"
    tolerance: float = 2e-4
    degrade_on_error: str = "sticky"  # sticky | transient | raise


@dataclass(frozen=True)
class BuildArtifact:
    kind: str  # onnx | dynamo | engine
    path: str | None
    provider: str | None
    precision: str
    batch: int | None
    fingerprint: str
    metrics: dict
    parity: dict | None
```

Fingerprint (reuse existing probes, don't invent new ones):

```
sha256(component, backend, module_class, ckpt_mtime+size,
       io_signature, dynamic_axes, precision,
       cc, torch, ort, trt_loadable, cuda_ep, provider_chain)
```

`env_fingerprint()` in `onnx/backends.py` already supplies `torch`, `cuda`, `cc`,
`vram_gb`, `ort`, `trt_ep_listed`, `trt_libs_loadable`, `trt`, `cuda_ep`.

Cache manifest: `onnx/.build_manifest.json`, keyed by fingerprint. The config
`target:` block already stores the same fingerprint — reuse it.

## 5. Builders (`fast_float/builders/`)

- `base_builder.py`: `BuilderSession` ABC — `build(spec) -> BuildArtifact`,
  `available() -> bool`, `fingerprint(spec) -> str`, `cache`.
- `onnx/`: wrap `convert_*.py` + `export_module` + `make_runtime_session` +
  `trt_optimization_profiles`. Two sub-steps: (a) export the graph,
  (b) optionally pre-warm the ORT-TRT engine into the TRT cache dir.
- `dynamo/`: wrap `torch.compile` (inductor cache dir) and/or `torch.export` AOT.
- `tensorrt/` (later): native engine via `trtexec`/TRT Builder → `.engine`.

Builder passes `opt_shapes`/`extra_dynamic` exactly as the selector does today so
auto-generated TRT profiles (`2e98eab`) keep working.

## 6. Runtimes (`fast_float/runtimes/`)

- `base_runtime.py`: `RuntimeSession` ABC — `forward(*args)`, `device`,
  `providers`, `replicate(device_id)`, `degrade(reason)`, `stats`, `close`.
- `onnx/`: **merge both runner families** — `_OnnxRunner` +
  `DirectionOnnx/RefEncOnnx/DecoderOnnx` (from `dispatch.py`) and
  `EmotionOnnxBackend/Wav2vecOnnxBackend/FMTStepBackend` (from `onnx/runtime.py`)
  — under `RuntimeSession`, registry-driven. Keep CUDA IOBinding with the cached
  `_io_failed` host fallback, and `_apply`/`_rebind` device rebinding.
- `dynamo/`: `CompiledRuntime` (with `_CompiledLeaf` extra-attr semantics) and
  `ExportedRuntime`.
- `eager/`: always-available fallback wrapper.
- **Degradation policy as data** (`RunSpec.degrade_on_error`): today `fmt` is
  sticky, dispatch runners are per-call. Unify.

Invariants every runtime must honor:

- device binding at construction (`device_id`); `replicate(device_id)` for the
  `replicable` components (decoder).
- host/dtype normalization: numpy fp32 host path; outputs `.to(device, fp32)`.
- never raise to the pipeline: on failure, follow `degrade_on_error`.
- parity is a **build-time** gate (converter/selector); runtime trusts config.

## 7. Integration seam — one registry

Add `fast_float/onnx/registry.py`: `component -> (target locator, spec factory,
runtime factory, install hook, replicable)`. Then:

- `apply_backends(model, opt)` becomes a loop over the registry (name/behavior
  preserved: default all-eager is a no-op; never raises).
- Model constructors stop self-wiring (emotion/audio/fmt delegate to the
  registry), removing the ctor-vs-post-load split and the `get_component` dead
  imports.

## 8. Build-on-miss policy

A served box must not compile at startup. On cache miss:

- `dynamo` — may compile on miss.
- `onnx` / `trt` — miss → `eager` + warning (building is an offline step).

## 9. Tooling

- `python -m fast_float.builders build --components ... --backend onnx|dynamo --out-dir onnx`
- `python -m fast_float.runtimes inspect` — per-component chosen runtime,
  providers, device, degradations.
- Refactor `select_backends.py` to consume builders/runtimes instead of
  `make_session_from_args` directly.

## 10. Phases (behavior-preserving, risk-ordered)

0. **Reconcile** — done: `fast_float/builders/` + `fast_float/runtimes/` are the
   layout; `spec.py` + `registry.py` hold the §3 matrix as data.
1. **Unify wiring** into the registry — partly done: `apply_backends` is
   registry-driven (1a) and the `get_component` dead imports are fixed;
   emotion/audio/fmt still self-wire in their constructors (1b, open).
2. **Promote runners** — done: both families live in `fast_float/runtimes/onnx/`
   under `RuntimeSession`; `dispatch.py` is the thin installer and
   `onnx/runtime.py` is a re-export shim.
3. **`runtimes/dynamo/`** — done: `CompiledRuntime` (+ `builders/dynamo`).
   `ExportedRuntime` (torch.export AOT) still open.
4. **Spec/Artifact/Fingerprint/cache** — core done: `builders/onnx.OnnxBuilder`
   wraps `export_module` and records into `ArtifactCache`. Pointing the selector
   at builders is still open.
5. **`inspect` CLI + startup degradation warnings** — done:
   `python -m fast_float.runtimes inspect` prints the resolved per-component
   plan and where the box cannot honour it; `inspect_model`/`warn_degraded`
   report live sessions at startup. Every `RuntimeSession` subclass's
   ``forward`` auto-reports a ``runtime:<component>`` stage into
   `utils/timing.py` (no-op when no timing run is open) and keeps
   `RuntimeStats` (calls/errors, plus io_bindings at the IOBinding sites).
   Still open: native TRT builder (ORT-TRT first).

- Cross-cutting: add `tests/` for spec/fingerprint/registry (pure-CPU — the only
  locally runnable pieces; the repo has no `tests/` today).

## 11. System-specific risks

- Ctor self-wiring builds before device placement; FMT's `alignment_mask` is
  created on `opt.rank` (`FMT.py:289`). Registry install-after-load avoids the
  mismatch.
- `decoder` static batch: one artifact vs a batch set (selector picks one via
  `--dec-batch` today).
- `fmt` sticky degradation can hide a broken build → `inspect` warns at startup.
- Weights fingerprint cost → use ckpt mtime+size, not a content hash.
- Real export/parity/TRT validation only on the GPU box (local lacks
  onnxruntime/TRT); watch `session.get_providers()` and `[backends:iobinding]`.
