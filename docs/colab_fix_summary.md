# Summary for the Colab LLM agent

## What happened so far (Kaggle run trace)

1. Weights ok: `float.pth` found; emotion dir found.
2. First crash: transformers on Kaggle (>=4.4x) breaks the forked `Wav2Vec2ForSpeechClassification` because `from_pretrained`'s finalizer expects `all_tied_weights_keys` (missing). Fixed: manual load + tie_weights no-op.
3. Deps: `onnxscript` missing — script installs `onnx<1.22 + onnxscript==0.7.2 + onnxruntime-gpu<1.24`.
4. Export tripped: `onnxscript` needs `onnx.external_data_helper` removed in `onnx>=1.22`. Pin `onnx<1.22`.
5. Emotion/audio exports succeeded. ORT TensorRT warning (missing `libnvinfer.so.10`) is benign — it's just the TRT probe.
6. **ORT session crash**: `return cls.__new__(cls, *args)` — root cause was dynamo export using `external_data=True`, writing a `.data` sidecar that ORT<1.24 cannot resolve from a plain path string. **Fixed in commit `da6c4c1`**: changed to `external_data=False` + always uses abs-path in `InferenceSession`.

## Current state (pushed to `feat/onnx-supporting`, commit `da6c4c1`)

- `fast_float/onnx/export.py`: `external_data=False` in dynamo path; `make_ort_session` uses abs-path + `SessionOptions`
- All other converters unchanged.

## Prompt to paste to Colab LLM (Kaggle notebook)

```
You are fixing ONNX export on Kaggle for this repo.

Repo: https://github.com/mohamed-em2m/float-fast
Branch: feat/onnx-supporting
Workdir: /kaggle/working/float-fast

Goal: run `./scripts/convert_to_onnx.sh` and produce all 6 ONNX artifacts with parity passing.

Context:
- weights: checkpoints/float.pth and checkpoints/wav2vec-english-speech-emotion-recognition
- float.pth has NO emotion_encoder.* keys; export warns but proceeds with base emo-dir weights
- onnx 1.22.0 removed onnx.external_data_helper; onnxscript 0.7.2 needs onnx<1.22
- transformers>=4.4x breaks forked SER: script uses manual load + no-op tie_weights
- onnxruntime>=1.24 dropped cp310 wheels; script pins onnxruntime(-gpu)<1.24
- ORT TensorRT warning (libnvinfer.so.10 missing) is BENIGN — ignore it
- Previous crash `return cls.__new__(cls, *args)`: fixed (external_data=False in export.py)

Steps:
1. cd /kaggle/working/float-fast && git fetch origin && git switch feat/onnx-supporting && git pull
2. chmod +x scripts/convert_to_onnx.sh
3. ./scripts/convert_to_onnx.sh 2>&1 | tee /kaggle/working/onnx_convert.log | tail -80
4. Verify artifacts: ls -la onnx/
5. Return the tail of the log (especially parity PASS/FAIL for all 6 units and onnx stack versions)

Do not edit unrelated files. Only fix env deps if needed; the code changes are already committed.
If a unit fails parity, report the exact error. Do not skip parity checks.
```

## Known issues resolved

- `onnx.external_data_helper` removed in onnx>=1.22 → pin onnx<1.22
- ORT cls.**new** crash → external_data=False (commit da6c4c1)
- transformers from_pretrained crash → manual load + tie_weights no-op
- onnxruntime cp310 missing wheel → pin onnxruntime<1.24

## Examples

```bash
# One shot: deps -> weights -> TensorRT probe -> six units -> selector
./scripts/convert_to_onnx.sh

# TensorRT must major-match ORT (10.9 for onnxruntime-gpu 1.23)
./scripts/install_tensorrt.sh

# Verify what this box actually sees
uv run --frozen --no-sync python -c "import onnxruntime as ort; print(ort.get_available_providers())"
uv run --frozen --no-sync python -m fast_float.runtimes inspect
```

Per-unit converters (`./scripts/convert_fmt_to_onnx.sh`,
`./scripts/convert_wav2vec_to_onnx.sh`) source the same
`scripts/_onnx_env.sh` helpers, so a fix there applies to all of them. Always
launch with `uv run --frozen --no-sync`: a plain `uv run` re-syncs the venv and
trades the GPU `onnxruntime` back for the portable CPU wheel.

Run modes: [USAGE.md](USAGE.md). Export/parity rules:
[BACKEND_SELECTION.md](BACKEND_SELECTION.md).
