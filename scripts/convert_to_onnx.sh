#!/bin/bash
# One-command ONNX conversion (Kaggle / any Linux box with python + torch).
#   ./scripts/convert_to_onnx.sh
# Env overrides: OUT_DIR (default ./onnx), EMO_OUT, FMT_OUT, DIR_OUT,
#   AUDIO_OUT, DEC_OUT, REF_OUT, CKPT, EMO_DIR, W2V_DIR, DEC_BATCH, AUDIO_FRAMES.
#
# NOTE: We do NOT use set -e globally. Weights/dep sections abort on failure
# (they are prerequisites). Each converter runs independently so all 6 are
# always attempted; a summary at the end shows PASS/FAIL per unit.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

OUT_DIR="${OUT_DIR:-./onnx}"
EMO_OUT="${EMO_OUT:-${OUT_DIR}/emotion_v1.onnx}"
FMT_OUT="${FMT_OUT:-${OUT_DIR}/fmt_step_v1.onnx}"
DIR_OUT="${DIR_OUT:-${OUT_DIR}/direction_v1.onnx}"
AUDIO_OUT="${AUDIO_OUT:-${OUT_DIR}/audio_enc_v1.onnx}"
DEC_OUT="${DEC_OUT:-${OUT_DIR}/decoder_v1.onnx}"
REF_OUT="${REF_OUT:-${OUT_DIR}/refenc_v1.onnx}"
CKPT="${CKPT:-./checkpoints/float.pth}"
EMO_DIR="${EMO_DIR:-./checkpoints/wav2vec-english-speech-emotion-recognition}"
W2V_DIR="${W2V_DIR:-./checkpoints/wav2vec2-base-960h}"
# Decoder bakes a STATIC batch (grouped conv); audio bakes a STATIC frame
# count T. Runtime must pad/trim to these (documented in the converters).
DEC_BATCH="${DEC_BATCH:-8}"
AUDIO_FRAMES="${AUDIO_FRAMES:-300}"
# Extra flags forwarded to every converter, e.g. ORT_ARGS="--no-trt" to force
# the CUDA EP, ORT_ARGS="--trt-fp32" for FP32 TensorRT, or
# ORT_ARGS="--trt-max-batch 16". Default: TensorRT FP16 engine-cached.
ORT_ARGS="${ORT_ARGS:-}"
mkdir -p "${OUT_DIR}" checkpoints

echo "=== [1/8] weights ==="
if [ ! -f "${CKPT}" ]; then
  echo "float.pth missing -> downloading (gdown)"
  command -v gdown >/dev/null || pip install -q gdown
  gdown --id 1rvWuM12cyvNvBQNCLmG4Fr2L1rpjQBF0 -O "${CKPT}"
else
  echo "found ${CKPT}"
fi
if [ ! -d "${EMO_DIR}" ]; then
  echo "emotion model missing -> downloading (huggingface-cli)"
  command -v huggingface-cli >/dev/null || pip install -q "huggingface_hub[hf-transfer]"
  huggingface-cli download r-f/wav2vec-english-speech-emotion-recognition \
    --local-dir "${EMO_DIR}"
else
  echo "found ${EMO_DIR}"
fi
if [ ! -d "${W2V_DIR}" ]; then
  echo "wav2vec2-base missing -> downloading (huggingface-cli)"
  command -v huggingface-cli >/dev/null || pip install -q "huggingface_hub[hf-transfer]"
  huggingface-cli download facebook/wav2vec2-base-960h \
    --local-dir "${W2V_DIR}"
else
  echo "found ${W2V_DIR}"
fi

echo "=== [2/8] python deps ==="
python -c "import torch; print('torch', torch.__version__, 'cuda=', torch.cuda.is_available())"
probe_onnx() { python -c "import onnx.external_data_helper, onnxscript, onnxruntime" 2>/dev/null; }
# On CUDA boxes the ORT build must ALSO ship a working CUDA EP.
# CPU-only torch boxes always pass this probe.
probe_ort_cuda() {
  python -c "
import sys
import torch
import onnxruntime as o
ok = (not torch.cuda.is_available()
      or 'CUDAExecutionProvider' in o.get_available_providers())
sys.exit(0 if ok else 1)" 2>/dev/null
}
probe_trt() {
  python -c "
import sys
import torch
import onnxruntime as o
ok = (not torch.cuda.is_available()
      or 'TensorrtExecutionProvider' in o.get_available_providers())
sys.exit(0 if ok else 1)" 2>/dev/null
}
probe_stack() { probe_onnx && probe_ort_cuda; }
onnx_rt_pkg() {
  # onnxruntime>=1.24 dropped cp310 wheels; the box runs CPython 3.10.
  if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "onnxruntime-gpu<1.24"
  else
    echo "onnxruntime<1.24"
  fi
}
if ! probe_stack; then
  echo "installing onnx stack (pin onnx<1.22, onnxscript 0.7.2, onnxruntime<1.24)"
  pip uninstall -q -y onnx onnxscript onnx-ir onnxruntime onnxruntime-gpu 2>/dev/null || true
  pip install -q --no-cache-dir "onnx<1.22" "onnxscript==0.7.2" "$(onnx_rt_pkg)" || pip install -q --no-cache-dir "onnx<1.22" "onnxscript==0.7.2" "onnxruntime<1.24"
fi
if ! probe_stack; then
  echo "retrying with force-reinstall (clears partial/corrupt installs)"
  pip uninstall -q -y onnx onnxscript onnx-ir onnxruntime onnxruntime-gpu 2>/dev/null || true
  pip install -q --no-cache-dir --force-reinstall "onnx<1.22" "onnxscript==0.7.2" "$(onnx_rt_pkg)" || pip install -q --no-cache-dir --force-reinstall "onnx<1.22" "onnxscript==0.7.2" "onnxruntime<1.24"
fi
echo "onnx stack versions:"
python -c "import importlib.metadata as m
for p in ('onnx','onnxscript','onnx-ir','onnxruntime','onnxruntime-gpu'):
    try: print(' ', p, m.version(p))
    except Exception: print(' ', p, 'MISSING')"
probe_stack || { echo "ERROR: onnx stack still broken — export cannot run"; exit 1; }

# ORT's CUDA EP dlopens libcublas/libcudnn by soname at session init. When
# those come from pip `nvidia-*` wheels — and especially when torch bundles a
# *different* CUDA major than the onnxruntime-gpu wheel (e.g. torch cu130 with
# ORT cu12) — the loader cannot find them, ORT silently drops the GPU EP, and
# every parity check dies with "fell back to CPU". Prepend the pip nvidia lib
# dirs so libonnxruntime_providers_cuda.so resolves its CUDA deps (a no-op when
# the libs already live on the default path, e.g. torch bundles matching CUDA).
if [ "$(python -c 'import torch;print(torch.cuda.is_available())' 2>/dev/null)" = "True" ]; then
  NV_LIB_DIRS="$(python -c 'import glob,os,site; d=set();
[d.update(glob.glob(os.path.join(b,"nvidia","*","lib"))) for b in site.getsitepackages()+[site.getusersitepackages()]];
print(":".join(sorted(d)))' 2>/dev/null || true)"
  if [ -n "${NV_LIB_DIRS}" ]; then
    export LD_LIBRARY_PATH="${NV_LIB_DIRS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    echo "ORT loader path += pip nvidia libs (${NV_LIB_DIRS//:/ })"
  fi
fi

echo "=== [2.5/8] tensorRT probe ==="
if [ "$(python -c 'import torch;print(torch.cuda.is_available())' 2>/dev/null)" = "True" ]; then
  if probe_trt; then
    echo "TensorRT EP available (engines cached under ${OUT_DIR}/trt_cache)"
  else
    echo "TensorRT EP NOT available -> attempting best-effort 'pip install tensorrt'"
    pip install -q tensorrt 2>/dev/null || true
    if probe_trt; then
      echo "TensorRT EP now available"
    else
      echo "WARN: TensorRT still unavailable — units fall back to the CUDA EP"
    fi
  fi
else
  echo "no CUDA on this box — skipping TensorRT"
fi


# Per-unit runner: captures output, records PASS/FAIL, never aborts other units.
UNIT_RESULTS=()
run_unit() {
  local label="$1"; shift
  echo ""
  echo "=== ${label} ==="
  if "$@"; then
    UNIT_RESULTS+=("PASS  ${label}")
  else
    UNIT_RESULTS+=("FAIL  ${label}")
    echo "[WARN] ${label} failed — continuing with remaining units"
  fi
}

# Temporarily disable pipefail so per-unit failures don't abort the whole script.
set +e

run_unit "[3/8] convert: emotion (unit 1)" \
  python -m fast_float.onnx.convert_emotion \
    --ckpt "${CKPT}" --emo-dir "${EMO_DIR}" --out "${EMO_OUT}" "$@" ${ORT_ARGS}

run_unit "[4/8] convert: fmt step (unit 2)" \
  python -m fast_float.onnx.convert_fmt \
    --ckpt "${CKPT}" --out "${FMT_OUT}" ${ORT_ARGS}

run_unit "[5/8] convert: direction (unit 3)" \
  python -m fast_float.onnx.convert_direction \
    --ckpt "${CKPT}" --out "${DIR_OUT}" ${ORT_ARGS}

run_unit "[6/8] convert: audio encoder (unit 4)" \
  python -m fast_float.onnx.convert_audio \
    --ckpt "${CKPT}" --w2v-dir "${W2V_DIR}" --out "${AUDIO_OUT}" \
    --frames "${AUDIO_FRAMES}" ${ORT_ARGS}

run_unit "[7/8] convert: frame decoder (unit 5)" \
  python -m fast_float.onnx.convert_decoder \
    --ckpt "${CKPT}" --out "${DEC_OUT}" --batch "${DEC_BATCH}" ${ORT_ARGS}

run_unit "[8/8] convert: reference encoder (unit 6)" \
  python -m fast_float.onnx.convert_refenc \
    --ckpt "${CKPT}" --out "${REF_OUT}" ${ORT_ARGS}

set -e

echo ""
echo "=== artifacts ==="
ls -la "${OUT_DIR}" 2>/dev/null || echo "(no artifacts dir)"

echo ""
echo "=== unit summary ==="
FAILED=0
for r in "${UNIT_RESULTS[@]}"; do
  echo "  ${r}"
  [[ "${r}" == FAIL* ]] && FAILED=$((FAILED + 1))
done
if [ "${FAILED}" -gt 0 ]; then
  echo ""
  echo "ERROR: ${FAILED} unit(s) failed — see logs above"
  exit 1
else
  echo ""
  echo "All units PASSED"
fi
