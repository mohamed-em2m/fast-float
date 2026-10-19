#!/bin/bash
# One-command ONNX conversion (Kaggle / any Linux box with python + torch).
#   ./scripts/convert_to_onnx.sh
# Env overrides: OUT_DIR (default ./onnx), EMO_OUT, FMT_OUT, DIR_OUT,
#   W2V_OUT, DEC_OUT, REF_OUT, CKPT, EMO_DIR, W2V_DIR, DEC_BATCH, DEC_BATCHES,
#   W2V_SECONDS.
#
# NOTE: We do NOT use set -e globally. Weights/dep sections abort on failure
# (they are prerequisites). Each converter runs independently so all 6 are
# always attempted; a summary at the end shows PASS/FAIL per unit.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# --- pin python/pip to the project venv and load the shared onnx-stack
# helpers. The per-unit scripts source scripts/_onnx_env.sh too, so the
# install traps (bare `pip` on a uv venv, the CPU onnxruntime shadowing the
# GPU build, the IR-version window) are fixed in exactly one place. ---
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/_onnx_env.sh"
use_project_venv

OUT_DIR="${OUT_DIR:-./onnx}"
EMO_OUT="${EMO_OUT:-${OUT_DIR}/emotion_v1.onnx}"
FMT_OUT="${FMT_OUT:-${OUT_DIR}/fmt_step_v1.onnx}"
DIR_OUT="${DIR_OUT:-${OUT_DIR}/direction_v1.onnx}"
W2V_OUT="${W2V_OUT:-${OUT_DIR}/wav2vec_v1.onnx}"
DEC_OUT="${DEC_OUT:-${OUT_DIR}/decoder_v1.onnx}"
REF_OUT="${REF_OUT:-${OUT_DIR}/refenc_v1.onnx}"
CKPT="${CKPT:-./checkpoints/float.pth}"
EMO_DIR="${EMO_DIR:-./checkpoints/wav2vec-english-speech-emotion-recognition}"
W2V_DIR="${W2V_DIR:-./checkpoints/wav2vec2-base-960h}"
# Decoder bakes a STATIC batch per artifact (grouped conv). Export a set
# (DEC_BATCHES) and let the runtime pick the smallest session covering a
# micro-batch; DEC_BATCH is the primary artifact (also the selector's
# --dec-batch). The wav2vec audio graph keeps batch AND frames dynamic
# (W2V_SECONDS only sizes the parity probe), so no pad/trim is needed there.
DEC_BATCH="${DEC_BATCH:-8}"
DEC_BATCHES="${DEC_BATCHES:-1,2,4,8,16}"
W2V_SECONDS="${W2V_SECONDS:-3.0}"
# Extra flags forwarded to every converter, e.g. ORT_ARGS="--no-trt" to force
# the CUDA EP, ORT_ARGS="--trt-fp32" for FP32 TensorRT, or
# ORT_ARGS="--trt-max-batch 16". Default: TensorRT FP16 engine-cached.
ORT_ARGS="${ORT_ARGS:-}"
read -r -a ORT_ARGS_ARR <<< "${ORT_ARGS}" || true
mkdir -p "${OUT_DIR}" checkpoints

echo "=== [1/8] weights ==="
if [ ! -f "${CKPT}" ]; then
  echo "float.pth missing -> downloading (gdown)"
  command -v gdown >/dev/null || pip_install gdown
  gdown --id 1rvWuM12cyvNvBQNCLmG4Fr2L1rpjQBF0 -O "${CKPT}"
else
  echo "found ${CKPT}"
fi
if [ ! -d "${EMO_DIR}" ]; then
  echo "emotion model missing -> downloading (huggingface-cli)"
  command -v huggingface-cli >/dev/null || pip_install "huggingface_hub[hf-transfer]"
  huggingface-cli download r-f/wav2vec-english-speech-emotion-recognition \
    --local-dir "${EMO_DIR}"
else
  echo "found ${EMO_DIR}"
fi
if [ ! -d "${W2V_DIR}" ]; then
  echo "wav2vec2-base missing -> downloading (huggingface-cli)"
  command -v huggingface-cli >/dev/null || pip_install "huggingface_hub[hf-transfer]"
  huggingface-cli download facebook/wav2vec2-base-960h \
    --local-dir "${W2V_DIR}"
else
  echo "found ${W2V_DIR}"
fi

echo "=== [2/8] python deps ==="
python -c "import torch; print('torch', torch.__version__, 'cuda=', torch.cuda.is_available())"
# Verify (and repair) the pinned onnx stack; aborts on failure (prerequisite).
ensure_onnx_stack

echo "=== [2.5/8] tensorRT probe ==="
if [ "$(python -c 'import torch;print(torch.cuda.is_available())' 2>/dev/null)" = "True" ]; then
  # Put the pip CUDA/TensorRT lib dirs on the loader path *before* probing:
  # probe_trt loads ORT's TRT provider lib, which dlopens libnvinfer/libcublas,
  # and a fresh python has none of them on its default path (this is why the
  # probe used to report "not available" even when TensorRT worked).
  augment_ort_libs
  if probe_trt; then
    echo "TensorRT EP available (engines cached under ${OUT_DIR}/trt_cache)"
  else
    echo "TensorRT EP NOT available -> running scripts/install_tensorrt.sh"
    PY="${PY}" bash "${SCRIPT_DIR}/install_tensorrt.sh" || true
    # The installer runs in a subshell (its exports don't reach us) and may have
    # installed new lib dirs, so re-augment before re-probing.
    augment_ort_libs
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
    --ckpt "${CKPT}" --emo-dir "${EMO_DIR}" --out "${EMO_OUT}" "${ORT_ARGS_ARR[@]}"

run_unit "[4/8] convert: fmt step (unit 2)" \
  python -m fast_float.onnx.convert_fmt \
    --ckpt "${CKPT}" --out "${FMT_OUT}" "${ORT_ARGS_ARR[@]}"

run_unit "[5/8] convert: direction (unit 3)" \
  python -m fast_float.onnx.convert_direction \
    --ckpt "${CKPT}" --out "${DIR_OUT}" "${ORT_ARGS_ARR[@]}"

run_unit "[6/8] convert: audio encoder (unit 4)" \
  python -m fast_float.onnx.convert_wav2vec \
    --ckpt "${CKPT}" --w2v-dir "${W2V_DIR}" --out "${W2V_OUT}" \
    --seconds "${W2V_SECONDS}" "${ORT_ARGS_ARR[@]}"

run_unit "[7/8] convert: frame decoder (unit 5)" \
  python -m fast_float.onnx.convert_decoder \
    --ckpt "${CKPT}" --out "${DEC_OUT}" --batch "${DEC_BATCH}" \
    --batches "${DEC_BATCHES}" "${ORT_ARGS_ARR[@]}"

run_unit "[8/8] convert: reference encoder (unit 6)" \
  python -m fast_float.onnx.convert_refenc \
    --ckpt "${CKPT}" --out "${REF_OUT}" "${ORT_ARGS_ARR[@]}"

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

  # Per-component backend selection: compares eager vs torch.compile vs ONNX
  # and writes onnx/backends.generated.yaml (read at runtime). Set
  # SKIP_SELECT=1 to skip and let the runtime default to all-eager.
  echo ""
  echo "=== [select] per-component backend selection ==="
  if [ "${SKIP_SELECT:-0}" = "1" ]; then
    echo "SKIP_SELECT=1 -> skipping (runtime defaults to all-eager)"
  else
    python -m fast_float.onnx.select_backends \
      --onnx-dir "${OUT_DIR}" --out "${OUT_DIR}/backends.generated.yaml" \
      --ckpt "${CKPT}" --emo-dir "${EMO_DIR}" --w2v-dir "${W2V_DIR}" \
      --dec-batch "${DEC_BATCH}" --seconds "${W2V_SECONDS}" \
      "${ORT_ARGS_ARR[@]}" || echo "[WARN] backend selection failed -> runtime keeps all-eager"
  fi
fi
