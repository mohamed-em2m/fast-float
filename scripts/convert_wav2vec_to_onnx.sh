#!/bin/bash
# One-command wav2vec2 ONNX conversion (Kaggle / any Linux box with python + torch).
#   ./scripts/convert_wav2vec_to_onnx.sh [--out ...] [converter args...]
# Env overrides: OUT_DIR (default ./onnx), W2V_OUT, CKPT, W2V_DIR.
#
# This is the *dynamic* conv_feats graph (`wav2vec_v1.onnx`) the runtime's
# `Wav2VecModel`/`AudioEncoder` path loads; `convert_to_onnx.sh` produces the
# same artifact as its audio unit.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# shellcheck source=/dev/null
source "${SCRIPT_DIR}/_onnx_env.sh"
use_project_venv

OUT_DIR="${OUT_DIR:-./onnx}"
W2V_OUT="${W2V_OUT:-${OUT_DIR}/wav2vec_v1.onnx}"
CKPT="${CKPT:-./checkpoints/float.pth}"
W2V_DIR="${W2V_DIR:-./checkpoints/wav2vec2-base-960h}"
mkdir -p "${OUT_DIR}" checkpoints

echo "=== [1/3] weights ==="
if [ ! -f "${CKPT}" ]; then
  echo "float.pth missing -> downloading (gdown)"
  command -v gdown >/dev/null || pip_install gdown
  gdown --id 1rvWuM12cyvNvBQNCLmG4Fr2L1rpjQBF0 -O "${CKPT}"
else
  echo "found ${CKPT}"
fi
if [ ! -d "${W2V_DIR}" ]; then
  echo "wav2vec2 model missing -> downloading (huggingface-cli)"
  command -v huggingface-cli >/dev/null || pip_install "huggingface_hub[hf-transfer]"
  huggingface-cli download facebook/wav2vec2-base-960h \
    --local-dir "${W2V_DIR}"
else
  echo "found ${W2V_DIR}"
fi

echo "=== [2/3] python deps ==="
"${PY}" -c "import torch; print('torch', torch.__version__, 'cuda=', torch.cuda.is_available())"
# Verify (and repair) the pinned onnx stack; aborts on failure (prerequisite).
ensure_onnx_stack

echo "=== [3/3] convert: wav2vec2 (unit 4) ==="
"${PY}" -m fast_float.onnx.convert_wav2vec \
  --ckpt "${CKPT}" --w2v-dir "${W2V_DIR}" --out "${W2V_OUT}" "$@"

echo "=== artifacts ==="
ls -la "${OUT_DIR}"
