#!/bin/bash
# One-command FMT single-step ONNX conversion (Kaggle / any Linux box with python + torch).
#   ./scripts/convert_fmt_to_onnx.sh [--out ...] [converter args...]
# Env overrides: OUT_DIR (default ./onnx), FMT_OUT, CKPT.
# Structural flags (--fmt-depth, --dim-h, ...) must match the deploy opt.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# shellcheck source=/dev/null
source "${SCRIPT_DIR}/_onnx_env.sh"
use_project_venv

OUT_DIR="${OUT_DIR:-./onnx}"
FMT_OUT="${FMT_OUT:-${OUT_DIR}/fmt_step_v1.onnx}"
CKPT="${CKPT:-./checkpoints/float.pth}"
mkdir -p "${OUT_DIR}" checkpoints

echo "=== [1/3] weights ==="
if [ ! -f "${CKPT}" ]; then
  echo "float.pth missing -> downloading (gdown)"
  command -v gdown >/dev/null || pip_install gdown
  gdown_download 1rvWuM12cyvNvBQNCLmG4Fr2L1rpjQBF0 "${CKPT}"
else
  echo "found ${CKPT}"
fi

echo "=== [2/3] python deps ==="
"${PY}" -c "import torch; print('torch', torch.__version__, 'cuda=', torch.cuda.is_available())"
# Verify (and repair) the pinned onnx stack; aborts on failure (prerequisite).
ensure_onnx_stack

echo "=== [3/3] convert: FMT step (unit 2) ==="
"${PY}" -m fast_float.onnx.convert_fmt \
  --ckpt "${CKPT}" --out "${FMT_OUT}" "$@"

echo "=== artifacts ==="
ls -la "${OUT_DIR}"
