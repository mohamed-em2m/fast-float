#!/bin/bash
# One-command ONNX conversion (Kaggle / any Linux box with python + torch).
#   ./scripts/convert_to_onnx.sh [--out ...] [converter args...]
# Env overrides: OUT_DIR (default ./onnx), EMO_OUT, CKPT, EMO_DIR.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

OUT_DIR="${OUT_DIR:-./onnx}"
EMO_OUT="${EMO_OUT:-${OUT_DIR}/emotion_v1.onnx}"
CKPT="${CKPT:-./checkpoints/float.pth}"
EMO_DIR="${EMO_DIR:-./checkpoints/wav2vec-english-speech-emotion-recognition}"
mkdir -p "${OUT_DIR}" checkpoints

echo "=== [1/3] weights ==="
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

echo "=== [2/3] python deps ==="
python -c "import torch; print('torch', torch.__version__, 'cuda=', torch.cuda.is_available())"
if ! python -c "import onnx, onnxruntime" 2>/dev/null; then
  echo "installing onnx + onnxruntime"
  if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    pip install -q onnx "onnxruntime-gpu" || pip install -q onnx onnxruntime
  else
    pip install -q onnx onnxruntime
  fi
fi

echo "=== [3/3] convert: emotion (unit 1) ==="
python -m fast_float.onnx.convert_emotion \
  --ckpt "${CKPT}" --emo-dir "${EMO_DIR}" --out "${EMO_OUT}" "$@"

echo "=== artifacts ==="
ls -la "${OUT_DIR}"
