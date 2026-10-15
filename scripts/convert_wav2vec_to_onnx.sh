#!/bin/bash
# One-command wav2vec2 ONNX conversion (Kaggle / any Linux box with python + torch).
#   ./scripts/convert_wav2vec_to_onnx.sh [--out ...] [converter args...]
# Env overrides: OUT_DIR (default ./onnx), W2V_OUT, CKPT, W2V_DIR.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

OUT_DIR="${OUT_DIR:-./onnx}"
W2V_OUT="${W2V_OUT:-${OUT_DIR}/wav2vec_v1.onnx}"
CKPT="${CKPT:-./checkpoints/float.pth}"
W2V_DIR="${W2V_DIR:-./checkpoints/wav2vec2-base-960h}"
mkdir -p "${OUT_DIR}" checkpoints

echo "=== [1/3] weights ==="
if [ ! -f "${CKPT}" ]; then
  echo "float.pth missing -> downloading (gdown)"
  command -v gdown >/dev/null || pip install -q gdown
  gdown --id 1rvWuM12cyvNvBQNCLmG4Fr2L1rpjQBF0 -O "${CKPT}"
else
  echo "found ${CKPT}"
fi
if [ ! -d "${W2V_DIR}" ]; then
  echo "wav2vec2 model missing -> downloading (huggingface-cli)"
  command -v huggingface-cli >/dev/null || pip install -q "huggingface_hub[hf-transfer]"
  huggingface-cli download facebook/wav2vec2-base-960h \
    --local-dir "${W2V_DIR}"
else
  echo "found ${W2V_DIR}"
fi

echo "=== [2/3] python deps ==="
python -c "import torch; print('torch', torch.__version__, 'cuda=', torch.cuda.is_available())"
probe_onnx() { python -c "import onnx.external_data_helper, onnxscript, onnxruntime" 2>/dev/null; }
onnx_rt_pkg() {
  if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "onnxruntime-gpu"
  else
    echo "onnxruntime"
  fi
}
if ! probe_onnx; then
  echo "installing onnx stack (joint resolve, pin onnx<1.22)"
  pip uninstall -q -y onnx onnxscript onnx-ir onnxruntime onnxruntime-gpu 2>/dev/null || true
  pip install -q --no-cache-dir "onnx<1.22" "onnxscript==0.7.2" "$(onnx_rt_pkg)" || pip install -q --no-cache-dir "onnx<1.22" "onnxscript==0.7.2" onnxruntime
fi
if ! probe_onnx; then
  echo "retrying with force-reinstall (clears partial/corrupt installs)"
  pip uninstall -q -y onnx onnxscript onnx-ir onnxruntime onnxruntime-gpu 2>/dev/null || true
  pip install -q --no-cache-dir --force-reinstall "onnx<1.22" "onnxscript==0.7.2" "$(onnx_rt_pkg)" || pip install -q --no-cache-dir --force-reinstall "onnx<1.22" "onnxscript==0.7.2" onnxruntime
fi
echo "onnx stack versions:"
python -c "import importlib.metadata as m
for p in ('onnx','onnxscript','onnx-ir','onnxruntime','onnxruntime-gpu'):
    try: print(' ', p, m.version(p))
    except Exception: print(' ', p, 'MISSING')"
probe_onnx || { echo "ERROR: onnx stack still broken — export cannot run"; exit 1; }

echo "=== [3/3] convert: wav2vec2 (unit 2) ==="
python -m fast_float.onnx.convert_wav2vec \
  --ckpt "${CKPT}" --w2v-dir "${W2V_DIR}" --out "${W2V_OUT}" "$@"

echo "=== artifacts ==="
ls -la "${OUT_DIR}"
