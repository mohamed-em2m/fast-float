#!/bin/bash
# One-command ONNX conversion (Kaggle / any Linux box with python + torch).
#   ./scripts/convert_to_onnx.sh
# Env overrides: OUT_DIR (default ./onnx), EMO_OUT, FMT_OUT, DIR_OUT,
#   AUDIO_OUT, DEC_OUT, CKPT, EMO_DIR, W2V_DIR, DEC_BATCH, AUDIO_FRAMES.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

OUT_DIR="${OUT_DIR:-./onnx}"
EMO_OUT="${EMO_OUT:-${OUT_DIR}/emotion_v1.onnx}"
FMT_OUT="${FMT_OUT:-${OUT_DIR}/fmt_step_v1.onnx}"
DIR_OUT="${DIR_OUT:-${OUT_DIR}/direction_v1.onnx}"
AUDIO_OUT="${AUDIO_OUT:-${OUT_DIR}/audio_enc_v1.onnx}"
DEC_OUT="${DEC_OUT:-${OUT_DIR}/decoder_v1.onnx}"
CKPT="${CKPT:-./checkpoints/float.pth}"
EMO_DIR="${EMO_DIR:-./checkpoints/wav2vec-english-speech-emotion-recognition}"
W2V_DIR="${W2V_DIR:-./checkpoints/wav2vec2-base-960h}"
# Decoder bakes a STATIC batch (grouped conv); audio bakes a STATIC frame
# count T. Runtime must pad/trim to these (documented in the converters).
DEC_BATCH="${DEC_BATCH:-8}"
AUDIO_FRAMES="${AUDIO_FRAMES:-300}"
mkdir -p "${OUT_DIR}" checkpoints

echo "=== [1/7] weights ==="
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

echo "=== [2/7] python deps ==="
python -c "import torch; print('torch', torch.__version__, 'cuda=', torch.cuda.is_available())"
probe_onnx() { python -c "import onnx.external_data_helper, onnxscript, onnxruntime" 2>/dev/null; }
onnx_rt_pkg() {
  # onnxruntime>=1.24 dropped cp310 wheels; the box runs CPython 3.10.
  if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "onnxruntime-gpu<1.24"
  else
    echo "onnxruntime<1.24"
  fi
}
if ! probe_onnx; then
  echo "installing onnx stack (joint resolve, pin onnx<1.22)"
  pip uninstall -q -y onnx onnxscript onnx-ir onnxruntime onnxruntime-gpu 2>/dev/null || true
  pip install -q --no-cache-dir "onnx<1.22" "onnxscript==0.7.2" "$(onnx_rt_pkg)" || pip install -q --no-cache-dir "onnx<1.22" "onnxscript==0.7.2" "onnxruntime<1.24"
fi
if ! probe_onnx; then
  echo "retrying with force-reinstall (clears partial/corrupt installs)"
  pip uninstall -q -y onnx onnxscript onnx-ir onnxruntime onnxruntime-gpu 2>/dev/null || true
  pip install -q --no-cache-dir --force-reinstall "onnx<1.22" "onnxscript==0.7.2" "$(onnx_rt_pkg)" || pip install -q --no-cache-dir --force-reinstall "onnx<1.22" "onnxscript==0.7.2" "onnxruntime<1.24"
fi
echo "onnx stack versions:"
python -c "import importlib.metadata as m
for p in ('onnx','onnxscript','onnx-ir','onnxruntime','onnxruntime-gpu'):
    try: print(' ', p, m.version(p))
    except Exception: print(' ', p, 'MISSING')"
probe_onnx || { echo "ERROR: onnx stack still broken — export cannot run"; exit 1; }

echo "=== [3/7] convert: emotion (unit 1) ==="
python -m fast_float.onnx.convert_emotion \
  --ckpt "${CKPT}" --emo-dir "${EMO_DIR}" --out "${EMO_OUT}" "$@"

echo "=== [4/7] convert: fmt step (unit 2) ==="
python -m fast_float.onnx.convert_fmt \
  --ckpt "${CKPT}" --out "${FMT_OUT}"

echo "=== [5/7] convert: direction (unit 3) ==="
python -m fast_float.onnx.convert_direction \
  --ckpt "${CKPT}" --out "${DIR_OUT}"

echo "=== [6/7] convert: audio encoder (unit 4) ==="
python -m fast_float.onnx.convert_audio \
  --ckpt "${CKPT}" --w2v-dir "${W2V_DIR}" --out "${AUDIO_OUT}" \
  --frames "${AUDIO_FRAMES}"

echo "=== [7/7] convert: frame decoder (unit 5) ==="
python -m fast_float.onnx.convert_decoder \
  --ckpt "${CKPT}" --out "${DEC_OUT}" --batch "${DEC_BATCH}"

echo "=== artifacts ==="
ls -la "${OUT_DIR}"
