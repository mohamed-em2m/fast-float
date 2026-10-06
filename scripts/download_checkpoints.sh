#!/bin/bash
set -e

# Resolve repo root (this script lives in scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

mkdir -p checkpoints

if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found, installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Use the venv's interpreter when available so `gdown` / `huggingface-cli` land there
if [ -x "${PROJECT_ROOT}/.venv/bin/python" ]; then
    _UV_PY="--python ${PROJECT_ROOT}/.venv/bin/python"
    GDOWN="${PROJECT_ROOT}/.venv/bin/gdown"
    HF="${PROJECT_ROOT}/.venv/bin/huggingface-cli"
else
    _UV_PY=""
    GDOWN="gdown"
    HF="huggingface-cli"
fi

# shellcheck disable=SC2086
uv pip install $_UV_PY -q gdown "huggingface_hub[cli]>=0.14.1,<1.0"

echo "=== Downloading float.pth ==="
if [ -f "checkpoints/float.pth" ]; then
    echo "checkpoints/float.pth already exists, skipping."
else
    "${GDOWN}" --id 1rvWuM12cyvNvBQNCLmG4Fr2L1rpjQBF0 -O checkpoints/float.pth
fi

echo "=== Downloading wav2vec emotion model ==="
"${HF}" download r-f/wav2vec-english-speech-emotion-recognition \
  --local-dir ./checkpoints/wav2vec-english-speech-emotion-recognition

echo "=== Downloading wav2vec2-base-960h ==="
"${HF}" download facebook/wav2vec2-base-960h \
  --local-dir ./checkpoints/wav2vec2-base-960h

echo "=== Checkpoints ready ==="
ls -la checkpoints/
