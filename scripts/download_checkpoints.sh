#!/bin/bash
set -e

# Resolve repo root (this script lives in scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

mkdir -p checkpoints

# gdown and hf are project dependencies (uv sync) — just locate them.
if [ -x "${PROJECT_ROOT}/.venv/bin/gdown" ]; then
    GDOWN="${PROJECT_ROOT}/.venv/bin/gdown"
    HF="${PROJECT_ROOT}/.venv/bin/hf"
elif [ -x "${PROJECT_ROOT}/.venv/Scripts/gdown.exe" ]; then
    GDOWN="${PROJECT_ROOT}/.venv/Scripts/gdown.exe"
    HF="${PROJECT_ROOT}/.venv/Scripts/hf.exe"
else
    echo "No .venv found — running 'uv sync --frozen' first."
    uv sync --frozen
    GDOWN="${PROJECT_ROOT}/.venv/bin/gdown"
    HF="${PROJECT_ROOT}/.venv/bin/hf"
fi

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
