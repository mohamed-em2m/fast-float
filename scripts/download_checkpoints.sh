#!/bin/bash
set -e

# Resolve repo root (this script lives in scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# Shared helpers: the gdown flag and the huggingface CLI were both renamed
# between majors, so the download step must not assume one spelling.
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/_onnx_env.sh"
use_project_venv

mkdir -p checkpoints

# gdown and the huggingface CLI are project dependencies (uv sync) — just
# locate gdown; the CLI itself is probed by the shared helper (it was renamed
# from huggingface-cli to hf).
if [ ! -x "${GDOWN:-}" ] && [ -x "${PROJECT_ROOT}/.venv/bin/gdown" ]; then
    GDOWN="${PROJECT_ROOT}/.venv/bin/gdown"
fi
if [ ! -x "${GDOWN:-}" ]; then
    echo "No .venv found — running 'uv sync --frozen' first."
    uv sync --frozen
    GDOWN="${PROJECT_ROOT}/.venv/bin/gdown"
fi
export GDOWN

echo "=== Downloading float.pth ==="
if [ -f "checkpoints/float.pth" ]; then
    echo "checkpoints/float.pth already exists, skipping."
else
    gdown_download 1rvWuM12cyvNvBQNCLmG4Fr2L1rpjQBF0 checkpoints/float.pth
fi

echo "=== Downloading wav2vec emotion model ==="
hf_download r-f/wav2vec-english-speech-emotion-recognition \
  ./checkpoints/wav2vec-english-speech-emotion-recognition

echo "=== Downloading wav2vec2-base-960h ==="
hf_download facebook/wav2vec2-base-960h \
  ./checkpoints/wav2vec2-base-960h

echo "=== Checkpoints ready ==="
ls -la checkpoints/
