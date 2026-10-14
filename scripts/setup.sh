#!/bin/bash
set -e

# Always resolve paths relative to the repo root (this script lives in scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# Detect if running in Docker
if [ "${IS_IT_RUN_IN_DOCKER:-}" = "true" ]; then
    echo "=== Running in Docker environment ==="
    USE_SUDO=""
else
    echo "=== Running in local environment ==="
    if command -v sudo >/dev/null 2>&1; then
        USE_SUDO="sudo"
    else
        USE_SUDO=""
    fi
fi

echo "=== Updating system packages ==="
$USE_SUDO apt-get update -y

echo "=== Installing system dependencies ==="
$USE_SUDO apt-get install -y python3.10 ffmpeg git wget curl

echo "=== Installing uv (package manager) ==="
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "uv version: $(uv --version)"

echo "=== Syncing project environment (uv sync) ==="
echo "--- Python comes from .python-version, deps from pyproject.toml + uv.lock ---"
uv venv .venv --python 3.10 --clear
source .venv/bin/activate 
uv pip install --python .venv/bin/python -e .
uv lock

echo "=== Installing optional video codecs (torchcodec) ==="
# torchcodec is optional; install if compatible binary wheel exists for torch/cuda
uv pip install --python .venv/bin/python torchcodec || echo "⚠️ torchcodec install skipped or unavailable for current platform/PyTorch version (PyAV and FFmpeg NVENC remain available)."

echo "=== Running checkpoint download script ==="
chmod +x "${SCRIPT_DIR}/download_checkpoints.sh"
"${SCRIPT_DIR}/download_checkpoints.sh"

echo "=== Creating results/checkpoints directories ==="
mkdir -p results checkpoints

echo "=== Setup completed successfully ==="
