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

echo "=== Installing Python and system dependencies ==="
$USE_SUDO apt-get install -y python3.10 ffmpeg git wget curl

echo "=== Installing uv (package manager) ==="
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "uv version: $(uv --version)"

echo "=== Creating Python virtual environment with uv ==="
if [ ! -d ".venv" ]; then
    uv venv --python 3.10 .venv
else
    echo ".venv already exists, reusing it."
fi

echo "=== Installing CUDA PyTorch (cu118, must come before requirements.txt) ==="
uv pip install --python .venv/bin/python torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

echo "=== Installing base requirements ==="
uv pip install --python .venv/bin/python -r requirements.txt

echo "=== Running checkpoint download script ==="
chmod +x "${SCRIPT_DIR}/download_checkpoints.sh"
"${SCRIPT_DIR}/download_checkpoints.sh"

echo "=== Creating results/checkpoints directories ==="
mkdir -p results checkpoints

echo "=== Setup completed successfully ==="
