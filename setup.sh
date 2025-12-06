#!/bin/bash
set -e

echo "=== Updating system packages ==="
sudo apt-get update -y

echo "=== Installing system dependencies ==="
sudo apt-get install -y python3.10 python3.10-distutils python3.10-venv ffmpeg git

echo "=== Ensuring uv is installed ==="
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

echo "=== Creating uv environment ==="
uv venv .venv

echo "=== Installing base requirements using uv ==="
uv pip install --python .venv/bin/python --no-cache-dir -r requirements.txt

echo "=== Installing additional packages using uv ==="
uv pip install --python .venv/bin/python IPython==9.8.0
uv pip install --python .venv/bin/python huggingface-hub==1.2.1
uv pip install --python .venv/bin/python ngrok==1.6.0 pyngrok==7.5.0 gdown==5.2.0
uv pip install --python .venv/bin/python brotli==1.2.0
uv pip install --python .venv/bin/python nest_asyncio==1.6.0

echo "=== Running checkpoint download script ==="
chmod +x download_checkpoints.sh
./download_checkpoints.sh

echo "=== Downloading test image ==="
.venv/bin/gdown "https://drive.google.com/uc?id=1E7atbSqYSgCHsH0V_oGtEN5imn2G-vxy" -O img.jpg

echo "=== Creating results directory ==="
mkdir -p results

echo "=== Setup completed successfully ==="
