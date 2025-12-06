#!/bin/bash
set -e

echo "=== Updating system packages ==="
sudo apt-get update -y

echo "=== Installing Python and system dependencies ==="
sudo apt-get install -y python3.10 python3.10-distutils python3.10-venv python3-pip ffmpeg git

echo "=== Creating Python virtual environment ==="
python3.10 -m venv .venv

echo "=== Activating virtual environment ==="
source .venv/bin/activate

echo "=== Upgrading pip ==="
pip install --upgrade pip

echo "=== Installing base requirements ==="
pip install -r requirements.txt



echo "=== Running checkpoint download script ==="
chmod +x download_checkpoints.sh
./download_checkpoints.sh

echo "=== Downloading test image ==="
gdown "https://drive.google.com/uc?id=1E7atbSqYSgCHsH0V_oGtEN5imn2G-vxy" -O img.jpg

echo "=== Creating results directory ==="
mkdir -p results

echo "=== Setup completed successfully ==="
