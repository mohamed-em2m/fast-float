#!/bin/bash
set -e

# Detect if running in Docker
if  [ -n "$IS_IT_RUN_IN_DOCKER" = "true" ] ; then
    echo "=== Running in Docker environment ==="
    USE_SUDO=""
else
    echo "=== Running in local environment ==="
    USE_SUDO="sudo"
fi

echo "=== Updating system packages ==="
$USE_SUDO apt-get update -y

echo "=== Installing Python and system dependencies ==="
$USE_SUDO apt-get install -y python3.10 python3.10-distutils python3.10-venv python3-pip ffmpeg git wget curl

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

echo "=== Creating results directory ==="
mkdir -p results

echo "=== Setup completed successfully ==="