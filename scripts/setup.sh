#!/bin/bash

# --- Automatically inject NVIDIA & CUDA library paths ---
export LD_LIBRARY_PATH="/usr/local/lib/python3.13/dist-packages/nvidia/cublas/lib:/usr/local/lib/python3.13/dist-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.13/dist-packages/nvidia/cudnn/lib:/usr/lib64-nvidia:/usr/local/cuda/lib64:$LD_LIBRARY_PATH"

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

echo "=== Installing ONNX export toolchain (onnxscript / ONNX Runtime / TensorRT) ==="
# onnx + onnxruntime come from pyproject (CPU runtime; the web app uses it). Add
# onnxscript (required by torch.export's dynamo ONNX exporter) and, on a CUDA
# box, onnxruntime-gpu + TensorRT so scripts/convert_to_onnx.sh can use the
# TensorRT EP (FP16). Best-effort: failures here don't block the web app.
uv pip install --python .venv/bin/python --no-cache-dir onnxscript \
    || echo "⚠️ onnxscript install failed — ONNX export will need it."
HAS_CUDA="$(python -c 'import torch;print(torch.cuda.is_available())' 2>/dev/null || echo False)"
if [ "${HAS_CUDA}" = "True" ]; then
    echo "CUDA detected → installing onnxruntime-gpu<1.24 + tensorrt (best effort)"
    uv pip install --python .venv/bin/python --no-cache-dir "onnxruntime-gpu<1.24" \
        || echo "⚠️ onnxruntime-gpu install skipped — the CUDA EP will be unavailable."
    uv pip install --python .venv/bin/python --no-cache-dir tensorrt \
        || echo "⚠️ tensorrt install skipped — the TensorRT EP will be unavailable."
else
    echo "No CUDA detected → CPU ONNX Runtime only."
fi

echo "=== ONNX / ONNX Runtime / TensorRT stack ==="
python - <<'PY' || true
import importlib.metadata as m
try:
    import torch
    print("  torch cuda:", torch.cuda.is_available())
except Exception as e:
    print("  torch import failed:", e)
for p in ("onnx", "onnxscript", "onnx-ir", "onnxruntime", "onnxruntime-gpu", "tensorrt"):
    try:
        print(f"  {p}: {m.version(p)}")
    except Exception:
        pass
try:
    import onnxruntime as ort
    print("  ORT providers:", ort.get_available_providers())
except Exception as e:
    print("  onnxruntime import failed:", e)
PY

echo "=== Running checkpoint download script ==="
chmod +x "${SCRIPT_DIR}/download_checkpoints.sh"
"${SCRIPT_DIR}/download_checkpoints.sh"

echo "=== Creating results/checkpoints directories ==="
mkdir -p results checkpoints

echo "=== Setup completed successfully ==="
