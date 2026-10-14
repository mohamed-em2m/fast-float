# Manual install equivalent of scripts/setup.sh (run from the repo root).
# Requires `uv` on PATH. Python version comes from .python-version,
# CUDA PyTorch pins and all deps come from pyproject.toml + uv.lock.
uv sync --frozen

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd .. && pwd)"
NVRTC_DIR=$(dirname "$(find "$PWD/.venv/lib/python3.10/site-packages/nvidia" -name 'libnvrtc.so.13' -print -quit)")

export LD_LIBRARY_PATH="$NVRTC_DIR:$PWD/.venv/lib/python3.10/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"