# Manual install equivalent of scripts/setup.sh (run from the repo root).
# Requires `uv` on PATH. Python version comes from .python-version,
# CUDA PyTorch pins and all deps come from pyproject.toml + uv.lock.

uv sync --frozen

NVRTC_DIR=$(dirname "$(find "$PWD/.venv/lib/python3.10/site-packages/nvidia" -name 'libnvrtc.so.13' -print -quit)")

VENV="${VIRTUAL_ENV:-$PWD/.venv}"

mapfile -t NVIDIA_LIB_DIRS < <(
    find "$VENV/lib" -type f \
        \( -name '*.so' -o -name '*.so.*' \) \
        -path '*/site-packages/nvidia/*' \
        -printf '%h\n' 2>/dev/null |
    sort -u
)

LD_LIBRARY_PATH="$NVRTC_DIR:$(IFS=:; echo "${NVIDIA_LIB_DIRS[*]}")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH


cat >> ~/.bashrc <<'EOF'
# CUDA/NVIDIA library paths
source "./scripts/setup_cuda_libs.sh"
EOF

# shellcheck source=/dev/null
source ~/.bashrc
