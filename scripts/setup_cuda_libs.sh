#!/usr/bin/env bash
set -euo pipefail

# Prefer the project venv. This file is *sourced* by the launchers
# (run_all.sh / run_api.sh / environments.sh) from a plain shell where `.venv`
# is not activated; falling back to the system `python` finds no
# onnxruntime/nvidia wheels, and `set -e` then aborts the launch.
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_PROJECT_ROOT="$(cd "${_SCRIPT_DIR}/.." && pwd)"
if [[ -x "${_PROJECT_ROOT}/.venv/bin/python" ]]; then
    PY="${_PROJECT_ROOT}/.venv/bin/python"
else
    PY="${PY:-python}"
fi

# Find Python site-packages directories
mapfile -t PACKAGE_PATHS < <(
    "${PY}" -c '
import site
for p in site.getsitepackages():
    print(p)
if hasattr(site, "getusersitepackages"):
    print(site.getusersitepackages())
'
)

NVIDIA_LIB_PATHS=()

# Search NVIDIA and TensorRT packages for shared libraries. ORT's CUDA EP
# needs libcublas/libcudnn and the TensorRT EP needs libnvinfer; the pip
# wheels place them under nvidia/*, tensorrt*/ and tensorrt_libs/.
for BASE_PATH in "${PACKAGE_PATHS[@]}"; do
    PACKAGE_DIRS=("${BASE_PATH}/nvidia" "${BASE_PATH}/tensorrt" "${BASE_PATH}/tensorrt_libs")
    for EXTRA_DIR in "${BASE_PATH}"/tensorrt*; do
        [[ -d "${EXTRA_DIR}" ]] && PACKAGE_DIRS+=("${EXTRA_DIR}")
    done

    for PKG_DIR in "${PACKAGE_DIRS[@]}"; do
        [[ -d "${PKG_DIR}" ]] || continue
        while IFS= read -r -d '' LIB_DIR; do
            NVIDIA_LIB_PATHS+=("$LIB_DIR")
        done < <(
            find "${PKG_DIR}" -type f \
                \( -name '*.so' -o -name '*.so.*' \) \
                -printf '%h\0' |
            sort -zu
        )
    done
done

# Check common system CUDA library directories
SYSTEM_CUDA_PATHS=(
    "/usr/local/cuda/lib64"
    "/usr/lib64-nvidia"
)

for PATH_DIR in "${SYSTEM_CUDA_PATHS[@]}"; do
    if [[ -d "$PATH_DIR" ]]; then
        NVIDIA_LIB_PATHS+=("$PATH_DIR")
    fi
done

# Deduplicate while preserving the first occurrence
declare -A SEEN=()
UNIQUE_PATHS=()

for PATH_DIR in "${NVIDIA_LIB_PATHS[@]}"; do
    if [[ -z "${SEEN[$PATH_DIR]+x}" ]]; then
        SEEN["$PATH_DIR"]=1
        UNIQUE_PATHS+=("$PATH_DIR")
    fi
done

# Prepend discovered paths to the existing environment
NEW_LD_PATH="$(IFS=:; echo "${UNIQUE_PATHS[*]}")"

if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
    NEW_LD_PATH="${NEW_LD_PATH}:${LD_LIBRARY_PATH}"
fi

export LD_LIBRARY_PATH="$NEW_LD_PATH"

echo "Discovered and added CUDA/NVIDIA library paths:"
printf ' - %s\n' "${UNIQUE_PATHS[@]}"

echo
echo "Testing ONNX Runtime..."

"${PY}" - <<'PY'
try:
    import onnxruntime as ort
    print("Available Execution Providers:", ort.get_available_providers())
except Exception as e:
    print("Failed to test ONNX Runtime:", e)
    raise
PY
