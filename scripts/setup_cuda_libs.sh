#!/usr/bin/env bash
set -euo pipefail

# Find Python site-packages directories
mapfile -t PACKAGE_PATHS < <(
    python -c '
import site
for p in site.getsitepackages():
    print(p)
if hasattr(site, "getusersitepackages"):
    print(site.getusersitepackages())
'
)

NVIDIA_LIB_PATHS=()

# Search NVIDIA packages for shared libraries
for BASE_PATH in "${PACKAGE_PATHS[@]}"; do
    NVIDIA_DIR="${BASE_PATH}/nvidia"

    if [[ -d "$NVIDIA_DIR" ]]; then
        while IFS= read -r -d '' LIB_DIR; do
            NVIDIA_LIB_PATHS+=("$LIB_DIR")
        done < <(
            find "$NVIDIA_DIR" -type f \
                \( -name '*.so' -o -name '*.so.*' \) \
                -printf '%h\0' |
            sort -zu
        )
    fi
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

python - <<'PY'
try:
    import onnxruntime as ort
    print("Available Execution Providers:", ort.get_available_providers())
except Exception as e:
    print("Failed to test ONNX Runtime:", e)
    raise
PY
