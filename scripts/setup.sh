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
# shellcheck source=/dev/null
source .venv/bin/activate
uv pip install --python .venv/bin/python -e .
uv lock

echo "=== Installing optional video codecs (torchcodec) ==="
# torchcodec is optional; install if compatible binary wheel exists for torch/cuda
uv pip install --python .venv/bin/python torchcodec || echo "⚠️ torchcodec install skipped or unavailable for current platform/PyTorch version (PyAV and FFmpeg NVENC remain available)."

echo "=== Installing ONNX export toolchain (onnxscript / ONNX Runtime / TensorRT) ==="
# onnx + onnxruntime come from pyproject (a CPU wheel; the web app uses it).
# Add onnxscript (required by torch.export's dynamo ONNX exporter). On a CUDA
# box the CPU onnxruntime wheel must be *replaced* by onnxruntime-gpu -- both
# provide the ``onnxruntime`` module, and layering one over the other leaves a
# mixed tree where the CUDA EP silently never initializes.
uv pip install --python .venv/bin/python --no-cache-dir onnxscript \
    || echo "⚠️ onnxscript install failed — ONNX export will need it."
HAS_CUDA="$(python -c 'import torch;print(torch.cuda.is_available())' 2>/dev/null || echo False)"
if [ "${HAS_CUDA}" = "True" ]; then
    echo "CUDA detected → onnxruntime-gpu<1.24 + TensorRT (CUDA 12); best effort"
    # Replace, never overlay: uninstall the CPU wheel first.
    uv pip uninstall --python .venv/bin/python onnxruntime 2>/dev/null || true
    uv pip install --python .venv/bin/python --no-cache-dir "onnxruntime-gpu<1.24" \
        || echo "⚠️ onnxruntime-gpu install failed — the CUDA EP will be unavailable."
    # TensorRT for CUDA 12: the versioned metapackage pulls the matching
    # tensorrt_libs; fall back to the unversioned wheel (older layout).
    if ! uv pip install --python .venv/bin/python --no-cache-dir "tensorrt-cu12"; then
        uv pip install --python .venv/bin/python --no-cache-dir tensorrt \
            || echo "⚠️ tensorrt install skipped — the TensorRT EP will be unavailable."
    fi
    # ORT dlopens libnvinfer/libcublas by soname; put the pip lib dirs on the
    # loader path for this shell (the app repeats this at runtime, see
    # fast_float/initialize_env.py).
    NV_LIB_DIRS="$(python -c 'import glob,os,site; d=set()
for b in site.getsitepackages()+[site.getusersitepackages()]:
    for p in [os.path.join(b,"nvidia"), *glob.glob(os.path.join(b,"tensorrt*"))]:
        for r,_,fs in os.walk(p):
            if any(f.endswith((".so",".dll")) or ".so." in f for f in fs): d.add(r)
print(":".join(sorted(d)))' 2>/dev/null || true)"
    if [ -n "${NV_LIB_DIRS}" ]; then
        export LD_LIBRARY_PATH="${NV_LIB_DIRS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
        echo "ORT loader path += pip CUDA/TensorRT libs"
    fi
else
    echo "No CUDA detected → CPU ONNX Runtime only."
fi

echo "=== ONNX / ONNX Runtime / TensorRT stack ==="
python - <<'PY' || true
import ctypes
import importlib.metadata as m

for p in ("onnx", "onnxscript", "onnx-ir", "onnxruntime", "onnxruntime-gpu", "tensorrt", "tensorrt-cu12"):
    try:
        print(f"  {p}: {m.version(p)}")
    except Exception:
        pass
try:
    import torch
    print("  torch cuda:", torch.cuda.is_available(), torch.__version__)
except Exception as e:
    print("  torch import failed:", e)

try:
    import onnxruntime as ort
    provs = ort.get_available_providers()
    print("  ORT providers (listed):", provs)

    # A listed CUDA EP can still fail to load (e.g. ORT cu12 wheel vs torch
    # cu13); only a real session shows what actually initializes.
    from onnx import TensorProto, helper
    node = helper.make_node("Identity", ["x"], ["y"])
    graph = helper.make_graph(
        [node],
        "probe",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    sess = ort.InferenceSession(
        model.SerializeToString(),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    active = sess.get_providers()
    print("  CUDA EP effective:", active)
    if torch.cuda.is_available() and active[0] == "CPUExecutionProvider":
        print("  ⚠️ CUDA EP dropped to CPU — onnxruntime-gpu CUDA major != torch CUDA.")
        print("     Install an onnxruntime-gpu built for torch's CUDA major.")

    trt_listed = "TensorrtExecutionProvider" in provs
    trt_ok = False
    try:
        import tensorrt  # noqa: F401
        trt_ok = True
    except Exception:
        for so in ("libnvinfer.so.10", "libnvinfer.so.9", "libnvinfer.so"):
            try:
                ctypes.CDLL(so)
                trt_ok = True
                break
            except OSError:
                continue
    print(f"  TensorRT EP listed: {trt_listed} | libnvinfer loadable: {trt_ok}")
    if trt_listed and not trt_ok:
        print("  ⚠️ TensorRT EP listed but libnvinfer missing: pip install 'tensorrt-cu12',")
        print("     then re-run: source scripts/setup_cuda_libs.sh")
except Exception as e:
    print("  onnxruntime import failed:", e)
PY

echo "=== Running checkpoint download script ==="
chmod +x "${SCRIPT_DIR}/download_checkpoints.sh"
"${SCRIPT_DIR}/download_checkpoints.sh"

echo "=== Creating results/checkpoints directories ==="
mkdir -p results checkpoints

echo "=== Setup completed successfully ==="
