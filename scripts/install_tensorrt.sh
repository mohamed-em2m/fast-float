#!/usr/bin/env bash
# Install a TensorRT build that onnxruntime-gpu<1.24 can actually load -- and,
# when the active ORT is a CPU-only build, replace it with the GPU wheel first.
#
# Two independent failures look identical from the outside ("TensorRT EP not
# available"):
#   1. the CPU `onnxruntime` wheel is imported (shadows onnxruntime-gpu): it
#      lists neither CUDA nor TensorRT, so nothing can load;
#   2. a GPU wheel is imported but ORT dlopens libnvinfer.so.10 (ORT 1.23 is
#      built against TensorRT 10.9) while an unpinned `pip install tensorrt` on
#      a cu13 box pulls TensorRT 11 (libnvinfer.so.11).
# This script fixes (1) then (2), and after each step proves the EP really
# initialises on this box rather than trusting a listed name.
#
# Usage:  bash scripts/install_tensorrt.sh
# Env:    TRT_SPEC  pip spec override (default tensorrt-cu12==10.9.0.34)
#         PY        python interpreter (default .venv/bin/python, else python)
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}" || exit 1

# --- interpreter ---
if [ -z "${PY:-}" ]; then
    if [ -x "${PROJECT_ROOT}/.venv/bin/python" ]; then
        PY="${PROJECT_ROOT}/.venv/bin/python"
    else
        PY="python"
    fi
fi

pip_install() {
    if command -v uv >/dev/null 2>&1; then
        uv pip install --python "${PY}" "$@"
    else
        "${PY}" -m pip install "$@"
    fi
}
pip_uninstall() {
    if command -v uv >/dev/null 2>&1; then
        uv pip uninstall --python "${PY}" "$@" >/dev/null 2>&1 || true
    else
        "${PY}" -m pip uninstall -y "$@" >/dev/null 2>&1 || true
    fi
}

# Put the pip nvidia/tensorrt lib dirs on the loader path for THIS shell (a
# freshly installed TRT/CUDA lib is otherwise invisible until the next shell).
add_lib_dirs() {
    local dirs
    dirs="$("${PY}" -c 'import glob, os, site
d = set()
for b in site.getsitepackages() + [site.getusersitepackages()]:
    for p in [os.path.join(b, "nvidia"), *glob.glob(os.path.join(b, "tensorrt*"))]:
        for r, _sub, fs in os.walk(p):
            if any(f.endswith((".so", ".dll")) or ".so." in f for f in fs):
                d.add(r)
print(":".join(sorted(d)))' 2>/dev/null || true)"
    if [ -n "${dirs}" ]; then
        export LD_LIBRARY_PATH="${dirs}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
        echo "ORT loader path += pip CUDA/TensorRT libs"
    fi
}

has_cuda_ep() {
    "${PY}" -c 'import onnxruntime as o, sys; sys.exit(0 if "CUDAExecutionProvider" in o.get_available_providers() else 1)' 2>/dev/null
}

# A CUDA session that comes back CPU-only means the GPU wheel is present but its
# CUDA deps are unresolvable (e.g. ORT cu12 vs torch cu13).
cuda_ep_active() {
    "${PY}" - <<'PY'
import sys
try:
    import onnxruntime as ort
except Exception:
    sys.exit(1)
from onnx import TensorProto, helper
node = helper.make_node("Identity", ["x"], ["y"])
graph = helper.make_graph(
    [node], "p",
    [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
    [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
try:
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
except Exception as e:
    print("  CUDA session creation failed:", e); sys.exit(1)
print("  active providers:", sess.get_providers())
sys.exit(0 if sess.get_providers() and sess.get_providers()[0] != "CPUExecutionProvider" else 1)
PY
}

# Honest TensorRT probe: a listed EP only means the wheel was built with TRT
# support; the real question is whether ORT's provider lib can load libnvinfer
# AND a session actually picks the TRT EP.
trt_probe() {
    "${PY}" - <<'PY'
import ctypes, os, sys
try:
    import onnxruntime as ort
except Exception as e:
    print("  onnxruntime import failed:", e); sys.exit(1)
provs = ort.get_available_providers()
if "TensorrtExecutionProvider" not in provs:
    print("  TensorRT EP not listed by this ORT build:", provs); sys.exit(1)
try:
    import tensorrt  # noqa: F401  (primes the pip lib dirs on the loader path)
except Exception:
    pass
capi = os.path.join(os.path.dirname(ort.__file__), "capi")
lib = os.path.join(
    capi,
    "onnxruntime_providers_tensorrt.dll" if sys.platform == "win32" else "libonnxruntime_providers_tensorrt.so",
)
if not os.path.exists(lib):
    print("  ORT TRT provider lib not found:", lib); sys.exit(1)
try:
    ctypes.CDLL(lib)
except OSError as e:
    print("  ORT TRT provider lib cannot load libnvinfer:", e); sys.exit(1)
from onnx import TensorProto, helper
node = helper.make_node("Identity", ["x"], ["y"])
graph = helper.make_graph(
    [node], "p",
    [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
    [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
try:
    sess = ort.InferenceSession(
        model.SerializeToString(),
        providers=["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
    )
except Exception as e:
    print("  TRT session creation failed:", e); sys.exit(1)
active = sess.get_providers()
print("  active providers:", active)
sys.exit(0 if "TensorrtExecutionProvider" in active else 1)
PY
}

echo "=== TensorRT installer ==="
echo "python: ${PY}"

if ! "${PY}" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "No CUDA GPU (torch.cuda.is_available() == False) — TensorRT is CUDA-only; nothing to do."
    exit 0
fi

ORT_VER="$("${PY}" -c 'import onnxruntime; print(onnxruntime.__version__)' 2>/dev/null || echo '?')"
echo "onnxruntime: ${ORT_VER}"
case "${ORT_VER}" in
1.23.*) : ;;
\?) echo "WARN: could not read the ORT version; proceeding with the 1.23 pin." ;;
*) echo "WARN: this script pins for ORT 1.23 (TensorRT 10.9); ORT ${ORT_VER} may need a" \
    "different TensorRT major. Override with TRT_SPEC=... if the probe below fails." ;;
esac
TRT_SPEC="${TRT_SPEC:-tensorrt-cu12==10.9.0.34}"
echo "target: ${TRT_SPEC} (libnvinfer.so.10)"

add_lib_dirs

echo "--- ORT diagnostics ---"
"${PY}" - <<'PY'
import importlib.metadata as m
try:
    import onnxruntime as ort
except Exception as e:
    print("  onnxruntime import failed:", e); raise SystemExit
print("  module:", ort.__file__)
try:
    print("  get_device():", ort.get_device())
except Exception:
    pass
print("  listed providers:", ort.get_available_providers())
for p in ("onnxruntime", "onnxruntime-gpu"):
    try:
        print(f"  {p}: {m.version(p)}")
    except Exception:
        pass
PY

# (1) A CPU `onnxruntime` wheel imported under the same module name shadows the
# GPU one and lists neither CUDA nor TensorRT. Replace it with the GPU wheel.
if ! has_cuda_ep; then
    echo "Active onnxruntime lists no CUDAExecutionProvider -> it is a CPU build"
    echo "(a GPU wheel always lists it). Replacing with onnxruntime-gpu<1.24 ..."
    pip_uninstall onnxruntime onnxruntime-gpu
    if ! pip_install --no-cache-dir "onnxruntime-gpu<1.24"; then
        echo "pinned onnxruntime-gpu<1.24 failed; trying unpinned onnxruntime-gpu."
        pip_install --no-cache-dir "onnxruntime-gpu" || {
            echo "ERROR: onnxruntime-gpu install failed; cannot enable CUDA/TensorRT."
            exit 1
        }
    fi
    add_lib_dirs
    "${PY}" -c 'import onnxruntime as o; print("  listed providers now:", o.get_available_providers())'
fi

# The GPU wheel lists CUDA, but ORT cu12 against torch cu130 leaves the CUDA EP
# unable to resolve libcublas.so.12/libcudnn.so.9 -> every session drops to CPU,
# and TensorRT cannot work either.
if has_cuda_ep; then
    if cuda_ep_active; then
        echo "CUDA EP is active."
    else
        CUDA_VER="$("${PY}" -c 'import torch; print(torch.version.cuda)' 2>/dev/null || echo '?')"
        echo "WARN: CUDAExecutionProvider is listed but no session activates it."
        echo "      Likely a CUDA-major mismatch: onnxruntime-gpu 1.23 is CUDA 12,"
        echo "      torch here is CUDA ${CUDA_VER}."
        echo "      Fix: install the cu12 runtime libs, e.g."
        echo "        uv pip install nvidia-cublas-cu12 nvidia-cudnn-cu12 nvidia-cuda-runtime-cu12"
        echo "      or use an onnxruntime-gpu built for this torch's CUDA major."
    fi
fi

echo "--- probing TensorRT before install ---"
if trt_probe; then
    echo "TensorRT EP already works — nothing to install."
    exit 0
fi

echo "--- evicting any mismatched tensorrt tree ---"
pip_uninstall \
    tensorrt tensorrt-cu12 tensorrt-cu13 \
    tensorrt-bindings tensorrt-libs \
    tensorrt-cu12-bindings tensorrt-cu12-libs \
    tensorrt-cu13-bindings tensorrt-cu13-libs

echo "--- installing ${TRT_SPEC} ---"
if ! pip_install --no-cache-dir "${TRT_SPEC}"; then
    echo "pinned ${TRT_SPEC} failed; trying unpinned tensorrt-cu12 (verify the probe below)."
    pip_install --no-cache-dir "tensorrt-cu12" || {
        echo "ERROR: TensorRT install failed."
        exit 1
    }
fi

echo "--- probing TensorRT after install ---"
add_lib_dirs
if trt_probe; then
    echo "OK: TensorRT EP is available (engines cache under <onnx-dir>/trt_cache)."
    exit 0
fi
echo "WARN: TensorRT still not loadable. Common causes:"
echo "  - the active ORT is still not a GPU build (see the diagnostics above);"
echo "  - the CUDA EP itself does not activate (CUDA-major mismatch; see above);"
echo "  - ORT expects a different TensorRT major (try another TRT_SPEC)."
echo "Units will fall back to the CUDA/CPU EP."
exit 1
