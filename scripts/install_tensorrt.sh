#!/usr/bin/env bash
# Install a TensorRT build that onnxruntime-gpu<1.24 can actually load.
#
# ORT 1.23 is built against TensorRT 10.9 and dlopens libnvinfer.so.10. An
# UNPINNED `pip install tensorrt` on a cu13 box resolves to TensorRT 11
# (libnvinfer.so.11): it imports fine, but ORT's provider library cannot load
# it, so the EP is silently dropped to CUDA/CPU. This script evicts any
# mismatched tensorrt tree, installs the pinned cu12 10.9 build that matches
# ORT, and then proves the EP really initialises on this box.
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

# Honest probe: a listed EP only means the wheel was built with TRT support; the
# real question is whether ORT's provider lib can load libnvinfer AND a session
# actually picks the TRT EP.
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

# Put the pip nvidia/tensorrt lib dirs on the loader path for THIS shell (a
# freshly installed TRT is otherwise invisible until the next shell).
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
add_lib_dirs

echo "--- probing before install ---"
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

echo "--- probing after install ---"
add_lib_dirs
if trt_probe; then
    echo "OK: TensorRT EP is available (engines cache under <onnx-dir>/trt_cache)."
    exit 0
fi
echo "WARN: TensorRT still not loadable. Common causes:"
echo "  - ORT build expects a different TensorRT major (try another TRT_SPEC)."
echo "  - pip nvidia libs (cudnn/cublas) missing from the loader path."
echo "Units will fall back to the CUDA EP."
exit 1
