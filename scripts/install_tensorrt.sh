#!/usr/bin/env bash
# Install a TensorRT build that onnxruntime-gpu<1.24 can actually load -- and
# fix the surrounding GPU stack that otherwise makes a correct TensorRT install
# look broken.
#
# Four independent failures all surface as "TensorRT EP not available":
#   1. the CPU `onnxruntime` wheel is imported (shadows onnxruntime-gpu): it
#      lists neither CUDA nor TensorRT, so nothing can load;
#   2. a GPU wheel is imported but its CUDA EP cannot resolve the cu12 sonames
#      it was built against (libcublas.so.12, libcudnn.so.9, libcufft.so.11,
#      ...) because the box's torch is cu118/cu13 -> sessions silently drop to
#      CPU and TensorRT cannot initialise either;
#   3. once LD_LIBRARY_PATH is exported for (2), the loader stops honouring the
#      interpreter's DT_RPATH, so /usr/lib64-nvidia/libcuda.so.1 becomes
#      unresolvable -> every CUDA call fails with error 35 ("CUDA driver
#      version is insufficient for CUDA runtime version");
#   4. a GPU wheel is imported but ORT dlopens libnvinfer.so.10 (ORT 1.23 is
#      built against TensorRT 10.9) while an unpinned `pip install tensorrt` on
#      a cu13 box pulls TensorRT 11 (libnvinfer.so.11).
# This script fixes (1)-(3) then installs (4), and after each step proves the EP
# really initialises on this box rather than trusting a listed name.
#
# Usage:  bash scripts/install_tensorrt.sh
# Env:    TRT_SPEC  pip spec override (default tensorrt-cu12==10.9.0.34)
#         PY        python interpreter (default .venv/bin/python, else python)
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}" || exit 1

# --- interpreter: pin to the project venv (never a system/Colab site-packages) ---
VENV_PY="${PROJECT_ROOT}/.venv/bin/python"
if [ -z "${PY:-}" ]; then
    if [ -x "${VENV_PY}" ]; then
        PY="${VENV_PY}"
    else
        PY="python"
    fi
fi
# Put the venv's bin/ first so bare `pip`/`python` also resolve inside it.
if [ -f "${PROJECT_ROOT}/.venv/bin/activate" ]; then
    # shellcheck source=/dev/null
    source "${PROJECT_ROOT}/.venv/bin/activate"
fi
if [ -x "${VENV_PY}" ] && [ "${PY}" != "${VENV_PY}" ]; then
    echo "NOTE: using PY=${PY}, but a project venv exists at ${VENV_PY}."
    echo "      Unset PY (or set PY=${VENV_PY}) to install into the project venv."
fi
echo "interpreter: ${PY}"

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

# Where did each package actually land? A system/Colab interpreter silently
# swallows the install while the project venv stays unchanged.
report_pkgs() {
    "${PY}" - "$@" <<'PY'
import importlib.metadata as m
import sys

print("  sys.prefix:", sys.prefix)
for name in sys.argv[1:]:
    try:
        dist = m.distribution(name)
    except Exception:
        print(f"  {name}: not installed"); continue
    try:
        loc = str(dist.locate_file(""))
    except Exception:
        loc = "?"
    print(f"  {name}: {dist.version} -> {loc}")
PY
}

# System CUDA/driver lib dirs. Not optional: exporting LD_LIBRARY_PATH makes the
# loader ignore the interpreter's DT_RPATH -- which on Colab is what exposes
# /usr/lib64-nvidia/libcuda.so.1 -- so without these libcuda.so.1 is
# unresolvable and every CUDA call dies with error 35. Mirrors
# fast_float/initialize_env.py's _SYSTEM_CUDA_LIB_DIRS (the runtime path).
_SYSTEM_CUDA_LIB_DIRS="/usr/local/cuda/lib64:/usr/lib64-nvidia"

# Put the pip nvidia/tensorrt lib dirs (plus the system CUDA/driver dirs) on
# the loader path for THIS shell; a freshly installed TRT/CUDA lib is otherwise
# invisible until the next shell.
add_lib_dirs() {
    local dirs sysdirs all d
    local -a sysdirs_arr
    dirs="$("${PY}" -c 'import glob, os, site
d = set()
for b in site.getsitepackages() + [site.getusersitepackages()]:
    for p in [os.path.join(b, "nvidia"), *glob.glob(os.path.join(b, "tensorrt*"))]:
        for r, _sub, fs in os.walk(p):
            if any(f.endswith((".so", ".dll")) or ".so." in f for f in fs):
                d.add(r)
print(":".join(sorted(d)))' 2>/dev/null || true)"
    IFS=':' read -ra sysdirs_arr <<< "${_SYSTEM_CUDA_LIB_DIRS}"
    sysdirs=""
    for d in "${sysdirs_arr[@]}"; do
        if [ -d "${d}" ]; then
            sysdirs="${sysdirs}${sysdirs:+:}${d}"
        fi
    done
    all="${dirs}"
    if [ -n "${sysdirs}" ]; then
        all="${all}${all:+:}${sysdirs}"
    fi
    if [ -n "${all}" ]; then
        export LD_LIBRARY_PATH="${all}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
        echo "ORT loader path += pip CUDA/TensorRT libs + system CUDA/driver dirs"
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
# onnx >= ~1.19 defaults a new model to an IR version newer than this ORT can
# read (ORT 1.23 tops out at IR 11); without the clamp the probe dies with
# "Unsupported model IR version: N" and masks the real execution-provider
# status behind a model-loading error.
model.ir_version = min(model.ir_version, 9)
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
# onnx >= ~1.19 defaults a new model to an IR version newer than this ORT can
# read (ORT 1.23 tops out at IR 11); without the clamp the probe dies with
# "Unsupported model IR version: N" and masks the real execution-provider
# status behind a model-loading error.
model.ir_version = min(model.ir_version, 9)
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

# cu12 runtime wheels ORT 1.23's CUDA/TensorRT EPs dlopen by soname
# (libcublas.so.12, libcudnn.so.9, libcufft.so.11, ...). A box whose torch is
# cu118/cu13 ships none of them, so a GPU ORT wheel boots straight into the CPU
# EP and TensorRT cannot initialise either.
ORT_CUDA12_LIBS=(
    nvidia-cuda-runtime-cu12
    nvidia-cuda-nvrtc-cu12
    nvidia-cublas-cu12
    nvidia-cudnn-cu12
    nvidia-cufft-cu12
    nvidia-curand-cu12
    nvidia-cusolver-cu12
    nvidia-cusparse-cu12
    nvidia-nvjitlink-cu12
)

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
    report_pkgs onnxruntime onnxruntime-gpu
    "${PY}" -c 'import onnxruntime as o; print("  listed providers now:", o.get_available_providers())'
fi

# The GPU wheel lists CUDA, but ORT 1.23's CUDA EP is built against CUDA 12 and
# dlopens the cu12 sonames (libcublas.so.12, libcudnn.so.9, libcufft.so.11, ...).
# A box whose torch is cu118/cu13 ships none of them -> every session drops to
# CPU, and TensorRT cannot work either. Install the matching runtime wheels
# (pure data wheels; they coexist with torch's own bundled libs).
if has_cuda_ep; then
    if cuda_ep_active; then
        echo "CUDA EP is active."
    else
        CUDA_VER="$("${PY}" -c 'import torch; print(torch.version.cuda)' 2>/dev/null || echo '?')"
        echo "CUDAExecutionProvider is listed but no session activates it"
        echo "(torch here is CUDA ${CUDA_VER}; onnxruntime-gpu 1.23 dlopens CUDA 12 sonames)."
        echo "Installing the cu12 runtime libs the CUDA/TensorRT EPs need ..."
        if pip_install --no-cache-dir "${ORT_CUDA12_LIBS[@]}"; then
            add_lib_dirs
            if cuda_ep_active; then
                echo "CUDA EP is now active."
            else
                echo "WARN: CUDA EP still inactive after installing the cu12 runtime libs."
                echo "      Check that ${_SYSTEM_CUDA_LIB_DIRS} exist (driver libcuda.so.1)"
                echo "      and that the ORT wheel matches this box's CUDA major."
            fi
        else
            echo "WARN: could not install the cu12 runtime libs; CUDA/TensorRT may stay CPU-only."
        fi
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

echo "--- installed into ---"
report_pkgs tensorrt-cu12 tensorrt-cu12-libs tensorrt-cu13

echo "--- probing TensorRT after install ---"
add_lib_dirs
if trt_probe; then
    echo "OK: TensorRT EP is available (engines cache under <onnx-dir>/trt_cache)."
    exit 0
fi
echo "WARN: TensorRT still not loadable. Common causes:"
echo "  - the active ORT is still not a GPU build (see the diagnostics above);"
echo "  - the CUDA EP itself does not activate (missing cu12 runtime libs or an"
echo "    unresolvable driver libcuda.so.1; see the loader path above);"
echo "  - ORT expects a different TensorRT major (try another TRT_SPEC)."
echo "Units will fall back to the CUDA/CPU EP."
exit 1
