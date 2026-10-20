# shellcheck shell=bash disable=SC2329
# Shared ONNX conversion environment for the scripts in this directory.
#
# Sourced (never executed directly) so convert_to_onnx.sh and the per-unit
# helpers share ONE fix for the install traps documented in AGENT.md §6:
#   * a uv venv ships no `pip`, so a bare `pip` installs into the *system*
#     site-packages (Colab especially) and the venv never changes;
#   * onnxscript pulls the CPU onnxruntime, whose files shadow onnxruntime-gpu
#     and kill every CUDA/TensorRT EP;
#   * onnx-built models default to an IR version newer than ORT 1.23 accepts.
#
# Callers keep their own SCRIPT_DIR/PROJECT_ROOT/cd, then:
#   source "${SCRIPT_DIR}/_onnx_env.sh"
#   use_project_venv
#   ensure_onnx_stack
#
# Functions assume the caller's `set -euo pipefail` unless noted; nothing here
# changes the caller's shell options.

# onnx window: onnxscript (the dynamo exporter's translator) needs onnx>=1.17,
# while ORT 1.23 refuses any model whose IR version exceeds 11. onnx 1.19+
# defaults to IR 12/13, so only 1.17/1.18 are safe despite `onnx<1.22` still
# shipping onnx.external_data_helper. (The dynamo exporter itself always writes
# IR 10, so conversion never trips this -- other ONNX callers in the repo do.)
ONNX_PIN="onnx>=1.17,<1.19"
ONNXSCRIPT_PIN="onnxscript==0.7.2"
ORT_MAX="1.24"

# Activate the project venv (if present) and route installs through uv.
# Sets PY, PIP_REINSTALL and defines pip_install/pip_uninstall.
use_project_venv() {
  if [ -f "${PROJECT_ROOT}/.venv/bin/activate" ]; then
    # shellcheck source=/dev/null
    source "${PROJECT_ROOT}/.venv/bin/activate"
  elif [ -f "${PROJECT_ROOT}/.venv/Scripts/activate" ]; then
    # shellcheck source=/dev/null
    source "${PROJECT_ROOT}/.venv/Scripts/activate"
  fi
  if [ -z "${PY:-}" ]; then
    if [ -x "${PROJECT_ROOT}/.venv/bin/python" ]; then
      PY="${PROJECT_ROOT}/.venv/bin/python"
    elif [ -x "${PROJECT_ROOT}/.venv/Scripts/python.exe" ]; then
      PY="${PROJECT_ROOT}/.venv/Scripts/python.exe"
    else
      PY="$(command -v python)"
    fi
  fi
  export PY
  echo "python: ${PY} ($("${PY}" -c 'import sys; print(sys.prefix)'))"
  if command -v uv >/dev/null 2>&1; then
    pip_install() { uv pip install -q --no-cache -p "${PY}" "$@"; }
    pip_uninstall() { uv pip uninstall -q -p "${PY}" "$@"; }
    PIP_REINSTALL=--reinstall
  else
    pip_install() { "${PY}" -m pip install -q --no-cache-dir "$@"; }
    pip_uninstall() { "${PY}" -m pip uninstall -q -y "$@"; }
    PIP_REINSTALL=--force-reinstall
  fi
}

# Importability alone is not enough: an onnx outside the pin (e.g. the
# pyproject portable `onnx<1.17`, or a stray 1.19+ pulled by another tool)
# imports fine but makes `make_model` default to IR 12/13, which ORT 1.23
# rejects ("Unsupported model IR version"). Enforce the pin here so the repair
# below actually fires.
probe_onnx() {
  "${PY}" -c "
import sys, importlib.metadata as m
import onnx.external_data_helper, onnxscript, onnxruntime  # noqa: F401
mm = lambda v: tuple(int(p) for p in v.split('.')[:2])
onnx_ok = (1, 17) <= mm(m.version('onnx')) < (1, 19)
sys.exit(0 if (onnx_ok and m.version('onnxscript') == '0.7.2') else 1)" 2>/dev/null
}

# On CUDA boxes the ORT build must ALSO ship a working CUDA EP. CPU-only torch
# boxes always pass this probe.
probe_ort_cuda() {
  "${PY}" -c "
import sys
import torch
import onnxruntime as o
ok = (not torch.cuda.is_available()
      or 'CUDAExecutionProvider' in o.get_available_providers())
sys.exit(0 if ok else 1)" 2>/dev/null
}

probe_stack() { probe_onnx && probe_ort_cuda; }

# onnxruntime>=1.24 dropped cp310 wheels; the box runs CPython 3.10.
onnx_rt_pkg() {
  if "${PY}" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "onnxruntime-gpu<${ORT_MAX}"
  else
    echo "onnxruntime<${ORT_MAX}"
  fi
}

# onnxscript depends on the CPU onnxruntime; both wheels ship the same
# `onnxruntime` package files, so on a CUDA box whichever installed last owns
# them. Install the stack, then hand the shared files back to the GPU wheel
# last -- otherwise a later `uv sync` silently shadows the GPU build.
install_onnx_stack() {
  local rt
  rt="$(onnx_rt_pkg)"
  pip_uninstall onnx onnxscript onnx-ir onnxruntime onnxruntime-gpu || true
  pip_install "$@" "${ONNX_PIN}" "${ONNXSCRIPT_PIN}" "$rt" \
    || pip_install "$@" "${ONNX_PIN}" "${ONNXSCRIPT_PIN}" "onnxruntime<${ORT_MAX}"
  if [ "${rt}" != "onnxruntime<${ORT_MAX}" ]; then
    pip_uninstall onnxruntime || true
    pip_install "$PIP_REINSTALL" --no-deps "$rt"
  fi
}

print_onnx_versions() {
  "${PY}" -c "import importlib.metadata as m
for p in ('onnx','onnxscript','onnx-ir','onnxruntime','onnxruntime-gpu'):
    try: print(' ', p, m.version(p))
    except Exception: print(' ', p, 'MISSING')"
}

# Verify the stack, repairing once (and once more with a reinstall), then abort
# the caller on failure (export cannot run without it).
ensure_onnx_stack() {
  if [ "${FLOAT_ONNX_STACK_READY:-0}" = "1" ]; then
    return 0
  fi
  if ! probe_stack; then
    echo "installing onnx stack (pin ${ONNX_PIN}, ${ONNXSCRIPT_PIN}, onnxruntime<${ORT_MAX})"
    install_onnx_stack
  fi
  if ! probe_stack; then
    echo "retrying with reinstall (clears partial/corrupt installs)"
    install_onnx_stack "$PIP_REINSTALL"
  fi
  echo "onnx stack versions:"
  print_onnx_versions
  probe_stack || { echo "ERROR: onnx stack still broken — export cannot run"; exit 1; }
  export FLOAT_ONNX_STACK_READY=1
}

# Listing the EP only proves the wheel was built with TRT support; it says
# nothing about whether libnvinfer (of the exact version ORT expects) is
# loadable. Delegate to the same honest check the exporters use.
probe_trt() {
  "${PY}" -c "
import sys
import torch
import onnxruntime as o
if not torch.cuda.is_available():
    sys.exit(0)
if 'TensorrtExecutionProvider' not in o.get_available_providers():
    sys.exit(1)
from fast_float.onnx.export import _trt_libs_loadable
sys.exit(0 if _trt_libs_loadable() else 1)" 2>/dev/null
}

# ORT's CUDA/TensorRT EPs dlopen libcublas/libcudnn/libnvinfer by soname at
# session init. When those come from pip `nvidia-*`/`tensorrt*` wheels the
# loader cannot find them, ORT silently drops the GPU EP, and every parity
# check dies with "fell back to CPU". The system CUDA/driver dirs are required,
# not optional: exporting LD_LIBRARY_PATH makes the loader ignore the
# interpreter's DT_RPATH -- which on Colab is what exposes
# /usr/lib64-nvidia/libcuda.so.1. Without them libcuda.so.1 is unresolvable and
# every CUDA/TRT session fails with error 35.
augment_ort_libs() {
  local dirs sysdirs d all
  local -a sysdirs_arr
  dirs="$("${PY}" -c 'import glob, os, site
d = set()
for b in site.getsitepackages() + [site.getusersitepackages()]:
    for p in [os.path.join(b, "nvidia"), *glob.glob(os.path.join(b, "tensorrt*"))]:
        for r, _sub, fs in os.walk(p):
            if any(f.endswith((".so", ".dll")) or ".so." in f for f in fs):
                d.add(r)
print(":".join(sorted(d)))' 2>/dev/null || true)"
  IFS=':' read -ra sysdirs_arr <<< "/usr/local/cuda/lib64:/usr/lib64-nvidia"
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
