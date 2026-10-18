#!/bin/bash
# One-command ONNX conversion (Kaggle / any Linux box with python + torch).
#   ./scripts/convert_to_onnx.sh
# Env overrides: OUT_DIR (default ./onnx), EMO_OUT, FMT_OUT, DIR_OUT,
#   AUDIO_OUT, DEC_OUT, REF_OUT, CKPT, EMO_DIR, W2V_DIR, DEC_BATCH, AUDIO_FRAMES.
#
# NOTE: We do NOT use set -e globally. Weights/dep sections abort on failure
# (they are prerequisites). Each converter runs independently so all 6 are
# always attempted; a summary at the end shows PASS/FAIL per unit.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# --- pin to the project venv: bare `python`/`pip` (Colab especially) otherwise
# land in a system site-packages and the venv stays unchanged. ---
if [ -f "${PROJECT_ROOT}/.venv/bin/activate" ]; then
  # shellcheck source=/dev/null
  source "${PROJECT_ROOT}/.venv/bin/activate"
fi
PY="${PY:-$(command -v python)}"
export PY
echo "python: ${PY} ($("${PY}" -c 'import sys; print(sys.prefix)'))"

# --- ...and the same for *installs*. Activating a uv venv does not give you a
# `pip`: uv venvs ship neither a `pip` binary nor the `pip` module, so a bare
# `pip` silently resolves to the system pip and installs into the wrong
# site-packages (Colab especially) -- the venv stays untouched and every
# "repair" below becomes a no-op. Route installs through uv (the repo's rule)
# and keep a `python -m pip` fallback for plain venvs. ---
if command -v uv >/dev/null 2>&1; then
  pip_install() { uv pip install -q --no-cache -p "${PY}" "$@"; }
  pip_uninstall() { uv pip uninstall -q -p "${PY}" "$@"; }
  PIP_REINSTALL=--reinstall
else
  pip_install() { "${PY}" -m pip install -q --no-cache-dir "$@"; }
  pip_uninstall() { "${PY}" -m pip uninstall -q -y "$@"; }
  PIP_REINSTALL=--force-reinstall
fi

OUT_DIR="${OUT_DIR:-./onnx}"
EMO_OUT="${EMO_OUT:-${OUT_DIR}/emotion_v1.onnx}"
FMT_OUT="${FMT_OUT:-${OUT_DIR}/fmt_step_v1.onnx}"
DIR_OUT="${DIR_OUT:-${OUT_DIR}/direction_v1.onnx}"
AUDIO_OUT="${AUDIO_OUT:-${OUT_DIR}/audio_enc_v1.onnx}"
DEC_OUT="${DEC_OUT:-${OUT_DIR}/decoder_v1.onnx}"
REF_OUT="${REF_OUT:-${OUT_DIR}/refenc_v1.onnx}"
CKPT="${CKPT:-./checkpoints/float.pth}"
EMO_DIR="${EMO_DIR:-./checkpoints/wav2vec-english-speech-emotion-recognition}"
W2V_DIR="${W2V_DIR:-./checkpoints/wav2vec2-base-960h}"
# Decoder bakes a STATIC batch (grouped conv); audio bakes a STATIC frame
# count T. Runtime must pad/trim to these (documented in the converters).
DEC_BATCH="${DEC_BATCH:-8}"
AUDIO_FRAMES="${AUDIO_FRAMES:-300}"
# Extra flags forwarded to every converter, e.g. ORT_ARGS="--no-trt" to force
# the CUDA EP, ORT_ARGS="--trt-fp32" for FP32 TensorRT, or
# ORT_ARGS="--trt-max-batch 16". Default: TensorRT FP16 engine-cached.
ORT_ARGS="${ORT_ARGS:-}"
read -r -a ORT_ARGS_ARR <<< "${ORT_ARGS}" || true
mkdir -p "${OUT_DIR}" checkpoints

echo "=== [1/8] weights ==="
if [ ! -f "${CKPT}" ]; then
  echo "float.pth missing -> downloading (gdown)"
  command -v gdown >/dev/null || pip_install gdown
  gdown --id 1rvWuM12cyvNvBQNCLmG4Fr2L1rpjQBF0 -O "${CKPT}"
else
  echo "found ${CKPT}"
fi
if [ ! -d "${EMO_DIR}" ]; then
  echo "emotion model missing -> downloading (huggingface-cli)"
  command -v huggingface-cli >/dev/null || pip_install "huggingface_hub[hf-transfer]"
  huggingface-cli download r-f/wav2vec-english-speech-emotion-recognition \
    --local-dir "${EMO_DIR}"
else
  echo "found ${EMO_DIR}"
fi
if [ ! -d "${W2V_DIR}" ]; then
  echo "wav2vec2-base missing -> downloading (huggingface-cli)"
  command -v huggingface-cli >/dev/null || pip_install "huggingface_hub[hf-transfer]"
  huggingface-cli download facebook/wav2vec2-base-960h \
    --local-dir "${W2V_DIR}"
else
  echo "found ${W2V_DIR}"
fi

echo "=== [2/8] python deps ==="
python -c "import torch; print('torch', torch.__version__, 'cuda=', torch.cuda.is_available())"
probe_onnx() { python -c "import onnx.external_data_helper, onnxscript, onnxruntime" 2>/dev/null; }
# On CUDA boxes the ORT build must ALSO ship a working CUDA EP.
# CPU-only torch boxes always pass this probe.
probe_ort_cuda() {
  python -c "
import sys
import torch
import onnxruntime as o
ok = (not torch.cuda.is_available()
      or 'CUDAExecutionProvider' in o.get_available_providers())
sys.exit(0 if ok else 1)" 2>/dev/null
}
probe_trt() {
  # Listing the EP only proves the onnxruntime-gpu wheel was built with TRT
  # support; it says nothing about whether libnvinfer (of the exact version ORT
  # expects) is loadable. Delegate to the same honest check the exporters use.
  python -c "
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
probe_stack() { probe_onnx && probe_ort_cuda; }
onnx_rt_pkg() {
  # onnxruntime>=1.24 dropped cp310 wheels; the box runs CPython 3.10.
  if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "onnxruntime-gpu<1.24"
  else
    echo "onnxruntime<1.24"
  fi
}
# onnx window: `onnxscript` (the dynamo exporter's translator) needs
# onnx>=1.17, and ORT 1.23 refuses any model whose IR version exceeds 11. An
# onnx built model defaults to the installed onnx's own IR version, which 1.19+
# pushes past that (1.19->12, 1.20/1.21->13), so only 1.17/1.18 are safe
# despite `onnx<1.22` still having `onnx.external_data_helper`. (The dynamo
# exporter itself always writes IR 10, so conversion never trips this -- other
# ONNX callers in the repo do.)
ONNX_PIN="onnx>=1.17,<1.19"
# `onnxscript` (needed by the dynamo exporter) depends on the *CPU*
# `onnxruntime`, and both distributions ship the same `onnxruntime` package
# files. On a CUDA box whichever installed last owns them, so a plain
# `uv sync` (which installs the CPU wheel declared in pyproject) silently
# shadows the GPU build -- killing every CUDA/TensorRT EP. The repair therefore
# has to hand the shared files back to the GPU wheel last.
install_onnx_stack() {
  local rt
  rt="$(onnx_rt_pkg)"
  pip_uninstall onnx onnxscript onnx-ir onnxruntime onnxruntime-gpu || true
  pip_install "$@" "${ONNX_PIN}" "onnxscript==0.7.2" "$rt" || pip_install "$@" "${ONNX_PIN}" "onnxscript==0.7.2" "onnxruntime<1.24"
  if [ "${rt}" != "onnxruntime<1.24" ]; then
    # Drop the CPU wheel onnxscript dragged in, then re-take the files (its dep
    # is not enforced at runtime -- onnxruntime-gpu provides the same module).
    pip_uninstall onnxruntime || true
    pip_install "$PIP_REINSTALL" --no-deps "$rt"
  fi
}
if ! probe_stack; then
  echo "installing onnx stack (pin ${ONNX_PIN}, onnxscript 0.7.2, onnxruntime<1.24)"
  install_onnx_stack
fi
if ! probe_stack; then
  echo "retrying with reinstall (clears partial/corrupt installs)"
  install_onnx_stack "$PIP_REINSTALL"
fi
echo "onnx stack versions:"
python -c "import importlib.metadata as m
for p in ('onnx','onnxscript','onnx-ir','onnxruntime','onnxruntime-gpu'):
    try: print(' ', p, m.version(p))
    except Exception: print(' ', p, 'MISSING')"
probe_stack || { echo "ERROR: onnx stack still broken — export cannot run"; exit 1; }

# ORT's CUDA/TensorRT EPs dlopen libcublas/libcudnn/libnvinfer by soname at
# session init. When those come from pip `nvidia-*`/`tensorrt*` wheels — and
# especially when torch bundles a *different* CUDA major than the
# onnxruntime-gpu wheel (e.g. torch cu130 with ORT cu12) — the loader cannot
# find them, ORT silently drops the GPU EP, and every parity check dies with
# "fell back to CPU". The loader path is augmented once, *after* the TensorRT
# step below, so a just-installed tensorrt lib dir is picked up too.
augment_ort_libs() {
  local dirs sysdirs d all
  local -a sysdirs_arr
  dirs="$(python -c 'import glob, os, site
d = set()
for b in site.getsitepackages() + [site.getusersitepackages()]:
    for p in [os.path.join(b, "nvidia"), *glob.glob(os.path.join(b, "tensorrt*"))]:
        for r, _sub, fs in os.walk(p):
            if any(f.endswith((".so", ".dll")) or ".so." in f for f in fs):
                d.add(r)
print(":".join(sorted(d)))' 2>/dev/null || true)"
  # The system CUDA/driver dirs are required, not optional: exporting
  # LD_LIBRARY_PATH makes the loader ignore the interpreter's DT_RPATH — which
  # on Colab is what exposes /usr/lib64-nvidia/libcuda.so.1. Without them
  # libcuda.so.1 is unresolvable and every CUDA/TRT session fails with error 35
  # ("CUDA driver version is insufficient for CUDA runtime version").
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

echo "=== [2.5/8] tensorRT probe ==="
if [ "$(python -c 'import torch;print(torch.cuda.is_available())' 2>/dev/null)" = "True" ]; then
  # Put the pip CUDA/TensorRT lib dirs on the loader path *before* probing:
  # probe_trt loads ORT's TRT provider lib, which dlopens libnvinfer/libcublas,
  # and a fresh python has none of them on its default path (this is why the
  # probe used to report "not available" even when TensorRT worked).
  augment_ort_libs
  if probe_trt; then
    echo "TensorRT EP available (engines cached under ${OUT_DIR}/trt_cache)"
  else
    echo "TensorRT EP NOT available -> running scripts/install_tensorrt.sh"
    PY="${PY}" bash "${SCRIPT_DIR}/install_tensorrt.sh" || true
    # The installer runs in a subshell (its exports don't reach us) and may have
    # installed new lib dirs, so re-augment before re-probing.
    augment_ort_libs
    if probe_trt; then
      echo "TensorRT EP now available"
    else
      echo "WARN: TensorRT still unavailable — units fall back to the CUDA EP"
    fi
  fi
else
  echo "no CUDA on this box — skipping TensorRT"
fi


# Per-unit runner: captures output, records PASS/FAIL, never aborts other units.
UNIT_RESULTS=()
run_unit() {
  local label="$1"; shift
  echo ""
  echo "=== ${label} ==="
  if "$@"; then
    UNIT_RESULTS+=("PASS  ${label}")
  else
    UNIT_RESULTS+=("FAIL  ${label}")
    echo "[WARN] ${label} failed — continuing with remaining units"
  fi
}

# Temporarily disable pipefail so per-unit failures don't abort the whole script.
set +e

run_unit "[3/8] convert: emotion (unit 1)" \
  python -m fast_float.onnx.convert_emotion \
    --ckpt "${CKPT}" --emo-dir "${EMO_DIR}" --out "${EMO_OUT}" "${ORT_ARGS_ARR[@]}"

run_unit "[4/8] convert: fmt step (unit 2)" \
  python -m fast_float.onnx.convert_fmt \
    --ckpt "${CKPT}" --out "${FMT_OUT}" "${ORT_ARGS_ARR[@]}"

run_unit "[5/8] convert: direction (unit 3)" \
  python -m fast_float.onnx.convert_direction \
    --ckpt "${CKPT}" --out "${DIR_OUT}" "${ORT_ARGS_ARR[@]}"

run_unit "[6/8] convert: audio encoder (unit 4)" \
  python -m fast_float.onnx.convert_audio \
    --ckpt "${CKPT}" --w2v-dir "${W2V_DIR}" --out "${AUDIO_OUT}" \
    --frames "${AUDIO_FRAMES}" "${ORT_ARGS_ARR[@]}"

run_unit "[7/8] convert: frame decoder (unit 5)" \
  python -m fast_float.onnx.convert_decoder \
    --ckpt "${CKPT}" --out "${DEC_OUT}" --batch "${DEC_BATCH}" "${ORT_ARGS_ARR[@]}"

run_unit "[8/8] convert: reference encoder (unit 6)" \
  python -m fast_float.onnx.convert_refenc \
    --ckpt "${CKPT}" --out "${REF_OUT}" "${ORT_ARGS_ARR[@]}"

set -e

echo ""
echo "=== artifacts ==="
ls -la "${OUT_DIR}" 2>/dev/null || echo "(no artifacts dir)"

echo ""
echo "=== unit summary ==="
FAILED=0
for r in "${UNIT_RESULTS[@]}"; do
  echo "  ${r}"
  [[ "${r}" == FAIL* ]] && FAILED=$((FAILED + 1))
done
if [ "${FAILED}" -gt 0 ]; then
  echo ""
  echo "ERROR: ${FAILED} unit(s) failed — see logs above"
  exit 1
else
  echo ""
  echo "All units PASSED"

  # Per-component backend selection: compares eager vs torch.compile vs ONNX
  # and writes onnx/backends.generated.yaml (read at runtime). Set
  # SKIP_SELECT=1 to skip and let the runtime default to all-eager.
  echo ""
  echo "=== [select] per-component backend selection ==="
  if [ "${SKIP_SELECT:-0}" = "1" ]; then
    echo "SKIP_SELECT=1 -> skipping (runtime defaults to all-eager)"
  else
    python -m fast_float.onnx.select_backends \
      --onnx-dir "${OUT_DIR}" --out "${OUT_DIR}/backends.generated.yaml" \
      --ckpt "${CKPT}" --emo-dir "${EMO_DIR}" --w2v-dir "${W2V_DIR}" \
      --dec-batch "${DEC_BATCH}" --audio-frames "${AUDIO_FRAMES}" \
      "${ORT_ARGS_ARR[@]}" || echo "[WARN] backend selection failed -> runtime keeps all-eager"
  fi
fi
