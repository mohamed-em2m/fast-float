#!/bin/bash

# CUDA/NVIDIA loader paths are resolved from the *project venv* after it is
# created (augment_ort_libs below) and again at app runtime by
# scripts/setup_cuda_libs.sh. The old block here hardcoded a python3.13
# site-packages layout that never matched this project (it ships a python 3.10
# venv), so it added nothing on any real box.

# Policy of this script:
#   * the environment build (uv -> venv -> deps -> ONNX stack) MUST succeed;
#   * everything that only *enables* something (system packages, the optional
#     codecs, the TensorRT layer, the checkpoint download) is best effort --
#     a failure there must not abort the run and leave the box half-built.
set -euo pipefail

# Always resolve paths relative to the repo root (this script lives in scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# Detect if running in Docker
if [ "${IS_IT_RUN_IN_DOCKER:-}" = "true" ]; then
    echo "=== Running in Docker environment ==="
elif [ "$(id -u)" -eq 0 ]; then
    echo "=== Running as root ==="
else
    echo "=== Running in local environment ==="
fi

# Run a command with root privileges -- apt needs them. Root runs it directly,
# a normal user goes through sudo when sudo is available.
run_as_root() {
    if [ "$(id -u)" -eq 0 ]; then
        "$@"
    elif command -v sudo >/dev/null 2>&1; then
        sudo "$@"
    else
        "$@"
    fi
}

# Cache the sudo credential once (a timestamp) so the apt calls below ask for
# the password a single time instead of once per package.
if [ "$(id -u)" -ne 0 ] && command -v sudo >/dev/null 2>&1; then
    sudo -v >/dev/null 2>&1 || true
fi

echo "=== Updating system packages ==="
# apt needs root. This block used to sit at the very top of the script, *before*
# the sudo detection above, so on any normal (non-root) desktop it died with
# "Could not open lock file /var/lib/apt/lists/lock - Permission denied" and
# `set -e` aborted the whole setup before uv, the venv and the deps were even
# touched. It also ran `apt-get update` twice (once here, once below).
if ! run_as_root apt-get update -y; then
    echo "⚠️ apt-get update failed — continuing (uv/pip do not need apt)."
fi

echo "=== Installing system dependencies ==="
# One package at a time: a single package the distro does not ship (e.g.
# python3.10 on Ubuntu 24.04 without the deadsnakes PPA) would otherwise abort
# the whole apt transaction and take ffmpeg/git/wget/curl down with it.
install_system_pkg() {
    local pkg="$1"
    if ! run_as_root apt-get install -y "${pkg}"; then
        echo "⚠️ could not install '${pkg}' — continuing."
        return 0
    fi
}
# ffmpeg: video encode/decode (NVENC + PyAV). git/wget/curl: tooling and the
# uv/Node installers. build-essential + python3.10-dev: compiling a wheel that
# has no prebuilt one (uv provisions its own interpreter when the system has
# no python3.10, so these two are conveniences, not requirements).
for pkg in ffmpeg git wget curl build-essential python3.10 python3.10-dev; do
    install_system_pkg "${pkg}"
done

echo "=== Installing uv (package manager) ==="
# uv installs into ~/.local/bin, which is not always on PATH: add it *before*
# looking for uv so an existing installation is reused instead of re-downloaded
# on every run.
export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"
fi
echo "uv version: $(uv --version)"

echo "=== Ensuring Node for the Studio frontend (Vite 8 needs 20.19+/22.12+) ==="
# Ubuntu 24.04's `nodejs` is 18.x, which Vite 8 / rolldown cannot run: they
# import `node:util`'s `styleText` (Node >= 20.12) and die at `vite dev` with a
# cryptic "does not provide an export named 'styleText'". Distro repos never
# carry the needed version, so pull LTS 22 from NodeSource when the system Node
# is missing or too old. Best effort -- a failure only disables the frontend
# (run_all.sh re-checks and reports it), never the backend/ONNX pipeline.
node_version_ok() {
    local mm="" major="" minor=""
    mm="$(node -p 'process.versions.node.split(".").slice(0,2).join(".")' 2>/dev/null)" || return 1
    major="${mm%%.*}"
    minor="${mm##*.}"
    [ "${major}" -gt 22 ] && return 0
    [ "${major}" -eq 22 ] && [ "${minor}" -ge 12 ] && return 0
    [ "${major}" -eq 20 ] && [ "${minor}" -ge 19 ] && return 0
    return 1
}
if command -v node >/dev/null 2>&1 && node_version_ok; then
    echo "found node $(node -v)"
else
    echo "node $(node -v 2>/dev/null || echo '(missing)') is too old for Vite 8 -> installing Node 22"
    if curl -fsSL https://deb.nodesource.com/setup_22.x | run_as_root bash - \
        && run_as_root apt-get install -y nodejs; then
        # Drop the hashed path entry, otherwise bash keeps resolving the old
        # binary and the version check below still sees Node 18.
        hash -r
        echo "installed node $(node -v)"
    else
        echo "⚠️ Node upgrade failed — the frontend needs Node 20.19+/22.12+; the backend is unaffected."
    fi
fi

# Refresh the frontend deps after a Node/npm change. npm's optional-dependency
# bug (npm/cli#4828) leaves platform natives out of an existing node_modules
# when the npm that wrote it was older, so `vite dev` dies with "Cannot find
# native binding ... @rolldown/binding-linux-x64-gnu" even though the install
# "succeeded". `npm ci` rebuilds it from the committed lockfile.
if [ -f "${PROJECT_ROOT}/frontend/package.json" ] && command -v npm >/dev/null 2>&1; then
    if node_version_ok; then
        echo "=== Installing frontend dependencies (npm ci) ==="
        (cd "${PROJECT_ROOT}/frontend" && npm ci) \
            || echo "⚠️ npm ci failed — run it in frontend/ before ./scripts/run_all.sh"
    fi
fi

echo "=== Syncing project environment (uv sync) ==="
echo "--- Python comes from .python-version, deps from pyproject.toml + uv.lock ---"
# The interpreter version is read from .python-version (the single source of
# truth). `uv venv --python 3.10` hardcoded it, so the venv silently drifted
# away from the file whenever the project moved to another patch release.
REQUIRED_PY_VERSION="3.10"
if [ -f "${PROJECT_ROOT}/.python-version" ]; then
    _py_version="$(tr -d '[:space:]' < "${PROJECT_ROOT}/.python-version" 2>/dev/null || true)"
    if [ -n "${_py_version}" ]; then
        REQUIRED_PY_VERSION="${_py_version}"
    fi
fi
echo "python version: ${REQUIRED_PY_VERSION}"

# Reuse an existing venv: `--clear` used to wipe it on every run, forcing a fresh
# ~7 GiB torch download plus a full re-extract of the layered ONNX/TensorRT
# stack. On a box that is even moderately full the rebuild runs out of space
# mid-way and leaves the env without TensorRT (or worse, without torch). Pass
# --clear only when you really want a from-scratch environment.
venv_python_version() {
    [ -x "${1}" ] || return 1
    "${1}" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null
}
if [ -x .venv/bin/python ]; then
    _venv_py="$(venv_python_version .venv/bin/python || true)"
    if [ "${_venv_py}" != "${REQUIRED_PY_VERSION}" ]; then
        # Wrong interpreter: `uv sync` cannot satisfy requires-python with it,
        # so the venv has to be rebuilt (torch et al. are re-installed by the
        # `uv sync` below).
        echo "existing .venv runs Python ${_venv_py:-?} but the project needs ${REQUIRED_PY_VERSION} -> recreating it"
        # --force: the directory may be a half-written/corrupted venv, which
        # uv refuses to clear without it.
        uv venv --clear --force .venv --python "${REQUIRED_PY_VERSION}"
    fi
fi
if [ ! -x .venv/bin/python ]; then
    uv venv .venv --python "${REQUIRED_PY_VERSION}"
fi
# shellcheck source=/dev/null
source .venv/bin/activate
# uv.lock is the source of truth (torch cu126 from the pytorch index, the
# app's onnx<1.17, ...). The ONNX *export* stack needs a different, conflicting
# window and is layered on top below.
#
# --inexact keeps packages the lock does not know about. onnxruntime-gpu and
# tensorrt-cu12 are layered on below; a plain `uv sync` prunes them, and the
# installer then has to re-extract ~4.4 GiB of TensorRT libs (the Linux wheel
# ships both the Linux and a Windows builder resource) -- the exact ENOSPC
# failure this script has to survive on a 50 GiB disk.
uv sync --inexact

echo "=== Installing LiveKit SDK (token API + RTC SDK) ==="
# 1. The token API (livekit-api) is numpy-free and a normal dependency group.
#    --inexact: preserve the ONNX / TensorRT layers that a plain uv sync
#    --group would prune (it treats any unlisted package as an orphan and
#    removes them).
if ! uv sync --group livekit --inexact; then
    echo "Warning: livekit-api install failed -- LiveKit token generation will be unavailable."
fi
# 2. The RTC SDK (livekit) — needed by the avatar worker / publisher tasks
#    (fast_float/livekit/worker.py). Its wheel *declares* numpy>=2.0, but uv
#    would then unpin the project's numpy==1.24.4 (needed by onnx<1.17).
#    Verified at runtime: the SDK works correctly with numpy 1.24.4, and all
#    of its real dependencies (protobuf, aiofiles, ...) are already in this
#    venv — so install it WITHOUT its dependency resolution (--no-deps) and
#    re-assert numpy afterwards to be safe against a drifted lock.
if ! uv pip install --python .venv/bin/python --no-deps "livekit>=1.1"; then
    echo "Warning: livekit RTC SDK install failed -- LiveKit live-transport publishing will be unavailable."
fi
if ! uv pip install --python .venv/bin/python "numpy==1.24.4"; then
    echo "Warning: numpy re-pin failed — check 'uv pip list | grep numpy'."
fi
# Probe: both halves of the LiveKit stack must import (with the pinned numpy).
if uv run --frozen --no-sync python - <<'PY'
from livekit import api as livekit_api
from livekit import rtc

token = livekit_api.AccessToken("probe-key", "probe-secret").with_grants(
    livekit_api.VideoGrants(room_join=True, room="probe")
).to_jwt()
import numpy  # noqa: F401
print("  livekit-api ok, rtc ok, numpy", numpy.__version__, "- token minted (len", len(token), ")")
PY
then
    echo "LiveKit SDK installed."
else
    echo "⚠️ LiveKit SDK probe failed — LiveKit transport will be unavailable."
fi

echo "=== Starting the LiveKit SFU server container (best effort) ==="
# The dev server (docker-compose.livekit.yml) plus the dev keys already set in
# the environment above make the live transport usable right after setup.
if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
    if docker compose -f "${PROJECT_ROOT}/docker-compose.livekit.yml" up -d; then
        echo "LiveKit SFU running on ws://127.0.0.1:7880 (stop: docker compose -f docker-compose.livekit.yml down)"
    else
        echo "⚠️ LiveKit server container failed to start — scripts/run_all.sh retries it."
    fi
else
    echo "docker not found — the live transport falls back to MP4 until the LiveKit"
    echo "server is started (docker compose -f docker-compose.livekit.yml up -d)."
fi

echo "=== Installing optional video codecs (torchcodec) ==="
# torchcodec is optional; install if compatible binary wheel exists for torch/cuda
uv pip install --python .venv/bin/python torchcodec || echo "⚠️ torchcodec install skipped or unavailable for current platform/PyTorch version (PyAV and FFmpeg NVENC remain available)."

echo "=== Installing ONNX export toolchain (pinned: onnxscript / ONNX Runtime / TensorRT) ==="
# The app (uv.lock / pyproject) pins onnx<1.17, but the dynamo exporter needs
# onnxscript (onnx>=1.17) while ORT 1.23 refuses model IR > 11 -> only onnx
# 1.17/1.18 is safe (AGENT.md §6). Install exactly that window — and the
# matching onnxruntime — from the one shared helper, so setup.sh and
# scripts/convert_to_onnx.sh can never disagree. Installing onnxscript by hand
# here (as this script used to) pulls whatever onnx PyPI currently serves
# (1.23.2 today): every session then dies with "Unsupported model IR version:
# 14", and onnxscript's CPU onnxruntime wheel silently shadows the GPU one.
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/_onnx_env.sh"
use_project_venv
ensure_onnx_stack

# TensorRT must *major-match* the ORT wheel (10.9 for ORT 1.23; libnvinfer.so.10).
# The dedicated installer pins tensorrt-cu12==10.9.0.34, evicts any TRT 11
# (cu13) tree, and re-asserts onnxruntime-gpu afterwards.
PY="${PY}" bash "${SCRIPT_DIR}/install_tensorrt.sh" || \
    echo "⚠️ TensorRT install failed — units fall back to the CUDA EP."

# ORT dlopens libnvinfer/libcublas/libcudnn by soname; put the pip CUDA/TensorRT
# lib dirs + the system driver dirs on the loader path for the probe below (the
# app repeats this at runtime via scripts/setup_cuda_libs.sh).
augment_ort_libs

echo "=== ONNX / ONNX Runtime / TensorRT stack ==="
python - <<'PY' || true
import ctypes
import importlib.metadata as m
import os
import sys

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
    # onnx >= ~1.19 defaults a new model to an IR version newer than ORT 1.23
    # can read (it tops out at IR 11), which would abort the probe with
    # "Unsupported model IR version" and be misreported as an ORT import error.
    model.ir_version = min(model.ir_version, 9)
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
    # Honest TRT test: load ORT's own provider library. A TensorRT tree whose
    # major disagrees with ORT (e.g. libnvinfer.so.11 vs ORT's expected .so.10)
    # imports fine yet is unusable, so a bare `import tensorrt` lies.
    trt_ok = False
    try:
        import tensorrt  # noqa: F401  (primes the pip lib dirs on the loader path)
    except Exception:
        pass
    capi = os.path.join(os.path.dirname(ort.__file__), "capi")
    provider = os.path.join(
        capi,
        "onnxruntime_providers_tensorrt.dll" if sys.platform == "win32" else "libonnxruntime_providers_tensorrt.so",
    )
    if os.path.exists(provider):
        try:
            ctypes.CDLL(provider)
            trt_ok = True
        except OSError:
            trt_ok = False
    print(f"  TensorRT EP listed: {trt_listed} | provider lib loadable: {trt_ok}")
    if trt_listed and not trt_ok:
        print("  ⚠️ TensorRT EP listed but ORT's provider lib cannot load libnvinfer.")
        print("     Likely a TensorRT major mismatch: ORT 1.23 needs TensorRT 10.9")
        print("     (libnvinfer.so.10). Install: uv pip install 'tensorrt-cu12==10.9.0.34'")
    # TensorRT dlopens libnvinfer_builder_resource.so.<ver> LAZILY, only when a
    # real network is built. The Identity-model probe above therefore passes on
    # a truncated install (e.g. the disk filled mid-extraction) while every
    # conversion unit later dies with "createInferBuilder: Unable to load
    # library: libnvinfer_builder_resource.so.10.9.0". Check the file itself.
    if trt_ok and sys.platform == "linux":
        import glob
        import site

        builder = []
        for base in site.getsitepackages() + [site.getusersitepackages()]:
            builder += glob.glob(os.path.join(base, "tensorrt*", "libnvinfer_builder_resource.so*"))
        print("  TensorRT builder resource:", "present" if builder else "MISSING")
        if not builder:
            print("  ⚠️ TensorRT loads, but it cannot build engines: the")
            print("     libnvinfer_builder_resource.so.10.* file is missing from the venv")
            print("     (usually a truncated install on a full disk). Free space, then:")
            print("     bash scripts/install_tensorrt.sh")
except Exception as e:
    print("  onnxruntime import failed:", e)
PY

# TensorRT was installed above, together with the pinned ORT/onnx stack.

echo "=== Creating results/checkpoints directories ==="
# Before the download: a failed download must not leave the box without the
# directories the app writes into.
mkdir -p results checkpoints

echo "=== Running checkpoint download script ==="
chmod +x "${SCRIPT_DIR}/download_checkpoints.sh"
# The weights live on the network (gdown / HuggingFace), so a flaky connection
# must not throw away an environment that is otherwise set up: warn instead of
# aborting, and let the user re-run scripts/download_checkpoints.sh.
if ! "${SCRIPT_DIR}/download_checkpoints.sh"; then
    echo "⚠️ checkpoint download failed — re-run scripts/download_checkpoints.sh when the network is back."
fi

echo "=== Setup completed successfully ==="
