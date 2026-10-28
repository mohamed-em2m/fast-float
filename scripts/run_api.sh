#!/bin/bash
# Launch the FLOAT FastAPI (WebRTC) server with hypercorn.
# Usage: ./scripts/run_api.sh  (override port with PORT=9000 ./scripts/run_api.sh)
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
PORT="${PORT:-8000}"
# Number of GPUs for decoding (read as opt.ngpus by fast_float.main).
export NGPUS="${NGPUS:-1}"
# Startup timeout via env (honored where hypercorn maps env -> config).
# Harmless otherwise: startup is fast since warmup runs in the background.
export HYPERCORN_STARTUP_TIMEOUT="${HYPERCORN_STARTUP_TIMEOUT:-1200}"
# Per-process JSONL log file (logs/api.log) + combined console capture.
export FLOAT_PROCESS_ROLE="${FLOAT_PROCESS_ROLE:-api}"
export FLOAT_LOG_DIR="${FLOAT_LOG_DIR:-${PROJECT_ROOT}/logs}"
export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8}"
mkdir -p "${FLOAT_LOG_DIR}"
# --no-sync: the lock pins the portable CPU onnxruntime, while setup.sh /
# convert_to_onnx.sh layer onnx + onnxruntime-gpu + TensorRT on top. A plain
# `uv run` re-syncs the venv and the server boots without a CUDA/TensorRT EP.
# Pass the repo-root .env to the API when present, same as run_all.sh: it is
# the fallback for boxes without fast_float/config/{settings,secerts}.toml
# (shared config files still win).
ENV_ARGS=()
if [ -f "${PROJECT_ROOT}/.env" ]; then
    ENV_ARGS=(--env-file "${PROJECT_ROOT}/.env")
fi
# Access logs are opt-in (FLOAT_ACCESS_LOG=1): one line per request, including
# every /progress poll, drowns the app's own records in logs/server.log.
ACCESS_LOG_ARGS=()
case "${FLOAT_ACCESS_LOG:-}" in
    [Tt][Rr][Uu][Ee] | [Yy][Ee][Ss] | [Oo][Nn] | 1) ACCESS_LOG_ARGS=(--access-logfile -) ;;
esac
exec uv run "${ENV_ARGS[@]}" --frozen --no-sync hypercorn fast_float.main:app \
    --bind "0.0.0.0:${PORT}" \
    --worker-class asyncio \
    --log-config "json:${PROJECT_ROOT}/config/logging.json" \
    ${ACCESS_LOG_ARGS[@]+"${ACCESS_LOG_ARGS[@]}"} \
    --log-level info \
    "$@"
