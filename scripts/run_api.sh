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
exec uv run --frozen hypercorn fast_float.main:app \
    --bind "0.0.0.0:${PORT}" \
    --worker-class asyncio \
    --log-level info \
    "$@"
