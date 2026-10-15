#!/bin/bash
# Launch the FLOAT FastAPI (WebRTC) server with hypercorn.
# Usage: ./scripts/run_api.sh  (override port with PORT=9000 ./scripts/run_api.sh)
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
PORT="${PORT:-8000}"
# Model load alone can exceed hypercorn's 60s default startup timeout, so
# allow a long startup window (warmup itself now runs in the background).
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-1200}"
exec uv run --frozen hypercorn fast_float.main:app \
    --bind "0.0.0.0:${PORT}" \
    --worker-class asyncio \
    --startup-timeout "${STARTUP_TIMEOUT}" \
    --log-level info \
    "$@"
