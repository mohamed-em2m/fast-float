#!/bin/bash
# Launch the FLOAT FastAPI (WebRTC) server with hypercorn.
# Usage: ./scripts/run_api.sh  (override port with PORT=9000 ./scripts/run_api.sh)
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
PORT="${PORT:-8000}"
exec uv run --frozen hypercorn main:app \
    --bind "0.0.0.0:${PORT}" \
    --worker-class asyncio \
    --log-level info \
    "$@"
