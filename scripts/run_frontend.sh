#!/bin/bash
# Launch the FLOAT React Studio frontend dev server.
# Usage:
#   ./scripts/run_frontend.sh
#   FRONTEND_PORT=3000 ./scripts/run_frontend.sh
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}/frontend"
# Same override run_all.sh honors; hardcoded before, so a standalone frontend
# start could not avoid colliding with another dev server on 3000.
PORT="${FRONTEND_PORT:-3000}"
exec npm run dev -- --host 0.0.0.0 --port "${PORT}"
