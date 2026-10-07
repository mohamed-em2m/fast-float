#!/bin/bash
# Launch the FLOAT React Studio frontend dev server.
# Usage: ./scripts/run_frontend.sh
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}/frontend"
exec npm run dev -- --host 0.0.0.0 --port 3000
