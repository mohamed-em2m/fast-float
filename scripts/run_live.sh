#!/bin/bash
# Launch the FLOAT live + batch Gradio app (mic -> chunked near-real-time video).
# Usage: ./scripts/run_live.sh [--port 7860] [--share] ...
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
exec uv run --frozen python fast_float/interfaces/float_live_app.py "$@"
