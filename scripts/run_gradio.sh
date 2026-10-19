#!/bin/bash
# Launch the FLOAT batch Gradio app (upload image + audio -> video).
# Usage: ./scripts/run_gradio.sh [--port 7860] [--share] ...
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
exec uv run --frozen --no-sync python fast_float/interfaces/gradio_interface.py "$@"
