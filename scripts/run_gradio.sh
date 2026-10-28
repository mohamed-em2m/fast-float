#!/bin/bash
# Launch the FLOAT batch Gradio app (upload image + audio -> video).
# Usage: ./scripts/run_gradio.sh [--port 7860] [--share] ...
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
# Pass the repo-root .env when present so the app sees the same fallback
# values as the API/Gemini worker (shared TOML config still wins).
ENV_ARGS=()
if [ -f "${PROJECT_ROOT}/.env" ]; then
    ENV_ARGS=(--env-file "${PROJECT_ROOT}/.env")
fi
# --no-sync: the lock pins the portable CPU onnxruntime, while setup.sh layers
# the export stack (onnx + onnxruntime-gpu + TensorRT) on top of it.
exec uv run "${ENV_ARGS[@]}" --frozen --no-sync python fast_float/legacy/gradio_interface.py "$@"
