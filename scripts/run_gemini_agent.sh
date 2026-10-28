#!/bin/bash
# Launch the Gemini Live worker (LiveKit Agents) as its own process.
#
#   ./scripts/run_gemini_agent.sh
#
# It shares the project venv/lockfile with the API; LiveKit dispatches jobs to
# the registered agent. Configuration comes from
# fast_float/config/secerts.toml + settings.toml (single source of truth) with
# the repo-root .env as a fallback. Override the config directory for both
# processes with FLOAT_CONFIG_DIR.
#
# Use `start`, not the deprecated `dev`/`console` modes: those need PortAudio
# for local mic/speaker I/O, which the worker does not.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

if [ ! -x "${PROJECT_ROOT}/.venv/bin/python" ]; then
    echo "❌ .venv missing — run ./scripts/setup.sh first."
    exit 1
fi

exec uv run --frozen --no-sync python -m fast_float.livekit.gemini.agent start "$@"
