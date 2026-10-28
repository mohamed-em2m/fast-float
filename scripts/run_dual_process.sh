#!/bin/bash
# Launch the FLOAT dual-process architecture:
#   1. FLOAT GPU Inference Worker (dedicated CUDA process via ZeroMQ/SHM)
#   2. Streaming Backend (FastAPI / hypercorn / LiveKit)
#   3. React Studio frontend
#   4. Optional Gemini Live worker (if configured)
#
# Usage:
#   ./scripts/run_dual_process.sh
#   PORT=8000 FRONTEND_PORT=3000 ./scripts/run_dual_process.sh
#
# Env overrides:
#   FLOAT_IPC_ENABLED       true (default: true for dual process)
#   FLOAT_ZMQ_CONTROL_PORT  worker REP socket port (default 5555)
#   FLOAT_ZMQ_FRAMES_PORT   worker PUB socket port (default 5556)
#   FLOAT_SHM_NAME          shared memory name (default float_frame_ring)
#   FLOAT_SHM_SLOTS         ring buffer capacity (default 64)
#   PORT                    backend port (default 8000)
#   FRONTEND_PORT           frontend dev port (default 3000)
#   VITE_API_URL            URL frontend uses to reach backend
#   LIVEKIT_START           auto (default) / yes / no
#   FLOAT_GEMINI_WORKER     auto (default) / yes / no
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# Enable IPC dual-process mode for both processes
export FLOAT_IPC_ENABLED="${FLOAT_IPC_ENABLED:-true}"
export FLOAT_ZMQ_CONTROL_PORT="${FLOAT_ZMQ_CONTROL_PORT:-5555}"
export FLOAT_ZMQ_FRAMES_PORT="${FLOAT_ZMQ_FRAMES_PORT:-5556}"
export FLOAT_SHM_NAME="${FLOAT_SHM_NAME:-float_frame_ring}"
export FLOAT_SHM_SLOTS="${FLOAT_SHM_SLOTS:-64}"

PORT="${PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-3000}"
export VITE_API_URL="${VITE_API_URL:-http://localhost:${PORT}}"
export CORS_ORIGINS="${CORS_ORIGINS:-http://localhost:${FRONTEND_PORT},http://127.0.0.1:${FRONTEND_PORT}}"
export NGPUS="${NGPUS:-1}"
export PYTHONUNBUFFERED=1
# Keep child logs UTF-8 (the console capture decodes UTF-8) so emojis in log
# lines do not raise UnicodeEncodeError when stdout is a pipe.
export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8}"

# --- Logging ------------------------------------------------------------------
# One rotating JSONL file per process (logs/worker.log, logs/api.log, ...) plus
# an optional interleaved human-readable timeline. This script's stdout is the
# single writer for combined.log, so no cross-process file lock is needed.
export FLOAT_LOG_DIR="${FLOAT_LOG_DIR:-${PROJECT_ROOT}/logs}"
mkdir -p "${FLOAT_LOG_DIR}"
case "${FLOAT_LOG_TEE:-1}" in
    0|false|no|off) ;;
    *) exec > >(tee -a "${FLOAT_LOG_DIR}/combined.log") 2>&1 ;;
esac

# --- LiveKit SFU (live-transport preview) -------------------------------------
LIVEKIT_START="${LIVEKIT_START:-auto}"
FLOAT_GEMINI_WORKER="${FLOAT_GEMINI_WORKER:-auto}"

livekit_configured_url() {
    local py="${PROJECT_ROOT}/.venv/bin/python"
    [ -x "${py}" ] || return 0
    (
        unset LIVEKIT_URL LIVEKIT_API_KEY LIVEKIT_API_SECRET
        "${py}" - <<'PY' 2>/dev/null
from fast_float.livekit.settings import LiveKitSettings

settings = LiveKitSettings.from_env()
print(settings.url if settings.configured else "")
PY
    ) || true
}

livekit_export_local_dev() {
    export LIVEKIT_API_KEY="${LIVEKIT_API_KEY:-devkey}"
    export LIVEKIT_API_SECRET="${LIVEKIT_API_SECRET:-secret}"
    export LIVEKIT_URL="${LIVEKIT_URL:-ws://127.0.0.1:7880}"
}

livekit_server_start() {
    if ! command -v docker >/dev/null 2>&1; then
        if [ "${LIVEKIT_START}" = "yes" ]; then
            echo "❌ LIVEKIT_START=yes but docker is not installed."
            return 1
        fi
        echo "ℹ️  docker not found — LiveKit SFU not started (live transport falls back to MP4)."
        return 0
    fi
    if docker compose -f "${PROJECT_ROOT}/docker-compose.livekit.yml" up -d; then
        echo "🟢 LiveKit SFU running at ${LIVEKIT_URL} (left running; stop with:"
        echo "   docker compose -f docker-compose.livekit.yml down)"
        if [ "${LIVEKIT_STUN:-no}" = "yes" ]; then
            docker compose -f "${PROJECT_ROOT}/docker-compose.livekit.yml" --profile stun up -d \
                && echo "🟢 STUN server running on 3478"
        fi
    else
        return 1
    fi
}

# --- Prerequisites -----------------------------------------------------------
command -v uv >/dev/null 2>&1 || { echo "❌ 'uv' not found on PATH. Run ./scripts/setup.sh first."; exit 1; }

if [ ! -x "${PROJECT_ROOT}/.venv/bin/python" ]; then
    echo "📦 .venv missing -> running full setup first (scripts/setup.sh)..."
    bash "${SCRIPT_DIR}/setup.sh"
fi

if [ "${LIVEKIT_START}" = "yes" ]; then
    livekit_export_local_dev
    livekit_server_start || { echo "LiveKit start failed — continuing without the live transport."; true; }
elif [ "${LIVEKIT_START}" = "auto" ]; then
    if [ -n "${LIVEKIT_URL:-}" ] || [ -n "$(livekit_configured_url)" ]; then
        echo "ℹ️  LiveKit endpoint already configured — not starting the local SFU container."
    else
        livekit_export_local_dev
        livekit_server_start || { echo "LiveKit start failed — continuing without the live transport."; true; }
    fi
fi

# --- Gemini Live worker -------------------------------------------------------
gemini_worker_ready() {
    local py="${PROJECT_ROOT}/.venv/bin/python"
    [ -x "${py}" ] || return 1
    "${py}" - <<'PY' >/dev/null 2>&1
from fast_float.livekit.settings import LiveKitSettings

settings = LiveKitSettings.from_env()
ready = (
    settings.configured
    and bool(settings.agent_name)
    and (settings.gemini_configured or settings.gemini.vertexai)
)
raise SystemExit(0 if ready else 1)
PY
}

gemini_worker_running() {
    pgrep -f "fast_float.livekit.gemini.agent start" >/dev/null 2>&1
}

AGENT_PID=""
if [ "${FLOAT_GEMINI_WORKER}" = "no" ]; then
    echo "ℹ️  Gemini worker disabled (FLOAT_GEMINI_WORKER=no)."
elif gemini_worker_ready; then
    if gemini_worker_running; then
        echo "ℹ️  Gemini Live worker already running — reusing it."
        AGENT_PID=""
    else
        echo "🤖 Starting Gemini Live worker..."
        FLOAT_PROCESS_ROLE=agent uv run --frozen --no-sync python -m fast_float.livekit.gemini.agent start 2>&1 \
            | sed -u 's/^/[agent] /' &
        AGENT_PID=$!
    fi
fi

# --- Node / pnpm frontend setup ----------------------------------------------
command -v node >/dev/null 2>&1 || { echo "❌ 'node' not found on PATH (Node 20.19+ / 22.12+ required). Run ./scripts/setup.sh."; exit 1; }
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/_frontend_env.sh"
if ! node_version_ok; then
    echo "❌ Node $(node -v) is too old for Vite 8 — need 20.19+ or 22.12+."
    echo "   Fix: re-run ./scripts/setup.sh (it upgrades Node when needed)."
    exit 1
fi
ensure_pnpm || { echo "❌ pnpm could not be enabled. Run ./scripts/setup.sh first."; exit 1; }

if ! pnpm_deps_current; then
    echo "📦 frontend dependencies missing or stale — installing with pnpm..."
    pnpm_install
fi

# shellcheck source=/dev/null
source "${PROJECT_ROOT}/scripts/setup_cuda_libs.sh"

# --- Cleanup Trap: Stop all processes -----------------------------------------
cleanup() {
    trap - INT TERM EXIT
    echo ""
    echo "🛑 Stopping GPU worker + streaming backend + frontend..."
    kill 0 2>/dev/null || true
}
trap cleanup INT TERM EXIT

UV_ENV_ARGS=()
if [ -f "${PROJECT_ROOT}/.env" ]; then
    UV_ENV_ARGS=(--env-file "${PROJECT_ROOT}/.env")
fi
# Access logs are opt-in (FLOAT_ACCESS_LOG=1); one line per request is noise.
ACCESS_LOG_ARGS=()
case "${FLOAT_ACCESS_LOG:-}" in
    [Tt][Rr][Uu][Ee] | [Yy][Ee][Ss] | [Oo][Nn] | 1) ACCESS_LOG_ARGS=(--access-logfile -) ;;
esac

# --- Process 1: FLOAT GPU Inference Worker -----------------------------------
echo "🧠 Starting Process 1: FLOAT GPU Inference Worker (ZeroMQ + Shared Memory)..."
echo "   (tcp://127.0.0.1:${FLOAT_ZMQ_CONTROL_PORT} ctrl, tcp://127.0.0.1:${FLOAT_ZMQ_FRAMES_PORT} frames)"
FLOAT_PROCESS_ROLE=worker uv run "${UV_ENV_ARGS[@]}" --frozen --no-sync python -m fast_float.workers.inference_worker 2>&1 \
    | sed -u 's/^/[worker] /' &
WORKER_PID=$!

# Brief pause to allow the worker to create the SHM ring and bind ZeroMQ sockets
sleep 2

# --- Process 2: Streaming Backend (FastAPI / LiveKit) -------------------------
echo "🚀 Starting Process 2: Streaming Backend → http://localhost:${PORT} (docs: http://localhost:${PORT}/docs)"
FLOAT_PROCESS_ROLE=api HYPERCORN_STARTUP_TIMEOUT=1200 uv run "${UV_ENV_ARGS[@]}" --frozen --no-sync hypercorn fast_float.main:app \
    --bind "0.0.0.0:${PORT}" \
    --worker-class asyncio \
    --log-config "json:${PROJECT_ROOT}/config/logging.json" \
    ${ACCESS_LOG_ARGS[@]+"${ACCESS_LOG_ARGS[@]}"} \
    --log-level info 2>&1 | sed -u 's/^/[api] /' &
BACKEND_PID=$!

# --- Process 3: React Studio Frontend -----------------------------------------
echo "🎨 Starting Process 3: Studio Frontend  → http://localhost:${FRONTEND_PORT} (api: ${VITE_API_URL})"
(cd "${PROJECT_ROOT}/frontend" && pnpm run dev --host 0.0.0.0 --port "${FRONTEND_PORT}") 2>&1 | sed -u 's/^/[web] /' &
FRONTEND_PID=$!

echo ""
echo "✅ Dual-Process Stack Active:"
echo "   - GPU Worker PID:        ${WORKER_PID}"
echo "   - Backend API PID:       ${BACKEND_PID}"
echo "   - Studio Frontend PID:   ${FRONTEND_PID}"
[ -n "${AGENT_PID}" ] && echo "   - Gemini Worker PID:     ${AGENT_PID}"
echo "   - Logs:                  ${FLOAT_LOG_DIR} (api.log, worker.log, combined.log)"
echo ""
echo "   Open http://localhost:${FRONTEND_PORT} in your browser. Press Ctrl+C to stop all."
echo ""

PIDS=("${WORKER_PID}" "${BACKEND_PID}" "${FRONTEND_PID}")
if [ -n "${AGENT_PID}" ]; then
    PIDS+=("${AGENT_PID}")
fi
wait -n "${PIDS[@]}" || true
