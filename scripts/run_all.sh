#!/bin/bash
# Launch the FLOAT backend (FastAPI/hypercorn) and the React Studio frontend
# together with a single command. Ctrl+C stops both.
#
# Usage:
#   ./scripts/run_all.sh
#   PORT=8000 FRONTEND_PORT=3000 ./scripts/run_all.sh
#
# Env overrides:
#   PORT            backend port (default 8000)
#   FRONTEND_PORT   frontend dev port (default 3000)
#   VITE_API_URL    URL the frontend uses to reach the backend
#                   (default http://localhost:${PORT})
#   CORS_ORIGINS    origins the backend allows
#                   (default http://localhost:${FRONTEND_PORT})
#   NGPUS           number of GPUs for decoding (default 1)
#   LIVEKIT_START   auto (default) = start the LiveKit SFU container when
#                   Docker is available; yes = require it; no = skip
#   LIVEKIT_URL / LIVEKIT_API_KEY / LIVEKIT_API_SECRET
#                   LiveKit SFU endpoint + dev credentials; must match the
#                   keys the LiveKit server was started with
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PORT="${PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-3000}"
export VITE_API_URL="${VITE_API_URL:-http://localhost:${PORT}}"
# Browsers treat localhost and 127.0.0.1 as different origins — allow both.
# The same-origin /api proxy normally avoids CORS entirely; this is a fallback
# for direct backend access. Override CORS_ORIGINS to add public origins
# (e.g. an ngrok host) when pointing VITE_API_URL straight at the backend.
export CORS_ORIGINS="${CORS_ORIGINS:-http://localhost:${FRONTEND_PORT},http://127.0.0.1:${FRONTEND_PORT}}"
export NGPUS="${NGPUS:-1}"
# Unbuffered Python so backend errors appear in the combined log immediately.
export PYTHONUNBUFFERED=1

# --- LiveKit SFU (live-transport preview) -------------------------------------
# The backend reads LIVEKIT_URL/KEY/SECRET (LiveKitSettings.from_env) and the
# docker-compose.livekit.yml server uses the same keys as its tokens, so both
# halves must agree: export one source of truth here. Defaults are the dev
# credentials the compose file also falls back to.
LIVEKIT_API_KEY="${LIVEKIT_API_KEY:-devkey}"
LIVEKIT_API_SECRET="${LIVEKIT_API_SECRET:-secret}"
LIVEKIT_URL="${LIVEKIT_URL:-ws://127.0.0.1:7880}"
export LIVEKIT_API_KEY LIVEKIT_API_SECRET LIVEKIT_URL
LIVEKIT_START="${LIVEKIT_START:-auto}"

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
        # Optional pure STUN for the raw WebRTC transport (LIVEKIT_STUN=yes)
        if [ "${LIVEKIT_STUN:-no}" = "yes" ]; then
            docker compose -f "${PROJECT_ROOT}/docker-compose.livekit.yml" --profile stun up -d \
                && echo "🟢 STUN server running on 3478 (set [webrtc].stun_urls in fast_float/config/settings.toml)"
        fi
    else
        return 1
    fi
}
if [ "${LIVEKIT_START}" = "auto" ] || [ "${LIVEKIT_START}" = "yes" ]; then
    livekit_server_start || { echo "LiveKit start failed — continuing without the live transport."; true; }
fi

# --- Prerequisites -----------------------------------------------------------
command -v uv >/dev/null 2>&1 || { echo "❌ 'uv' not found on PATH. Run ./scripts/setup.sh first."; exit 1; }

# --- Auto-bootstrap: one command must run everything --------------------------
# No project venv yet -> run the full setup (uv, Node, deps, checkpoints)
# before starting anything.
if [ ! -x "${PROJECT_ROOT}/.venv/bin/python" ]; then
    echo "📦 .venv missing -> running the full setup first (scripts/setup.sh)..."
    bash "${SCRIPT_DIR}/setup.sh"
fi
# Top up the LiveKit RTC SDK when a fresh `uv sync` removed it again (it is
# installed --no-deps on top of the lock, so uv sync can prune it silently).
if ! uv run --frozen --no-sync python -c "import livekit.rtc" 2>/dev/null; then
    echo "📦 LiveKit RTC SDK missing from the venv -> installing (--no-deps)..."
    uv pip install --python .venv/bin/python --no-deps "livekit>=1.1" \
        || echo "⚠️ livekit install failed — the live transport falls back to MP4."
fi

# Vite 8 / rolldown import `node:util`'s `styleText`, which only exists on Node
# >= 20.12 (LTS 20.19+/22.12+). Node 18 resolves the import fine and then dies
# deep inside vite with "does not provide an export named 'styleText'", so
# check the real requirement up front and point at setup.sh (which installs
# Node 22 when the system Node is too old).
command -v node >/dev/null 2>&1 || { echo "❌ 'node' not found on PATH (Node 20.19+ / 22.12+ required). Run ./scripts/setup.sh."; exit 1; }
node_version_ok() {
    local mm major minor
    mm="$(node -p 'process.versions.node.split(".").slice(0,2).join(".")' 2>/dev/null)" || return 1
    major="${mm%%.*}"
    minor="${mm##*.}"
    [ "${major}" -gt 22 ] && return 0
    [ "${major}" -eq 22 ] && [ "${minor}" -ge 12 ] && return 0
    [ "${major}" -eq 20 ] && [ "${minor}" -ge 19 ] && return 0
    return 1
}
if ! node_version_ok; then
    echo "❌ Node $(node -v) is too old for Vite 8 — need 20.19+ or 22.12+ (rolldown needs node:util.styleText)."
    echo "   Fix: re-run ./scripts/setup.sh (it upgrades Node when the system Node is too old)."
    exit 1
fi
command -v npm >/dev/null 2>&1 || { echo "❌ 'npm' not found on PATH."; exit 1; }

if [ ! -d "${PROJECT_ROOT}/frontend/node_modules" ]; then
    echo "📦 frontend/node_modules missing — installing..."
    # npm ci (not install): rebuilds exactly from the committed lockfile,
    # including the platform-native optional deps npm can silently skip.
    (cd "${PROJECT_ROOT}/frontend" && npm ci)
fi

# shellcheck source=/dev/null
source "${PROJECT_ROOT}/scripts/setup_cuda_libs.sh"
# --- Cleanup: Ctrl+C (or TERM) stops both servers -----------------------------
cleanup() {
    # Disarm first: `kill 0` signals this shell too, which would re-enter the
    # trap and print the banner in an endless loop (and abort with a core dump
    # when the launcher is stopped with SIGTERM).
    trap - INT TERM EXIT
    echo ""
    echo "🛑 Stopping backend + frontend..."
    # Kill the whole process group started by this script (backend + frontend).
    kill 0 2>/dev/null || true
}
trap cleanup INT TERM EXIT

# --- Backend ------------------------------------------------------------------
echo "🚀 Starting backend  →  http://localhost:${PORT} (docs: http://localhost:${PORT}/docs)"
echo "   (model load + warmup can take several minutes on first start)"
# Pin to the lockfile: a floating `uv run` re-resolves on a stale lock and can
# drag in phantom deps (seen: onnxruntime==1.24.3 with no cp310 wheel).
# `--no-sync` is required: the lock pins the *portable* stack (onnx<1.17 + the
# CPU `onnxruntime`), but setup.sh / convert_to_onnx.sh layer the export stack
# (onnx 1.18 + onnxruntime-gpu + TensorRT) ON TOP of it. A plain `uv run`
# re-syncs the venv back to the lock, so the backend silently boots without the
# CUDA/TensorRT EPs ("Available Execution Providers: ['Azure','CPU']").
HYPERCORN_STARTUP_TIMEOUT=1200 uv run --frozen --no-sync hypercorn fast_float.main:app \
    --bind "0.0.0.0:${PORT}" \
    --worker-class asyncio \
    --log-level info 2>&1 | sed -u 's/^/[api] /' &

BACKEND_PID=$!

# --- Frontend ------------------------------------------------------------------
echo "🎨 Starting frontend →  http://localhost:${FRONTEND_PORT} (api: ${VITE_API_URL})"
(cd "${PROJECT_ROOT}/frontend" && npm run dev -- --host 0.0.0.0 --port "${FRONTEND_PORT}") 2>&1 | sed -u 's/^/[web] /' &
FRONTEND_PID=$!

echo ""
echo "✅ Both servers running — backend PID ${BACKEND_PID}, frontend PID ${FRONTEND_PID}."
echo "   Open http://localhost:${FRONTEND_PORT} in your browser. Press Ctrl+C to stop both."
echo ""

# Wait on both; if either exits, shut the other down via the EXIT trap.
wait -n "${BACKEND_PID}" "${FRONTEND_PID}" || true
