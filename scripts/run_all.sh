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
# Unbuffered Python so backend errors appear in the combined log immediately.
export PYTHONUNBUFFERED=1

# --- Prerequisites -----------------------------------------------------------
command -v uv >/dev/null 2>&1 || { echo "❌ 'uv' not found on PATH. Run ./scripts/setup.sh first."; exit 1; }
command -v node >/dev/null 2>&1 || { echo "❌ 'node' not found on PATH (Node 20.19+ / 22.12+ required)."; exit 1; }
command -v npm >/dev/null 2>&1 || { echo "❌ 'npm' not found on PATH."; exit 1; }

if [ ! -d "${PROJECT_ROOT}/frontend/node_modules" ]; then
    echo "📦 frontend/node_modules missing — installing..."
    (cd "${PROJECT_ROOT}/frontend" && npm install)
fi

# --- Cleanup: Ctrl+C (or TERM) stops both servers -----------------------------
cleanup() {
    echo ""
    echo "🛑 Stopping backend + frontend..."
    # Kill the whole process group started by this script (backend + frontend).
    kill 0 2>/dev/null || true
}
trap cleanup INT TERM EXIT

# --- Backend ------------------------------------------------------------------
echo "🚀 Starting backend  →  http://localhost:${PORT} (docs: http://localhost:${PORT}/docs)"
echo "   (model load + warmup can take several minutes on first start)"
# 2>&1 merged and [api]-prefixed so backend errors are visible inline.
HYPERCORN_STARTUP_TIMEOUT=300 uv run hypercorn fast_float.main:app \
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
