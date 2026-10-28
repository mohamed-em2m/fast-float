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
#   FLOAT_GEMINI_WORKER
#                   auto (default) = start the Gemini Live worker when LiveKit
#                   and Gemini are both configured; yes = require it; no = skip
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
# Keep child logs UTF-8 (the console capture decodes UTF-8).
export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8}"

# --- Logging ------------------------------------------------------------------
export FLOAT_LOG_DIR="${FLOAT_LOG_DIR:-${PROJECT_ROOT}/logs}"
mkdir -p "${FLOAT_LOG_DIR}"
case "${FLOAT_LOG_TEE:-1}" in
    0|false|no|off) ;;
    *) exec > >(tee -a "${FLOAT_LOG_DIR}/combined.log") 2>&1 ;;
esac

# --- LiveKit SFU (live-transport preview) -------------------------------------
# The backend reads LIVEKIT_URL/KEY/SECRET (LiveKitSettings.from_env) from the
# environment, then secerts.toml, then settings.toml. Do NOT export the local
# dev defaults unconditionally: that used to override a configured LiveKit
# Cloud deployment in settings.toml with ws://127.0.0.1:7880, so the browser
# tried to join a local SFU that was never started ("LiveKit SFU failed").
LIVEKIT_START="${LIVEKIT_START:-auto}"
FLOAT_GEMINI_WORKER="${FLOAT_GEMINI_WORKER:-auto}"

# The endpoint configured in the TOML files (env not exported yet). Uses the
# project venv once it exists; empty when nothing is configured.
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

# Local SFU: only export dev credentials when we are actually starting the
# compose container, so its keys and the backend's token keys agree.
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
        # Optional coturn STUN for the raw WebRTC transport (LIVEKIT_STUN=yes).
        # The API also has a built-in responder; set [webrtc].stun_urls to use
        # this container instead.
        if [ "${LIVEKIT_STUN:-no}" = "yes" ]; then
            docker compose -f "${PROJECT_ROOT}/docker-compose.livekit.yml" --profile stun up -d \
                && echo "🟢 STUN server running on 3478 (set [webrtc].stun_urls in fast_float/config/settings.toml)"
        fi
    else
        return 1
    fi
}

# --- Prerequisites -----------------------------------------------------------
command -v uv >/dev/null 2>&1 || { echo "❌ 'uv' not found on PATH. Run ./scripts/setup.sh first."; exit 1; }

# --- Auto-bootstrap: one command must run everything --------------------------
# No project venv yet -> run the full setup (uv, Node, deps, checkpoints)
# before starting anything.
if [ ! -x "${PROJECT_ROOT}/.venv/bin/python" ]; then
    echo "📦 .venv missing -> running the full setup first (scripts/setup.sh)..."
    bash "${SCRIPT_DIR}/setup.sh"
fi
# Resolve LiveKit only after setup, so the configured endpoint in
# settings.toml can be read without falling back to the local dev defaults.
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
# Same project/venv, separate process: LiveKit dispatches jobs to a registered
# worker. Only start it when both halves are configured; the Studio can still
# use the builtin engine otherwise.
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
    # pgrep -f matches full command lines, including this script's own whenever
    # the pattern appears in it — split the string so it can never self-match.
    # shellcheck disable=SC2140
    pgrep -f "fast_float.livekit.gemini.age"nt" start" >/dev/null 2>&1
}

AGENT_PID=""
if [ "${FLOAT_GEMINI_WORKER}" = "no" ]; then
    echo "ℹ️  Gemini worker disabled (FLOAT_GEMINI_WORKER=no)."
elif gemini_worker_ready; then
    if gemini_worker_running; then
        echo "ℹ️  Gemini Live worker already running — reusing it instead of starting a second one."
        echo "   (Stop that process first if you want a fresh worker: pkill -f 'fast_float.livekit.gemini.agent start')"
        AGENT_PID=""
    else
        echo "🤖 Starting Gemini Live worker (separate process, shared config)..."
        FLOAT_PROCESS_ROLE=agent uv run --frozen --no-sync python -m fast_float.livekit.gemini.agent start 2>&1 \
            | sed -u 's/^/[agent] /' &
        AGENT_PID=$!
    fi
else
    if [ "${FLOAT_GEMINI_WORKER}" = "yes" ]; then
        echo "❌ FLOAT_GEMINI_WORKER=yes but LiveKit + Gemini are not configured."
        echo "   Set [livekit] url/api_key/api_secret and [gemini].api_key in fast_float/config/secerts.toml"
        echo "   (or export LIVEKIT_* / GOOGLE_API_KEY)."
        exit 1
    fi
    echo "ℹ️  Gemini worker not started (LiveKit + Gemini not both configured)."
fi

# Vite 8 / rolldown import `node:util`'s `styleText`, which only exists on Node
# >= 20.12 (LTS 20.19+/22.12+). Node 18 resolves the import fine and then dies
# deep inside vite with "does not provide an export named 'styleText'", so
# check the real requirement up front and point at setup.sh (which installs
# Node 22 when the system Node is too old).
command -v node >/dev/null 2>&1 || { echo "❌ 'node' not found on PATH (Node 20.19+ / 22.12+ required). Run ./scripts/setup.sh."; exit 1; }
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/_frontend_env.sh"
if ! node_version_ok; then
    echo "❌ Node $(node -v) is too old for Vite 8 — need 20.19+ or 22.12+ (rolldown needs node:util.styleText)."
    echo "   Fix: re-run ./scripts/setup.sh (it upgrades Node when the system Node is too old)."
    exit 1
fi

# pnpm is provisioned by Corepack at the version pinned in package.json.
# `ensure_pnpm` is a no-op on every start after setup has done it once.
ensure_pnpm || { echo "❌ pnpm could not be enabled. Run ./scripts/setup.sh first."; exit 1; }

if ! pnpm_deps_current; then
    echo "📦 frontend dependencies missing or stale — installing with pnpm..."
    # --frozen-lockfile: the committed pnpm-lock.yaml is authoritative, so the
    # launcher can never silently install a different tree than setup.sh.
    pnpm_install
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
    echo "🛑 Stopping backend + frontend + Gemini worker (if running)..."
    # Kill the whole process group started by this script.
    kill 0 2>/dev/null || true
}
trap cleanup INT TERM EXIT

# --- Backend ------------------------------------------------------------------
echo "🚀 Starting backend  →  http://localhost:${PORT} (docs: http://localhost:${PORT}/docs)"
echo "   (model load + warmup can take several minutes on first start)"
# This launcher runs inference inside the API process: it never starts the
# ZeroMQ worker. settings.toml ships [ipc] enabled = true for the dual-process
# stack, so force the override off here — otherwise every live conversation
# builds a ZMQ generator and dies attaching `/float_frame_ring`
# (`FileNotFoundError: /float_frame_ring`). Use ./scripts/run_dual_process.sh
# for the worker + API split.
export FLOAT_IPC_ENABLED="${FLOAT_IPC_ENABLED:-false}"
# Pin to the lockfile: a floating `uv run` re-resolves on a stale lock and can
# drag in phantom deps (seen: onnxruntime==1.24.3 with no cp310 wheel).
# `--no-sync` is required: the lock pins the *portable* stack (onnx<1.17 + the
# CPU `onnxruntime`), but setup.sh / convert_to_onnx.sh layer the export stack
# (onnx 1.18 + onnxruntime-gpu + TensorRT) ON TOP of it. A plain `uv run`
# re-syncs the venv back to the lock, so the backend silently boots without the
# CUDA/TensorRT EPs ("Available Execution Providers: ['Azure','CPU']").
# Pass the repo-root .env to the API when present so it sees the same fallback
# values as the worker (shared secerts.toml/settings.toml still win).
UV_ENV_ARGS=()
if [ -f "${PROJECT_ROOT}/.env" ]; then
    UV_ENV_ARGS=(--env-file "${PROJECT_ROOT}/.env")
fi
# Access logs are opt-in (FLOAT_ACCESS_LOG=1); one line per request is noise.
ACCESS_LOG_ARGS=()
case "${FLOAT_ACCESS_LOG:-}" in
    [Tt][Rr][Uu][Ee] | [Yy][Ee][Ss] | [Oo][Nn] | 1) ACCESS_LOG_ARGS=(--access-logfile -) ;;
esac
FLOAT_PROCESS_ROLE=api HYPERCORN_STARTUP_TIMEOUT=1200 uv run "${UV_ENV_ARGS[@]}" --frozen --no-sync hypercorn fast_float.main:app \
    --bind "0.0.0.0:${PORT}" \
    --worker-class asyncio \
    --log-config "json:${PROJECT_ROOT}/config/logging.json" \
    ${ACCESS_LOG_ARGS[@]+"${ACCESS_LOG_ARGS[@]}"} \
    --log-level info 2>&1 | sed -u 's/^/[api] /' &

BACKEND_PID=$!

# --- Frontend ------------------------------------------------------------------
echo "🎨 Starting frontend →  http://localhost:${FRONTEND_PORT} (api: ${VITE_API_URL})"
(cd "${PROJECT_ROOT}/frontend" && pnpm run dev --host 0.0.0.0 --port "${FRONTEND_PORT}") 2>&1 | sed -u 's/^/[web] /' &
FRONTEND_PID=$!

echo ""
echo "✅ Running — backend PID ${BACKEND_PID}, frontend PID ${FRONTEND_PID}${AGENT_PID:+, gemini worker PID ${AGENT_PID}}."
echo "   Open http://localhost:${FRONTEND_PORT} in your browser. Press Ctrl+C to stop."
echo ""

# Wait on all; if any exits, shut the rest down via the EXIT trap.
PIDS=("${BACKEND_PID}" "${FRONTEND_PID}")
if [ -n "${AGENT_PID}" ]; then
    PIDS+=("${AGENT_PID}")
fi
wait -n "${PIDS[@]}" || true
