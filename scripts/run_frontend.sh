#!/bin/bash
# Launch the FLOAT React Studio frontend dev server.
# Usage:
#   ./scripts/run_frontend.sh
#   FRONTEND_PORT=3000 ./scripts/run_frontend.sh
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PROJECT_ROOT

# Node + pnpm come from the shared helper so this launcher behaves exactly
# like run_all.sh / setup.sh.
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/_frontend_env.sh"
command -v node >/dev/null 2>&1 || { echo "❌ 'node' not found on PATH (Node 20.19+ / 22.12+ required). Run ./scripts/setup.sh."; exit 1; }
node_version_ok || { echo "❌ Node $(node -v) is too old for Vite 8 — need 20.19+ or 22.12+. Run ./scripts/setup.sh."; exit 1; }
ensure_pnpm || { echo "❌ pnpm could not be enabled. Run ./scripts/setup.sh first."; exit 1; }

if ! pnpm_deps_current; then
    echo "📦 frontend dependencies missing or stale — installing with pnpm..."
    pnpm_install || exit 1
fi

cd "${FRONTEND_DIR}"
# Same override run_all.sh honors; hardcoded before, so a standalone frontend
# start could not avoid colliding with another dev server on 3000.
PORT="${FRONTEND_PORT:-3000}"
exec pnpm run dev --host 0.0.0.0 --port "${PORT}"
