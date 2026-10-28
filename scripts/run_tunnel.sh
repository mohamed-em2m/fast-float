#!/bin/bash
# Expose the Studio frontend through a public HTTPS URL.
#
# Why this exists: the frontend is usually run on a remote/containerized box
# (Colab, a VM, a devcontainer). http://localhost:3000 there only exists inside
# that box, and a runtime port-proxy URL (e.g. *.prod.r.colab.dev) returns
# HTTP 404 whenever it cannot reach the container's port. A cloudflared quick
# tunnel dials out from inside the box instead, so it works behind NAT and
# container boundaries.
#
# It also gives the browser a real https origin, which is what makes the
# microphone available at all: Chrome exposes getUserMedia only on secure
# contexts (https or localhost), so the record tab is dead on a plain-http
# remote URL no matter what the frontend does.
#
# Usage:
#   ./scripts/run_tunnel.sh          # start (or reuse) a tunnel for :3000
#   FRONTEND_PORT=3001 ./scripts/run_tunnel.sh
#   ./scripts/run_tunnel.sh url      # print the current URL
#   ./scripts/run_tunnel.sh stop     # stop the tunnel
#
# Quick tunnels are anonymous and have no uptime guarantee: the hostname changes
# every time cloudflared starts, so treat the printed URL as ephemeral.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PORT="${FRONTEND_PORT:-3000}"
TARGET="http://127.0.0.1:${PORT}"
LOG_DIR="${FLOAT_LOG_DIR:-${PROJECT_ROOT}/logs}"
LOG_FILE="${LOG_DIR}/cloudflared.log"
URL_FILE="${LOG_DIR}/tunnel-url.txt"
# Earlier manual runs logged here; still read it so a tunnel started by hand is
# reused instead of starting a second one.
LEGACY_LOG="/tmp/cloudflared.log"
PATTERN="cloudflared tunnel --url ${TARGET}"

tunnel_pid() {
    pgrep -f "${PATTERN}" 2>/dev/null | head -1
}

# Last quick-tunnel hostname in a cloudflared log, if any.
url_from_log() {
    local file="$1"
    [ -f "${file}" ] || return 0
    grep -oE "https://[a-z0-9-]+\.trycloudflare\.com" "${file}" 2>/dev/null | tail -1
}

# Prefer the newest log: cloudflared may have been restarted by hand (logging
# to LEGACY_LOG) after a script run left an older hostname in LOG_FILE, and the
# stale hostname would 404 forever.
current_url() {
    local url="" best="" best_mtime=-1 mtime
    for f in "${LOG_FILE}" "${LEGACY_LOG}"; do
        [ -f "${f}" ] || continue
        url="$(url_from_log "${f}")"
        [ -n "${url}" ] || continue
        mtime="$(stat -c %Y "${f}" 2>/dev/null || echo 0)"
        if [ "${mtime}" -gt "${best_mtime}" ]; then
            best_mtime="${mtime}"
            best="${url}"
        fi
    done
    # Last resort: the hostname this script saved for its own tunnel.
    if [ -z "${best}" ] && [ -f "${URL_FILE}" ]; then
        best="$(tr -d '[:space:]' < "${URL_FILE}")"
    fi
    printf '%s' "${best}"
}

require_cloudflared() {
    command -v cloudflared >/dev/null 2>&1 && return 0
    echo "❌ 'cloudflared' not found on PATH."
    echo "   Install it once with:"
    echo "     curl -fsSL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb -o /tmp/cloudflared.deb"
    echo "     sudo dpkg -i /tmp/cloudflared.deb"
    return 1
}

cmd_url() {
    local url
    url="$(current_url)"
    if [ -z "${url}" ]; then
        echo "No tunnel URL recorded. Start one with: ./scripts/run_tunnel.sh"
        return 1
    fi
    if ! tunnel_pid >/dev/null; then
        echo "⚠️  ${url} is stale — cloudflared is not running. Start it with: ./scripts/run_tunnel.sh"
        return 1
    fi
    echo "${url}"
}

cmd_stop() {
    local pids
    pids="$(pgrep -f "${PATTERN}" 2>/dev/null || true)"
    if [ -z "${pids}" ]; then
        echo "No tunnel running for ${TARGET}."
        return 0
    fi
    # shellcheck disable=SC2086
    kill ${pids} 2>/dev/null || true
    rm -f "${URL_FILE}"
    echo "🛑 Stopped the tunnel for ${TARGET}."
}

cmd_start() {
    require_cloudflared || return 1

    if ! curl -s -o /dev/null --max-time 3 "${TARGET}/"; then
        echo "⚠️  Nothing is listening on ${TARGET} — start the frontend first:"
        echo "     ./scripts/run_all.sh    (or ./scripts/run_frontend.sh)"
        echo "   Continuing anyway; the tunnel will serve errors until it is up."
    fi

    if tunnel_pid >/dev/null; then
        local existing
        existing="$(current_url)"
        if [ -n "${existing}" ]; then
            echo "ℹ️  Tunnel already running for ${TARGET} — reusing it."
            echo "${existing}" > "${URL_FILE}"
            echo "🔗 ${existing}"
            return 0
        fi
        echo "⚠️  cloudflared is running but no URL was logged yet; check ${LOG_FILE}."
        return 1
    fi

    mkdir -p "${LOG_DIR}"
    # A fresh tunnel must not inherit a previous run's hostname from the log.
    : > "${LOG_FILE}"

    echo "🚇 Starting a quick tunnel for ${TARGET}..."
    nohup cloudflared tunnel --url "${TARGET}" --no-autoupdate > "${LOG_FILE}" 2>&1 &
    local pid=$!

    # cloudflared needs a couple of seconds to be assigned a hostname.
    local url="" i=0
    while [ "${i}" -lt 40 ]; do
        url="$(url_from_log "${LOG_FILE}")"
        [ -n "${url}" ] && break
        if ! kill -0 "${pid}" 2>/dev/null; then
            echo "❌ cloudflared exited before a URL was assigned. Last lines:"
            tail -5 "${LOG_FILE}"
            return 1
        fi
        sleep 0.5
        i=$((i + 1))
    done

    if [ -z "${url}" ]; then
        echo "❌ No URL assigned within 20s. Check ${LOG_FILE}."
        return 1
    fi

    echo "${url}" > "${URL_FILE}"
    echo "🔗 ${url}"
    echo "   (also saved to ${URL_FILE})"
    echo ""

    # The hostname is handed out before the edge route is live, so a first
    # request can 404/502 for a few seconds. Report what it actually serves.
    local code=""
    for i in 1 2 3 4 5 6; do
        code="$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 "${url}/" || true)"
        [ "${code}" = "200" ] && break
        sleep 2
    done
    case "${code}" in
        200) echo "✅ ${url} is serving the Studio." ;;
        "") echo "⚠️  ${url} did not answer yet; it may take a few more seconds." ;;
        *) echo "⚠️  ${url} answered HTTP ${code}. If the frontend is up, retry in a few seconds." ;;
    esac
}

case "${1:-start}" in
    start) cmd_start ;;
    url) cmd_url ;;
    stop) cmd_stop ;;
    *)
        echo "Usage: $0 [start|url|stop]"
        exit 2
        ;;
esac
