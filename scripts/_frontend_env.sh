#!/bin/bash
# Shared helpers for the React Studio frontend (pnpm).
#
# Sourced by setup.sh, run_all.sh, run_dual_process.sh and run_frontend.sh so
# every entry point agrees on the Node requirement, the pnpm version and the
# install flags. This file only defines functions; it has no side effects.
#
# pnpm is used instead of npm because its content-addressable store hardlinks
# packages into node_modules: the first install downloads once, later installs
# (and worktrees) reuse the same files instead of re-extracting them.

FRONTEND_DIR="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/frontend"

# Vite 8 / rolldown import node:util's styleText (Node >= 20.12), so the real
# floor is 20.19+/22.12+. Keep this in sync with frontend/package.json engines.
node_version_ok() {
    local mm="" major="" minor=""
    mm="$(node -p 'process.versions.node.split(".").slice(0,2).join(".")' 2>/dev/null)" || return 1
    major="${mm%%.*}"
    minor="${mm##*.}"
    [ "${major}" -gt 22 ] && return 0
    [ "${major}" -eq 22 ] && [ "${minor}" -ge 12 ] && return 0
    [ "${major}" -eq 20 ] && [ "${minor}" -ge 19 ] && return 0
    return 1
}

# The exact pnpm release is declared once in package.json ("packageManager"),
# which is what Corepack reads as well. Parsing it here keeps setup.sh and the
# launchers from silently drifting to different pnpm versions.
pnpm_version() {
    local from_pkg="" fallback="10.18.0"
    if command -v node >/dev/null 2>&1 && [ -f "${FRONTEND_DIR}/package.json" ]; then
        from_pkg="$(node -e \
            "process.stdout.write(require(process.argv[1]).packageManager || '')" \
            "${FRONTEND_DIR}/package.json" 2>/dev/null || true)"
        from_pkg="${from_pkg#pnpm@}"
        # Corepack may rewrite this to `pnpm@x.y.z+sha512...`; compare the
        # version, not the integrity suffix.
        from_pkg="${from_pkg%%+*}"
    fi
    printf '%s' "${from_pkg:-${fallback}}"
}

# Corepack ships with Node and provisions the pinned pnpm on demand, so setup
# never depends on whatever global pnpm happens to be installed. Returns
# non-zero only when neither Corepack nor npm can provide it.
ensure_pnpm() {
    local want="" have="" shim_dir=""
    want="$(pnpm_version)"
    # Corepack may ask "Do you want to download pnpm?" on a fresh box; setup.sh
    # is non-interactive and must not hang waiting for an answer.
    export COREPACK_ENABLE_DOWNLOAD_PROMPT=0

    if command -v pnpm >/dev/null 2>&1; then
        have="$(pnpm --version 2>/dev/null || true)"
        [ "${have}" = "${want}" ] && return 0
    fi

    if command -v corepack >/dev/null 2>&1; then
        # Install the shim in ~/.local/bin and prepend it to PATH: on a box
        # that already has an outdated global pnpm earlier on PATH, this makes
        # the Corepack shim (and therefore the pinned version) win. Fall back
        # to node's own bin dir if the directory is not usable.
        shim_dir="${HOME}/.local/bin"
        mkdir -p "${shim_dir}" 2>/dev/null || shim_dir=""
        if [ -z "${shim_dir}" ] \
            || ! corepack enable --install-directory "${shim_dir}" pnpm >/dev/null 2>&1; then
            corepack enable pnpm >/dev/null 2>&1 || true
        fi
        if [ -n "${shim_dir}" ]; then
            case ":${PATH}:" in
                *":${shim_dir}:"*) ;;
                *) export PATH="${shim_dir}:${PATH}" ;;
            esac
        fi
        corepack prepare "pnpm@${want}" --activate >/dev/null 2>&1 || true
        hash -r
    fi

    # Last resort for a Node build without Corepack.
    if ! command -v pnpm >/dev/null 2>&1 && command -v npm >/dev/null 2>&1; then
        npm install --global "pnpm@${want}" >/dev/null 2>&1 || true
        hash -r
    fi

    command -v pnpm >/dev/null 2>&1 || return 1
    have="$(pnpm --version 2>/dev/null || true)"
    [ "${have}" = "${want}" ]
}

# One install path for every entry point: the committed lockfile is
# authoritative, and cached metadata/tarballs are preferred over the network.
# pnpm hardlinks from its store, so an up-to-date rerun is sub-second.
pnpm_install() {
    local dir="${1:-${FRONTEND_DIR}}"
    (
        cd "${dir}" || exit 1
        export COREPACK_ENABLE_DOWNLOAD_PROMPT=0
        pnpm install --frozen-lockfile --prefer-offline --reporter=append-only
    )
}

# True when node_modules was produced by pnpm and is not older than the
# lockfile. Used by the launchers to skip the (cheap, but non-zero) install
# check on every start.
pnpm_deps_current() {
    local dir="${1:-${FRONTEND_DIR}}"
    [ -f "${dir}/node_modules/.modules.yaml" ] || return 1
    [ -f "${dir}/pnpm-lock.yaml" ] || return 1
    [ ! "${dir}/pnpm-lock.yaml" -nt "${dir}/node_modules/.modules.yaml" ]
}
