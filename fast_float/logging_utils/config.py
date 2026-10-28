# SPDX-License-Identifier: Apache-2.0
"""Central logging configuration for the FLOAT stack.

One call configures every process::

    from fast_float.logging_utils import configure_logging

    configure_logging(role="api")

What you get by default:

* a human-readable console handler on stderr (colors only on a TTY),
* a rotating JSONL file per process: ``logs/<role>.log`` (50 MB x 5),
* request/session context on every record (see :mod:`.context`),
* quiet third-party loggers, crash dumps (see :mod:`.crash`).

Precedence (highest first): explicit arguments → environment → the
``[logging]`` section of ``fast_float/config/settings.toml`` → defaults.
The function is idempotent: calling it again only updates levels unless a
structural option (role, sink, format, rotation) actually changes.
"""

from __future__ import annotations

import contextlib
import dataclasses
import logging
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fast_float.logging_utils.context import current_context
from fast_float.logging_utils.crash import install_crash_handlers
from fast_float.logging_utils.filters import ContextFilter, HealthPathFilter
from fast_float.logging_utils.formatter import ConsoleFormatter, JsonFormatter
from fast_float.logging_utils.handlers import (
    DEFAULT_BACKUP_COUNT,
    DEFAULT_MAX_BYTES,
    build_console_handler,
    build_file_handler,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG_DIR = PROJECT_ROOT / "logs"

# Loggers owned by the application.  Names that do not live under
# ``fast_float.*`` are legacy roots used by the Gradio app and the Gemini
# worker; they are configured too so their records share the same sinks.
DEFAULT_EXTRA_LOGGERS = ("float_gemini", "float_live_app", "py.warnings")
SERVER_LOGGERS = ("hypercorn", "hypercorn.error", "hypercorn.access", "uvicorn", "uvicorn.error", "uvicorn.access")
HEALTH_PATHS = ("/health", "/perf/requests", "/perf/matrix", "/livekit/status", "/favicon.ico")

_TRUTHY = frozenset({"1", "true", "yes", "on", "debug"})
_FALSY = frozenset({"0", "false", "no", "off", "none", "disabled"})

_QUIET_LOGGERS: dict[str, int] = {
    "libav": logging.ERROR,
    "ffmpeg": logging.ERROR,
    "urllib3": logging.WARNING,
    "asyncio": logging.WARNING,
    "numba": logging.WARNING,
    "numba.core": logging.WARNING,
    "numba.typed": logging.WARNING,
    "httpcore": logging.WARNING,
    "httpx": logging.WARNING,
    "matplotlib": logging.WARNING,
    "PIL": logging.WARNING,
    "zmq": logging.WARNING,
    "aioice": logging.WARNING,
    "aiortc": logging.WARNING,
    "google_genai": logging.WARNING,
    "multipart": logging.WARNING,
}

_STATE_LOCK = threading.RLock()


# --------------------------------------------------------------------------- #
# Level / role resolution
# --------------------------------------------------------------------------- #
def coerce_level(level: int | str | None, default: int = logging.INFO) -> int:
    """Normalise an int/name log level to a numeric level."""
    if level is None:
        return default
    if isinstance(level, int):
        return level
    resolved = logging.getLevelName(str(level).strip().upper())
    return resolved if isinstance(resolved, int) else default


def _env_truthy(*names: str) -> bool:
    return any(os.environ.get(name, "").strip().lower() in _TRUTHY for name in names)


def debug_enabled() -> bool:
    """True when ``FLOAT_DEBUG``/``FLOAT_TIMING`` request debug mode."""
    return _env_truthy("FLOAT_DEBUG", "FLOAT_TIMING")


def access_log_enabled() -> bool:
    """True when ``FLOAT_ACCESS_LOG`` asks for per-request access lines.

    Off by default: every request (each ``/progress`` poll, each ``/stream``
    range read) writes one ``hypercorn.access``/``uvicorn.access`` record to
    the console and ``logs/server.log``, which buries the app's own records
    for no operational gain - the request middleware already logs the things
    worth keeping. Health-path polling is filtered even when this is on.
    """
    return _env_truthy("FLOAT_ACCESS_LOG")


def resolve_level(default: int = logging.INFO) -> int:
    """``FLOAT_LOG_LEVEL`` → ``FLOAT_DEBUG``/``FLOAT_TIMING`` → ``default``."""
    env = os.environ.get("FLOAT_LOG_LEVEL", "").strip()
    if env:
        return coerce_level(env, default)
    return logging.DEBUG if debug_enabled() else default


def resolve_role(explicit: str | None = None) -> str:
    """Pick the process role used in the format and the default file name."""
    if explicit:
        return explicit
    env = os.environ.get("FLOAT_PROCESS_ROLE", "").strip()
    if env:
        return env
    argv = " ".join(sys.argv).lower()
    if "pytest" in argv or "py.test" in argv:
        return "test"
    if "inference_worker" in argv:
        return "worker"
    if "gemini.agent" in argv or "gemini_agent" in argv:
        return "agent"
    if "float_live_app" in argv or "gradio" in argv:
        return "gradio"
    if "fast_float.main" in argv or "hypercorn" in argv or "uvicorn" in argv:
        return "api"
    if "fast_float.cli" in argv or "fast-float" in argv:
        return "cli"
    return "app"


# --------------------------------------------------------------------------- #
# Option resolution (explicit → env → settings.toml → default)
# --------------------------------------------------------------------------- #
def _toml_logging() -> dict[str, Any]:
    try:
        from fast_float.config.dynamic import load_settings

        section = load_settings().get("logging", {})
        return section if isinstance(section, dict) else {}
    except Exception:  # malformed operator edit must not break logging
        return {}


def _env_str(name: str) -> str | None:
    value = os.environ.get(name)
    return value.strip() if value is not None and value.strip() else None


def _env_bool(name: str) -> bool | None:
    raw = _env_str(name)
    if raw is None:
        return None
    lowered = raw.lower()
    if lowered in _TRUTHY:
        return True
    if lowered in _FALSY:
        return False
    return None


def _env_int(name: str) -> int | None:
    raw = _env_str(name)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _pick(explicit: Any, env_name: str, toml_value: Any, default: Any, *, cast=None) -> Any:
    if explicit is not None:
        return explicit
    env = _env_str(env_name)
    if env is not None:
        if cast is bool:
            parsed = _env_bool(env_name)
            if parsed is not None:
                return parsed
        elif cast is int:
            parsed_int = _env_int(env_name)
            if parsed_int is not None:
                return parsed_int
        else:
            return env
    if toml_value is not None and toml_value != "":
        if cast is bool:
            if isinstance(toml_value, bool):
                return toml_value
            return str(toml_value).strip().lower() in _TRUTHY
        if cast is int:
            try:
                return int(toml_value)
            except (TypeError, ValueError):
                return default
        return toml_value
    return default


def _resolve_level(level: int | str | None, debug: bool, toml: dict[str, Any]) -> int:
    if debug:
        return logging.DEBUG
    if level is not None:
        return coerce_level(level)
    env = os.environ.get("FLOAT_LOG_LEVEL", "").strip()
    if env:
        return coerce_level(env)
    if debug_enabled():
        return logging.DEBUG
    toml_level = toml.get("level")
    if isinstance(toml_level, str) and toml_level.strip():
        return coerce_level(toml_level)
    return logging.INFO


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LoggingState:
    """What the last :func:`configure_logging` installed."""

    level: int
    role: str
    root_loggers: tuple[str, ...]
    handlers: tuple[logging.Handler, ...]
    log_file: Path | None
    console: bool
    json_file: bool
    shared: bool
    color: bool | None
    rate_limit: bool
    quiet: bool
    max_bytes: int
    backup_count: int


_STATE: LoggingState | None = None


def _is_managed(name: str, roots: tuple[str, ...]) -> bool:
    return any(name == root or name.startswith(root + ".") for root in roots)


def _remove_owned(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        if getattr(handler, "_float_owned", False):
            logger.removeHandler(handler)


def _attach(logger: logging.Logger, handlers: tuple[logging.Handler, ...], level: int, *, propagate: bool) -> None:
    _remove_owned(logger)
    for handler in handlers:
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = propagate


def _route_server_loggers(handlers: tuple[logging.Handler, ...], level: int) -> None:
    """Make hypercorn/uvicorn logs use the same sinks and format.

    The servers configure their own handlers before importing the app (CLI), so
    their handlers are replaced on purpose rather than appended to.
    """
    access_filter = HealthPathFilter(HEALTH_PATHS)
    for name in SERVER_LOGGERS:
        logger = logging.getLogger(name)
        # An explicit server logging config (hypercorn --log-config / uvicorn
        # log_config) already writes its own file; do not replace it.
        if logger.handlers and _uses_our_formatters(logger):
            continue
        logger.handlers = list(handlers)
        logger.setLevel(level)
        logger.propagate = False
        if name.endswith(".access") and not any(isinstance(f, HealthPathFilter) for f in logger.filters):
            logger.addFilter(access_filter)


def _quiet_third_party() -> None:
    for name, level in _QUIET_LOGGERS.items():
        logging.getLogger(name).setLevel(level)
    logging.captureWarnings(True)


def _resolve_log_file(
    role: str, log_dir: Path, explicit_file: str | os.PathLike[str] | None, shared: bool
) -> Path | None:
    raw = explicit_file if explicit_file is not None else os.getenv("FLOAT_LOG_FILE")
    if raw is not None and str(raw).strip():
        raw_str = str(raw).strip()
        if raw_str.lower() in _FALSY:
            return None
        path = Path(raw_str).expanduser()
        return path if path.is_absolute() else log_dir / path
    return log_dir / ("shared.log" if shared else f"{role}.log")


def _structural_change(
    state: LoggingState,
    *,
    role: str,
    log_file: Path | None,
    console: bool,
    json_file: bool,
    shared: bool,
    color: bool | None,
    rate_limit: bool,
    quiet: bool,
    roots: tuple[str, ...],
    max_bytes: int,
    backup_count: int,
) -> bool:
    return (
        state.role != role
        or state.log_file != log_file
        or state.console != console
        or state.json_file != json_file
        or state.shared != shared
        or state.color != color
        or state.rate_limit != rate_limit
        or state.quiet != quiet
        or state.root_loggers != roots
        or state.max_bytes != max_bytes
        or state.backup_count != backup_count
    )


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def configure_logging(
    level: int | str | None = None,
    *,
    debug: bool = False,
    role: str | None = None,
    log_dir: str | os.PathLike[str] | None = None,
    console: bool | None = None,
    file: bool | None = None,
    json_file: bool | None = None,
    color: bool | None = None,
    max_bytes: int | None = None,
    backup_count: int | None = None,
    shared: bool | None = None,
    rate_limit: bool | None = None,
    quiet: bool | None = None,
    extra_loggers: tuple[str, ...] = (),
    unify_servers: bool | None = None,
    crash: bool | None = None,
    force: bool = False,
) -> int:
    """Configure console + per-process file logging. Returns the level."""
    global _STATE

    with _STATE_LOCK:
        toml = _toml_logging()
        resolved_role = resolve_role(role)
        resolved_level = _resolve_level(level, debug, toml)

        resolved_dir = Path(_pick(log_dir, "FLOAT_LOG_DIR", toml.get("dir"), str(DEFAULT_LOG_DIR))).expanduser()
        resolved_shared = bool(_pick(shared, "FLOAT_LOG_SHARED", toml.get("shared"), False, cast=bool))
        file_enabled = bool(_pick(file, "FLOAT_LOG_TO_FILE", toml.get("file"), resolved_role != "test", cast=bool))
        # Tests must not scribble into the repository's logs/ directory, no
        # matter what settings.toml says; pass file=True explicitly to opt in.
        if resolved_role == "test" and file is None and _env_str("FLOAT_LOG_TO_FILE") is None:
            file_enabled = False
        resolved_file = (
            _resolve_log_file(resolved_role, resolved_dir, os.environ.get("FLOAT_LOG_FILE"), resolved_shared)
            if file_enabled
            else None
        )
        resolved_json = bool(_pick(json_file, "FLOAT_LOG_JSON", toml.get("json"), True, cast=bool))
        resolved_console = bool(_pick(console, "FLOAT_LOG_CONSOLE", toml.get("console"), True, cast=bool))
        resolved_color = _pick(color, "FLOAT_LOG_COLOR", toml.get("color"), None, cast=bool)
        resolved_max_bytes = int(
            _pick(max_bytes, "FLOAT_LOG_MAX_BYTES", toml.get("max_bytes"), DEFAULT_MAX_BYTES, cast=int)
        )
        resolved_backups = int(
            _pick(backup_count, "FLOAT_LOG_BACKUPS", toml.get("backup_count"), DEFAULT_BACKUP_COUNT, cast=int)
        )
        resolved_rate_limit = bool(_pick(rate_limit, "FLOAT_LOG_RATE_LIMIT", toml.get("rate_limit"), True, cast=bool))
        resolved_quiet = bool(_pick(quiet, "FLOAT_LOG_QUIET", toml.get("quiet"), True, cast=bool))
        resolved_unify = bool(
            _pick(unify_servers, "FLOAT_LOG_UNIFY_SERVERS", toml.get("unify_servers"), True, cast=bool)
        )

        roots = ("fast_float", *DEFAULT_EXTRA_LOGGERS, *extra_loggers)
        roots = tuple(dict.fromkeys(roots))  # de-dupe, keep order

        state = _STATE
        if (
            state is not None
            and not force
            and not _structural_change(
                state,
                role=resolved_role,
                log_file=resolved_file,
                console=resolved_console,
                json_file=resolved_json,
                shared=resolved_shared,
                color=resolved_color,
                rate_limit=resolved_rate_limit,
                quiet=resolved_quiet,
                roots=roots,
                max_bytes=resolved_max_bytes,
                backup_count=resolved_backups,
            )
        ):
            _apply_level(state, resolved_level)
            return resolved_level

        # Rebuild: detach and close the handlers from the previous generation.
        if state is not None:
            for name in state.root_loggers + SERVER_LOGGERS:
                _remove_owned(logging.getLogger(name))
            for handler in state.handlers:
                with contextlib.suppress(Exception):
                    handler.close()

        context_filter = ContextFilter(resolved_role)
        handlers: list[logging.Handler] = []

        if resolved_console:
            console_handler = build_console_handler(
                resolved_level,
                color=resolved_color,
                rate_limit=resolved_rate_limit,
                filters=[context_filter],
            )
            console_handler._float_owned = True  # type: ignore[attr-defined]
            handlers.append(console_handler)

        if resolved_file is not None:
            file_handler = build_file_handler(
                resolved_file,
                resolved_level,
                json=resolved_json,
                max_bytes=resolved_max_bytes,
                backup_count=resolved_backups,
                shared=resolved_shared,
                filters=[context_filter],
            )
            file_handler._float_owned = True  # type: ignore[attr-defined]
            handlers.append(file_handler)

        handler_tuple = tuple(handlers)
        propagate = not handler_tuple

        for name in roots:
            _attach(logging.getLogger(name), handler_tuple, resolved_level, propagate=propagate)

        if resolved_unify:
            if handler_tuple:
                _route_server_loggers(handler_tuple, resolved_level)
            else:
                for name in SERVER_LOGGERS:
                    logger = logging.getLogger(name)
                    logger.handlers = []
                    logger.propagate = True

        if resolved_quiet:
            _quiet_third_party()

        _STATE = LoggingState(
            level=resolved_level,
            role=resolved_role,
            root_loggers=roots,
            handlers=handler_tuple,
            log_file=resolved_file,
            console=resolved_console,
            json_file=resolved_json,
            shared=resolved_shared,
            color=resolved_color,
            rate_limit=resolved_rate_limit,
            quiet=resolved_quiet,
            max_bytes=resolved_max_bytes,
            backup_count=resolved_backups,
        )

        if crash is None:
            crash = resolved_role not in {"test", "app"}
        if crash:
            install_crash_handlers(resolved_role, resolved_dir)

        return resolved_level


def _apply_level(state: LoggingState, level: int) -> None:
    global _STATE
    for name in state.root_loggers + SERVER_LOGGERS:
        logging.getLogger(name).setLevel(level)
    for handler in state.handlers:
        handler.setLevel(level)
    _STATE = dataclasses.replace(state, level=level)


def get_logger(name: str = "fast_float") -> logging.Logger:
    """Return a module logger, configuring the app roots on first use."""
    if _STATE is None:
        configure_logging()
    return logging.getLogger(name)


def setup_logger(name: str = "fast_float", level: int | str | None = None) -> logging.Logger:
    """Backwards-compatible helper: configure logging and return ``name``.

    Kept for the Gradio app and older call sites.  A top-level logger that is
    not under a managed root (e.g. ``float_live_app``) is adopted, so its
    records reach the same console/file sinks.
    """
    configure_logging()
    state = _STATE
    if state is not None and not _is_managed(name, state.root_loggers):
        configure_logging(extra_loggers=(name,), force=True)
    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(coerce_level(level))
    return logger


def set_log_level(level: int | str) -> int:
    """Change the level of every managed logger (used by ``--debug``)."""
    resolved = coerce_level(level)
    if _STATE is None:
        return configure_logging(level=resolved)
    with _STATE_LOCK:
        _apply_level(_STATE, resolved)
    return resolved


def current_log_file() -> Path | None:
    """The file handler's target, or ``None`` when file logging is off."""
    return _STATE.log_file if _STATE is not None else None


def _uses_our_formatters(logger: logging.Logger) -> bool:
    return any(isinstance(handler.formatter, (ConsoleFormatter, JsonFormatter)) for handler in logger.handlers)


def reroute_server_loggers() -> None:
    """Re-attach the app handlers to hypercorn/uvicorn loggers.

    Servers may install their own handlers around the app import.  This leaves
    a server alone when it is already configured with our formatters (the
    ``--log-config`` / ``logconfig_dict`` path writes ``logs/server.log``), and
    adopts it otherwise so its records share the app sinks.
    """
    state = _STATE
    if state is None or not state.handlers:
        return
    for name in SERVER_LOGGERS:
        logger = logging.getLogger(name)
        if logger.handlers and _uses_our_formatters(logger):
            continue
        logger.handlers = list(state.handlers)
        logger.setLevel(state.level)
        logger.propagate = False
        if name.endswith(".access") and not any(isinstance(f, HealthPathFilter) for f in logger.filters):
            logger.addFilter(HealthPathFilter(HEALTH_PATHS))


def current_role() -> str:
    return _STATE.role if _STATE is not None else resolve_role()


def current_context_fields() -> dict[str, Any]:
    """Convenience re-export used by entry points for startup banners."""
    return current_context().fields()


def _reset_for_tests() -> None:
    """Tear down handlers/state (tests only)."""
    global _STATE
    with _STATE_LOCK:
        state = _STATE
        if state is not None:
            for name in state.root_loggers + SERVER_LOGGERS:
                _remove_owned(logging.getLogger(name))
            for handler in state.handlers:
                with contextlib.suppress(Exception):
                    handler.close()
        _STATE = None


__all__ = [
    "DEFAULT_LOG_DIR",
    "LoggingState",
    "access_log_enabled",
    "coerce_level",
    "configure_logging",
    "current_context_fields",
    "current_log_file",
    "current_role",
    "debug_enabled",
    "get_logger",
    "reroute_server_loggers",
    "resolve_level",
    "resolve_role",
    "set_log_level",
    "setup_logger",
]
