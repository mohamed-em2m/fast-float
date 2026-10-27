# SPDX-License-Identifier: Apache-2.0
"""ASGI-server logging configs (uvicorn / hypercorn) wired to our formatters.

The app itself is served by hypercorn in production; uvicorn is a supported
alternative (``fast-float serve --server-type uvicorn``).  Both servers install
their own handlers before/around the app import, so the app re-routes their
loggers at startup (:func:`fast_float.logging_utils.config.reroute_server_loggers`).
These dict factories are the explicit alternative for the
``hypercorn --log-config json:<file>`` / ``uvicorn(log_config=...)`` paths.
"""

from __future__ import annotations

from typing import Literal

from fast_float.logging_utils.filters import ContextFilter, HealthPathFilter


def _formatter_config(formatter: Literal["text", "json"], *, fmt: str | None = None) -> dict:
    if formatter == "json":
        return {
            "()": "fast_float.logging_utils.formatter.JsonFormatter",
            "redact": True,
        }
    entry: dict = {
        "()": "fast_float.logging_utils.formatter.ConsoleFormatter",
        "color": False,
    }
    if fmt is not None:
        entry["fmt"] = fmt
    return entry


def create_uvicorn_log_config(
    excluded_paths: list[str] | None = None,
    log_level: str = "info",
    formatter: Literal["text", "json"] = "text",
    role: str = "api",
) -> dict:
    """A ``uvicorn.Config(log_config=...)`` compatible dict."""
    excluded = list(excluded_paths or ())
    default_formatter: dict
    access_formatter: dict
    if formatter == "json":
        default_formatter = _formatter_config("json")
        access_formatter = _formatter_config("json")
    else:
        # Keep uvicorn's own formatters: AccessFormatter is required for the
        # %(request_line)s / %(status_code)s fields.
        default_formatter = {
            "()": "uvicorn.logging.DefaultFormatter",
            "fmt": "%(levelprefix)s %(message)s",
            "use_colors": None,
        }
        access_formatter = {
            "()": "uvicorn.logging.AccessFormatter",
            "fmt": '%(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s',
        }

    config: dict = {
        "version": 1,
        "disable_existing_loggers": False,
        "filters": {
            "access_log_filter": {
                "()": HealthPathFilter,
                "excluded_paths": excluded,
            },
            "context": {
                "()": ContextFilter,
                "role": role,
            },
        },
        "formatters": {"default": default_formatter, "access": access_formatter},
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
                "filters": ["context"],
            },
            "access": {
                "formatter": "access",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
                "filters": ["access_log_filter", "context"],
            },
        },
        "loggers": {
            "uvicorn": {"handlers": ["default"], "level": log_level.upper(), "propagate": False},
            "uvicorn.error": {"handlers": ["default"], "level": log_level.upper(), "propagate": False},
            "uvicorn.access": {"handlers": ["access"], "level": log_level.upper(), "propagate": False},
        },
    }
    return config


def create_hypercorn_log_config(
    excluded_paths: list[str] | None = None,
    log_level: str = "info",
    formatter: Literal["text", "json"] = "text",
    log_file: str | None = None,
    max_bytes: int = 20 * 1024 * 1024,
    backup_count: int = 3,
    role: str = "api",
) -> dict:
    """A ``hypercorn.Config.logconfig_dict`` compatible dict.

    Console/stderr always; ``log_file`` optionally adds a rotating JSONL sink
    for the server's own records (``logs/server.log``).  The handler instance
    is shared by both server loggers, so rotation has a single writer — never
    point this at the application's ``logs/<role>.log`` while the app's own
    handler is also writing there.
    """
    handlers: dict[str, dict] = {
        "default": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",
            "filters": ["context"],
        },
        "access": {
            "formatter": "access",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
            "filters": ["access_log_filter", "context"],
        },
    }
    error_handlers = ["default"]
    access_handlers = ["access"]
    if log_file is not None:
        handlers["file"] = {
            "formatter": "json",
            "class": "logging.handlers.RotatingFileHandler",
            "filename": log_file,
            "maxBytes": max(0, max_bytes),
            "backupCount": max(0, backup_count),
            "encoding": "utf-8",
            "delay": True,
            "filters": ["context"],
        }
        error_handlers.append("file")
        access_handlers.append("file")

    return {
        "version": 1,
        "disable_existing_loggers": False,
        "filters": {
            "access_log_filter": {
                "()": HealthPathFilter,
                "excluded_paths": list(excluded_paths or ()),
            },
            "context": {
                "()": ContextFilter,
                "role": role,
            },
        },
        "formatters": {
            "default": _formatter_config(formatter),
            "access": _formatter_config(formatter),
            "json": _formatter_config("json"),
        },
        "handlers": handlers,
        "loggers": {
            "hypercorn.error": {
                "handlers": error_handlers,
                "level": log_level.upper(),
                "propagate": False,
            },
            "hypercorn.access": {
                "handlers": access_handlers,
                "level": log_level.upper(),
                "propagate": False,
            },
        },
    }


__all__ = ["create_hypercorn_log_config", "create_uvicorn_log_config"]
