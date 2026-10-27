# SPDX-License-Identifier: Apache-2.0
"""Console and JSON log formatters.

Two consumers, two formats:

* :class:`ConsoleFormatter` — human-readable, optional ANSI color on the level
  name only, millisecond timestamps, short source location in DEBUG.
* :class:`JsonFormatter` — one JSON object per line, no third-party
  dependency, stable keys for ``jq``/Loki/Vector ingestion.

Both redact obvious credentials (see :mod:`fast_float.logging_utils.filters`).
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any

from fast_float.logging_utils.filters import is_sensitive_key, redact_text

CONSOLE_FORMAT = (
    "%(asctime)s.%(msecs)03d | %(levelname)-8s | %(process_role)s:%(process)d"
    " | %(name)s%(location)s | %(context)s%(message)s"
)
CONSOLE_DATEFMT = "%Y-%m-%d %H:%M:%S"

_LEVEL_COLORS = {
    logging.DEBUG: "\x1b[37m",  # white
    logging.INFO: "\x1b[32m",  # green
    logging.WARNING: "\x1b[33m",  # yellow
    logging.ERROR: "\x1b[31m",  # red
    logging.CRITICAL: "\x1b[1;31m",  # bold red
}
_RESET = "\x1b[0m"

# Attributes that every LogRecord carries (plus the ones our filters add).
# Everything else was passed via ``logger.info(..., extra={...})``.
_STANDARD_ATTRS = frozenset(vars(logging.LogRecord("", 0, "", 0, "", (), None))) | {
    "message",
    "asctime",
    "context",
    "location",
    "process_role",
    "req_id",
    "session_id",
    "gen",
    "room",
    "taskName",
}

# Structured fields always emitted when non-empty (stable JSON consumers).
_CONTEXT_KEYS = ("req_id", "session_id", "gen", "room")


def _use_color() -> bool:
    """ANSI colors only for an interactive terminal that did not opt out."""
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FLOAT_NO_COLOR", "").strip().lower() in {"1", "true", "yes", "on"}:
        return False
    try:
        return bool(sys.stderr.isatty())
    except Exception:
        return False


def _ensure_defaults(record: logging.LogRecord) -> None:
    """Fill fields the format strings rely on when no ContextFilter ran."""
    d = record.__dict__
    d.setdefault("process_role", "app")
    d.setdefault("location", "")
    d.setdefault("context", "")
    d.setdefault("req_id", "")
    d.setdefault("session_id", "")


class ConsoleFormatter(logging.Formatter):
    """Colored, aligned, single-line-per-record console formatter."""

    def __init__(
        self,
        fmt: str = CONSOLE_FORMAT,
        datefmt: str = CONSOLE_DATEFMT,
        *,
        color: bool | None = None,
        redact: bool = True,
    ) -> None:
        super().__init__(fmt, datefmt, "%")
        self.color = _use_color() if color is None else color
        self.redact = redact

    def _prefix(self, record: logging.LogRecord) -> str:
        """The formatted head of the line, up to (excluding) the message."""
        saved_message = getattr(record, "message", "")
        record.message = ""
        try:
            return self.formatMessage(record)
        finally:
            record.message = saved_message

    def format(self, record: logging.LogRecord) -> str:
        _ensure_defaults(record)
        original_level = record.levelname
        if self.color:
            color = _LEVEL_COLORS.get(record.levelno)
            if color:
                record.levelname = f"{color}{original_level}{_RESET}"
        try:
            message = super().format(record)
            if "\n" in message:
                prefix = self._prefix(record)
                message = message.replace("\n", "\n" + prefix)
        finally:
            record.levelname = original_level
        note = getattr(record, "_console_note", "")
        if note:
            message += note
        return redact_text(message) if self.redact else message


class JsonFormatter(logging.Formatter):
    """One JSON object per line (JSONL), dependency-free."""

    def __init__(self, *, redact: bool = True, include_extras: bool = True) -> None:
        super().__init__()
        self.redact = redact
        self.include_extras = include_extras

    def _clean(self, value: Any, key: str = "") -> Any:
        if self.redact and key and is_sensitive_key(key):
            return "***"
        if isinstance(value, str):
            if len(value) > 4000:
                value = value[:4000] + "…"
            return redact_text(value) if self.redact else value
        if isinstance(value, (int, float, bool)) or value is None:
            return value
        if isinstance(value, (list, tuple)):
            return [self._clean(item) for item in value][:50]
        if isinstance(value, dict):
            return {str(k): self._clean(v, str(k)) for k, v in list(value.items())[:50]}
        return redact_text(repr(value)) if self.redact else repr(value)

    def format(self, record: logging.LogRecord) -> str:
        _ensure_defaults(record)
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
            "role": getattr(record, "process_role", "app"),
            "pid": record.process,
            "thread": record.threadName,
            "msg": record.getMessage(),
        }

        for key in _CONTEXT_KEYS:
            value = getattr(record, key, None)
            if value not in (None, "", 0):
                payload[key] = value

        if self.include_extras:
            for key, value in record.__dict__.items():
                if key in _STANDARD_ATTRS or key in payload or key.startswith("_"):
                    continue
                payload[key] = self._clean(value, key)

        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)

        if self.redact:
            payload["msg"] = redact_text(payload["msg"])
            if "exc" in payload:
                payload["exc"] = redact_text(payload["exc"])
        if len(payload["msg"]) > 8000:
            payload["msg"] = payload["msg"][:8000] + "…"

        return json.dumps(payload, ensure_ascii=False, default=str)


__all__ = [
    "CONSOLE_DATEFMT",
    "CONSOLE_FORMAT",
    "ConsoleFormatter",
    "JsonFormatter",
]
