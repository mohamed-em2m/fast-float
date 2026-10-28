# SPDX-License-Identifier: Apache-2.0
"""Logging filters: context injection, noisy-repeat suppression, redaction."""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Iterable
from urllib.parse import urlparse

from fast_float.logging_utils.context import current_context

# Values that must never reach a log file.  Matched case-insensitively against
# ``key=value`` / ``key: value`` / JSON ``"key": "value"`` shapes.
_SENSITIVE_KEY_RE = re.compile(
    r"(?i)\b(api[_-]?key|api[_-]?secret|secret|token|access[_-]?token|"
    r"refresh[_-]?token|password|passwd|authorization|credential|"
    r"livekit[_-]?api[_-]?secret|gemini[_-]?api[_-]?key|turn[_-]?secret)\b"
)

# ``key=value`` / ``key: value`` where the value is unquoted or quoted.
_REDACT_KV_RE = re.compile(
    r"(?i)(\b(?:api[_-]?key|api[_-]?secret|secret|token|access[_-]?token|"
    r"refresh[_-]?token|password|passwd|authorization|credential|"
    r"livekit[_-]?api[_-]?secret|gemini[_-]?api[_-]?key|turn[_-]?secret)\b"
    r"\s*[:=]\s*)([\"']?)([^\s\"'&,;]+)(\2)"
)
# ``Bearer <token>`` and query-string credentials.
_REDACT_BEARER_RE = re.compile(r"(?i)(\bbearer\s+)[A-Za-z0-9._~+/=-]+")
_REDACT_QUERY_RE = re.compile(r"(?i)([?&](?:access_token|token|api_key|key|secret)=)[^&\s]+")
# Google API keys (Gemini).
_REDACT_GOOGLE_KEY_RE = re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")

_REDACTED = "***"


def redact_text(text: str) -> str:
    """Mask credentials in an already-formatted log line."""
    if not text:
        return text
    text = _REDACT_BEARER_RE.sub(rf"\1{_REDACTED}", text)
    text = _REDACT_KV_RE.sub(lambda m: f"{m.group(1)}{m.group(2)}{_REDACTED}{m.group(4)}", text)
    text = _REDACT_QUERY_RE.sub(rf"\1{_REDACTED}", text)
    return _REDACT_GOOGLE_KEY_RE.sub(_REDACTED, text)


def is_sensitive_key(key: str) -> bool:
    """True when a structured field name looks like a credential."""
    return bool(_SENSITIVE_KEY_RE.search(key.replace("-", "_")))


class ContextFilter(logging.Filter):
    """Inject process role and request context into every record.

    The formatter-friendly derived fields are:

    ``process_role``
        ``api`` / ``worker`` / ``agent`` / ``cli`` / ``test``.
    ``context``
        ``"[req=… sess=… gen=…] "`` or ``""`` when nothing is bound.
    ``location``
        ``" @ file.py:123"`` for DEBUG records only, so INFO stays readable.
    ``req_id`` / ``session_id`` / ``gen`` / ``room``
        Always present (empty/0 defaults) so the JSON formatter is stable.
    """

    def __init__(self, role: str = "app") -> None:
        super().__init__()
        self.role = role

    def filter(self, record: logging.LogRecord) -> bool:
        ctx = current_context()
        req_id = getattr(record, "req_id", "") or ctx.req_id or ""
        session_id = getattr(record, "session_id", "") or ctx.session_id or ""
        gen = getattr(record, "gen", 0) or ctx.gen or 0
        room = getattr(record, "room", "") or ctx.room or ""

        record.process_role = getattr(record, "process_role", None) or self.role
        record.req_id = req_id
        record.session_id = session_id
        record.gen = gen
        record.room = room

        parts = []
        if req_id:
            parts.append(f"req={req_id}")
        if session_id:
            parts.append(f"sess={session_id}")
        if gen:
            parts.append(f"gen={gen}")
        if room:
            parts.append(f"room={room}")
        record.context = ("[" + " ".join(parts) + "] ") if parts else ""

        if record.levelno <= logging.DEBUG:
            record.location = f" @ {record.filename}:{record.lineno}"
        else:
            record.location = ""
        return True


class RateLimitFilter(logging.Filter):
    """Collapse repeated identical messages inside a time window.

    Attached to the console handler only: the rotating file keeps every record
    for post-mortems.  The first ``burst`` records in a window pass, one extra
    line announces the suppression, the rest are dropped.
    """

    def __init__(self, burst: int = 5, window_s: float = 10.0, max_keys: int = 512) -> None:
        super().__init__()
        self.burst = max(1, burst)
        self.window_s = max(0.1, window_s)
        self.max_keys = max(16, max_keys)
        self._state: dict[tuple, list[float]] = {}

    @staticmethod
    def _key(record: logging.LogRecord) -> tuple:
        args = record.args
        try:
            hash(args)
        except TypeError:
            args = str(args)
        return (record.name, record.levelno, str(record.msg), args)

    def filter(self, record: logging.LogRecord) -> bool:
        now = time.monotonic()
        key = self._key(record)
        state = self._state.get(key)
        if state is None or now - state[0] > self.window_s:
            if len(self._state) > self.max_keys:
                self._state.clear()
            self._state[key] = [now, 1.0, 0.0]
            return True

        state[1] += 1
        if state[1] <= self.burst:
            return True
        if state[2] == 0:
            state[2] = 1
            # Do not rewrite ``record.msg``: the same record is handed to the
            # file handler next, which must keep the original message.  The
            # console formatter picks this private note up (JSON skips _keys).
            record._console_note = f" (further similar records suppressed for {self.window_s:g}s)"
            return True
        return False


def _record_path(record: logging.LogRecord) -> str:
    """Best-effort request path from uvicorn or hypercorn access records."""
    args = record.args
    if isinstance(args, dict):
        # hypercorn AccessLogAtoms: U = path, Uq = path + query string.
        for key in ("U", "Uq", "r"):
            value = args.get(key)
            if isinstance(value, str) and value:
                return value
        return ""
    if isinstance(args, tuple) and len(args) >= 3 and isinstance(args[2], str):
        # uvicorn access: (client, method, path, version, status)
        return args[2]
    return ""


class HealthPathFilter(logging.Filter):
    """Drop access-log records for polling endpoints (``/health`` & friends)."""

    def __init__(self, excluded_paths: Iterable[str] | None = None) -> None:
        super().__init__()
        self.excluded_paths = set(excluded_paths or ())

    def filter(self, record: logging.LogRecord) -> bool:
        if not self.excluded_paths:
            return True
        raw = _record_path(record)
        if not raw:
            return True
        # Hypercorn's ``r`` atom is "GET /path HTTP/1.1"; peel the method off.
        if " " in raw:
            parts = raw.split()
            raw = parts[1] if len(parts) > 1 else raw
        path = urlparse(raw).path
        return path not in self.excluded_paths


__all__ = [
    "ContextFilter",
    "HealthPathFilter",
    "RateLimitFilter",
    "is_sensitive_key",
    "redact_text",
]
