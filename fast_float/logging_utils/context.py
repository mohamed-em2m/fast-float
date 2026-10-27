# SPDX-License-Identifier: Apache-2.0
"""Request/session context propagated to every log record.

The FLOAT stack is multi-process (API, GPU worker, Gemini agent), and one live
session spans at least two of them.  ``req_id``/``session_id`` already travel
over IPC (:mod:`fast_float.ipc.messages`), so binding them to a
:class:`contextvars.ContextVar` is enough to join the two sides in the logs.

``contextvars`` are copied into tasks created with ``asyncio.create_task`` and
into ``asyncio.to_thread``.  They are **not** copied by
``loop.run_in_executor`` unless the caller wraps the callable, so the worker
binds the context explicitly from the incoming :class:`IPCRequest`.
"""

from __future__ import annotations

import contextvars
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any


@dataclass(frozen=True)
class LogContext:
    """Immutable slice of request metadata attached to the current task."""

    req_id: str = ""
    session_id: str = ""
    gen: int = 0
    room: str = ""

    def fields(self) -> dict[str, Any]:
        """Non-empty fields, in log-friendly order."""
        out: dict[str, Any] = {}
        if self.req_id:
            out["req_id"] = self.req_id
        if self.session_id:
            out["session_id"] = self.session_id
        if self.gen:
            out["gen"] = self.gen
        if self.room:
            out["room"] = self.room
        return out


# LogContext is a frozen dataclass; the linter cannot see that it is immutable.
_CTX: contextvars.ContextVar[LogContext] = contextvars.ContextVar(
    "fast_float_log_context",
    default=LogContext(),  # noqa: B039
)


def current_context() -> LogContext:
    """The context bound to the current task/thread."""
    return _CTX.get()


def new_request_id() -> str:
    """Short id used when a client does not send ``X-Request-ID``."""
    return uuid.uuid4().hex[:12]


def set_log_context(**fields: Any) -> contextvars.Token:
    """Bind ``fields`` for the current task; returns a token for :func:`reset`."""
    clean = {key: value for key, value in fields.items() if value is not None}
    return _CTX.set(replace(_CTX.get(), **clean))


def reset_log_context(token: contextvars.Token) -> None:
    """Undo a previous :func:`set_log_context`."""
    _CTX.reset(token)


@contextmanager
def log_context(**fields: Any) -> Iterator[LogContext]:
    """Bind request metadata for the duration of the block.

    Example::

        with log_context(req_id=req.req_id, session_id=req.session_id):
            handle(req)
    """
    token = set_log_context(**fields)
    try:
        yield _CTX.get()
    finally:
        reset_log_context(token)


__all__ = [
    "LogContext",
    "current_context",
    "log_context",
    "new_request_id",
    "reset_log_context",
    "set_log_context",
]
