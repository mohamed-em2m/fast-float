# SPDX-License-Identifier: Apache-2.0
"""Pure-ASGI middleware that binds a request id to the logging context.

Deliberately *not* ``BaseHTTPMiddleware``: FLOAT streams MP4/WebRTC responses
and Starlette's BaseHTTPMiddleware wraps the response in an extra streaming
task, which interferes with large streaming bodies and client disconnects.
This middleware touches headers only.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from fast_float.logging_utils.context import log_context, new_request_id

logger = logging.getLogger("fast_float.http")

_HEADER_REQ = b"x-request-id"
_HEADER_SESSION = b"x-session-id"


def _header(scope: dict[str, Any], name: bytes) -> str:
    for key, value in scope.get("headers") or ():
        if key.lower() == name:
            return value.decode("latin-1")
    return ""


class RequestContextMiddleware:
    """Bind ``X-Request-ID``/``X-Session-ID`` (or a generated id) per request."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope.get("type") not in {"http", "websocket"}:
            await self.app(scope, receive, send)
            return

        req_id = _header(scope, _HEADER_REQ) or new_request_id()
        session_id = _header(scope, _HEADER_SESSION)
        method = scope.get("method", "WS")
        path = scope.get("path", "?")
        start = time.perf_counter()
        status = 0

        async def send_wrapper(message: dict) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = int(message.get("status", 0))
                headers = list(message.get("headers") or ())
                headers.append((b"x-request-id", req_id.encode("ascii")))
                message["headers"] = headers
            await send(message)

        with log_context(req_id=req_id, session_id=session_id):
            logger.debug("[http] -> %s %s", method, path)
            try:
                await self.app(scope, receive, send_wrapper)
            except Exception:
                logger.exception("[http] %s %s failed", method, path)
                raise
            finally:
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                logger.debug(
                    "[http] <- %s %s %s %.1fms",
                    method,
                    path,
                    status or "-",
                    elapsed_ms,
                )


__all__ = ["RequestContextMiddleware"]
