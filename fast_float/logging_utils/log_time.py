# SPDX-License-Identifier: Apache-2.0
"""Timeslice logging decorator (sync and async)."""

from __future__ import annotations

import functools
import inspect
import logging
import time
from collections.abc import Callable
from typing import Any, TypeVar

_F = TypeVar("_F", bound=Callable[..., Any])


def logtime(logger: logging.Logger, msg: str | None = None, level: int = logging.DEBUG) -> Callable[[_F], _F]:
    """Log the execution time of the decorated function.

    Works on both regular and ``async def`` callables.  Place it beneath other
    decorators so it measures the actual work.  The default level is DEBUG, so
    ``FLOAT_DEBUG=1`` enables it without code changes.
    """

    def _inner(func: _F) -> _F:
        prefix = f"Function '{func.__module__}.{func.__qualname__}'" if msg is None else msg

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def _async_wrapper(*args: Any, **kwargs: Any) -> Any:
                start = time.perf_counter()
                try:
                    return await func(*args, **kwargs)
                finally:
                    logger.log(level, "%s: Elapsed time %.7f secs", prefix, time.perf_counter() - start)

            return _async_wrapper  # type: ignore[return-value]

        @functools.wraps(func)
        def _wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                logger.log(level, "%s: Elapsed time %.7f secs", prefix, time.perf_counter() - start)

        return _wrapper  # type: ignore[return-value]

    return _inner


__all__ = ["logtime"]
