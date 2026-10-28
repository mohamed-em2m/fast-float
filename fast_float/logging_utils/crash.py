# SPDX-License-Identifier: Apache-2.0
"""Crash diagnostics: faulthandler dumps and uncaught-exception hooks.

A CUDA worker that segfaults leaves nothing useful in a normal log file.  Each
entry point installs:

* ``faulthandler`` writing to ``logs/crash-<role>-<pid>.log`` (segfaults,
  aborts, ``SIGQUIT`` dumps), and
* hooks for uncaught exceptions in the main thread, worker threads, asyncio
  tasks and ``__del__`` (``sys.unraisablehook``).

Everything is idempotent and best-effort: a read-only checkout must not stop
the app from starting because ``logs/`` cannot be created.
"""

from __future__ import annotations

import asyncio
import contextlib
import faulthandler
import logging
import os
import sys
import threading
from pathlib import Path

_FAULT_FILE = None  # keep the file object alive for faulthandler
_INSTALLED = False

logger = logging.getLogger("fast_float.crash")


def _log_uncaught(exc_type, exc_value, exc_tb) -> None:
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_tb)
        return
    logger.critical("uncaught exception", exc_info=(exc_type, exc_value, exc_tb))


def _log_thread_exception(args: threading.ExceptHookArgs) -> None:
    if args.exc_type is SystemExit:
        return
    name = args.thread.name if args.thread is not None else "?"
    logger.critical(
        "uncaught exception in thread %s",
        name,
        exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
    )


def _log_unraisable(unraisable) -> None:
    exc = unraisable.exc_value
    logger.error(
        "unraisable exception in %s: %s",
        type(unraisable.object).__name__ if unraisable.object is not None else "?",
        unraisable.err_msg or "no message",
        exc_info=(type(exc), exc, exc.__traceback__) if exc is not None else None,
    )


def install_crash_handlers(role: str, log_dir: str | os.PathLike[str] | None) -> Path | None:
    """Enable faulthandler + exception hooks. Returns the crash-dump path.

    Safe to call from several entry points; only the first call installs the
    hooks, and a second call with a different role re-points faulthandler.
    """
    global _FAULT_FILE, _INSTALLED

    path: Path | None = None
    if log_dir is not None:
        try:
            directory = Path(log_dir)
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"crash-{role}-{os.getpid()}.log"
            # Kept open for the process lifetime; faulthandler writes to it.
            file = open(  # noqa: SIM115
                path, "a", encoding="utf-8", buffering=1, errors="backslashreplace"
            )
            if _FAULT_FILE is not None:
                with contextlib.suppress(Exception):
                    faulthandler.disable()
                with contextlib.suppress(Exception):
                    _FAULT_FILE.close()
            faulthandler.enable(file=file, all_threads=True)
            _FAULT_FILE = file
            logger.info("crash dumps -> %s", path)
        except Exception as exc:  # read-only fs, permission error, ...
            logger.warning("could not enable faulthandler crash dumps: %s", exc)
            path = None

    if not _INSTALLED:
        sys.excepthook = _log_uncaught
        threading.excepthook = _log_thread_exception
        if hasattr(sys, "unraisablehook"):
            sys.unraisablehook = _log_unraisable  # type: ignore[assignment]
        _INSTALLED = True
    return path


def install_asyncio_exception_handler(loop: asyncio.AbstractEventLoop) -> None:
    """Log unhandled task/transport errors, then defer to the previous handler."""
    previous = loop.get_exception_handler()

    def _handler(active_loop: asyncio.AbstractEventLoop, context: dict) -> None:
        exc = context.get("exception")
        logger.error(
            "unhandled asyncio exception: %s",
            context.get("message") or "no message",
            exc_info=(type(exc), exc, exc.__traceback__) if exc is not None else None,
        )
        if previous is not None:
            previous(active_loop, context)

    loop.set_exception_handler(_handler)


__all__ = ["install_asyncio_exception_handler", "install_crash_handlers"]
