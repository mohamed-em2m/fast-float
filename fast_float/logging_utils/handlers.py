# SPDX-License-Identifier: Apache-2.0
"""Handler factories, including an opt-in cross-process locked file handler.

The default deployment writes **one file per process** (``logs/api.log``,
``logs/worker.log``, …) through :class:`logging.handlers.RotatingFileHandler`.
Rotation is safe there because there is a single writer.

``LockedFileHandler`` exists for the case where several processes must append
to *one* file and no supervisor/tee is available.  It serializes each record
with a thread lock plus an OS file lock on a sidecar ``<file>.lock``:

* POSIX: ``fcntl.flock``
* Windows: ``msvcrt.locking`` on the first byte

Two deliberate limits, both documented in ``docs/LOGGING.md``:

* The lock has a timeout; on timeout the record is written anyway (unlocked)
  and counted in ``unlocked_writes``.  A stalled GPU worker is worse than a
  possibly interleaved line.
* No in-process rotation.  ``RotatingFileHandler`` is *not* multi-process safe
  (two writers can roll over at once or keep writing to the renamed file), so
  shared mode uses dated names or external rotation instead.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
import threading
import time
from pathlib import Path

from fast_float.logging_utils.filters import RateLimitFilter
from fast_float.logging_utils.formatter import ConsoleFormatter, JsonFormatter

DEFAULT_MAX_BYTES = 50 * 1024 * 1024
DEFAULT_BACKUP_COUNT = 5

if os.name == "nt":  # pragma: no cover - platform split
    import msvcrt
else:  # pragma: no cover - platform split
    import fcntl


class LockedFileHandler(logging.FileHandler):
    """Append to one file from several processes, serialized by an OS lock."""

    def __init__(
        self,
        filename: str | os.PathLike[str],
        *,
        mode: str = "a",
        encoding: str = "utf-8",
        delay: bool = True,
        timeout: float = 0.25,
    ) -> None:
        super().__init__(filename, mode=mode, encoding=encoding, delay=delay)
        self.lock_path = f"{os.fspath(filename)}.lock"
        self.timeout = max(0.0, timeout)
        self.unlocked_writes = 0
        self._thread_lock = threading.Lock()
        # "+" keeps the fd writable; the byte range itself carries no data.
        # Kept open for the handler's lifetime; closed in ``close()``.
        self._lock_stream = open(self.lock_path, "a+b")  # noqa: SIM115

    # -- OS lock ---------------------------------------------------------
    def _try_lock(self) -> bool:
        fd = self._lock_stream.fileno()
        try:
            if os.name == "nt":  # pragma: no cover - Windows
                self._lock_stream.seek(0)
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            else:  # pragma: no cover - POSIX
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            return False

    def _acquire(self) -> bool:
        deadline = time.monotonic() + self.timeout
        while True:
            if self._try_lock():
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.002)

    def _release(self) -> None:
        fd = self._lock_stream.fileno()
        try:
            if os.name == "nt":  # pragma: no cover - Windows
                self._lock_stream.seek(0)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            else:  # pragma: no cover - POSIX
                fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass

    # -- logging ---------------------------------------------------------
    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
        except RecursionError:  # pragma: no cover - logging guard
            raise
        except Exception:
            self.handleError(record)
            return

        try:
            with self._thread_lock:
                if self.stream is None:
                    self.stream = self._open()
                locked = self._acquire()
                try:
                    self.stream.write(message + self.terminator)
                    self.flush()
                finally:
                    if locked:
                        self._release()
                if not locked:
                    self.unlocked_writes += 1
        except RecursionError:  # pragma: no cover - logging guard
            raise
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        try:
            self._lock_stream.close()
        finally:
            super().close()


def build_console_handler(
    level: int = logging.NOTSET,
    *,
    color: bool | None = None,
    rate_limit: bool = True,
    filters: list[logging.Filter] | None = None,
) -> logging.Handler:
    """stderr handler with the human-readable formatter."""
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(ConsoleFormatter(color=color))
    handler.setLevel(level)
    if rate_limit:
        handler.addFilter(RateLimitFilter())
    for filt in filters or ():
        handler.addFilter(filt)
    return handler


def build_file_handler(
    path: str | os.PathLike[str],
    level: int = logging.NOTSET,
    *,
    json: bool = True,
    max_bytes: int = DEFAULT_MAX_BYTES,
    backup_count: int = DEFAULT_BACKUP_COUNT,
    shared: bool = False,
    timeout: float = 0.25,
    filters: list[logging.Filter] | None = None,
) -> logging.Handler:
    """Rotating per-process handler, or a locked shared handler when ``shared``."""
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)

    if shared:
        handler: logging.Handler = LockedFileHandler(resolved, timeout=timeout)
    else:
        handler = logging.handlers.RotatingFileHandler(
            resolved,
            maxBytes=max(0, max_bytes),
            backupCount=max(0, backup_count),
            encoding="utf-8",
            delay=True,
        )
    handler.setFormatter(JsonFormatter() if json else ConsoleFormatter(color=False))
    handler.setLevel(level)
    for filt in filters or ():
        handler.addFilter(filt)
    return handler


__all__ = [
    "DEFAULT_BACKUP_COUNT",
    "DEFAULT_MAX_BYTES",
    "LockedFileHandler",
    "build_console_handler",
    "build_file_handler",
]
