"""Backwards-compatible facade for application logging.

The implementation lives in :mod:`fast_float.logging_utils`:

* :func:`configure_logging` — console + per-process JSONL file, roles,
  request context, rotation (see ``docs/LOGGING.md``).
* :func:`setup_logger` / :func:`set_log_level` — legacy helpers kept so the
  Gradio app and older call sites keep working unchanged.
* :data:`logger` — the ``fast_float`` root logger.

Importing this module ensures logging is configured once (idempotent), so code
that only does ``from fast_float.utils.logger import logger`` still gets output
without an explicit entry-point call.
"""

from __future__ import annotations

from fast_float.logging_utils.config import (
    coerce_level,
    configure_logging,
    debug_enabled,
    get_logger,
    resolve_level,
    resolve_role,
    set_log_level,
    setup_logger,
)
from fast_float.logging_utils.formatter import ConsoleFormatter

# Historical name; the vLLM copy used to export its own class.
ColoredFormatter = ConsoleFormatter

logger = get_logger("fast_float")

__all__ = [
    "ColoredFormatter",
    "coerce_level",
    "configure_logging",
    "debug_enabled",
    "get_logger",
    "logger",
    "resolve_level",
    "resolve_role",
    "set_log_level",
    "setup_logger",
]
