# SPDX-License-Identifier: Apache-2.0
"""Logging engine for the FLOAT stack.

Typical use in an entry point::

    from fast_float.logging_utils import configure_logging

    configure_logging(role="api")

and everywhere else::

    import logging

    logger = logging.getLogger(__name__)  # or fast_float.logging_utils.get_logger

See ``docs/LOGGING.md`` for the environment/TOML reference and the
multi-process (per-role file + supervisor tee) design.
"""

from fast_float.logging_utils.asgi import RequestContextMiddleware
from fast_float.logging_utils.config import (
    DEFAULT_LOG_DIR,
    LoggingState,
    coerce_level,
    configure_logging,
    current_log_file,
    current_role,
    debug_enabled,
    get_logger,
    reroute_server_loggers,
    resolve_level,
    resolve_role,
    set_log_level,
    setup_logger,
)
from fast_float.logging_utils.context import (
    LogContext,
    current_context,
    log_context,
    new_request_id,
    reset_log_context,
    set_log_context,
)
from fast_float.logging_utils.crash import (
    install_asyncio_exception_handler,
    install_crash_handlers,
)
from fast_float.logging_utils.filters import (
    ContextFilter,
    HealthPathFilter,
    RateLimitFilter,
    is_sensitive_key,
    redact_text,
)
from fast_float.logging_utils.formatter import ConsoleFormatter, JsonFormatter
from fast_float.logging_utils.handlers import (
    LockedFileHandler,
    build_console_handler,
    build_file_handler,
)
from fast_float.logging_utils.lazy import lazy
from fast_float.logging_utils.log_time import logtime
from fast_float.logging_utils.torch_tensor import tensors_str_no_data
from fast_float.logging_utils.uvicorn_logging import (
    create_hypercorn_log_config,
    create_uvicorn_log_config,
)

__all__ = [
    "DEFAULT_LOG_DIR",
    "ConsoleFormatter",
    "ContextFilter",
    "HealthPathFilter",
    "JsonFormatter",
    "LockedFileHandler",
    "LogContext",
    "LoggingState",
    "RateLimitFilter",
    "RequestContextMiddleware",
    "build_console_handler",
    "build_file_handler",
    "coerce_level",
    "configure_logging",
    "create_hypercorn_log_config",
    "create_uvicorn_log_config",
    "current_context",
    "current_log_file",
    "current_role",
    "debug_enabled",
    "get_logger",
    "install_asyncio_exception_handler",
    "install_crash_handlers",
    "is_sensitive_key",
    "lazy",
    "log_context",
    "logtime",
    "new_request_id",
    "redact_text",
    "reroute_server_loggers",
    "reset_log_context",
    "resolve_level",
    "resolve_role",
    "set_log_context",
    "set_log_level",
    "setup_logger",
    "tensors_str_no_data",
]
