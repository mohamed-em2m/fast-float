import logging
import os
import sys
from typing import ClassVar

# Logger names owned by this application. ``configure_logging`` / ``set_log_level``
# apply the requested level to every managed logger, so flipping debug mode once
# reaches module loggers (``fast_float.*``) *and* the app logger.
_MANAGED_LOGGERS: set[str] = {"fast_float"}

# Env values treated as "on" for the boolean debug/timing flags.
_TRUTHY: frozenset[str] = frozenset({"1", "true", "yes", "on", "debug"})

# Third-party loggers that stay quiet regardless of our level.
_QUIET_LOGGERS: dict[str, int] = {
    "libav": logging.ERROR,
    "ffmpeg": logging.ERROR,
    "urllib3": logging.WARNING,
    "asyncio": logging.WARNING,
    "numba": logging.WARNING,
    "numba.core": logging.WARNING,
    "numba.typed": logging.WARNING,
    "httpcore": logging.WARNING,
    "httpx": logging.WARNING,
    "matplotlib": logging.WARNING,
}


class ColoredFormatter(logging.Formatter):
    """Custom log formatter with ANSI colors and structured format."""

    grey = "\x1b[38;21m"
    blue = "\x1b[38;5;39m"
    yellow = "\x1b[38;5;226m"
    red = "\x1b[38;5;196m"
    bold_red = "\x1b[31;1m"
    green = "\x1b[38;5;46m"
    reset = "\x1b[0m"

    FORMATS: ClassVar[dict[int, str]] = {
        logging.DEBUG: grey + "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s" + reset,
        logging.INFO: green + "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s" + reset,
        logging.WARNING: yellow + "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s" + reset,
        logging.ERROR: red + "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s" + reset,
        logging.CRITICAL: bold_red + "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s" + reset,
    }

    def format(self, record: logging.LogRecord) -> str:
        log_fmt = self.FORMATS.get(record.levelno, self.FORMATS[logging.INFO])
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)


def _env_truthy(*names: str) -> bool:
    """True when any named env var carries an on-style value."""
    return any(os.environ.get(name, "").strip().lower() in _TRUTHY for name in names)


def debug_enabled() -> bool:
    """Whether debug mode is requested via the environment (``FLOAT_DEBUG``/``FLOAT_TIMING``)."""
    return _env_truthy("FLOAT_DEBUG", "FLOAT_TIMING")


def coerce_level(level: int | str | None, default: int = logging.INFO) -> int:
    """Normalise an int/name log level to a numeric level, falling back to ``default``."""
    if level is None:
        return default
    if isinstance(level, int):
        return level
    resolved = logging.getLevelName(str(level).strip().upper())
    return resolved if isinstance(resolved, int) else default


def resolve_level(default: int = logging.INFO) -> int:
    """Resolve the effective level: ``$FLOAT_LOG_LEVEL`` → ``$FLOAT_DEBUG`` → ``default``."""
    if os.environ.get("FLOAT_LOG_LEVEL", "").strip():
        return coerce_level(os.environ["FLOAT_LOG_LEVEL"], default)
    return logging.DEBUG if debug_enabled() else default


def _quiet_third_party() -> None:
    for name, lvl in _QUIET_LOGGERS.items():
        logging.getLogger(name).setLevel(lvl)


def setup_logger(name: str = "fast_float", level: int | str | None = None) -> logging.Logger:
    """Setup or retrieve a logger with standard colored stream handling.

    ``level`` defaults to :func:`resolve_level` (env-driven), so importing the
    package honours ``FLOAT_LOG_LEVEL`` / ``FLOAT_DEBUG`` without extra wiring.
    """
    logger = logging.getLogger(name)
    logger.setLevel(resolve_level() if level is None else coerce_level(level))

    # Remove stale handlers (important when the module is re-imported in a
    # Jupyter / Kaggle notebook without restarting the kernel).
    logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(ColoredFormatter())
    logger.addHandler(console_handler)

    # Prevent the root logger (which Gradio/IPython may rewire) from
    # intercepting our records and potentially swallowing them.
    logger.propagate = False

    _MANAGED_LOGGERS.add(name)

    # Suppress verbose 3rd party loggers
    _quiet_third_party()
    return logger


def set_log_level(level: int | str) -> int:
    """Bump every managed logger (``fast_float.*`` + app loggers) to ``level``.

    Children created with ``getLogger(__name__)`` inherit the level from the
    ``fast_float`` parent, so a single call flips the whole tree — this is what
    ``--debug`` uses instead of fiddling with the (unrelated) root logger.
    """
    lvl = coerce_level(level)
    for name in _MANAGED_LOGGERS:
        logging.getLogger(name).setLevel(lvl)
    return lvl


def configure_logging(level: int | str | None = None, *, debug: bool = False) -> int:
    """Apply a level to all managed loggers, installing the handler if needed.

    Priority: ``debug=True`` → ``level`` → ``$FLOAT_LOG_LEVEL`` → ``$FLOAT_DEBUG``
    → INFO. Safe to call repeatedly; returns the effective numeric level.
    """
    if debug:
        lvl = logging.DEBUG
    elif level is not None:
        lvl = coerce_level(level)
    else:
        lvl = resolve_level()
    for name in _MANAGED_LOGGERS:
        lg = logging.getLogger(name)
        if not lg.handlers:
            setup_logger(name)
        lg.setLevel(lvl)
    return lvl


logger = setup_logger("fast_float")
