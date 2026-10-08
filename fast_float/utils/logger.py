import logging
import sys


class ColoredFormatter(logging.Formatter):
    """Custom log formatter with ANSI colors and structured format."""

    grey = "\x1b[38;21m"
    blue = "\x1b[38;5;39m"
    yellow = "\x1b[38;5;226m"
    red = "\x1b[38;5;196m"
    bold_red = "\x1b[31;1m"
    green = "\x1b[38;5;46m"
    reset = "\x1b[0m"

    FORMATS = {
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


def setup_logger(name: str = "float_fast", level: int = logging.INFO) -> logging.Logger:
    """Setup or retrieve a logger with standard colored stream handling."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(ColoredFormatter())
        logger.addHandler(console_handler)

    # Suppress verbose 3rd party loggers
    logging.getLogger("libav").setLevel(logging.ERROR)
    logging.getLogger("ffmpeg").setLevel(logging.ERROR)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("numba").setLevel(logging.WARNING)
    logging.getLogger("numba.core").setLevel(logging.WARNING)
    logging.getLogger("numba.typed").setLevel(logging.WARNING)

    return logger


logger = setup_logger("fast_float")
