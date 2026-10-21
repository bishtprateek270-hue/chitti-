"""
Chitti Logging Module.
Provides clean terminal logging with customizable levels and formatted tags.
"""

import sys
import logging
from typing import Optional


class CustomFormatter(logging.Formatter):
    """Custom formatter with clean bracketed prefixes."""

    PREFIX_MAP = {
        logging.DEBUG: "[DEBUG]",
        logging.INFO: "[CHITTI]",
        logging.WARNING: "[WARNING]",
        logging.ERROR: "[ERROR]",
        logging.CRITICAL: "[FATAL]",
    }

    def format(self, record: logging.LogRecord) -> str:
        prefix = self.PREFIX_MAP.get(record.levelno, "[CHITTI]")
        # Allow custom tag overrides via record.args if passed as a tag dict
        tag = getattr(record, "tag", None)
        if tag:
            prefix = f"[{tag}]"
        return f"{prefix} {record.getMessage()}"


def setup_logger(name: str = "chitti", level: str = "INFO") -> logging.Logger:
    """Configures and returns the application logger."""
    logger = logging.getLogger(name)
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    # Avoid duplicate handlers if reconfigured
    if not logger.handlers:
        import io
        if sys.stdout is None:
            sys.stdout = io.StringIO()
        if hasattr(sys.stdout, "reconfigure"):
            try:
                sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(numeric_level)
        handler.setFormatter(CustomFormatter())
        logger.addHandler(handler)

        # File handler for persistent background daemon diagnostics
        try:
            from logging.handlers import RotatingFileHandler
            log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            log_file = os.path.join(log_dir, "chitti_daemon.log")
            file_handler = RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=2, encoding="utf-8")
            file_handler.setLevel(numeric_level)
            file_formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        except Exception:
            pass

        logger.propagate = False
    else:
        logger.setLevel(numeric_level)
        for h in logger.handlers:
            h.setLevel(numeric_level)

    return logger


# Default application logger
logger = setup_logger()


def log_state(state: str, message: str = ""):
    """Prints special state banners like [LISTENING], [THINKING], etc."""
    if message:
        print(f"\n[{state}] {message}")
    else:
        print(f"\n[{state}]")


def log_chitti(message: str):
    """Logs a standard Chitti output message."""
    logger.info(message)


def log_error(message: str, exc: Optional[Exception] = None):
    """Logs an error message gracefully."""
    logger.error(message)
    if exc and logger.isEnabledFor(logging.DEBUG):
        logger.exception(exc)


def log_warning(message: str):
    """Logs a warning message."""
    logger.warning(message)


def log_debug(message: str):
    """Logs a debug message."""
    logger.debug(message)


# Aliases for convenience
log_info = log_chitti
log_warn = log_warning
