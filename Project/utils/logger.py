"""
Centralized logging configuration for the Agentic RAG System.
Provides a single get_logger factory used by every module so log
formatting and destinations stay consistent across the project.
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler

from config import settings

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str) -> logging.Logger:
    """
    Create (or retrieve) a configured logger instance.

    Args:
        name: The name of the logger, typically __name__ of the caller.

    Returns:
        A configured logging.Logger instance with console and file handlers.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    try:
        log_file_path = os.path.join(settings.paths.logs_dir, "app.log")
        file_handler = RotatingFileHandler(
            log_file_path, maxBytes=2 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError:
        logger.warning("Could not attach file handler for logging; using console only.")

    logger.propagate = False
    return logger
