"""Centralised logging configuration.

Every subsystem logs through the standard :mod:`logging` module using the
``maya1`` logger namespace so output is consistent and greppable:

    [2026-07-22 11:40:01] INFO  maya1.engine: Loading model maya-research/maya1
    [2026-07-22 11:40:33] INFO  maya1.api:    /v1/audio/speech text_len=42 ...

Call :func:`configure_logging` once at process start (the app factory does this).
"""

from __future__ import annotations

import logging
import sys

_LOG_FORMAT = "[%(asctime)s] %(levelname)-5s %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Root logger name for the whole package. Submodules use logging.getLogger(__name__).
ROOT_LOGGER_NAME = "maya1"


def configure_logging(level: str = "INFO") -> None:
    """Configure the ``maya1`` logger to write to stdout at ``level``.

    Idempotent: calling more than once will not attach duplicate handlers.
    """
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    logger.setLevel(level.upper())

    if not logger.handlers:
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
        logger.addHandler(handler)

    # Don't propagate to the root logger (avoids duplicate lines under uvicorn).
    logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the ``maya1`` namespace."""
    return logging.getLogger(name)
