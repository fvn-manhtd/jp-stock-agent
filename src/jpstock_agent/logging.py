"""Structured logging for jpstock-agent.

Provides JSON-formatted logging with contextual information for
debugging, monitoring, and observability.

Usage
-----
    from jpstock_agent.logging import get_logger

    logger = get_logger(__name__)
    logger.info("Fetching stock data", extra={"symbol": "7203", "source": "yfinance"})
"""

from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any


class JSONFormatter(logging.Formatter):
    """Format log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add extra fields (passed via extra={} in logging calls)
        for key in ("symbol", "source", "function", "duration_ms",
                     "error", "retry_attempt", "cache_hit", "params"):
            value = getattr(record, key, None)
            if value is not None:
                log_entry[key] = value

        # Add exception info if present
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, default=str, ensure_ascii=False)


def get_logger(name: str, level: int | None = None) -> logging.Logger:
    """Return a logger with JSON formatting.

    Parameters
    ----------
    name : str
        Logger name (typically ``__name__``).
    level : int, optional
        Logging level.  Defaults to ``logging.INFO``.
        Set to ``logging.DEBUG`` for verbose output.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)

    logger.setLevel(level or logging.INFO)
    return logger


class LogTimer:
    """Context manager for timing operations and logging duration.

    Usage
    -----
        with LogTimer(logger, "stock_history", symbol="7203"):
            result = fetch_data()
    """

    def __init__(self, logger: logging.Logger, operation: str, **context: Any):
        self.logger = logger
        self.operation = operation
        self.context = context
        self._start: float = 0.0

    def __enter__(self) -> LogTimer:
        self._start = time.perf_counter()
        self.logger.debug(
            f"Starting {self.operation}",
            extra={"function": self.operation, **self.context},
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        duration_ms = round((time.perf_counter() - self._start) * 1000, 2)
        extra = {
            "function": self.operation,
            "duration_ms": duration_ms,
            **self.context,
        }

        if exc_type is not None:
            extra["error"] = f"{exc_type.__name__}: {exc_val}"
            self.logger.warning(
                f"{self.operation} failed after {duration_ms}ms",
                extra=extra,
            )
        else:
            self.logger.info(
                f"{self.operation} completed in {duration_ms}ms",
                extra=extra,
            )
        return False  # Don't suppress exceptions
