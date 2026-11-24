"""Logging for the whole project.

Levels:
  INFO  (default) : the main steps with counts, paths, durations and results.
  DEBUG (-v)      : additionally every detail (per sequence, per transform, resolved
                    config, Ultralytics overrides, ...), with module and line number.

Every module gets its logger with ``get_logger(__name__)``. All of them are children
of the project logger "main", so one ``setup_logging`` call configures everything.
Ultralytics' own messages are additionally written to the same log files.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import config

PROJECT_LOGGER = "main"
ULTRALYTICS_LOGGER = "ultralytics"  # ultralytics.utils.LOGGING_NAME

_FMT_INFO = "%(asctime)s | %(levelname)-7s | %(message)s"
_FMT_DEBUG = "%(asctime)s | %(levelname)-7s | %(name)s:%(lineno)d | %(message)s"
_DATEFMT = "%d-%m-%Y %H:%M:%S"


class _PlainFormatter(logging.Formatter):
    """Removes terminal color codes (Ultralytics uses them) so log files stay readable."""
    _ANSI = re.compile(r"\x1b\[[0-9;]*m")

    def format(self, record: logging.LogRecord) -> str:
        return self._ANSI.sub("", super().format(record))


def get_logger(name: str) -> logging.Logger:
    """Module logger below the project logger, e.g. get_logger(__name__) -> 'main.data_handler.formats'."""
    return logging.getLogger(f"{PROJECT_LOGGER}.{name}")


def _formatter(level: int) -> logging.Formatter:
    return _PlainFormatter(_FMT_DEBUG if level <= logging.DEBUG else _FMT_INFO, datefmt=_DATEFMT)


def setup_logging(level: int = logging.INFO, log_file: Path = config.LOGS_DIR / "output.log") -> logging.Logger:
    """Console + log file (appended) at `level`. Call once at program start."""
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(PROJECT_LOGGER)
    logger.setLevel(level)
    logger.propagate = False  # no duplicates via the root logger
    for h in list(logger.handlers):  # safe to call twice (e.g. in a notebook)
        logger.removeHandler(h)
        h.close()

    console = logging.StreamHandler()
    console.setFormatter(_formatter(level))
    logger.addHandler(console)
    add_log_file(log_file, level)
    return logger


def add_log_file(path: Path, level: int | None = None) -> logging.Handler:
    """Also write the project log and Ultralytics' log to `path` (e.g. a run folder).

    Returns the handler; remove it again with ``remove_log_file``.
    """
    level = logging.getLogger(PROJECT_LOGGER).level if level is None else level
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setLevel(level)
    handler.setFormatter(_formatter(level))
    logging.getLogger(PROJECT_LOGGER).addHandler(handler)
    logging.getLogger(ULTRALYTICS_LOGGER).addHandler(handler)  # keeps its own console output
    return handler


def remove_log_file(handler: logging.Handler) -> None:
    for name in (PROJECT_LOGGER, ULTRALYTICS_LOGGER):
        logging.getLogger(name).removeHandler(handler)
    handler.close()
