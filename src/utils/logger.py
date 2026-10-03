"""
src/utils/logger.py
-------------------
Project-wide logging configuration.

Usage
-----
In any module::

    from src.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Processing record %s", record_id)

Calling ``setup_logging()`` once from the entry-point script configures
the root logger with a console handler and (optionally) a rotating file
handler.  All child loggers created with ``get_logger(__name__)`` inherit
this configuration automatically.

Log format
----------
``2024-01-15 10:23:45 | INFO     | src.data.loader:load_record:82 | Loading record 100``
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_logger(name: str) -> logging.Logger:
    """Return a named logger.

    Parameters
    ----------
    name:
        Typically ``__name__`` of the calling module.

    Returns
    -------
    logging.Logger
    """
    return logging.getLogger(name)


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str | Path] = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,
) -> None:
    """Configure the root logger.

    Should be called **once** at the start of any entry-point script
    (training, evaluation, UI, etc.).  Calling it multiple times is safe
    — duplicate handlers are removed first.

    Parameters
    ----------
    level:
        Logging level string: ``"DEBUG"``, ``"INFO"``, ``"WARNING"``,
        ``"ERROR"``, or ``"CRITICAL"``.
    log_file:
        Optional path to a rotating log file.  If *None*, only the
        console (stderr) handler is attached.
    max_bytes:
        Maximum size of a single log file before rotation.
    backup_count:
        Number of rotated log files to keep.

    Raises
    ------
    ValueError
        If *level* is not a recognised log level string.
    """
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(
            f"Invalid log level: '{level}'. "
            "Choose from DEBUG, INFO, WARNING, ERROR, CRITICAL."
        )

    root = logging.getLogger()
    # Remove any pre-existing handlers to avoid duplicate output.
    for handler in root.handlers[:]:
        root.removeHandler(handler)
        handler.close()

    root.setLevel(numeric_level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler (stderr so it doesn't pollute stdout pipelines)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    # Optional rotating file handler
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            filename=str(log_path),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    # Suppress overly verbose third-party loggers
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("tensorflow").setLevel(logging.WARNING)
    logging.getLogger("absl").setLevel(logging.WARNING)

    log = get_logger(__name__)
    log.debug(
        "Logging initialised | level=%s | file=%s",
        level.upper(),
        str(log_file) if log_file else "none",
    )
