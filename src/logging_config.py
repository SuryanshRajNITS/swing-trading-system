"""Configure simple console and local-file application logging."""

from __future__ import annotations

import logging
from pathlib import Path

from src.configuration import LoggingSettings


def configure_logging(settings: LoggingSettings) -> logging.Logger:
    """Configure and return the root logger without emitting configuration data."""
    settings.path.parent.mkdir(parents=True, exist_ok=True)
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.level))

    # Make repeated setup deterministic (for example, in tests or a restarted CLI).
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(Path(settings.path), encoding="utf-8")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    return root_logger
