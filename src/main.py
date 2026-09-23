"""Run the project foundation checks."""

from __future__ import annotations

import logging

from src.configuration import load_settings
from src.database import initialize_database
from src.logging_config import configure_logging


def main() -> None:
    settings = load_settings()
    logger = configure_logging(settings.logging)
    initialize_database(settings.database)
    logger.info("Foundation initialized; trading mode is %s.", settings.trading_mode)


if __name__ == "__main__":
    main()
