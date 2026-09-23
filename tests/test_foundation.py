from __future__ import annotations

import logging
from pathlib import Path

import pytest

from src.configuration import DatabaseSettings, LoggingSettings, load_settings
from src.database import database_is_healthy, initialize_database
from src.logging_config import configure_logging


def test_default_settings_are_loaded_and_live_trading_is_disabled() -> None:
    settings = load_settings()

    assert settings.environment == "development"
    assert settings.trading_mode == "paper"
    assert settings.live_trading_enabled is False
    assert settings.initial_capital == 10000
    assert settings.max_risk_per_trade == 0.01
    assert settings.timezone == "Asia/Kolkata"


def test_invalid_configuration_fails_clearly(tmp_path: Path) -> None:
    settings_file = tmp_path / "settings.yaml"
    settings_file.write_text("live_trading_enabled: maybe\n", encoding="utf-8")

    with pytest.raises(ValueError, match="environment"):
        load_settings(settings_file)


def test_database_initialization_and_health_check(tmp_path: Path) -> None:
    database = DatabaseSettings(tmp_path / "nested" / "foundation.sqlite3")

    initialize_database(database)

    assert database.path.exists()
    assert database_is_healthy(database)


def test_logging_configures_console_and_file(tmp_path: Path) -> None:
    log_path = tmp_path / "logs" / "application.log"
    logger = configure_logging(LoggingSettings(level="INFO", path=log_path))
    logger.info("foundation test message")

    for handler in logger.handlers:
        handler.flush()
    assert log_path.exists()
    assert "foundation test message" in log_path.read_text(encoding="utf-8")
    assert any(isinstance(handler, logging.StreamHandler) for handler in logger.handlers)
