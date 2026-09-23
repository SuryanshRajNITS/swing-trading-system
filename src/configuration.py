"""Load and validate the application's YAML settings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.yaml"


@dataclass(frozen=True)
class DatabaseSettings:
    path: Path


@dataclass(frozen=True)
class LoggingSettings:
    level: str
    path: Path


@dataclass(frozen=True)
class HistoricalDataSettings:
    start_date: date
    end_date: date


@dataclass(frozen=True)
class AppSettings:
    environment: str
    trading_mode: str
    live_trading_enabled: bool
    initial_capital: float
    max_risk_per_trade: float
    database: DatabaseSettings
    logging: LoggingSettings
    timezone: str
    historical_data: HistoricalDataSettings


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Setting '{name}' must be a mapping.")
    return value


def _required_string(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Setting '{key}' must be a non-empty string.")
    return value.strip()


def _parse_config_date(value: Any, name: str, timezone: ZoneInfo) -> date:
    if isinstance(value, datetime):
        raise ValueError(f"Setting '{name}' must be a date, not a timestamp.")
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        raise ValueError(f"Setting '{name}' must be an ISO date or 'yesterday'.")
    if value.strip().lower() == "yesterday":
        return datetime.now(timezone).date() - timedelta(days=1)
    try:
        return date.fromisoformat(value.strip())
    except ValueError:
        raise ValueError(f"Setting '{name}' must be an ISO date (YYYY-MM-DD) or 'yesterday'.") from None


def load_settings(path: str | Path = DEFAULT_SETTINGS_PATH) -> AppSettings:
    """Load settings from YAML, resolving local paths from the project root."""
    settings_path = Path(path).expanduser().resolve()
    try:
        with settings_path.open("r", encoding="utf-8") as settings_file:
            raw = yaml.safe_load(settings_file)
    except OSError as exc:
        raise ValueError(f"Could not read settings file '{settings_path}': {exc}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in settings file '{settings_path}': {exc}") from exc

    data = _require_mapping(raw, "root")
    environment = _required_string(data, "environment")
    trading_mode = _required_string(data, "trading_mode")
    if trading_mode not in {"backtest", "paper", "alert_only", "live"}:
        raise ValueError("Setting 'trading_mode' must be backtest, paper, alert_only, or live.")

    live_enabled = data.get("live_trading_enabled")
    if not isinstance(live_enabled, bool):
        raise ValueError("Setting 'live_trading_enabled' must be a boolean.")
    if trading_mode == "live" and not live_enabled:
        raise ValueError("Live trading mode requires 'live_trading_enabled: true'.")

    capital = data.get("initial_capital")
    risk = data.get("max_risk_per_trade")
    if isinstance(capital, bool) or not isinstance(capital, (int, float)) or capital <= 0:
        raise ValueError("Setting 'initial_capital' must be a positive number.")
    if isinstance(risk, bool) or not isinstance(risk, (int, float)) or not 0 < risk <= 1:
        raise ValueError("Setting 'max_risk_per_trade' must be greater than 0 and at most 1.")

    database = _require_mapping(data.get("database"), "database")
    logging_data = _require_mapping(data.get("logging"), "logging")
    log_level = _required_string(logging_data, "level").upper()
    if log_level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        raise ValueError("Setting 'logging.level' must be a valid standard log level.")

    def local_path(value: str) -> Path:
        target = Path(value).expanduser()
        return target.resolve() if target.is_absolute() else (PROJECT_ROOT / target).resolve()

    timezone_name = _required_string(data, "timezone")
    try:
        market_timezone = ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError:
        raise ValueError(f"Setting 'timezone' is not a recognized timezone: {timezone_name}.") from None

    historical = _require_mapping(data.get("data"), "data")
    start_date = _parse_config_date(historical.get("history_start_date"), "data.history_start_date", market_timezone)
    end_date = _parse_config_date(historical.get("history_end_date"), "data.history_end_date", market_timezone)
    if start_date > end_date:
        raise ValueError("Setting 'data.history_start_date' must not be later than 'data.history_end_date'.")

    return AppSettings(
        environment=environment,
        trading_mode=trading_mode,
        live_trading_enabled=live_enabled,
        initial_capital=float(capital),
        max_risk_per_trade=float(risk),
        database=DatabaseSettings(local_path(_required_string(database, "path"))),
        logging=LoggingSettings(
            level=log_level,
            path=local_path(_required_string(logging_data, "path")),
        ),
        timezone=timezone_name,
        historical_data=HistoricalDataSettings(start_date=start_date, end_date=end_date),
    )
