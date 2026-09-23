"""Read-only Kite historical candle download and local CSV persistence."""

from __future__ import annotations

import csv
import argparse
import math
from datetime import date, datetime, timedelta, time
from pathlib import Path
from typing import Any, Callable, Iterable
from zoneinfo import ZoneInfo

from kiteconnect import KiteConnect
from kiteconnect.exceptions import KiteException

from src.broker.auth import AuthenticationError
from src.broker.session import SessionVerificationError, create_authenticated_client
from src.configuration import PROJECT_ROOT, load_settings


INTERVAL_MAX_DAYS = {
    "day": 2000,
    "60minute": 400,
    "15minute": 100,
}
CANDLE_FIELDS = ("date", "open", "high", "low", "close", "volume", "oi")
DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "data" / "raw" / "historical"


class HistoricalDataError(RuntimeError):
    """Raised when historical candles cannot be retrieved or persisted safely."""


def validate_candles(rows: Iterable[dict[str, Any]], *, timezone: str = "Asia/Kolkata") -> list[dict[str, Any]]:
    """Validate Kite SDK candles and normalize them into deterministic records."""
    zone = ZoneInfo(timezone)
    if isinstance(rows, (str, bytes, dict)):
        raise HistoricalDataError("Kite returned an invalid candle response.")
    try:
        candles = list(rows)
    except TypeError:
        raise HistoricalDataError("Kite returned an invalid candle response.") from None

    normalized: list[dict[str, Any]] = []
    seen: set[datetime] = set()
    for index, candle in enumerate(candles):
        if not isinstance(candle, dict):
            raise HistoricalDataError(f"Candle {index} is not a mapping.")
        timestamp = candle.get("date")
        if isinstance(timestamp, str):
            try:
                timestamp = datetime.fromisoformat(timestamp)
            except ValueError:
                timestamp = None
        if not isinstance(timestamp, datetime):
            raise HistoricalDataError(f"Candle {index} has an invalid timestamp.")
        timestamp = timestamp.replace(tzinfo=zone) if timestamp.tzinfo is None else timestamp.astimezone(zone)
        if timestamp in seen:
            raise HistoricalDataError("Kite returned duplicate candle timestamps.")
        seen.add(timestamp)

        prices: dict[str, float] = {}
        for field in ("open", "high", "low", "close"):
            value = candle.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise HistoricalDataError(f"Candle {index} has an invalid {field} value.")
            prices[field] = float(value)
        if prices["high"] < max(prices["open"], prices["close"], prices["low"]):
            raise HistoricalDataError(f"Candle {index} has inconsistent OHLC values.")
        if prices["low"] > min(prices["open"], prices["close"], prices["high"]):
            raise HistoricalDataError(f"Candle {index} has inconsistent OHLC values.")

        volume = candle.get("volume")
        if isinstance(volume, bool) or not isinstance(volume, (int, float)) or not math.isfinite(volume) or volume < 0:
            raise HistoricalDataError(f"Candle {index} has an invalid volume value.")
        oi = candle.get("oi", 0)
        if isinstance(oi, bool) or not isinstance(oi, (int, float)) or not math.isfinite(oi) or oi < 0:
            raise HistoricalDataError(f"Candle {index} has an invalid open-interest value.")

        normalized.append({"date": timestamp.isoformat(), **prices, "volume": int(volume), "oi": int(oi)})

    normalized.sort(key=lambda candle: candle["date"])
    return normalized


def _date_chunks(start_date: date, end_date: date, interval: str) -> Iterable[tuple[datetime, datetime]]:
    max_days = INTERVAL_MAX_DAYS.get(interval)
    if max_days is None:
        raise HistoricalDataError(f"Unsupported interval '{interval}'. Use day, 60minute, or 15minute.")
    cursor = start_date
    while cursor <= end_date:
        chunk_end = min(end_date, cursor + timedelta(days=max_days - 1))
        yield datetime.combine(cursor, time.min), datetime.combine(chunk_end, time.max)
        cursor = chunk_end + timedelta(days=1)


def save_candles(
    candles: Iterable[dict[str, Any]], *, symbol: str, interval: str,
    directory: str | Path = DEFAULT_OUTPUT_DIRECTORY,
) -> Path:
    """Atomically save a validated symbol/interval CSV beneath the raw data directory."""
    if not symbol or any(character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789&-_" for character in symbol.upper()):
        raise HistoricalDataError("Instrument symbol is not safe for use as a local filename.")
    if interval not in INTERVAL_MAX_DAYS:
        raise HistoricalDataError(f"Unsupported interval '{interval}'. Use day, 60minute, or 15minute.")
    validated = validate_candles(candles)
    output = Path(directory) / interval
    destination = output / f"{symbol.upper()}.csv"
    temporary = destination.with_suffix(".csv.tmp")
    try:
        output.mkdir(parents=True, exist_ok=True)
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CANDLE_FIELDS)
            writer.writeheader()
            writer.writerows(validated)
        temporary.replace(destination)
    except OSError:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise HistoricalDataError("Could not write the local historical candle file.") from None
    return destination


def download_instrument_history(
    instrument: dict[str, Any],
    *,
    start_date: date,
    end_date: date,
    intervals: tuple[str, ...] = ("day", "60minute", "15minute"),
    client: KiteConnect | None = None,
    client_factory: Callable[[], KiteConnect] = create_authenticated_client,
    output_directory: str | Path = DEFAULT_OUTPUT_DIRECTORY,
    timezone: str = "Asia/Kolkata",
) -> dict[str, Path]:
    """Download requested candle intervals for one explicit Kite instrument record."""
    symbol, token = instrument.get("tradingsymbol"), instrument.get("instrument_token")
    if not isinstance(symbol, str) or not symbol.strip():
        raise HistoricalDataError("Instrument record has no trading symbol.")
    if isinstance(token, bool) or not isinstance(token, int) or token <= 0:
        raise HistoricalDataError("Instrument record has an invalid instrument token.")
    if start_date > end_date:
        raise HistoricalDataError("Historical start date must not be later than end date.")
    if not intervals:
        raise HistoricalDataError("At least one historical interval is required.")
    for interval in intervals:
        if interval not in INTERVAL_MAX_DAYS:
            raise HistoricalDataError(f"Unsupported interval '{interval}'. Use day, 60minute, or 15minute.")

    try:
        client = client or client_factory()
        saved: dict[str, Path] = {}
        for interval in intervals:
            candles: list[dict[str, Any]] = []
            for chunk_start, chunk_end in _date_chunks(start_date, end_date, interval):
                candles.extend(client.historical_data(token, chunk_start, chunk_end, interval))
            normalized = validate_candles(candles, timezone=timezone)
            saved[interval] = save_candles(
                normalized, symbol=symbol, interval=interval, directory=output_directory
            )
        return saved
    except (AuthenticationError, SessionVerificationError) as error:
        raise HistoricalDataError(str(error)) from None
    except KiteException:
        raise HistoricalDataError(f"Kite rejected the historical-data request for {symbol}.") from None
    except HistoricalDataError:
        raise
    except Exception:
        raise HistoricalDataError(f"Could not download historical candles for {symbol}.") from None


def download_configured_history(
    instruments: Iterable[dict[str, Any]],
    *,
    intervals: tuple[str, ...] = ("day", "60minute", "15minute"),
    client: KiteConnect | None = None,
    output_directory: str | Path = DEFAULT_OUTPUT_DIRECTORY,
) -> dict[str, dict[str, Path]]:
    """Download configured historical dates for explicitly supplied instruments."""
    settings = load_settings()
    client = client or create_authenticated_client()
    results: dict[str, dict[str, Path]] = {}
    for instrument in instruments:
        symbol = instrument.get("tradingsymbol", "")
        results[symbol] = download_instrument_history(
            instrument,
            start_date=settings.historical_data.start_date,
            end_date=settings.historical_data.end_date,
            intervals=intervals,
            client=client,
            output_directory=output_directory,
            timezone=settings.timezone,
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Kite historical candles for explicit NSE symbols.")
    parser.add_argument("--symbols", required=True, help="Comma-separated symbols from the latest NSE snapshot.")
    parser.add_argument(
        "--intervals", default="day,60minute,15minute",
        help="Comma-separated intervals: day, 60minute, 15minute (default: all three).",
    )
    arguments = parser.parse_args()
    requested_symbols = {value.strip().upper() for value in arguments.symbols.split(",") if value.strip()}
    requested_intervals = tuple(value.strip() for value in arguments.intervals.split(",") if value.strip())
    if not requested_symbols:
        parser.error("provide at least one symbol")

    snapshots = sorted((PROJECT_ROOT / "data" / "raw" / "instruments").glob("nse_instruments_*.csv"))
    if not snapshots:
        parser.error("no NSE instrument snapshot found; run python -m src.broker.instruments first")
    with snapshots[-1].open("r", encoding="utf-8", newline="") as handle:
        available = {
            row["tradingsymbol"].upper(): {
                **row,
                "instrument_token": int(row["instrument_token"]),
            }
            for row in csv.DictReader(handle)
            if row.get("tradingsymbol", "").upper() in requested_symbols
            and row.get("instrument_type") == "EQ"
        }
    missing = requested_symbols - available.keys()
    if missing:
        parser.error(f"symbols missing from latest NSE equity snapshot: {', '.join(sorted(missing))}")

    try:
        results = download_configured_history(available.values(), intervals=requested_intervals)
    except (HistoricalDataError, ValueError) as error:
        print(f"Historical-data download failed: {error}")
        raise SystemExit(1) from None
    for symbol, files in results.items():
        for interval, path in files.items():
            print(f"Saved {symbol} {interval} candles to {path}.")


if __name__ == "__main__":
    main()
