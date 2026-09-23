"""Stable security-id storage and provenance for normalized daily market data."""

from __future__ import annotations

import csv
import re
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

from src.configuration import PROJECT_ROOT


MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "market_data_manifest.csv"
DATA_ROOT = PROJECT_ROOT / "data" / "processed" / "market_data"
MANIFEST_FIELDS = (
    "universe_id", "security_id", "source", "symbol_used", "interval",
    "requested_start", "requested_end", "actual_start", "actual_end",
    "row_count", "acquired_at", "provenance", "validation_status", "relative_path",
)
CANDLE_FIELDS = ("date", "open", "high", "low", "close", "volume")


class MarketDataError(ValueError):
    """Invalid market data or manifest state."""


def validate_ohlcv(
    rows: Iterable[dict[str, Any]], *, timezone_name: str = "Asia/Kolkata"
) -> list[dict[str, Any]]:
    """Validate, de-duplicate-check, sort and normalize OHLCV rows."""
    zone = ZoneInfo(timezone_name)
    result: list[dict[str, Any]] = []
    seen: set[datetime] = set()
    for index, row in enumerate(rows):
        stamp = row.get("date")
        if isinstance(stamp, date) and not isinstance(stamp, datetime):
            stamp = datetime.combine(stamp, datetime.min.time())
        elif isinstance(stamp, str):
            try:
                stamp = datetime.fromisoformat(stamp.strip())
            except ValueError:
                try:
                    stamp = datetime.combine(date.fromisoformat(stamp.strip()), datetime.min.time())
                except ValueError:
                    stamp = None
        if not isinstance(stamp, datetime):
            raise MarketDataError(f"Row {index} has an invalid timestamp.")
        stamp = stamp.replace(tzinfo=zone) if stamp.tzinfo is None else stamp.astimezone(zone)
        if stamp in seen:
            raise MarketDataError("Market data contains duplicate timestamps.")
        seen.add(stamp)

        prices: dict[str, float] = {}
        for field in ("open", "high", "low", "close"):
            try:
                value = float(str(row[field]).replace(",", "").strip())
            except (KeyError, TypeError, ValueError):
                raise MarketDataError(f"Row {index} has an invalid {field} price.") from None
            if value <= 0 or value != value or value in (float("inf"), float("-inf")):
                raise MarketDataError(f"Row {index} has an invalid {field} price.")
            prices[field] = value
        if prices["high"] < max(prices.values()) or prices["low"] > min(prices.values()):
            raise MarketDataError(f"Row {index} has inconsistent OHLC prices.")

        try:
            volume_number = float(str(row["volume"]).replace(",", "").strip())
        except (KeyError, TypeError, ValueError):
            raise MarketDataError(f"Row {index} has an invalid volume.") from None
        if volume_number < 0 or not volume_number.is_integer():
            raise MarketDataError(f"Row {index} has an invalid volume.")
        result.append({"date": stamp.isoformat(), **prices, "volume": int(volume_number)})

    return sorted(result, key=lambda item: item["date"])


def dataset_path(
    security_id: str, symbol_used: str, interval: str = "day", *,
    universe_id: str = "NIFTY50_PIT_TRADABLE", root: str | Path = DATA_ROOT,
) -> Path:
    if not re.fullmatch(r"[A-Z0-9_-]+", security_id):
        raise MarketDataError("Invalid security_id for dataset path.")
    if not re.fullmatch(r"[A-Z0-9&_-]+", symbol_used.upper()):
        raise MarketDataError("Invalid symbol for dataset path.")
    if interval != "day":
        raise MarketDataError("This daily-dataset milestone supports only the day interval.")
    return Path(root) / universe_id / interval / security_id / f"{symbol_used.upper()}.csv"


def read_market_data(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return validate_ohlcv(list(csv.DictReader(handle)))


def _atomic_csv(path: Path, fields: tuple[str, ...], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise MarketDataError(f"Could not write market data file {path.name}.") from None


def write_market_data(
    path: str | Path, rows: Iterable[dict[str, Any]], *, timezone_name: str = "Asia/Kolkata"
) -> list[dict[str, Any]]:
    normalized = validate_ohlcv(rows, timezone_name=timezone_name)
    if not normalized:
        raise MarketDataError("Cannot mark an empty candle set as market-data coverage.")
    _atomic_csv(Path(path), CANDLE_FIELDS, normalized)
    return normalized


def read_manifest(path: str | Path = MANIFEST_PATH) -> list[dict[str, str]]:
    manifest_path = Path(path)
    if not manifest_path.exists():
        return []
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def record_coverage(
    *, universe_id: str, security_id: str, source: str, symbol_used: str,
    interval: str, requested_start: date, requested_end: date,
    rows: list[dict[str, Any]], provenance: str, validation_status: str,
    relative_path: str, manifest_path: str | Path = MANIFEST_PATH,
    acquired_at: datetime | None = None,
) -> dict[str, str]:
    """Upsert coverage metadata; no broker token field is required or stored."""
    if not rows:
        raise MarketDataError("Empty data cannot be recorded as coverage.")
    now = acquired_at or datetime.now(timezone.utc)
    record = {
        "universe_id": universe_id,
        "security_id": security_id,
        "source": source,
        "symbol_used": symbol_used,
        "interval": interval,
        "requested_start": requested_start.isoformat(),
        "requested_end": requested_end.isoformat(),
        "actual_start": rows[0]["date"][:10],
        "actual_end": rows[-1]["date"][:10],
        "row_count": str(len(rows)),
        "acquired_at": now.astimezone(timezone.utc).isoformat(),
        "provenance": provenance,
        "validation_status": validation_status,
        "relative_path": relative_path,
    }
    path = Path(manifest_path)
    existing = read_manifest(path)
    key_fields = ("universe_id", "security_id", "source", "symbol_used", "interval")
    updated = False
    for index, old in enumerate(existing):
        if all(old.get(key) == record[key] for key in key_fields):
            existing[index] = record
            updated = True
            break
    if not updated:
        existing.append(record)
    _atomic_csv(path, MANIFEST_FIELDS, existing)
    return record
