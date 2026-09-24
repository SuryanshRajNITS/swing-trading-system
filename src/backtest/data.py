"""Load manifested normalized daily market data for simulation."""

from __future__ import annotations

import hashlib
import csv
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any

from src.data.market_data import DATA_ROOT, MANIFEST_PATH, MarketDataError, read_manifest, read_market_data, validate_ohlcv


class BacktestDataError(ValueError):
    """The manifested daily dataset is absent, invalid, or ambiguous."""


def load_daily_dataset_through(
    *, universe_id: str, start_date: date, end_date: date,
    data_root: str | Path = DATA_ROOT, manifest_path: str | Path = MANIFEST_PATH,
) -> tuple[dict[str, list[dict[str, Any]]], str]:
    """Load a dated prefix only; rows after ``end_date`` are not parsed or hashed.

    This is used for restricted research windows where later validation/OOS candles
    must remain outside the run's data information set. Normalized CSV files must
    be chronological, as enforced when they are written by the data pipeline.
    """
    if start_date > end_date:
        raise BacktestDataError("Dataset start_date must not be after end_date.")
    manifest = Path(manifest_path)
    if not manifest.exists():
        raise BacktestDataError(f"Daily data manifest not found: {manifest}")
    digest = hashlib.sha256()
    grouped: dict[str, dict[date, dict[str, Any]]] = defaultdict(dict)
    root = Path(data_root)
    eligible_rows = [row for row in read_manifest(manifest)
                     if row.get("universe_id") == universe_id and row.get("interval") == "day"
                     and row.get("validation_status") in {"VALIDATED", "VALIDATED_EVALUATION_PARTIAL_WARMUP"}
                     and row.get("actual_start", "9999-12-31") <= end_date.isoformat()]
    if not eligible_rows:
        raise BacktestDataError(f"No validated daily data is manifested for {universe_id}.")
    for item in sorted(eligible_rows, key=lambda row: (row.get("security_id", ""), row.get("symbol_used", ""), row.get("source", ""))):
        relative_path = item.get("relative_path", "")
        candle_path = root / relative_path
        try:
            with candle_path.open("r", encoding="utf-8-sig", newline="") as handle:
                header = handle.readline()
                if not header:
                    raise OSError("empty data file")
                prefix_rows: list[dict[str, str]] = []
                prefix_bytes = header.encode("utf-8")
                previous_day: date | None = None
                for line in handle:
                    raw_date = line.split(",", 1)[0].strip().strip('"')
                    try:
                        row_day = date.fromisoformat(raw_date[:10])
                    except ValueError:
                        raise BacktestDataError(f"Invalid date in manifested daily data file: {relative_path}") from None
                    if row_day > end_date:
                        break
                    if previous_day and row_day < previous_day:
                        raise BacktestDataError(f"Daily data is not chronological: {relative_path}")
                    previous_day = row_day
                    parsed = next(csv.DictReader([header, line]))
                    prefix_rows.append(parsed)
                    prefix_bytes += line.encode("utf-8")
        except OSError:
            raise BacktestDataError(f"Manifested daily data file is missing: {relative_path or 'unknown'}") from None
        try:
            candles = validate_ohlcv(prefix_rows)
        except MarketDataError as error:
            raise BacktestDataError(f"Could not validate dated daily data ({item.get('security_id', 'unknown')}): {error}") from None
        digest.update(item.get("security_id", "").encode("utf-8"))
        digest.update(relative_path.encode("utf-8"))
        digest.update(prefix_bytes)
        for candle in candles:
            day = date.fromisoformat(candle["date"][:10])
            existing = grouped[item["security_id"]].get(day)
            if existing is not None and any(existing[field] != candle[field] for field in ("open", "high", "low", "close", "volume")):
                raise BacktestDataError(
                    f"Conflicting source candles for {item['security_id']} on {day}; resolve provenance before simulation."
                )
            grouped[item["security_id"]][day] = candle
    data = {security_id: [row for day, row in sorted(by_day.items()) if day <= end_date]
            for security_id, by_day in grouped.items()}
    if not any(start_date <= date.fromisoformat(row["date"][:10]) <= end_date
               for candles in data.values() for row in candles):
        raise BacktestDataError("No manifested daily candles overlap the requested evaluation period.")
    return data, digest.hexdigest()


def load_daily_dataset(
    *, universe_id: str, start_date: date, end_date: date,
    data_root: str | Path = DATA_ROOT, manifest_path: str | Path = MANIFEST_PATH,
) -> tuple[dict[str, list[dict[str, Any]]], str]:
    manifest = Path(manifest_path)
    if not manifest.exists():
        raise BacktestDataError(f"Daily data manifest not found: {manifest}")
    digest = hashlib.sha256(manifest.read_bytes())
    rows = [row for row in read_manifest(manifest)
            if row.get("universe_id") == universe_id and row.get("interval") == "day"
            and row.get("validation_status") in {"VALIDATED", "VALIDATED_EVALUATION_PARTIAL_WARMUP"}]
    if not rows:
        raise BacktestDataError(f"No validated daily data is manifested for {universe_id}.")
    grouped: dict[str, dict[date, dict[str, Any]]] = defaultdict(dict)
    root = Path(data_root)
    for item in sorted(rows, key=lambda row: (row.get("security_id", ""), row.get("symbol_used", ""), row.get("source", ""))):
        candle_path = root / item.get("relative_path", "")
        try:
            digest.update(item.get("security_id", "").encode("utf-8"))
            digest.update(item.get("relative_path", "").encode("utf-8"))
            digest.update(candle_path.read_bytes())
        except OSError:
            raise BacktestDataError(f"Manifested daily data file is missing: {item.get('relative_path', 'unknown')}") from None
        try:
            candles = read_market_data(candle_path)
        except (OSError, KeyError, MarketDataError) as error:
            raise BacktestDataError(f"Could not load manifested daily data ({item.get('security_id', 'unknown')}): {error}") from None
        for candle in candles:
            day = date.fromisoformat(candle["date"][:10])
            existing = grouped[item["security_id"]].get(day)
            if existing is not None:
                comparable = ("open", "high", "low", "close", "volume")
                if any(existing[field] != candle[field] for field in comparable):
                    raise BacktestDataError(
                        f"Conflicting source candles for {item['security_id']} on {day}; resolve provenance before simulation."
                    )
                continue
            grouped[item["security_id"]][day] = candle
    data = {
        security_id: [row for day, row in sorted(by_day.items())
                      if day <= end_date]
        for security_id, by_day in grouped.items()
    }
    if not any(start_date <= date.fromisoformat(row["date"][:10]) <= end_date
               for candles in data.values() for row in candles):
        raise BacktestDataError("No manifested daily candles overlap the requested evaluation period.")
    return data, digest.hexdigest()


def load_hourly_dataset_through(
    *, universe_id: str, start_date: date, end_date: date,
    data_root: str | Path = DATA_ROOT,
    manifest_path: str | Path | None = None,
    daily_manifest_path: str | Path = MANIFEST_PATH,
    include_flagged_identities: bool = False,
    identity_exclusions_path: str | Path | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], str]:
    """Load manifested 60-minute candles through the requested date boundary.

    This data-only loader does not alter daily strategy inputs or execution rules.
    The end date is inclusive; all provider candle-start timestamps on that date
    are eligible.
    """
    from src.data.hourly import (
        FIELDS as HOURLY_FIELDS, HOURLY_MANIFEST, INTERVAL, HourlyDataError,
        _daily_sessions, normalize_hourly,
    )

    if start_date > end_date:
        raise BacktestDataError("Dataset start_date must not be after end_date.")
    manifest = Path(manifest_path or HOURLY_MANIFEST)
    if not manifest.exists():
        raise BacktestDataError(f"Hourly data manifest not found: {manifest}")
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))
    eligible = [row for row in manifest_rows
                if row.get("universe_id") == universe_id and row.get("interval") == INTERVAL
                and row.get("validation_status") == "VALIDATED"
                and row.get("actual_first_timestamp", "9999")[:10] <= end_date.isoformat()]
    exclusions_path = Path(identity_exclusions_path) if identity_exclusions_path else (
        Path(__file__).resolve().parents[2] / "data" / "processed" / "hourly" / "price_integrity" / "invalid_hourly_identities.csv")
    excluded_ids: set[str] = set()
    if not include_flagged_identities and exclusions_path.is_file():
        with exclusions_path.open("r", encoding="utf-8-sig", newline="") as handle:
            excluded_ids = {row.get("security_id", "") for row in csv.DictReader(handle)
                            if row.get("identity_status") == "HOURLY_IDENTITY_PRICE_MISMATCH"}
        eligible = [row for row in eligible if row.get("security_id") not in excluded_ids]
    if not eligible:
        raise BacktestDataError(f"No validated hourly data is manifested for {universe_id}.")
    root = Path(data_root)
    earliest = min(date.fromisoformat(row["actual_first_timestamp"][:10]) for row in eligible)
    session_dates = _daily_sessions(earliest, end_date, daily_manifest_path, root)
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    digest = hashlib.sha256()
    if excluded_ids and exclusions_path.is_file():
        digest.update(exclusions_path.read_bytes())
    for item in sorted(eligible, key=lambda row: (row.get("security_id", ""), row.get("mapping_effective_from", ""))):
        relative = item.get("relative_path", "")
        target = root / relative
        try:
            with target.open("r", encoding="utf-8-sig", newline="") as handle:
                header = handle.readline()
                if not header:
                    raise OSError("empty hourly data file")
                rows: list[dict[str, str]] = []
                prefix_bytes = header.encode("utf-8")
                previous: datetime | None = None
                for line in handle:
                    try:
                        row_stamp = datetime.fromisoformat(line.split(",", 1)[0].strip().strip('"'))
                    except ValueError:
                        raise BacktestDataError(f"Invalid timestamp in manifested hourly data file: {relative}") from None
                    if row_stamp.date() > end_date:
                        break
                    if previous is not None and row_stamp < previous:
                        raise BacktestDataError(f"Hourly data is not chronological: {relative}")
                    previous = row_stamp
                    rows.append(next(csv.DictReader([header, line])))
                    prefix_bytes += line.encode("utf-8")
                raw_bytes = prefix_bytes
        except OSError:
            raise BacktestDataError(f"Manifested hourly data file is missing: {relative or 'unknown'}") from None
        try:
            candles = normalize_hourly(rows, session_dates=session_dates)
        except (ValueError, HourlyDataError) as error:
            raise BacktestDataError(f"Could not validate hourly data ({item.get('security_id', 'unknown')}): {error}") from None
        digest.update(item.get("security_id", "").encode("utf-8"))
        digest.update(item.get("mapping_source_id", "").encode("utf-8"))
        digest.update(relative.encode("utf-8"))
        digest.update(raw_bytes)
        for candle in candles:
            stamp = datetime.fromisoformat(candle["date"])
            if stamp.date() > end_date:
                continue
            previous = grouped[item["security_id"]].get(candle["date"])
            if previous is not None and any(previous[field] != candle[field] for field in HOURLY_FIELDS[1:]):
                raise BacktestDataError(
                    f"Conflicting hourly source candles for {item['security_id']} at {candle['date']}."
                )
            grouped[item["security_id"]][candle["date"]] = candle
    data = {security_id: [candle for _, candle in sorted(by_time.items())]
            for security_id, by_time in grouped.items()}
    if not any(start_date <= datetime.fromisoformat(row["date"]).date() <= end_date
               for rows in data.values() for row in rows):
        raise BacktestDataError("No manifested hourly candles overlap the requested evaluation period.")
    return data, digest.hexdigest()
