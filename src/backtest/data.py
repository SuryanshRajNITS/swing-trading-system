"""Load manifested normalized daily market data for simulation."""

from __future__ import annotations

import hashlib
import csv
from collections import defaultdict
from datetime import date
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
