"""Load manifested normalized daily market data for simulation."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

from src.data.market_data import DATA_ROOT, MANIFEST_PATH, MarketDataError, read_manifest, read_market_data


class BacktestDataError(ValueError):
    """The manifested daily dataset is absent, invalid, or ambiguous."""


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
