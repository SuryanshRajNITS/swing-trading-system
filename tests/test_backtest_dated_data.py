from __future__ import annotations

import csv
from datetime import date

from src.backtest.data import load_daily_dataset_through


def test_dated_prefix_loader_does_not_parse_or_hash_rows_after_end(tmp_path):
    data_root = tmp_path / "market_data"
    relative = "TEST/day/SEC1/TEST.csv"
    candle_path = data_root / relative
    candle_path.parent.mkdir(parents=True)
    header = "date,open,high,low,close,volume\n"
    prefix = (
        "2023-01-02T00:00:00+05:30,100,105,98,102,1000\n"
        "2024-12-31T00:00:00+05:30,102,106,101,105,1200\n"
    )
    candle_path.write_text(header + prefix + "2025-01-02T00:00:00+05:30,not-a-price\n", encoding="utf-8")
    manifest = tmp_path / "manifest.csv"
    fields = ("universe_id", "security_id", "source", "symbol_used", "interval",
              "requested_start", "requested_end", "actual_start", "actual_end", "row_count",
              "acquired_at", "provenance", "validation_status", "relative_path")
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({"universe_id": "TEST", "security_id": "SEC1", "source": "fixture",
                         "symbol_used": "TEST", "interval": "day", "actual_start": "2023-01-02",
                         "actual_end": "2026-09-22", "validation_status": "VALIDATED",
                         "relative_path": relative})

    first, first_digest = load_daily_dataset_through(
        universe_id="TEST", start_date=date(2023, 1, 1), end_date=date(2024, 12, 31),
        data_root=data_root, manifest_path=manifest,
    )
    assert len(first["SEC1"]) == 2
    assert first["SEC1"][-1]["close"] == 105

    candle_path.write_text(header + prefix + "2025-01-02T00:00:00+05:30,still-not-a-price\n", encoding="utf-8")
    second, second_digest = load_daily_dataset_through(
        universe_id="TEST", start_date=date(2023, 1, 1), end_date=date(2024, 12, 31),
        data_root=data_root, manifest_path=manifest,
    )
    assert second == first
    assert second_digest == first_digest
