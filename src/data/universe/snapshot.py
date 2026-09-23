"""Import a saved Kite NSE instrument CSV as a one-day, date-stamped mapping."""

from __future__ import annotations

import argparse
import csv
import re
from datetime import date, timedelta
from pathlib import Path

from src.configuration import PROJECT_ROOT


def import_snapshot(snapshot_path: str | Path, reference_directory: str | Path) -> int:
    snapshot = Path(snapshot_path)
    match = re.fullmatch(r"nse_instruments_(\d{4}-\d{2}-\d{2})\.csv", snapshot.name)
    if not match:
        raise ValueError("Snapshot filename must be nse_instruments_YYYY-MM-DD.csv.")
    observed_on = date.fromisoformat(match.group(1))
    reference = Path(reference_directory)
    with (reference / "symbol_history.csv").open(encoding="utf-8", newline="") as handle:
        symbols = list(csv.DictReader(handle))
    active_symbols: dict[str, str] = {}
    for row in symbols:
        start = date.fromisoformat(row["effective_from"])
        end = date.fromisoformat(row["effective_to"]) if row["effective_to"] else None
        if start <= observed_on and (end is None or observed_on < end):
            active_symbols[row["symbol"]] = row["security_id"]
    with snapshot.open(encoding="utf-8", newline="") as handle:
        instruments = [
            row for row in csv.DictReader(handle)
            if row.get("exchange") == "NSE" and row.get("instrument_type") == "EQ"
        ]
    destination = reference / "broker_instrument_history.csv"
    with destination.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    fields = ["security_id", "broker", "exchange", "tradingsymbol", "instrument_token", "effective_from", "effective_to", "source_id"]
    rows = [
        row for row in rows
        if not (row["source_id"] == f"kite_snapshot_{observed_on.isoformat()}")
    ]
    added = 0
    for row in instruments:
        symbol = row.get("tradingsymbol", "")
        security_id = active_symbols.get(symbol)
        token = row.get("instrument_token", "")
        if not security_id or not token.isdigit() or int(token) <= 0:
            continue
        rows.append({
            "security_id": security_id,
            "broker": "kite",
            "exchange": "NSE",
            "tradingsymbol": symbol,
            "instrument_token": token,
            "effective_from": observed_on.isoformat(),
            "effective_to": (observed_on + timedelta(days=1)).isoformat(),
            "source_id": f"kite_snapshot_{observed_on.isoformat()}",
        })
        added += 1
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    sources_path = reference / "sources.csv"
    with sources_path.open(encoding="utf-8", newline="") as handle:
        sources = list(csv.DictReader(handle))
    source_id = f"kite_snapshot_{observed_on.isoformat()}"
    if not any(row["source_id"] == source_id for row in sources):
        sources.append({
            "source_id": source_id,
            "url": f"local:{snapshot.name}",
            "description": f"Saved Kite NSE instrument snapshot; mappings apply only on {observed_on.isoformat()}",
            "announcement_date": observed_on.isoformat(),
        })
        with sources_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["source_id", "url", "description", "announcement_date"])
            writer.writeheader()
            writer.writerows(sources)
    return added


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", required=True, help="Existing local NSE instrument snapshot CSV")
    parser.add_argument("--reference-dir", default=str(PROJECT_ROOT / "data" / "reference"))
    args = parser.parse_args()
    try:
        count = import_snapshot(args.path, args.reference_dir)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"Imported {count} dated Kite NSE equity mappings from the saved snapshot.")


if __name__ == "__main__":
    main()
