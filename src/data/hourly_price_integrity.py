"""Cross-check manifested 60-minute prices against same-day daily OHLC.

The audit is read-only with respect to market prices. It writes a local report and
an explicit exclusion list for hourly identities with a persistent price-scale
mismatch. It never repairs, adjusts, or substitutes candles.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.backtest.data import BacktestDataError, load_daily_dataset_through, load_hourly_dataset_through
from src.configuration import PROJECT_ROOT
from src.data.market_data import DATA_ROOT, MANIFEST_PATH, read_manifest
from src.data.hourly import HOURLY_MANIFEST
from src.data.universe.provider import PIT_TRADABLE


START = date(2023, 1, 1)
END = date(2024, 12, 31)
TOLERANCE = 0.05  # rupees; allows only source/display rounding, not price-scale differences
REPORT_DIR = PROJECT_ROOT / "data" / "processed" / "hourly" / "price_integrity"
REPORT_CSV = REPORT_DIR / "cross_timeframe_price_integrity_2023_2024.csv"
REPORT_JSON = REPORT_DIR / "cross_timeframe_price_integrity_2023_2024.json"
EXCLUSIONS_CSV = REPORT_DIR / "invalid_hourly_identities.csv"


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns or ["security_id"])
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=True, sort_keys=True)
                             if isinstance(v, (dict, list, tuple)) else v for k, v in row.items()})


def _manifest_context(hourly_manifest: Path, access_manifest: Path) -> dict[str, dict[str, str]]:
    access_by_id = {r.get("security_id", ""): r for r in _read_rows(access_manifest)}
    result = {}
    for row in _read_rows(hourly_manifest):
        if row.get("universe_id") != PIT_TRADABLE or row.get("interval") != "60minute":
            continue
        if row.get("validation_status") != "VALIDATED":
            continue
        sid = row.get("security_id", "")
        probe = access_by_id.get(sid, {})
        result[sid] = {
            "symbol": row.get("symbol_used", ""),
            "instrument_token": row.get("instrument_token", ""),
            "hourly_provenance": row.get("market_data_access_provenance", ""),
            "broker_mapping_provenance": row.get("broker_mapping_provenance", ""),
            "identity_status": probe.get("identity_status", ""),
            "current_symbol": probe.get("current_symbol", ""),
            "probe_result": probe.get("probe_result", ""),
            "validation_status": row.get("validation_status", ""),
        }
    return result


def _audit_security_days(
    daily_rows: Mapping[date, Mapping[str, Any]],
    hourly_rows: Mapping[date, Sequence[Mapping[str, Any]]],
    *, tolerance: float,
) -> dict[str, Any]:
    common_days = sorted(set(daily_rows) & set(hourly_rows))
    mismatches: list[dict[str, Any]] = []
    close_ratios: list[float] = []
    max_discrepancy = 0.0
    for day in common_days:
        bars = sorted(hourly_rows[day], key=lambda r: str(r["date"]))
        daily_row = daily_rows[day]
        aggregate = {"open": float(bars[0]["open"]),
                     "high": max(float(r["high"]) for r in bars),
                     "low": min(float(r["low"]) for r in bars),
                     "close": float(bars[-1]["close"])}
        daily_values = {field: float(daily_row[field]) for field in aggregate}
        deltas = {field: aggregate[field] - daily_values[field] for field in aggregate}
        discrepancy = max(abs(delta) for delta in deltas.values())
        max_discrepancy = max(max_discrepancy, discrepancy)
        if daily_values["close"]:
            close_ratios.append(aggregate["close"] / daily_values["close"])
        failed = {k: round(v, 6) for k, v in deltas.items() if abs(v) > tolerance}
        if failed:
            mismatches.append({"date": day.isoformat(), "daily": daily_values,
                               "hourly_aggregate": aggregate, "deltas_hourly_minus_daily": failed,
                               "hourly_bar_count": len(bars),
                               "session_types": sorted({str(r.get("session_type", "")) for r in bars})})
    median_ratio = statistics.median(close_ratios) if close_ratios else None
    return {"days_compared": len(common_days),
            "matched_days": len(common_days) - len(mismatches),
            "mismatched_days": len(mismatches),
            "median_ratio": median_ratio,
            "max_discrepancy": max_discrepancy,
            "first_mismatch": mismatches[0] if mismatches else "",
            "examples": mismatches[:5]}


def audit_cross_timeframe_prices(
    *, start: date = START, end: date = END,
    data_root: str | Path = DATA_ROOT,
    daily_manifest: str | Path = MANIFEST_PATH,
    hourly_manifest: str | Path = HOURLY_MANIFEST,
    access_manifest: str | Path = PROJECT_ROOT / "data" / "processed" / "hourly_access_probe_manifest.csv",
    report_dir: str | Path = REPORT_DIR,
    tolerance: float = TOLERANCE,
) -> dict[str, Any]:
    if start > end or tolerance < 0:
        raise ValueError("Invalid audit date range or tolerance.")
    # Use the existing manifested loaders so duplicate/conflicting sources and
    # invalid normalized records remain errors instead of being silently ignored.
    daily, daily_hash = load_daily_dataset_through(
        universe_id=PIT_TRADABLE, start_date=start, end_date=end,
        data_root=data_root, manifest_path=daily_manifest)
    hourly, hourly_hash = load_hourly_dataset_through(
        universe_id=PIT_TRADABLE, start_date=start, end_date=end,
        data_root=data_root, manifest_path=hourly_manifest,
        daily_manifest_path=daily_manifest, include_flagged_identities=True)
    context = _manifest_context(Path(hourly_manifest), Path(access_manifest))
    daily_by_id_day = {sid: {date.fromisoformat(str(r["date"])[:10]): r for r in rows
                             if start <= date.fromisoformat(str(r["date"])[:10]) <= end}
                       for sid, rows in daily.items()}
    hourly_by_id_day: dict[str, dict[date, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for sid, rows in hourly.items():
        for row in rows:
            day = date.fromisoformat(str(row["date"])[:10])
            if start <= day <= end:
                hourly_by_id_day[sid][day].append(row)

    # The audit population is the 53 validated 60-minute datasets. Daily-only
    # departed/unmapped securities are not members of that hourly population.
    security_ids = sorted(context)
    reports: list[dict[str, Any]] = []
    for sid in security_ids:
        context_row = context.get(sid, {})
        dmap, hmap = daily_by_id_day.get(sid, {}), hourly_by_id_day.get(sid, {})
        audited = _audit_security_days(dmap, hmap, tolerance=tolerance)
        common_days = audited["days_compared"]
        mismatches = audited["mismatched_days"]
        median_ratio = audited["median_ratio"]
        # A persistent scale/identity divergence is fail-closed. This threshold
        # identifies series that differ materially from daily OHLC; it is not a
        # strategy or price-correction rule.
        systematic_identity_mismatch = bool(
            median_ratio is not None and abs(median_ratio - 1.0) > 0.01
            and common_days >= 5
        )
        status = "HOURLY_IDENTITY_PRICE_MISMATCH" if systematic_identity_mismatch else (
            "OHLC_MISMATCH" if mismatches else "PRICE_CONSISTENT")
        reports.append({
            "security_id": sid, **context_row,
            "matched_days": audited["matched_days"],
            "mismatched_days": mismatches,
            "days_compared": common_days,
            "median_close_price_ratio_hourly_over_daily": round(median_ratio, 8) if median_ratio is not None else "",
            "max_discrepancy_rupees": round(audited["max_discrepancy"], 6),
            "first_mismatch": audited["first_mismatch"],
            "examples": audited["examples"],
            "identity_quality_status": status,
            "tolerance_rupees": tolerance,
        })

    report_path = Path(report_dir)
    csv_path = report_path / "cross_timeframe_price_integrity_2023_2024.csv"
    json_path = report_path / "cross_timeframe_price_integrity_2023_2024.json"
    exclusions_path = report_path / "invalid_hourly_identities.csv"
    _write_rows(csv_path, reports)
    payload = {
        "audit": "daily_vs_kite_60minute_ohlc",
        "period": {"start": start.isoformat(), "end": end.isoformat()},
        "universe_id": PIT_TRADABLE,
        "tolerance_rupees": tolerance,
        "aggregation": {"open": "first hourly open", "high": "maximum hourly high",
                         "low": "minimum hourly low", "close": "last hourly close"},
        "identity_mismatch_rule": "median daily-close ratio differs from 1 by more than 1%, with at least 5 common sessions",
        "data_hashes": {"daily_prefix_sha256": daily_hash, "hourly_prefix_sha256": hourly_hash},
        "security_count": len(reports),
        "mismatched_security_count": sum(r["mismatched_days"] > 0 for r in reports),
        "identity_mismatch_security_count": sum(r["identity_quality_status"] == "HOURLY_IDENTITY_PRICE_MISMATCH" for r in reports),
        "securities": reports,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    exclusions = [{"security_id": r["security_id"], "symbol": r.get("symbol", ""),
                   "identity_status": r["identity_quality_status"],
                   "median_close_price_ratio_hourly_over_daily": r["median_close_price_ratio_hourly_over_daily"],
                   "mismatched_days": r["mismatched_days"], "report": json_path.name}
                  for r in reports if r["identity_quality_status"] == "HOURLY_IDENTITY_PRICE_MISMATCH"]
    _write_rows(exclusions_path, exclusions)
    print(f"Cross-timeframe audit: {len(reports)} securities; {payload['mismatched_security_count']} with one or more OHLC mismatches; {len(exclusions)} systematic identity mismatches.")
    print(f"CSV: {csv_path}\nJSON: {json_path}\nExclusions: {exclusions_path}")
    for row in reports:
        if row["mismatched_days"] or row["identity_quality_status"] != "PRICE_CONSISTENT":
            print(f"{row['security_id']} {row.get('symbol','')}: matched={row['matched_days']} mismatch={row['mismatched_days']} median_ratio={row['median_close_price_ratio_hourly_over_daily']} max_abs={row['max_discrepancy_rupees']} first={row['first_mismatch']}")
    return {"payload": payload, "csv_path": csv_path, "json_path": json_path,
            "exclusions_path": exclusions_path, "reports": reports}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, default=START)
    parser.add_argument("--end", type=date.fromisoformat, default=END)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--daily-manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--hourly-manifest", type=Path, default=HOURLY_MANIFEST)
    parser.add_argument("--access-manifest", type=Path, default=PROJECT_ROOT / "data" / "processed" / "hourly_access_probe_manifest.csv")
    parser.add_argument("--output-directory", type=Path, default=REPORT_DIR)
    parser.add_argument("--tolerance", type=float, default=TOLERANCE)
    args = parser.parse_args()
    try:
        audit_cross_timeframe_prices(start=args.start, end=args.end, data_root=args.data_root,
            daily_manifest=args.daily_manifest, hourly_manifest=args.hourly_manifest,
            access_manifest=args.access_manifest, report_dir=args.output_directory,
            tolerance=args.tolerance)
    except (ValueError, OSError, BacktestDataError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
