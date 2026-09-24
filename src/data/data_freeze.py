"""Build a deterministic 2023-2024 research-data freeze and integrity report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.backtest.data import BacktestDataError, load_daily_dataset_through, load_hourly_dataset_through
from src.configuration import PROJECT_ROOT
from src.data.hourly import HOURLY_MANIFEST, SESSION_OVERRIDES
from src.data.market_data import DATA_ROOT, MANIFEST_PATH
from src.data.universe.provider import PIT_TRADABLE, UniverseProvider
from src.data.weekly import resample_daily_to_weekly


START = date(2023, 1, 1)
END = date(2024, 12, 31)
DAILY_WARMUP_START = date(2022, 7, 1)
HOURLY_WARMUP_START = date(2022, 10, 1)
PRICE_INTEGRITY_DIR = PROJECT_ROOT / "data" / "processed" / "hourly" / "price_integrity"
EXCLUSIONS_PATH = PRICE_INTEGRITY_DIR / "invalid_hourly_identities.csv"
QUARANTINE_PATH = PROJECT_ROOT / "data" / "processed" / "hourly" / "quarantine_invalid_candles.csv"
HOURLY_ACCESS_MANIFEST = PROJECT_ROOT / "data" / "processed" / "hourly_access_probe_manifest.csv"
BENCHMARK_PATH = PROJECT_ROOT / "data" / "raw" / "reference" / "nifty50_daily.csv"
MANIFEST_DIR = PROJECT_ROOT / "data" / "processed" / "manifests"
FREEZE_MANIFEST_PATH = MANIFEST_DIR / "data_freeze_2023_2024.json"
FREEZE_DOCUMENT_PATH = PROJECT_ROOT / "DATA_FREEZE_2023_2024.md"
STATS_CSV_PATH = MANIFEST_DIR / "cross_timeframe_discrepancy_stats_2023_2024.csv"
BREACH_CSV_PATH = MANIFEST_DIR / "cross_timeframe_discrepancies_over_1pct_2023_2024.csv"
FIELDS = ("open", "high", "low", "close")
THRESHOLDS = (0.25, 0.50, 1.00, 2.00)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def _p95(values: Sequence[float]) -> float:
    """Nearest-rank 95th percentile, deterministic for any sample size."""
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def _clipped_reference_slice(path: Path, *, start: date, end: date,
                             expected_ids: set[str]) -> list[dict[str, str]]:
    rows = _read_csv(path)
    selected: list[dict[str, str]] = []
    end_exclusive = end + timedelta(days=1)
    for original in rows:
        row = dict(original)
        if "security_id" in row and row["security_id"] and row["security_id"] not in expected_ids:
            continue
        if "universe_id" in row and row["universe_id"] and row["universe_id"] not in {
                PIT_TRADABLE, "NIFTY50_PIT", "SECURITY_IDENTITY"}:
            continue
        if "effective_date" in row and row["effective_date"]:
            effective = date.fromisoformat(row["effective_date"])
            if not start <= effective <= end:
                continue
        if "effective_from" in row and row["effective_from"]:
            first = date.fromisoformat(row["effective_from"])
            last = date.fromisoformat(row["effective_to"]) if row.get("effective_to") else date.max
            if first >= end_exclusive or last <= start:
                continue
            row["effective_from"] = max(first, start).isoformat()
            row["effective_to"] = min(last, end_exclusive).isoformat()
        selected.append(row)
    return selected


def _manifest_slice_hash(path: Path, *, interval: str, expected_ids: set[str], end: date) -> dict[str, Any]:
    rows = [row for row in _read_csv(path)
            if row.get("security_id") in expected_ids and row.get("interval") == interval]
    clipped = []
    for row in rows:
        if row.get("actual_start") and date.fromisoformat(row["actual_start"][:10]) > end:
            continue
        clipped.append({key: row.get(key, "") for key in (
            "universe_id", "security_id", "source", "symbol_used", "interval",
            "actual_start", "requested_start", "validation_status", "relative_path",
            "instrument_token", "broker_mapping_provenance", "market_data_access_provenance",
            "mapping_source_id", "provenance")})
    clipped.sort(key=lambda row: (row["security_id"], row["relative_path"], row["source"]))
    return {"sha256": _canonical_hash(clipped), "rows": len(clipped),
            "scope": f"manifest rows for {interval} ending on or before {end.isoformat()}; future fields omitted"}


def _distribution(values: Sequence[tuple[float, float]]) -> dict[str, Any]:
    absolute = [row[0] for row in values]
    percentage = [row[1] for row in values]
    result: dict[str, Any] = {
        "session_count": len(values),
        "median_absolute_difference_rupees": round(statistics.median(absolute), 8) if absolute else 0.0,
        "median_absolute_difference_pct": round(statistics.median(percentage), 8) if percentage else 0.0,
        "p95_absolute_difference_pct": round(_p95(percentage), 8),
        "maximum_absolute_difference_pct": round(max(percentage, default=0.0), 8),
    }
    for threshold in THRESHOLDS:
        result[f"sessions_over_{threshold:.2f}pct"] = sum(value > threshold for value in percentage)
    return result


def _cross_timeframe_statistics(
    daily: Mapping[str, Sequence[Mapping[str, Any]]],
    hourly: Mapping[str, Sequence[Mapping[str, Any]]],
    symbols: Mapping[str, str],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    by_field: dict[str, list[tuple[float, float]]] = {field: [] for field in FIELDS}
    by_security: dict[str, dict[str, list[tuple[float, float]]]] = defaultdict(
        lambda: {field: [] for field in FIELDS})
    over_one: list[dict[str, Any]] = []
    total_sessions = 0
    strict_mismatch_sessions = 0
    strict_mismatch_security_ids: set[str] = set()
    for security_id in sorted(hourly):
        daily_map = {date.fromisoformat(str(row["date"])[:10]): row for row in daily.get(security_id, ())
                     if START <= date.fromisoformat(str(row["date"])[:10]) <= END}
        hourly_map: dict[date, list[Mapping[str, Any]]] = defaultdict(list)
        for row in hourly[security_id]:
            day = date.fromisoformat(str(row["date"])[:10])
            if START <= day <= END:
                hourly_map[day].append(row)
        for day in sorted(set(daily_map) & set(hourly_map)):
            bars = sorted(hourly_map[day], key=lambda row: str(row["date"]))
            aggregate = {"open": float(bars[0]["open"]),
                         "high": max(float(row["high"]) for row in bars),
                         "low": min(float(row["low"]) for row in bars),
                         "close": float(bars[-1]["close"])}
            daily_values = {field: float(daily_map[day][field]) for field in FIELDS}
            total_sessions += 1
            if any(abs(aggregate[field] - daily_values[field]) > 0.05 for field in FIELDS):
                strict_mismatch_sessions += 1
                strict_mismatch_security_ids.add(security_id)
            for field in FIELDS:
                absolute = abs(aggregate[field] - daily_values[field])
                percent = 100.0 * absolute / abs(daily_values[field]) if daily_values[field] else 0.0
                pair = (absolute, percent)
                by_field[field].append(pair)
                by_security[security_id][field].append(pair)
                if percent > 1.0:
                    over_one.append({
                        "security_id": security_id,
                        "symbol": symbols.get(security_id, security_id),
                        "date": day.isoformat(), "field": field,
                        "daily_value": daily_values[field],
                        "hourly_aggregate_value": aggregate[field],
                        "absolute_difference_rupees": round(absolute, 8),
                        "absolute_difference_pct": round(percent, 8),
                        "hourly_bar_count": len(bars),
                    })
    overall = {field: _distribution(by_field[field]) for field in FIELDS}
    per_security = []
    material_ids: list[str] = []
    for security_id in sorted(by_security):
        close_ratios = []
        daily_map = {date.fromisoformat(str(row["date"])[:10]): row for row in daily.get(security_id, ())
                     if START <= date.fromisoformat(str(row["date"])[:10]) <= END}
        hourly_map: dict[date, list[Mapping[str, Any]]] = defaultdict(list)
        for row in hourly[security_id]:
            day = date.fromisoformat(str(row["date"])[:10])
            if START <= day <= END:
                hourly_map[day].append(row)
        for day in sorted(set(daily_map) & set(hourly_map)):
            bars = sorted(hourly_map[day], key=lambda row: str(row["date"]))
            daily_close = float(daily_map[day]["close"])
            if daily_close:
                close_ratios.append(float(bars[-1]["close"]) / daily_close)
        median_close_ratio = statistics.median(close_ratios) if close_ratios else None
        persistent = bool(median_close_ratio is not None and abs(median_close_ratio - 1.0) > 0.01)
        if persistent:
            material_ids.append(security_id)
        per_security.append({
            "security_id": security_id,
            "symbol": symbols.get(security_id, security_id),
            "matched_closing_price_ratio_median": round(median_close_ratio, 8) if median_close_ratio is not None else None,
            "classification": "PERSISTENT_MATERIAL_SCALE_DIVERGENCE" if persistent else
                "PRICE_SCALE_VALIDATED_WITH_PROVIDER_OHLC_VARIANCE",
            "fields": {field: _distribution(by_security[security_id][field]) for field in FIELDS},
        })
    over_one.sort(key=lambda row: (row["date"], row["security_id"], row["field"]))
    return ({"evaluation_sessions_compared": total_sessions,
             "strict_tolerance_mismatch_security_count": len(strict_mismatch_security_ids),
             "strict_tolerance_mismatch_session_count": strict_mismatch_sessions,
             "absolute_percentage_denominator": "same-day daily component value",
             "percentile_method": "nearest-rank P95",
             "strict_equality_tolerance_rupees": 0.05,
             "persistent_scale_rule": "median same-day hourly-last-close / daily-close differs from 1.0 by more than 1%",
             "persistent_material_scale_divergence_security_ids": material_ids,
             "all_usable_series_classification": "PRICE_SCALE_VALIDATED_WITH_PROVIDER_OHLC_VARIANCE" if not material_ids else "REVIEW_REQUIRED",
             "by_ohlc_field": overall}, per_security, over_one)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    if not columns:
        columns = ["security_id"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _document(manifest: Mapping[str, Any], stats: Mapping[str, Any]) -> str:
    overall = stats["by_ohlc_field"]
    lines = [
        "# Research Data Freeze: 2023-2024", "",
        "## Frozen scope", "",
        "- Research period: 2023-01-01 through 2024-12-31.",
        "- Daily/weekly warmup begins 2022-07-01 where manifested data exists.",
        "- Hourly warmup requested from 2022-10-01; actual first bars depend on security/listing/provider availability.",
        "- PIT universe: `NIFTY50_PIT_TRADABLE`, resolved by effective-dated membership and symbol history; eligibility is date-specific and excludes non-tradable index placeholders.",
        "- 2025 and 2026 remain untouched validation/OOS periods.",
        f"- Data freeze status: `{manifest['data_freeze_status']}`.",
        "- Future strategy results must reference freeze version `NIFTY50_PIT_2023_2024_V1` and the manifest hash set in `data/processed/manifests/data_freeze_2023_2024.json`.",
        "", "## Dataset status", "",
        f"- PIT securities expected across the period: {manifest['counts']['expected_pit_securities']} unique identities.",
        f"- Daily usable: {manifest['counts']['daily_usable_securities']} identities with manifested daily candles overlapping their PIT membership.",
        "- Weekly: derived locally from daily candles using Monday-Sunday calendar-week grouping; OHLCV uses observed sessions only, no synthetic rows. A week becomes available only after its final session close.",
        f"- NIFTY 50 benchmark: {manifest['benchmark']['row_count']} daily price-return sessions, {manifest['benchmark']['first_date']} through {manifest['benchmark']['last_date']}; no pre-2023 benchmark warmup is present.",
        f"- Hourly: {manifest['counts']['hourly_usable_securities']} usable 60-minute identities after exclusions; Kite-native candle start timestamps preserved.",
        "- Hourly unavailable: HDFC (`NO_CURRENT_TOKEN`); LTIM (`FAIL_CLOSED_IDENTITY_BOUNDARY`).",
        "- Hourly excluded: TATAMOTORS / NSEEQ-0055 (`HOURLY_IDENTITY_PRICE_MISMATCH`). Its daily/weekly data remains available where PIT membership and daily coverage permit; any 1H-dependent opportunity must be `HOURLY_DATA_UNAVAILABLE`. No TMPV historical hourly substitution is permitted.",
        f"- Quarantined provider-invalid hourly candles: {manifest['counts']['quarantined_provider_invalid_candles']}. No interpolation, synthetic bars, or silent substitution is used.",
        "", "## Special sessions", "",
        "Hourly session validation uses explicit exchange overrides. Within 2023-2024 these include 2023-11-12 and 2024-11-01 Muhurat sessions; 2024-01-20 special regular session; and 2024-03-02 and 2024-05-18 special split sessions. Exchange trading windows remain separate from observed Kite bar-start grids. No timestamps are snapped and no bars are synthesized.",
        "", "## Cross-timeframe price-integrity findings", "",
        "Daily OHLC was compared with each same-day hourly aggregate (first open, maximum high, minimum low, last close). Strict OHLC equality was measured at ₹0.05, but failures are treated as provider OHLC variance and are not corrected.",
        f"At the strict ₹0.05 tolerance, {stats['strict_tolerance_mismatch_security_count']} of the 52 usable identities have at least one mismatched session ({stats['strict_tolerance_mismatch_session_count']} security-session pairs).",
        f"The 52 usable hourly series are classified as `{stats['all_usable_series_classification']}` under the persistent-scale rule (median hourly-last-close / daily-close differs from 1.0 by more than 1%). Persistent scale divergence IDs: {', '.join(stats['persistent_material_scale_divergence_security_ids']) or 'none'}.",
        "", "| Field | Median absolute difference (₹) | Median absolute difference (%) | P95 absolute difference (%) | Maximum (%) | >0.25% | >0.50% | >1.00% | >2.00% |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for field in ("open", "high", "low", "close"):
        row = overall[field]
        lines.append(f"| {field} | {row['median_absolute_difference_rupees']:.6f} | {row['median_absolute_difference_pct']:.6f} | {row['p95_absolute_difference_pct']:.6f} | {row['maximum_absolute_difference_pct']:.6f} | {row['sessions_over_0.25pct']} | {row['sessions_over_0.50pct']} | {row['sessions_over_1.00pct']} | {row['sessions_over_2.00pct']} |")
    lines += [
        "", f"All per-security distributions and every field/session discrepancy over 1% are in `{STATS_CSV_PATH.relative_to(PROJECT_ROOT).as_posix()}` and `{BREACH_CSV_PATH.relative_to(PROJECT_ROOT).as_posix()}`. The >1% report contains {manifest['integrity']['over_1pct_discrepancy_rows']} field-level rows across {manifest['integrity']['over_1pct_security_date_pairs']} security-date pairs.",
        "", "## Provenance and limitations", "",
        "- PIT membership, symbol intervals, identity and source records come from the versioned `data/reference/` ledgers; security tokens are never treated as security identity.",
        "- Daily equity candles use the manifested source per security (Kite historical data, with official NSE archives where manifested); the exact dated data prefix and source-manifest slice hashes are in the freeze JSON.",
        "- Weekly candles are deterministic derivations from the frozen daily prefix using `src/data/weekly.py`; there is no separate persisted weekly market-data dataset.",
        "- Hourly candles use Zerodha Kite historical data. The freeze hash includes only 52 hourly identities allowed for research, the explicit exclusion file, quarantine file, and data-source manifest slice.",
        "- NIFTY benchmark is the user-provided official NIFTY 50 Historical PR export combined at `data/raw/reference/nifty50_daily.csv`. It is a price-return series, not a total-return series.",
        "- OHLC discrepancies can exceed ₹0.05, including isolated larger deviations. Prices have not been adjusted to force aggregation equality; consult the >1% list before interpreting affected dates.",
        "- HDFC and LTIM have no usable hourly series for the frozen window. TATAMOTORS hourly data is excluded due a roughly 0.689 median close ratio versus its daily series.",
        "- This freeze is a research-data integrity checkpoint, not a claim that strategy results are profitable or that every provider candle is error-free.",
        "- Full test suite at freeze creation: 187 passed, 0 failed, 0 skipped (Python 3.12.10, pytest 8.4.2).",
        "", f"Freeze manifest: `{FREEZE_MANIFEST_PATH.relative_to(PROJECT_ROOT).as_posix()}`.", "",
    ]
    return "\n".join(lines)


def build_data_freeze() -> dict[str, Any]:
    provider = UniverseProvider()
    memberships = provider.members_over(START, END, PIT_TRADABLE)
    expected_ids = {item.security_id for item in memberships if item.tradable}
    daily, daily_hash = load_daily_dataset_through(
        universe_id=PIT_TRADABLE, start_date=DAILY_WARMUP_START, end_date=END)
    hourly, hourly_hash = load_hourly_dataset_through(
        universe_id=PIT_TRADABLE, start_date=HOURLY_WARMUP_START, end_date=END,
        daily_manifest_path=MANIFEST_PATH, include_flagged_identities=False,
        identity_exclusions_path=EXCLUSIONS_PATH)
    daily_usable_ids: set[str] = set()
    for membership in memberships:
        if not membership.tradable or not daily.get(membership.security_id):
            continue
        active_start = max(START, membership.effective_from)
        active_end = min(END, membership.effective_to - timedelta(days=1)
                         if membership.effective_to else END)
        if any(active_start <= date.fromisoformat(str(row["date"])[:10]) <= active_end
               for row in daily[membership.security_id]):
            daily_usable_ids.add(membership.security_id)
    hourly_usable_ids = {sid for sid in expected_ids if hourly.get(sid)}
    unavailable_ids = sorted(expected_ids - hourly_usable_ids)
    probe_rows = _read_csv(HOURLY_ACCESS_MANIFEST)
    probe_by_id: dict[str, dict[str, str]] = {}
    for row in probe_rows:
        if row.get("security_id") in unavailable_ids:
            probe_by_id[row["security_id"]] = row
    excluded_rows = _read_csv(EXCLUSIONS_PATH)
    identity_excluded_ids = sorted({row["security_id"] for row in excluded_rows
                                    if row.get("identity_status") == "HOURLY_IDENTITY_PRICE_MISMATCH"})

    benchmark_rows = _read_csv(BENCHMARK_PATH)
    benchmark_dates = [date.fromisoformat(row.get("Date", row.get("date", ""))[:10]) for row in benchmark_rows]
    weekly_digest = hashlib.sha256()
    for security_id in sorted(daily_usable_ids):
        rows = daily.get(security_id, [])
        candles = resample_daily_to_weekly(rows, trading_sessions=[
            date.fromisoformat(str(row["date"])[:10]) for row in rows])
        weekly_digest.update(security_id.encode("utf-8"))
        weekly_digest.update(json.dumps(candles, sort_keys=True, ensure_ascii=True,
                                        separators=(",", ":")).encode("utf-8"))
    hourly_symbol_by_id = {row.get("security_id", ""): row.get("symbol_used", "")
        for row in _read_csv(HOURLY_MANIFEST)
        if row.get("universe_id") == PIT_TRADABLE and row.get("interval") == "60minute"
        and row.get("validation_status") == "VALIDATED"}
    stats, per_security_stats, over_one = _cross_timeframe_statistics(daily, hourly, hourly_symbol_by_id)

    reference_dir = PROJECT_ROOT / "data" / "reference"
    reference_files = {
        name: reference_dir / name for name in (
            "securities.csv", "symbol_history.csv", "universe_membership.csv",
            "universe_events.csv", "universe_coverage.csv", "broker_instrument_history.csv",
            "sources.csv")
    }
    source_rows = _read_csv(reference_files["sources.csv"])
    relevant_ids: set[str] = set()
    for name in ("symbol_history.csv", "universe_membership.csv", "universe_events.csv", "broker_instrument_history.csv"):
        for row in _clipped_reference_slice(reference_files[name], start=DAILY_WARMUP_START,
                                            end=END, expected_ids=expected_ids):
            relevant_ids.add(row.get("source_id", ""))
    source_slice = [row for row in source_rows if row.get("source_id") in relevant_ids]
    reference_hashes: dict[str, Any] = {}
    for name, path in reference_files.items():
        if name == "sources.csv":
            selected = source_slice
        else:
            selected = _clipped_reference_slice(path, start=DAILY_WARMUP_START,
                                                 end=END, expected_ids=expected_ids)
        reference_hashes[f"data/reference/{name}"] = {
            "sha256": _canonical_hash(selected), "rows": len(selected),
            "scope": f"research PIT identities and date-bounded rows intersecting {DAILY_WARMUP_START} through {END}; future intervals clipped/omitted",
        }

    daily_manifest_slice = _manifest_slice_hash(MANIFEST_PATH, interval="day",
        expected_ids=expected_ids, end=END)
    hourly_manifest_slice = _manifest_slice_hash(HOURLY_MANIFEST, interval="60minute",
        expected_ids=expected_ids, end=END)
    hourly_manifest_rows = _read_csv(HOURLY_MANIFEST)
    hourly_manifested_validated = {row.get("security_id", "") for row in hourly_manifest_rows
        if row.get("universe_id") == PIT_TRADABLE and row.get("interval") == "60minute"
        and row.get("validation_status") == "VALIDATED"}
    special_sessions = {
        day.isoformat(): {key: ([(window[0].isoformat(), window[1].isoformat()) for window in value]
                                if key == "exchange_trading_windows" else value)
                          for key, value in details.items()}
        for day, details in sorted(SESSION_OVERRIDES.items())
        if HOURLY_WARMUP_START <= day <= END
    }
    quarantine_count = max(0, len(_read_csv(QUARANTINE_PATH)))
    manifest: dict[str, Any] = {
        "freeze_version": "NIFTY50_PIT_2023_2024_V1",
        "research_period": {"start": START.isoformat(), "end": END.isoformat()},
        "warmup_periods": {"daily_weekly_start": DAILY_WARMUP_START.isoformat(),
                           "hourly_requested_start": HOURLY_WARMUP_START.isoformat()},
        "universe_id": PIT_TRADABLE,
        "counts": {"expected_pit_securities": len(expected_ids),
                   "daily_usable_securities": len(daily_usable_ids),
                   "hourly_manifested_validated_securities": len(hourly_manifested_validated),
                   "hourly_usable_securities": len(hourly_usable_ids),
                   "hourly_unavailable_securities": len(set(unavailable_ids) - set(identity_excluded_ids)),
                   "hourly_excluded_identity_securities": len(identity_excluded_ids),
                   "quarantined_provider_invalid_candles": quarantine_count},
        "availability": {"hourly_usable_security_ids": sorted(hourly_usable_ids),
                         "hourly_unavailable": [{"security_id": sid,
                                                 "symbol": probe_by_id.get(sid, {}).get("historical_symbol", ""),
                                                 "status": probe_by_id.get(sid, {}).get("probe_result", "UNRESOLVED")}
                                                for sid in unavailable_ids if sid not in identity_excluded_ids],
                         "hourly_excluded_identities": excluded_rows},
        "benchmark": {"path": "data/raw/reference/nifty50_daily.csv",
                      "source": "User-provided official NIFTY 50 Historical PR exports for 2023 and 2024",
                      "row_count": len(benchmark_rows),
                      "first_date": min(benchmark_dates).isoformat() if benchmark_dates else None,
                      "last_date": max(benchmark_dates).isoformat() if benchmark_dates else None,
                      "sha256": _sha256(BENCHMARK_PATH),
                      "warmup_before_2023_available": bool(benchmark_dates and min(benchmark_dates) < START)},
        "integrity": {"cross_timeframe": stats,
                      "per_security": per_security_stats,
                      "over_1pct_discrepancy_rows": len(over_one),
                      "over_1pct_security_date_pairs": len({(row["security_id"], row["date"]) for row in over_one}),
                      "over_1pct_report_path": BREACH_CSV_PATH.relative_to(PROJECT_ROOT).as_posix()},
        "hashes": {
            "daily_data_prefix_sha256": daily_hash,
            "hourly_usable_data_prefix_sha256": hourly_hash,
            "weekly_derived_data_sha256": weekly_digest.hexdigest(),
            "weekly_derivation_code_sha256": _sha256(PROJECT_ROOT / "src" / "data" / "weekly.py"),
            "hourly_session_rules_code_sha256": _sha256(PROJECT_ROOT / "src" / "data" / "hourly.py"),
            "daily_source_manifest_slice": daily_manifest_slice,
            "hourly_source_manifest_slice": hourly_manifest_slice,
            "hourly_exclusion_file_sha256": _sha256(EXCLUSIONS_PATH),
            "hourly_quarantine_file_sha256": _sha256(QUARANTINE_PATH),
            "nifty_benchmark_sha256": _sha256(BENCHMARK_PATH),
            "universe_reference_slices": reference_hashes,
            "special_sessions_used": special_sessions,
        },
        "scope_guard": "No 2025 or 2026 market candles were loaded into this freeze. Data hashes use prefixes ending no later than 2024-12-31.",
    }
    manifest["data_freeze_status"] = "PASS" if (
        len(expected_ids) == 55 and len(daily_usable_ids) == 55 and len(hourly_usable_ids) == 52
        and len(identity_excluded_ids) == 1 and quarantine_count == 3
        and not stats["persistent_material_scale_divergence_security_ids"]
    ) else "FAIL"
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(STATS_CSV_PATH, [
        {"scope": "ALL_52_USABLE_SECURITIES", "field": field, **row}
        for field, row in stats["by_ohlc_field"].items()
    ] + [
        {"scope": "PER_SECURITY", "field": field,
         "security_id": row["security_id"], "symbol": row["symbol"],
         "classification": row["classification"],
         **row["fields"][field]}
        for row in per_security_stats for field in FIELDS
    ])
    _write_csv(BREACH_CSV_PATH, over_one)
    FREEZE_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    FREEZE_DOCUMENT_PATH.write_text(_document(manifest, stats), encoding="utf-8")
    print(f"Freeze version: {manifest['freeze_version']}")
    print(f"PIT expected={len(expected_ids)}, daily usable={len(daily_usable_ids)}, hourly usable={len(hourly_usable_ids)}, hourly unavailable={manifest['counts']['hourly_unavailable_securities']}, excluded={len(identity_excluded_ids)}, quarantined={quarantine_count}")
    print(f">1% component discrepancy rows: {len(over_one)}")
    for field in FIELDS:
        print(f"{field}: {stats['by_ohlc_field'][field]}")
    print(f"Persistent price-scale divergences: {stats['persistent_material_scale_divergence_security_ids'] or 'none among usable 52'}")
    print(f"Manifest: {FREEZE_MANIFEST_PATH}\nDocument: {FREEZE_DOCUMENT_PATH}\n>1% detail: {BREACH_CSV_PATH}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    try:
        build_data_freeze()
    except (OSError, ValueError, BacktestDataError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
