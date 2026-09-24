"""Resumable, date-aware Kite 60-minute acquisition and audit for PIT equities."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from collections import defaultdict
from datetime import date, datetime, time as day_time, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable
from zoneinfo import ZoneInfo

from kiteconnect.exceptions import KiteException

from src.broker.session import create_authenticated_client
from src.configuration import PROJECT_ROOT
from src.data.dataset import NEW_LISTING_SECURITY_IDS, RequestThrottle
from src.data.historical import INTERVAL_MAX_DAYS
from src.data.kite_probe import KiteProbeError, latest_instrument_snapshot
from src.data.market_data import DATA_ROOT, MANIFEST_PATH, read_manifest
from src.data.universe.provider import PIT_TRADABLE, UniverseCoverageError, UniverseProvider


HOURLY_MANIFEST = PROJECT_ROOT / "data" / "processed" / "hourly_market_data_manifest.csv"
HOURLY_ACCESS_MANIFEST = PROJECT_ROOT / "data" / "processed" / "hourly_access_probe_manifest.csv"
INTERVAL = "60minute"
SOURCE = "Kite Connect historical API"
TIMEZONE = "Asia/Kolkata"
ZONE = ZoneInfo(TIMEZONE)
FIELDS = (
    "date", "open", "high", "low", "close", "volume", "session_type",
    "provider_interval", "provider_bar_start_semantics", "provider_timestamp_raw",
    "provider_grid_status", "data_quality_status",
)
HOURLY_QUARANTINE_PATH = PROJECT_ROOT / "data" / "processed" / "hourly" / "quarantine_invalid_candles.csv"
MANIFEST_FIELDS = (
    "universe_id", "security_id", "source", "broker", "instrument_token",
    "broker_mapping_provenance", "market_data_access_provenance",
    "mapping_source_id", "mapping_effective_from", "mapping_effective_to",
    "symbol_used", "interval", "requested_start", "requested_end",
    "actual_first_timestamp", "actual_last_timestamp", "row_count",
    "missing_session_diagnostics", "downloaded_at", "content_sha256",
    "provenance", "validation_status", "relative_path", "invalid_provider_candles",
    "quarantined_candles", "recovered_by_refetch", "missing_hourly_bars_due_to_quarantine",
)
QUARANTINE_FIELDS = (
    "security_id", "symbol", "token", "timestamp", "raw_ohlcv", "failed_invariant",
    "retry_result", "source", "retry_count", "data_quality_status",
    "refetched_raw_response", "quarantined_at",
)
ACCESS_FIELDS = (
    "universe_id", "security_id", "historical_symbol", "symbol_effective_from",
    "symbol_effective_to", "evaluation_start", "evaluation_end", "warmup_start",
    "requested_start", "requested_end", "snapshot_date", "current_symbol",
    "current_resolved_token", "current_instrument_lookup_source", "identity_status",
    "instrument_token", "broker_mapping_source_id",
    "mapping_effective_from", "mapping_effective_to", "probe_date",
    "probe_actual_first_timestamp", "probe_actual_last_timestamp", "probe_row_count",
    "probe_result", "market_data_access_result", "failure_detail", "market_data_access_provenance",
)
VERIFIED_CONTINUITY_RENAME_SOURCES = {
    "zomato_rename", "nse_ltm_symbol_change", "nse_ltm_name_change",
}
OPTIONAL_FINAL_PARTIAL_START = day_time(15, 15)
REGULAR_REQUIRED_BAR_STARTS = frozenset(day_time(hour, 15) for hour in range(9, 15))
REGULAR_VALID_BAR_STARTS = REGULAR_REQUIRED_BAR_STARTS | {OPTIONAL_FINAL_PARTIAL_START}
EXPECTED_FULL_BAR_STARTS = tuple(sorted(REGULAR_REQUIRED_BAR_STARTS))
EXPECTED_FULL_BAR_START_SET = REGULAR_REQUIRED_BAR_STARTS
SESSION_OVERRIDES = {
    # NSE Muhurat session, Monday 24-Oct-2022. Official session notice: NSE/CMTR/54023.
    date(2022, 10, 24): {
        "session_type": "MUHURAT", "source": "NSE/CMTR/54023",
        "exchange_trading_windows": ((day_time(18, 15), day_time(19, 15)),),
    },
    # NSE live session on Saturday 20-Jan-2024 used regular cash-market timings.
    # Official schedule: NSE/MSD/60340, https://nsearchives.nseindia.com/content/circulars/MSD60340.pdf
    date(2024, 1, 20): {
        "session_type": "SPECIAL_REGULAR_SESSION",
        "exchange_trading_windows": ((day_time(9, 15), day_time(15, 30)),),
        "source": "NSE/MSD/60340",
    },
    # NSE live session on Saturday 02-Mar-2024 split across Primary and DR sites.
    # Official schedule: NSE/MSD/60677, https://nsearchives.nseindia.com/content/circulars/MSD60677.pdf
    date(2024, 3, 2): {
        "session_type": "SPECIAL_SPLIT_SESSION",
        "exchange_trading_windows": ((day_time(9, 15), day_time(10, 0)),
                                     (day_time(11, 30), day_time(12, 30))),
        "source": "NSE/MSD/60677",
    },
    # NSE live session on Saturday 18-May-2024 split across Primary and DR sites.
    # Official schedule: NSE/MSD/61893, https://nsearchives.nseindia.com/content/circulars/MSD61893.pdf
    date(2024, 5, 18): {
        "session_type": "SPECIAL_SPLIT_SESSION",
        "exchange_trading_windows": ((day_time(9, 15), day_time(10, 0)),
                                     (day_time(11, 30), day_time(12, 30))),
        "source": "NSE/MSD/61893",
    },
    # NSE capital-market Muhurat session, Sunday 12-Nov-2023, open 18:15 close 19:15.
    # Official schedule: NSE/CMTR/59124, https://nsearchives.nseindia.com/content/circulars/CMTR59124.pdf
    date(2023, 11, 12): {
        "session_type": "MUHURAT",
        "exchange_trading_windows": ((day_time(18, 15), day_time(19, 15)),),
        "source": "NSE/CMTR/59124",
    },
    # NSE capital-market Muhurat session, Friday 01-Nov-2024, open 18:00 close 19:00.
    # Official schedule: NSE/CMTR/64628, https://nsearchives.nseindia.com/content/circulars/CMTR64628.pdf
    date(2024, 11, 1): {
        "session_type": "MUHURAT",
        "exchange_trading_windows": ((day_time(18, 0), day_time(19, 0)),),
        "source": "NSE/CMTR/64628",
    },
}


class HourlyDataError(RuntimeError):
    """Hourly data acquisition, persistence, or validation error."""


class _InvalidProviderCandle(ValueError):
    def __init__(self, failed_invariant: str):
        super().__init__(failed_invariant)
        self.failed_invariant = failed_invariant


def _aware(value: Any) -> datetime:
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.strip())
        except ValueError:
            raise HourlyDataError("Hourly candle has an invalid timestamp.") from None
    if not isinstance(value, datetime):
        raise HourlyDataError("Hourly candle has an invalid timestamp.")
    return value.replace(tzinfo=ZONE) if value.tzinfo is None else value.astimezone(ZONE)


def session_type_on(trading_date: date, session_dates: set[date] | None = None) -> str:
    """Classify only exchange dates present in the calendar or explicit NSE overrides."""
    if trading_date in SESSION_OVERRIDES:
        return str(SESSION_OVERRIDES[trading_date]["session_type"])
    if (session_dates is not None and trading_date in session_dates
            and trading_date.weekday() < 5):
        return "NORMAL_SESSION"
    return "CLOSED"


def required_hourly_starts(trading_date: date, session_dates: set[date] | None = None) -> frozenset[day_time]:
    """Return required 60-minute start times for a known trading session date."""
    if session_type_on(trading_date, session_dates) == "NORMAL_SESSION":
        return REGULAR_REQUIRED_BAR_STARTS
    return frozenset()


def _is_expected_special_session_gap(previous: datetime, current: datetime) -> bool:
    if previous.date() != current.date():
        return False
    previous_time = previous.time().replace(tzinfo=None)
    current_time = current.time().replace(tzinfo=None)
    # Special schedules may contain long, legitimate breaks. The explicit date tag
    # prevents these from being confused with ordinary-session gaps in the audit.
    return session_type_on(current.date()) in {
        "SPECIAL_REGULAR_SESSION", "SPECIAL_SPLIT_SESSION", "MUHURAT"
    } and current_time > previous_time


def valid_session_timestamp(value: Any, session_dates: set[date] | None = None) -> bool:
    """Validate the provider candle start against a date-specific NSE cash session."""
    original = value
    if isinstance(value, str):
        try:
            original = datetime.fromisoformat(value.strip())
        except ValueError:
            return False
    if isinstance(original, datetime) and original.tzinfo is not None:
        if original.utcoffset() != timedelta(hours=5, minutes=30):
            return False
    try:
        stamp = _aware(value)
    except HourlyDataError:
        return False
    kind = session_type_on(stamp.date(), session_dates)
    local = stamp.timetz()
    if kind == "NORMAL_SESSION":
        return local.replace(tzinfo=None) in REGULAR_VALID_BAR_STARTS and not stamp.second and not stamp.microsecond
    if kind not in {"SPECIAL_REGULAR_SESSION", "SPECIAL_SPLIT_SESSION", "MUHURAT"}:
        return False
    return (
        local.hour * 60 + local.minute >= 8 * 60 + 30
        and local.hour * 60 + local.minute <= 20 * 60 + 30
        and local.minute in {0, 15}
        and not stamp.second and not stamp.microsecond
    )


def _raw_candle_summary(row: Any) -> str:
    if not isinstance(row, dict):
        return repr(row)
    fields = {key: row.get(key) for key in ("date", "open", "high", "low", "close", "volume")}
    return repr(fields)


def _timestamp_detail(value: Any) -> tuple[str, str]:
    raw = repr(value)
    try:
        local = _aware(value).isoformat()
    except HourlyDataError:
        local = "<cannot convert to Asia/Kolkata>"
    return raw, local


def _ohlcv_failure_diagnostic(
    rows: list[Any], index: int, *, failed_invariant: str,
    context: dict[str, Any] | None = None,
) -> str:
    """Describe a malformed provider candle without hiding or changing its values."""
    context = context or {}
    row = rows[index]
    raw_stamp = row.get("date") if isinstance(row, dict) else None
    raw_timestamp, local_timestamp = _timestamp_detail(raw_stamp)
    nearby = lambda left, right: [
        _raw_candle_summary(rows[position]) for position in range(left, right)
    ]
    return "\n".join((
        f"Hourly candle {index} has inconsistent OHLCV values.",
        f"security_id={context.get('security_id', '<unknown>')}; "
        f"symbol={context.get('symbol', '<unknown>')}; "
        f"instrument_token={context.get('instrument_token', '<unknown>')}; "
        f"requested_chunk={context.get('chunk_start', '<unknown>')}..{context.get('chunk_end', '<unknown>')}",
        f"candle_index={index}; raw_timestamp={raw_timestamp}; "
        f"asia_kolkata_timestamp={local_timestamp}",
        f"raw_OHLCV={_raw_candle_summary(row)}",
        f"failed_invariant={failed_invariant}",
        f"previous_3={nearby(max(0, index - 3), index)}",
        f"next_3={nearby(index + 1, min(len(rows), index + 4))}",
    ))


def _hourly_grid_diagnostic(
    rows: list[Any], index: int, *, context: dict[str, Any] | None = None,
    session_dates: set[date] | None = None,
) -> str:
    context = context or {}
    row = rows[index]
    raw_stamp = row.get("date") if isinstance(row, dict) else None
    raw_timestamp, local_timestamp = _timestamp_detail(raw_stamp)
    try:
        stamp = _aware(raw_stamp)
        kind = session_type_on(stamp.date(), session_dates)
        if kind == "CLOSED":
            reason = "local date is not in the validated NSE session calendar and is not a known special session"
        elif kind == "NORMAL_SESSION" and not day_time(9, 15) <= stamp.time().replace(tzinfo=None) <= OPTIONAL_FINAL_PARTIAL_START:
            reason = "local time is outside 09:15 through 15:15 normal-session starts"
        elif kind == "NORMAL_SESSION" and (stamp.second or stamp.microsecond):
            reason = "seconds or microseconds are non-zero"
        elif kind == "NORMAL_SESSION" and stamp.minute != 15:
            reason = "minute is off the verified :15 hourly start grid"
        elif kind in {"SPECIAL_REGULAR_SESSION", "SPECIAL_SPLIT_SESSION", "MUHURAT"}:
            reason = "special-session timestamp fails native-time plausibility checks"
        else:
            reason = "local date is absent from the validated NSE daily-session calendar"
    except HourlyDataError:
        reason = "timestamp cannot be parsed"
    nearby = lambda left, right: [
        _raw_candle_summary(rows[position])
        for position in range(left, right)
    ]
    distinct_times: set[str] = set()
    for item in rows:
        candidate = item.get("date") if isinstance(item, dict) else None
        try:
            distinct_times.add(_aware(candidate).strftime("%H:%M:%S.%f"))
        except HourlyDataError:
            distinct_times.add(f"UNPARSEABLE:{candidate!r}")
    details = [
        f"Hourly candle {index} is outside the NSE hourly session grid ({reason}).",
        f"security_id={context.get('security_id', '<unknown>')}; symbol={context.get('symbol', '<unknown>')}; "
        f"instrument_token={context.get('instrument_token', '<unknown>')}; "
        f"requested_chunk={context.get('chunk_start', '<unknown>')}..{context.get('chunk_end', '<unknown>')}",
        f"raw_timestamp={raw_timestamp}; asia_kolkata_timestamp={local_timestamp}",
        f"OHLCV={_raw_candle_summary(row)}",
        f"previous_3={nearby(max(0, index - 3), index)}",
        f"next_3={nearby(index + 1, min(len(rows), index + 4))}",
        f"distinct_intraday_times={sorted(distinct_times)}",
    ]
    return "\n".join(details)


def _validate_provider_ohlcv(row: dict[str, Any]) -> tuple[dict[str, float], int]:
    values: dict[str, float] = {}
    for field in ("open", "high", "low", "close"):
        raw = row.get(field)
        if isinstance(raw, bool):
            raise _InvalidProviderCandle(f"invalid_{field}: boolean value")
        try:
            value = float(raw)
        except (TypeError, ValueError):
            raise _InvalidProviderCandle(f"invalid_{field}: not numeric") from None
        if not math.isfinite(value):
            raise _InvalidProviderCandle(f"non_finite_{field}")
        if value <= 0:
            raise _InvalidProviderCandle(f"non_positive_{field}")
        values[field] = value
    violations = [
        name for name, invalid in (
            ("high < open", values["high"] < values["open"]),
            ("high < close", values["high"] < values["close"]),
            ("high < low", values["high"] < values["low"]),
            ("low > open", values["low"] > values["open"]),
            ("low > close", values["low"] > values["close"]),
        ) if invalid
    ]
    if violations:
        raise _InvalidProviderCandle(", ".join(violations))
    volume = row.get("volume")
    if isinstance(volume, bool):
        raise _InvalidProviderCandle("invalid volume: boolean value")
    try:
        volume_number = float(volume)
    except (TypeError, ValueError):
        raise _InvalidProviderCandle("invalid volume: not numeric") from None
    if not math.isfinite(volume_number):
        raise _InvalidProviderCandle("non_finite_volume")
    if volume_number < 0:
        raise _InvalidProviderCandle("negative volume")
    if not volume_number.is_integer():
        raise _InvalidProviderCandle("fractional volume")
    return values, int(volume_number)


def normalize_hourly(
    rows: Iterable[dict[str, Any]], *, session_dates: set[date] | None = None,
    diagnostic_context: dict[str, Any] | None = None,
    on_invalid_ohlcv: Callable[[dict[str, Any], int, str], tuple[dict[str, Any] | None, str]] | None = None,
) -> list[dict[str, Any]]:
    """Normalize Kite rows deterministically without synthesizing candles."""
    try:
        source = list(rows)
    except TypeError:
        raise HourlyDataError("Kite returned an invalid hourly response.") from None
    if not source:
        return []
    seen_source_stamps: set[str] = set()
    for row in source:
        if not isinstance(row, dict):
            continue
        try:
            source_stamp = _aware(row.get("date")).isoformat()
        except HourlyDataError:
            continue
        if source_stamp in seen_source_stamps:
            raise HourlyDataError("Hourly data contains duplicate timestamps.")
        seen_source_stamps.add(source_stamp)
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(source):
        if not isinstance(row, dict):
            raise HourlyDataError(f"Hourly candle {index} is not a mapping.")
        try:
            stamp = _aware(row.get("date"))
        except HourlyDataError:
            raise HourlyDataError(_hourly_grid_diagnostic(
                source, index, context=diagnostic_context, session_dates=session_dates
            )) from None
        if not valid_session_timestamp(row.get("date"), session_dates):
            raise HourlyDataError(_hourly_grid_diagnostic(
                source, index, context=diagnostic_context, session_dates=session_dates
            ))
        try:
            values, volume_number = _validate_provider_ohlcv(row)
        except _InvalidProviderCandle as error:
            if on_invalid_ohlcv is None:
                raise HourlyDataError(_ohlcv_failure_diagnostic(
                    source, index, failed_invariant=error.failed_invariant, context=diagnostic_context
                )) from None
            replacement, quality_status = on_invalid_ohlcv(row, index, error.failed_invariant)
            if replacement is None:
                # The invalid provider bar has been recorded in quarantine.
                continue
            recovered = normalize_hourly(
                [replacement], session_dates=session_dates, diagnostic_context=diagnostic_context,
            )[0]
            recovered["data_quality_status"] = quality_status
            normalized.append(recovered)
            continue
        special = SESSION_OVERRIDES.get(stamp.date())
        kind = session_type_on(stamp.date(), session_dates)
        if kind in {"SPECIAL_REGULAR_SESSION", "SPECIAL_SPLIT_SESSION", "MUHURAT"}:
            previous_stamp = None
            for prior in source[:index]:
                try:
                    prior_stamp = _aware(prior.get("date"))
                except (HourlyDataError, AttributeError):
                    continue
                if prior_stamp.date() == stamp.date():
                    previous_stamp = prior_stamp
            if previous_stamp is not None and stamp <= previous_stamp:
                raise HourlyDataError(
                    f"Special-session Kite timestamps must be strictly increasing; {stamp.isoformat()} "
                    f"follows {previous_stamp.isoformat()} in provider response."
                )
        raw_timestamp = row.get("provider_timestamp_raw")
        if raw_timestamp is None:
            raw_timestamp = row.get("date")
        if isinstance(raw_timestamp, datetime):
            raw_timestamp = raw_timestamp.isoformat()
        normalized.append({
            "date": stamp.isoformat(), **values, "volume": int(volume_number),
            "session_type": special["session_type"] if special else kind,
            "provider_interval": row.get("provider_interval") or INTERVAL,
            "provider_bar_start_semantics": row.get("provider_bar_start_semantics") or "KITE_NATIVE",
            "provider_timestamp_raw": str(raw_timestamp),
            "provider_grid_status": "NATIVE_SPECIAL_SESSION" if kind in {
                "SPECIAL_REGULAR_SESSION", "SPECIAL_SPLIT_SESSION", "MUHURAT"
            } else "NORMAL_GRID_VALIDATED",
            "data_quality_status": row.get("data_quality_status") or "VALIDATED",
        })
    normalized.sort(key=lambda row: row["date"])
    seen: set[str] = set()
    for row in normalized:
        if row["date"] in seen:
            raise HourlyDataError("Hourly data contains duplicate timestamps.")
        seen.add(row["date"])
    return normalized


def hourly_chunks(start_date: date, end_date: date) -> list[tuple[datetime, datetime]]:
    if start_date > end_date:
        raise HourlyDataError("Hourly start_date must not be after end_date.")
    max_days = INTERVAL_MAX_DAYS[INTERVAL]
    chunks = []
    cursor = start_date
    while cursor <= end_date:
        chunk_end = min(end_date, cursor + timedelta(days=max_days - 1))
        chunks.append((datetime.combine(cursor, day_time.min), datetime.combine(chunk_end, day_time.max)))
        cursor = chunk_end + timedelta(days=1)
    return chunks


def _atomic_csv(path: Path, fields: tuple[str, ...], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    try:
        with temp.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        temp.replace(path)
    except OSError:
        temp.unlink(missing_ok=True)
        raise HourlyDataError("Could not write hourly data safely.") from None


def _manifest_rows(path: str | Path) -> list[dict[str, str]]:
    target = Path(path)
    if not target.exists():
        return []
    with target.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_manifest(path: str | Path, rows: list[dict[str, str]]) -> None:
    _atomic_csv(Path(path), MANIFEST_FIELDS, rows)


def _json_provider_value(value: Any) -> str:
    def safe(item: Any) -> Any:
        if isinstance(item, float) and not math.isfinite(item):
            return repr(item)
        if isinstance(item, (datetime, date)):
            return item.isoformat()
        if isinstance(item, dict):
            return {str(key): safe(value) for key, value in item.items()}
        if isinstance(item, (list, tuple)):
            return [safe(value) for value in item]
        return item

    return json.dumps(safe(value), ensure_ascii=True, sort_keys=True, allow_nan=False,
                      default=lambda item: repr(item))


def _record_quarantine(path: str | Path, record: dict[str, Any]) -> None:
    target = Path(path)
    rows = _manifest_rows(target)
    key = (record.get("security_id"), record.get("symbol"), record.get("timestamp"))
    for index, old in enumerate(rows):
        if (old.get("security_id"), old.get("symbol"), old.get("timestamp")) == key:
            rows[index] = {**old, **record}
            break
    else:
        rows.append(record)
    _atomic_csv(target, QUARANTINE_FIELDS, rows)


def hourly_window_coverage_status(
    *, security_id: str, start: datetime, end: datetime,
    quarantine_path: str | Path = HOURLY_QUARANTINE_PATH,
    available_timestamps: Iterable[datetime | str] = (),
) -> str:
    """Flag analysis windows containing a quarantined missing provider bar."""
    present: set[str] = set()
    for value in available_timestamps:
        try:
            present.add(_aware(value).isoformat())
        except HourlyDataError:
            continue
    for row in _manifest_rows(quarantine_path):
        if row.get("security_id") != security_id:
            continue
        try:
            stamp = _aware(row.get("timestamp"))
        except HourlyDataError:
            continue
        if _aware(start) <= stamp <= _aware(end) and stamp.isoformat() not in present:
            return "PARTIAL_HOURLY_DATA"
    return "COMPLETE_HOURLY_DATA"


def _read_existing(path: Path, *, session_dates: set[date] | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return normalize_hourly(list(csv.DictReader(handle)), session_dates=session_dates)
    except (OSError, csv.Error, KeyError, HourlyDataError):
        # Corrupt files are not overwritten silently; force explicit repair.
        raise HourlyDataError(f"Existing hourly file is invalid and needs review: {path.name}") from None


def _merge(existing: Iterable[dict[str, Any]], incoming: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    by_timestamp: dict[str, dict[str, Any]] = {}
    for candle in [*existing, *incoming]:
        stamp = candle["date"]
        prior = by_timestamp.get(stamp)
        if prior is not None:
            comparable = [key for key in FIELDS[1:] if key != "data_quality_status"]
            if any(prior.get(key) != candle.get(key) for key in comparable):
                raise HourlyDataError(f"Conflicting hourly candles at {stamp}; refusing unsafe overwrite.")
            if prior.get("data_quality_status") == "RECOVERED_BY_REFETCH":
                candle["data_quality_status"] = "RECOVERED_BY_REFETCH"
        by_timestamp[stamp] = candle
    return [by_timestamp[key] for key in sorted(by_timestamp)]


def _daily_sessions(start: date, end: date, daily_manifest: str | Path, data_root: str | Path) -> set[date]:
    sessions: set[date] = set()
    root = Path(data_root)
    for item in read_manifest(daily_manifest):
        if item.get("interval") != "day" or item.get("validation_status") not in {
            "VALIDATED", "VALIDATED_EVALUATION_PARTIAL_WARMUP"
        }:
            continue
        target = root / item.get("relative_path", "")
        if not target.is_file():
            continue
        try:
            with target.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    try:
                        day = date.fromisoformat(row["date"][:10])
                    except (KeyError, ValueError):
                        continue
                    if start <= day <= end:
                        sessions.add(day)
        except OSError:
            continue
    sessions.update(day for day in SESSION_OVERRIDES if start <= day <= end)
    return sessions


def _missing_session_diagnostics(rows: list[dict[str, Any]], sessions: set[date]) -> str:
    observed = {datetime.fromisoformat(row["date"]).date(): set() for row in rows}
    for row in rows:
        stamp = datetime.fromisoformat(row["date"])
        observed.setdefault(stamp.date(), set()).add(stamp.time().replace(tzinfo=None))
    missing = []
    for session in sorted(sessions):
        required = required_hourly_starts(session, sessions)
        absent = [slot.strftime("%H:%M") for slot in sorted(required) if slot not in observed.get(session, set())]
        if absent:
            missing.append(f"{session.isoformat()}:{','.join(absent)}")
    return ";".join(missing)


def _request_windows(
    *, existing: list[dict[str, Any]], start: date, end: date,
    daily_manifest: str | Path, data_root: str | Path,
) -> list[tuple[date, date]]:
    """Return only missing daily sessions, refreshing the last existing session."""
    sessions = _daily_sessions(start, end, daily_manifest, data_root)
    if not existing:
        return [(start, end)]
    observed: dict[date, set[day_time]] = defaultdict(set)
    for row in existing:
        stamp = datetime.fromisoformat(row["date"])
        observed[stamp.date()].add(stamp.time().replace(tzinfo=None))
    missing_days: set[date] = set()
    extra_ranges: list[tuple[date, date]] = []
    if sessions:
        for session in sessions:
            required = required_hourly_starts(session, sessions)
            if required and not required.issubset(observed.get(session, set())):
                missing_days.add(session)
        if max(sessions) < end:
            last_existing = max(datetime.fromisoformat(row["date"]).date() for row in existing)
            if last_existing < end:
                extra_ranges.append((max(max(sessions) + timedelta(days=1), last_existing + timedelta(days=1)), end))
    else:
        last = max(datetime.fromisoformat(row["date"]).date() for row in existing)
        if start <= last <= end:
            missing_days.add(last)
        if last < end:
            extra_ranges.append((last + timedelta(days=1), end))
    # Refresh the latest observed day so an unfinished/final provider candle can update.
    last_day = max(datetime.fromisoformat(row["date"]).date() for row in existing)
    if start <= last_day <= end:
        missing_days.add(last_day)
    ordered = sorted(missing_days)
    windows: list[tuple[date, date]] = []
    for day in ordered:
        if windows and day <= windows[-1][1] + timedelta(days=1):
            windows[-1] = (windows[-1][0], day)
        else:
            windows.append((day, day))
    windows.extend(extra_ranges)
    merged: list[tuple[date, date]] = []
    for left, right in sorted(windows):
        if merged and left <= merged[-1][1] + timedelta(days=1):
            merged[-1] = (merged[-1][0], max(merged[-1][1], right))
        else:
            merged.append((left, right))
    return merged


def _record(
    *, manifest_path: str | Path, record: dict[str, str],
) -> None:
    rows = _manifest_rows(manifest_path)
    if record.get("validation_status") == "VALIDATED":
        rows = [old for old in rows if not (
            old.get("validation_status") == "NO_KITE_HISTORICAL_ACCESS"
            and old.get("universe_id") == record.get("universe_id")
            and old.get("security_id") == record.get("security_id")
            and old.get("symbol_used") == record.get("symbol_used")
            and old.get("requested_start") == record.get("requested_start")
            and old.get("requested_end") == record.get("requested_end")
        )]
    key_fields = ("universe_id", "security_id", "instrument_token", "mapping_source_id",
                  "symbol_used", "requested_start", "requested_end")
    key = tuple(record.get(field, "") for field in key_fields)
    replaced = False
    for index, old in enumerate(rows):
        if tuple(old.get(field, "") for field in key_fields) == key:
            rows[index] = record
            replaced = True
            break
    if not replaced:
        rows.append(record)
    _write_manifest(manifest_path, rows)


def _write_access_manifest(path: str | Path, rows: list[dict[str, str]]) -> None:
    _atomic_csv(Path(path), ACCESS_FIELDS, rows)


def _probe_intervals(
    *, provider: UniverseProvider, security_id: str, request_start: date, request_end: date,
    evaluation_start: date, members: list[Any], symbol_row: Any,
    snapshot_date: date, current_symbol: str, current_token: int | None,
    current_lookup_source: str, identity_status: str,
    session_days: set[date],
) -> list[dict[str, Any]]:
    """Split history into evidenced token ranges and probe-only gaps."""
    symbol_start = max(request_start, symbol_row.effective_from)
    symbol_end = min(request_end, symbol_row.effective_to - timedelta(days=1)
                     if symbol_row.effective_to else request_end)
    if symbol_start > symbol_end:
        return []
    evidence = provider.broker_instruments_over(security_id, symbol_start, symbol_end)
    segments: list[tuple[date, date, Any | None]] = []
    cursor = symbol_start
    for mapping in evidence:
        left = max(symbol_start, mapping.effective_from)
        right = min(symbol_end, mapping.effective_to - timedelta(days=1)
                    if mapping.effective_to else symbol_end)
        if left > right:
            continue
        if cursor < left:
            segments.append((cursor, left - timedelta(days=1), None))
        segments.append((left, right, mapping))
        cursor = max(cursor, right + timedelta(days=1))
    if cursor <= symbol_end:
        segments.append((cursor, symbol_end, None))

    output: list[dict[str, Any]] = []
    for segment_start, segment_end, mapping in segments:
        member_ranges = []
        for member in members:
            if member.security_id != security_id:
                continue
            member_start = max(evaluation_start, member.effective_from)
            member_end = min(request_end, member.effective_to - timedelta(days=1)
                             if member.effective_to else request_end)
            left, right = max(segment_start, member_start), min(segment_end, member_end)
            if left <= right:
                member_ranges.append((left, right))
        preferred = sorted(day for day in session_days if any(left <= day <= right for left, right in member_ranges))
        candidates = preferred or sorted(day for day in session_days if segment_start <= day <= segment_end)
        probe_day = candidates[0] if candidates else None
        token = mapping.instrument_token if mapping is not None else current_token
        provenance = "EVIDENCED_HISTORICAL_MAPPING" if mapping is not None else "CURRENT_TOKEN_HISTORICAL_PROBE"
        renamed_probe = mapping is None and current_symbol != symbol_row.symbol
        failure_detail = ""
        status = ""
        output_identity_status = (
            "SAME_SYMBOL_STABLE_IDENTITY" if current_symbol == symbol_row.symbol else identity_status
        )
        if token is None:
            status = "NO_CURRENT_TOKEN"
            failure_detail = "No date-valid historical mapping or current NSE equity token was resolved."
        elif probe_day is None:
            status = "FAIL_HISTORICAL_DATA"
            failure_detail = "No validated local NSE session date is available for the probe range."
        elif mapping is None:
            identity_events = [event for event in provider.identity_events_over(
                security_id, probe_day + timedelta(days=1), snapshot_date
            ) if event.event_type == "SYMBOL_CHANGED"]
            verified_events = [event for event in identity_events if _verified_rename_continuity(
                provider, security_id, symbol_row.symbol, current_symbol, event
            )]
            if any(not _verified_rename_continuity(
                provider, security_id, symbol_row.symbol, current_symbol, event
            ) for event in identity_events):
                status = "FAIL_CLOSED_IDENTITY_BOUNDARY"
                output_identity_status = "UNRESOLVED_IDENTITY_BOUNDARY"
                failure_detail = "Current instrument identity crosses an explicitly unsafe corporate-action boundary."
            elif renamed_probe and not verified_events:
                status = "FAIL_CLOSED_IDENTITY_BOUNDARY"
                output_identity_status = "UNRESOLVED_IDENTITY_BOUNDARY"
                failure_detail = "Changed current symbol has no explicit stable-identity continuity evidence."
            elif renamed_probe:
                output_identity_status = "VERIFIED_RENAME_CONTINUITY"
                provenance = "CURRENT_RENAMED_TOKEN_HISTORICAL_PROBE"
        output.append({
            "security_id": security_id, "historical_symbol": symbol_row.symbol,
            "symbol_effective_from": symbol_row.effective_from,
            "symbol_effective_to": symbol_row.effective_to,
            "requested_start": segment_start, "requested_end": segment_end,
            "current_symbol": current_symbol, "current_resolved_token": current_token,
            "current_instrument_lookup_source": current_lookup_source,
            "identity_status": output_identity_status,
            "instrument_token": token, "broker_mapping_source_id": mapping.source_id if mapping else "",
            "mapping_effective_from": mapping.effective_from if mapping else None,
            "mapping_effective_to": mapping.effective_to if mapping else None,
            "market_data_access_provenance": provenance, "probe_date": probe_day,
            "renamed_probe": renamed_probe,
            "probe_result": status, "failure_detail": failure_detail,
        })
    return output


def _verified_rename_continuity(
    provider: UniverseProvider, security_id: str, historical_symbol: str,
    current_symbol: str, event: Any,
) -> bool:
    """Allow only named, source-backed same-ID rename paths with identity evidence."""
    if event.from_symbol != historical_symbol or event.to_symbol != current_symbol:
        return False
    if event.source_id in VERIFIED_CONTINUITY_RENAME_SOURCES:
        return True
    # Tata's 2025 passenger-vehicle demerger boundary is allowed for data access
    # only when the source's own security-ID history and ISIN document continuity.
    if (security_id, event.source_id, historical_symbol, current_symbol) != (
        "NSEEQ-0055", "tata_symbol_change", "TATAMOTORS", "TMPV"
    ):
        return False
    try:
        identity_isin, identity_source = provider.security_identity_for(security_id)
        symbols = provider.symbol_history_over(security_id, date.min, date.max)
    except (AttributeError, UniverseCoverageError):
        return False
    return bool(
        identity_isin == "INE155A01022" and identity_source == "tata_pv_isin"
        and any(item.symbol == historical_symbol and item.effective_to == event.effective_date for item in symbols)
        and any(item.symbol == current_symbol and item.effective_from == event.effective_date for item in symbols)
    )


def probe_hourly_access(
    *, start_date: date, end_date: date, warmup_start: date = date(2022, 10, 1),
    universe_id: str = PIT_TRADABLE, provider: UniverseProvider | None = None,
    snapshot_directory: str | Path = PROJECT_ROOT / "data" / "raw" / "instruments",
    data_root: str | Path = DATA_ROOT, daily_manifest_path: str | Path = MANIFEST_PATH,
    output_path: str | Path = HOURLY_ACCESS_MANIFEST, client: Any | None = None,
    client_factory: Callable[[], Any] = create_authenticated_client,
    throttle: RequestThrottle | None = None, sleep: Callable[[float], None] = time.sleep,
) -> list[dict[str, str]]:
    """Probe historical 60-minute access without changing broker reference data."""
    if warmup_start > start_date or start_date > end_date:
        raise HourlyDataError("Probe dates must satisfy warmup_start <= start <= end.")
    provider = provider or UniverseProvider()
    members = [member for member in provider.members_over(start_date, end_date, universe_id) if member.tradable]
    first_member: dict[str, date] = {}
    for member in members:
        first_member[member.security_id] = min(first_member.get(member.security_id, member.effective_from), member.effective_from)
    snapshot_date, instruments = latest_instrument_snapshot(snapshot_directory)
    snapshot_source = f"FULL_NSE_EQUITY_SNAPSHOT:{snapshot_date.isoformat()}"
    session_days = _daily_sessions(warmup_start, end_date, daily_manifest_path, data_root)
    plans: list[dict[str, Any]] = []
    for security_id, member_start in sorted(first_member.items()):
        try:
            latest_symbol = provider.latest_symbol_for(security_id)
            current_symbol = latest_symbol.symbol.upper()
            identity_status = "CURRENT_SYMBOL_FROM_STABLE_SECURITY_HISTORY"
        except UniverseCoverageError:
            current_symbol = ""
            identity_status = "UNRESOLVED_STABLE_SECURITY_SYMBOL"
        current_record = instruments.get(current_symbol) if current_symbol else None
        try:
            current_token = int(current_record["instrument_token"]) if current_record else None
            if current_token is not None and current_token <= 0:
                current_token = None
        except (KeyError, TypeError, ValueError):
            current_token = None
        acquisition_start = max(warmup_start, member_start - timedelta(days=180))
        if security_id in NEW_LISTING_SECURITY_IDS:
            acquisition_start = member_start
        symbol_intervals = provider.symbol_history_over(security_id, acquisition_start, end_date)
        if not symbol_intervals:
            plans.append({
                "security_id": security_id, "historical_symbol": "",
                "symbol_effective_from": acquisition_start, "symbol_effective_to": None,
                "requested_start": acquisition_start, "requested_end": end_date,
                "current_symbol": current_symbol, "current_resolved_token": current_token,
                "current_instrument_lookup_source": snapshot_source,
                "identity_status": identity_status,
                "instrument_token": current_token, "broker_mapping_source_id": "",
                "mapping_effective_from": None, "mapping_effective_to": None,
                "market_data_access_provenance": "CURRENT_TOKEN_HISTORICAL_PROBE",
                "probe_date": None, "probe_result": "FAIL_CLOSED_IDENTITY_BOUNDARY",
                "failure_detail": "No source-backed historical symbol identity covers the requested range.",
            })
            continue
        for symbol_row in symbol_intervals:
            plans.extend(_probe_intervals(
                provider=provider, security_id=security_id, request_start=acquisition_start,
                request_end=end_date, evaluation_start=start_date, members=members,
                symbol_row=symbol_row, snapshot_date=snapshot_date,
                current_symbol=current_symbol, current_token=current_token,
                current_lookup_source=snapshot_source, identity_status=identity_status,
                session_days=session_days,
            ))

    pending = [plan for plan in plans if not plan["probe_result"]]
    if pending:
        try:
            client = client or client_factory()
        except Exception:
            for plan in pending:
                plan["probe_result"] = "FAIL_HISTORICAL_DATA"
                plan["failure_detail"] = "Kite session could not be established for the read-only probe."
    throttle = throttle or RequestThrottle(sleep=sleep)
    for plan in pending:
        if plan["probe_result"]:
            continue
        throttle.wait()
        try:
            response = client.historical_data(
                int(plan["instrument_token"]),
                datetime.combine(plan["probe_date"], day_time.min),
                datetime.combine(plan["probe_date"], day_time.max), INTERVAL,
            )
            valid = normalize_hourly(response, session_dates={plan["probe_date"]})
            if not valid:
                plan["probe_result"] = "FAIL_HISTORICAL_DATA"
                plan["failure_detail"] = "Kite returned no valid 60-minute candles for the probe date."
            else:
                plan["probe_result"] = (
                    "PASS_RENAMED_TOKEN_HISTORICAL_DATA" if plan.get("renamed_probe")
                    else "PASS_HISTORICAL_DATA"
                )
                plan["probe_actual_first_timestamp"] = valid[0]["date"]
                plan["probe_actual_last_timestamp"] = valid[-1]["date"]
                plan["probe_row_count"] = len(valid)
        except Exception:
            plan["probe_result"] = "FAIL_HISTORICAL_DATA"
            plan["failure_detail"] = "Kite rejected the probe or returned invalid hourly candles."

    results = []
    for plan in plans:
        results.append({
            "universe_id": universe_id, "security_id": plan["security_id"],
            "historical_symbol": plan["historical_symbol"],
            "symbol_effective_from": plan["symbol_effective_from"].isoformat(),
            "symbol_effective_to": plan["symbol_effective_to"].isoformat() if plan["symbol_effective_to"] else "",
            "evaluation_start": start_date.isoformat(), "evaluation_end": end_date.isoformat(),
            "warmup_start": warmup_start.isoformat(),
            "requested_start": plan["requested_start"].isoformat(), "requested_end": plan["requested_end"].isoformat(),
            "snapshot_date": snapshot_date.isoformat(), "current_symbol": plan["current_symbol"],
            "current_resolved_token": str(plan["current_resolved_token"] or ""),
            "current_instrument_lookup_source": plan.get("current_instrument_lookup_source", ""),
            "identity_status": plan.get("identity_status", ""),
            "instrument_token": str(plan["instrument_token"] or ""),
            "broker_mapping_source_id": plan["broker_mapping_source_id"],
            "mapping_effective_from": plan["mapping_effective_from"].isoformat() if plan["mapping_effective_from"] else "",
            "mapping_effective_to": plan["mapping_effective_to"].isoformat() if plan["mapping_effective_to"] else "",
            "probe_date": plan["probe_date"].isoformat() if plan["probe_date"] else "",
            "probe_actual_first_timestamp": plan.get("probe_actual_first_timestamp", ""),
            "probe_actual_last_timestamp": plan.get("probe_actual_last_timestamp", ""),
            "probe_row_count": str(plan.get("probe_row_count", 0)),
            "probe_result": plan["probe_result"], "market_data_access_result": plan["probe_result"],
            "failure_detail": plan["failure_detail"],
            "market_data_access_provenance": plan["market_data_access_provenance"],
        })
    _write_access_manifest(output_path, results)
    return sorted(results, key=lambda row: (row["security_id"], row["requested_start"], row["historical_symbol"]))


def build_hourly_dataset(
    *, start_date: date, end_date: date, warmup_start: date = date(2022, 10, 1),
    universe_id: str = PIT_TRADABLE, provider: UniverseProvider | None = None,
    client: Any | None = None, client_factory: Callable[[], Any] = create_authenticated_client,
    data_root: str | Path = DATA_ROOT, manifest_path: str | Path = HOURLY_MANIFEST,
    daily_manifest_path: str | Path = MANIFEST_PATH,
    access_manifest_path: str | Path = HOURLY_ACCESS_MANIFEST,
    snapshot_directory: str | Path = PROJECT_ROOT / "data" / "raw" / "instruments",
    sleep: Callable[[float], None] = time.sleep,
    throttle: RequestThrottle | None = None,
    force: bool = False,
    quarantine_path: str | Path = HOURLY_QUARANTINE_PATH,
) -> dict[str, Any]:
    """Acquire ranges authorized by evidence-backed mapping or a successful hourly probe."""
    if warmup_start > start_date or start_date > end_date:
        raise HourlyDataError("Dates must satisfy warmup_start <= start <= end.")
    provider = provider or UniverseProvider()
    memberships = [m for m in provider.members_over(start_date, end_date, universe_id) if m.tradable]
    expected_ids = {m.security_id for m in memberships}
    if not expected_ids:
        raise HourlyDataError("No tradable PIT securities were found in the requested period.")
    access_rows = _manifest_rows(access_manifest_path)
    if not access_rows:
        raise HourlyDataError(
            f"Hourly access probes are missing; run probe first to create {Path(access_manifest_path).name}."
        )
    snapshot_date, current_instruments = latest_instrument_snapshot(snapshot_directory)
    matching = [
        row for row in access_rows
        if row.get("universe_id") == universe_id
        and row.get("evaluation_start") == start_date.isoformat()
        and row.get("evaluation_end") == end_date.isoformat()
        and row.get("warmup_start") == warmup_start.isoformat()
    ]
    if not matching:
        raise HourlyDataError("No hourly access probes match this universe and date range; run probe first.")
    stale = [row for row in matching if row.get("snapshot_date") != snapshot_date.isoformat()]
    if stale:
        raise HourlyDataError("The hourly access probe uses a stale Kite instrument snapshot; rerun probe.")
    for row in matching:
        if (row.get("probe_result") not in {"PASS_HISTORICAL_DATA", "PASS_RENAMED_TOKEN_HISTORICAL_DATA"}
                or row.get("market_data_access_provenance") not in {
                    "CURRENT_TOKEN_HISTORICAL_PROBE", "CURRENT_RENAMED_TOKEN_HISTORICAL_PROBE"
                }):
            continue
        current_symbol = row.get("current_symbol", "")
        current_record = current_instruments.get(current_symbol)
        try:
            snapshot_token = int(current_record["instrument_token"]) if current_record else None
        except (KeyError, TypeError, ValueError):
            snapshot_token = None
        if (not snapshot_token or row.get("current_resolved_token") != str(snapshot_token)
                or row.get("instrument_token") != str(snapshot_token)):
            raise HourlyDataError("A probed current Kite token no longer matches the saved instrument snapshot; rerun probe.")
    pass_results = {"PASS_HISTORICAL_DATA", "PASS_RENAMED_TOKEN_HISTORICAL_DATA"}
    accessible = [row for row in matching if row.get("probe_result") in pass_results]
    failed_rows = [row for row in matching if row.get("probe_result") not in pass_results]
    records = _manifest_rows(manifest_path)
    for row in failed_rows:
        _record(manifest_path=manifest_path, record={
            "universe_id": universe_id, "security_id": row["security_id"], "source": SOURCE,
            "broker": "kite", "instrument_token": row.get("instrument_token", ""),
            "broker_mapping_provenance": (
                f"HISTORICALLY_EVIDENCED:{row['broker_mapping_source_id']}"
                if row.get("broker_mapping_source_id") else "NOT_HISTORICALLY_EVIDENCED"
            ),
            "market_data_access_provenance": row.get("market_data_access_provenance", ""),
            "mapping_source_id": row.get("broker_mapping_source_id", ""),
            "mapping_effective_from": row.get("mapping_effective_from", ""),
            "mapping_effective_to": row.get("mapping_effective_to", ""),
            "symbol_used": row.get("historical_symbol", ""), "interval": INTERVAL,
            "requested_start": row.get("requested_start", ""), "requested_end": row.get("requested_end", ""),
            "actual_first_timestamp": "", "actual_last_timestamp": "", "row_count": "0",
            "missing_session_diagnostics": "", "downloaded_at": datetime.now(timezone.utc).isoformat(),
            "content_sha256": "", "provenance": row.get("failure_detail", ""),
            "validation_status": "NO_KITE_HISTORICAL_ACCESS", "relative_path": "",
        })
    missing_probe_ids = sorted(expected_ids - {row.get("security_id", "") for row in matching})
    if not accessible:
        return {
            "expected_securities": len(expected_ids), "mapped_ranges": 0,
            "resolved_access_ranges": 0, "evidenced_broker_mapping_ranges": 0,
            "current_token_probe_ranges": 0, "downloaded_rows": 0, "downloads": [],
            "access_failures": [
                f"{row['security_id']}:{row['historical_symbol']}:{row['probe_result']}"
                for row in failed_rows
            ] + [f"{security_id}:FAIL_CLOSED_IDENTITY_BOUNDARY:NO_PROBE_RESULT" for security_id in missing_probe_ids],
        }
    try:
        client = client or client_factory()
    except Exception:
        raise HourlyDataError("Kite session could not be established for hourly download.") from None
    throttle = throttle or RequestThrottle(sleep=sleep)
    downloads: list[str] = []
    downloaded_rows = 0
    reused_securities: list[str] = []
    total_invalid = total_quarantined = total_recovered = 0
    access_failures = [
        f"{row['security_id']}:{row['historical_symbol']}:{row['probe_result']}"
        for row in failed_rows
    ] + [f"{security_id}:FAIL_CLOSED_IDENTITY_BOUNDARY:NO_PROBE_RESULT" for security_id in missing_probe_ids]
    for access in accessible:
        token = int(access["instrument_token"])
        mapped_start = date.fromisoformat(access["requested_start"])
        mapped_end = date.fromisoformat(access["requested_end"])
        symbol = access["historical_symbol"]
        security_id = access["security_id"]
        mapping_source_id = access.get("broker_mapping_source_id", "")
        mapping_provenance = (
            f"HISTORICALLY_EVIDENCED:{mapping_source_id}"
            if mapping_source_id else "NOT_HISTORICALLY_EVIDENCED"
        )
        access_provenance = access["market_data_access_provenance"]
        path = Path(data_root) / universe_id / INTERVAL / security_id / (
            f"{symbol}_{token}.csv"
        )
        complete_record = next((record for record in records if
            record.get("universe_id") == universe_id
            and record.get("security_id") == security_id
            and record.get("instrument_token") == str(token)
            and record.get("symbol_used") == symbol
            and record.get("requested_start") == mapped_start.isoformat()
            and record.get("requested_end") == mapped_end.isoformat()
            and record.get("validation_status") == "VALIDATED"
        ), None)
        content_is_complete = (
            complete_record is not None and path.is_file()
            and hashlib.sha256(path.read_bytes()).hexdigest() == complete_record.get("content_sha256")
        )
        if not force and content_is_complete:
            reused_securities.append(security_id)
            downloads.append(f"{security_id}:{symbol}:reused existing completed dataset")
            continue
        session_dates = _daily_sessions(mapped_start, mapped_end, daily_manifest_path, data_root)
        if not session_dates:
            raise HourlyDataError("No validated daily session calendar is available; hourly session dates cannot be verified.")
        existing = _read_existing(path, session_dates=session_dates)
        # Re-fetch only sessions with missing full bars plus the last observed session.
        chunks: list[dict[str, Any]] = []
        quality_counts = {"invalid": 0, "quarantined": 0, "recovered": 0}
        quarantine_events: list[dict[str, Any]] = []
        returned_candles = 0
        last_error: Exception | None = None
        windows = _request_windows(
            existing=existing, start=mapped_start, end=mapped_end,
            daily_manifest=daily_manifest_path, data_root=data_root,
        )
        request_chunks = [chunk for window_start, window_end in windows
                          for chunk in hourly_chunks(window_start, window_end)]
        for chunk_start, chunk_end in request_chunks:
            response: list[dict[str, Any]] | None = None
            for attempt in range(4):
                throttle.wait()
                try:
                    response = client.historical_data(
                        token, chunk_start, chunk_end, INTERVAL
                    )
                    if not isinstance(response, list):
                        raise HourlyDataError(
                            f"Kite returned an invalid hourly response for {security_id} ({symbol}), "
                            f"token {token}, chunk {chunk_start.isoformat()}..{chunk_end.isoformat()}."
                        )
                    last_error = None
                    break
                except HourlyDataError:
                    raise
                except Exception as error:
                    last_error = error
                    if attempt < 3:
                        sleep(2 ** attempt)
            if last_error is not None:
                if isinstance(last_error, KiteException):
                    raise HourlyDataError(f"Kite rejected hourly history for {security_id} ({symbol}).") from None
                raise HourlyDataError(f"Could not retrieve hourly history for {security_id} ({symbol}).") from None
            chunk_sessions = {day for day in session_dates if chunk_start.date() <= day <= chunk_end.date()}
            returned_candles += len(response or [])

            def handle_invalid_candle(
                bad_row: dict[str, Any], candle_index: int, failed_invariant: str,
            ) -> tuple[dict[str, Any] | None, str]:
                del candle_index
                quality_counts["invalid"] += 1
                bad_stamp = _aware(bad_row.get("date"))
                retry_response: list[dict[str, Any]] = []
                matching_rows: list[dict[str, Any]] = []
                retry_result = "REFETCH_FAILED"
                diagnostic_start = datetime.combine(bad_stamp.date(), day_time.min)
                diagnostic_end = datetime.combine(bad_stamp.date(), day_time.max)
                try:
                    throttle.wait()
                    raw_retry = client.historical_data(token, diagnostic_start, diagnostic_end, INTERVAL)
                    if isinstance(raw_retry, list):
                        retry_response = raw_retry
                        for candidate in raw_retry:
                            if not isinstance(candidate, dict):
                                continue
                            try:
                                if _aware(candidate.get("date")) == bad_stamp:
                                    matching_rows.append(candidate)
                            except HourlyDataError:
                                continue
                        if len(matching_rows) == 1:
                            try:
                                recovered = normalize_hourly(
                                    matching_rows, session_dates=chunk_sessions,
                                    diagnostic_context={
                                        "security_id": security_id, "symbol": symbol,
                                        "instrument_token": token,
                                        "chunk_start": diagnostic_start.isoformat(),
                                        "chunk_end": diagnostic_end.isoformat(),
                                    },
                                )[0]
                                recovered["data_quality_status"] = "RECOVERED_BY_REFETCH"
                                quality_counts["recovered"] += 1
                                return recovered, "RECOVERED_BY_REFETCH"
                            except HourlyDataError:
                                retry_result = "PERSISTENT_INVALID_CANDLE"
                        elif matching_rows:
                            retry_result = "REFETCH_AMBIGUOUS_TIMESTAMP"
                        else:
                            retry_result = "REFETCH_DID_NOT_RETURN_TIMESTAMP"
                    else:
                        retry_result = "REFETCH_INVALID_RESPONSE"
                except Exception:
                    retry_result = "REFETCH_FAILED"

                quality_counts["quarantined"] += 1
                event = {
                    "security_id": security_id, "symbol": symbol, "token": str(token),
                    "timestamp": bad_stamp.isoformat(),
                    "raw_ohlcv": _json_provider_value({key: bad_row.get(key) for key in
                                                       ("date", "open", "high", "low", "close", "volume")}),
                    "failed_invariant": failed_invariant, "retry_result": retry_result,
                    "source": "KITE", "retry_count": "1",
                    "data_quality_status": "PROVIDER_INVALID_OHLC",
                    "data_quality_status": "PROVIDER_INVALID_OHLC",
                    "refetched_raw_response": _json_provider_value(retry_response),
                    "quarantined_at": datetime.now(timezone.utc).isoformat(),
                }
                _record_quarantine(quarantine_path, event)
                quarantine_events.append(event)
                return None, "PROVIDER_INVALID_OHLC"

            chunks.extend(normalize_hourly(
                response or [], session_dates=chunk_sessions,
                diagnostic_context={
                    "security_id": security_id, "symbol": symbol, "instrument_token": token,
                    "chunk_start": chunk_start.isoformat(), "chunk_end": chunk_end.isoformat(),
                },
                on_invalid_ohlcv=handle_invalid_candle,
            ))
        total_invalid += quality_counts["invalid"]
        total_quarantined += quality_counts["quarantined"]
        total_recovered += quality_counts["recovered"]
        corruption_rate = quality_counts["invalid"] / returned_candles if returned_candles else 0.0
        if quality_counts["invalid"] > 3 or corruption_rate > 0.001:
            failure = (
                f"{security_id}:{symbol}:FAILED_CORRUPTION_THRESHOLD:"
                f"{quality_counts['invalid']} malformed/{returned_candles} returned candles "
                f"({corruption_rate:.6%})"
            )
            access_failures.append(failure)
            if complete_record is None:
                _record(manifest_path=manifest_path, record={
                    "universe_id": universe_id, "security_id": security_id,
                    "source": SOURCE, "broker": "kite", "instrument_token": str(token),
                    "symbol_used": symbol, "interval": INTERVAL,
                    "requested_start": mapped_start.isoformat(), "requested_end": mapped_end.isoformat(),
                    "invalid_provider_candles": str(quality_counts["invalid"]),
                    "quarantined_candles": str(quality_counts["quarantined"]),
                    "recovered_by_refetch": str(quality_counts["recovered"]),
                    "missing_hourly_bars_due_to_quarantine": str(quality_counts["quarantined"]),
                    "provenance": failure, "validation_status": "FAILED_CORRUPTION_THRESHOLD",
                    "downloaded_at": datetime.now(timezone.utc).isoformat(),
                })
            continue
        incoming = chunks
        incoming = [row for row in incoming if mapped_start <= date.fromisoformat(row["date"][:10]) <= mapped_end]
        merged = _merge(existing, incoming)
        if not merged:
            access_failures.append(f"{security_id}:{symbol}:FAIL_HISTORICAL_DATA:no candles in requested range")
            continue
        _atomic_csv(path, FIELDS, merged)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        relative = path.relative_to(Path(data_root)).as_posix()
        relevant_sessions = _daily_sessions(mapped_start, mapped_end, daily_manifest_path, data_root)
        merged_timestamps = {row["date"] for row in merged}
        missing_due_to_quarantine = sum(
            datetime.fromisoformat(event["timestamp"]).isoformat() not in merged_timestamps
            for event in quarantine_events
        )
        first_stamp = datetime.fromisoformat(merged[0]["date"])
        last_stamp = datetime.fromisoformat(merged[-1]["date"])
        record = {
            "universe_id": universe_id, "security_id": security_id, "source": SOURCE,
            "broker": "kite", "instrument_token": str(token),
            "broker_mapping_provenance": mapping_provenance,
            "market_data_access_provenance": access_provenance,
            "mapping_source_id": mapping_source_id,
            "mapping_effective_from": access.get("mapping_effective_from", ""),
            "mapping_effective_to": access.get("mapping_effective_to", ""),
            "symbol_used": symbol, "interval": INTERVAL,
            "requested_start": mapped_start.isoformat(), "requested_end": mapped_end.isoformat(),
            "actual_first_timestamp": first_stamp.isoformat(), "actual_last_timestamp": last_stamp.isoformat(),
            "row_count": str(len(merged)),
            "missing_session_diagnostics": _missing_session_diagnostics(merged, relevant_sessions),
            "downloaded_at": datetime.now(timezone.utc).isoformat(), "content_sha256": digest,
            "provenance": (
                f"broker_mapping={mapping_provenance}; market_data_access={access_provenance}; "
                f"historical symbol range {access['symbol_effective_from']} to "
                f"{access.get('symbol_effective_to') or 'open'}; Kite interval={INTERVAL}"
            ),
            "invalid_provider_candles": str(quality_counts["invalid"]),
            "quarantined_candles": str(quality_counts["quarantined"]),
            "recovered_by_refetch": str(quality_counts["recovered"]),
            "missing_hourly_bars_due_to_quarantine": str(missing_due_to_quarantine),
            "validation_status": "VALIDATED", "relative_path": relative,
        }
        _record(manifest_path=manifest_path, record=record)
        downloaded_rows += len(incoming)
        downloads.append(f"{security_id}:{symbol}:{len(incoming)} fetched/{len(merged)} total")
    return {"expected_securities": len(expected_ids), "mapped_ranges": len(accessible),
            "resolved_access_ranges": len(accessible),
            "evidenced_broker_mapping_ranges": sum(bool(row.get("broker_mapping_source_id")) for row in accessible),
            "current_token_probe_ranges": sum(row.get("market_data_access_provenance") in {
                "CURRENT_TOKEN_HISTORICAL_PROBE", "CURRENT_RENAMED_TOKEN_HISTORICAL_PROBE"
            } for row in accessible),
            "downloaded_rows": downloaded_rows,
            "invalid_provider_candles": total_invalid,
            "quarantined_candles": total_quarantined,
            "recovered_by_refetch": total_recovered,
            "reused_securities": reused_securities,
            "downloads": downloads, "access_failures": access_failures}


def audit_hourly_dataset(
    *, start_date: date, end_date: date, universe_id: str = PIT_TRADABLE,
    provider: UniverseProvider | None = None, data_root: str | Path = DATA_ROOT,
    manifest_path: str | Path = HOURLY_MANIFEST,
    daily_manifest_path: str | Path = MANIFEST_PATH,
    quarantine_path: str | Path = HOURLY_QUARANTINE_PATH,
) -> dict[str, Any]:
    provider = provider or UniverseProvider()
    members = [m for m in provider.members_over(start_date, end_date, universe_id) if m.tradable]
    expected = {m.security_id for m in members}
    manifest_target = Path(manifest_path)
    manifest_exists = manifest_target.is_file()
    rows = []
    for row in _manifest_rows(manifest_target):
        if row.get("universe_id") != universe_id or row.get("interval") != INTERVAL:
            continue
        try:
            row_start = date.fromisoformat(row.get("requested_start", ""))
            row_end = date.fromisoformat(row.get("requested_end", ""))
        except ValueError:
            # Keep malformed records visible in the audit as incomplete evidence.
            rows.append(row)
            continue
        if row_start <= end_date and start_date <= row_end:
            rows.append(row)
    valid_sessions = _daily_sessions(start_date, end_date, daily_manifest_path, data_root)
    quarantined_by_security: dict[str, list[dict[str, str]]] = defaultdict(list)
    for item in _manifest_rows(quarantine_path):
        try:
            stamp = _aware(item.get("timestamp"))
        except HourlyDataError:
            continue
        if start_date <= stamp.date() <= end_date:
            quarantined_by_security[item.get("security_id", "")].append(item)
    by_security: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_security[row.get("security_id", "")].append(row)
    reports: dict[str, Any] = {}
    present: set[str] = set()
    audit_totals = {
        "normal_session_rows": 0, "special_regular_rows": 0, "special_split_rows": 0,
        "muhurat_rows": 0, "native_special_session_rows": 0,
        "unexpected_normal_grid_rows": 0, "invalid_provider_candles": 0,
        "quarantined_candles": 0, "recovered_by_refetch": 0,
        "missing_hourly_bars_due_to_quarantine": 0,
    }
    for security_id, records in sorted(by_security.items()):
        all_stamps: list[datetime] = []
        duplicate_count = non_monotonic = outside = ohlc_errors = 0
        annual: dict[str, int] = defaultdict(int)
        suspicious_gaps: list[str] = []
        previous: datetime | None = None
        seen: set[datetime] = set()
        row_total = 0
        row_session_counts = {key: 0 for key in audit_totals}
        recovered_count = 0
        first_values: list[datetime] = []
        last_values: list[datetime] = []
        per_file_issues = []
        access_failures = [record.get("provenance", "") for record in records
                           if record.get("validation_status") == "NO_KITE_HISTORICAL_ACCESS"]
        for record in records:
            if record.get("validation_status") != "VALIDATED":
                continue
            path = Path(data_root) / record.get("relative_path", "")
            try:
                with path.open("r", encoding="utf-8-sig", newline="") as handle:
                    file_rows = list(csv.DictReader(handle))
            except OSError:
                per_file_issues.append(f"missing file: {record.get('relative_path', '')}")
                continue
            stamps_in_file: list[datetime] = []
            for row in file_rows:
                try:
                    stamp = _aware(row.get("date"))
                    prices = {key: float(row[key]) for key in ("open", "high", "low", "close")}
                    int(row["volume"])
                except (KeyError, ValueError, TypeError, HourlyDataError):
                    ohlc_errors += 1
                    continue
                if not start_date <= stamp.date() <= end_date:
                    continue
                row_total += 1
                stamps_in_file.append(stamp)
                if stamp in seen:
                    duplicate_count += 1
                if previous is not None:
                    if stamp < previous:
                        non_monotonic += 1
                    elif (stamp - previous > timedelta(hours=1) and stamp.date() == previous.date()
                          and not _is_expected_special_session_gap(previous, stamp)):
                        suspicious_gaps.append(f"{previous.isoformat()}->{stamp.isoformat()}")
                previous = stamp
                seen.add(stamp)
                all_stamps.append(stamp)
                annual[str(stamp.year)] += 1
                if row.get("data_quality_status") == "RECOVERED_BY_REFETCH":
                    recovered_count += 1
                session_kind = session_type_on(stamp.date(), valid_sessions)
                counter = {
                    "NORMAL_SESSION": "normal_session_rows",
                    "SPECIAL_REGULAR_SESSION": "special_regular_rows",
                    "SPECIAL_SPLIT_SESSION": "special_split_rows",
                    "MUHURAT": "muhurat_rows",
                }.get(session_kind)
                if counter:
                    row_session_counts[counter] += 1
                if session_kind in {"SPECIAL_REGULAR_SESSION", "SPECIAL_SPLIT_SESSION", "MUHURAT"}:
                    row_session_counts["native_special_session_rows"] += 1
                valid_stamp = valid_session_timestamp(row.get("date"), valid_sessions)
                if session_kind == "NORMAL_SESSION" and not valid_stamp:
                    row_session_counts["unexpected_normal_grid_rows"] += 1
                if not valid_stamp:
                    outside += 1
                if prices["high"] < max(prices.values()) or prices["low"] > min(prices.values()):
                    ohlc_errors += 1
            if stamps_in_file:
                first_values.append(min(stamps_in_file)); last_values.append(max(stamps_in_file))
        if all_stamps:
            present.add(security_id)
        quarantine_rows = quarantined_by_security.get(security_id, [])
        present_timestamps = {stamp.isoformat() for stamp in all_stamps}
        missing_quarantined = sum(item.get("timestamp", "") not in present_timestamps
                                  for item in quarantine_rows)
        row_session_counts.update({
            "quarantined_candles": len(quarantine_rows),
            "recovered_by_refetch": recovered_count,
            "invalid_provider_candles": len(quarantine_rows) + recovered_count,
            "missing_hourly_bars_due_to_quarantine": missing_quarantined,
        })
        reports[security_id] = {
            "data_status": "DATA_AVAILABLE" if row_total else "MISSING_HOURLY_DATA",
            "row_count": row_total,
            "first_timestamp": min(all_stamps).isoformat() if all_stamps else None,
            "last_timestamp": max(all_stamps).isoformat() if all_stamps else None,
            "duplicate_timestamps": duplicate_count, "non_monotonic_timestamps": non_monotonic,
            "ohlc_violations": ohlc_errors, "outside_session_timestamps": outside,
            **row_session_counts,
            "suspicious_large_gaps": suspicious_gaps,
            "missing_full_bar_sessions": (
                _missing_session_diagnostics([{"date": stamp.isoformat()} for stamp in all_stamps], valid_sessions)
                if all_stamps and valid_sessions else
                "no validated hourly file" if not all_stamps else "daily session calendar unavailable"
            ),
            "rows_by_year": dict(sorted(annual.items())), "file_issues": per_file_issues,
            "access_failures": access_failures,
        }
    represented = {row.get("security_id", "") for row in rows}
    dataset_not_built_ids: list[str] = []
    incomplete_ids: list[str] = []
    genuinely_missing_ids: list[str] = []
    for security_id in sorted(expected):
        if security_id in reports:
            report = reports[security_id]
            record_rows = by_security.get(security_id, [])
            statuses = {row.get("validation_status", "") for row in record_rows}
            if report["row_count"]:
                report["data_status"] = "DATA_AVAILABLE"
            elif not rows:
                report["data_status"] = "DATASET_NOT_BUILT"
                dataset_not_built_ids.append(security_id)
            elif statuses and statuses.issubset({"NO_KITE_HISTORICAL_ACCESS"}):
                report["data_status"] = "MISSING_HOURLY_DATA"
                genuinely_missing_ids.append(security_id)
            else:
                report["data_status"] = "BUILD_INCOMPLETE"
                incomplete_ids.append(security_id)
            continue
        if not rows:
            data_status = "DATASET_NOT_BUILT"
            dataset_not_built_ids.append(security_id)
        else:
            data_status = "BUILD_INCOMPLETE"
            incomplete_ids.append(security_id)
        reports[security_id] = {
            "data_status": data_status, "row_count": 0,
            "first_timestamp": None, "last_timestamp": None,
            "duplicate_timestamps": 0, "non_monotonic_timestamps": 0,
            "ohlc_violations": 0, "outside_session_timestamps": 0,
            "suspicious_large_gaps": [], "missing_full_bar_sessions": "no validated hourly file",
            "rows_by_year": {}, "file_issues": [], "access_failures": [],
            **{key: 0 for key in audit_totals},
        }
    has_unfinished_records = any(
        row.get("validation_status") not in {"VALIDATED", "NO_KITE_HISTORICAL_ACCESS"}
        for row in rows
    )
    has_invalid_artifact = any(
        report["file_issues"] or (any(r.get("validation_status") == "VALIDATED"
                                       for r in by_security.get(security_id, [])) and not report["row_count"])
        for security_id, report in reports.items()
    )
    if not manifest_exists or not rows:
        dataset_status = "DATASET_NOT_BUILT"
    elif expected - represented or has_unfinished_records or has_invalid_artifact:
        dataset_status = "BUILD_INCOMPLETE"
    elif expected - present:
        dataset_status = "BUILT_WITH_GAPS"
    else:
        dataset_status = "BUILT"
    for report in reports.values():
        for key in audit_totals:
            audit_totals[key] += report.get(key, 0)
    return {"universe_id": universe_id, "period": [start_date.isoformat(), end_date.isoformat()],
            "pit_securities_expected": len(expected), "securities_with_hourly_data": len(expected & present),
            "missing_securities": sorted(expected - present), "dataset_status": dataset_status,
            "dataset_not_built_securities": dataset_not_built_ids,
            "build_incomplete_securities": incomplete_ids,
            "genuinely_missing_hourly_data_securities": genuinely_missing_ids,
            **audit_totals,
            "securities": reports}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("probe", "build", "audit"):
        item = sub.add_parser(name)
        item.add_argument("--start", type=date.fromisoformat, default=date(2023, 1, 1))
        item.add_argument("--end", type=date.fromisoformat, default=date(2024, 12, 31))
        item.add_argument("--warmup-start", type=date.fromisoformat, default=date(2022, 10, 1))
        item.add_argument("--universe", default=PIT_TRADABLE)
        item.add_argument("--data-root", type=Path, default=DATA_ROOT)
        item.add_argument("--manifest", type=Path, default=HOURLY_MANIFEST)
        item.add_argument("--access-manifest", type=Path, default=HOURLY_ACCESS_MANIFEST)
        item.add_argument("--daily-manifest", type=Path, default=MANIFEST_PATH)
        item.add_argument("--snapshot-directory", type=Path, default=PROJECT_ROOT / "data" / "raw" / "instruments")
        item.add_argument("--quarantine", type=Path, default=HOURLY_QUARANTINE_PATH)
        if name == "build":
            item.add_argument("--force", action="store_true", help="Re-download even completed security datasets.")
    args = parser.parse_args()
    try:
        if args.command == "probe":
            results = probe_hourly_access(
                start_date=args.start, end_date=args.end, warmup_start=args.warmup_start,
                universe_id=args.universe, data_root=args.data_root,
                daily_manifest_path=args.daily_manifest, output_path=args.access_manifest,
                snapshot_directory=args.snapshot_directory,
            )
            print("security_id | historical_symbol | current_symbol | current_token | current_instrument_lookup_source | identity_status | probe_date | market_data_access_result | market_data_access_provenance")
            for row in results:
                print(" | ".join((row["security_id"], row["historical_symbol"], row["current_symbol"],
                                  row["current_resolved_token"], row["current_instrument_lookup_source"],
                                  row["identity_status"], row["probe_date"],
                                  row["market_data_access_result"], row["market_data_access_provenance"])))
            print(f"Probe rows: {len(results)}; report: {args.access_manifest}")
        elif args.command == "build":
            result = build_hourly_dataset(start_date=args.start, end_date=args.end,
                                          warmup_start=args.warmup_start, universe_id=args.universe,
                                          data_root=args.data_root, manifest_path=args.manifest,
                                          daily_manifest_path=args.daily_manifest,
                                          access_manifest_path=args.access_manifest,
                                          snapshot_directory=args.snapshot_directory,
                                          force=args.force, quarantine_path=args.quarantine)
            print(json.dumps(result, indent=2))
        else:
            result = audit_hourly_dataset(start_date=args.start, end_date=args.end,
                                          universe_id=args.universe, data_root=args.data_root,
                                          manifest_path=args.manifest,
                                          daily_manifest_path=args.daily_manifest,
                                          quarantine_path=args.quarantine)
            print(json.dumps(result, indent=2))
    except (HourlyDataError, KiteProbeError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
