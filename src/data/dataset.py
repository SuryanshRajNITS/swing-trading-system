"""Resumable daily PIT dataset acquisition and separate coverage audits."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from datetime import date, datetime, time as day_time, timedelta, timezone
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

from kiteconnect.exceptions import KiteException

from src.broker.auth import AuthenticationError
from src.broker.session import SessionVerificationError, create_authenticated_client
from src.configuration import PROJECT_ROOT
from src.data.kite_probe import (
    DEFAULT_PROBE_MANIFEST,
    KiteProbeError,
    latest_instrument_snapshot,
    read_probe_manifest,
    probe_universe,
    write_probe_manifest,
)
from src.data.market_data import (
    DATA_ROOT,
    MANIFEST_PATH,
    MarketDataError,
    dataset_path,
    read_manifest,
    read_market_data,
    record_coverage,
    validate_ohlcv,
    write_market_data,
)
from src.data.nse_history import NseHistoryError, NSE_PROVENANCE, NSE_SOURCE, import_nse_daily
from src.data.universe.provider import PIT_TRADABLE, UniverseProvider


SUPPORTED_INTERVALS = {"day": 2000}
NSE_TZ = ZoneInfo("Asia/Kolkata")
KITE_SOURCE = "Kite Connect historical API"
LEGACY_SOURCE = "Kite legacy local historical file"
LEGACY_ROOT = PROJECT_ROOT / "data" / "raw" / "historical" / "day"
TATA_DEMERGER_EFFECTIVE = date(2025, 10, 24)
NEW_LISTING_SECURITY_IDS = {"NSEEQ-0035", "NSEEQ-0036", "NSEEQ-0039", "NSEEQ-0060"}


class DatasetError(RuntimeError):
    """Daily dataset acquisition or audit failure."""


class RequestThrottle:
    """Simple process-local Kite throttle, defaulting to at most 3 calls/sec."""

    def __init__(self, min_interval: float = 1 / 3, *, sleep: Callable[[float], None] = time.sleep) -> None:
        self.min_interval = min_interval
        self.sleep = sleep
        self.last_call: float | None = None

    def wait(self) -> None:
        now = time.monotonic()
        if self.last_call is not None:
            delay = self.min_interval - (now - self.last_call)
            if delay > 0:
                self.sleep(delay)
        self.last_call = time.monotonic()


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _security_names(provider: UniverseProvider) -> dict[str, str]:
    return {
        row["security_id"]: row["canonical_symbol"].upper()
        for row in _load_rows(provider.reference_directory / "securities.csv")
    }


def _member_windows(
    provider: UniverseProvider, universe_id: str, start: date, end: date
) -> dict[str, tuple[date, date, list[Any]]]:
    intervals = [
        row for row in provider.members_over(start, end, universe_id)
        if row.tradable
    ]
    grouped: dict[str, list[Any]] = defaultdict(list)
    for row in intervals:
        grouped[row.security_id].append(row)
    return {
        security_id: (
            min(max(start, row.effective_from) for row in rows),
            max(min(end, row.effective_to - timedelta(days=1)) if row.effective_to else end for row in rows),
            rows,
        )
        for security_id, rows in grouped.items()
    }


def _request_start(security_id: str, first_member: date, start: date, warmup_start: date) -> date:
    if security_id in NEW_LISTING_SECURITY_IDS:
        return first_member
    if security_id == "NSEEQ-0055":
        return max(warmup_start, start)
    # Six months of indicator warmup before a later universe entry, or the global warmup.
    return max(warmup_start, first_member - timedelta(days=180))


def _current_symbol(
    provider: UniverseProvider, security_id: str, snapshot_date: date, canonical: str
) -> str:
    try:
        return provider.symbol_for(security_id, snapshot_date).upper()
    except LookupError:
        return canonical


def _legacy_symbol_to_id(provider: UniverseProvider) -> dict[str, str]:
    securities = _security_names(provider)
    aliases: dict[str, str] = {symbol: security_id for security_id, symbol in securities.items()}
    for row in _load_rows(provider.reference_directory / "symbol_history.csv"):
        aliases[row["symbol"].upper()] = row["security_id"]
    return aliases


def adopt_legacy_daily_files(
    *, universe_id: str = PIT_TRADABLE, evaluation_start: date,
    evaluation_end: date, warmup_start: date, provider: UniverseProvider | None = None,
    legacy_directory: str | Path = LEGACY_ROOT, data_root: str | Path = DATA_ROOT,
    manifest_path: str | Path = MANIFEST_PATH,
) -> int:
    """Copy known Kite daily CSVs into stable security-id paths and record provenance."""
    provider = provider or UniverseProvider()
    aliases = _legacy_symbol_to_id(provider)
    copied = 0
    for legacy in Path(legacy_directory).glob("*.csv"):
        symbol = legacy.stem.upper()
        security_id = aliases.get(symbol)
        if not security_id:
            continue
        try:
            rows = read_market_data(legacy)
        except (OSError, MarketDataError, csv.Error):
            continue
        if not rows:
            continue
        destination = dataset_path(security_id, symbol, root=data_root, universe_id=universe_id)
        normalized = write_market_data(destination, rows)
        try:
            relative = destination.relative_to(Path(data_root)).as_posix()
        except ValueError:
            relative = destination.as_posix()
        acquired = datetime.fromtimestamp(legacy.stat().st_mtime, timezone.utc)
        status = "VALIDATED_EVALUATION_PARTIAL_WARMUP"
        record_coverage(
            universe_id=universe_id,
            security_id=security_id,
            source=LEGACY_SOURCE,
            symbol_used=symbol,
            interval="day",
            requested_start=warmup_start,
            requested_end=evaluation_end,
            rows=normalized,
            provenance=f"Imported local file {legacy.as_posix()}; original acquisition timestamp unavailable; file mtime used as ingest proxy",
            validation_status=status,
            relative_path=relative,
            manifest_path=manifest_path,
            acquired_at=acquired,
        )
        copied += 1
    return copied


def _existing_valid_file(
    *, records: list[dict[str, str]], security_id: str, symbol: str,
    source: str, start: date, end: date, manifest_path: str | Path,
    data_root: str | Path = DATA_ROOT,
) -> bool:
    record = next((
        row for row in records
        if row.get("security_id") == security_id
        and row.get("symbol_used", "").upper() == symbol.upper()
        and row.get("source") == source
        and row.get("interval") == "day"
    ), None)
    if not record or record.get("validation_status") not in {"VALIDATED", "VALIDATED_EVALUATION_PARTIAL_WARMUP"}:
        return False
    try:
        path = Path(data_root) / record["relative_path"]
        rows = read_market_data(path)
        return bool(rows and rows[0]["date"][:10] <= start.isoformat() and rows[-1]["date"][:10] >= end.isoformat())
    except (OSError, KeyError, MarketDataError, csv.Error):
        return False


def _download_kite_daily(
    *, client: Any, token: int, security_id: str, symbol: str, start: date, end: date,
    universe_id: str, snapshot_path: Path, data_root: str | Path, manifest_path: str | Path,
    throttle: RequestThrottle, sleep: Callable[[float], None],
) -> tuple[Path, int]:
    chunks: list[dict[str, Any]] = []
    cursor = start
    try:
        while cursor <= end:
            chunk_end = min(end, cursor + timedelta(days=SUPPORTED_INTERVALS["day"] - 1))
            last_error: Exception | None = None
            for attempt in range(4):
                throttle.wait()
                try:
                    chunk = client.historical_data(
                        token,
                        datetime.combine(cursor, day_time.min),
                        datetime.combine(chunk_end, day_time.max),
                        "day",
                    )
                    if not isinstance(chunk, list):
                        raise DatasetError("Kite returned an invalid historical response.")
                    chunks.extend(chunk)
                    last_error = None
                    break
                except KiteException as error:
                    last_error = error
                except DatasetError:
                    raise
                except Exception as error:
                    last_error = error
                if attempt < 3:
                    sleep(2 ** attempt)
            if last_error is not None:
                if isinstance(last_error, KiteException):
                    raise DatasetError(f"Kite rejected daily history for {security_id} ({symbol}).") from None
                raise DatasetError(f"Could not retrieve daily history for {security_id} ({symbol}).") from None
            cursor = chunk_end + timedelta(days=1)
    except DatasetError:
        raise

    try:
        normalized = validate_ohlcv(chunks)
    except MarketDataError as error:
        raise DatasetError(f"Historical rows failed validation for {security_id}: {error}") from None
    normalized = [row for row in normalized if start.isoformat() <= row["date"][:10] <= end.isoformat()]
    if not normalized:
        raise DatasetError(f"Kite returned no daily candles for {security_id} ({symbol}).")
    destination = dataset_path(security_id, symbol, root=data_root, universe_id=universe_id)
    written = write_market_data(destination, normalized)
    try:
        relative = destination.relative_to(Path(data_root)).as_posix()
    except ValueError:
        relative = destination.as_posix()
    record_coverage(
        universe_id=universe_id,
        security_id=security_id,
        source=KITE_SOURCE,
        symbol_used=symbol,
        interval="day",
        requested_start=start,
        requested_end=end,
        rows=written,
        provenance=f"Current Kite instrument snapshot {snapshot_path.name}; historical_data response; token not stored in market-data manifest",
        validation_status="VALIDATED",
        relative_path=relative,
        manifest_path=manifest_path,
    )
    return destination, len(written)


def _official_secondary_targets(
    windows: dict[str, tuple[date, date, list[Any]]],
    *, warmup_start: date, evaluation_end: date,
) -> list[tuple[str, str, date, date]]:
    targets: list[tuple[str, str, date, date]] = []
    if "NSEEQ-0024" in windows:
        first, last, _ = windows["NSEEQ-0024"]
        targets.append(("NSEEQ-0024", "HDFC", warmup_start, min(last, evaluation_end)))
    if "NSEEQ-0055" in windows:
        targets.append((
            "NSEEQ-0055", "TATAMOTORS", warmup_start,
            min(evaluation_end, TATA_DEMERGER_EFFECTIVE - timedelta(days=1)),
        ))
    return targets


def _import_secondaries(
    targets: list[tuple[str, str, date, date]], *, universe_id: str,
    data_root: str | Path, manifest_path: str | Path,
) -> tuple[int, list[str]]:
    imported = 0
    failures: list[str] = []
    records = read_manifest(manifest_path)
    for security_id, symbol, start, end in targets:
        if _existing_valid_file(
            records=records, security_id=security_id, symbol=symbol,
            source=NSE_SOURCE, start=start, end=end, manifest_path=manifest_path,
            data_root=data_root,
        ):
            continue
        try:
            import_nse_daily(
                security_id=security_id, symbol=symbol, start_date=start, end_date=end,
                universe_id=universe_id, data_root=data_root, manifest_path=manifest_path,
            )
            imported += 1
            records = read_manifest(manifest_path)
        except (NseHistoryError, MarketDataError):
            failures.append(f"{security_id}:{symbol}")
    return imported, failures


def build_dataset(
    *, universe_id: str, start_date: date, end_date: date, intervals: tuple[str, ...],
    warmup_start: date, include_nse_secondary: bool = False,
    provider: UniverseProvider | None = None,
    snapshot_directory: str | Path = PROJECT_ROOT / "data" / "raw" / "instruments",
    data_root: str | Path = DATA_ROOT,
    manifest_path: str | Path = MANIFEST_PATH,
    probe_manifest_path: str | Path = DEFAULT_PROBE_MANIFEST,
    client_factory: Callable[[], Any] = create_authenticated_client,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    if intervals != ("day",):
        raise DatasetError("This milestone supports only --intervals day.")
    if start_date > end_date or warmup_start > start_date:
        raise DatasetError("Dates must satisfy warmup_start <= start <= end.")
    provider = provider or UniverseProvider()
    windows = _member_windows(provider, universe_id, start_date, end_date)
    if not windows:
        raise DatasetError("No tradable PIT securities were found in the requested period.")
    adopted = adopt_legacy_daily_files(
        universe_id=universe_id,
        evaluation_start=start_date,
        evaluation_end=end_date,
        warmup_start=warmup_start,
        provider=provider,
        data_root=data_root,
        manifest_path=manifest_path,
    )

    throttle = RequestThrottle(sleep=sleep)

    secondary_imported = 0
    secondary_failures: list[str] = []
    if include_nse_secondary:
        secondary_imported, secondary_failures = _import_secondaries(
            _official_secondary_targets(windows, warmup_start=warmup_start, evaluation_end=end_date),
            universe_id=universe_id,
            data_root=data_root,
            manifest_path=manifest_path,
        )

    snapshot_date, instruments = latest_instrument_snapshot(snapshot_directory)
    cached_probes = read_probe_manifest(probe_manifest_path)
    cached_by_id: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in cached_probes:
        if (
            row.get("universe_id") == universe_id
            and row.get("evaluation_start") == start_date.isoformat()
            and row.get("evaluation_end") == end_date.isoformat()
            and row.get("warmup_start") == warmup_start.isoformat()
            and row.get("snapshot_date") == snapshot_date.isoformat()
        ):
            cached_by_id[row["security_id"]].append(row)
    reusable_statuses = {"PASS_HISTORICAL_DATA", "NO_CURRENT_TOKEN", "FAIL_CLOSED_IDENTITY_BOUNDARY"}
    reusable = set(cached_by_id) >= set(windows) and all(
        any(row["status"] in reusable_statuses for row in cached_by_id[security_id])
        for security_id in windows
    )
    if reusable:
        probes = [row for rows in cached_by_id.values() for row in rows]
    else:
        try:
            probes = probe_universe(
                start_date=start_date,
                end_date=end_date,
                warmup_start=warmup_start,
                universe_id=universe_id,
                provider=provider,
                snapshot_directory=snapshot_directory,
                request_wait=throttle.wait,
                client_factory=client_factory,
            )
        except KiteProbeError as error:
            raise DatasetError(str(error)) from None
        write_probe_manifest(probes, probe_manifest_path)

    probe_by_id: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in probes:
        probe_by_id[row["security_id"]].append(row)

    pass_ids = {
        security_id for security_id, rows in probe_by_id.items()
        if any(row["status"] == "PASS_HISTORICAL_DATA" for row in rows)
    }
    kite_results: list[str] = []
    kite_failures = [
        f"{row['security_id']}:{row['status']}"
        for row in probes
        if row["status"] not in {"PASS_HISTORICAL_DATA", "FAIL_CLOSED_IDENTITY_BOUNDARY"}
    ]
    kite_client: Any | None = None
    if pass_ids:
        try:
            kite_client = client_factory()
        except (AuthenticationError, SessionVerificationError):
            kite_client = None
        except Exception:
            kite_client = None

    symbols = _security_names(provider)
    probe_symbol = {
        row["security_id"]: row["symbol_used"]
        for row in probes
        if row["status"] == "PASS_HISTORICAL_DATA"
    }
    records = read_manifest(manifest_path)
    for security_id, (first_member, last_member, _) in sorted(windows.items()):
        if security_id == "NSEEQ-0024":
            continue  # Delisted HDFC: handled only by the explicit NSE daily source.
        symbol = probe_symbol.get(security_id)
        if not symbol:
            continue
        # Do not use TMPV's current token for the pre-demerger TATAMOTORS series.
        if security_id == "NSEEQ-0055":
            acquire_start = TATA_DEMERGER_EFFECTIVE
        else:
            acquire_start = _request_start(security_id, first_member, start_date, warmup_start)
        acquire_end = end_date
        if acquire_start > acquire_end:
            continue
        if _existing_valid_file(
            records=records, security_id=security_id, symbol=symbol,
            source=KITE_SOURCE, start=acquire_start, end=acquire_end,
            manifest_path=manifest_path,
            data_root=data_root,
        ):
            kite_results.append(f"{security_id}:cached")
            continue
        if kite_client is None:
            kite_failures.append(f"{security_id}:session unavailable")
            continue
        instrument = instruments.get(symbol)
        if not instrument:
            kite_failures.append(f"{security_id}:no current token")
            continue
        try:
            path, count = _download_kite_daily(
                client=kite_client,
                token=int(instrument["instrument_token"]),
                security_id=security_id,
                symbol=symbol,
                start=acquire_start,
                end=acquire_end,
                universe_id=universe_id,
                snapshot_path=Path(snapshot_directory) / f"nse_instruments_{snapshot_date.isoformat()}.csv",
                data_root=data_root,
                manifest_path=manifest_path,
                throttle=throttle,
                sleep=sleep,
            )
            kite_results.append(f"{security_id}:{count} rows")
            records = read_manifest(manifest_path)
        except (DatasetError, MarketDataError) as error:
            kite_failures.append(f"{security_id}:{error}")

    return {
        "security_count": len(windows),
        "legacy_files_adopted": adopted,
        "kite_probe_statuses": {status: sum(row["status"] == status for row in probes) for status in sorted({row["status"] for row in probes})},
        "kite_downloads": kite_results,
        "kite_failures": kite_failures,
        "nse_secondary_imported": secondary_imported,
        "nse_secondary_failures": secondary_failures,
    }


def audit_dataset(
    *, universe_id: str, start_date: date, end_date: date,
    provider: UniverseProvider | None = None,
    data_root: str | Path = DATA_ROOT,
    manifest_path: str | Path = MANIFEST_PATH,
) -> dict[str, Any]:
    provider = provider or UniverseProvider()
    windows = _member_windows(provider, universe_id, start_date, end_date)
    mappings = provider.unresolved_broker_mappings(start_date, end_date, universe_id=universe_id)
    manifests = [row for row in read_manifest(manifest_path) if row.get("universe_id") == universe_id and row.get("interval") == "day"]
    by_security: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in manifests:
        by_security[row.get("security_id", "")].append(row)

    validated_rows: dict[str, list[list[dict[str, Any]]]] = defaultdict(list)
    session_calendar: set[date] = set()
    for security_id, rows in by_security.items():
        for row in rows:
            if row.get("validation_status") not in {"VALIDATED", "VALIDATED_EVALUATION_PARTIAL_WARMUP"}:
                continue
            try:
                candles = read_market_data(Path(data_root) / row["relative_path"])
            except (OSError, KeyError, MarketDataError, csv.Error):
                continue
            if not candles:
                continue
            validated_rows[security_id].append(candles)
            session_calendar.update(date.fromisoformat(candle["date"][:10]) for candle in candles)

    complete: list[str] = []
    gaps: list[dict[str, str]] = []
    for security_id, (required_start, required_end, memberships) in sorted(windows.items()):
        records = validated_rows.get(security_id, [])
        if not records:
            gaps.append({"security_id": security_id, "reason": "no validated daily market data"})
            continue
        starts = [date.fromisoformat(candles[0]["date"][:10]) for candles in records]
        ends = [date.fromisoformat(candles[-1]["date"][:10]) for candles in records]
        actual_start, actual_end = min(starts), max(ends)
        observed = {
            date.fromisoformat(candle["date"][:10])
            for candles in records for candle in candles
        }
        expected_sessions = {
            session for session in session_calendar
            if any(
                max(start_date, membership.effective_from) <= session
                and session <= min(
                    end_date,
                    membership.effective_to - timedelta(days=1) if membership.effective_to else end_date,
                )
                for membership in memberships
            )
        }
        missing_sessions = sorted(expected_sessions - observed)
        # Endpoint allowance covers a weekend/exchange holiday if the observed global calendar begins or ends nearby.
        start_ok = actual_start <= required_start + timedelta(days=7)
        end_ok = actual_end >= required_end - timedelta(days=7)
        if start_ok and end_ok and not missing_sessions:
            complete.append(security_id)
        else:
            reasons = []
            if not start_ok:
                reasons.append(f"starts {actual_start}, expected by {(required_start + timedelta(days=7)).isoformat()}")
            if not end_ok:
                reasons.append(f"ends {actual_end}, expected through {(required_end - timedelta(days=7)).isoformat()}")
            if missing_sessions:
                reasons.append(
                    f"missing {len(missing_sessions)} observed exchange-session dates "
                    f"({missing_sessions[0]} through {missing_sessions[-1]})"
                )
            gaps.append({"security_id": security_id, "reason": "; ".join(reasons)})

    return {
        "broker_mapping_unresolved_ranges": mappings,
        "market_data_complete_security_ids": complete,
        "market_data_gaps": gaps,
        "market_data_total_security_ids": len(windows),
    }


def _print_audit(result: dict[str, Any]) -> None:
    print("BROKER MAPPING COVERAGE")
    print(f"Unresolved date ranges: {len(result['broker_mapping_unresolved_ranges'])}")
    for security_id, symbol, from_date, to_date in result["broker_mapping_unresolved_ranges"]:
        print(f"  {security_id} {symbol}: {from_date} through {to_date}")
    print("MARKET DATA COVERAGE")
    print(
        f"Complete daily security series: {len(result['market_data_complete_security_ids'])}/"
        f"{result['market_data_total_security_ids']}"
    )
    for gap in result["market_data_gaps"]:
        print(f"  {gap['security_id']}: {gap['reason']}")
    print("Interior dates are checked against the union of dates in validated local series; missing candles are never filled.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build", help="Build resumable PIT daily OHLCV files.")
    audit_parser = subparsers.add_parser("audit", help="Compare broker mappings and daily data coverage.")
    for command_parser in (build_parser, audit_parser):
        command_parser.add_argument("--universe", default=PIT_TRADABLE)
        command_parser.add_argument("--start", required=True, type=date.fromisoformat)
        command_parser.add_argument("--end", required=True, type=date.fromisoformat)
    build_parser.add_argument("--intervals", default="day")
    build_parser.add_argument("--warmup-start", type=date.fromisoformat, default=date(2022, 7, 1))
    build_parser.add_argument(
        "--include-nse-secondary", action="store_true",
        help="Explicitly import HDFC and pre-demerger TATAMOTORS daily data from NSE's official archive.",
    )
    build_parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    build_parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    build_parser.add_argument("--probe-manifest", type=Path, default=DEFAULT_PROBE_MANIFEST)
    audit_parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    audit_parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    audit_parser.add_argument("--interval", choices=("day",), default="day")
    args = parser.parse_args()
    try:
        if args.command == "build":
            intervals = tuple(value.strip() for value in args.intervals.split(",") if value.strip())
            result = build_dataset(
                universe_id=args.universe,
                start_date=args.start,
                end_date=args.end,
                intervals=intervals,
                warmup_start=args.warmup_start,
                include_nse_secondary=args.include_nse_secondary,
                data_root=args.data_root,
                manifest_path=args.manifest,
                probe_manifest_path=args.probe_manifest,
            )
            print(json.dumps(result, indent=2))
            audit = audit_dataset(
                universe_id=args.universe,
                start_date=args.start,
                end_date=args.end,
                data_root=args.data_root,
                manifest_path=args.manifest,
            )
            _print_audit(audit)
        else:
            _print_audit(audit_dataset(
                universe_id=args.universe,
                start_date=args.start,
                end_date=args.end,
                data_root=args.data_root,
                manifest_path=args.manifest,
            ))
    except (DatasetError, KiteProbeError, MarketDataError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
