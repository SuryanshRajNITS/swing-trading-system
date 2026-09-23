"""Read-only probes that verify historical data availability without changing mappings."""

from __future__ import annotations

import argparse
import csv
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

from kiteconnect.exceptions import KiteException

from src.broker.auth import AuthenticationError
from src.broker.session import SessionVerificationError, create_authenticated_client
from src.configuration import PROJECT_ROOT
from src.data.universe.provider import PIT_TRADABLE, UniverseProvider


DEFAULT_PROBE_MANIFEST = PROJECT_ROOT / "data" / "processed" / "kite_probe_manifest.csv"
SNAPSHOT_DIRECTORY = PROJECT_ROOT / "data" / "raw" / "instruments"
PROBE_FIELDS = (
    "universe_id", "security_id", "symbol_used", "requested_start", "requested_end",
    "actual_start", "actual_end", "row_count", "status", "provenance",
)
NSE_TZ = ZoneInfo("Asia/Kolkata")
TATA_DEMERGER_EFFECTIVE = date(2025, 10, 24)
LISTING_BOUNDARIES = {"NSEEQ-0035", "NSEEQ-0036", "NSEEQ-0039", "NSEEQ-0060"}


class KiteProbeError(RuntimeError):
    """A safe, sanitized error during read-only historical-data probing."""


def latest_instrument_snapshot(directory: str | Path = SNAPSHOT_DIRECTORY) -> tuple[date, dict[str, dict[str, Any]]]:
    paths = sorted(Path(directory).glob("nse_instruments_*.csv"))
    if not paths:
        raise KiteProbeError("No saved Kite NSE instrument snapshot is available.")
    try:
        observed_on = date.fromisoformat(paths[-1].stem.removeprefix("nse_instruments_"))
    except ValueError:
        raise KiteProbeError("The latest NSE instrument snapshot has an invalid date name.") from None
    with paths[-1].open("r", encoding="utf-8-sig", newline="") as handle:
        instruments = {
            row["tradingsymbol"].upper(): row
            for row in csv.DictReader(handle)
            if row.get("exchange") == "NSE" and row.get("instrument_type") == "EQ"
            and row.get("tradingsymbol")
        }
    return observed_on, instruments


def _security_symbols(reference_dir: Path) -> dict[str, str]:
    securities_path = reference_dir / "securities.csv"
    with securities_path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {row["security_id"]: row["canonical_symbol"].upper() for row in csv.DictReader(handle)}


def _probe_start(
    provider: UniverseProvider, security_id: str, first_member: date,
    end_date: date, warmup_start: date,
) -> date:
    if security_id == "NSEEQ-0055":
        return max(first_member, TATA_DEMERGER_EFFECTIVE)
    if security_id in LISTING_BOUNDARIES:
        return first_member
    return max(warmup_start, first_member - timedelta(days=180))


def probe_universe(
    *, start_date: date, end_date: date, warmup_start: date,
    universe_id: str = PIT_TRADABLE,
    provider: UniverseProvider | None = None,
    snapshot_directory: str | Path = SNAPSHOT_DIRECTORY,
    client: Any | None = None,
    client_factory: Callable[[], Any] = create_authenticated_client,
    request_wait: Callable[[], None] | None = None,
    throttle_seconds: float = 1 / 3,
    sleep: Callable[[float], None] = time.sleep,
) -> list[dict[str, str]]:
    if start_date > end_date or warmup_start > start_date:
        raise KiteProbeError("Probe dates must satisfy warmup_start <= start <= end.")
    provider = provider or UniverseProvider()
    members = provider.members_over(start_date, end_date, universe_id)
    tradable = [row for row in members if row.tradable]
    first_dates: dict[str, date] = {}
    for member in tradable:
        first_dates[member.security_id] = min(first_dates.get(member.security_id, member.effective_from), member.effective_from)

    snapshot_date, instruments = latest_instrument_snapshot(snapshot_directory)
    symbols = _security_symbols(provider.reference_directory)
    # Current symbol history is used to find today's contract only. It does not change the dated broker map.
    current_symbols: dict[str, str] = {}
    for security_id, canonical in symbols.items():
        try:
            current_symbols[security_id] = provider.symbol_for(security_id, snapshot_date).upper()
        except LookupError:
            current_symbols[security_id] = canonical

    # Avoid opening a session if no usable current token exists at all.
    requested: list[tuple[str, str, date, date, str]] = []
    results: list[dict[str, str]] = []
    for security_id, first_member in sorted(first_dates.items()):
        current_symbol = current_symbols.get(security_id, "")
        token_record = instruments.get(current_symbol)
        if token_record is None:
            results.append(_probe_row(universe_id, security_id, current_symbol, None, None, [], "NO_CURRENT_TOKEN"))
            continue
        probe_start = _probe_start(provider, security_id, first_member, end_date, warmup_start)
        if probe_start > end_date:
            results.append(_probe_row(
                universe_id, security_id, current_symbol, probe_start, end_date, [],
                "FAIL_CLOSED_IDENTITY_BOUNDARY",
            ))
            continue
        probe_end = min(end_date, probe_start + timedelta(days=7))
        # Never use the TMPV contract to probe the pre-demerger Tata Motors identity.
        if security_id == "NSEEQ-0055" and probe_start == TATA_DEMERGER_EFFECTIVE:
            results.append(_probe_row(
                universe_id, security_id, "TATAMOTORS", first_member,
                TATA_DEMERGER_EFFECTIVE - timedelta(days=1), [],
                "FAIL_CLOSED_IDENTITY_BOUNDARY",
            ))
        requested.append((security_id, current_symbol, probe_start, probe_end, token_record["instrument_token"]))

    if requested:
        try:
            client = client or client_factory()
        except (AuthenticationError, SessionVerificationError):
            results.extend(
                _probe_row(
                    universe_id, security_id, symbol, probe_start, probe_end, [],
                    "SESSION_UNAVAILABLE",
                )
                for security_id, symbol, probe_start, probe_end, _ in requested
            )
            return sorted(results, key=lambda row: (row["security_id"], row["symbol_used"]))
        except Exception:
            results.extend(
                _probe_row(
                    universe_id, security_id, symbol, probe_start, probe_end, [],
                    "SESSION_UNAVAILABLE",
                )
                for security_id, symbol, probe_start, probe_end, _ in requested
            )
            return sorted(results, key=lambda row: (row["security_id"], row["symbol_used"]))

    last_request = 0.0
    for security_id, symbol, probe_start, probe_end, token_text in requested:
        if request_wait is not None:
            request_wait()
        else:
            elapsed = time.monotonic() - last_request
            if last_request and elapsed < throttle_seconds:
                sleep(throttle_seconds - elapsed)
        try:
            rows = client.historical_data(
                int(token_text),
                datetime.combine(probe_start, datetime.min.time()),
                datetime.combine(probe_end, datetime.max.time()),
                "day",
            )
        except KiteException:
            rows = None
            status = "KITE_REJECTED"
        except Exception:
            rows = None
            status = "REQUEST_FAILED"
        last_request = time.monotonic()
        if rows is None:
            results.append(_probe_row(universe_id, security_id, symbol, probe_start, probe_end, [], status))
        else:
            dated = [row for row in rows if isinstance(row, dict) and row.get("date") is not None]
            status = "PASS_HISTORICAL_DATA" if dated else "NO_CANDLES"
            results.append(_probe_row(universe_id, security_id, symbol, probe_start, probe_end, dated, status))

    return sorted(results, key=lambda row: (row["security_id"], row["symbol_used"]))


def _probe_row(
    universe_id: str, security_id: str, symbol: str, start: date | None,
    end: date | None, candles: list[dict[str, Any]], status: str,
) -> dict[str, str]:
    dates = []
    for row in candles:
        value = row.get("date")
        dates.append(value.date().isoformat() if isinstance(value, datetime) else str(value)[:10])
    return {
        "universe_id": universe_id,
        "security_id": security_id,
        "symbol_used": symbol,
        "requested_start": start.isoformat() if start else "",
        "requested_end": end.isoformat() if end else "",
        "actual_start": min(dates, default=""),
        "actual_end": max(dates, default=""),
        "row_count": str(len(candles)),
        "status": status,
        "provenance": "Current Kite token; read-only historical_data probe; no broker mapping modified",
    }


def write_probe_manifest(rows: list[dict[str, str]], path: str | Path = DEFAULT_PROBE_MANIFEST) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=PROBE_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(destination)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise KiteProbeError("Could not save the Kite data-probe manifest.") from None


def read_probe_manifest(path: str | Path = DEFAULT_PROBE_MANIFEST) -> list[dict[str, str]]:
    manifest = Path(path)
    if not manifest.exists():
        return []
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", default=PIT_TRADABLE)
    parser.add_argument("--start", required=True, type=date.fromisoformat)
    parser.add_argument("--end", required=True, type=date.fromisoformat)
    parser.add_argument("--warmup-start", type=date.fromisoformat, default=date(2022, 7, 1))
    parser.add_argument("--output", type=Path, default=DEFAULT_PROBE_MANIFEST)
    args = parser.parse_args()
    try:
        rows = probe_universe(
            start_date=args.start,
            end_date=args.end,
            warmup_start=args.warmup_start,
            universe_id=args.universe,
        )
        write_probe_manifest(rows, args.output)
    except KiteProbeError as error:
        print(f"Kite historical-data probe failed: {error}")
        raise SystemExit(1) from None
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    print(f"Read-only historical-data probes: {counts}; report: {args.output}")


if __name__ == "__main__":
    main()
