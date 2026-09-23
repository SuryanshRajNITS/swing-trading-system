"""Download, validate, and save a local NSE instrument-master snapshot."""

from __future__ import annotations

import csv
import math
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Iterable
from zoneinfo import ZoneInfo

from kiteconnect import KiteConnect
from kiteconnect.exceptions import KiteException

from src.broker.auth import AuthenticationError
from src.broker.session import SessionVerificationError, create_authenticated_client
from src.configuration import PROJECT_ROOT


INSTRUMENT_FIELDS = (
    "instrument_token",
    "exchange_token",
    "tradingsymbol",
    "name",
    "last_price",
    "expiry",
    "strike",
    "tick_size",
    "lot_size",
    "instrument_type",
    "segment",
    "exchange",
)
NSE_EXCHANGE = "NSE"
INDIA_TIMEZONE = ZoneInfo("Asia/Kolkata")
DEFAULT_SNAPSHOT_DIRECTORY = PROJECT_ROOT / "data" / "raw" / "instruments"


class InstrumentMasterError(RuntimeError):
    """Raised when the instrument master cannot be safely downloaded or saved."""


def validate_nse_instruments(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate and deterministically sort rows returned by Kite's NSE endpoint."""
    if isinstance(rows, (str, bytes, dict)):
        raise InstrumentMasterError("Kite returned an empty or invalid NSE instrument list.")
    try:
        rows = list(rows)
    except TypeError:
        raise InstrumentMasterError("Kite returned an empty or invalid NSE instrument list.") from None
    if not rows:
        raise InstrumentMasterError("Kite returned an empty or invalid NSE instrument list.")

    normalized: list[dict[str, Any]] = []
    seen_tokens: set[int] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise InstrumentMasterError(f"NSE instrument row {index} is not a mapping.")
        missing = [field for field in INSTRUMENT_FIELDS if field not in row]
        if missing:
            raise InstrumentMasterError(f"NSE instrument row {index} is missing required fields.")

        symbol = row["tradingsymbol"]
        if not isinstance(symbol, str) or not symbol.strip():
            raise InstrumentMasterError(f"NSE instrument row {index} has no trading symbol.")
        if row["exchange"] != NSE_EXCHANGE:
            raise InstrumentMasterError(f"NSE instrument row {index} has an unexpected exchange.")
        if not isinstance(row["instrument_type"], str) or not row["instrument_type"].strip():
            raise InstrumentMasterError(f"NSE instrument row {index} has no instrument type.")
        if not isinstance(row["segment"], str) or not row["segment"].strip():
            raise InstrumentMasterError(f"NSE instrument row {index} has no segment.")

        token = row["instrument_token"]
        if isinstance(token, bool) or not isinstance(token, int) or token <= 0:
            raise InstrumentMasterError(f"NSE instrument row {index} has an invalid instrument token.")
        if token in seen_tokens:
            raise InstrumentMasterError("NSE instrument master contains duplicate instrument tokens.")
        seen_tokens.add(token)

        for field in ("last_price", "strike", "tick_size"):
            value = row[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise InstrumentMasterError(f"NSE instrument row {index} has an invalid {field} value.")
        if isinstance(row["lot_size"], bool) or not isinstance(row["lot_size"], int) or row["lot_size"] < 0:
            raise InstrumentMasterError(f"NSE instrument row {index} has an invalid lot size.")

        normalized.append({field: row[field] for field in INSTRUMENT_FIELDS})

    normalized.sort(
        key=lambda row: (
            row["tradingsymbol"],
            row["instrument_type"],
            str(row["expiry"] or ""),
            float(row["strike"]),
            row["instrument_token"],
        )
    )
    return normalized


def save_nse_snapshot(
    rows: Iterable[dict[str, Any]],
    *,
    directory: str | Path = DEFAULT_SNAPSHOT_DIRECTORY,
    snapshot_date: date | None = None,
) -> Path:
    """Write a normalized, date-stamped CSV snapshot atomically."""
    validated = validate_nse_instruments(rows)
    snapshot_date = snapshot_date or datetime.now(INDIA_TIMEZONE).date()
    output_directory = Path(directory)
    destination = output_directory / f"nse_instruments_{snapshot_date.isoformat()}.csv"
    temporary = destination.with_suffix(destination.suffix + ".tmp")

    try:
        output_directory.mkdir(parents=True, exist_ok=True)
        with temporary.open("w", encoding="utf-8", newline="") as snapshot:
            writer = csv.DictWriter(snapshot, fieldnames=INSTRUMENT_FIELDS, extrasaction="raise")
            writer.writeheader()
            writer.writerows(validated)
        temporary.replace(destination)
    except OSError:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise InstrumentMasterError("Could not write the local NSE instrument snapshot.") from None
    return destination


def download_nse_snapshot(
    *,
    client: KiteConnect | None = None,
    client_factory: Callable[[], KiteConnect] = create_authenticated_client,
    directory: str | Path = DEFAULT_SNAPSHOT_DIRECTORY,
    snapshot_date: date | None = None,
) -> tuple[Path, int]:
    """Fetch the NSE-only Kite instrument master and save a validated CSV copy."""
    try:
        client = client or client_factory()
        rows = client.instruments(NSE_EXCHANGE)
    except (AuthenticationError, SessionVerificationError) as error:
        raise InstrumentMasterError(str(error)) from None
    except KiteException:
        raise InstrumentMasterError("Kite rejected the NSE instrument-master request.") from None
    except Exception:
        raise InstrumentMasterError("Could not download the NSE instrument master from Kite.") from None

    validated = validate_nse_instruments(rows)
    path = save_nse_snapshot(validated, directory=directory, snapshot_date=snapshot_date)
    return path, len(validated)


def main() -> None:
    try:
        path, row_count = download_nse_snapshot()
    except InstrumentMasterError as error:
        print(f"NSE instrument-master download failed: {error}")
        raise SystemExit(1) from None
    print(f"Saved {row_count} NSE instrument records to {path}.")


if __name__ == "__main__":
    main()
