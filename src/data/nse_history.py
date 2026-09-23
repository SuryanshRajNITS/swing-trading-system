"""Explicit importer for the NSE website's official daily price-volume CSV export."""

from __future__ import annotations

import argparse
import csv
import io
import re
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable

import requests

from src.data.market_data import (
    DATA_ROOT,
    MANIFEST_PATH,
    MarketDataError,
    dataset_path,
    record_coverage,
    validate_ohlcv,
    write_market_data,
)


NSE_ARCHIVE_PAGE = "https://www.nseindia.com/report-detail/eq_security"
NSE_EXPORT_ENDPOINT = "https://www.nseindia.com/api/historicalOR/generateSecurityWiseHistoricalData"
NSE_SOURCE = "NSE official security-wise historical archive"
NSE_PROVENANCE = (
    "NSE official security-wise archives page / CSV export; type=priceVolume; series=EQ"
)


class NseHistoryError(RuntimeError):
    """NSE historical export failed or returned malformed data."""


def parse_nse_price_volume_csv(content: bytes, *, symbol: str) -> list[dict[str, Any]]:
    try:
        reader = csv.DictReader(io.StringIO(content.decode("utf-8-sig")))
        raw_rows = list(reader)
    except (UnicodeDecodeError, csv.Error):
        raise NseHistoryError("NSE returned an invalid CSV export.") from None
    if not raw_rows:
        raise NseHistoryError("NSE returned no daily rows for the requested security and dates.")

    def normalized_keys(row: dict[str, Any]) -> dict[str, Any]:
        return {str(key).replace("\ufeff", "").strip().lower(): value for key, value in row.items()}

    rows: list[dict[str, Any]] = []
    for row in raw_rows:
        values = normalized_keys(row)
        try:
            row_symbol = str(values["symbol"]).strip().upper()
            series = str(values["series"]).strip().upper()
            candle = {
                "date": datetime.strptime(str(values["date"]).strip(), "%d-%b-%Y").date(),
                "open": values["open price"],
                "high": values["high price"],
                "low": values["low price"],
                "close": values["close price"],
                "volume": values["total traded quantity"],
            }
        except (KeyError, ValueError, TypeError):
            raise NseHistoryError("NSE CSV did not contain the expected equity OHLCV columns.") from None
        if row_symbol != symbol.upper() or series != "EQ":
            raise NseHistoryError("NSE returned a row for a different symbol or series.")
        rows.append(candle)
    try:
        return validate_ohlcv(rows)
    except MarketDataError as error:
        raise NseHistoryError(f"NSE daily data failed validation: {error}") from None


def fetch_nse_daily(
    symbol: str,
    start_date: date,
    end_date: date,
    *,
    session: requests.Session | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> list[dict[str, Any]]:
    if not re.fullmatch(r"[A-Z0-9&-]+", symbol.upper()):
        raise NseHistoryError("Invalid NSE symbol.")
    if start_date > end_date:
        raise NseHistoryError("Start date must not be later than end date.")
    if (end_date - start_date).days > 1826:
        raise NseHistoryError("NSE's official export form limits one request to five years.")

    http = session or requests.Session()
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": NSE_ARCHIVE_PAGE,
        "Accept": "text/csv,*/*",
    }
    try:
        page = http.get(NSE_ARCHIVE_PAGE, headers=headers, timeout=30)
        page.raise_for_status()
        sleep(0.25)
        response = http.get(
            NSE_EXPORT_ENDPOINT,
            params={
                "from": start_date.strftime("%d-%m-%Y"),
                "to": end_date.strftime("%d-%m-%Y"),
                "symbol": symbol.upper(),
                "type": "priceVolume",
                "series": "EQ",
                "csv": "true",
            },
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()
    except requests.RequestException:
        raise NseHistoryError("Could not retrieve daily data from NSE's official historical archive.") from None
    if "text/csv" not in response.headers.get("content-type", "").lower():
        raise NseHistoryError("NSE did not return a CSV response for the requested history.")
    return parse_nse_price_volume_csv(response.content, symbol=symbol)


def import_nse_daily(
    *, security_id: str, symbol: str, start_date: date, end_date: date,
    universe_id: str = "NIFTY50_PIT_TRADABLE", data_root: str | Path = DATA_ROOT,
    manifest_path: str | Path = MANIFEST_PATH,
    session: requests.Session | None = None,
) -> tuple[Path, int]:
    candles = fetch_nse_daily(symbol, start_date, end_date, session=session)
    bounded = [row for row in candles if start_date.isoformat() <= row["date"][:10] <= end_date.isoformat()]
    if not bounded:
        raise NseHistoryError("NSE returned no candles inside the requested date range.")
    destination = dataset_path(
        security_id, symbol, "day", universe_id=universe_id, root=data_root
    )
    normalized = write_market_data(destination, bounded)
    base = Path(data_root)
    try:
        relative = destination.relative_to(base).as_posix()
    except ValueError:
        relative = destination.as_posix()
    record_coverage(
        universe_id=universe_id,
        security_id=security_id,
        source=NSE_SOURCE,
        symbol_used=symbol.upper(),
        interval="day",
        requested_start=start_date,
        requested_end=end_date,
        rows=normalized,
        provenance=f"{NSE_PROVENANCE}; page={NSE_ARCHIVE_PAGE}",
        validation_status="VALIDATED",
        relative_path=relative,
        manifest_path=manifest_path,
    )
    return destination, len(normalized)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--security-id", required=True)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--start", required=True, type=date.fromisoformat)
    parser.add_argument("--end", required=True, type=date.fromisoformat)
    args = parser.parse_args()
    try:
        path, count = import_nse_daily(
            security_id=args.security_id,
            symbol=args.symbol,
            start_date=args.start,
            end_date=args.end,
        )
    except NseHistoryError as error:
        parser.error(str(error))
    print(f"Imported {count} official NSE daily rows to {path}.")


if __name__ == "__main__":
    main()
