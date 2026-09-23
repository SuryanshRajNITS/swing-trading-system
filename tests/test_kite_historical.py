from __future__ import annotations

import csv
from datetime import date, datetime
from pathlib import Path

import pytest
from kiteconnect.exceptions import TokenException

from src.data.historical import (
    HistoricalDataError,
    download_instrument_history,
    save_candles,
    validate_candles,
)


def candle(timestamp: datetime) -> dict[str, object]:
    return {
        "date": timestamp,
        "open": 10.0,
        "high": 12.0,
        "low": 9.0,
        "close": 11.0,
        "volume": 100,
        "oi": 0,
    }


def test_validate_candles_normalizes_timezone_and_sorts() -> None:
    rows = [candle(datetime(2023, 1, 2, 9, 15)), candle(datetime(2023, 1, 1, 9, 15))]

    result = validate_candles(rows)

    assert result[0]["date"] == "2023-01-01T09:15:00+05:30"
    assert result[1]["date"] == "2023-01-02T09:15:00+05:30"


def test_validate_rejects_bad_ohlc_and_duplicate_timestamps() -> None:
    invalid = candle(datetime(2023, 1, 1, 9, 15))
    invalid["high"] = 8.0
    with pytest.raises(HistoricalDataError, match="inconsistent OHLC"):
        validate_candles([invalid])

    timestamp = datetime(2023, 1, 1, 9, 15)
    with pytest.raises(HistoricalDataError, match="duplicate"):
        validate_candles([candle(timestamp), candle(timestamp)])


def test_save_candles_writes_normalized_csv(tmp_path: Path) -> None:
    path = save_candles(
        [candle(datetime(2023, 1, 1, 9, 15))], symbol="INFY", interval="day", directory=tmp_path
    )

    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert path.name == "INFY.csv"
    assert rows[0]["date"] == "2023-01-01T09:15:00+05:30"
    assert rows[0]["volume"] == "100"


def test_download_chunks_requests_and_saves_intervals(tmp_path: Path) -> None:
    class FakeKite:
        def __init__(self) -> None:
            self.requests: list[tuple[int, datetime, datetime, str]] = []

        def historical_data(self, token: int, start: datetime, end: datetime, interval: str):
            self.requests.append((token, start, end, interval))
            return [candle(datetime(2023, 1, 2, 9, 15))]

    client = FakeKite()
    paths = download_instrument_history(
        {"tradingsymbol": "INFY", "instrument_token": 123},
        start_date=date(2023, 1, 1), end_date=date(2023, 1, 2),
        intervals=("day", "15minute"), client=client, output_directory=tmp_path,  # type: ignore[arg-type]
    )

    assert set(paths) == {"day", "15minute"}
    assert len(client.requests) == 2
    assert all(request[0] == 123 for request in client.requests)
    assert paths["day"].exists() and paths["15minute"].exists()


def test_kite_errors_are_sanitized(tmp_path: Path) -> None:
    class FakeKite:
        def historical_data(self, *args: object) -> list[dict[str, object]]:
            raise TokenException("private-access-token")

    with pytest.raises(HistoricalDataError, match="Kite rejected") as error:
        download_instrument_history(
            {"tradingsymbol": "INFY", "instrument_token": 123},
            start_date=date(2023, 1, 1), end_date=date(2023, 1, 2),
            intervals=("day",), client=FakeKite(), output_directory=tmp_path,  # type: ignore[arg-type]
        )
    assert "private-access-token" not in str(error.value)
