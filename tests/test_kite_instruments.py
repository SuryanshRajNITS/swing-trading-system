from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

import pytest
from kiteconnect.exceptions import TokenException

from src.broker.instruments import (
    INSTRUMENT_FIELDS,
    InstrumentMasterError,
    download_nse_snapshot,
    save_nse_snapshot,
    validate_nse_instruments,
)


def instrument(token: int, symbol: str) -> dict[str, object]:
    return {
        "instrument_token": token,
        "exchange_token": str(token),
        "tradingsymbol": symbol,
        "name": symbol,
        "last_price": 100.0,
        "expiry": None,
        "strike": 0.0,
        "tick_size": 0.05,
        "lot_size": 1,
        "instrument_type": "EQ",
        "segment": "NSE",
        "exchange": "NSE",
    }


def test_validate_nse_instruments_sorts_rows_deterministically() -> None:
    rows = [instrument(2, "TCS"), instrument(1, "INFY")]

    result = validate_nse_instruments(rows)

    assert [row["tradingsymbol"] for row in result] == ["INFY", "TCS"]
    assert tuple(result[0]) == INSTRUMENT_FIELDS


def test_validate_rejects_empty_wrong_exchange_and_duplicate_tokens() -> None:
    with pytest.raises(InstrumentMasterError, match="empty or invalid"):
        validate_nse_instruments([])

    wrong_exchange = instrument(1, "INFY")
    wrong_exchange["exchange"] = "BSE"
    with pytest.raises(InstrumentMasterError, match="unexpected exchange"):
        validate_nse_instruments([wrong_exchange])

    with pytest.raises(InstrumentMasterError, match="duplicate instrument tokens"):
        validate_nse_instruments([instrument(1, "INFY"), instrument(1, "TCS")])


def test_save_snapshot_writes_date_stamped_normalized_csv(tmp_path: Path) -> None:
    destination = save_nse_snapshot(
        [instrument(2, "TCS"), instrument(1, "INFY")],
        directory=tmp_path / "raw" / "instruments",
        snapshot_date=date(2026, 9, 23),
    )

    assert destination.name == "nse_instruments_2026-09-23.csv"
    with destination.open(encoding="utf-8", newline="") as snapshot:
        reader = csv.DictReader(snapshot)
        assert tuple(reader.fieldnames or ()) == INSTRUMENT_FIELDS
        rows = list(reader)
    assert [row["tradingsymbol"] for row in rows] == ["INFY", "TCS"]
    assert rows[0]["exchange"] == "NSE"


def test_download_requests_only_nse_and_saves_validated_snapshot(tmp_path: Path) -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.exchange: str | None = None

        def instruments(self, exchange: str) -> list[dict[str, object]]:
            self.exchange = exchange
            return [instrument(1, "INFY")]

    client = FakeClient()
    path, row_count = download_nse_snapshot(
        client=client,  # type: ignore[arg-type]
        directory=tmp_path,
        snapshot_date=date(2026, 9, 23),
    )

    assert client.exchange == "NSE"
    assert path.exists()
    assert row_count == 1


def test_kite_failures_are_sanitized(tmp_path: Path) -> None:
    class FakeClient:
        def instruments(self, exchange: str) -> list[dict[str, object]]:
            raise TokenException("contains private-access-token")

    with pytest.raises(InstrumentMasterError, match="Kite rejected") as error:
        download_nse_snapshot(client=FakeClient(), directory=tmp_path)  # type: ignore[arg-type]

    assert "private-access-token" not in str(error.value)
