from __future__ import annotations

import csv
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.data.dataset import (
    KITE_SOURCE,
    DatasetError,
    RequestThrottle,
    _download_kite_daily,
    adopt_legacy_daily_files,
    audit_dataset,
)
from src.data.kite_probe import probe_universe, write_probe_manifest
from src.data.market_data import MarketDataError, dataset_path, read_manifest, read_market_data, validate_ohlcv
from src.data.nse_history import NseHistoryError, parse_nse_price_volume_csv
from src.data.universe.provider import PIT_TRADABLE, UniverseProvider
from src.data.weekly import last_completed_weekly_candle, resample_daily_to_weekly


def _candle(day: date, *, opening: float = 10.0, high: float = 12.0, low: float = 9.0,
            close: float = 11.0, volume: int = 100) -> dict[str, object]:
    return {
        "date": datetime.combine(day, datetime.min.time()),
        "open": opening,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }


def test_daily_validation_rejects_duplicate_and_bad_volume() -> None:
    day = date(2023, 1, 2)
    with pytest.raises(MarketDataError, match="duplicate"):
        validate_ohlcv([_candle(day), _candle(day)])
    invalid = _candle(day, volume=-1)
    with pytest.raises(MarketDataError, match="volume"):
        validate_ohlcv([invalid])


def test_nse_parser_normalizes_official_csv_and_rejects_wrong_symbol() -> None:
    sample = (
        '"Symbol  ","Series  ","Date  ","Open Price  ","High Price  ","Low Price  ",'
        '"Close Price  ","Total Traded Quantity  "\r\n'
        'HDFC,EQ,12-Jul-2023,"2,755.80","2,777.65","2,710.60","2,724.30","1,234,567"\r\n'
    ).encode("utf-8")
    rows = parse_nse_price_volume_csv(sample, symbol="HDFC")
    assert rows == [{
        "date": "2023-07-12T00:00:00+05:30",
        "open": 2755.8,
        "high": 2777.65,
        "low": 2710.6,
        "close": 2724.3,
        "volume": 1234567,
    }]
    with pytest.raises(NseHistoryError, match="different symbol"):
        parse_nse_price_volume_csv(sample.replace(b"HDFC,EQ", b"OTHER,EQ"), symbol="HDFC")


def test_legacy_infys_are_adopted_as_market_data_without_mapping_claim(tmp_path: Path) -> None:
    source = tmp_path / "legacy"
    source.mkdir()
    for symbol in ("INFY", "TCS"):
        with (source / f"{symbol}.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("date", "open", "high", "low", "close", "volume", "oi"))
            writer.writeheader()
            writer.writerow({"date": "2023-01-02T00:00:00+05:30", "open": 10, "high": 12,
                             "low": 9, "close": 11, "volume": 100, "oi": 0})
    root = tmp_path / "processed" / "market_data"
    manifest = tmp_path / "processed" / "market_data_manifest.csv"

    count = adopt_legacy_daily_files(
        evaluation_start=date(2023, 1, 1),
        evaluation_end=date(2023, 1, 3),
        warmup_start=date(2022, 7, 1),
        legacy_directory=source,
        data_root=root,
        manifest_path=manifest,
    )

    assert count == 2
    rows = read_manifest(manifest)
    assert {row["security_id"] for row in rows} == {"NSEEQ-0033", "NSEEQ-0057"}
    assert all(row["source"] != "Kite Connect historical API" for row in rows)
    assert not any("instrument_token" in row for row in rows)
    assert read_market_data(root / PIT_TRADABLE / "day" / "NSEEQ-0033" / "INFY.csv")[0]["volume"] == 100


def test_kite_probe_keeps_mappings_unchanged_and_respects_tata_identity_boundary(tmp_path: Path) -> None:
    # Make a dated synthetic snapshot with current symbols; token values are test fixtures only.
    provider = UniverseProvider()
    securities_path = provider.reference_directory / "securities.csv"
    with securities_path.open("r", encoding="utf-8", newline="") as handle:
        securities = list(csv.DictReader(handle))
    overrides = {"NSEEQ-0041": "LTM", "NSEEQ-0055": "TMPV", "NSEEQ-0065": "ETERNAL"}
    snapshot = tmp_path / "nse_instruments_2026-09-23.csv"
    with snapshot.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("tradingsymbol", "instrument_token", "instrument_type", "exchange"))
        writer.writeheader()
        for token, row in enumerate(securities, start=10000):
            if row["identity_type"] == "EQUITY" and row["security_id"] != "NSEEQ-0024":
                symbol = overrides.get(row["security_id"], row["canonical_symbol"])
                writer.writerow({"tradingsymbol": symbol, "instrument_token": token,
                                 "instrument_type": "EQ", "exchange": "NSE"})

    class FakeKite:
        def __init__(self) -> None:
            self.calls: list[tuple[int, datetime, datetime]] = []

        def historical_data(self, token: int, start: datetime, end: datetime, interval: str):
            assert interval == "day"
            self.calls.append((token, start, end))
            return [_candle(start.date())]

    client = FakeKite()
    original = (provider.reference_directory / "broker_instrument_history.csv").read_bytes()
    results = probe_universe(
        start_date=date(2023, 1, 1), end_date=date(2026, 9, 22),
        warmup_start=date(2022, 7, 1), provider=provider,
        snapshot_directory=tmp_path, client=client, throttle_seconds=0,
    )
    write_probe_manifest(results, tmp_path / "probes.csv")

    tata_rows = [row for row in results if row["security_id"] == "NSEEQ-0055"]
    assert any(row["symbol_used"] == "TATAMOTORS" and row["status"] == "FAIL_CLOSED_IDENTITY_BOUNDARY" for row in tata_rows)
    assert any(row["symbol_used"] == "TMPV" and row["status"] == "PASS_HISTORICAL_DATA" for row in tata_rows)
    et = next(row for row in results if row["security_id"] == "NSEEQ-0065")
    assert et["symbol_used"] == "ETERNAL" and et["status"] == "PASS_HISTORICAL_DATA"
    assert any(row["status"] == "NO_CURRENT_TOKEN" and row["security_id"] == "NSEEQ-0024" for row in results)
    assert not any("instrument_token" in row for row in results)
    assert original == (provider.reference_directory / "broker_instrument_history.csv").read_bytes()
    assert all(call[1].date() >= date(2025, 10, 24) for call in client.calls if call[0] == 10000 + next(
        i for i, item in enumerate(securities) if item["security_id"] == "NSEEQ-0055"
    ))


def test_kite_daily_writer_is_atomic_and_manifest_is_vendor_neutral(tmp_path: Path) -> None:
    target = dataset_path("NSEEQ-0001", "ADANIENT", root=tmp_path / "market")
    manifest = tmp_path / "manifest.csv"
    snapshot = tmp_path / "nse_instruments_2026-09-23.csv"
    snapshot.write_text("", encoding="utf-8")

    class FakeKite:
        def historical_data(self, token: int, start: datetime, end: datetime, interval: str):
            return [_candle(date(2023, 1, 2))]

    result, count = _download_kite_daily(
        client=FakeKite(), token=1001, security_id="NSEEQ-0001", symbol="ADANIENT",
        start=date(2023, 1, 1), end=date(2023, 1, 3), universe_id=PIT_TRADABLE,
        snapshot_path=snapshot, data_root=tmp_path / "market", manifest_path=manifest,
        throttle=RequestThrottle(0), sleep=lambda _: None,
    )
    assert result == target and count == 1 and target.exists()
    row = read_manifest(manifest)[0]
    assert row["source"] == KITE_SOURCE
    assert "instrument_token" not in row


def test_dataset_audit_separates_market_coverage_from_broker_mapping(tmp_path: Path) -> None:
    # No manifest means market coverage is missing while broker coverage is independently reported.
    result = audit_dataset(
        universe_id=PIT_TRADABLE,
        start_date=date(2023, 1, 1),
        end_date=date(2026, 9, 22),
        data_root=tmp_path / "market",
        manifest_path=tmp_path / "missing.csv",
    )
    assert result["broker_mapping_unresolved_ranges"]
    assert result["market_data_total_security_ids"] == 61
    assert not result["market_data_complete_security_ids"]


def test_weekly_aggregation_uses_observed_sessions_and_hides_current_week_until_close() -> None:
    days = [date(2023, 1, 2), date(2023, 1, 3), date(2023, 1, 4), date(2023, 1, 5), date(2023, 1, 6)]
    rows = [_candle(day, opening=10 + i, high=15 + i, low=8 + i, close=12 + i, volume=10) for i, day in enumerate(days)]
    weekly = resample_daily_to_weekly(rows, trading_sessions=days)
    assert len(weekly) == 1
    candle = weekly[0]
    assert (candle["open"], candle["high"], candle["low"], candle["close"], candle["volume"]) == (10, 19, 8, 16, 50)
    assert last_completed_weekly_candle(datetime(2023, 1, 5, 15, 30), weekly) is None
    assert last_completed_weekly_candle(datetime(2023, 1, 6, 15, 29), weekly) is None
    assert last_completed_weekly_candle(datetime(2023, 1, 6, 15, 30), weekly) == candle


def test_weekly_short_holiday_week_completes_on_last_observed_exchange_session() -> None:
    days = [date(2023, 1, 2), date(2023, 1, 3), date(2023, 1, 5)]
    rows = [_candle(day, volume=3) for day in days]
    weekly = resample_daily_to_weekly(rows, trading_sessions=days)
    assert weekly[0]["last_session"] == "2023-01-05"
    assert weekly[0]["volume"] == 9
    assert last_completed_weekly_candle(datetime(2023, 1, 4, 15, 30), weekly) is None
    assert last_completed_weekly_candle(datetime(2023, 1, 5, 15, 30), weekly) == weekly[0]
