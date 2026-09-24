from __future__ import annotations

import csv
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from src.backtest.data import load_hourly_dataset_through
from src.data.hourly import (
    ACCESS_FIELDS,
    HourlyDataError,
    SESSION_OVERRIDES,
    audit_hourly_dataset,
    build_hourly_dataset,
    hourly_chunks,
    hourly_window_coverage_status,
    normalize_hourly,
    probe_hourly_access,
    session_type_on,
    valid_session_timestamp,
)
from src.data.market_data import MANIFEST_FIELDS
from src.data.kite_probe import latest_instrument_snapshot
from src.data.universe.provider import BrokerInstrument, MembershipInterval, SymbolInterval, UniverseEvent
from src.data.universe.provider import DEFAULT_PROVIDER


def _bar(stamp: datetime, *, close: float = 10) -> dict:
    return {"date": stamp, "open": close, "high": close + 1, "low": close - 1,
            "close": close, "volume": 100, "oi": 0}


def _daily_inputs(root: Path, manifest: Path, days: list[date]) -> None:
    path = root / "daily" / "one.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("date", "open", "high", "low", "close", "volume"))
        writer.writeheader()
        for day in days:
            writer.writerow({"date": day.isoformat(), "open": 10, "high": 11, "low": 9, "close": 10, "volume": 100})
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerow({"universe_id": "NIFTY50_PIT_TRADABLE", "security_id": "SEC-1",
                         "interval": "day", "validation_status": "VALIDATED", "relative_path": "daily/one.csv"})


def _build_inputs(root: Path, days: list[date]) -> tuple[Path, Path, Path]:
    data_root = root / "data"
    daily_manifest = root / "daily.csv"
    access_manifest = root / "access.csv"
    _daily_inputs(data_root, daily_manifest, days)
    with access_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ACCESS_FIELDS)
        writer.writeheader()
        writer.writerow({
            "universe_id": "NIFTY50_PIT_TRADABLE", "security_id": "SEC-1",
            "historical_symbol": "OLD", "symbol_effective_from": "2020-01-01",
            "evaluation_start": days[0].isoformat(), "evaluation_end": days[-1].isoformat(),
            "warmup_start": days[0].isoformat(), "requested_start": days[0].isoformat(),
            "requested_end": days[-1].isoformat(), "snapshot_date": "2026-09-23",
            "current_symbol": "NEW", "current_resolved_token": "987", "instrument_token": "987",
            "probe_date": days[0].isoformat(), "probe_result": "PASS_HISTORICAL_DATA",
            "market_data_access_result": "PASS_HISTORICAL_DATA",
            "market_data_access_provenance": "CURRENT_TOKEN_HISTORICAL_PROBE",
        })
    snapshots = root / "instruments"
    _snapshot(snapshots)
    return data_root, daily_manifest, access_manifest


def _weekday_dates(first: date, count: int) -> list[date]:
    days = []
    cursor = first
    while len(days) < count:
        if cursor.weekday() < 5:
            days.append(cursor)
        cursor += timedelta(days=1)
    return days


def _snapshot(directory: Path, *, include_current: bool = True, symbol: str = "NEW", token: int = 987) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "nse_instruments_2026-09-23.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("tradingsymbol", "instrument_token", "instrument_type", "exchange"))
        writer.writeheader()
        if include_current:
            writer.writerow({"tradingsymbol": symbol, "instrument_token": token,
                             "instrument_type": "EQ", "exchange": "NSE"})
    return path


class _HourlyProvider:
    def __init__(self, *, mapping: BrokerInstrument | None = None, boundary: bool = False,
                 event_source: str = "zomato_rename", current_symbol: str = "NEW"):
        self.mapping = mapping
        self.boundary = boundary
        self.event_source = event_source
        self.current_symbol = current_symbol

    def members_over(self, start, end, universe_id):
        return [MembershipInterval("SEC-1", universe_id, date(2020, 1, 1), None,
                                   None, "STANDARD", True, "membership-source")]

    def symbol_for(self, security_id, day):
        raise AssertionError("Current index membership/date lookup must not be used for token resolution")

    def latest_symbol_for(self, security_id):
        assert security_id == "SEC-1"
        return SymbolInterval(security_id, self.current_symbol, "Example Ltd", date(2025, 1, 1),
                              None, "rename-source")

    def symbol_history_over(self, security_id, start, end):
        return [SymbolInterval(security_id, "OLD", "Example Ltd", date(2020, 1, 1),
                               None, "rename-source")]

    def broker_instruments_over(self, security_id, start, end):
        return [self.mapping] if self.mapping else []

    def identity_events_over(self, security_id, start, end):
        if start > date(2025, 10, 24) or end < date(2025, 10, 24):
            return []
        return [UniverseEvent("SYMBOL_CHANGED", "SECURITY_IDENTITY", security_id, None,
                              date(2025, 10, 24), "OLD", self.current_symbol,
                              self.event_source, "Corporate action identity boundary")]

    def security_identity_for(self, security_id):
        return "INE155A01022", "tata_pv_isin"


class _FakeHourlyClient:
    def __init__(self, days: list[date], *, empty: bool = False):
        self.days = days
        self.empty = empty
        self.calls: list[tuple[int, date, date, str]] = []

    def historical_data(self, token, start, end, interval):
        self.calls.append((token, start.date(), end.date(), interval))
        if self.empty:
            return []
        rows = []
        for day in self.days:
            if start.date() <= day <= end.date():
                rows.extend(_bar(datetime.combine(day, time(hour, 15))) for hour in range(9, 15))
                rows.append(_bar(datetime.combine(day, time(15, 15))))
        return rows


def test_hourly_chunks_respect_400_day_kite_limit():
    chunks = hourly_chunks(date(2022, 10, 1), date(2024, 12, 31))
    assert len(chunks) == 3
    assert (chunks[0][1] - chunks[0][0]).days == 399
    assert chunks[0][1].date() + timedelta(days=1) == chunks[1][0].date()


def test_existing_broker_mapping_is_not_backdated_for_stable_security_id():
    assert DEFAULT_PROVIDER.broker_instruments_over(
        "NSEEQ-0001", date(2023, 1, 1), date(2024, 12, 31)
    ) == []
    current = DEFAULT_PROVIDER.broker_instruments_over(
        "NSEEQ-0001", date(2026, 9, 23), date(2026, 9, 23)
    )
    assert len(current) == 1
    assert current[0].source_id == "kite_snapshot_2026-09-23"
    renamed = DEFAULT_PROVIDER.symbol_history_over(
        "NSEEQ-0065", date(2025, 3, 28), date(2025, 4, 10)
    )
    assert [item.symbol for item in renamed] == ["ZOMATO", "ETERNAL"]
    assert all(item.security_id == "NSEEQ-0065" for item in renamed)
    events = DEFAULT_PROVIDER.identity_events_over(
        "NSEEQ-0065", date(2025, 3, 28), date(2025, 4, 10)
    )
    assert [event.source_id for event in events] == ["zomato_rename"]


def test_former_index_members_resolve_from_full_nse_snapshot_without_membership_fallback():
    snapshot_date, instruments = latest_instrument_snapshot()
    expected = {
        "NSEEQ-0011": ("BPCL", "134657"),
        "NSEEQ-0012": ("BRITANNIA", "140033"),
        "NSEEQ-0016": ("DIVISLAB", "2800641"),
        "NSEEQ-0027": ("HEROMOTOCO", "345089"),
        "NSEEQ-0032": ("INDUSINDBK", "1346049"),
        "NSEEQ-0063": ("UPL", "2889473"),
    }
    for security_id, (symbol, token) in expected.items():
        assert DEFAULT_PROVIDER.latest_symbol_for(security_id).symbol == symbol
        assert instruments[symbol]["instrument_token"] == token
        assert instruments[symbol]["exchange"] == "NSE"
        assert instruments[symbol]["instrument_type"] == "EQ"
    assert DEFAULT_PROVIDER.latest_symbol_for("NSEEQ-0024").symbol == "HDFC"
    assert "HDFC" not in instruments
    assert instruments["HDFCBANK"]["instrument_token"] == "341249"


def test_timezone_session_grid_and_final_partial_are_preserved():
    sessions = {date(2023, 1, 3)}
    rows = normalize_hourly([
        _bar(datetime.fromisoformat("2023-01-03T15:15:00+05:30")),
        _bar(datetime.fromisoformat("2023-01-03T09:15:00")),
    ], session_dates=sessions)
    assert rows[0]["date"] == "2023-01-03T09:15:00+05:30"
    assert rows[-1]["date"] == "2023-01-03T15:15:00+05:30"
    assert valid_session_timestamp("2023-01-03T09:15:00+05:30", {date(2023, 1, 3)})
    assert not valid_session_timestamp("2023-01-03T15:30:00+05:30", {date(2023, 1, 3)})
    assert not valid_session_timestamp("2023-01-04T09:15:00+05:30", {date(2023, 1, 3)})
    normal_grid = [datetime.fromisoformat(f"2023-01-03T{hour:02d}:15:00+05:30") for hour in range(9, 16)]
    assert all(valid_session_timestamp(stamp, {date(2023, 1, 3)}) for stamp in normal_grid)
    assert {row["session_type"] for row in normalize_hourly(
        [_bar(stamp) for stamp in normal_grid], session_dates=sessions
    )} == {"NORMAL_SESSION"}


def test_known_muhurat_sessions_are_explicit_and_off_grid_times_fail_closed():
    muhurt_2022 = datetime.fromisoformat("2022-10-24T18:15:00+05:30")
    assert session_type_on(date(2022, 10, 24)) == "MUHURAT"
    assert valid_session_timestamp(muhurt_2022, {date(2022, 10, 24)})
    muhurt_2023 = datetime.fromisoformat("2023-11-12T18:15:00+05:30")
    assert valid_session_timestamp(muhurt_2023, {date(2023, 11, 12)})
    assert normalize_hourly([_bar(muhurt_2023)], session_dates={date(2023, 11, 12)})[0]["session_type"] == "MUHURAT"
    muhurt_2024 = datetime.fromisoformat("2024-11-01T18:00:00+05:30")
    assert valid_session_timestamp(muhurt_2024, {date(2024, 11, 1)})
    assert not valid_session_timestamp("2023-11-12T18:16:00+05:30", {date(2023, 11, 12)})
    assert not valid_session_timestamp("2023-11-19T18:15:00+05:30", {date(2023, 11, 19)})
    assert valid_session_timestamp("2024-11-01T18:00:00+05:30", {date(2024, 11, 1)})
    assert not valid_session_timestamp("2024-11-01T18:15:00+00:00", {date(2024, 11, 1)})


def test_regular_special_saturday_uses_date_override_without_allowing_other_weekends():
    saturday = date(2024, 1, 20)
    all_regular_starts = [datetime.fromisoformat(f"2024-01-20T{hour:02d}:15:00+05:30")
                          for hour in range(9, 16)]
    # Explicit override label and all regular-session hourly candle starts.
    assert session_type_on(saturday) == "SPECIAL_REGULAR_SESSION"
    assert SESSION_OVERRIDES[saturday]["source"] == "NSE/MSD/60340"
    assert all(valid_session_timestamp(stamp, {saturday}) for stamp in all_regular_starts)
    assert normalize_hourly([_bar(stamp) for stamp in all_regular_starts], session_dates={saturday})[0]["session_type"] == "SPECIAL_REGULAR_SESSION"
    assert not valid_session_timestamp("2024-01-20T10:30:00+05:30", {saturday})
    assert session_type_on(date(2024, 1, 27)) == "CLOSED"
    assert not valid_session_timestamp("2024-01-27T09:15:00+05:30", {date(2024, 1, 27)})


def test_special_session_native_kite_grid_is_accepted_without_date_specific_whitelist():
    march_2 = date(2024, 3, 2)
    override = SESSION_OVERRIDES[march_2]
    assert session_type_on(march_2) == "SPECIAL_SPLIT_SESSION"
    assert override["exchange_trading_windows"] == (
        (time(9, 15), time(10, 0)), (time(11, 30), time(12, 30))
    )
    assert override["source"] == "NSE/MSD/60677"
    provider_starts = (time(9, 15), time(11, 15), time(12, 15))
    assert all(valid_session_timestamp(datetime.combine(march_2, slot), {march_2})
               for slot in provider_starts)
    # Kite's 11:15 timestamp is a provider bar start, not a continuous exchange open.
    assert not any(start <= time(11, 15) < end
                   for start, end in override["exchange_trading_windows"])
    for trading_day in (march_2, date(2024, 5, 18)):
        assert session_type_on(trading_day) == "SPECIAL_SPLIT_SESSION"
        candles = [_bar(datetime.combine(trading_day, slot)) for slot in provider_starts]
        normalized = normalize_hourly(candles, session_dates={trading_day})
        assert [row["date"][11:16] for row in normalized] == ["09:15", "11:15", "12:15"]
        assert {row["provider_grid_status"] for row in normalized} == {"NATIVE_SPECIAL_SESSION"}
        assert all(row["provider_timestamp_raw"].endswith(slot.strftime("%H:%M:00"))
                   for row, slot in zip(normalized, provider_starts))
    assert not valid_session_timestamp(datetime.combine(date(2024, 1, 27), time(9, 15)),
                                       {date(2024, 1, 27)})
    assert not valid_session_timestamp(datetime.combine(date(2024, 3, 3), time(9, 15)),
                                       {date(2024, 3, 3)})
    # NSE/MSD circular 64037 identifies September 21 as a mock, not live trading.
    mock_date = date(2024, 9, 21)
    assert session_type_on(mock_date) == "CLOSED"
    assert not valid_session_timestamp(datetime.combine(mock_date, time(11, 15)), {mock_date})
    assert not valid_session_timestamp("2024-03-02T08:15:00+05:30", {march_2})
    assert not valid_session_timestamp("2024-03-02T20:45:00+05:30", {march_2})
    assert not valid_session_timestamp("2024-03-02T11:07:00+05:30", {march_2})


def test_special_session_rejects_duplicate_nonmonotonic_and_bad_ohlc_bars():
    trading_day = date(2024, 3, 2)
    def bar(hhmm: str) -> dict:
        return _bar(datetime.fromisoformat(f"2024-03-02T{hhmm}:00+05:30"))

    with pytest.raises(HourlyDataError, match="strictly increasing"):
        normalize_hourly([bar("11:15"), bar("09:15")], session_dates={trading_day})
    with pytest.raises(HourlyDataError, match="duplicate timestamps"):
        normalize_hourly([bar("09:15"), bar("09:15")], session_dates={trading_day})
    invalid = bar("09:15")
    invalid["high"] = 1
    with pytest.raises(HourlyDataError, match="inconsistent OHLC"):
        normalize_hourly([invalid], session_dates={trading_day})
    assert session_type_on(date(2023, 11, 12)) == "MUHURAT"
    assert session_type_on(date(2024, 11, 1)) == "MUHURAT"


def test_split_session_metadata_preserves_kite_timestamp_and_session_identity():
    raw = "2024-03-02T11:15:00+05:30"
    normalized = normalize_hourly([_bar(datetime.fromisoformat(raw))], session_dates={date(2024, 3, 2)})[0]
    assert normalized["date"] == raw
    assert normalized["provider_timestamp_raw"] == raw
    assert normalized["provider_interval"] == "60minute"
    assert normalized["provider_bar_start_semantics"] == "KITE_NATIVE"
    assert normalized["session_type"] == "SPECIAL_SPLIT_SESSION"
    assert normalized["provider_grid_status"] == "NATIVE_SPECIAL_SESSION"
    assert normalized["session_type"] != SESSION_OVERRIDES[date(2024, 1, 20)]["session_type"]
    assert normalized["session_type"] != "MUHURAT"


def test_invalid_candle_error_includes_full_chunk_context_neighbors_and_intraday_grid():
    rows = [
        _bar(datetime.fromisoformat(f"2023-01-03T{hour:02d}:15:00+05:30"))
        for hour in (9, 10, 11, 12, 13)
    ]
    rows[3]["date"] = "2023-01-03T12:45:00+05:30"
    with pytest.raises(HourlyDataError) as error:
        normalize_hourly(rows, session_dates={date(2023, 1, 3)}, diagnostic_context={
            "security_id": "SEC-1", "symbol": "TEST", "instrument_token": 77,
            "chunk_start": "2023-01-01T00:00:00", "chunk_end": "2023-01-31T23:59:59",
        })
    diagnostic = str(error.value)
    for expected in (
        "Hourly candle 3", "security_id=SEC-1", "symbol=TEST", "instrument_token=77",
        "requested_chunk=2023-01-01T00:00:00..2023-01-31T23:59:59",
        "2023-01-03T12:45:00+05:30", "asia_kolkata_timestamp=2023-01-03T12:45:00+05:30",
        "'open': 10", "previous_3=", "next_3=", "distinct_intraday_times=",
        "09:15:00.000000", "12:45:00.000000",
    ):
        assert expected in diagnostic


def test_seconds_are_not_snapped_to_hourly_grid():
    candle = _bar(datetime.fromisoformat("2023-01-03T09:15:01+05:30"))
    assert not valid_session_timestamp(candle["date"], {date(2023, 1, 3)})
    with pytest.raises(HourlyDataError, match="seconds or microseconds are non-zero"):
        normalize_hourly([candle], session_dates={date(2023, 1, 3)})


def test_normalization_is_deterministic_and_rejects_duplicates_and_bad_ohlc():
    a = _bar(datetime.fromisoformat("2023-01-03T10:15:00+05:30"))
    b = _bar(datetime.fromisoformat("2023-01-03T09:15:00+05:30"))
    sessions = {date(2023, 1, 3)}
    assert normalize_hourly([a, b], session_dates=sessions) == normalize_hourly([b, a], session_dates=sessions)
    with pytest.raises(HourlyDataError, match="duplicate"):
        normalize_hourly([a, a], session_dates=sessions)
    broken = _bar(datetime.fromisoformat("2023-01-03T09:15:00+05:30"))
    broken["high"] = 8
    with pytest.raises(HourlyDataError, match="inconsistent OHLC"):
        normalize_hourly([broken], session_dates=sessions)


@pytest.mark.parametrize(("field", "value", "expected_invariant"), [
    ("high", 9, "high < close"),
    ("low", 11, "low > open"),
    ("low", 12, "high < low"),
    ("close", float("nan"), "non_finite_close"),
    ("volume", float("nan"), "non_finite_volume"),
])
def test_ohlcv_diagnostic_names_failed_invariant_and_provider_context(field, value, expected_invariant):
    timestamps = [datetime.fromisoformat(f"2023-01-03T{hour:02d}:15:00+05:30") for hour in range(9, 14)]
    rows = [_bar(stamp) for stamp in timestamps]
    rows[2][field] = value
    with pytest.raises(HourlyDataError) as error:
        normalize_hourly(rows, session_dates={date(2023, 1, 3)}, diagnostic_context={
            "security_id": "SEC-X", "symbol": "SAMPLE", "instrument_token": 42,
            "chunk_start": "2023-01-01T00:00:00", "chunk_end": "2023-01-31T23:59:59",
        })
    diagnostic = str(error.value)
    for required in (
        "security_id=SEC-X", "symbol=SAMPLE", "instrument_token=42",
        "requested_chunk=2023-01-01T00:00:00..2023-01-31T23:59:59",
        "candle_index=2", "raw_timestamp=", "asia_kolkata_timestamp=",
        "raw_OHLCV=", "failed_invariant=",
        "previous_3=", "next_3=",
    ):
        assert required in diagnostic
    assert expected_invariant in diagnostic


def test_renamed_stable_security_current_token_probe_authorizes_download_without_claiming_mapping(tmp_path: Path):
    data_root = tmp_path / "data"
    manifest = tmp_path / "hourly.csv"
    daily_manifest = tmp_path / "daily.csv"
    access_manifest = tmp_path / "access.csv"
    snapshots = tmp_path / "instruments"
    session_days = [date(2023, 1, 3), date(2023, 1, 4)]
    _daily_inputs(data_root, daily_manifest, session_days)
    _snapshot(snapshots)
    # An explicitly sourced stable rename may bridge historical and current symbols.
    provider = _HourlyProvider(boundary=True, event_source="zomato_rename")
    client = _FakeHourlyClient(session_days)
    probes = probe_hourly_access(
        start_date=session_days[0], end_date=session_days[-1], warmup_start=date(2022, 10, 1),
        provider=provider, snapshot_directory=snapshots, data_root=data_root,
        daily_manifest_path=daily_manifest, output_path=access_manifest, client=client,
        sleep=lambda _: None,
    )
    assert len(probes) == 1
    assert probes[0]["historical_symbol"] == "OLD"
    assert probes[0]["current_symbol"] == "NEW"
    assert probes[0]["current_resolved_token"] == "987"
    assert probes[0]["probe_date"] == "2023-01-03"
    assert probes[0]["probe_result"] == "PASS_RENAMED_TOKEN_HISTORICAL_DATA"
    assert probes[0]["market_data_access_result"] == "PASS_RENAMED_TOKEN_HISTORICAL_DATA"
    assert probes[0]["market_data_access_provenance"] == "CURRENT_RENAMED_TOKEN_HISTORICAL_PROBE"
    assert probes[0]["current_instrument_lookup_source"] == "FULL_NSE_EQUITY_SNAPSHOT:2026-09-23"
    assert probes[0]["identity_status"] == "VERIFIED_RENAME_CONTINUITY"

    client.calls.clear()
    result = build_hourly_dataset(
        start_date=date(2023, 1, 3), end_date=date(2023, 1, 4),
        warmup_start=date(2022, 10, 1), provider=provider, client=client,
        data_root=data_root, manifest_path=manifest, daily_manifest_path=daily_manifest,
        access_manifest_path=access_manifest, snapshot_directory=snapshots,
        sleep=lambda _: None,
    )
    assert result["mapped_ranges"] == 1
    assert result["evidenced_broker_mapping_ranges"] == 0
    assert result["current_token_probe_ranges"] == 1
    assert client.calls[0][0] == 987
    rows = list(csv.DictReader(manifest.open(encoding="utf-8", newline="")))
    assert rows[0]["security_id"] == "SEC-1"
    assert rows[0]["instrument_token"] == "987"
    assert rows[0]["broker_mapping_provenance"] == "NOT_HISTORICALLY_EVIDENCED"
    assert rows[0]["market_data_access_provenance"] == "CURRENT_RENAMED_TOKEN_HISTORICAL_PROBE"
    assert rows[0]["actual_last_timestamp"].startswith("2023-01-04T15:15")
    assert rows[0]["content_sha256"]
    # Completed security datasets are reused without broker calls unless forced.
    client.calls.clear()
    reused = build_hourly_dataset(
        start_date=date(2023, 1, 3), end_date=date(2023, 1, 4),
        warmup_start=date(2022, 10, 1), provider=provider, client=client,
        data_root=data_root, manifest_path=manifest, daily_manifest_path=daily_manifest,
        access_manifest_path=access_manifest, snapshot_directory=snapshots,
        sleep=lambda _: None,
    )
    assert len(client.calls) == 0
    assert reused["reused_securities"] == ["SEC-1"]
    build_hourly_dataset(
        start_date=date(2023, 1, 3), end_date=date(2023, 1, 4),
        warmup_start=date(2022, 10, 1), provider=provider, client=client,
        data_root=data_root, manifest_path=manifest, daily_manifest_path=daily_manifest,
        access_manifest_path=access_manifest, snapshot_directory=snapshots,
        sleep=lambda _: None, force=True,
    )
    assert len(client.calls) == 1


def test_evidenced_historical_mapping_is_preferred_over_current_token(tmp_path: Path):
    data_root, manifest, daily, access = (tmp_path / name for name in ("data", "hourly.csv", "daily.csv", "access.csv"))
    days = [date(2023, 1, 3)]
    _daily_inputs(data_root, daily, days)
    snapshots = tmp_path / "instruments"
    _snapshot(snapshots)
    mapping = BrokerInstrument("SEC-1", "kite", "NSE", "OLD", 123,
                               date(2020, 1, 1), None, "historic-source")
    provider = _HourlyProvider(mapping=mapping)
    client = _FakeHourlyClient(days)
    rows = probe_hourly_access(start_date=days[0], end_date=days[0], warmup_start=date(2022, 10, 1),
                               provider=provider, snapshot_directory=snapshots, data_root=data_root,
                               daily_manifest_path=daily, output_path=access, client=client, sleep=lambda _: None)
    assert rows[0]["probe_result"] == "PASS_HISTORICAL_DATA"
    assert rows[0]["instrument_token"] == "123"
    assert rows[0]["broker_mapping_source_id"] == "historic-source"
    assert rows[0]["market_data_access_provenance"] == "EVIDENCED_HISTORICAL_MAPPING"
    assert client.calls[0][0] == 123
    result = build_hourly_dataset(
        start_date=days[0], end_date=days[0], warmup_start=date(2022, 10, 1),
        provider=provider, client=client, data_root=data_root, manifest_path=manifest,
        daily_manifest_path=daily, access_manifest_path=access, snapshot_directory=snapshots,
        sleep=lambda _: None,
    )
    assert result["evidenced_broker_mapping_ranges"] == 1
    data_rows = list(csv.DictReader(manifest.open(encoding="utf-8", newline="")))
    assert data_rows[0]["broker_mapping_provenance"] == "HISTORICALLY_EVIDENCED:historic-source"
    assert data_rows[0]["market_data_access_provenance"] == "EVIDENCED_HISTORICAL_MAPPING"


def test_missing_current_token_reports_no_access_without_calling_kite(tmp_path: Path):
    data_root, daily, access = tmp_path / "data", tmp_path / "daily.csv", tmp_path / "access.csv"
    days = [date(2023, 1, 3)]
    _daily_inputs(data_root, daily, days)
    snapshots = tmp_path / "instruments"
    _snapshot(snapshots, include_current=False)
    client = _FakeHourlyClient(days)
    rows = probe_hourly_access(start_date=days[0], end_date=days[0], warmup_start=date(2022, 10, 1),
                               provider=_HourlyProvider(), snapshot_directory=snapshots, data_root=data_root,
                               daily_manifest_path=daily, output_path=access, client=client, sleep=lambda _: None)
    assert rows[0]["probe_result"] == "NO_CURRENT_TOKEN"
    assert client.calls == []
    result = build_hourly_dataset(
        start_date=days[0], end_date=days[0], warmup_start=date(2022, 10, 1),
        provider=_HourlyProvider(), client_factory=lambda: pytest.fail("must not create Kite client"),
        data_root=data_root, manifest_path=tmp_path / "hourly.csv", daily_manifest_path=daily,
        access_manifest_path=access, snapshot_directory=snapshots,
    )
    assert result["mapped_ranges"] == 0
    assert result["access_failures"] == ["SEC-1:OLD:NO_CURRENT_TOKEN"]
    recorded = list(csv.DictReader((tmp_path / "hourly.csv").open(encoding="utf-8", newline="")))
    assert recorded[0]["validation_status"] == "NO_KITE_HISTORICAL_ACCESS"


def test_unsafe_identity_boundary_fails_closed_without_probe(tmp_path: Path):
    data_root, daily, access = tmp_path / "data", tmp_path / "daily.csv", tmp_path / "access.csv"
    days = [date(2023, 1, 3)]
    _daily_inputs(data_root, daily, days)
    snapshots = tmp_path / "instruments"
    _snapshot(snapshots)
    client = _FakeHourlyClient(days)
    rows = probe_hourly_access(start_date=days[0], end_date=days[0], warmup_start=date(2022, 10, 1),
                               provider=_HourlyProvider(boundary=True, event_source="unknown_rename"), snapshot_directory=snapshots,
                               data_root=data_root, daily_manifest_path=daily, output_path=access,
                               client=client, sleep=lambda _: None)
    assert rows[0]["probe_result"] == "FAIL_CLOSED_IDENTITY_BOUNDARY"
    assert client.calls == []


def test_tata_identity_with_same_security_id_isin_and_symbol_history_can_probe(tmp_path: Path):
    data_root, daily, access = tmp_path / "data", tmp_path / "daily.csv", tmp_path / "access.csv"
    days = [date(2023, 1, 3)]
    _daily_inputs(data_root, daily, days)
    snapshots = tmp_path / "instruments"
    _snapshot(snapshots, symbol="TMPV", token=884737)

    class TataProvider(_HourlyProvider):
        def members_over(self, start, end, universe_id):
            return [MembershipInterval("NSEEQ-0055", universe_id, date(2023, 1, 1), None,
                                       None, "STANDARD", True, "membership-source")]

        def symbol_for(self, security_id, day):
            raise AssertionError("PIT symbol lookup must not resolve the current contract")

        def latest_symbol_for(self, security_id):
            return SymbolInterval(security_id, "TMPV", "Tata Motors Passenger Vehicles Ltd.",
                                  date(2025, 10, 24), None, "tata_symbol_change")

        def symbol_history_over(self, security_id, start, end):
            return [SymbolInterval(security_id, "TATAMOTORS", "Tata Motors Ltd.",
                                   date(2023, 1, 1), date(2025, 10, 24), "sebi_dec2022"),
                    SymbolInterval(security_id, "TMPV", "Tata Motors Passenger Vehicles Ltd.",
                                   date(2025, 10, 24), None, "tata_symbol_change")]

        def identity_events_over(self, security_id, start, end):
            if start <= date(2025, 10, 24) <= end:
                return [UniverseEvent("SYMBOL_CHANGED", "SECURITY_IDENTITY", security_id,
                                      date(2025, 10, 16), date(2025, 10, 24), "TATAMOTORS", "TMPV",
                                      "tata_symbol_change", "")]
            return []

        def security_identity_for(self, security_id):
            return "INE155A01022", "tata_pv_isin"

    provider = TataProvider(boundary=True, event_source="tata_symbol_change", current_symbol="TMPV")
    client = _FakeHourlyClient(days)
    rows = probe_hourly_access(start_date=days[0], end_date=days[0], warmup_start=date(2022, 10, 1),
                               provider=provider, snapshot_directory=snapshots, data_root=data_root,
                               daily_manifest_path=daily, output_path=access, client=client, sleep=lambda _: None)
    assert rows[0]["probe_result"] == "PASS_RENAMED_TOKEN_HISTORICAL_DATA"
    assert rows[0]["market_data_access_provenance"] == "CURRENT_RENAMED_TOKEN_HISTORICAL_PROBE"


def test_unverified_rename_fails_closed_even_if_current_token_is_present(tmp_path: Path):
    data_root, daily, access = tmp_path / "data", tmp_path / "daily.csv", tmp_path / "access.csv"
    days = [date(2023, 1, 3)]
    _daily_inputs(data_root, daily, days)
    snapshots = tmp_path / "instruments"
    _snapshot(snapshots)
    provider = _HourlyProvider(boundary=True, event_source="unknown_rename")
    client = _FakeHourlyClient(days)
    rows = probe_hourly_access(start_date=days[0], end_date=days[0], warmup_start=date(2022, 10, 1),
                               provider=provider, snapshot_directory=snapshots, data_root=data_root,
                               daily_manifest_path=daily, output_path=access, client=client, sleep=lambda _: None)
    assert rows[0]["probe_result"] == "FAIL_CLOSED_IDENTITY_BOUNDARY"
    assert client.calls == []


def test_empty_hourly_probe_fails_and_does_not_authorize_download(tmp_path: Path):
    data_root, daily, access = tmp_path / "data", tmp_path / "daily.csv", tmp_path / "access.csv"
    days = [date(2023, 1, 3)]
    _daily_inputs(data_root, daily, days)
    snapshots = tmp_path / "instruments"
    _snapshot(snapshots)
    client = _FakeHourlyClient(days, empty=True)
    rows = probe_hourly_access(start_date=days[0], end_date=days[0], warmup_start=date(2022, 10, 1),
                               provider=_HourlyProvider(), snapshot_directory=snapshots, data_root=data_root,
                               daily_manifest_path=daily, output_path=access, client=client, sleep=lambda _: None)
    assert rows[0]["probe_result"] == "FAIL_HISTORICAL_DATA"
    assert rows[0]["failure_detail"]


def test_audit_and_backtest_hourly_loader(tmp_path: Path):
    # Reuse a tiny deterministic fixture, then verify diagnostics and dated loading.
    data_root = tmp_path / "data"
    hourly_manifest = tmp_path / "hourly.csv"
    session_day = date(2023, 1, 3)
    daily_manifest = tmp_path / "daily.csv"
    _daily_inputs(data_root, daily_manifest, [session_day])
    relative = Path("NIFTY50_PIT_TRADABLE/60minute/SEC-1/SYMBOL.csv")
    file_path = data_root / relative
    file_path.parent.mkdir(parents=True)
    with file_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("date", "open", "high", "low", "close", "volume"))
        writer.writeheader()
        writer.writerow({"date": "2023-01-03T09:15:00+05:30", "open": 10, "high": 11,
                         "low": 9, "close": 10, "volume": 100})
    with hourly_manifest.open("w", encoding="utf-8", newline="") as handle:
        fields = ("universe_id", "security_id", "source", "broker", "instrument_token", "mapping_source_id",
                  "mapping_effective_from", "mapping_effective_to", "symbol_used", "interval", "requested_start",
                  "requested_end", "actual_first_timestamp", "actual_last_timestamp", "row_count",
                  "missing_session_diagnostics", "downloaded_at", "content_sha256", "provenance",
                  "validation_status", "relative_path")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({"universe_id": "NIFTY50_PIT_TRADABLE", "security_id": "SEC-1", "interval": "60minute",
                         "validation_status": "VALIDATED", "actual_first_timestamp": "2023-01-03T09:15:00+05:30",
                         "relative_path": relative.as_posix()})

    class Provider:
        def members_over(self, start, end, universe_id):
            return [MembershipInterval("SEC-1", universe_id, session_day, None,
                                       None, "STANDARD", True, "membership-source")]

    audit = audit_hourly_dataset(start_date=session_day, end_date=session_day, provider=Provider(),
                                 data_root=data_root, manifest_path=hourly_manifest,
                                 daily_manifest_path=daily_manifest)
    assert audit["pit_securities_expected"] == 1
    assert audit["securities_with_hourly_data"] == 1
    assert audit["securities"]["SEC-1"]["row_count"] == 1
    assert audit["normal_session_rows"] == 1
    assert audit["unexpected_normal_grid_rows"] == 0
    assert audit["dataset_status"] == "BUILT"
    loaded, digest = load_hourly_dataset_through(
        universe_id="NIFTY50_PIT_TRADABLE", start_date=session_day, end_date=session_day,
        data_root=data_root, manifest_path=hourly_manifest, daily_manifest_path=daily_manifest,
    )
    assert len(loaded["SEC-1"]) == 1
    assert len(digest) == 64

    with hourly_manifest.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writerow({"universe_id": "NIFTY50_PIT_TRADABLE", "security_id": "SEC-2", "interval": "60minute",
                         "validation_status": "VALIDATED", "actual_first_timestamp": "2023-01-03T09:15:00+05:30",
                         "relative_path": relative.as_posix()})

    excluded_identities = tmp_path / "invalid-hourly-identities.csv"
    with excluded_identities.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("security_id", "identity_status"))
        writer.writeheader()
        writer.writerow({"security_id": "SEC-1", "identity_status": "HOURLY_IDENTITY_PRICE_MISMATCH"})
    filtered, _ = load_hourly_dataset_through(
        universe_id="NIFTY50_PIT_TRADABLE", start_date=session_day, end_date=session_day,
        data_root=data_root, manifest_path=hourly_manifest, daily_manifest_path=daily_manifest,
        identity_exclusions_path=excluded_identities)
    assert "SEC-1" not in filtered
    assert "SEC-2" in filtered
    raw_included, _ = load_hourly_dataset_through(
        universe_id="NIFTY50_PIT_TRADABLE", start_date=session_day, end_date=session_day,
        data_root=data_root, manifest_path=hourly_manifest, daily_manifest_path=daily_manifest,
        identity_exclusions_path=excluded_identities, include_flagged_identities=True)
    assert len(raw_included["SEC-1"]) == 1

    # The audit reads raw order and duplicates instead of normalizing them away.
    with file_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("date", "open", "high", "low", "close", "volume"))
        writer.writeheader()
        for stamp in ("2023-01-03T10:15:00+05:30", "2023-01-03T09:15:00+05:30",
                      "2023-01-03T09:15:00+05:30"):
            writer.writerow({"date": stamp, "open": 10, "high": 11, "low": 9, "close": 10, "volume": 100})
    audited = audit_hourly_dataset(start_date=session_day, end_date=session_day, provider=Provider(),
                                   data_root=data_root, manifest_path=hourly_manifest,
                                   daily_manifest_path=daily_manifest)
    assert audited["securities"]["SEC-1"]["duplicate_timestamps"] == 1
    assert audited["securities"]["SEC-1"]["non_monotonic_timestamps"] == 1


def test_audit_distinguishes_not_built_from_incomplete_and_explicit_data_gap(tmp_path: Path):
    start = date(2023, 1, 3)

    class Provider:
        def members_over(self, first, last, universe_id):
            return [MembershipInterval(f"SEC-{number}", universe_id, start, None,
                                       None, "STANDARD", True, "membership-source")
                    for number in (1, 2)]

    missing_manifest = tmp_path / "not-built.csv"
    not_built = audit_hourly_dataset(start_date=start, end_date=start, provider=Provider(),
                                     data_root=tmp_path / "data", manifest_path=missing_manifest)
    assert not_built["dataset_status"] == "DATASET_NOT_BUILT"
    assert set(not_built["dataset_not_built_securities"]) == {"SEC-1", "SEC-2"}

    partial_manifest = tmp_path / "partial.csv"
    with partial_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("universe_id", "security_id", "interval",
                                                    "requested_start", "requested_end",
                                                    "validation_status", "provenance"))
        writer.writeheader()
        writer.writerow({"universe_id": "NIFTY50_PIT_TRADABLE", "security_id": "SEC-1",
                         "interval": "60minute", "requested_start": start.isoformat(),
                         "requested_end": start.isoformat(), "validation_status": "NO_KITE_HISTORICAL_ACCESS",
                         "provenance": "No current token"})
    incomplete = audit_hourly_dataset(start_date=start, end_date=start, provider=Provider(),
                                      data_root=tmp_path / "data", manifest_path=partial_manifest)
    assert incomplete["dataset_status"] == "BUILD_INCOMPLETE"
    assert incomplete["securities"]["SEC-1"]["data_status"] == "MISSING_HOURLY_DATA"
    assert incomplete["securities"]["SEC-2"]["data_status"] == "BUILD_INCOMPLETE"


def test_failed_security_build_does_not_persist_unvalidated_candles(tmp_path: Path):
    day = date(2023, 1, 3)
    data_root, daily, access = tmp_path / "data", tmp_path / "daily.csv", tmp_path / "access.csv"
    _daily_inputs(data_root, daily, [day])
    snapshots = tmp_path / "instruments"
    _snapshot(snapshots)
    with access.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ACCESS_FIELDS)
        writer.writeheader()
        writer.writerow({
            "universe_id": "NIFTY50_PIT_TRADABLE", "security_id": "SEC-1",
            "historical_symbol": "OLD", "symbol_effective_from": "2020-01-01",
            "evaluation_start": day.isoformat(), "evaluation_end": day.isoformat(),
            "warmup_start": day.isoformat(), "requested_start": day.isoformat(),
            "requested_end": day.isoformat(), "snapshot_date": "2026-09-23",
            "current_symbol": "NEW", "current_resolved_token": "987", "instrument_token": "987",
            "probe_date": day.isoformat(), "probe_result": "PASS_HISTORICAL_DATA",
            "market_data_access_result": "PASS_HISTORICAL_DATA",
            "market_data_access_provenance": "CURRENT_TOKEN_HISTORICAL_PROBE",
        })

    class OffGridClient:
        def historical_data(self, token, first, last, interval):
            return [_bar(datetime.fromisoformat("2023-01-03T10:45:00+05:30"))]

    hourly_manifest = tmp_path / "hourly.csv"
    with pytest.raises(HourlyDataError, match="distinct_intraday_times"):
        build_hourly_dataset(
            start_date=day, end_date=day, warmup_start=day, provider=_HourlyProvider(),
            client=OffGridClient(), data_root=data_root, manifest_path=hourly_manifest,
            daily_manifest_path=daily, access_manifest_path=access, snapshot_directory=snapshots,
            sleep=lambda _: None,
        )
    expected_file = data_root / "NIFTY50_PIT_TRADABLE" / "60minute" / "SEC-1" / "OLD_987.csv"
    assert not expected_file.exists()
    assert not hourly_manifest.exists()


def test_completed_security_is_persisted_before_later_security_failure(tmp_path: Path):
    day = date(2023, 1, 3)
    data_root, daily, access = tmp_path / "data", tmp_path / "daily.csv", tmp_path / "access.csv"
    _daily_inputs(data_root, daily, [day])
    snapshots = tmp_path / "instruments"
    _snapshot(snapshots)
    with access.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ACCESS_FIELDS)
        writer.writeheader()
        for security_id in ("SEC-1", "SEC-2"):
            writer.writerow({
                "universe_id": "NIFTY50_PIT_TRADABLE", "security_id": security_id,
                "historical_symbol": "OLD", "symbol_effective_from": "2020-01-01",
                "evaluation_start": day.isoformat(), "evaluation_end": day.isoformat(),
                "warmup_start": day.isoformat(), "requested_start": day.isoformat(),
                "requested_end": day.isoformat(), "snapshot_date": "2026-09-23",
                "current_symbol": "NEW", "current_resolved_token": "987", "instrument_token": "987",
                "probe_date": day.isoformat(), "probe_result": "PASS_HISTORICAL_DATA",
                "market_data_access_result": "PASS_HISTORICAL_DATA",
                "market_data_access_provenance": "CURRENT_TOKEN_HISTORICAL_PROBE",
            })

    class TwoSecurityProvider(_HourlyProvider):
        def members_over(self, start, end, universe_id):
            return [MembershipInterval(security_id, universe_id, date(2020, 1, 1), None,
                                       None, "STANDARD", True, "membership-source")
                    for security_id in ("SEC-1", "SEC-2")]

    class FailSecondClient:
        def __init__(self):
            self.calls = 0

        def historical_data(self, token, first, last, interval):
            self.calls += 1
            if self.calls == 2:
                return [_bar(datetime.fromisoformat("2023-01-03T09:30:00+05:30"))]
            return [_bar(datetime.combine(day, time(hour, 15))) for hour in range(9, 16)]

    manifest = tmp_path / "hourly.csv"
    with pytest.raises(HourlyDataError, match="outside the NSE hourly session grid"):
        build_hourly_dataset(
            start_date=day, end_date=day, warmup_start=day, provider=TwoSecurityProvider(),
            client=FailSecondClient(), data_root=data_root, manifest_path=manifest,
            daily_manifest_path=daily, access_manifest_path=access, snapshot_directory=snapshots,
            sleep=lambda _: None,
        )
    first_file = data_root / "NIFTY50_PIT_TRADABLE" / "60minute" / "SEC-1" / "OLD_987.csv"
    second_file = data_root / "NIFTY50_PIT_TRADABLE" / "60minute" / "SEC-2" / "OLD_987.csv"
    assert first_file.is_file()
    assert not second_file.exists()
    persisted = list(csv.DictReader(manifest.open(encoding="utf-8", newline="")))
    assert [row["security_id"] for row in persisted] == ["SEC-1"]

    # A rerun skips SEC-1 completely and resumes at SEC-2.
    retry = FailSecondClient()
    resumed = build_hourly_dataset(
        start_date=day, end_date=day, warmup_start=day, provider=TwoSecurityProvider(),
        client=retry, data_root=data_root, manifest_path=manifest,
        daily_manifest_path=daily, access_manifest_path=access, snapshot_directory=snapshots,
        sleep=lambda _: None,
    )
    assert retry.calls == 1
    assert resumed["reused_securities"] == ["SEC-1"]
    assert (data_root / "NIFTY50_PIT_TRADABLE" / "60minute" / "SEC-2" / "OLD_987.csv").is_file()


@pytest.mark.parametrize("retry_is_valid", [True, False])
def test_malformed_provider_candle_is_recovered_or_quarantined_and_counted(tmp_path: Path, retry_is_valid: bool):
    days = _weekday_dates(date(2023, 1, 3), 143)  # 1 bad / 1001 source candles stays below 0.1%.
    data_root, daily_manifest, access_manifest = _build_inputs(tmp_path, days)
    snapshots = tmp_path / "instruments"
    manifest = tmp_path / "hourly.csv"
    quarantine = tmp_path / "hourly" / "quarantine_invalid_candles.csv"
    bad_day = days[35]
    bad_stamp = datetime.combine(bad_day, time(9, 15))

    def candles(start: datetime, end: datetime, *, bad: bool) -> list[dict]:
        rows = []
        for day in days:
            if start.date() <= day <= end.date():
                for hour in range(9, 16):
                    row = _bar(datetime.combine(day, time(hour, 15)))
                    if bad and day == bad_day and hour == 9:
                        row.update(open=1434, high=1430.6, low=1420.65, close=1423, volume=1493823)
                    elif not bad and day == bad_day and hour == 9:
                        row.update(open=1434, high=1436, low=1420.65, close=1423, volume=1493823)
                    rows.append(row)
        return rows

    class RefetchClient:
        def __init__(self):
            self.calls = []

        def historical_data(self, token, start, end, interval):
            self.calls.append((start.date(), end.date()))
            if len(self.calls) == 1:
                return candles(start, end, bad=True)
            return candles(start, end, bad=not retry_is_valid)

    client = RefetchClient()
    result = build_hourly_dataset(
        start_date=days[0], end_date=days[-1], warmup_start=days[0], provider=_HourlyProvider(),
        client=client, data_root=data_root, manifest_path=manifest,
        daily_manifest_path=daily_manifest, access_manifest_path=access_manifest,
        snapshot_directory=snapshots, sleep=lambda _: None, quarantine_path=quarantine,
    )
    assert client.calls[0] == (days[0], days[-1])
    assert client.calls[1] == (bad_day, bad_day)
    output_file = data_root / "NIFTY50_PIT_TRADABLE" / "60minute" / "SEC-1" / "OLD_987.csv"
    rows = list(csv.DictReader(output_file.open(encoding="utf-8", newline="")))
    bad_stamp_local = bad_stamp.replace(tzinfo=ZoneInfo("Asia/Kolkata")).isoformat()
    bad_rows = [row for row in rows if row["date"] == bad_stamp_local]
    manifest_row = next(row for row in csv.DictReader(manifest.open(encoding="utf-8", newline="")))
    assert result["invalid_provider_candles"] == 1
    if retry_is_valid:
        assert result["recovered_by_refetch"] == 1
        assert result["quarantined_candles"] == 0
        assert bad_rows[0]["high"] == "1436.0"
        assert bad_rows[0]["data_quality_status"] == "RECOVERED_BY_REFETCH"
        assert not quarantine.exists()
        assert manifest_row["recovered_by_refetch"] == "1"
        assert manifest_row["missing_hourly_bars_due_to_quarantine"] == "0"
    else:
        assert result["recovered_by_refetch"] == 0
        assert result["quarantined_candles"] == 1
        assert not bad_rows
        assert manifest_row["quarantined_candles"] == "1"
        assert manifest_row["missing_hourly_bars_due_to_quarantine"] == "1"
        quarantine_row = next(row for row in csv.DictReader(quarantine.open(encoding="utf-8", newline="")))
        assert quarantine_row["security_id"] == "SEC-1"
        assert quarantine_row["symbol"] == "OLD"
        assert quarantine_row["token"] == "987"
        assert quarantine_row["timestamp"] == bad_stamp_local
        assert quarantine_row["failed_invariant"] == "high < open"
        assert quarantine_row["retry_result"] == "PERSISTENT_INVALID_CANDLE"
        assert quarantine_row["source"] == "KITE"
        assert quarantine_row["retry_count"] == "1"
        assert quarantine_row["data_quality_status"] == "PROVIDER_INVALID_OHLC"
        assert "1430.6" in quarantine_row["refetched_raw_response"]
        assert hourly_window_coverage_status(
            security_id="SEC-1", start=bad_stamp, end=bad_stamp, quarantine_path=quarantine,
        ) == "PARTIAL_HOURLY_DATA"
        assert hourly_window_coverage_status(
            security_id="SEC-1", start=bad_stamp, end=bad_stamp, quarantine_path=quarantine,
            available_timestamps=[bad_stamp_local],
        ) == "COMPLETE_HOURLY_DATA"
        assert result["downloads"]
        audit = audit_hourly_dataset(
            start_date=days[0], end_date=days[-1], provider=_HourlyProvider(), data_root=data_root,
            manifest_path=manifest, daily_manifest_path=daily_manifest, quarantine_path=quarantine,
        )
        assert audit["invalid_provider_candles"] == 1
        assert audit["quarantined_candles"] == 1
        assert audit["recovered_by_refetch"] == 0
        assert audit["missing_hourly_bars_due_to_quarantine"] == 1


def test_excessive_corruption_marks_only_security_failed_and_continues(tmp_path: Path):
    days = _weekday_dates(date(2023, 1, 3), 143)
    data_root, daily_manifest, access_manifest = _build_inputs(tmp_path, days)
    manifest = tmp_path / "hourly.csv"
    quarantine = tmp_path / "hourly" / "quarantine_invalid_candles.csv"
    bad_day = days[35]

    def response(start: datetime, end: datetime, *, invalid: bool) -> list[dict]:
        rows = []
        for day in days:
            if start.date() <= day <= end.date():
                for hour in range(9, 16):
                    row = _bar(datetime.combine(day, time(hour, 15)))
                    if invalid and day == bad_day and hour in (9, 10, 11, 12):
                        row["high"] = row["open"] - 1
                    rows.append(row)
        return rows

    class ExcessiveClient:
        def __init__(self):
            self.calls = 0

        def historical_data(self, token, start, end, interval):
            self.calls += 1
            return response(start, end, invalid=True)

    client = ExcessiveClient()
    result = build_hourly_dataset(
        start_date=days[0], end_date=days[-1], warmup_start=days[0], provider=_HourlyProvider(),
        client=client, data_root=data_root, manifest_path=manifest,
        daily_manifest_path=daily_manifest, access_manifest_path=access_manifest,
        snapshot_directory=tmp_path / "instruments", sleep=lambda _: None, quarantine_path=quarantine,
    )
    assert "FAILED_CORRUPTION_THRESHOLD" in result["access_failures"][0]
    assert result["invalid_provider_candles"] == 4
    assert result["quarantined_candles"] == 4
    assert not (data_root / "NIFTY50_PIT_TRADABLE" / "60minute" / "SEC-1" / "OLD_987.csv").exists()
    manifest_row = next(row for row in csv.DictReader(manifest.open(encoding="utf-8", newline="")))
    assert manifest_row["validation_status"] == "FAILED_CORRUPTION_THRESHOLD"
    assert manifest_row["invalid_provider_candles"] == "4"
    assert manifest_row["quarantined_candles"] == "4"
