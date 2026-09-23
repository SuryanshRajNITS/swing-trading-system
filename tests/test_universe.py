from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from src.data.universe.provider import (
    CURRENT,
    PIT_INDEX,
    PIT_TRADABLE,
    UniverseCoverageError,
    UniverseProvider,
)


@pytest.fixture
def universe() -> UniverseProvider:
    return UniverseProvider()


def symbols(provider: UniverseProvider, day: str, universe_id: str = PIT_TRADABLE) -> set[str]:
    return {security.symbol for security in provider.members_on(date.fromisoformat(day), universe_id)}


def test_initial_roster_and_normal_membership_count(universe: UniverseProvider) -> None:
    members = universe.members_on(date(2023, 1, 3))
    assert len(members) == 50
    assert {member.symbol for member in members} == {
        "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK", "BAJAJ-AUTO",
        "BAJFINANCE", "BAJAJFINSV", "BPCL", "BHARTIARTL", "BRITANNIA", "CIPLA", "COALINDIA",
        "DIVISLAB", "DRREDDY", "EICHERMOT", "GRASIM", "HCLTECH", "HDFCBANK", "HDFCLIFE", "HDFC",
        "HEROMOTOCO", "HINDALCO", "HINDUNILVR", "ICICIBANK", "INDUSINDBK", "INFY", "ITC",
        "JSWSTEEL", "KOTAKBANK", "LT", "M&M", "MARUTI", "NESTLEIND", "NTPC", "ONGC", "POWERGRID",
        "RELIANCE", "SBILIFE", "SBIN", "SUNPHARMA", "TCS", "TATACONSUM", "TATAMOTORS", "TATASTEEL",
        "TECHM", "TITAN", "UPL", "ULTRACEMCO", "WIPRO",
    }
    assert all(member.universe_id == PIT_TRADABLE for member in members)


def test_hdfc_ltim_effective_boundary(universe: UniverseProvider) -> None:
    assert "HDFC" in symbols(universe, "2023-07-12")
    assert "LTIM" not in symbols(universe, "2023-07-12")
    assert "HDFC" not in symbols(universe, "2023-07-13")
    assert "LTIM" in symbols(universe, "2023-07-13")
    assert len(symbols(universe, "2023-07-13")) == 50


def test_jio_demerger_index_period_and_tradability(universe: UniverseProvider) -> None:
    assert "JIOFIN" not in symbols(universe, "2023-07-19", PIT_INDEX)
    assert "JIOFIN" in symbols(universe, "2023-07-20", PIT_INDEX)
    assert "JIOFIN" not in symbols(universe, "2023-07-20", PIT_TRADABLE)
    assert "JIOFIN" in symbols(universe, "2023-08-21", PIT_TRADABLE)
    assert "JIOFIN" not in symbols(universe, "2023-09-07", PIT_TRADABLE)
    assert "JIOFIN" in symbols(universe, "2025-03-28", PIT_TRADABLE)


def test_scheduled_reconstitution_boundaries(universe: UniverseProvider) -> None:
    assert "UPL" in symbols(universe, "2024-03-27")
    assert "SHRIRAMFIN" not in symbols(universe, "2024-03-27")
    assert "UPL" not in symbols(universe, "2024-03-28")
    assert "SHRIRAMFIN" in symbols(universe, "2024-03-28")

    assert {"DIVISLAB", "LTIM"} <= symbols(universe, "2024-09-29")
    assert {"BEL", "TRENT"} <= symbols(universe, "2024-09-30")
    assert not ({"DIVISLAB", "LTIM"} & symbols(universe, "2024-09-30"))

    assert {"BPCL", "BRITANNIA"} <= symbols(universe, "2025-03-27")
    assert {"JIOFIN", "ZOMATO"} <= symbols(universe, "2025-03-28")
    assert not ({"BPCL", "BRITANNIA"} & symbols(universe, "2025-03-28"))

    assert {"HEROMOTOCO", "INDUSINDBK"} <= symbols(universe, "2025-09-29")
    assert {"INDIGO", "MAXHEALTH"} <= symbols(universe, "2025-09-30")
    assert len(symbols(universe, "2025-09-30")) == 50


def test_zomato_eternal_is_one_security_with_dated_symbols(universe: UniverseProvider) -> None:
    before = next(item for item in universe.members_on(date(2025, 4, 8)) if item.symbol == "ZOMATO")
    after = next(item for item in universe.members_on(date(2025, 4, 9)) if item.symbol == "ETERNAL")
    assert before.security_id == after.security_id
    assert before.company_name == "Zomato Ltd."
    assert after.company_name == "Eternal Ltd."
    assert universe.symbol_for(before.security_id, date(2025, 4, 8)) == "ZOMATO"
    assert universe.symbol_for(before.security_id, date(2025, 4, 9)) == "ETERNAL"


def test_ltim_ltm_symbol_history_is_independent_of_universe_membership(universe: UniverseProvider) -> None:
    security_id = "NSEEQ-0041"
    assert universe.symbol_for(security_id, date(2024, 9, 30)) == "LTIM"
    assert universe.symbol_for(security_id, date(2026, 2, 26)) == "LTIM"
    assert universe.symbol_for(security_id, date(2026, 2, 27)) == "LTM"
    assert universe.symbol_for(security_id, date(2026, 4, 16)) == "LTM"


def test_tata_motors_name_and_symbol_change(universe: UniverseProvider) -> None:
    previous = next(item for item in universe.members_on(date(2025, 10, 23)) if item.symbol == "TATAMOTORS")
    current = next(item for item in universe.members_on(date(2025, 10, 24)) if item.symbol == "TMPV")
    assert previous.security_id == current.security_id
    assert previous.company_name == "Tata Motors Ltd."
    assert current.company_name == "Tata Motors Passenger Vehicles Ltd."


def test_change_ledger_preserves_announcement_and_effective_dates(universe: UniverseProvider) -> None:
    dummy_id = next(
        item.security_id for item in universe.members_on(date(2025, 1, 6), PIT_INDEX)
        if item.symbol == "DUMMYITC"
    )
    itc_dummy = next(
        event for event in universe.changes_on(date(2025, 1, 6), PIT_INDEX)
        if event.security_id == dummy_id
    )
    assert itc_dummy.announcement_date == date(2024, 12, 30)
    assert itc_dummy.effective_date == date(2025, 1, 6)

    ticker_change = next(
        event for event in universe.changes_on(date(2025, 10, 24))
        if event.event_type == "SYMBOL_CHANGED" and event.from_symbol == "TATAMOTORS"
    )
    assert ticker_change.announcement_date == date(2025, 10, 16)
    assert ticker_change.to_symbol == "TMPV"

    review = universe.changes_on(date(2026, 3, 30))
    assert len(review) == 1
    assert review[0].event_type == "REVIEW_NO_CHANGE"


def test_corporate_action_placeholders_are_index_only(universe: UniverseProvider) -> None:
    assert "DUMMYITC" in symbols(universe, "2025-01-06", PIT_INDEX)
    assert "DUMMYITC" not in symbols(universe, "2025-01-06", PIT_TRADABLE)
    assert "ITCHOTELS" in symbols(universe, "2025-01-29", PIT_TRADABLE)
    assert "ITCHOTELS" not in symbols(universe, "2025-02-10", PIT_TRADABLE)
    itc_hotels = next(item for item in universe.members_on(date(2025, 1, 29)) if item.symbol == "ITCHOTELS")
    assert itc_hotels.isin == "INE379A01028"

    assert "DUMMYTATAM" in symbols(universe, "2025-10-14", PIT_INDEX)
    assert "DUMMYTATAM" not in symbols(universe, "2025-10-14", PIT_TRADABLE)
    assert "TMCV" in symbols(universe, "2025-11-12", PIT_TRADABLE)
    assert "TMCV" not in symbols(universe, "2025-11-17", PIT_TRADABLE)

    assert "DUMMYHDLVR" in symbols(universe, "2025-12-05", PIT_INDEX)
    assert "DUMMYHDLVR" not in symbols(universe, "2025-12-05", PIT_TRADABLE)
    assert "KWIL" in symbols(universe, "2026-02-16", PIT_TRADABLE)
    assert "KWIL" not in symbols(universe, "2026-02-24", PIT_TRADABLE)


def test_oos_cutoff_and_future_scheduled_replacement(universe: UniverseProvider) -> None:
    assert "WIPRO" in symbols(universe, "2026-09-22")
    assert "BSE" not in symbols(universe, "2026-09-22")
    assert "WIPRO" not in symbols(universe, "2026-09-30")
    assert "BSE" in symbols(universe, "2026-09-30")
    assert len(symbols(universe, "2026-09-30")) == 50


def test_current_universe_is_explicit_and_never_a_fallback(universe: UniverseProvider) -> None:
    # Pin the fixture's known current-snapshot date instead of relying on wall-clock time.
    current = universe.current_members(date(2026, 9, 23))
    assert len(current) == 50
    assert all(item.universe_id == CURRENT for item in current)
    assert {item.security_id for item in current} == {
        item.security_id for item in universe.members_on(date(2026, 9, 22), PIT_TRADABLE)
        if item.membership_type == "NORMAL"
    }
    with pytest.raises(UniverseCoverageError, match="no fallback"):
        universe.members_on(date(2025, 3, 28), CURRENT)
    with pytest.raises(UniverseCoverageError, match="no fallback"):
        universe.current_members(date(2026, 9, 24))
    with pytest.raises(UniverseCoverageError, match="no fallback"):
        universe.members_on(date(2020, 1, 1), PIT_TRADABLE)


def test_broker_mapping_only_covers_snapshot_date(universe: UniverseProvider) -> None:
    tmpv = next(item for item in universe.members_on(date(2026, 9, 23), CURRENT) if item.symbol == "TMPV")
    mapped = universe.broker_instrument_for(tmpv.security_id, date(2026, 9, 23))
    assert mapped.tradingsymbol == "TMPV"
    with pytest.raises(UniverseCoverageError, match="No unique kite/NSE"):
        universe.broker_instrument_for(tmpv.security_id, date(2026, 9, 22))


def test_audit_returns_historical_gaps_without_guessing(universe: UniverseProvider) -> None:
    unresolved = universe.unresolved_broker_mappings(date(2023, 1, 1), date(2026, 9, 22))
    assert unresolved
    assert all(start <= end for _, _, start, end in unresolved)
    assert all(end <= date(2026, 9, 22) for _, _, _, end in unresolved)
    tmpv = [row for row in unresolved if row[0] == next(
        member.security_id for member in universe.members_on(date(2025, 10, 23))
        if member.symbol == "TATAMOTORS"
    )]
    assert ("TATAMOTORS", date(2025, 10, 23)) in {(row[1], row[3]) for row in tmpv}
    assert ("TMPV", date(2026, 9, 22)) in {(row[1], row[3]) for row in tmpv}


def test_members_over_honors_inclusive_requested_dates(universe: UniverseProvider) -> None:
    intervals = universe.members_over(date(2023, 7, 12), date(2023, 7, 13))
    hdfc_id = next(item.security_id for item in universe.members_on(date(2023, 7, 12)) if item.symbol == "HDFC")
    hdfc = [row for row in intervals if row.security_id == hdfc_id]
    assert len(hdfc) == 1
    assert hdfc[0].effective_to == date(2023, 7, 13)


def test_snapshot_importer_is_date_bounded_and_idempotent(tmp_path: Path) -> None:
    import csv

    from src.data.universe.snapshot import import_snapshot

    reference = tmp_path / "reference"
    reference.mkdir()
    for filename in ("symbol_history.csv", "broker_instrument_history.csv", "sources.csv"):
        source = Path("data/reference") / filename
        (reference / filename).write_bytes(source.read_bytes())
    snapshot = tmp_path / "nse_instruments_2026-09-23.csv"
    with snapshot.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["exchange", "instrument_type", "tradingsymbol", "instrument_token"])
        writer.writeheader()
        writer.writerow({"exchange": "NSE", "instrument_type": "EQ", "tradingsymbol": "INFY", "instrument_token": 111})
    assert import_snapshot(snapshot, reference) == 1
    assert import_snapshot(snapshot, reference) == 1
    rows = list(csv.DictReader((reference / "broker_instrument_history.csv").open(encoding="utf-8", newline="")))
    assert len(rows) == 1
    assert rows[0]["effective_from"] == "2026-09-23"
    assert rows[0]["effective_to"] == "2026-09-24"
