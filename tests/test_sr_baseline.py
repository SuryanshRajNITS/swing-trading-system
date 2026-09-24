from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from src.decision.interfaces import EntrySignal
from src.risk.position_sizing import FixedFractionRiskPolicy
from src.strategy.sr_baseline import (
    PriceZone, SRBaselineConfig, SRBaselineStrategy, SwingPoint, _is_broken,
    _weekly_filter, cluster_zones, find_confirmed_swings,
)


def _row(day: int, *, high: float, low: float, close: float, volume: int = 100) -> dict:
    return {"date": f"2023-01-{day:02d}T00:00:00+05:30", "open": close,
            "high": high, "low": low, "close": close, "volume": volume}


def _support_rows() -> list[dict]:
    return [
        _row(1, high=106, low=104, close=105),
        _row(2, high=102, low=100, close=101),
        _row(3, high=105, low=103, close=104),
        _row(4, high=107, low=104, close=106),
        _row(5, high=102, low=100, close=101),
        _row(6, high=105, low=103, close=104),
        _row(7, high=111, low=106, close=110),
        _row(8, high=112, low=108, close=110),
        _row(9, high=102, low=100, close=101.5),
    ]


def _breakout_rows(*, retest: bool = True) -> list[dict]:
    rows = [
        _row(1, high=105, low=103, close=104),
        _row(2, high=110, low=107, close=108),
        _row(3, high=106, low=104, close=105),
        _row(4, high=108, low=105, close=107),
        _row(5, high=110, low=107, close=109),
        _row(6, high=106, low=103, close=104),
        _row(7, high=108, low=105, close=107),
        _row(8, high=113, low=108, close=112),
        _row(9, high=114, low=112, close=113),
    ]
    if retest:
        rows.append(_row(10, high=112, low=109, close=111.5))
    else:
        rows.extend([_row(10, high=118, low=114, close=117),
                     _row(11, high=120, low=117, close=119),
                     _row(12, high=121, low=118, close=120)])
    return rows


def _context(rows: list[dict], day: date | None = None):
    session = day or date(2023, 1, len(rows))
    timestamp = datetime.combine(session, datetime.min.time(), tzinfo=ZoneInfo("Asia/Kolkata")).replace(hour=15, minute=30)
    sec = SimpleNamespace(security_id="SEC1", symbol="TEST")
    return SimpleNamespace(
        session_date=session, timestamp=timestamp,
        next_session_open=timestamp.replace(day=min(28, session.day + 1), hour=9, minute=15),
        latest_completed_weekly={"SEC1": {"close": 120.0}},
        daily_history={"SEC1": rows}, eligible=(sec,), open_security_ids=frozenset(),
    )


def test_swings_are_unavailable_until_right_confirmation_bars_exist():
    rows = [_row(i+1, high=value, low=1, close=10) for i, value in enumerate([2, 3, 8, 4, 2])]
    assert find_confirmed_swings(rows[:4], left=2, right=2) == []
    swings = find_confirmed_swings(rows, left=2, right=2)
    assert [(swing.kind, swing.pivot_index, swing.confirmation_index) for swing in swings] == [
        ("resistance", 2, 4),
    ]


def test_same_side_pivots_cluster_into_percentage_zones():
    swings = [SwingPoint("support", 2, 4, 100), SwingPoint("support", 8, 10, 100.5),
              SwingPoint("resistance", 3, 5, 120)]
    zones = cluster_zones(swings, tolerance_pct=0.01)
    support = next(zone for zone in zones if zone.kind == "support")
    assert support.pivot_count == 2
    assert support.center == pytest.approx(100.25)
    assert support.lower < support.center < support.upper


def test_support_second_touch_requires_reaction_and_structural_stop():
    strategy = SRBaselineStrategy(config=SRBaselineConfig(swing_left=1, swing_right=1,
        zone_tolerance_pct=0.01, minimum_touch_separation_sessions=2,
        support_min_close_location=0.6), setup="support_test", target_r=2)
    rows = _support_rows()
    context = _context(rows)
    signal = strategy._support_signal("SEC1", "TEST", rows, context)
    assert signal is not None
    assert signal.stop_price < 99
    assert signal.metadata["days_since_previous_touch"] >= 2
    assert signal.metadata["touch_count"] >= 3
    assert signal.metadata["rejection_strength"] >= 0.6


def test_decisively_broken_support_is_not_valid():
    zone = PriceZone("support", 100, 99, 101, 2, 0, 5)
    rows = [_row(1, high=102, low=98, close=98)]
    assert _is_broken(rows, zone, 0.005)
    strategy = SRBaselineStrategy(config=SRBaselineConfig(swing_left=1, swing_right=1,
        minimum_touch_separation_sessions=2), setup="support_test")
    assert strategy._support_signal("SEC1", "TEST", _support_rows() + rows, _context(_support_rows() + rows)) is None


def test_breakout_retest_emits_only_after_separate_retest_and_freezes_level():
    strategy = SRBaselineStrategy(config=SRBaselineConfig(swing_left=1, swing_right=1,
        zone_tolerance_pct=0.01, breakout_distance_pct=0.001,
        retest_tolerance_pct=0.01, maximum_retest_sessions=5),
        setup="breakout_retest", target_r=2)
    rows = _breakout_rows()
    signal = strategy._advance_breakouts("SEC1", rows, context=_context(rows), symbol="TEST", collect_signal=True)
    assert signal is not None
    assert signal.metadata["breakout_date"] != signal.metadata["retest_date"]
    assert signal.metadata["retest_age"] >= 1
    assert signal.stop_price < signal.metadata["zone"]["lower"]
    assert signal.metadata["breakout_level"] == pytest.approx(111.1)


def test_breakout_retest_timeout_is_reported():
    strategy = SRBaselineStrategy(config=SRBaselineConfig(swing_left=1, swing_right=1,
        zone_tolerance_pct=0.01, breakout_distance_pct=0.001,
        maximum_retest_sessions=2), setup="breakout_retest")
    rows = _breakout_rows(retest=False)
    for end in range(1, len(rows)+1):
        prefix = rows[:end]
        strategy._advance_breakouts("SEC1", prefix, context=_context(prefix), symbol="TEST", collect_signal=True)
    assert any(row.get("rejection_reason") == "RETEST_TIMEOUT" for row in strategy.diagnostics)


def test_risk_policy_zeroes_unaffordable_structural_distance():
    decision = FixedFractionRiskPolicy().size(equity=10_000, available_cash=10_000,
                                               entry_price=150, stop_price=140, risk_fraction=0.01)
    assert decision.quantity == 10
    too_wide = FixedFractionRiskPolicy().size(equity=10_000, available_cash=10_000,
                                               entry_price=150, stop_price=120, risk_fraction=0.01)
    assert too_wide.quantity == 3
    assert too_wide.quantity * 30 <= 100


def test_next_session_signal_contract_and_target_r_are_explicit():
    rows = _support_rows()
    context = _context(rows)
    signal = EntrySignal("SEC1", "TEST", context.timestamp, context.next_session_open,
                         98.0, metadata={"target_r": 2.5}, strategy_version="sr_baseline_v1")
    assert signal.execution_timestamp == context.next_session_open
    assert signal.metadata["target_r"] == 2.5


def test_weekly_filter_uses_only_prior_calendar_weeks_even_on_friday():
    # A sharply rising current week must not affect the weekly filter reference.
    rows = []
    current = date(2023, 1, 2)
    for close in [100]*5 + [90]*5 + [500, 510, 520, 530, 540]:
        while current.weekday() >= 5:
            current = date.fromordinal(current.toordinal() + 1)
        rows.append({"date": current.isoformat()+"T00:00:00+05:30", "open": close-1,
                     "high": close+1, "low": close-2, "close": close, "volume": 100})
        current = date.fromordinal(current.toordinal() + 1)
    friday = date(2023, 1, 20)
    context = SimpleNamespace(session_date=friday,
        timestamp=datetime(2023, 1, 20, 15, 30, tzinfo=ZoneInfo("Asia/Kolkata")),
        latest_completed_weekly={"SEC1": {"close": 530}}, daily_history={"SEC1": rows})
    cfg = SRBaselineConfig(weekly_sma_weeks=2, weekly_require_higher_close=True)
    assert _weekly_filter(context, "SEC1", cfg) is False


def test_diagnostics_and_decisions_are_repeatable():
    outputs = []
    for _ in range(2):
        strategy = SRBaselineStrategy(config=SRBaselineConfig(swing_left=1, swing_right=1,
            minimum_touch_separation_sessions=2), setup="support_test", target_r=1.5)
        context = _context(_support_rows())
        signals = strategy.generate_entries(context)
        outputs.append(([(item.security_id, item.stop_price, dict(item.metadata)) for item in signals],
                        list(strategy.diagnostics)))
    assert outputs[0] == outputs[1]
    assert outputs[0][0]
    assert outputs[0][0][0][2]["target_r"] == 1.5


def test_weekly_filter_rejection_updates_the_candidate_record_once():
    strategy = SRBaselineStrategy(config=SRBaselineConfig(swing_left=1, swing_right=1,
        minimum_touch_separation_sessions=2), setup="support_test", weekly_filter=True)
    context = _context(_support_rows())
    assert strategy.generate_entries(context) == ()
    candidates = [row for row in strategy.diagnostics if row["setup_type"] == "support_test"]
    assert len(candidates) == 1
    assert candidates[0]["accepted"] is False
    assert candidates[0]["rejection_reason"] == "WEEKLY_FILTER_REJECTED"
    assert candidates[0]["weekly_filter_state"] is False
