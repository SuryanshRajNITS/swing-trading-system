from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from src.backtest.models import CostModel
from src.backtest.top_down_swing_portfolio import _rank, _run_portfolio

TZ = ZoneInfo("Asia/Kolkata")


def ts(day: int, hour: int = 10) -> datetime:
    return datetime(2023, 1, day, hour, 15, tzinfo=TZ)


def sig(sid="A", *, rr=2.0, stamp=None, setup="setup-A", entry=100.0, stop=90.0, target=120.0):
    stamp = stamp or ts(2)
    return {"security_id": sid, "symbol": sid, "setup_instance_id": setup,
        "setup_type": "SUPPORT_REACTION", "setup_signal_date": "2023-01-01",
        "executable_timestamp": stamp, "confirmation_timestamp": stamp-timedelta(hours=1),
        "weekly_selection_path": "MOMENTUM_SELECTION",
        "result": {"confirmation_type": "SUPPORT_HOLD_CONFIRMATION", "confirmation_features": {}},
        "plan": {"entry": entry, "stop": stop, "target": target,
            "planned_reward_r": rr, "target_timeframe": "DAILY",
            "target_reference": "DAILY", "target_zone_strength": "MEANINGFUL"}}


def bar(stamp, op, hi, lo, close):
    return {"date": stamp.isoformat(), "open": op, "high": hi, "low": lo,
            "close": close, "volume": 1000}


def test_rank_uses_rr_then_structural_target_quality_then_security_id():
    higher_rr = sig("Z", rr=2.5)
    daily = sig("B", rr=2.0)
    fixed = sig("A", rr=2.0)
    fixed["plan"]["target_timeframe"] = "FIXED_2R_RESEARCH_FALLBACK"
    assert sorted([fixed, daily, higher_rr], key=_rank) == [higher_rr, daily, fixed]


def test_entry_bar_does_not_trigger_stop_or_target_on_same_bar():
    entry = ts(2)
    result = _run_portfolio([sig(stamp=entry)], {"A": [bar(entry, 100, 125, 89, 110)]},
                            CostModel.raw_zero())
    assert result["trades"][0]["status"] == "OPEN"
    assert result["trades"][0]["holding_hourly_bars"] == 0


def test_same_bar_stop_and_target_is_stop_first_and_flagged():
    entry, later = ts(2), ts(2, 11)
    result = _run_portfolio([sig(stamp=entry)], {"A": [bar(entry, 100, 110, 95, 105),
        bar(later, 105, 125, 89, 115)]}, CostModel.raw_zero())
    trade = result["trades"][0]
    assert trade["exit_reason"] == "AMBIGUOUS_INTRABAR_STOP_FIRST"
    assert trade["ambiguous_intrabar_stop_first"] is True
    assert trade["exit_price"] == pytest.approx(90)


def test_entry_sizing_uses_prior_mark_not_same_bar_future_close():
    first, second = ts(2), ts(2, 11)
    signals = [sig("A", stamp=first, setup="one"),
               sig("B", stamp=second, setup="two", entry=200, stop=180, target=240)]
    bars = {"A": [bar(first, 100, 101, 99, 100), bar(second, 100, 1000, 99, 1000)],
            "B": [bar(first, 200, 201, 199, 200), bar(second, 200, 201, 199, 200)]}
    result = _run_portfolio(signals, bars, CostModel.raw_zero())
    second_entry = next(r for r in result["trades"] if r["setup_instance_id"] == "two")
    # Prior marks are A=100 and B=200; the same-timestamp 1000 close is not known yet.
    assert second_entry["equity_before_entry"] < 10_100


def test_repeated_runs_are_deterministic_and_cash_never_negative():
    entry = ts(2)
    signals = [sig("A", stamp=entry, setup="a", rr=2.5), sig("B", stamp=entry, setup="b", rr=2.0)]
    bars = {sid: [bar(entry, 100, 101, 99, 100)] for sid in ("A", "B")}
    first = _run_portfolio(signals, bars, CostModel.current_zerodha_equity_delivery_2026())
    second = _run_portfolio(signals, bars, CostModel.current_zerodha_equity_delivery_2026())
    assert first == second
    assert all(row["cash"] >= -1e-8 for row in first["equity_curve"])
    assert len(first["simultaneous_groups"]) == 1


def test_risk_sizing_and_cash_cap_are_applied_without_leverage():
    entry = ts(2)
    signal = sig(stamp=entry, entry=1000, stop=990, target=1020)
    result = _run_portfolio([signal], {"A": [bar(entry, 1000, 1001, 999, 1000)]},
                            CostModel.raw_zero())
    trade = result["trades"][0]
    assert trade["quantity"] == 10
    assert trade["initial_risk_rupees"] == pytest.approx(100)
    assert trade["position_value"] == pytest.approx(10000)
    assert result["equity_curve"][0]["cash"] >= 0


def test_zero_quantity_and_insufficient_cash_are_explicit_skips():
    entry = ts(2)
    costly_share = sig("A", stamp=entry, entry=15000, stop=14990, target=15020)
    result = _run_portfolio([costly_share], {"A": [bar(entry, 15000, 15001, 14999, 15000)]},
                            CostModel.raw_zero())
    assert result["rejections"]["INSUFFICIENT_CASH"] == 1
    assert result["trades"] == []
    no_risk_quantity = sig("A", stamp=entry, entry=1000, stop=0, target=3000)
    # A small account and large share risk make floor(risk budget / risk/share) zero.
    result = _run_portfolio([no_risk_quantity], {"A": [bar(entry, 1000, 1001, 999, 1000)]},
                            CostModel.raw_zero(), initial_capital=100)
    assert result["rejections"]["ZERO_QUANTITY_CAPITAL_CONSTRAINT"] == 1


def test_simultaneous_signal_rank_controls_cash_allocation_deterministically():
    entry = ts(2)
    first = sig("Z", rr=2.5, stamp=entry, setup="first", entry=600, stop=599, target=602)
    second = sig("A", rr=2.0, stamp=entry, setup="second", entry=600, stop=599, target=602)
    bars = {sid: [bar(entry, 600, 601, 599, 600)] for sid in ("Z", "A")}
    result = _run_portfolio([second, first], bars, CostModel.raw_zero(), initial_capital=1000)
    assert [r["setup_instance_id"] for r in result["trades"]] == ["first"]
    assert result["rejections"]["CAPITAL_ALLOCATED_TO_PRIOR_SIGNAL"] == 1


@pytest.mark.parametrize("op,hi,lo,reason,expected", [
    (89, 95, 88, "GAP_STOP", 89),
    (121, 125, 120, "GAP_TARGET", 121),
    (100, 110, 89, "STOP", 90),
    (100, 121, 95, "TARGET", 120),
])
def test_gap_and_single_level_exits(op, hi, lo, reason, expected):
    entry, later = ts(2), ts(2, 11)
    result = _run_portfolio([sig(stamp=entry)], {"A": [bar(entry, 100, 101, 99, 100),
        bar(later, op, hi, lo, 105)]}, CostModel.raw_zero())
    assert result["trades"][0]["exit_reason"] == reason
    assert result["trades"][0]["exit_price"] == pytest.approx(expected)


def test_open_position_is_marked_to_market_without_forced_exit():
    entry, later = ts(2), ts(2, 11)
    result = _run_portfolio([sig(stamp=entry)], {"A": [bar(entry, 100, 101, 99, 100),
        bar(later, 100, 110, 99, 108)]}, CostModel.raw_zero())
    trade = result["trades"][0]
    assert trade["status"] == "OPEN"
    assert trade["exit_reason"] == "OPEN_AT_PERIOD_END"
    assert trade["mark_price"] == 108
    assert result["metrics"]["ending_equity_mtm"] > 10000


def test_dp_is_charged_once_per_security_per_sell_date():
    t1, t2, t3 = ts(2), ts(2, 11), ts(2, 12)
    first = sig("A", stamp=t1, setup="one", entry=100, stop=90, target=120)
    second = sig("A", stamp=t2, setup="two", entry=89, stop=80, target=120)
    bars = {"A": [bar(t1, 100, 101, 99, 100), bar(t2, 89, 90, 88, 89),
                   bar(t3, 79, 80, 78, 79)]}
    result = _run_portfolio([first, second], bars, CostModel.raw_zero())
    costed = _run_portfolio([first, second], bars, CostModel.current_zerodha_equity_delivery_2026())
    assert len(result["trades"]) == 2
    model = CostModel.current_zerodha_equity_delivery_2026()
    dp_component = sum(float(r["exit_costs"]) - model.calculate(
        float(r["exit_price"])*int(r["quantity"]), side="sell", include_dp_charge=False).total
        for r in costed["trades"])
    assert dp_component == pytest.approx(15.34)


def test_missing_execution_bar_fails_closed_and_entry_must_match_saved_open():
    entry = ts(2)
    signal = sig(stamp=entry)
    no_bars = _run_portfolio([signal], {}, CostModel.raw_zero())
    assert no_bars["rejections"]["MISSING_EXECUTABLE_HOURLY_BAR"] == 1
    prior = ts(2, 9)
    missing = _run_portfolio([signal], {"A": [bar(prior, 100, 101, 99, 100)],
        "B": [bar(entry, 100, 101, 99, 100)]}, CostModel.raw_zero())
    assert missing["rejections"]["MISSING_EXECUTABLE_HOURLY_BAR"] == 1
    with pytest.raises(ValueError, match="differs from next hourly open"):
        _run_portfolio([signal], {"A": [bar(entry, 101, 102, 100, 101)]}, CostModel.raw_zero())


def test_duplicate_setup_instance_id_is_blocked():
    entry = ts(2)
    one = sig("A", stamp=entry, setup="same")
    two = sig("B", stamp=entry, setup="same")
    result = _run_portfolio([one, two], {"A": [bar(entry, 100, 101, 99, 100)],
        "B": [bar(entry, 100, 101, 99, 100)]}, CostModel.raw_zero())
    assert result["rejections"]["DUPLICATE_SETUP_BLOCKED"] == 1
    assert len(result["trades"]) == 1
