from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from src.backtest.models import CostModel
from src.backtest.top_down_swing_v2_portfolio_research import (
    family_edge, run_portfolio, size_position, stagnation_decision,
)

TZ = ZoneInfo("Asia/Kolkata")
CFG = {
    "initial_capital": 100000,
    "max_open_positions": 4,
    "base_risk_fraction": .0075,
    "max_total_open_risk_fraction": .035,
    "max_single_position_notional_fraction": .35,
    "base_slot_notional_divisor": 4,
    "family_weighting": {
        "enabled_in_weighted_arm": True, "minimum_multiplier": .5,
        "maximum_multiplier": 1.5, "high_expectancy_r": .15,
        "high_net_profit_factor": 1.25, "weak_expectancy_r": -.10,
        "weak_net_profit_factor": .85, "minimum_closed_trades_for_confident_edge": 10,
        "full_confidence_closed_trades": 20,
    },
    "trade_management": {"early_stagnation_check_session": 18,
                          "stagnation_exit_session": 20,
                          "absolute_max_hold_sessions": 25},
}


def _signal(sid: str, symbol: str, day: date, *, entry: float = 100,
            stop: float = 90, target: float = 130, family: str = "SUPPORT_REACTION",
            suffix: str = "one") -> dict:
    stamp = datetime.combine(day, datetime.min.time().replace(hour=9, minute=15), TZ)
    return {
        "security_id": sid, "symbol": symbol,
        "setup_instance_id": f"{sid}-{suffix}", "setup_event_id": f"event-{sid}",
        "setup_type": family, "setup_family": family,
        "setup_signal_date": day.isoformat(), "executable_timestamp": stamp,
        "confirmation_timestamp": stamp - timedelta(minutes=60),
        "daily_zones": [], "plan": {"entry": entry, "stop": stop, "target": target,
                                      "planned_reward_r": (target-entry)/(entry-stop),
                                      "target_timeframe": "DAILY", "volume_confirmation_passed": True},
        "result": {"confirmation_type": "SUPPORT_REACTION", "confirmation_features": {"volume": {"passed": True, "relative_volume": 1.2}}},
        "support": None, "resistance": [],
    }


def _bar(stamp: datetime, *, op: float = 100, hi: float = 101, lo: float = 99,
         close: float = 100) -> dict:
    return {"date": stamp.isoformat(), "open": op, "high": hi, "low": lo,
            "close": close, "volume": 1000, "session_type": "NORMAL_SESSION"}


def _one_bar_per_day(sid: str, start: date, days: int, *, close: float = 100,
                     high: float = 101) -> dict[str, list[dict]]:
    return {sid: [_bar(datetime.combine(start + timedelta(days=i), datetime.min.time().replace(hour=9, minute=15), TZ),
                       op=100, hi=high, lo=99, close=close) for i in range(days)]}


def test_family_confidence_starts_at_one_until_ten_closed_trades():
    assert family_edge([], enabled=True, thresholds=CFG["family_weighting"]).multiplier == 1
    sample = [{"net_R": 1, "net_pnl": 100} for _ in range(9)]
    assert family_edge(sample, enabled=True, thresholds=CFG["family_weighting"]).multiplier == 1


def test_family_edge_uses_expectancy_and_pf_not_win_rate_alone():
    high = [{"net_R": .3, "net_pnl": 30} for _ in range(10)]
    weak = [{"net_R": -.2, "net_pnl": -20} for _ in range(20)]
    assert family_edge(high, thresholds=CFG["family_weighting"]).multiplier == 1.25
    assert family_edge(weak, thresholds=CFG["family_weighting"]).multiplier == .5
    many_wins_but_bad_r = [{"net_R": .01, "net_pnl": 1} for _ in range(19)] + [{"net_R": -2, "net_pnl": -200}]
    assert family_edge(many_wins_but_bad_r, thresholds=CFG["family_weighting"]).multiplier == 1.0


def test_size_caps_notional_risk_cash_and_aggregate_risk_without_moving_stop():
    model = CostModel.current_zerodha_equity_delivery_2026()
    sized = size_position(equity=100000, cash=100000, entry_price=100,
                          structural_stop=90, family_multiplier=1,
                          open_risk=0, cfg=CFG, costs=model)
    assert sized["quantity"] == 75  # risk cap is tighter than the 25% slot notional
    assert sized["planned_risk"] <= 750
    assert sized["quantity"] * 100 <= sized["notional_cap"]
    weighted = size_position(equity=100000, cash=100000, entry_price=100,
                             structural_stop=90, family_multiplier=1.5,
                             open_risk=0, cfg=CFG, costs=model)
    assert weighted["risk_budget"] == 1125
    assert weighted["quantity"] <= 112
    capped = size_position(equity=100000, cash=100000, entry_price=100,
                           structural_stop=90, family_multiplier=.5,
                           open_risk=3450, cfg=CFG, costs=model)
    assert capped["quantity_by_aggregate_risk"] <= 5
    min_weight = size_position(equity=100000, cash=100000, entry_price=100,
                               structural_stop=90, family_multiplier=.5,
                               open_risk=0, cfg=CFG, costs=model)
    assert min_weight["risk_budget"] == 375
    cash_bound = size_position(equity=100000, cash=201, entry_price=100,
                               structural_stop=90, family_multiplier=1,
                               open_risk=0, cfg=CFG, costs=model)
    assert cash_bound["quantity"] == 2


def test_stagnation_rules_preserve_slow_winners_and_schedule_fixed_sessions():
    assert stagnation_decision(completed_sessions=20, mfe_r=1.0, current_r=0) == (None, False)
    assert stagnation_decision(completed_sessions=20, mfe_r=.2, current_r=.5) == (None, False)
    assert stagnation_decision(completed_sessions=19, mfe_r=.2, current_r=0) == (None, True)
    assert stagnation_decision(completed_sessions=20, mfe_r=.2, current_r=0) == ("TIME_STAGNATION_EXIT", True)
    assert stagnation_decision(completed_sessions=25, mfe_r=1.1, current_r=.6) == ("TIME_MAX_HOLD_EXIT", False)


def test_portfolio_has_four_position_cap_same_security_block_and_reproducible_execution():
    start = date(2023, 1, 2)
    signals = [_signal("A", "AAA", start, suffix="a"),
               _signal("B", "BBB", start, suffix="b"),
               _signal("C", "CCC", start, suffix="c"),
               _signal("D", "DDD", start, suffix="d"),
               _signal("A", "AAA", start, suffix="a2"),
               _signal("E", "EEE", start, suffix="e")]
    bars = {sid: _one_bar_per_day(sid, start, 2)[sid] for sid in "ABCDE"}
    costs = CostModel.current_zerodha_equity_delivery_2026()
    first = run_portfolio(signals, bars, costs, CFG, weighted=False, time_exits=False)
    second = run_portfolio(signals, bars, costs, CFG, weighted=False, time_exits=False)
    assert first == second
    assert len(first["trades"]) == 4
    assert len({t["security_id"] for t in first["trades"]}) == 4
    assert first["summary"]["maximum_open_positions"] <= 4
    assert first["rejections"] == [{**r} for r in first["rejections"]]
    reasons = {r["reason"] for r in first["rejections"]}
    assert "SAME_SECURITY_ALREADY_OPEN" in reasons
    assert "MAX_OPEN_POSITIONS" in reasons
    assert first["summary"]["zero_cost_same_execution_invariants_passed"]
    assert all(t["strategy_version"] == "top_down_swing_v2_daily_structure" for t in first["trades"])
    assert all(t["quantity"] * t["entry_price"] <= t["equity_before_entry"] * .35 + 1e-6
               for t in first["trades"])
    assert all(float(row["cash"]) >= 0 for row in first["equity_curve"])


def test_unused_cash_can_fund_a_later_valid_signal():
    start = date(2023, 1, 2)
    first = _signal("A", "AAA", start, suffix="first")
    second_day = start + timedelta(days=1)
    second = _signal("B", "BBB", second_day, suffix="second")
    bars = {
        "A": [_bar(datetime.combine(start, datetime.min.time().replace(hour=9, minute=15), TZ)),
              _bar(datetime.combine(second_day, datetime.min.time().replace(hour=9, minute=15), TZ))],
        "B": [_bar(datetime.combine(second_day, datetime.min.time().replace(hour=9, minute=15), TZ))],
    }
    result = run_portfolio([first, second], bars, CostModel.current_zerodha_equity_delivery_2026(),
                           CFG, weighted=False, time_exits=False)
    assert len(result["trades"]) == 2
    assert result["trades"][1]["cash_before_entry"] > result["trades"][1]["position_notional"]


def test_session_20_time_exit_executes_next_open_and_retains_bracket_counterfactual():
    start = date(2023, 1, 2)
    sig = _signal("A", "AAA", start, entry=100, stop=90, target=130)
    bars = _one_bar_per_day("A", start, 22, close=100, high=101)
    # Use 22 calendar dates, mimicking consecutive exchange sessions.
    next_signal_day = start + timedelta(days=20)
    next_signal = _signal("B", "BBB", next_signal_day, suffix="capacity_after_time_exit")
    bars["B"] = [_bar(datetime.combine(next_signal_day, datetime.min.time().replace(hour=9, minute=15), TZ)),
                 _bar(datetime.combine(next_signal_day + timedelta(days=1), datetime.min.time().replace(hour=9, minute=15), TZ))]
    result = run_portfolio([sig, next_signal], bars, CostModel.current_zerodha_equity_delivery_2026(),
                           CFG, weighted=False, time_exits=True)
    trade = next(t for t in result["trades"] if t["setup_instance_id"] == sig["setup_instance_id"])
    assert trade["exit_reason"] == "TIME_STAGNATION_EXIT"
    assert date.fromisoformat(trade["time_exit_scheduled_at"][:10]) == start + timedelta(days=19)
    assert date.fromisoformat(trade["exit_timestamp"][:10]) == start + timedelta(days=20)
    assert trade["stop_price"] == 90
    assert trade["counterfactual_original_bracket"]["label"].startswith("COUNTERFACTUAL_ONLY")
    assert any(t["setup_instance_id"] == next_signal["setup_instance_id"] for t in result["trades"])


def test_session_25_maximum_holding_exit_uses_next_open():
    start = date(2023, 1, 2)
    sig = _signal("A", "AAA", start, entry=100, stop=90, target=130)
    bars = _one_bar_per_day("A", start, 27, close=106, high=106)
    result = run_portfolio([sig], bars, CostModel.current_zerodha_equity_delivery_2026(),
                           CFG, weighted=False, time_exits=True)
    trade = result["trades"][0]
    assert trade["exit_reason"] == "TIME_MAX_HOLD_EXIT"
    assert date.fromisoformat(trade["time_exit_scheduled_at"][:10]) == start + timedelta(days=24)
    assert date.fromisoformat(trade["exit_timestamp"][:10]) == start + timedelta(days=25)


def test_weighting_at_entry_uses_only_family_trades_closed_before_signal_time():
    start = date(2023, 1, 2)
    signals, bars = [], {}
    for i in range(11):
        day = start + timedelta(days=i * 2)
        sid = f"S{i:02d}"
        signals.append(_signal(sid, sid, day, target=110, suffix=f"t{i}"))
        bars[sid] = [
            _bar(datetime.combine(day, datetime.min.time().replace(hour=9, minute=15), TZ)),
            _bar(datetime.combine(day + timedelta(days=1), datetime.min.time().replace(hour=9, minute=15), TZ),
                 op=100, hi=111, lo=99, close=110),
        ]
    result = run_portfolio(signals, bars, CostModel.current_zerodha_equity_delivery_2026(),
                           CFG, weighted=True, time_exits=False)
    last = next(t for t in result["trades"] if t["setup_instance_id"] == "S10-t10")
    assert last["family_closed_trades_before"] == 10
    assert last["family_multiplier"] == 1.25
    assert len(last["family_prior_closed_trade_ids"]) == 10
    assert all(exit_stamp < last["entry_timestamp"] for exit_stamp in last["family_prior_closed_exit_timestamps"])
