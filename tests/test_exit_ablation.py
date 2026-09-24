from datetime import date, datetime, time, timedelta
from types import SimpleNamespace

import pytest

from src.backtest.engine import BacktestConfig, BacktestEngine
from src.backtest.exit_ablation import (ARMS, AblationArmStrategy, _display, _summary)
from src.backtest.models import CostModel, OperatingCostModel
from src.decision.interfaces import EntrySignal
from src.strategy.sr_zone_v2 import ZoneConfig
from src.strategy.structural_exit import StructuralExitConfig


class _Universe:
    def members_on(self, day, universe_id):
        return [SimpleNamespace(security_id="S1", symbol="S1", tradable=True)]


class _SignalSource:
    name = "test_source"
    version = "test-v1"
    parameters = {}
    diagnostics = []
    zone_config = ZoneConfig()

    def generate_entries(self, context):
        if context.session_date != date(2024, 1, 2) or context.next_session_open is None:
            return ()
        return (EntrySignal("S1", "S1", context.timestamp, context.next_session_open,
                            90.0, None, (), {"setup_type": "support_test",
                                             "proposed_entry": 100.0,
                                             "zone": {"lower": 90.0}}, self.version),)

    def generate_exits(self, context):
        return ()


def _run(*, arm, open_price, target_high=149.0, plan=None):
    source = _SignalSource()
    source.diagnostics = []
    strategy = AblationArmStrategy(arm=arm, signal_strategy=source,
                                   structural_config=StructuralExitConfig())
    if plan is not None:
        strategy._structural_plan = lambda signal, context, full: plan
    days = [date(2024, 1, 2), date(2024, 1, 3)]
    rows = [
        {"date": datetime.combine(days[0], time.min), "open": 100.0,
         "high": 101.0, "low": 99.0, "close": 100.0, "volume": 100},
        {"date": datetime.combine(days[1], time.min), "open": open_price,
         "high": target_high, "low": min(open_price, 95.0), "close": open_price,
         "volume": 100},
    ]
    config = BacktestConfig("TOY", days[0], days[-1], initial_capital=10_000,
                            max_risk_per_trade=.01, max_open_positions=2,
                            costs=CostModel.raw_zero(),
                            operating_costs=OperatingCostModel(0, 0, 0))
    result = BacktestEngine(config=config, provider=_Universe()).run(
        strategy=strategy, daily_data={"S1": rows})
    return result, strategy


def test_four_registered_arms_are_exact():
    assert ARMS == ("A_BASELINE", "B_TIME_EXIT", "C_STRUCTURAL_STOP_TIME",
                    "D_FULL_STRUCTURAL")


def test_arm_c_fixed_two_r_target_uses_actual_fill_and_structural_stop():
    plan = {"accepted": True, "reason_code": "STRUCTURAL_PLAN_ACCEPTED",
            "stop_price": 90.0, "stop_distance_percent": 10.0,
            "target_price": 140.0, "target_status": "VALID_STRUCTURAL_TARGET"}
    result, _ = _run(arm="C_STRUCTURAL_STOP_TIME", open_price=110.0, plan=plan)
    # Actual fill 110, stop 90: engine must use 110 + 2 * (110 - 90) = 150.
    assert result.open_positions[0]["target_price"] == pytest.approx(150.0)


def test_arm_b_exits_after_15_completed_sessions_at_the_next_open():
    source = _SignalSource()
    source.diagnostics = []
    strategy = AblationArmStrategy(arm="B_TIME_EXIT", signal_strategy=source,
                                   structural_config=StructuralExitConfig())
    days = []
    current = date(2024, 1, 2)
    while len(days) < 17:
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    rows = [{"date": datetime.combine(day, time.min), "open": 100.0,
             "high": 101.0, "low": 99.0, "close": 100.0, "volume": 100}
            for day in days]
    config = BacktestConfig("TOY", days[0], days[-1], initial_capital=10_000,
                            max_risk_per_trade=.01, max_open_positions=2,
                            costs=CostModel.raw_zero(),
                            operating_costs=OperatingCostModel(0, 0, 0))
    result = BacktestEngine(config=config, provider=_Universe()).run(
        strategy=strategy, daily_data={"S1": rows})
    assert result.trades[0].exit_reason == "TIME_EXIT"
    assert result.trades[0].exit_timestamp.date() == days[16]
    assert result.trades[0].holding_period == 15


def test_arm_d_recalculates_actual_rr_and_rejects_opening_gap_below_minimum():
    plan = {"accepted": True, "reason_code": "STRUCTURAL_PLAN_ACCEPTED",
            "stop_price": 90.0, "stop_distance_percent": 10.0,
            "target_price": 120.0, "target_resistance_zone_id": "R-1",
            "target_status": "VALID_STRUCTURAL_TARGET"}
    result, _ = _run(arm="D_FULL_STRUCTURAL", open_price=105.0,
                     target_high=119.0, plan=plan)
    assert not result.trades
    rejection = next(event for event in result.events
                     if event.event_type == "ENTRY_REJECTED")
    assert rejection.details["reason_code"] == "ACTUAL_RR_BELOW_MINIMUM"
    assert rejection.details["actual_reward_risk_ratio"] == pytest.approx(1.0)


def test_arm_d_rejects_fill_at_or_above_structural_target_explicitly():
    plan = {"accepted": True, "reason_code": "STRUCTURAL_PLAN_ACCEPTED",
            "stop_price": 90.0, "stop_distance_percent": 10.0,
            "target_price": 120.0, "target_resistance_zone_id": "R-1",
            "target_status": "VALID_STRUCTURAL_TARGET"}
    result, _ = _run(arm="D_FULL_STRUCTURAL", open_price=121.0,
                     target_high=122.0, plan=plan)
    rejection = next(event for event in result.events
                     if event.event_type == "ENTRY_REJECTED")
    assert rejection.details["reason_code"] == "ACTUAL_ENTRY_AT_OR_ABOVE_STRUCTURAL_TARGET"


def test_comparison_output_is_deterministic():
    from src.backtest.exit_ablation import _deltas

    arm = {"overall": _summary([], 10_000), "by_year": {
        "2023": _summary([], 10_000), "2024": _summary([], 10_000)},
        "time_exit": _summary([], 10_000)["time_exit"],
        "by_setup": {"SUPPORT": _summary([], 10_000),
                     "BREAKOUT_RETEST": _summary([], 10_000)}}
    report = {"arms": {name: arm for name in ARMS},
              "deltas": _deltas({name: arm for name in ARMS}),
              "structural_rejection_reasons_D": {}}
    assert _display(report) == _display(report)
