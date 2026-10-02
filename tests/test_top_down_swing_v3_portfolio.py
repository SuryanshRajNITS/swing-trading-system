from datetime import date

import pytest

from src.backtest.models import CostModel
from src.backtest.top_down_swing_v3_portfolio import _protection_counterfactual, _protected_run, _year_breakdown


def _signal():
    return {"security_id":"S1","symbol":"TEST","setup_family":"TRENDLINE_PULLBACK",
        "setup_instance_id":"S1-E1","setup_signal_date":"2023-01-02","regime":"TREND_UP",
        "primary_geometry":"INCLINED","plan":{"entry":100.0,"stop":90.0,"target":115.0,
        "planned_reward_r":1.5,"target_timeframe":"CHANNEL"},
        "confirmation_timestamp":"2023-01-02T09:15:00+05:30",
        "executable_timestamp":__import__("datetime").datetime.fromisoformat("2023-01-02T10:15:00+05:30")}


def _bars():
    return {"S1":[
        {"date":"2023-01-02T10:15:00+05:30","open":100,"high":103,"low":99,"close":102,"volume":100},
        {"date":"2023-01-02T11:15:00+05:30","open":102,"high":116,"low":101,"close":115,"volume":120},
        {"date":"2023-01-02T12:15:00+05:30","open":115,"high":116,"low":114,"close":115,"volume":90},
    ]}


def test_fixed_bracket_arm_keeps_target_execution_and_portfolio_caps():
    result=_protected_run([_signal()],_bars(),CostModel.raw_zero(),initial=100000,risk_fraction=.0075,
        max_positions=4,max_notional_fraction=.35,total_risk_fraction=.035,enable_protection=False)
    trade=result["trades"][0]
    assert trade["status"]=="CLOSED"
    assert trade["exit_reason"]=="TARGET"
    assert result["metrics"]["negative_cash_observed"] is False
    assert result["metrics"]["max_open_positions_observed"]<=4


def test_protected_arm_is_deterministic_and_never_lowers_stop():
    bars=_bars()
    one=_protected_run([_signal()],bars,CostModel.raw_zero(),initial=100000,risk_fraction=.0075,
        max_positions=4,max_notional_fraction=.35,total_risk_fraction=.035,enable_protection=True)
    two=_protected_run([_signal()],bars,CostModel.raw_zero(),initial=100000,risk_fraction=.0075,
        max_positions=4,max_notional_fraction=.35,total_risk_fraction=.035,enable_protection=True)
    assert one==two
    assert one["trades"][0]["exit_reason"]=="TARGET"
    assert one["trades"][0]["final_current_stop"]>=one["trades"][0]["original_stop"]


def test_entry_rejected_when_next_open_reduces_actual_rr_below_one_point_five():
    signal=_signal(); signal["plan"]["target"]=110.0
    signal["executable_timestamp"]=__import__("datetime").datetime.fromisoformat("2023-01-02T10:15:00+05:30")
    bars=_bars(); bars["S1"][0]["open"]=104
    result=_protected_run([signal],bars,CostModel.raw_zero(),initial=100000,risk_fraction=.0075,
        max_positions=4,max_notional_fraction=.35,total_risk_fraction=.035,enable_protection=True)
    assert not result["trades"]
    assert result["rejections"]["ACTUAL_RR_BELOW_MINIMUM"]==1


def test_same_bar_stop_and_target_uses_conservative_stop_first():
    bars=_bars(); bars["S1"][1]["low"]=89
    result=_protected_run([_signal()],bars,CostModel.raw_zero(),initial=100000,risk_fraction=.0075,
        max_positions=4,max_notional_fraction=.35,total_risk_fraction=.035,enable_protection=False)
    trade=result["trades"][0]
    assert trade["exit_reason"]=="ORIGINAL_STOP"
    assert trade["ambiguous_intrabar_stop_first"] is True


def test_protective_stop_activates_after_completed_bar_and_gap_fill_is_realized_at_open():
    signal=_signal(); signal["plan"]["target"]=130.0; signal["plan"]["planned_reward_r"]=3.0
    bars={"S1":[
        {"date":"2023-01-02T10:15:00+05:30","open":100,"high":102,"low":99,"close":101,"volume":100},
        {"date":"2023-01-02T11:15:00+05:30","open":101,"high":118,"low":100,"close":116,"volume":200},
        {"date":"2023-01-02T12:15:00+05:30","open":99,"high":100,"low":98,"close":99,"volume":100},
    ]}
    result=_protected_run([signal],bars,CostModel.raw_zero(),initial=100000,risk_fraction=.0075,
        max_positions=4,max_notional_fraction=.35,total_risk_fraction=.035,enable_protection=True)
    trade=result["trades"][0]
    assert trade["exit_reason"]=="GAP_PROTECTIVE_STOP"
    assert trade["raw_exit_price"]==99
    assert trade["final_current_stop"]>trade["original_stop"]
    assert trade["realized_R_after_gap"]<trade["intended_protected_R"]


def test_protection_report_separates_fixed_gap_and_structural_trail_exits():
    trades=[{"status":"CLOSED","exit_reason":reason,"security_id":"S1",
        "exit_timestamp":"2023-01-02T10:15:00+05:30","original_stop":90,"target_price":115,
        "realized_R":.75,"setup_instance_id":str(i)}
        for i,reason in enumerate(("PROTECTIVE_STOP","GAP_PROTECTIVE_STOP","TRAILING_STRUCTURAL_STOP"))]
    report=_protection_counterfactual(trades,_bars())
    assert report["protected_stop_exits"]==3
    assert report["protective_stop_exits"]==1
    assert report["gap_protective_stop_exits"]==1
    assert report["structural_trail_exits"]==1


def test_year_drawdown_uses_same_daily_equity_high_water_mark_as_total():
    equity=[{"timestamp":"2023-12-29","equity":120000,"drawdown_pct":0},
        {"timestamp":"2024-01-02","equity":110000,"drawdown_pct":-15}]
    rows=_year_breakdown([],equity)
    assert rows[1]["max_dd_pct"]==pytest.approx(-10000/120000*100)
