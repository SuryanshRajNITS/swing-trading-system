from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from src.backtest.top_down_swing_portfolio import _run_portfolio
from src.backtest.models import CostModel
from src.backtest.top_down_swing_sniper import _sniper_candidates, _support_hold_delta, _tier_rank, classify_candidate

TZ = ZoneInfo("Asia/Kolkata")


def candidate(tier, sid, rr=2.0):
    return {"security_id": sid, "setup_instance_id": sid, "tier": tier,
        "quality": {"location_quality_rank": 2, "structural_rr": rr,
            "target_quality_rank": 2, "confirmation_quality_rank": 3, "volume_support": True}}


def test_quality_tiers_are_deterministic_and_ignore_future_pnl_fields():
    s = {"plan": {"entry": 100, "stop": 90, "target": 120, "planned_reward_r": 2,
                   "target_timeframe": "DAILY", "target_zone_strength": "MAJOR"},
         "result": {"confirmation_type": "SUPPORT_HOLD_CONFIRMATION", "confirmation_features": {
             "higher_low_to_higher_high": True, "same_time_expansion": True,
             "held_support_zone_eligibility": [{"eligible_for_support_hold": True, "strength": "MAJOR",
                 "role_state": "ACTIVE", "original_side": "support"}]}},
         "setup_zone": {"strength": "MAJOR"}, "structural_location_category": "SUPPORT_LOCATION_ACTIONABLE",
         "weekly_structure": {"higher_high": True, "higher_low": True}, "net_pnl": 1}
    first = classify_candidate(s)
    changed_outcome = classify_candidate({**s, "net_pnl": -999999, "gross_pnl": 123456})
    assert first == changed_outcome
    assert first["quality_tier"] == "TIER_A_ROBUST"
    assert first["hostile_volume_status"] == "NOT_AVAILABLE_IN_FROZEN_FEATURES"


def test_support_hold_removal_cause_counts_reconcile_to_unique_removed_instances():
    old = [{"setup_instance_id":"one","hourly_result":'{"confirmation_type":"SUPPORT_HOLD_CONFIRMATION","reason_codes":["HOURLY_HIGHER_LOW_TO_HIGHER_HIGH"]}'},
           {"setup_instance_id":"two","hourly_result":'{"confirmation_type":"SUPPORT_HOLD_CONFIRMATION","confirmation_features":{"involved_zones":[{"role_state":"RESISTANCE_CANDIDATE","broken":true}]}}'}]
    new = [{"setup_instance_id":"two","hourly_result":'{"confirmation_type":"SUPPORT_HOLD_CONFIRMATION"}'}]
    audit = _support_hold_delta(old, new)
    assert audit["removed"] == 1
    assert audit["added"] == 0
    assert audit["net_delta"] == -1
    assert sum(audit["causes"].values()) == audit["removed"]
    assert audit["causes"]["legacy_hl_hh_label"] == 1


def test_fallback_target_can_be_tier_b_but_not_tier_a_and_c_is_deterministic():
    s = {"plan": {"entry": 100, "stop": 90, "target": 120, "planned_reward_r": 2,
                   "target_reference": "FIXED_2R_RESEARCH_FALLBACK"},
         "result": {"confirmation_type": "SUPPORT_HOLD_CONFIRMATION", "confirmation_features": {}},
         "structural_location_category": "SUPPORT_LOCATION_ACTIONABLE", "setup_zone": {"strength": "MEANINGFUL"}}
    assert classify_candidate(s)["quality_tier"] == "TIER_B_VALID"
    assert classify_candidate(s) == classify_candidate(s)
    assert [x["security_id"] for x in sorted([candidate("TIER_B_VALID", "B"),
        candidate("TIER_A_ROBUST", "Z")], key=_tier_rank)] == ["Z", "B"]
    assert [s["tier"] for s in _sniper_candidates([
        candidate("TIER_A_ROBUST", "A"), candidate("TIER_B_VALID", "B"), candidate("TIER_C_WEAK", "C")])] == ["TIER_A_ROBUST", "TIER_B_VALID"]
    tied = sorted([candidate("TIER_A_ROBUST", "Z"), candidate("TIER_A_ROBUST", "A")], key=_tier_rank)
    assert [s["security_id"] for s in tied] == ["A", "Z"]


def test_patient_allocator_executes_tier_b_and_deployment_respects_risk_and_cash():
    stamp = datetime(2023, 1, 2, 10, 15, tzinfo=TZ)
    s = {"security_id": "A", "symbol": "A", "setup_instance_id": "A", "tier": "TIER_B_VALID",
        "quality": {"location_quality_rank": 1, "structural_rr": 2, "target_quality_rank": 1,
                    "confirmation_quality_rank": 2, "volume_support": False},
        "setup_type": "SUPPORT_REACTION", "setup_signal_date": "2023-01-01",
        "executable_timestamp": stamp, "confirmation_timestamp": stamp-timedelta(hours=1),
        "result": {"confirmation_type": "SUPPORT_HOLD_CONFIRMATION", "confirmation_features": {}},
        "plan": {"entry": 100, "stop": 90, "target": 120, "planned_reward_r": 2,
                 "target_timeframe": "DAILY", "target_zone_strength": "MEANINGFUL"}}
    bars = {"A": [{"date": stamp.isoformat(), "open": 100, "high": 101, "low": 99, "close": 100, "volume": 1}]}
    result = _run_portfolio(_sniper_candidates([s]), bars, CostModel.raw_zero(), initial_capital=10000,
        risk_fraction=.01, max_positions=1, preferred_deployment_pct=.8)
    trade = result["trades"][0]
    assert trade["quantity"] == 10
    assert trade["capital_deployed_pct"] == .1
    assert trade["actual_risk_pct"] == .01
    assert trade["risk_budget_utilization"] == 1
    assert result["equity_curve"][0]["cash"] >= 0
    assert result["equity_curve"][0]["cash"] == 9000


def test_cost_decomposition_and_r_metrics_are_recorded():
    start = datetime(2023, 1, 2, 10, 15, tzinfo=TZ); later = start + timedelta(hours=1)
    s = {"security_id":"A","symbol":"A","setup_instance_id":"A","setup_type":"SUPPORT_REACTION",
        "setup_signal_date":"2023-01-01","executable_timestamp":start,"confirmation_timestamp":start-timedelta(hours=1),
        "result":{"confirmation_type":"SUPPORT_HOLD_CONFIRMATION","confirmation_features":{}},
        "plan":{"entry":100,"stop":90,"target":120,"planned_reward_r":2,"target_timeframe":"DAILY"}}
    bars={"A":[{"date":start.isoformat(),"open":100,"high":101,"low":99,"close":100,"volume":1},
                {"date":later.isoformat(),"open":100,"high":121,"low":99,"close":120,"volume":1}]}
    run=_run_portfolio([s],bars,CostModel.current_zerodha_equity_delivery_2026())
    trade=run["trades"][0]
    assert trade["entry_cost_breakdown"]["stamp_duty"] > 0
    assert trade["exit_cost_breakdown"]["dp_charge"] > 0
    assert trade["gross_R"] == 2


def test_two_position_sniper_allows_two_distinct_positions_without_leverage():
    stamp = datetime(2023, 1, 2, 10, 15, tzinfo=TZ)
    signals=[]; bars={}
    for sid in ("A", "B"):
        signals.append({"security_id":sid,"symbol":sid,"setup_instance_id":sid,"setup_type":"SUPPORT_REACTION",
            "setup_signal_date":"2023-01-01","executable_timestamp":stamp,"confirmation_timestamp":stamp-timedelta(hours=1),
            "result":{"confirmation_type":"SUPPORT_HOLD_CONFIRMATION","confirmation_features":{}},
            "plan":{"entry":100,"stop":90,"target":120,"planned_reward_r":2,"target_timeframe":"DAILY"}})
        bars[sid]=[{"date":stamp.isoformat(),"open":100,"high":101,"low":99,"close":100,"volume":1}]
    run=_run_portfolio(signals,bars,CostModel.raw_zero(),initial_capital=10000,risk_fraction=.01,max_positions=2)
    assert len(run["trades"]) == 2
    assert run["equity_curve"][0]["open_positions"] == 2
    assert run["equity_curve"][0]["cash"] >= 0
