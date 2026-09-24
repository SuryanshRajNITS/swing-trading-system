from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

import src.strategy.structural_exit as structural_exit
from src.backtest.engine import BacktestConfig, BacktestEngine
from src.backtest.models import CostModel, OperatingCostModel
from src.decision.interfaces import EntrySignal, StrategyContext
from src.strategy.sr_zone_v2 import ZoneConfig
from src.strategy.structural_exit import (StructuralExitConfig, StructuralExitStrategy,
                                          active_resistance_zones_asof,
                                          build_structural_exit_plan)


def _resistance(zone_id, lower, upper, *, broken=False):
    return SimpleNamespace(zone_id=zone_id,side="resistance",lower_bound=lower,
                           upper_bound=upper,broken=broken,pivots=("p1","p2"))


def test_support_stop_is_buffered_below_the_responsible_support_zone():
    plan=build_structural_exit_plan(setup_type="support_test",
        signal_metadata={"proposed_entry":100,"zone_id":"SUP-1",
                         "zone":{"lower":90,"upper":94}},
        active_resistances=[_resistance("RES-1",130,135)],atr=4,
        config=StructuralExitConfig(minimum_reward_risk=.5))
    assert plan["accepted"]
    assert plan["structural_reference_price"]==90
    assert plan["stop_buffer"]==1
    assert plan["stop_price"]==89
    assert plan["stop_price"]<90
    assert plan["structural_zone_id"]=="SUP-1"
    assert plan["stop_distance_percent"]==pytest.approx(11.0)
    assert plan["stop_distance_atr"]==pytest.approx(2.75)


def test_breakout_stop_uses_lower_of_retest_low_and_broken_zone_boundary():
    plan=build_structural_exit_plan(setup_type="breakout_retest",
        signal_metadata={"proposed_entry":105,"retest_low":96,"zone_id":"BROKEN-1",
                         "zone":{"lower":93,"upper":98}},
        active_resistances=[_resistance("RES-1",140,145)],atr=4,
        config=StructuralExitConfig(minimum_reward_risk=.5))
    assert plan["structural_reference_price"]==93
    assert plan["structural_reference_type"]=="BROKEN_ZONE_LOWER_BOUNDARY"
    assert plan["structural_reference_candidates"]=={
        "retest_low":96.0,"broken_zone_lower_boundary":93.0}
    assert plan["stop_price"]==92


def test_nearest_active_resistance_is_selected_and_target_is_below_it():
    zones=(_resistance("BROKEN",105,108,broken=True),
           _resistance("NEAREST",110,114),_resistance("FARTHER",120,124))
    plan=build_structural_exit_plan(setup_type="support_test",
        signal_metadata={"proposed_entry":100,"zone":{"lower":80},"zone_id":"SUP"},
        active_resistances=zones,atr=2,
        config=StructuralExitConfig(minimum_reward_risk=.1))
    assert plan["accepted"]
    assert plan["target_resistance_zone_id"]=="NEAREST"
    assert plan["target_zone_lower_bound"]==110
    assert plan["target_buffer"]==pytest.approx(0.2)
    assert plan["target_price"]==pytest.approx(109.8)
    assert plan["target_price"]<plan["target_zone_lower_bound"]
    assert plan["target_price"]<plan["target_zone_upper_bound"]


def test_future_resistance_cannot_enter_asof_zone_set(monkeypatch):
    captured={}
    future_zone=_resistance("FUTURE",120,124)
    def fake_detect_zones(rows,**kwargs):
        captured["max_date"]=max(row["date"] for row in rows)
        return (future_zone,)
    monkeypatch.setattr(structural_exit,"detect_zones",fake_detect_zones)
    rows=[{"date":"2024-01-02","open":100,"high":101,"low":99,"close":100,"volume":1},
          {"date":"2024-01-03","open":100,"high":101,"low":99,"close":100,"volume":1},
          {"date":"2024-01-04","open":100,"high":101,"low":99,"close":100,"volume":1}]
    known=active_resistance_zones_asof(daily_rows=rows,security_id="S",
                                       as_of=date(2024,1,3),zone_config=ZoneConfig())
    assert captured["max_date"]=="2024-01-03"
    assert known==(future_zone,)


def test_minimum_reward_risk_rejection():
    plan=build_structural_exit_plan(setup_type="support_test",
        signal_metadata={"proposed_entry":100,"zone":{"lower":90}},
        active_resistances=[_resistance("NEAR",108,112)],atr=2)
    assert not plan["accepted"]
    assert plan["reason_code"]=="INSUFFICIENT_STRUCTURAL_RR"
    assert plan["resulting_reward_risk_ratio"]<1.5


def test_missing_higher_resistance_rejects_without_fixed_r_fallback():
    plan=build_structural_exit_plan(setup_type="support_test",
        signal_metadata={"proposed_entry":100,"zone":{"lower":90}},
        active_resistances=[_resistance("BELOW",95,99)],atr=2)
    assert not plan["accepted"]
    assert plan["reason_code"]=="NO_STRUCTURAL_TARGET"
    assert plan["target_status"]=="NO_STRUCTURAL_TARGET"
    assert "target_price" not in plan


def test_fixed_r_fallback_requires_explicit_configuration_and_is_labeled():
    plan=build_structural_exit_plan(setup_type="support_test",
        signal_metadata={"proposed_entry":100,"zone":{"lower":90}},
        active_resistances=(),atr=2,
        config=StructuralExitConfig(no_target_policy="FIXED_R_FALLBACK",fallback_reward_risk=2))
    assert plan["accepted"]
    assert plan["reason_code"]=="EXPLICIT_FIXED_R_FALLBACK_NO_STRUCTURAL_TARGET"
    assert plan["target_type"]=="EXPLICIT_FIXED_R_FALLBACK"
    assert plan["target_price"]==pytest.approx(121)


def test_structural_plan_is_deterministic():
    args=dict(setup_type="support_test",signal_metadata={"proposed_entry":100,"zone":{"lower":90}},
              active_resistances=[_resistance("R",120,124)],atr=2)
    assert build_structural_exit_plan(**args)==build_structural_exit_plan(**args)


@dataclass(frozen=True)
class _Member:
    security_id: str="A"
    symbol: str="A"
    tradable: bool=True


class _Universe:
    def __init__(self, sessions): self.sessions=sessions
    def members_on(self, day, universe_id): return [_Member()]


class _SignalSource:
    name="fixture"
    version="fixture-v1"
    parameters={}
    diagnostics=[]
    zone_config=ZoneConfig()
    def __init__(self, signal_day): self.signal_day=signal_day
    def generate_entries(self, context):
        if context.session_date!=self.signal_day or context.next_session_open is None: return ()
        return (EntrySignal("A","A",context.timestamp,context.next_session_open,90,200,
                            metadata={"setup_type":"support_test","maximum_holding_sessions":15},
                            strategy_version=self.version),)
    def generate_exits(self, context): return ()


class _TimeOnlyStrategy(StructuralExitStrategy):
    def generate_entries(self, context):
        return self.signal_strategy.generate_entries(context)


def _time_exit_run(*, touch=None):
    days=[]
    current=date(2024,1,2)
    while len(days)<17:
        if current.weekday()<5: days.append(current)
        current+=timedelta(days=1)
    rows=[]
    for index,day in enumerate(days):
        high,low=102.0,98.0
        if touch and index==15:
            if touch=="stop": low=89.0
            elif touch=="target": high=201.0
            elif touch=="both": high,low=201.0,89.0
        rows.append({"date":datetime.combine(day,time.min),"open":100.0,"high":high,
                     "low":low,"close":100.0,"volume":1000})
    config=BacktestConfig(universe_id="TOY",start_date=days[0],end_date=days[-1],
        initial_capital=10000,max_risk_per_trade=.01,max_open_positions=2,
        costs=CostModel.raw_zero(),operating_costs=OperatingCostModel(0,0,0))
    strategy=_TimeOnlyStrategy(signal_strategy=_SignalSource(days[0]))
    result=BacktestEngine(config=config,provider=_Universe(days)).run(
        strategy=strategy,daily_data={"A":rows})
    return result,days,strategy


def test_time_exit_fires_after_exactly_15_completed_sessions_at_next_open():
    result,days,strategy=_time_exit_run()
    trade=result.trades[0]
    assert trade.exit_reason=="TIME_EXIT"
    assert strategy.time_exit_decisions[0]["date"]==days[15].isoformat()
    assert trade.exit_timestamp.date()==days[16]
    assert trade.exit_timestamp.time()==time(9,15)
    assert trade.holding_period==15
    assert trade.metadata["time_exit"]["holding_sessions"]==15
    assert trade.metadata["time_exit"]["net_pnl"]==trade.net_pnl
    assert trade.metadata["time_exit"]["R_multiple"]==trade.R_multiple
    assert trade.metadata["time_exit"]["profitable"]==(trade.net_pnl>0)


@pytest.mark.parametrize(("touch","expected"),[("stop","STOP"),("target","TARGET"),("both","STOP")])
def test_stop_or_target_on_maximum_hold_session_precedes_time_exit(touch,expected):
    result,days,strategy=_time_exit_run(touch=touch)
    assert result.trades[0].exit_reason==expected
    assert result.trades[0].exit_timestamp.date()==days[15]
    assert not strategy.time_exit_decisions
