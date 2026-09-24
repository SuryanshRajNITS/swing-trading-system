from datetime import date, timedelta
from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo
from types import SimpleNamespace

from src.strategy.weekly_selection import (WeeklySelectionStrategy, completed_weekly_rows,
                                           weekly_features)
from src.decision.interfaces import StrategyContext
from src.strategy.weekly_selection import nearest_weekly_zones


def _daily(weekly_closes, *, start=date(2022,7,4), step=1.0):
    rows=[]
    monday=start
    for i,close in enumerate(weekly_closes):
        for offset in range(5):
            day=monday+timedelta(days=offset)
            value=100+i*step+offset*.1 if i==0 else close-step*.3+offset*(step*.075)
            rows.append({"date":day.isoformat(),"open":value-.1,"high":value+1,
                         "low":value-1,"close":value,"volume":1000})
        monday+=timedelta(days=7)
    return rows


def test_completed_weekly_rows_exclude_current_partial_week():
    rows=_daily([100+i for i in range(30)])
    asof=date.fromisoformat(rows[-2]["date"]) # Thursday in final week
    current_week=date.fromordinal(asof.toordinal()-asof.weekday())
    result=completed_weekly_rows(rows,asof)
    assert all(date.fromisoformat(row["week_start"])<current_week for row in result)
    changed=rows+[dict(rows[-1],date=(asof+timedelta(days=1)).isoformat(),close=99999)]
    assert result==completed_weekly_rows(changed,asof)


def test_weekly_features_calculate_returns_and_nifty_relative_strength():
    stock=_daily([100+i*2 for i in range(35)],step=2)
    nifty=_daily([100+i for i in range(35)],step=1)
    result=weekly_features(stock_daily=stock,benchmark_daily=nifty,as_of=date(2023,3,1),
                           security_id="PIT-SEC")
    assert result["features_available"]
    assert result["return_4w"]>0 and result["return_13w"]>0
    assert result["relative_strength_4w"]>0
    assert result["relative_strength_13w"]>0
    assert result["status"]=="eligible"
    assert set(result["score_components"])=={"structure","momentum","relative_strength","weekly_zone_context"}


def test_weekly_feature_history_remains_asof_when_later_rows_are_appended():
    stock=_daily([100+i for i in range(35)])
    nifty=_daily([100+i*.5 for i in range(35)],step=.5)
    asof=date(2023,3,1)
    before=weekly_features(stock_daily=stock,benchmark_daily=nifty,as_of=asof,security_id="S")
    stock.append(dict(stock[-1],date="2023-03-02",close=5000))
    nifty.append(dict(nifty[-1],date="2023-03-02",close=100))
    after=weekly_features(stock_daily=stock,benchmark_daily=nifty,as_of=asof,security_id="S")
    assert before==after


def test_insufficient_history_is_neutral_and_reproducible():
    rows=_daily([100+i for i in range(10)])
    args=dict(stock_daily=rows,benchmark_daily=rows,as_of=date(2022,10,1),security_id="S")
    first=weekly_features(**args)
    assert first["features_available"] is False
    assert first==weekly_features(**args)


def test_weekly_eligibility_is_explicit_and_reproducible():
    stock=_daily([100+i*2 for i in range(35)],step=2)
    benchmark=_daily([100+i for i in range(35)])
    args=dict(stock_daily=stock,benchmark_daily=benchmark,as_of=date(2023,3,1),security_id="S")
    eligible=weekly_features(**args)
    assert eligible["status"]=="eligible"
    assert eligible["reason"]=="weekly_gates_passed"
    assert eligible==weekly_features(**args)
    # Equal stock and benchmark returns fail the strict relative-strength gate.
    no_relative_edge=weekly_features(stock_daily=benchmark,benchmark_daily=benchmark,
        as_of=date(2023,3,1),security_id="S")
    assert no_relative_edge["status"]=="rejected"
    assert no_relative_edge["reason"]=="weekly_component_gate_failed"


def test_nearest_weekly_support_and_resistance_distances():
    zones=(SimpleNamespace(side="support",center=95),
           SimpleNamespace(side="support",center=98),
           SimpleNamespace(side="resistance",center=104),
           SimpleNamespace(side="resistance",center=110),
           SimpleNamespace(side="support",center=105))
    result=nearest_weekly_zones(zones,price=100,atr=2)
    assert result=={"nearest_support":98,"support_distance_atr":1.0,
                    "support_distance_pct":.02,"nearest_resistance":104,
                    "resistance_distance_atr":2.0,"resistance_distance_pct":.04}


def test_shortlist_is_limited_to_pit_eligible_members():
    stock=_daily([100+i*2 for i in range(35)],step=2)
    benchmark=_daily([100+i for i in range(35)])
    seen=[]
    dummy=SimpleNamespace(parameters={},zone_config=None,diagnostics=[],
                          generate_entries=lambda context: seen.extend(context.eligible) or (),
                          generate_exits=lambda context: ())
    selector=WeeklySelectionStrategy(daily_strategy=dummy,benchmark_daily=benchmark)
    members=(SimpleNamespace(security_id="PIT-A",symbol="A"),)
    context=StrategyContext(timestamp=datetime(2023,3,1,15,30,tzinfo=ZoneInfo("Asia/Kolkata")),
                            session_date=date(2023,3,1),next_session_open=None,
                            universe_id="NIFTY50_PIT_TRADABLE",eligible=members,
                            daily_history={"PIT-A":stock},latest_completed_weekly={"PIT-A":None},
                            open_security_ids=frozenset(),open_positions={})
    selector.generate_entries(context)
    assert {row["security_id"] for row in selector.decisions}=={"PIT-A"}
    assert selector.weekly_snapshots[-1]["pit_universe_count"]==1
    assert [security.security_id for security in seen]==["PIT-A"]


def test_daily_strategy_receives_only_weekly_eligible_pit_members():
    seen=[]
    dummy=SimpleNamespace(parameters={},zone_config=None,diagnostics=[],
                          generate_entries=lambda context: seen.extend(context.eligible) or (),
                          generate_exits=lambda context: ())
    selection=WeeklySelectionStrategy(daily_strategy=dummy,benchmark_daily=())
    states={
        "PIT-A":{"status":"eligible","reason":"weekly_gates_passed"},
        "PIT-B":{"status":"rejected","reason":"weekly_component_gate_failed"},
        "PIT-C":{"status":"neutral","reason":"benchmark_history_insufficient","resistance_blocked":True},
        "PIT-D":{"status":"neutral","reason":"nearby_weekly_resistance_breakout_exception_only",
                 "resistance_blocked":True},
    }
    selection._features=lambda sid,rows,as_of: {"security_id":sid,"features_available":True,
        "total_score":5,**states[sid]}
    members=tuple(SimpleNamespace(security_id=sid,symbol=sid) for sid in states)
    context=StrategyContext(timestamp=datetime(2023,3,1,15,30,tzinfo=ZoneInfo("Asia/Kolkata")),
        session_date=date(2023,3,1),next_session_open=None,universe_id="NIFTY50_PIT_TRADABLE",
        eligible=members,daily_history={},latest_completed_weekly={},open_security_ids=frozenset(),open_positions={})
    selection.generate_entries(context)
    assert [security.security_id for security in seen]==["PIT-A","PIT-D"]
    assert {row["security_id"]:row["status"] for row in selection.decisions}=={
        "PIT-A":"eligible","PIT-B":"rejected","PIT-C":"neutral","PIT-D":"neutral"}
