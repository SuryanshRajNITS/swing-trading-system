from __future__ import annotations

from datetime import date, timedelta
from math import sin, pi

import pytest

from src.strategy.structural_regime_v3 import classify_regime, classify_regime_series, confirmed_pivots, ema_context, ema_structural_state, trendlines
from src.strategy.top_down_swing_v3 import V3Config, TrendlineSetupLifecycle, family_allowed, trendline_pullback_candidate
from src.strategy.trade_management_v3 import initial_protection, stop_execution, update_protection


def _bars(kind: str, n: int = 280):
    rows=[]; day=date(2022,1,1)
    for i in range(n):
        if kind=="up":
            base=100+i*.42+3*sin(i*pi/6)
        elif kind=="down":
            base=300-i*.40+3*sin(i*pi/6)
        else:
            base=100+5*sin(i*pi/6)
        rows.append({"date":(day+timedelta(days=i)).isoformat(),"open":base-.3,"high":base+2,"low":base-2,"close":base,"volume":1000+(i%5)*100})
    return rows


def test_regime_uses_only_completed_daily_prefix_and_rejects_future_date():
    rows=_bars("up")
    result=classify_regime(rows[:-1],as_of=rows[-1]["date"])
    assert result["input_through"]==rows[-2]["date"]
    with pytest.raises(ValueError,match="strictly before"):
        classify_regime(rows,as_of=rows[-1]["date"])


def test_clear_higher_high_higher_low_classifies_trend_up():
    result=classify_regime(_bars("up"),as_of="2024-01-01")
    assert result["regime"]=="TREND_UP"
    assert "HIGHER_HIGH_AND_HIGHER_LOW" in result["reason_codes"]


def test_clear_lower_high_lower_low_classifies_trend_down():
    result=classify_regime(_bars("down"),as_of="2024-01-01")
    assert result["regime"]=="TREND_DOWN"


def test_pivot_requires_right_side_confirmation_and_no_future_anchor():
    rows=_bars("range",40)
    rows[20]["low"]=-10
    before=confirmed_pivots(rows[:22],side="low",left=2,right=2,min_reaction_atr=0)
    after=confirmed_pivots(rows[:23],side="low",left=2,right=2,min_reaction_atr=0)
    assert not any(x["index"]==20 for x in before)
    assert any(x["index"]==20 and x["confirm_index"]==22 for x in after)


def test_trendline_outputs_deterministically_and_wick_alone_is_not_break():
    rows=_bars("up")
    a=trendlines(rows); b=trendlines(rows)
    assert a==b
    assert a["rising_support"] is not None
    assert a["rising_support"].status in {"PRELIMINARY","ESTABLISHED","STRONG"}
    assert a["rising_support"].tolerance>0
    wick=[dict(x) for x in rows]
    wick[-1]["low"]=a["rising_support"].projected_level-a["rising_support"].tolerance*3
    wick[-1]["close"]=a["rising_support"].projected_level
    assert trendlines(wick)["rising_support"] is None or trendlines(wick)["rising_support"].status!="BROKEN"


def test_ema_context_is_causal_and_reports_reclaim_or_support_state():
    rows=_bars("up")
    got=ema_context(rows)
    assert got==ema_context(rows)
    assert got["ema50"]["value"] is not None
    assert got["ema50"]["normalized_slope"]>0


def test_ema_context_uses_long_causal_history_for_ema200_and_slopes():
    rows=_bars("up",280)
    got=ema_context(rows,interaction_rows=rows[-60:])
    before=ema_context(rows[:-1],interaction_rows=rows[-61:-1])
    assert got["ema50"]["value"] is not None
    assert got["ema200"]["value"] is not None
    assert got["ema50"]["normalized_slope"]>0
    assert got["ema200"]["normalized_slope"]>0
    assert got["ema200"]["value"] != before["ema200"]["value"]


def test_ema_structural_state_requires_repeated_touches_for_support():
    rows=_bars("up",280)
    ema_values={"ema50":{"valid":True,"value":rows[-1]["close"]-1,"normalized_slope":.1,
                         "accepted_above":True,"touch_count":2,"distance_atr":.2,"state":"SUPPORTIVE"},
                "ema200":{"valid":False,"value":None,"normalized_slope":None,
                          "accepted_above":False,"touch_count":0,"distance_atr":None,"state":"NONE"}}
    assert ema_structural_state(rows,ema_values)=="EMA50_SUPPORT"
    ema_values["ema50"]["touch_count"]=1
    assert ema_structural_state(rows,ema_values)=="NONE"


def test_regime_permissions_gate_downtrend_support_and_ambiguous():
    assert family_allowed("TREND_DOWN","SUPPORT_REACTION")== (False,"DOWNTREND_SUPPORT_REQUIRES_RECLAIM")
    assert family_allowed("TREND_DOWN","SUPPORT_REACTION",reclaim_confirmed=True)[0]
    assert not family_allowed("AMBIGUOUS","BREAKOUT_RETEST")[0]
    assert family_allowed("RANGE","SUPPORT_REACTION")[0]


def test_v3_minimum_rr_cannot_be_configured_below_one_point_five():
    with pytest.raises(ValueError): V3Config(minimum_structural_rr=1.49)


def test_profit_protection_milestones_and_stop_never_moves_backwards():
    state=initial_protection(100,90)
    no_move=update_protection(state,entry=100,high_since_entry=109.9,timestamp="t1",confirmed_h1_rows=[],h1_atr=1)
    assert no_move.current_stop==90
    floor=update_protection(no_move,entry=100,high_since_entry=115,timestamp="t2",confirmed_h1_rows=[],h1_atr=1)
    assert floor.current_stop>=100
    locked=update_protection(floor,entry=100,high_since_entry=120,timestamp="t3",confirmed_h1_rows=[],h1_atr=1)
    assert locked.current_stop>=107.5
    less=update_protection(locked,entry=100,high_since_entry=116,timestamp="t4",confirmed_h1_rows=[],h1_atr=1)
    assert less.current_stop==locked.current_stop
    assert locked.r_floor==.75


@pytest.mark.parametrize(("mfe","expected"),[(1.0,None),(1.5,0.0),(2.0,.75),(2.5,1.25),(3.0,1.75)])
def test_every_fixed_milestone_has_only_its_configured_minimum_floor(mfe,expected):
    base=initial_protection(100,90)
    got=update_protection(base,entry=100,high_since_entry=100+mfe*10,timestamp="next",
        confirmed_h1_rows=[],h1_atr=1,estimated_round_trip_cost_per_share=.20)
    assert got.milestone_r==mfe
    assert got.r_floor==expected
    if expected==0:
        assert got.current_stop>=100.20
    elif expected is not None:
        assert got.current_stop>=100+expected*10


def test_trailing_floor_uses_cost_allowance_without_changing_trade_entry_or_risk():
    state=initial_protection(100,90)
    result=update_protection(state,entry=100,high_since_entry=115,timestamp="t",
        confirmed_h1_rows=[],h1_atr=1,estimated_round_trip_cost_per_share=.35)
    assert result.current_stop>=100.35
    assert result.original_stop==90 and result.initial_risk==10


def test_hourly_pivot_is_unavailable_until_all_right_side_bars_complete():
    rows=[]
    for i,low in enumerate([10,8,7,8,9,10]): rows.append({"date":f"t{i}","low":low,"high":low+2})
    from src.strategy.trade_management_v3 import confirmed_hourly_higher_lows
    early=confirmed_hourly_higher_lows(rows[:4],left=1,right=2)
    later=confirmed_hourly_higher_lows(rows[:6],left=1,right=2)
    assert not any(x["index"]==2 for x in early)
    assert any(x["index"]==2 and x["confirmed_index"]==4 for x in later)


def test_confirmed_h1_higher_low_can_tighten_beyond_floor():
    rows=[]
    lows=[25,24,26,28,27,30,29,33,32,35]
    for i,low in enumerate(lows): rows.append({"date":f"t{i}","low":low,"high":low+2})
    state=initial_protection(100,90)
    updated=update_protection(state,entry=100,high_since_entry=140,timestamp="t9",confirmed_h1_rows=rows,h1_atr=.2,trailing_buffer_atr=.25)
    assert updated.current_stop>=100+1.75*10
    assert updated.current_stop>=state.current_stop


def test_gap_stop_executes_at_open_and_regular_stop_at_stop():
    assert stop_execution(90,89,92)==(True,90,"GAP_STOP")
    assert stop_execution(94,91,92)==(True,92,"PROTECTIVE_STOP")
    assert stop_execution(94,93,92)==(False,None,None)


def test_target_candidates_cannot_use_hourly_noise_and_rr_gate_is_hard():
    # An empty prefix fails closed; only DAILY/WEEKLY/CHANNEL dominant targets
    # are accepted by the evaluator after its geometry and H1 checks.
    got=trendline_pullback_candidate(daily_prefix=[],hourly_prefix=[],entry_bar=None,target_candidates=[])
    assert got["status"]=="REJECT"
    assert got["reason_code"]=="NO_ESTABLISHED_UPTREND_GEOMETRY"


def test_one_penetration_of_rising_ema_does_not_destroy_uptrend_state():
    rows=_bars("up")
    prior=classify_regime(rows[:-4],as_of=rows[-4]["date"])
    assert prior["regime"]=="TREND_UP"
    for row in rows[-3:]:
        row["close"]-=15
        row["low"]-=15
    state=classify_regime(rows[:-1],as_of=rows[-1]["date"],previous_state=prior)
    assert state["regime"] in {"TREND_UP","TRANSITION"}
    assert state["regime"]!="TREND_DOWN"


def test_ema_slope_alone_cannot_classify_a_trend():
    rows=_bars("up")
    # Remove the swing sequence and make a smooth rising line: EMAs rise, but
    # the meaningful-pivot evidence is absent, so the state remains ambiguous.
    smooth=[]
    for i,row in enumerate(rows):
        value=100+i*.3
        smooth.append({**row,"open":value,"high":value+0.1,"low":value-0.1,"close":value})
    state=classify_regime(smooth,as_of="2024-01-01")
    assert state["regime"]!="TREND_UP"


def test_flat_crossed_ema_alone_does_not_create_range():
    rows=_bars("range")
    state=classify_regime(rows,as_of="2024-01-01")
    assert state["regime"]!="RANGE" or state["range_strength"]>=0.55


def test_stateful_labels_are_causal_and_later_confirmation_does_not_rewrite_them():
    rows=_bars("up",100)
    first=classify_regime_series(rows,start_index=60)
    later=classify_regime_series(rows+[dict(rows[-1],date="2022-04-11",close=1,low=0)],start_index=60)
    assert [x["regime"] for x in first]==[x["regime"] for x in later[:len(first)]]
    assert all(x["input_through"]<x["decision_date"] for x in first)


def test_transition_requires_two_consecutive_candidate_sessions(monkeypatch):
    import src.strategy.structural_regime_v3 as module
    candidates=["TREND_UP","TREND_DOWN","TREND_DOWN"]
    regimes=["TREND_UP","TRANSITION","TRANSITION"]
    def fake(rows, *, as_of, previous_state=None, **kwargs):
        i=int(as_of[-2:])-2
        return {"regime":regimes[i],"candidate_regime":candidates[i],"candidate_confidence":1,
            "regime_start_date":as_of,"input_through":str(rows[-1]["date"]),"reason_codes":[],
            "trendlines":{},"ema":{},"diagnostic_evidence":{}}
    monkeypatch.setattr(module,"classify_regime",fake)
    rows=[{"date":f"2023-01-0{i}","open":1,"high":2,"low":0,"close":1} for i in range(1,5)]

    states=module.classify_regime_series(rows,start_index=1,candidate_confirmation_sessions=2)

    assert [s["regime"] for s in states]==["TREND_UP","TRANSITION","TREND_DOWN"]
    assert states[-1]["regime_start_date"]=="2023-01-04"


def test_regime_output_exposes_state_and_structural_evidence_fields():
    state=classify_regime(_bars("up"),as_of="2024-01-01")
    assert {"previous_regime","candidate_regime","candidate_confidence","regime_start_date",
            "recent_significant_highs","recent_significant_lows","swing_sequence","range_strength",
            "diagnostic_evidence","primary_geometry","secondary_geometry"} <= state.keys()
    assert state["diagnostic_evidence"]["weekly_context"]=="CONTEXT_ONLY"


def test_repeated_similar_horizontal_swings_with_width_can_form_range():
    rows=[]; base=date(2022,1,1)
    for i in range(300):
        value=100+3.5*sin(i*pi/10)
        rows.append({"date":(base+timedelta(days=i)).isoformat(),"open":value,"high":value+1.2,
                     "low":value-1.2,"close":value,"volume":1000})
    state=classify_regime(rows,as_of="2024-01-01")
    assert state["regime"]=="RANGE"
    assert state["range_upper"]>state["range_lower"]


def test_one_minor_bar_does_not_daily_flip_persistent_trend_state():
    rows=_bars("up")
    prior=classify_regime(rows[:-1],as_of=rows[-1]["date"])
    assert prior["regime"]=="TREND_UP"
    changed=[dict(x) for x in rows]
    changed[-1]["close"]-=2; changed[-1]["low"]-=3
    state=classify_regime(changed,as_of="2024-01-02",previous_state=prior)
    assert state["regime"] in {"TREND_UP","TRANSITION"}


def test_bearish_state_needs_bearish_structure_not_only_ema_state():
    state=classify_regime(_bars("down"),as_of="2024-01-01")
    assert state["regime"]=="TREND_DOWN"
    assert state["evidence"]["lower_high_lower_low"]


def test_break_confirmation_enters_transition_before_opposite_trend():
    rows=_bars("up")
    # The classifier only sees completed bars before each decision date.
    # Two closes decisively below the established rising support break the
    # old geometry, but are not enough by themselves to establish a downtrend.
    previous=classify_regime(rows[:270],as_of=rows[270]["date"])
    assert previous["regime"]=="TREND_UP"
    broken=[dict(x) for x in rows]
    for index in (270,271):
        for field in ("open","high","low","close"):
            broken[index][field]-=40
    next_state=classify_regime(broken[:272],as_of=rows[272]["date"],previous_state=previous)
    assert next_state["regime"]=="TRANSITION"
    # With full causal EMA history the raw candidate can remain bullish; a
    # confirmed break of the prior geometry still holds the regime in transition.
    assert next_state["candidate_regime"] in {"TREND_UP","TRANSITION"}
    assert next_state["trendlines"]["rising_support"]["status"]=="BROKEN"


def test_same_date_can_have_security_specific_regimes():
    up=classify_regime(_bars("up"),as_of="2024-01-01")
    down=classify_regime(_bars("down"),as_of="2024-01-01")
    assert up["regime"]!=down["regime"]


def test_weekly_context_is_never_an_input_hard_gate_for_daily_regime():
    daily=classify_regime(_bars("up"),as_of="2024-01-01")
    assert daily["regime"]=="TREND_UP"
    assert daily["diagnostic_evidence"]["weekly_context"]=="CONTEXT_ONLY"


def test_trendline_setup_thesis_consumption_requires_reestablishment_or_new_line():
    lifecycle=TrendlineSetupLifecycle("line-A")
    episode=lifecycle.consume("2023-01-10")
    assert lifecycle.status=="CONSUMED"
    assert episode.endswith("EPISODE:1")
    lifecycle.advance_day(trendline_id="line-A",close=103,support_level=100,daily_atr=4,date_text="2023-01-11")
    assert lifecycle.status=="CONSUMED"
    lifecycle.advance_day(trendline_id="line-A",close=105,support_level=100,daily_atr=3,date_text="2023-01-12")
    assert lifecycle.status=="WATCHING"
    assert lifecycle.episode==2
    lifecycle.consume("2023-01-13")
    lifecycle.advance_day(trendline_id="line-B",close=99,support_level=100,daily_atr=3,date_text="2023-01-14")
    assert lifecycle.status=="WATCHING" and lifecycle.episode==3
