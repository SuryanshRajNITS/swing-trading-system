from dataclasses import replace

import pytest

from src.strategy.sr_zone_v2 import Zone
from src.strategy.sr_zone_v2 import TouchEpisode
from src.strategy import top_down_swing_v2 as v2


def zone(side="support", low=100, high=102, strength="MEANINGFUL", seed="z"):
    return Zone(seed,"SEC",side,low,high,(low+high)/2,"2023-01-01","2023-01-01","2023-01-02",
        ("2023-01-01","2023-01-02"),(),(),False,None,"FRESH",strength=strength,reaction_count=3,
        mean_reaction_atr=1.0)


def test_daily_meaningful_support_triggers_watchlist_and_hourly_support_alone_does_not():
    rows=[{"date":f"2023-01-{i:02d}","open":104,"high":105,"low":103,"close":104,"volume":100} for i in range(1,15)]
    rows += [{"date":f"2023-01-{i:02d}","open":101,"high":103,"low":100,"close":101,"volume":100} for i in range(15,20)]
    location=v2.daily_location(rows,[zone()],v2.V2Config())
    assert location["state"]=="INSIDE_RECENT_DAILY_SUPPORT"
    assert v2.is_h1_watchlisted(location)
    no_daily=v2.daily_location(rows,[],v2.V2Config())
    assert no_daily["state"]=="NO_CLEAR_RECENT_DAILY_STRUCTURE"
    assert not v2.is_h1_watchlisted(no_daily)


def test_daily_watchlist_can_change_each_sod_with_location():
    support=zone(); resistance=zone("resistance",120,122,"MEANINGFUL","r")
    near=[{"date":f"2023-01-{i:02d}","open":104,"high":105,"low":103,"close":104,"volume":100} for i in range(1,15)]
    near += [{"date":f"2023-01-{i:02d}","open":101,"high":103,"low":100,"close":101,"volume":100} for i in range(15,20)]
    middle=[{"date":f"2023-01-{i:02d}","open":110,"high":112,"low":109,"close":110,"volume":100} for i in range(1,20)]
    assert v2.is_h1_watchlisted(v2.daily_location(near,[support,resistance],v2.V2Config()))
    assert not v2.is_h1_watchlisted(v2.daily_location(middle,[support,resistance],v2.V2Config()))


def test_daily_resistance_location_activates_breakout_watch_not_an_entry():
    resistance=zone("resistance",100,102,"MEANINGFUL","r")
    rows=[{"date":f"2023-02-{i:02d}","open":98,"high":100,"low":97,"close":99,"volume":100} for i in range(1,20)]
    location=v2.daily_location(rows,[resistance],v2.V2Config())
    assert location["state"]=="NEAR_DAILY_RESISTANCE_BREAKOUT_WATCH"
    activation=v2.daily_setup_activations(location,v2.V2Config())
    assert [x["family"] for x in activation]==["CONSOLIDATION_BREAKOUT"]


def test_role_flip_daily_location_activates_resistance_retest():
    role_flip=replace(zone("support",100,102,"MEANINGFUL","flip"),original_side="resistance",
                      role_state="SUPPORT_CONFIRMED",broken=True)
    rows=[{"date":f"2023-02-{i:02d}","open":103,"high":104,"low":102,"close":103,"volume":100} for i in range(1,15)]
    rows += [{"date":f"2023-02-{i:02d}","open":101,"high":103,"low":100,"close":101,"volume":100} for i in range(15,20)]
    location=v2.daily_location(rows,[role_flip],v2.V2Config())
    assert location["state"]=="RETESTING_BROKEN_RESISTANCE"
    assert v2.daily_setup_activations(location,v2.V2Config())[0]["family"]=="BREAKOUT_RETEST"


def test_waiting_for_role_flip_hold_is_not_reported_as_resistance_retest():
    waiting=replace(v2._as_v2_zone(zone("resistance",100,102,"MEANINGFUL","waiting")),
        original_side="resistance",role_state="BROKEN_RESISTANCE_WAITING_HOLD",broken=True)
    rows=[{"date":f"2023-02-{i:02d}","open":101,"high":102,"low":100,"close":101,"volume":100}
          for i in range(1,20)]
    location=v2.daily_location(rows,[waiting],v2.V2Config())
    assert location["state"]!="RETESTING_BROKEN_RESISTANCE"


def test_sod_recent_window_is_exact_and_excludes_current_day():
    rows=[{"date":f"2023-01-{i:02d}","close":float(i)} for i in range(1,32)]
    window=v2.recent_daily_window(rows,"2023-02-01",30)
    assert len(window)==30
    assert window[0]["date"]=="2023-01-02"
    assert window[-1]["date"]=="2023-01-31"
    assert "2023-02-01" not in [r["date"] for r in window]


def test_adaptive_window_uses_minimum_when_structure_is_sufficient(monkeypatch):
    support=zone("support",100,102,"MEANINGFUL","s"); resistance=zone("resistance",120,122,"MEANINGFUL","r")
    monkeypatch.setattr(v2,"enriched_zones",lambda rows,*a,**k:(support,resistance))
    rows=[{"date":f"2023-01-{i:02d}","open":105,"high":110,"low":100,"close":105,"volume":100} for i in range(1,51)]
    cfg=v2.V2Config(recent_daily_structure_min_sessions=45,recent_daily_structure_max_sessions=60)
    out=v2.adaptive_recent_structure(rows,sod_date="2023-02-20",security_id="SEC",cfg=cfg)
    assert out["sessions_used"]==45
    assert out["structure_sufficient"] is True


def test_adaptive_local_context_uses_persistent_map_without_rebuilding_it(monkeypatch):
    support=zone("support",100,102,"MEANINGFUL","s"); resistance=zone("resistance",120,122,"MEANINGFUL","r")
    calls=[]
    monkeypatch.setattr(v2,"enriched_zones",lambda rows,*a,**k:(calls.append(len(rows)) or (support,resistance)))
    rows=[{"date":f"2023-01-{i:02d}","open":105,"high":110,"low":100,"close":105,"volume":100} for i in range(1,71)]
    cfg=v2.V2Config(recent_daily_structure_min_sessions=45,recent_daily_structure_max_sessions=60)
    out=v2.adaptive_recent_structure(rows,sod_date="2023-02-20",security_id="SEC",cfg=cfg)
    assert out["sessions_used"]==45
    assert out["structure_sufficient"] is True
    assert calls==[70]


def test_adaptive_window_fails_sufficiency_closed_at_maximum(monkeypatch):
    monkeypatch.setattr(v2,"enriched_zones",lambda *a,**k:())
    rows=[{"date":f"2023-01-{i:02d}","open":105,"high":110,"low":100,"close":105,"volume":100} for i in range(1,71)]
    cfg=v2.V2Config(recent_daily_structure_min_sessions=45,recent_daily_structure_max_sessions=60)
    out=v2.adaptive_recent_structure(rows,sod_date="2023-02-20",security_id="SEC",cfg=cfg)
    assert out["sessions_used"]==60
    assert out["structure_sufficient"] is False


def test_adaptive_window_reports_insufficient_history_before_30_sessions(monkeypatch):
    monkeypatch.setattr(v2,"enriched_zones",lambda *a,**k:())
    rows=[{"date":f"2023-01-{i:02d}","open":105,"high":110,"low":100,"close":105,"volume":100} for i in range(1,20)]
    cfg=v2.V2Config(recent_daily_structure_min_sessions=45,recent_daily_structure_max_sessions=60)
    out=v2.adaptive_recent_structure(rows,sod_date="2023-01-20",security_id="SEC",cfg=cfg)
    assert out["sessions_used"]==19
    assert out["structure_sufficient"] is False
    assert out["extension_reason"]=="MAX_LOCAL_CONTEXT_REACHED_OR_INSUFFICIENT_HISTORY"


def test_near_support_state_can_use_older_major_only_when_revisited():
    rows=[{"date":f"2023-01-{i:02d}","open":104,"high":105,"low":103,"close":104,"volume":100} for i in range(1,15)]
    rows += [{"date":f"2023-01-{i:02d}","open":101,"high":103,"low":100,"close":101,"volume":100} for i in range(15,20)]
    out=v2.daily_location(rows,[],v2.V2Config(),older_zones=[replace(zone(),strength="MAJOR")])
    assert out["interesting_today"] is True
    assert out["support_provenance"]=="PERSISTENT_DAILY_STRUCTURE"


def test_distant_older_major_zone_does_not_dominate_recent_map():
    rows=[{"date":f"2023-01-{i:02d}","open":101,"high":103,"low":100,"close":101,"volume":100} for i in range(1,20)]
    far=replace(zone("support",70,72,"MAJOR","old-far"),first_pivot="2022-08-01")
    out=v2.daily_location(rows,[],v2.V2Config(),older_zones=[far])
    assert out["state"]=="NO_CLEAR_RECENT_DAILY_STRUCTURE"
    assert out["interesting_today"] is False


def test_zone_age_is_unavailable_not_negative_when_rendered_against_wrong_index_space():
    old_zone=replace(zone(),last_touch_index=30)
    rows=[{"date":f"2023-01-{i:02d}","close":101} for i in range(1,6)]
    out=v2.zone_row(old_zone,rows)
    assert out["age"] is None
    assert out["age_reference_mismatch"] is True
    assert out["last_touch_date"] is None
    # A valid-looking index can still belong to the full-history sequence.
    in_bounds_old=replace(old_zone,last_touch_index=3)
    out=v2.zone_row(in_bounds_old,rows,index_space_matches=False)
    assert out["age"] is None
    assert out["age_reference_mismatch"] is True


def test_older_major_resistance_near_current_price_remains_secondary_target_memory():
    rows=[{"date":f"2023-01-{i:02d}","open":104,"high":105,"low":103,"close":104,"volume":100} for i in range(1,15)]
    rows += [{"date":f"2023-01-{i:02d}","open":101,"high":103,"low":100,"close":101,"volume":100} for i in range(15,20)]
    old_r=replace(zone("resistance",103,105,"MAJOR","old-r"),first_pivot="2022-12-01")
    out=v2.daily_location(rows,[zone() ],v2.V2Config(),older_zones=[old_r])
    assert out["resistance"]==old_r
    assert out["resistance_provenance"]=="PERSISTENT_DAILY_STRUCTURE"


def test_long_memory_is_built_once_through_sod_and_local_window_stays_short(monkeypatch):
    calls=[]; support=zone(); resistance=zone("resistance",120,122,"MEANINGFUL","r")
    monkeypatch.setattr(v2,"enriched_zones",lambda rows,*a,**k:(calls.append(len(rows)) or (support,resistance)))
    rows=[{"date":f"2022-{(i//28)+1:02d}-{(i%28)+1:02d}","open":105,"high":110,"low":100,"close":105,"volume":100}
          for i in range(300)]
    out=v2.adaptive_recent_structure(rows,sod_date="2023-12-01",security_id="SEC",cfg=v2.V2Config())
    assert calls==[250]
    assert 45<=len(out["rows"])<=60
    assert out["structure_rows"][-1]["date"]<"2023-12-01"


def test_first_meaningful_reaction_validates_but_does_not_make_zone_tradeable(monkeypatch):
    rows=[{"date":f"2023-01-{i:02d}","open":101,"high":104,"low":100.5,"close":101,"volume":100}
          for i in range(1,31)]
    seed=replace(zone(),touch_episodes=(TouchEpisode(1,1,None,1),),original_side="support")
    captured={}
    def lifecycle(zones,data,zone_cfg,lifecycle_cfg):
        captured["recency_limit"]=lifecycle_cfg.meaningful_recency_bars
        return (v2._as_v2_zone(seed),)
    monkeypatch.setattr(v2,"_zones",lambda *a,**k:(seed,))
    monkeypatch.setattr(v2,"_lifecycle_snapshot",lifecycle)
    out=v2.enriched_zones(rows,"SEC",v2.V2Config().daily_zone,v2.V2Config())[0]
    assert captured["recency_limit"]>len(rows)
    assert out.validated is True
    assert out.tradeable is False
    assert "MEANINGFUL_POST_TOUCH_REACTION" in out.validation_reasons
    assert out.tradeability_reasons == ()
    assert out.strength=="MEANINGFUL"


def test_broken_support_stays_inactive_until_causal_reclaim_and_hold():
    original=zone("support",100,102,"MEANINGFUL","old-support")
    def bar(day,close,low=None,high=None):
        return {"date":f"2023-01-{day:02d}","open":close,"high":high or close+1,
            "low":low or close-1,"close":close,"volume":100}
    broken=[bar(1,101),bar(2,98),bar(3,98)]
    assert v2._v2_current_role(original,broken,v2.V2Config().daily_zone)[0]=="RESISTANCE_CANDIDATE"
    reclaimed=broken+[bar(4,103),bar(5,103),bar(6,103),bar(7,102.1,101,103)]
    assert v2._v2_current_role(original,reclaimed,v2.V2Config().daily_zone)[0]=="SUPPORT_CONFIRMED"


def test_broken_resistance_is_not_support_candidate_until_causal_hold_retest():
    resistance=zone("resistance",100,102,"MEANINGFUL","old-resistance")
    def bar(day,close,low=None,high=None):
        return {"date":f"2023-01-{day:02d}","open":close,"high":high or close+1,
            "low":low or close-1,"close":close,"volume":100}
    accepted=[bar(1,101),bar(2,103),bar(3,104),bar(4,105)]
    assert v2._v2_current_role(resistance,accepted,v2.V2Config().daily_zone)[0] == "BROKEN_RESISTANCE_WAITING_HOLD"
    held=accepted+[bar(5,103,101,104)]
    assert v2._v2_current_role(resistance,held,v2.V2Config().daily_zone)[0] == "SUPPORT_CONFIRMED"


def test_dominant_older_support_outranks_nearer_weaker_broken_zone(monkeypatch):
    dominant=replace(v2._as_v2_zone(zone("support",96,101,"MAJOR","dominant")),
        dominance_score=80,evidence_score=80,tradeable=True,validated=True)
    nearer=replace(v2._as_v2_zone(zone("resistance",100,104,"MAJOR","nearer")),
        original_side="support",side="resistance",role_state="RESISTANCE_CANDIDATE",
        broken=True,dominance_score=12,evidence_score=12,tradeable=True,validated=True)
    rows=[{"date":f"2023-01-{i:02d}","open":103,"high":105,"low":101,"close":103,"volume":100}
          for i in range(1,15)]
    rows.append({"date":"2023-01-15","open":101,"high":103,"low":100,"close":100.5,"volume":100})
    monkeypatch.setattr(v2,"ema",lambda rows,period:None)
    location=v2.daily_location(rows,[dominant,nearer],v2.V2Config())
    assert location["state"]=="INSIDE_RECENT_DAILY_SUPPORT"
    assert location["support"].zone_id=="dominant"
    assert location["dominant_zone"]["zone_id"]=="dominant"
    assert location["secondary_zones"][0]["priority_class"]=="LOCAL_SECONDARY_ZONE"
    assert location["distance_is_tiebreaker_only"] is True


def test_breakout_retest_is_not_required_after_hovering_near_established_resistance():
    resistance=replace(v2._as_v2_zone(zone("resistance",100,102,"MEANINGFUL","hovered")),hovering_bars=8)
    rows=[{"date":f"2023-02-{i:02d}","open":98,"high":100,"low":97,"close":99,"volume":100}
          for i in range(1,20)]
    location=v2.daily_location(rows,[resistance],v2.V2Config())
    activation=v2.daily_setup_activations(location,v2.V2Config())
    assert activation[0]["family"]=="CONSOLIDATION_BREAKOUT"
    assert activation[0]["setup_event_id"]


def test_support_deterioration_plus_available_ema_loss_waits_for_reclaim(monkeypatch):
    support=replace(v2._as_v2_zone(zone("support",100,102,"MAJOR","damaged")),
        dominance_score=40,tradeable=True,validated=True)
    rows=[{"date":f"2023-01-{i:02d}","open":104,"high":105,"low":103,"close":104,"volume":100}
          for i in range(1,53)]
    rows[-3].update(open=101.8,high=102.2,low=101.2,close=101.8)
    rows[-2].update(open=101.7,high=102.0,low=101.1,close=101.6)
    rows[-1].update(open=101.5,high=101.9,low=101.0,close=101.4)
    monkeypatch.setattr(v2,"ema",lambda rows,period:103.0 if period==50 else None)
    location=v2.daily_location(rows,[support],v2.V2Config())
    assert location["state"]=="WAIT_FOR_RECLAIM"
    assert "CLOSE_BELOW_EMA50" in location["support_damage_reasons"]
    assert "EMA200_UNAVAILABLE_EMA50_USED_WITH_LOCAL_SUPPORT_LOSS" in location["support_damage_reasons"]
    assert v2.daily_setup_activations(location,v2.V2Config())[0]["family"]=="RECLAIM"


def test_reclaim_watch_keeps_distinct_established_resistance_breakout_path():
    broken_support=replace(v2._as_v2_zone(zone("support",100,102,"MAJOR","broken")),
        broken=True,role_state="RESISTANCE_CANDIDATE",side="resistance",original_side="support")
    long_respected_resistance=replace(v2._as_v2_zone(zone("resistance",110,112,"MAJOR","long-respected")),
        original_side="resistance",role_state="RESISTANCE_CANDIDATE",hovering_bars=12,
        validated=True,tradeable=True,dominance_score=20)
    overlapping_resistance=replace(long_respected_resistance,zone_id="overlapping",lower_bound=111,
        upper_bound=113,center=112,dominance_score=10)
    location={"state":"WAIT_FOR_RECLAIM","broken_zone":broken_support,
        "activation_state":"BROKEN_SUPPORT_RECLAIM_WATCH",
        "direct_breakout_candidates":[long_respected_resistance,overlapping_resistance]}
    activations=v2.daily_setup_activations(location,v2.V2Config())
    assert [item["family"] for item in activations]==["RECLAIM","CONSOLIDATION_BREAKOUT"]
    assert activations[1]["zone"].zone_id=="long-respected"


def test_ema_loss_alone_does_not_globally_suppress_intact_support(monkeypatch):
    support=replace(v2._as_v2_zone(zone("support",100,102,"MAJOR","intact")),
        dominance_score=40,tradeable=True,validated=True)
    rows=[{"date":f"2023-01-{i:02d}","open":103,"high":104,"low":102.5,"close":103,"volume":100}
          for i in range(1,20)]
    monkeypatch.setattr(v2,"ema",lambda rows,period:105.0 if period==50 else None)
    location=v2.daily_location(rows,[support],v2.V2Config())
    assert location["state"]=="NEAR_RECENT_DAILY_SUPPORT"
    assert location["interesting_today"] is True


def test_daily_support_break_does_not_remain_a_buy_location():
    rows=[{"date":f"2023-01-{i:02d}","open":90,"high":92,"low":85,"close":88,"volume":100} for i in range(1,20)]
    broken=replace(zone(),broken=True,role_state="RESISTANCE_CONFIRMED",side="resistance",original_side="support")
    out=v2.daily_location(rows,[broken],v2.V2Config())
    assert out["state"]=="NO_CLEAR_RECENT_DAILY_STRUCTURE"
    assert not v2.is_h1_watchlisted(out)


def test_stop_and_target_use_daily_structure_and_no_fallback():
    cfg=v2.V2Config()
    resistance=zone("resistance",125,127,"MAJOR","r")
    p=v2.structural_target(105,95,resistance,[],2,cfg)
    assert p["target"] < resistance.lower_bound
    assert p["target_timeframe"]=="DAILY"
    assert p["planned_structural_rr"] >= 1.5
    assert v2.structural_target(105,95,None,[],2,cfg)["reason_code"]=="NO_MEANINGFUL_STRUCTURAL_TARGET"


def test_nearer_weekly_resistance_caps_daily_target():
    daily=zone("resistance",140,142,"MAJOR","d")
    weekly=zone("resistance",125,127,"MAJOR","w")
    plan=v2.structural_target(105,95,daily,[weekly],2,v2.V2Config())
    assert plan["target_timeframe"]=="WEEKLY"
    assert plan["target"] < weekly.lower_bound


def test_nearest_meaningful_resistance_is_intermediate_not_terminal_target():
    near=replace(v2._as_v2_zone(zone("resistance",110,112,"MEANINGFUL","near")),
        dominance_score=10,dominance_reasons=("two_interaction_episodes",))
    dominant=replace(v2._as_v2_zone(zone("resistance",130,133,"MAJOR","dominant")),
        dominance_score=40,dominance_reasons=("four_reactions",))
    plan=v2.structural_target(105,95,[near,dominant],[],2,v2.V2Config())
    assert plan["target_zone_id"]=="dominant"
    assert plan["target_classification"]=="DOMINANT"
    assert plan["planned_structural_rr"]>2
    assert plan["intermediate_structural_friction"][0]["zone_id"]=="near"
    assert plan["intermediate_structural_friction"][0]["classification"]=="SECONDARY"


def test_overlapping_daily_weekly_dominant_structure_is_one_target_area():
    daily=replace(v2._as_v2_zone(zone("resistance",130,133,"MAJOR","daily")),dominance_score=30)
    weekly=replace(v2._as_v2_zone(zone("resistance",131,134,"MEANINGFUL","weekly")),dominance_score=20)
    near=replace(v2._as_v2_zone(zone("resistance",110,112,"MEANINGFUL","near")),dominance_score=5)
    plan=v2.structural_target(105,95,[near,daily],[weekly],2,v2.V2Config())
    assert plan["target_zone_id"]=="daily"
    assert {x["timeframe"] for x in plan["structural_resistance_ladder"]
            if x["classification"]=="DOMINANT"}=={"DAILY","WEEKLY"}
    assert plan["intermediate_structural_friction"][0]["zone_id"]=="near"


def test_parent_zone_uses_evidenced_inner_boundary_for_breakout_activation():
    inner=replace(v2._as_v2_zone(zone("resistance",711,725,"MEANINGFUL","inner")),
        parent_zone_id="parent",structure_level_kind="INNER_BOUNDARY",validated=True,dominance_score=12)
    parent=replace(v2._as_v2_zone(zone("resistance",725,740,"MAJOR","parent")),
        nested_boundaries=(inner,),validated=True,dominance_score=30)
    location={"state":"NEAR_DAILY_RESISTANCE_BREAKOUT_WATCH","nearest_resistance":parent,
        "resistance_hovering_bars":8,"current_reference_price":729.7,"daily_atr":10}
    activation=v2.daily_setup_activations(location,v2.V2Config())[0]
    assert activation["zone"].zone_id=="inner"
    assert activation["zone"].parent_zone_id=="parent"


def test_parent_preserves_lower_edge_only_after_independent_revisit_evidence(monkeypatch):
    parent=replace(v2._as_v2_zone(zone("resistance",105,110,"MEANINGFUL","parent")),
        original_side="resistance",validated=True)
    monkeypatch.setattr(v2,"_zones",lambda *a,**k:(parent,))
    monkeypatch.setattr(v2,"detect_zones",lambda *a,**k:())
    episodes=(TouchEpisode(0,2,None,3),TouchEpisode(8,12,5,5))
    monkeypatch.setattr(v2,"_lifecycle_snapshot",lambda source,*a,**k:
        tuple(replace(z,touch_episodes=episodes) for z in source))
    rows=[{"date":f"2023-01-{i+1:02d}","open":103,"high":111,"low":99,"close":104,"volume":100}
          for i in range(30)]
    zones=v2.enriched_zones(rows,"SEC",v2.V2Config().daily_zone,v2.V2Config())
    enriched_parent=next(z for z in zones if z.zone_id=="parent")
    edge=next(z for z in enriched_parent.nested_boundaries if z.structure_level_kind=="INNER_BOUNDARY")
    assert edge.parent_zone_id=="parent"
    assert edge.boundary_price==105
    assert (edge.lower_bound,edge.upper_bound)==(100,105)
    assert len(edge.touch_episodes)==2


def test_nested_breakout_boundary_does_not_become_a_duplicate_terminal_target():
    parent=replace(v2._as_v2_zone(zone("resistance",125,127,"MAJOR","parent")),dominance_score=30)
    edge=replace(v2._as_v2_zone(zone("resistance",123,125,"MAJOR","edge")),
        structure_level_kind="INNER_BOUNDARY",parent_zone_id="parent",boundary_price=125,
        dominance_score=50)
    plan=v2.structural_target(105,95,[parent,edge],[],2,v2.V2Config())
    assert plan["target_zone_id"]=="parent"
    assert [item["zone_id"] for item in plan["structural_resistance_ladder"]]==["parent"]


def test_breakout_acceptance_can_be_recovered_from_active_resistance_state(monkeypatch):
    reference=replace(v2._as_v2_zone(zone("resistance",105,107,"MEANINGFUL","active")),
        original_side="resistance",role_state="ACTIVE")
    bars=[{"date":f"2023-01-02T{9+i%6:02d}:15:00+05:30","open":104,"high":105,"low":103,"close":104,"volume":100}
          for i in range(24)]
    bars[20].update(open=107.2,high=108.5,low=107.1,close=108.2,volume=1000)
    bars[21].update(open=108.1,high=109.5,low=107.8,close=109.0,volume=1100)
    bars[22].update(open=107.5,high=109.0,low=106.9,close=108.2,volume=120)
    bars[23].update(open=108.5,high=109,low=108,close=108.7,volume=1000)
    monkeypatch.setattr(v2,"volume_features",lambda rows,lookback:{
        "same_time_expansion":float(rows[-1]["volume"])>=1000,
        "expansion_vs_prior_mean":float(rows[-1]["volume"])>=1000,
        "same_time_contraction":float(rows[-1]["volume"])<1000})
    result,rejected=v2.hourly_entry(hourly_rows=bars,start_index=22,support=reference,daily_stop=100,
        daily_resistance=[zone("resistance",130,132,"MAJOR","target")],weekly_zones=[],daily_atr=2,
        security_id="SEC",cfg=v2.V2Config(),end_index=23,setup_family="BREAKOUT_RETEST")
    assert not rejected
    assert result is not None
    assert result["breakout_retest_provenance"]["breakout_timestamp"]==bars[20]["date"]
    assert result["breakout_retest_provenance"]["acceptance_state_after_breakout"]=="ACCEPTED"


def _bars():
    return [
      {"date":"2023-01-02T15:15:00+05:30","open":101,"high":102,"low":100,"close":101,"volume":100},
      {"date":"2023-01-03T10:15:00+05:30","open":101,"high":104,"low":100,"close":103,"volume":150},
      {"date":"2023-01-03T11:15:00+05:30","open":104,"high":106,"low":103,"close":105,"volume":120},
    ]


def test_confirmation_uses_completed_hourly_bar_and_next_open(monkeypatch):
    monkeypatch.setattr(v2,"enriched_zones",lambda *a,**k:())
    monkeypatch.setattr(v2,"volume_features",lambda *a,**k:{"same_time_contraction":True,"same_time_expansion":False,"expansion_vs_prior_mean":False})
    bars=_bars(); bars[0]["close"]=102.5
    result,rejected=v2.hourly_entry(hourly_rows=bars,start_index=1,support=zone(),daily_stop=90,
        daily_resistance=zone("resistance",130,132,"MAJOR","r"),weekly_zones=[],daily_atr=2,security_id="SEC",cfg=v2.V2Config())
    assert not rejected
    assert result["confirmation_timestamp"]==bars[1]["date"]
    assert result["entry_timestamp"]==bars[2]["date"]
    assert result["plan"]["entry"]==104


def test_strong_resistance_blocks_until_close_and_volume_clearance(monkeypatch):
    friction=zone("resistance",103,104,"MEANINGFUL","friction")
    calls={"n":0}
    def zones(*a,**k):
        calls["n"]+=1
        return (friction,) if calls["n"]==1 else ()
    monkeypatch.setattr(v2,"enriched_zones",zones)
    monkeypatch.setattr(v2,"volume_features",lambda *a,**k:{"same_time_expansion":True,"expansion_vs_prior_mean":False,"same_time_contraction":False})
    bars=_bars(); bars[0]["close"]=102.5
    bars[1].update(high=103.5,close=102.5) # wick through, close remains beneath the zone
    bars[2].update(open=103,high=108,low=101,close=107,volume=300)
    bars.append({"date":"2023-01-03T12:15:00+05:30","open":107,"high":108,"low":106,"close":107,"volume":100})
    result,rejected=v2.hourly_entry(hourly_rows=bars,start_index=1,support=zone(),daily_stop=95,
        daily_resistance=zone("resistance",140,142,"MAJOR","r"),weekly_zones=[],daily_atr=2,security_id="SEC",cfg=v2.V2Config())
    assert result is not None
    assert result["confirmation_timestamp"]==bars[2]["date"]
    assert result["entry_timestamp"]==bars[3]["date"]
    assert result["plan"]["friction_clearance_required"]


def test_breakout_retest_requires_volume_confirmed_acceptance_then_retest(monkeypatch):
    reference=replace(zone("resistance",105,107,"MEANINGFUL","breakout"),original_side="resistance")
    target=zone("resistance",130,132,"MAJOR","target")
    bars=[{"date":f"2023-01-02T{9+i%6:02d}:15:00+05:30","open":104,"high":105,"low":103,"close":104,"volume":100}
          for i in range(24)]
    bars[20].update(open=107.2,high=108.5,low=107.1,close=108.2,volume=1000)
    bars[21].update(open=108.1,high=109.5,low=107.8,close=109.0,volume=1100)
    bars[22].update(open=107.5,high=109.0,low=106.9,close=108.2,volume=120)
    bars[23].update(open=108.5,high=109,low=108,close=108.7,volume=1000)
    monkeypatch.setattr(v2,"volume_features",lambda rows,lookback: {
        "same_time_expansion":float(rows[-1]["volume"])>=1000,
        "expansion_vs_prior_mean":float(rows[-1]["volume"])>=1000,
        "same_time_contraction":float(rows[-1]["volume"])<1000,
        "relative_volume":float(rows[-1]["volume"])/1000,"volume":float(rows[-1]["volume"])})
    result,why=v2.hourly_entry(hourly_rows=bars,start_index=20,support=reference,daily_stop=100,
        daily_resistance=[target],weekly_zones=[],daily_atr=2,security_id="SEC",cfg=v2.V2Config(),
        end_index=23,setup_family="BREAKOUT_RETEST")
    assert not why
    assert result["setup_family"]=="BREAKOUT_RETEST"
    assert result["confirmation_timestamp"]==bars[22]["date"]
    assert result["entry_timestamp"]==bars[23]["date"]
    assert result["plan"]["volume_confirmation_passed"] is True
    evidence=result["breakout_retest_provenance"]
    assert evidence["breakout_timestamp"]==bars[20]["date"]
    assert evidence["acceptance_state_after_breakout"]=="ACCEPTED"
    assert evidence["breakout_volume_passed"] is True
    assert evidence["retest"]["retest_tested_from_above"] is True
    assert evidence["retest"]["retest_subtype"]=="BREAKOUT_RETEST_TOUCH"
    assert evidence["retest"]["touched_zone"] is True
    assert evidence["retest"]["minimum_price_during_retest"]==106.9
    assert evidence["retest"]["minimum_distance_to_zone"]==0
    assert evidence["retest"]["retest_response_timestamp"]==bars[22]["date"]
    assert evidence["retest"]["retest_response_type"]=="BULLISH_CLOSE_HOLD_ABOVE_CENTER"
    assert evidence["retest"]["retest_volume_confirmation"]["passed"] is True
    assert evidence["original_breakout_timestamp"]==bars[20]["date"]
    assert evidence["original_breakout_volume_confirmation"] is True
    assert evidence["retest"]["volume_evidence"]["features"]["same_time_contraction"] is True
    assert evidence["retest"]["volume_passed"] is True
    assert evidence["final_confirmation_timestamp"]==bars[22]["date"]


def test_breakout_front_run_requires_later_bullish_response_and_records_subtype(monkeypatch):
    reference=replace(zone("resistance",105,107,"MEANINGFUL","breakout"),original_side="resistance")
    target=zone("resistance",130,132,"MAJOR","target")
    bars=[{"date":f"2023-01-02T{9+i%6:02d}:15:00+05:30","open":104,"high":105,"low":103,"close":104,"volume":100}
          for i in range(25)]
    bars[20].update(open=107.2,high=108.5,low=107.1,close=108.2,volume=1000)
    bars[21].update(open=108.1,high=109.5,low=107.8,close=109.0,volume=1100)
    # Near-only approach: no zone contact and no bullish response on this bar.
    bars[22].update(open=108.1,high=108.4,low=107.5,close=107.8,volume=120)
    bars[23].update(open=107.8,high=109.0,low=107.6,close=108.4,volume=1000)
    bars[24].update(open=108.5,high=109,low=108,close=108.7,volume=1000)
    monkeypatch.setattr(v2,"volume_features",lambda rows,lookback:{
        "same_time_expansion":float(rows[-1]["volume"])>=1000,
        "expansion_vs_prior_mean":float(rows[-1]["volume"])>=1000,
        "same_time_contraction":float(rows[-1]["volume"])<1000,
        "relative_volume":float(rows[-1]["volume"])/1000,"volume":float(rows[-1]["volume"])})
    result,why=v2.hourly_entry(hourly_rows=bars,start_index=20,support=reference,daily_stop=100,
        daily_resistance=[target],weekly_zones=[],daily_atr=2,security_id="SEC",cfg=v2.V2Config(),
        end_index=22,setup_family="BREAKOUT_RETEST")
    assert result is None
    assert not any(x.get("reason_code")=="ENTRY_READY" for x in why)
    result,why=v2.hourly_entry(hourly_rows=bars,start_index=22,support=reference,daily_stop=100,
        daily_resistance=[target],weekly_zones=[],daily_atr=2,security_id="SEC",cfg=v2.V2Config(),
        end_index=24,setup_family="BREAKOUT_RETEST",setup_state={
            "accepted_breakout":True,"breakout_acceptance_count":2,"breakout_acceptance_index":21,
            "breakout_provenance":{"breakout_zone":{"zone_id":"breakout","lower_bound":105,"upper_bound":107},
                "breakout_timestamp":bars[20]["date"],"breakout_volume_passed":True,
                "acceptance_state_after_breakout":"ACCEPTED"}})
    assert not why
    assert result is not None
    retest=result["breakout_retest_provenance"]["retest"]
    assert retest["retest_subtype"]=="BREAKOUT_RETEST_FRONT_RUN"
    assert retest["touched_zone"] is False
    assert retest["approached_from_above"] is True
    assert retest["retest_tested_from_above"] is False
    assert retest["minimum_price_during_retest"]==107.5
    assert retest["minimum_distance_to_zone"]>0
    assert retest["minimum_distance_to_zone_atr"]>0
    assert retest["retest_timestamp"]==bars[22]["date"]
    assert retest["retest_response_timestamp"]==bars[23]["date"]
    assert retest["retest_response_type"]=="BULLISH_CLOSE_HOLD_ABOVE_CENTER"
    assert retest["retest_volume_confirmation"]["passed"] is True


def test_breakout_retest_cannot_skip_original_breakout_volume_confirmation(monkeypatch):
    reference=replace(zone("resistance",105,107,"MEANINGFUL","breakout"),original_side="resistance")
    bars=[{"date":f"2023-01-02T{9+i%6:02d}:15:00+05:30","open":104,"high":105,"low":103,"close":104,"volume":100}
          for i in range(24)]
    bars[20].update(open=107.2,high=108.5,low=107.1,close=108.2,volume=1000)
    bars[21].update(open=108.1,high=109.5,low=107.8,close=109.0,volume=1100)
    bars[22].update(open=107.5,high=109.0,low=106.9,close=108.2,volume=1200)
    monkeypatch.setattr(v2,"volume_features",lambda *a,**k:{"same_time_expansion":False,
        "expansion_vs_prior_mean":False,"same_time_contraction":True,"volume":100})
    result,_=v2.hourly_entry(hourly_rows=bars,start_index=20,support=reference,daily_stop=100,
        daily_resistance=[zone("resistance",130,132,"MAJOR","target")],weekly_zones=[],daily_atr=2,
        security_id="SEC",cfg=v2.V2Config(),end_index=23,setup_family="BREAKOUT_RETEST")
    assert result is None


def test_role_flipped_breakout_retest_records_causal_historical_breakout(monkeypatch):
    reference=replace(zone("support",105,107,"MEANINGFUL","breakout"),original_side="resistance",
                     role_state="SUPPORT_CONFIRMED",broken=True)
    bars=[{"date":f"2023-01-02T{9+i%6:02d}:15:00+05:30","open":104,"high":105,"low":103,"close":104,"volume":100}
          for i in range(24)]
    bars[20].update(open=107.2,high=108.5,low=107.1,close=108.2,volume=1000)
    bars[21].update(open=108.1,high=109.5,low=107.8,close=109.0,volume=1100)
    bars[22].update(open=107.5,high=109.0,low=106.9,close=108.2,volume=120)
    bars[23].update(open=108.5,high=109,low=108,close=108.7,volume=1000)
    monkeypatch.setattr(v2,"volume_features",lambda rows,lookback: {
        "same_time_expansion":float(rows[-1]["volume"])>=1000,
        "expansion_vs_prior_mean":float(rows[-1]["volume"])>=1000,
        "same_time_contraction":float(rows[-1]["volume"])<1000,
        "relative_volume":float(rows[-1]["volume"])/1000,"volume":float(rows[-1]["volume"])})
    result,_=v2.hourly_entry(hourly_rows=bars,start_index=22,support=reference,daily_stop=100,
        daily_resistance=[zone("resistance",130,132,"MAJOR","target")],weekly_zones=[],daily_atr=2,
        security_id="SEC",cfg=v2.V2Config(),end_index=23,setup_family="BREAKOUT_RETEST")
    assert result is not None
    provenance=result["breakout_retest_provenance"]
    assert provenance["breakout_timestamp"]==bars[20]["date"]
    assert provenance["accepted_timestamp"]==bars[21]["date"]
    assert provenance["breakout_volume_passed"] is True
    assert provenance["retest"]["volume_passed"] is True


def test_historical_breakout_acceptance_is_reset_by_close_below_zone(monkeypatch):
    reference=replace(zone("resistance",105,107,"MEANINGFUL","breakout"),original_side="resistance")
    bars=[{"date":f"2023-01-02T{9+i:02d}:15:00+05:30","open":104,"high":105,"low":103,"close":104,"volume":100}
          for i in range(5)]
    bars[0].update(open=108,high=110,low=107,close=109,volume=1000)
    bars[1].update(open=109,high=111,low=108,close=110,volume=1100)
    bars[2].update(open=104,high=105,low=99,close=99,volume=100)
    monkeypatch.setattr(v2,"volume_features",lambda rows,lookback:{
        "same_time_expansion":float(rows[-1]["volume"])>=1000,
        "expansion_vs_prior_mean":float(rows[-1]["volume"])>=1000})
    assert v2._historical_breakout_acceptance(bars,reference,v2.V2Config(),3) is None


def test_next_open_below_daily_zone_is_rejected_even_after_bullish_h1_close(monkeypatch):
    monkeypatch.setattr(v2,"enriched_zones",lambda *a,**k:())
    monkeypatch.setattr(v2,"volume_features",lambda *a,**k:{"same_time_contraction":True,
        "same_time_expansion":False,"expansion_vs_prior_mean":False})
    bars=_bars(); bars[0]["close"]=102.5
    bars[1].update(open=101,high=104,low=100,close=103)
    bars[2].update(open=99,high=102,low=98,close=101)
    result,rejected=v2.hourly_entry(hourly_rows=bars,start_index=1,support=zone(),daily_stop=95,
        daily_resistance=zone("resistance",130,132,"MAJOR","target"),weekly_zones=[],daily_atr=2,
        security_id="SEC",cfg=v2.V2Config())
    assert result is None
    assert any(x["reason_code"]=="ENTRY_GAPPED_BELOW_STRUCTURAL_ZONE" for x in rejected)


def test_role_flipped_support_close_below_zone_invalidates_breakout_retest_watch(monkeypatch):
    flipped=replace(zone("support",100,102,"MAJOR","flip"),original_side="resistance",
                    role_state="SUPPORT_CONFIRMED",broken=True)
    bars=_bars(); bars[0]["close"]=103; bars[1].update(open=99,high=101,low=98,close=99)
    result,rejected=v2.hourly_entry(hourly_rows=bars,start_index=1,support=flipped,daily_stop=95,
        daily_resistance=zone("resistance",130,132,"MAJOR","target"),weekly_zones=[],daily_atr=2,
        security_id="SEC",cfg=v2.V2Config(),setup_family="BREAKOUT_RETEST")
    assert result is None
    assert rejected[0]["reason_code"]=="DAILY_SUPPORT_BROKEN"


def test_breakout_acceptance_state_carries_across_incremental_replay_calls():
    reference=replace(zone("resistance",105,107,"MEANINGFUL","breakout"),original_side="resistance")
    target=zone("resistance",130,132,"MAJOR","target")
    bars=[{"date":f"2023-01-02T{9+i%6:02d}:15:00+05:30","open":104,"high":105,"low":103,"close":104,"volume":100}
          for i in range(24)]
    bars[20].update(open=107.2,high=108.5,low=107.1,close=108.2,volume=1000)
    bars[21].update(open=108.1,high=109.5,low=107.8,close=109.0,volume=1100)
    bars[22].update(open=107.5,high=109.0,low=106.9,close=108.2,volume=1200)
    bars[23].update(open=108.5,high=109,low=108,close=108.7,volume=1000)
    state={}
    first,why=v2.hourly_entry(hourly_rows=bars,start_index=20,support=reference,daily_stop=100,
        daily_resistance=[target],weekly_zones=[],daily_atr=2,security_id="SEC",cfg=v2.V2Config(),
        end_index=22,setup_family="BREAKOUT_RETEST",setup_state=state)
    assert first is None and not why
    assert state["accepted_breakout"] is True
    second,why=v2.hourly_entry(hourly_rows=bars,start_index=22,support=reference,daily_stop=100,
        daily_resistance=[target],weekly_zones=[],daily_atr=2,security_id="SEC",cfg=v2.V2Config(),
        end_index=23,setup_family="BREAKOUT_RETEST",setup_state=state)
    assert not why
    assert second is not None
    assert second["confirmation_timestamp"]==bars[22]["date"]


def test_consolidation_breakout_requires_compression_break_and_expansion():
    support=zone("support",90,95,"MAJOR","support")
    target=zone("resistance",130,132,"MAJOR","target")
    bars=[]
    for i in range(24):
        if i<14: bar={"open":97,"high":100,"low":96,"close":98,"volume":100}
        elif i<20: bar={"open":99.2,"high":100,"low":99,"close":99.7,"volume":100}
        else: bar={"open":100.1,"high":101,"low":100,"close":100.8,"volume":1000}
        bar["date"]=f"2023-01-02T{9+i%6:02d}:15:00+05:30";bars.append(bar)
    bars[21].update(open=100.8,high=101.8,low=100.7,close=101.5,volume=1100)
    bars[22].update(open=101.4,high=102,low=101.2,close=101.8,volume=1200)
    bars[23].update(open=101.7,high=102,low=101.5,close=101.9,volume=900)
    result,why=v2.hourly_entry(hourly_rows=bars,start_index=20,support=support,daily_stop=88,
        daily_resistance=[target],weekly_zones=[],daily_atr=2,security_id="SEC",cfg=v2.V2Config(),
        end_index=23,setup_family="CONSOLIDATION_BREAKOUT")
    assert not why
    assert result["setup_family"]=="CONSOLIDATION_BREAKOUT"
    assert result["plan"]["volume_confirmation_passed"] is True


def _direct_breakout_bars():
    bars=[]
    for i in range(15):
        bars.append({"date":f"2023-11-06T09:{15+i:02d}:00+05:30","open":100.2,
            "high":101.0,"low":99.0,"close":100.6,"volume":100})
    bars[14].update(open=102.1,high=106.2,low=101.8,close=106.0,volume=1000)
    bars.append({"date":"2023-11-06T14:30:00+05:30","open":106.1,
        "high":107.0,"low":105.5,"close":106.7,"volume":1000})
    return bars


def _long_respected_resistance():
    return replace(v2._as_v2_zone(zone("resistance",100,102,"MAJOR","direct")),
        side="resistance",original_side="support",role_state="RESISTANCE_CANDIDATE",
        broken=True,validated=True,tradeable=True,hovering_bars=12,failed_crosses=4,
        dominance_score=60)


def test_direct_structural_breakout_executes_without_retest(monkeypatch):
    monkeypatch.setattr(v2,"enriched_zones",lambda *a,**k:())
    monkeypatch.setattr(v2,"volume_features",lambda *a,**k:{"same_time_expansion":True,
        "expansion_vs_prior_mean":True,"same_time_contraction":False,"relative_volume":2.0})
    bars=_direct_breakout_bars()
    result,rejected=v2.hourly_entry(hourly_rows=bars,start_index=0,support=_long_respected_resistance(),
        daily_stop=95,daily_resistance=zone("resistance",130,132,"MAJOR","target"),weekly_zones=[],
        daily_atr=2,security_id="SEC",cfg=v2.V2Config(),end_index=15,
        setup_family="CONSOLIDATION_BREAKOUT")
    assert result is not None
    assert result["setup_family"]=="CONSOLIDATION_BREAKOUT"
    assert result["breakout_path"]=="DIRECT_STRUCTURAL_BREAKOUT"
    assert result["entry_type"]=="NEXT_AVAILABLE_1H_OPEN"
    assert result["direct_breakout_evidence"]["established_resistance"] is True
    assert result["direct_breakout_evidence"]["volume_expansion_passed"] is True
    assert result["direct_breakout_evidence"]["range_expansion_passed"] is True
    assert result["plan"]["planned_reward_r"]>=v2.V2Config().minimum_structural_rr
    assert not result["breakout_retest_provenance"]


def test_direct_breakout_rejects_weak_candle_and_missing_volume(monkeypatch):
    monkeypatch.setattr(v2,"enriched_zones",lambda *a,**k:())
    monkeypatch.setattr(v2,"volume_features",lambda *a,**k:{"same_time_expansion":True,
        "expansion_vs_prior_mean":True,"same_time_contraction":False})
    weak=_direct_breakout_bars(); weak[14].update(open=104.4,high=106.2,low=101.8,close=106.0)
    result,_=v2.hourly_entry(hourly_rows=weak,start_index=0,support=_long_respected_resistance(),
        daily_stop=95,daily_resistance=zone("resistance",130,132,"MAJOR","target"),weekly_zones=[],
        daily_atr=2,security_id="SEC",cfg=v2.V2Config(),end_index=15,
        setup_family="CONSOLIDATION_BREAKOUT")
    assert result is None
    strong=_direct_breakout_bars()
    monkeypatch.setattr(v2,"volume_features",lambda *a,**k:{"same_time_expansion":False,
        "expansion_vs_prior_mean":False,"same_time_contraction":False})
    result,_=v2.hourly_entry(hourly_rows=strong,start_index=0,support=_long_respected_resistance(),
        daily_stop=95,daily_resistance=zone("resistance",130,132,"MAJOR","target"),weekly_zones=[],
        daily_atr=2,security_id="SEC",cfg=v2.V2Config(),end_index=15,
        setup_family="CONSOLIDATION_BREAKOUT")
    assert result is None


def test_direct_breakout_requires_room_to_next_structural_resistance(monkeypatch):
    monkeypatch.setattr(v2,"enriched_zones",lambda *a,**k:())
    monkeypatch.setattr(v2,"volume_features",lambda *a,**k:{"same_time_expansion":True,
        "expansion_vs_prior_mean":True,"same_time_contraction":False})
    result,rejected=v2.hourly_entry(hourly_rows=_direct_breakout_bars(),start_index=0,
        support=_long_respected_resistance(),daily_stop=95,
        daily_resistance=zone("resistance",108,109,"MAJOR","too-close"),weekly_zones=[],
        daily_atr=2,security_id="SEC",cfg=v2.V2Config(),end_index=15,
        setup_family="CONSOLIDATION_BREAKOUT")
    assert result is None
    assert any(x["reason_code"] in {"RR_INSUFFICIENT","NO_MEANINGFUL_STRUCTURAL_TARGET"} for x in rejected)


def test_direct_breakout_waits_for_strong_hourly_resistance_friction_clearance(monkeypatch):
    overhead=zone("resistance",106.1,106.5,"MAJOR","h1-friction")
    monkeypatch.setattr(v2,"enriched_zones",lambda *a,**k:(overhead,))
    monkeypatch.setattr(v2,"volume_features",lambda *a,**k:{"same_time_expansion":True,
        "expansion_vs_prior_mean":True,"same_time_contraction":False})
    result,rejected=v2.hourly_entry(hourly_rows=_direct_breakout_bars(),start_index=0,
        support=_long_respected_resistance(),daily_stop=95,
        daily_resistance=zone("resistance",130,132,"MAJOR","target"),weekly_zones=[],
        daily_atr=2,security_id="SEC",cfg=v2.V2Config(),end_index=15,
        setup_family="CONSOLIDATION_BREAKOUT")
    assert result is None
    assert any(item["reason_code"]=="WAITING_FOR_HOURLY_FRICTION_CLEARANCE" for item in rejected)


def test_minor_hourly_resistance_does_not_block_and_ema_without_respect_is_soft(monkeypatch):
    minor=zone("resistance",103,104,"MINOR","minor")
    monkeypatch.setattr(v2,"enriched_zones",lambda *a,**k:(minor,))
    monkeypatch.setattr(v2,"volume_features",lambda *a,**k:{"same_time_contraction":True,"same_time_expansion":False,"expansion_vs_prior_mean":False})
    monkeypatch.setattr(v2,"ema",lambda rows,period:101.5)
    monkeypatch.setattr(v2,"_strong_ema_rejections",lambda *a,**k:0)
    bars=_bars();bars[0]["close"]=102.5
    out,_=v2.hourly_entry(hourly_rows=bars,start_index=1,support=zone(),daily_stop=90,
        daily_resistance=zone("resistance",130,132,"MAJOR","r"),weekly_zones=[],daily_atr=2,security_id="SEC",cfg=v2.V2Config())
    assert out is not None
    assert out["plan"]["friction_clearance_required"] is False


def test_respected_ema_is_strong_friction_and_requires_clearance(monkeypatch):
    monkeypatch.setattr(v2,"enriched_zones",lambda *a,**k:())
    monkeypatch.setattr(v2,"volume_features",lambda *a,**k:{"same_time_expansion":True,"same_time_contraction":False,"expansion_vs_prior_mean":True})
    monkeypatch.setattr(v2,"ema",lambda rows,period:101.5)
    monkeypatch.setattr(v2,"_strong_ema_rejections",lambda *a,**k:3)
    bars=_bars();bars[0]["close"]=102.5;bars[1].update(open=101,high=102,low=100,close=101.5);bars[2].update(open=102,high=107,low=101,close=106)
    bars.append({"date":"2023-01-03T12:15:00+05:30","open":106,"high":107,"low":105,"close":106,"volume":100})
    out,_=v2.hourly_entry(hourly_rows=bars,start_index=1,support=zone(),daily_stop=90,
        daily_resistance=zone("resistance",140,142,"MAJOR","r"),weekly_zones=[],daily_atr=2,security_id="SEC",cfg=v2.V2Config())
    assert out is not None
    assert out["plan"]["friction_clearance_required"]
    assert any(x["type"]=="EMA50" and x["strong"] for x in out["plan"]["friction"])


def test_hourly_resistance_and_ema_confluence_is_recorded_as_strong_block(monkeypatch):
    resistance=zone("resistance",103,104,"MEANINGFUL","friction")
    monkeypatch.setattr(v2,"enriched_zones",lambda *a,**k:(resistance,))
    monkeypatch.setattr(v2,"volume_features",lambda *a,**k:{"same_time_contraction":True,"same_time_expansion":False,"expansion_vs_prior_mean":False})
    monkeypatch.setattr(v2,"ema",lambda rows,period:103.5)
    monkeypatch.setattr(v2,"_strong_ema_rejections",lambda *a,**k:0)
    bars=_bars();bars[0]["close"]=102.5
    out,rejected=v2.hourly_entry(hourly_rows=bars,start_index=1,support=zone(),daily_stop=90,
        daily_resistance=zone("resistance",130,132,"MAJOR","r"),weekly_zones=[],daily_atr=2,security_id="SEC",cfg=v2.V2Config())
    assert out is None
    assert any("HOURLY_SR_EMA_CONFLUENCE" in x.get("friction_types",[]) for x in rejected)


def test_clearance_with_consumed_rr_is_rejected(monkeypatch):
    friction=zone("resistance",102,103,"MAJOR","f")
    monkeypatch.setattr(v2,"enriched_zones",lambda *a,**k:(friction,))
    monkeypatch.setattr(v2,"volume_features",lambda *a,**k:{"same_time_expansion":True,"expansion_vs_prior_mean":True,"same_time_contraction":False})
    bars=_bars(); bars[0]["close"]=102.5; bars[1].update(open=101,high=106,low=100,close=105)
    bars[2].update(open=104,high=105,low=103,close=104)
    result,rejected=v2.hourly_entry(hourly_rows=bars,start_index=1,support=zone(),daily_stop=90,
        daily_resistance=zone("resistance",110,112,"MAJOR","r"),weekly_zones=[],daily_atr=2,security_id="SEC",cfg=v2.V2Config())
    assert result is None
    assert any(x["reason_code"] in {"CLEARANCE_CONSUMED_STRUCTURAL_ROOM","RR_INSUFFICIENT"} for x in rejected)


def test_volume_required_and_hostile_bearish_acceptance_rejected(monkeypatch):
    monkeypatch.setattr(v2,"enriched_zones",lambda *a,**k:())
    monkeypatch.setattr(v2,"volume_features",lambda *a,**k:{"same_time_contraction":False,"same_time_expansion":False,"expansion_vs_prior_mean":False})
    bars=_bars(); bars[0]["close"]=102.5
    result,rejected=v2.hourly_entry(hourly_rows=bars,start_index=1,support=zone(),daily_stop=90,
        daily_resistance=zone("resistance",125,127,"MAJOR","r"),weekly_zones=[],daily_atr=2,security_id="SEC",cfg=v2.V2Config())
    assert result is None
    assert any(x["reason_code"]=="VOLUME_CONFIRMATION_MISSING" for x in rejected)
    bearish=[dict(x) for x in bars];bearish[1].update(open=103,high=104,low=98,close=99)
    monkeypatch.setattr(v2,"volume_features",lambda *a,**k:{"same_time_contraction":False,"same_time_expansion":True,"expansion_vs_prior_mean":True})
    result,rejected=v2.hourly_entry(hourly_rows=bearish,start_index=1,support=zone(),daily_stop=90,
        daily_resistance=zone("resistance",125,127,"MAJOR","r"),weekly_zones=[],daily_atr=2,security_id="SEC",cfg=v2.V2Config())
    assert result is None
    assert any(x["reason_code"]=="HOSTILE_BEARISH_VOLUME_AT_DAILY_SUPPORT" for x in rejected)
