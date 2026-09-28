"""Causal daily-structure-first signal replay for top_down_swing_v2."""
from __future__ import annotations

import argparse, csv, json
from bisect import bisect_left
from collections import Counter, defaultdict
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo
import yaml

from src.backtest.data import BacktestDataError, load_daily_dataset_through, load_hourly_dataset_through
from src.backtest.weekly_selection_run import load_nifty_benchmark
from src.configuration import PROJECT_ROOT
from src.data.market_data import DATA_ROOT, MANIFEST_PATH
from src.data.universe.provider import PIT_TRADABLE, UniverseProvider
from src.strategy.sr_zone_v2 import ZoneConfig
from src.strategy.top_down_swing import _as_datetime
from src.strategy.top_down_swing_v2 import V2Config, adaptive_recent_structure, daily_location, daily_setup_activations, enriched_zones, hourly_entry, zone_row
from src.strategy.weekly_selection import WeeklySelectionConfig, completed_weekly_rows, weekly_features
from src.strategy.weekly_regime import WeeklyRegimeConfig, weekly_regime

START, END = date(2023,1,1), date(2024,12,31)
DAILY_WARMUP, HOURLY_WARMUP = date(2022,7,1), date(2022,10,1)
TZ=ZoneInfo("Asia/Kolkata")
CONFIG=PROJECT_ROOT/"config"/"top_down_swing_v2.yaml"
BENCHMARK=PROJECT_ROOT/"data"/"raw"/"reference"/"nifty50_daily.csv"
OUTPUT=PROJECT_ROOT/"data"/"processed"/"backtests"/"top_down_swing_v2_daily_structure"


def _csv(path: Path, rows: Sequence[Mapping[str,Any]]) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for r in rows for k in r)) or ["empty"]
    with path.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore"); w.writeheader()
        for row in rows:
            w.writerow({k:json.dumps(v,sort_keys=True,ensure_ascii=True,default=str) if isinstance(v,(dict,list,tuple)) else v for k,v in row.items()})


def _cfg(raw: Mapping[str,Any]) -> V2Config:
    dz=raw.get("daily_zones",{}); hz=raw.get("hourly_zones",{}); wz=raw.get("weekly_zones",{})
    common=dict(minimum_pivots=int(dz.get("minimum_pivots",2)), maximum_age=int(dz.get("maximum_age",252)),
                minimum_width_pct=float(dz.get("minimum_width_pct",.0025)), maximum_width_pct=float(dz.get("maximum_width_pct",.01)),
                cluster_distance_atr=float(dz.get("cluster_distance_atr",.75)), departure_atr=float(dz.get("departure_atr",.5)),
                reaction_horizon=int(dz.get("reaction_horizon",5)))
    loc=raw.get("location",{}); hourly=raw.get("hourly",{}); vol=raw.get("volume",{})
    daily_lifecycle=raw.get("daily_zone_lifecycle",{})
    return V2Config(structural_memory_sessions=int(loc.get("structural_memory_sessions",250)),
       recent_daily_structure_min_sessions=int(loc.get("recent_daily_structure_min_sessions",45)),
       recent_daily_structure_max_sessions=int(loc.get("recent_daily_structure_max_sessions",60)),
       support_watch_distance_atr=float(loc.get("support_watch_distance_atr",1)),
       stop_buffer_atr=float(loc.get("stop_buffer_atr",.25)), target_front_run_buffer_atr=float(loc.get("target_front_run_buffer_atr",.1)),
       minimum_structural_rr=float(loc.get("minimum_structural_rr",1.5)), friction_distance_atr=float(hourly.get("friction_distance_atr",.5)),
       zone_break_buffer_atr=float(daily_lifecycle.get("break_buffer_atr",.10)),
       zone_acceptance_closes=int(daily_lifecycle.get("acceptance_closes",2)),
       direct_breakout_min_body_fraction=float(hourly.get("direct_breakout_min_body_fraction",.5)),
       direct_breakout_min_close_location=float(hourly.get("direct_breakout_min_close_location",.75)),
       direct_breakout_min_range_ratio=float(hourly.get("direct_breakout_min_range_ratio",1.0)),
       clearance_buffer_atr=float(hourly.get("clearance_buffer_atr",.1)), confirmation_expiry_sessions=int(hourly.get("confirmation_expiry_sessions",10)),
       consolidation_bars=int(hourly.get("consolidation_bars",6)),breakout_acceptance_closes=int(hourly.get("breakout_acceptance_closes",2)),
       ema_rejection_lookback_bars=int(hourly.get("ema_rejection_lookback_bars",20)), ema_min_rejections=int(hourly.get("ema_min_rejections",3)),
       volume_lookback=int(vol.get("lookback",20)),
       daily_zone=ZoneConfig(swing_left=int(dz.get("swing_left",3)),swing_right=int(dz.get("swing_right",3)),**common),
       weekly_zone=ZoneConfig(swing_left=int(wz.get("swing_left",2)),swing_right=int(wz.get("swing_right",2)), minimum_pivots=2,
          maximum_age=104,minimum_width_pct=.003,maximum_width_pct=.02),
       hourly_zone=ZoneConfig(swing_left=int(hourly.get("swing_left",3)),swing_right=int(hourly.get("swing_right",3)),minimum_pivots=2,maximum_age=130))


def _refresh_consumed_theses(consumed: dict[tuple[str,str],dict[str,Any]], security_id: str,
                             zones: Sequence[Any], current_close: float) -> list[dict[str,Any]]:
    """Release a consumed zone thesis only after map removal or role/invalidation change."""
    current={zone.zone_id:zone for zone in zones}
    released=[]
    for key,record in list(consumed.items()):
        if key[0]!=security_id:
            continue
        zone=current.get(key[1])
        if zone is None:
            zone=next((candidate for candidate in zones if
                min(candidate.upper_bound,float(record["upper_bound"]))>=
                max(candidate.lower_bound,float(record["lower_bound"]))),None)
        reason=None
        if zone is None:
            reason="ZONE_REMOVED_FROM_CURRENT_STRUCTURE_MAP"
        elif zone.role_state!=record["role_state"] or zone.side!=record["side"]:
            reason="ZONE_ROLE_CHANGED"
        elif record["side"]=="support" and current_close<float(record["lower_bound"]):
            reason="LONG_SUPPORT_THESIS_INVALIDATED_BY_CLOSE_BELOW_ZONE"
        if reason:
            released.append({"security_id":security_id,"zone_id":key[1],"reason_code":reason})
            consumed.pop(key,None)
    return released


def _consume_zone_thesis(consumed: dict[tuple[str,str],dict[str,Any]], security_id: str, zone: Any,
                         lifecycle: Sequence[Mapping[str,Any]], setup_event_id: str = "LEGACY") -> None:
    consumed[(security_id,zone.zone_id)]={"role_state":zone.role_state,"side":zone.side,
        "lower_bound":zone.lower_bound,"upper_bound":zone.upper_bound,
        "consumed_lifecycle":list(lifecycle),"setup_event_id":setup_event_id}


def _zone_thesis_consumed(consumed: Mapping[tuple[str,str],Any], security_id: str,
                          zone: Any, setup_event_id: str = "LEGACY") -> bool:
    """Suppress duplicate executions from one interaction, allow later revisits."""
    record=consumed.get((security_id,zone.zone_id))
    return bool(record and record.get("setup_event_id")==setup_event_id)


def _lifecycle_transition(watch: dict[str,Any], status: str, timestamp: str,
                          **evidence: Any) -> None:
    if any(item.get("status")==status for item in watch.get("lifecycle",[])):
        return
    watch.setdefault("lifecycle",[]).append({"status":status,"timestamp":timestamp,**evidence})
    watch["lifecycle_status"]=status


def replay(*, start:date=START,end:date=END,config_path:Path=CONFIG,output_dir:Path=OUTPUT,
           data_root:Path=DATA_ROOT,daily_manifest:Path=MANIFEST_PATH,hourly_manifest:Path|None=None,
           benchmark_path:Path=BENCHMARK,max_days:int|None=None) -> dict[str,Any]:
    if start<START or end>END or start>end: raise ValueError("V2 replay is restricted to the frozen 2023-2024 period.")
    raw=yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if raw.get("version")!="top_down_swing_v2_daily_structure": raise ValueError("V2 config version mismatch.")
    cfg=_cfg(raw)
    daily,dhash=load_daily_dataset_through(universe_id=PIT_TRADABLE,start_date=DAILY_WARMUP,end_date=end,data_root=data_root,manifest_path=daily_manifest)
    hourly,hhash=load_hourly_dataset_through(universe_id=PIT_TRADABLE,start_date=HOURLY_WARMUP,end_date=end,data_root=data_root,manifest_path=hourly_manifest,daily_manifest_path=daily_manifest)
    # The accepted user-supplied benchmark is the frozen bounded 2023-2024 file;
    # per-date weekly helpers exclude every week not completed by that SOD.
    benchmark=load_nifty_benchmark(benchmark_path,start=DAILY_WARMUP,end=END)
    provider=UniverseProvider(); dates=sorted({date.fromisoformat(r["date"][:10]) for rows in daily.values() for r in rows if start<=date.fromisoformat(r["date"][:10])<=end})
    if max_days: dates=dates[:max_days]
    trace=[]; ready=[]; rejects=[]; zone_dump=[]; location_rows=[]; counts=Counter(); archetypes=Counter(); setup_counts=Counter()
    daily_idx={sid:{date.fromisoformat(r["date"][:10]):i for i,r in enumerate(rows)} for sid,rows in daily.items()}
    hourly_stamps={sid:[_as_datetime(r["date"]) for r in rows] for sid,rows in hourly.items()}
    hourly_day_bounds={}
    hourly_days={}
    for sid,stamps in hourly_stamps.items():
        bounds={}
        for i,stamp in enumerate(stamps):
            d=stamp.date()
            bounds[d]=(bounds[d][0],i) if d in bounds else (i,i)
        hourly_day_bounds[sid]=bounds; hourly_days[sid]=sorted(bounds)
    session_ordinal={sid:{d:i for i,d in enumerate(sorted(index))} for sid,index in daily_idx.items()}
    weekly_cfg=WeeklySelectionConfig(); regime_cfg=WeeklyRegimeConfig()
    feature_cache={}; nifty_cache={}; active_watches={}; consumed_theses={}; consumed_suppressed=0
    first_causal_date=None; first_ready_date=None
    sod_activations=Counter(); weekly_statuses=Counter(); setup_order={"SUPPORT_REACTION":0,"RECLAIM":1,
        "BREAKOUT_RETEST":2,"FAILED_BREAKDOWN_RECLAIM":3,"HL_HH_CONFIRMATION":4,"CONSOLIDATION_BREAKOUT":5}
    for day in dates:
        members={m.security_id:m for m in provider.members_on(day,PIT_TRADABLE)}
        for sid,member in sorted(members.items()):
            if sid not in daily_idx or day not in daily_idx[sid]: continue
            idx=daily_idx[sid][day]
            # Start Of Day D has only D-1 and earlier daily candles available.
            dp=daily[sid][:idx]
            if not dp: continue
            week_key=date.fromordinal(day.toordinal()-day.weekday()).isoformat()
            cache_key=(sid,week_key)
            if cache_key not in feature_cache:
                feature_cache[cache_key]=weekly_features(stock_daily=dp,benchmark_daily=benchmark,as_of=day,
                    security_id=sid,config=weekly_cfg,zone_config=cfg.weekly_zone)
            weekly_selection=feature_cache[cache_key]
            weekly_status=str(weekly_selection.get("status") or "insufficient_history")
            weekly_statuses[weekly_status]+=1
            benchmark_ready=(weekly_selection.get("relative_strength_4w") is not None and
                             weekly_selection.get("relative_strength_13w") is not None)
            resistance_exception=(weekly_status=="neutral" and
                weekly_selection.get("reason")=="nearby_weekly_resistance_breakout_exception_only" and benchmark_ready)
            weekly_pass=weekly_status=="eligible" or resistance_exception
            if weekly_pass and first_causal_date is None:
                first_causal_date=day.isoformat()
            if weekly_pass: counts["WEEKLY_SELECTION_PASSED"]+=1
            if week_key not in nifty_cache:
                nifty_cache[week_key]=weekly_regime(nifty_daily=benchmark,as_of=day,config=regime_cfg)
            nifty_context=nifty_cache[week_key]
            if not weekly_pass:
                counts["WEEKLY_SELECTION_REJECTED"]+=1
                active_watches={key:value for key,value in active_watches.items() if key[0]!=sid}
                location_rows.append({"sod_date":day.isoformat(),"security_id":sid,"symbol":member.symbol,
                    "weekly_selector_status":weekly_status,"weekly_selector_reason":weekly_selection.get("reason"),
                    "weekly_selection_components":weekly_selection,"nifty_regime":nifty_context.get("regime"),
                    "recent_daily_location_state":"WEEKLY_SELECTION_REJECTED","weekly_gate_passed":False,
                    "daily_window_candles":[]})
                trace.append({"date":day.isoformat(),"security_id":sid,"symbol":member.symbol,
                    "state":"WEEKLY_SELECTION_REJECTED","reason_codes":[weekly_selection.get("reason") or weekly_status],
                    "weekly_selection":weekly_selection,"nifty_regime":nifty_context.get("regime")})
                continue
            adaptive=adaptive_recent_structure(daily[sid],sod_date=day.isoformat(),security_id=sid,cfg=cfg)
            recent=adaptive["rows"]
            if not recent: continue
            weeks=completed_weekly_rows(dp,day)
            dz=adaptive["zones"]; structure_rows=adaptive["structure_rows"]
            loc=daily_location(recent,dz,cfg,structure_rows=structure_rows); state=loc["state"]
            older=[]
            activations=daily_setup_activations(loc,cfg)
            if resistance_exception:
                activations=[item for item in activations if item["family"] in {"BREAKOUT_RETEST","CONSOLIDATION_BREAKOUT"}]
            current_structure_zones=tuple(dz)
            released=_refresh_consumed_theses(consumed_theses,sid,current_structure_zones,float(recent[-1]["close"]))
            for item in released:
                trace.append({"date":day.isoformat(),"security_id":sid,"symbol":member.symbol,
                    "state":"SETUP_INVALIDATED","reason_codes":[item["reason_code"]],
                    "setup_lifecycle":{"status":"INVALIDATED","zone_id":item["zone_id"]}})
            unconsumed_activations=[]
            for item in activations:
                if _zone_thesis_consumed(consumed_theses,sid,item["zone"],item["setup_event_id"]):
                    consumed_suppressed+=1
                else:
                    unconsumed_activations.append(item)
            activations=unconsumed_activations
            # SOD activation counters distinguish location from an H1 trigger.
            if state in {"NEAR_RECENT_DAILY_SUPPORT","NEAR_DAILY_SUPPORT"}: sod_activations["NEAR_DAILY_SUPPORT"]+=1
            elif state in {"INSIDE_RECENT_DAILY_SUPPORT","INSIDE_DAILY_SUPPORT"}: sod_activations["INSIDE_DAILY_SUPPORT"]+=1
            elif state=="NEAR_DAILY_RESISTANCE_BREAKOUT_WATCH": sod_activations["NEAR_DAILY_RESISTANCE_BREAKOUT_WATCH"]+=1
            elif state=="RETESTING_BROKEN_RESISTANCE": sod_activations["RETESTING_BROKEN_RESISTANCE"]+=1
            elif state in {"DAILY_SUPPORT_BROKEN","BROKEN_SUPPORT_RECLAIM_WATCH"}: sod_activations["BROKEN_SUPPORT_RECLAIM_WATCH"]+=1
            elif state in {"MID_RANGE","MID_RECENT_DAILY_RANGE"}: sod_activations["MID_RANGE"]+=1
            counts["daily_observations"]+=1
            sod_fields={"sod_date":day.isoformat(),"security_id":sid,"symbol":member.symbol,
                "recent_window_start":str(recent[0]["date"])[:10],"recent_window_end":str(recent[-1]["date"])[:10],
                "recent_window_sessions":len(recent),"recent_window_sessions_used":adaptive["sessions_used"],
                "recent_window_extension_reason":adaptive["extension_reason"],
                "recent_window_structure_sufficient":adaptive["structure_sufficient"],
                "local_context_sufficient":adaptive["local_context_sufficient"],
                "structural_memory_sessions":adaptive["memory_sessions"],
                "structural_memory_start":str(structure_rows[0]["date"])[:10],
                "structural_memory_end":str(structure_rows[-1]["date"])[:10],
                "weekly_gate_passed":weekly_pass,"weekly_selector_status":weekly_status,
                "weekly_selector_reason":weekly_selection.get("reason"),"weekly_selection_components":weekly_selection,
                "weekly_structure":{"higher_high":weekly_selection.get("higher_high"),"higher_low":weekly_selection.get("higher_low"),
                    "structure_score":weekly_selection.get("structure_score")},
                "nifty_regime":nifty_context.get("regime"),"nifty_regime_components":nifty_context.get("component_values"),
                "sod_activated_families":[item["family"] for item in activations],
                "persistent_support_zone_ids":[z.zone_id for z in dz if z.side=="support" and z.strength in {"MEANINGFUL","MAJOR"}],
                "persistent_resistance_zone_ids":[z.zone_id for z in dz if z.side=="resistance" and z.strength in {"MEANINGFUL","MAJOR"}],
                "nearest_recent_support_lower":loc.get("support_row",{}).get("lower_bound"),"nearest_recent_support_upper":loc.get("support_row",{}).get("upper_bound"),
                "nearest_recent_resistance_lower":(loc.get("resistance_row") or loc.get("nearest_resistance_row") or {}).get("lower_bound"),
                "nearest_recent_resistance_upper":(loc.get("resistance_row") or loc.get("nearest_resistance_row") or {}).get("upper_bound"),
                "resistance_provenance":loc.get("resistance_provenance","RECENT_DAILY_STRUCTURE"),
                "support_provenance":loc.get("support_provenance","RECENT_DAILY_STRUCTURE"),
                "current_reference_price":float(recent[-1]["close"]),"distance_to_support_pct":loc.get("support_distance_pct"),
                "distance_to_support_atr":loc.get("support_distance_atr"),"distance_to_resistance_pct":loc.get("resistance_distance_pct"),
                "distance_to_resistance_atr":loc.get("resistance_distance_atr"),"recent_daily_location_state":state,
                "interesting_today":bool(loc.get("interesting_today",False)),"interesting_today_reason":loc.get("interesting_today_reason"),
                "older_structure_memory_used":loc.get("older_structure_memory_used",False),"older_structure_reference":loc.get("older_structure_reference"),
                "daily_window_candles":recent}
            location_rows.append(sod_fields)
            if activations:
                counts["watchlist_candidates"]+=1; counts["approaching_support_events"]+=1
                if "DAILY_SUPPORT_TESTING" in loc.get("reason_codes",[]): counts["daily_support_tests"]+=1
                zone_dump.extend({**zone_row(z,structure_rows),"date":day.isoformat(),"symbol":member.symbol,"timeframe":"DAILY","provenance":"PERSISTENT_DAILY_STRUCTURE"} for z in dz)
            for activation in activations:
                family=activation["family"]; zone=activation["zone"]
                key=(sid,family,zone.zone_id)
                existing_watch=active_watches.get(key)
                if existing_watch and existing_watch.get("setup_event_id")!=activation["setup_event_id"]:
                    _lifecycle_transition(existing_watch,"INVALIDATED",day.isoformat(),
                        reason_code="NEW_DISTINCT_STRUCTURAL_REVISIT")
                    active_watches.pop(key,None)
                watch=active_watches.setdefault(key,{"security_id":sid,"symbol":member.symbol,"family":family,
                    "zone":zone,"activation_date":day,"activation_session":session_ordinal[sid][day],
                    "setup_event_id":activation["setup_event_id"],
                    "activation_state":state,"activation_location":sod_fields,"activation_atr":float(loc.get("daily_atr") or atr(recent,14)),
                    "zone_diagnostic":zone_row(zone,structure_rows),
                    "last_processed_index":None,"hourly_state":{},"lifecycle_status":"WATCHING",
                    "lifecycle":[{"status":"WATCHING","timestamp":day.isoformat()}]})
                watch["last_location_date"]=day.isoformat()
            # Existing daily-location watches remain active for their configured expiry, like V1 SETUP_ACTIVE.
            same_day_watches=sorted((w for k,w in active_watches.items() if k[0]==sid),
                key=lambda w:(setup_order.get(w["family"],99),w["activation_date"],w["zone"].zone_id))
            if same_day_watches and sid not in hourly:
                counts["HOURLY_DATA_UNAVAILABLE"]+=len(same_day_watches)
                for w in same_day_watches:
                    rejects.append({"date":day.isoformat(),"security_id":sid,"symbol":member.symbol,
                        "setup_family":w["family"],"reason_code":"HOURLY_DATA_UNAVAILABLE"})
                active_watches={key:value for key,value in active_watches.items() if key[0]!=sid}
                continue
            if same_day_watches:
                bars=hourly[sid]; stamps=hourly_stamps[sid]
                final=hourly_day_bounds.get(sid,{}).get(day,(-1,-1))[1]
                weekly_zones=enriched_zones(weeks,sid,cfg.weekly_zone,cfg) if weeks else ()
                for watch in same_day_watches:
                    age=session_ordinal[sid][day]-int(watch["activation_session"])
                    key=(sid,watch["family"],watch["zone"].zone_id)
                    if age>cfg.confirmation_expiry_sessions:
                        _lifecycle_transition(watch,"INVALIDATED",day.isoformat(),reason_code="SETUP_EXPIRED")
                        rejects.append({"date":day.isoformat(),"security_id":sid,"symbol":member.symbol,
                            "setup_family":watch["family"],"reason_code":"SETUP_EXPIRED",
                            "setup_lifecycle":watch["lifecycle"]})
                        active_watches.pop(key,None); continue
                    days=hourly_days.get(sid,[]); day_pos=bisect_left(days,watch["activation_date"])
                    first=hourly_day_bounds[sid][days[day_pos]][0] if day_pos<len(days) else len(bars)
                    previously_processed=watch.get("last_processed_index")
                    next_unprocessed=first if previously_processed is None else max(first,int(previously_processed)+1)
                    target_floor=(watch["zone"].upper_bound if watch["family"]=="BREAKOUT_RETEST"
                                  else float(recent[-1]["close"]))
                    daily_resistance=[z for z in dz if z.side=="resistance" and z.lower_bound>target_floor
                                      and z.strength in {"MEANINGFUL","MAJOR"}
                                      and (not z.broken or z.role_state in {"RESISTANCE_CONFIRMED","RESISTANCE_CANDIDATE"})]
                    daily_stop=watch["zone"].lower_bound-watch["activation_atr"]*cfg.stop_buffer_atr
                    result,why=hourly_entry(hourly_rows=bars,start_index=next_unprocessed,support=watch["zone"],daily_stop=daily_stop,
                        daily_resistance=daily_resistance,weekly_zones=weekly_zones,daily_atr=watch["activation_atr"],
                        security_id=sid,cfg=cfg,end_index=final,setup_family=watch["family"],
                        setup_state=watch["hourly_state"])
                    # hourly_entry reserves the last available bar because its
                    # next open is the executable fill. Revisit that bar on the
                    # next SOD when the next session's open becomes available.
                    watch["last_processed_index"]=max(int(previously_processed if previously_processed is not None else first-1),final-1)
                    h1state=watch["hourly_state"]
                    if watch["family"]=="BREAKOUT_RETEST" and h1state.get("accepted_breakout"):
                        provenance=h1state.get("breakout_provenance") or {}
                        accepted_at=str(provenance.get("accepted_timestamp") or day.isoformat())
                        _lifecycle_transition(watch,"BREAKOUT_ACCEPTED",accepted_at,
                            breakout_timestamp=provenance.get("breakout_timestamp"),
                            breakout_volume_passed=provenance.get("breakout_volume_passed"))
                        _lifecycle_transition(watch,"RETEST_WAIT",accepted_at)
                    counts["hourly_confirmation_count"]+=int(bool(result) or bool(why))
                    for item in why:
                        rejects.append({"date":day.isoformat(),"security_id":sid,"symbol":member.symbol,
                            "setup_family":watch["family"],**item})
                        counts[str(item.get("reason_code") or "UNKNOWN")]+=1
                    if result:
                        family=result.get("setup_family",watch["family"])
                        result["setup_family"]=family
                        _lifecycle_transition(watch,"CONFIRMED",str(result["confirmation_timestamp"]))
                        _lifecycle_transition(watch,"ENTRY_READY",str(result["confirmation_timestamp"]),
                            executable_timestamp=str(result["entry_timestamp"]))
                        _lifecycle_transition(watch,"CONSUMED",str(result["entry_timestamp"]))
                        fill=float(result["plan"]["entry"]); target=float(result["plan"]["target"])
                        result["plan"]["intermediate_daily_friction"]=[zone_row(z,structure_rows) for z in dz
                            if z.side=="resistance" and z.strength in {"MEANINGFUL","MAJOR"}
                            and fill<z.lower_bound<target and z.zone_id!=result["plan"].get("target_zone_id")
                            and z.strength!="MAJOR"]
                        setup_id=f"{sid}-{family}-{watch['zone'].zone_id}-{watch['activation_date'].isoformat()}"
                        row={"date":day.isoformat(),"setup_signal_date":watch["activation_date"].isoformat(),
                            "security_id":sid,"symbol":member.symbol,"state":"ENTRY_READY","setup_type":family,
                            "setup_family":family,"setup_instance_id":setup_id,
                            "setup_event_id":watch["setup_event_id"],"daily_location_state":watch["activation_state"],
                            "setup_lifecycle":watch["lifecycle"],
                            "support":watch["zone_diagnostic"],
                            "resistance":[zone_row(z,structure_rows) for z in daily_resistance],
                            "daily_zones":[zone_row(z,structure_rows) for z in dz],"weekly_selection_path":"WEEKLY_STOCK_SELECTION",
                            "weekly_selection_components":weekly_selection,"weekly_structure":{"higher_high":weekly_selection.get("higher_high"),
                                "higher_low":weekly_selection.get("higher_low"),"structure_score":weekly_selection.get("structure_score")},
                            "nifty_regime":nifty_context.get("regime"),"result":result,"plan":result["plan"]}
                        br_provenance=result.get("breakout_retest_provenance") or {}
                        row["retest_subtype"]=(br_provenance.get("retest") or {}).get("retest_subtype")
                        trace.append(row); ready.append({**row,"confirmation_timestamp":result["confirmation_timestamp"],
                            "executable_timestamp":result["entry_timestamp"],"result":result,"plan":result["plan"]})
                        counts["entry_ready_events"]+=1; archetypes[str(result.get("confirmation_type"))]+=1; setup_counts[family]+=1
                        first_ready_date=day.isoformat() if first_ready_date is None else min(first_ready_date,day.isoformat())
                        active_watches.pop(key,None)
                        consumed_zone=next((z for z in current_structure_zones if z.zone_id==watch["zone"].zone_id),watch["zone"])
                        _consume_zone_thesis(consumed_theses,sid,consumed_zone,watch["lifecycle"],watch["setup_event_id"])
                        active_watches={active_key:active_value for active_key,active_value in active_watches.items()
                            if not (active_key[0]==sid and active_key[2]==watch["zone"].zone_id)}
                        # Only one position candidate per stock/day; other families remain auditable watches.
                        break
                    if any(x.get("reason_code") in {"DAILY_SUPPORT_BROKEN","SETUP_INVALIDATED"} for x in why) and watch["family"] not in {"RECLAIM","FAILED_BREAKDOWN_RECLAIM"}:
                        reason=next(x for x in why if x.get("reason_code") in {"DAILY_SUPPORT_BROKEN","SETUP_INVALIDATED"})
                        _lifecycle_transition(watch,"INVALIDATED",str(reason.get("timestamp") or day.isoformat()),
                            reason_code=reason.get("reason_code"))
                        trace.append({"date":day.isoformat(),"security_id":sid,"symbol":member.symbol,
                            "state":"SETUP_INVALIDATED","reason_codes":[reason.get("reason_code")],
                            "setup_family":watch["family"],"setup_instance_id":key,
                            "setup_lifecycle":watch["lifecycle"]})
                        active_watches.pop(key,None)
                if not any(row.get("security_id")==sid and row.get("date")==day.isoformat() and row.get("state")=="ENTRY_READY" for row in trace[-max(1,len(same_day_watches)):]):
                    trace.append({"date":day.isoformat(),"security_id":sid,"symbol":member.symbol,"state":"SETUP_ACTIVE",
                        "reason_codes":["WAITING_FOR_CAUSAL_H1_SETUP"],"daily_location_state":state,
                        "weekly_selection":weekly_selection,"nifty_regime":nifty_context.get("regime")})
            else:
                if state in {"MID_RANGE","MID_RECENT_DAILY_RANGE"}: counts["MID_RANGE"]+=1
                trace.append({"date":day.isoformat(),"security_id":sid,"symbol":member.symbol,"state":state,
                    "reason_codes":loc.get("reason_codes",[]),"daily_zones":loc.get("daily_zones",[]),
                    "weekly_selection":weekly_selection,"nifty_regime":nifty_context.get("regime")})
    output_dir.mkdir(parents=True,exist_ok=True)
    setup_families=("SUPPORT_REACTION","BREAKOUT_RETEST","RECLAIM","CONSOLIDATION_BREAKOUT",
                    "HL_HH_CONFIRMATION","FAILED_BREAKDOWN_RECLAIM")
    setup_family_counts={family:int(setup_counts.get(family,0)) for family in setup_families}
    activation_names=("NEAR_DAILY_SUPPORT","INSIDE_DAILY_SUPPORT","NEAR_DAILY_RESISTANCE_BREAKOUT_WATCH",
        "RETESTING_BROKEN_RESISTANCE","BROKEN_SUPPORT_RECLAIM_WATCH","MID_RANGE")
    sod_activation_counts={name:int(sod_activations.get(name,0)) for name in activation_names}
    examples=[]; per_family=Counter(); breakout_retest_subtypes=Counter()
    for item in sorted(ready,key=lambda r:(r["setup_family"],r["confirmation_timestamp"],r["security_id"])):
        family=item["setup_family"]
        provenance=(item.get("result",{}).get("breakout_retest_provenance") or {})
        retest=(provenance.get("retest") or {})
        subtype=retest.get("retest_subtype") if family=="BREAKOUT_RETEST" else None
        if family=="BREAKOUT_RETEST": breakout_retest_subtypes[str(subtype or "MISSING")]+=1
        example_group=(family,subtype) if family=="BREAKOUT_RETEST" else (family,None)
        if per_family[example_group]>=5: continue
        plan=item["plan"]; result=item["result"]
        features=result.get("confirmation_features",{})
        examples.append({"setup_family":family,"retest_subtype":subtype,"setup_instance_id":item["setup_instance_id"],
            "signal_date":item["setup_signal_date"],"confirmation_timestamp":item["confirmation_timestamp"],
            "entry_timestamp":item["executable_timestamp"],"security_id":item["security_id"],"symbol":item["symbol"],
            "nifty_regime":item.get("nifty_regime"),"weekly_selection":item.get("weekly_selection_components"),
            "weekly_structure":item.get("weekly_structure"),"daily_location":item.get("daily_location_state"),
            "recent_window_sessions_used":next((r.get("recent_window_sessions_used") for r in location_rows
                if r.get("sod_date")==item["setup_signal_date"] and r.get("security_id")==item["security_id"]),None),
            "support_zone":item.get("support"),"resistance_zones":item.get("resistance"),
            "daily_zones":item.get("daily_zones"),"h1_confirmation":result.get("confirmation_type"),
            "h1_confirmation_features":features,"volume_confirmation_type":plan.get("volume_confirmation_type"),
            "volume_confirmation_features":plan.get("volume_confirmation_features"),
            "volume_confirmation_passed":plan.get("volume_confirmation_passed"),"entry":plan.get("entry"),
            "daily_structural_stop":plan.get("stop"),"intermediate_friction":plan.get("intermediate_daily_friction"),
            "primary_target":plan.get("target"),"target_timeframe":plan.get("target_timeframe"),
            "planned_rr":plan.get("planned_reward_r"),
            "breakout_retest_provenance":result.get("breakout_retest_provenance"),
            "setup_lifecycle":item.get("setup_lifecycle")})
        per_family[example_group]+=1
    _csv(output_dir/"v2_manual_setup_examples.csv",examples)
    _csv(output_dir/"v2_daily_zones.csv",zone_dump); _csv(output_dir/"v2_entry_ready_events.csv",ready); _csv(output_dir/"v2_rejected_candidates.csv",rejects)
    _csv(output_dir/"v2_signal_trace.csv",trace)
    # Stable deterministic sample of candidate locations for plotting.
    sample=sorted(location_rows,key=lambda r:__import__("hashlib").sha256(f"{r['security_id']}|{r['sod_date']}".encode()).hexdigest())[:max(30,min(100,len(location_rows)))]
    _csv(output_dir/"v2_sod_watchlist_diagnostics.csv",location_rows)
    _csv(output_dir/"daily_zone_visual_audit.csv",sample)
    bpcl_window=[]
    for r in location_rows:
        if r["symbol"]=="BPCL" and str(r["sod_date"])[:7]=="2023-09":
            bpcl_window.extend({"sod_date":r["sod_date"],"daily_candle":bar} for bar in r["daily_window_candles"])
    _csv(output_dir/"bpcl_recent_daily_windows.csv",bpcl_window)
    friction_types=Counter(str(f.get("type")) for r in ready for f in r.get("plan",{}).get("friction",[]))
    blocked_pairs={(r.get("date"),r.get("security_id"),t) for r in rejects if r.get("reason_code")=="WAITING_FOR_HOURLY_FRICTION_CLEARANCE" for t in r.get("friction_types",[])}
    friction_blocks=Counter(t for _,_,t in blocked_pairs)
    def unique_rejections(codes:set[str])->int:
        return len({(r.get("date"),r.get("security_id"),r.get("reason_code")) for r in rejects if r.get("reason_code") in codes})
    summary={"strategy_version":raw["version"],"period":{"start":start.isoformat(),"end":end.isoformat()},"warmup":{"daily":DAILY_WARMUP.isoformat(),"hourly":HOURLY_WARMUP.isoformat()},
      "universe":PIT_TRADABLE,"data_freeze_version":"NIFTY50_PIT_2023_2024_V1","2025_2026_loaded":False,
      "hourly_data_securities":len(hourly),"price_mismatch_hourly_excluded":"NSEEQ-0055","data_hashes":{"daily_prefix_sha256":dhash,"hourly_prefix_sha256":hhash},
      "counts":dict(sorted(counts.items())),"confirmation_archetypes":dict(sorted(archetypes.items())),"ready_events":len(ready),
      "setup_family_entry_ready_counts":setup_family_counts,"sod_activation_counts":sod_activation_counts,
      "weekly_selection":{"status_counts":dict(sorted(weekly_statuses.items())),
        "eligible_or_breakout_exception_observations":counts.get("WEEKLY_SELECTION_PASSED",0),
        "first_causally_eligible_date":first_causal_date,"nifty_regime_method":"existing causal weekly_regime classifier; context only"},
      "first_entry_ready_date":first_ready_date,"manual_setup_example_counts":{
        (f"{family}:{subtype}" if subtype else family):count
        for (family,subtype),count in sorted(per_family.items())},
      "breakout_retest_entry_ready_by_subtype":dict(sorted(breakout_retest_subtypes.items())),
      "breakout_retest_original_breakout_volume_mandatory":True,
      "setup_lifecycle":{"consumed_zone_theses":len(consumed_theses),
        "subsequent_sod_activations_suppressed":consumed_suppressed,
        "reset_conditions":["zone removed from current structure map","zone role or side changed",
            "daily close below a consumed long support zone","new non-overlapping structural zone event"]},
      "friction_observations_by_type":dict(sorted(friction_types.items())),
      "hourly_friction_diagnostics":{"blocked_by_hourly_resistance":friction_blocks.get("HOURLY_RESISTANCE",0),
        "blocked_by_ema50":friction_blocks.get("EMA50",0),"blocked_by_ema200":friction_blocks.get("EMA200",0),
        "blocked_by_sr_ema_confluence":friction_blocks.get("HOURLY_SR_EMA_CONFLUENCE",0),
        "later_cleared_friction_candidates":sum(bool(r.get("plan",{}).get("friction_clearance_required")) for r in ready),
        "clearance_consumed_rr":sum(r.get("reason_code")=="CLEARANCE_CONSUMED_STRUCTURAL_ROOM" for r in rejects),
        "missing_volume_confirmation":unique_rejections({"VOLUME_CONFIRMATION_MISSING","BREAKOUT_CLEARANCE_VOLUME_INSUFFICIENT"}),
        "hostile_bearish_volume":unique_rejections({"HOSTILE_BEARISH_VOLUME_AT_DAILY_SUPPORT"})},
      "candidate_level_rejection_counts":{"MID_RANGE":counts.get("MID_RANGE",0),
        "NEAR_DAILY_RESISTANCE_POOR_RR":counts.get("NEAR_DAILY_RESISTANCE_POOR_RR",0),
        "DAILY_SUPPORT_BROKEN":unique_rejections({"DAILY_SUPPORT_BROKEN"})+counts.get("DAILY_SUPPORT_BROKEN",0),
        "NO_MEANINGFUL_DAILY_TARGET":unique_rejections({"NO_MEANINGFUL_STRUCTURAL_TARGET"}),
        "RR_INSUFFICIENT":unique_rejections({"RR_INSUFFICIENT"}),
        "STOP_TOO_TIGHT_FOR_DAILY_STRUCTURE":unique_rejections({"STOP_TOO_TIGHT_FOR_DAILY_STRUCTURE"}),
        "HOURLY_CONFIRMATION_MISSING":counts.get("HOURLY_CONFIRMATION_MISSING",0),
        "HOURLY_DATA_UNAVAILABLE":counts.get("HOURLY_DATA_UNAVAILABLE",0),
        "SETUP_EXPIRED":counts.get("SETUP_EXPIRED",0)},
      "configuration":raw,"daily_structure_window":{"structural_memory_sessions":cfg.structural_memory_sessions,
        "local_min_sessions":cfg.recent_daily_structure_min_sessions,"local_max_sessions":cfg.recent_daily_structure_max_sessions,
        "selection":"persistent structure is rebuilt causally from up to the configured completed-session memory; local 45-60 session context expands only when it lacks an enclosing support/resistance map"},
      "daily_zone_method":{"swing_left":cfg.daily_zone.swing_left,"swing_right":cfg.daily_zone.swing_right,
        "pivot_cluster_distance_atr":cfg.daily_zone.cluster_distance_atr,"same_side_overlapping_zones":"merged by the shared deterministic overlap merge rule",
        "nested_structure":"validated repeated parent lower-edge interactions are retained as separately identified inner breakout/retest boundaries; parent macro zone remains intact",
        "target_hierarchy":"MAJOR parent zones and overlapping daily/weekly parent zones are dominant target candidates; when neither exists, highest-evidence parent zone is used. Other meaningful parent zones remain secondary, nested inner boundaries remain intermediate friction; H1/EMA friction is separate",
        "meaningful":"validated pivot zone with a meaningful reaction, sustained hover, failed cross, or accepted role flip; age within structural memory does not demote it",
        "tradeable":"requires validation and at least two separate touch episodes; first validation does not itself trigger an entry",
        "major":"at least 4 separate touch episodes and mean reaction >=1.0 ATR, without recency demotion",
        "minor":"retained for context; not an executable daily location"},
      "rules":{"daily_structure_primary":True,"completed_hourly_bars_only":True,"next_bar_open_entry":True,
        "weekly_stock_selection_precedes_daily_map":True,"nifty_regime_is_context_not_hard_gate":True,
        "all_entry_families_require_setup_specific_volume_confirmation":True,
        "same_confirmation_bar_entry":False,"outcome_trained_rules":False,"threshold_sweeps":False}}
    (output_dir/"v2_signal_summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True,ensure_ascii=True,default=str)+"\n",encoding="utf-8")
    print(json.dumps({"counts":summary["counts"],"entry_ready_by_setup_family":setup_family_counts,
        "sod_activation_counts":sod_activation_counts,"confirmation_archetypes":summary["confirmation_archetypes"],
        "first_causally_eligible_date":first_causal_date,"ready_events":len(ready),"output":str(output_dir)},indent=2))
    return {"summary":summary,"ready":ready,"trace":trace,"rejections":rejects,"output_directory":output_dir}


def main()->None:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--start",type=date.fromisoformat,default=START);p.add_argument("--end",type=date.fromisoformat,default=END)
    p.add_argument("--config",type=Path,default=CONFIG);p.add_argument("--output-directory",type=Path,default=OUTPUT);p.add_argument("--data-root",type=Path,default=DATA_ROOT)
    p.add_argument("--daily-manifest",type=Path,default=MANIFEST_PATH);p.add_argument("--hourly-manifest",type=Path);p.add_argument("--benchmark",type=Path,default=BENCHMARK);p.add_argument("--max-days",type=int)
    a=p.parse_args()
    try: replay(start=a.start,end=a.end,config_path=a.config,output_dir=a.output_directory,data_root=a.data_root,daily_manifest=a.daily_manifest,hourly_manifest=a.hourly_manifest,benchmark_path=a.benchmark,max_days=a.max_days)
    except (ValueError,OSError,BacktestDataError) as e: p.error(str(e))

if __name__=="__main__": main()
