"""Run the frozen V3 causal regime/geometry signal replay on 2023-2024 only.

The frozen V2 ready-event file is the horizontal candidate source. V3 evaluates
those events with a date-causal per-security state machine and independently
scans H1 bars for inclined TRENDLINE_PULLBACK candidates. It never invokes V2
signal generation or reads post-2024 market data.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from src.backtest.data import load_daily_dataset_through, load_hourly_dataset_through
from src.backtest.top_down_swing_portfolio import _cost_config
from src.configuration import PROJECT_ROOT
from src.data.hourly import HOURLY_MANIFEST
from src.data.market_data import DATA_ROOT, MANIFEST_PATH
from src.data.universe.provider import PIT_TRADABLE, UniverseProvider
from src.strategy.structural_regime_v3 import classify_regime_series
from src.strategy.top_down_swing_v3 import (V3Config, TrendlineSetupLifecycle, annotate_v2_signal,
    trendline_pullback_candidate)

START=date(2023,1,1); END=date(2024,12,31)
V2_ROOT=PROJECT_ROOT/"data/processed/backtests/top_down_swing_v2_daily_structure"
OUTPUT=PROJECT_ROOT/"data/processed/backtests/top_down_swing_v3"
CONFIG=PROJECT_ROOT/"config/top_down_swing_v3.yaml"
V2_SIGNALS=V2_ROOT/"v2_entry_ready_events.csv"


def _read(path: Path) -> list[dict[str,str]]:
    with path.open("r",encoding="utf-8-sig",newline="") as f: return list(csv.DictReader(f))


def _write(path: Path, rows: Sequence[Mapping[str,Any]]) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for r in rows for k in r)) or ["empty"]
    with path.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore"); w.writeheader()
        for row in rows:
            w.writerow({k:json.dumps(v,sort_keys=True,ensure_ascii=True,default=str) if isinstance(v,(dict,list,tuple)) else v for k,v in row.items()})


def _dt(value: Any) -> datetime:
    parsed=value if isinstance(value,datetime) else datetime.fromisoformat(str(value))
    return parsed


def _date(value: Any) -> date:
    return value if isinstance(value,date) else date.fromisoformat(str(value))


def _atr_proxy(rows: Sequence[Mapping[str,Any]],period:int=14)->float:
    tail=rows[-period:]
    return sum(float(r["high"])-float(r["low"]) for r in tail)/len(tail) if tail else 0.0


def _parse_signal(row: Mapping[str,str])->dict[str,Any]:
    result=json.loads(row.get("result") or "{}")
    plan=json.loads(row.get("plan") or "{}")
    return {**dict(row),"result":result,"plan":plan,
        "confirmation_timestamp":_dt(row.get("confirmation_timestamp") or result.get("confirmation_timestamp")),
        "executable_timestamp":_dt(row.get("executable_timestamp") or result.get("entry_timestamp"))}


def _trendline_signals(*,daily:Mapping[str,Sequence[Mapping[str,Any]]],hourly:Mapping[str,Sequence[Mapping[str,Any]]],
                       regimes:Mapping[str,Mapping[str,Mapping[str,Any]]],zone_rows:Sequence[Mapping[str,str]],
                    symbols:Mapping[Any,str],cfg:V3Config)->tuple[list[dict[str,Any]],list[dict[str,Any]]]:
    """Find causal daily-uptrend/H1-response events; no manual date is forced."""
    zones_by_sid:dict[str,list[dict[str,str]]]=defaultdict(list)
    for z in zone_rows: zones_by_sid[z["security_id"]].append(dict(z))
    ready=[]; diagnostics=[]; lifecycles:dict[str,TrendlineSetupLifecycle]={}
    for sid,daily_rows in daily.items():
        day_index={str(r["date"])[:10]:i for i,r in enumerate(daily_rows)}
        hrows=list(hourly.get(sid,()))
        by_day:dict[str,list[tuple[int,Mapping[str,Any]]]]=defaultdict(list)
        for i,h in enumerate(hrows): by_day[str(h["date"])[:10]].append((i,h))
        for day,items in by_day.items():
            d=date.fromisoformat(day)
            if not START<=d<=END: continue
            idx=day_index.get(day)
            if idx is None or idx<1: continue
            state=regimes.get(sid,{}).get(day)
            if not state or state["regime"]!="TREND_UP": continue
            line=state.get("trendlines",{}).get("rising_support")
            if not line or line.get("status") not in {"ESTABLISHED","STRONG"} or not line.get("channel_opposite_level"): continue
            prefix=daily_rows[:idx]
            atr=_atr_proxy(prefix)
            lifecycle=lifecycles.get(sid)
            if lifecycle is None:
                lifecycle=TrendlineSetupLifecycle(str(line["trendline_id"])); lifecycles[sid]=lifecycle
            lifecycle.advance_day(trendline_id=str(line["trendline_id"]),close=float(prefix[-1]["close"]),
                support_level=float(line["projected_level"]),daily_atr=atr,date_text=day)
            if lifecycle.status=="CONSUMED":
                diagnostics.append({"date":day,"security_id":sid,"symbol":symbols.get((sid,day),sid),
                    "setup_family":"TRENDLINE_PULLBACK","regime":state["regime"],"trendline_id":line["trendline_id"],
                    "status":"THESIS_CONSUMED","reason_code":"WAIT_FOR_STRUCTURAL_RESET",
                    "consumed_on":lifecycle.consumed_on,"episode":lifecycle.episode})
                continue
            # Decision-day projected channel estimate moves one daily session
            # along the already-confirmed parallel geometry.
            channel=float(line["channel_opposite_level"])+float(line["slope_per_session"])
            targets=[]
            if int(line.get("contact_count",0))>=3 and int(line.get("channel_contact_count",0))>=2:
                targets.append({"price":channel,"timeframe":"CHANNEL","classification":"DOMINANT","target_id":line["trendline_id"]})
            # Use the latest prior saved persistent-map evidence per zone ID.
            # Major zones dominate; absent those, only the strongest meaningful
            # zone is promoted. Other nearby levels remain friction, not target.
            latest={}
            for z in zones_by_sid.get(sid,()):
                if str(z.get("date",""))[:10]>=day:continue
                zid=str(z.get("zone_id",""))
                if not zid or z.get("timeframe") not in {"DAILY","WEEKLY"}:continue
                if zid not in latest or str(z["date"])>str(latest[zid]["date"]):latest[zid]=z
            valid=[]
            for z in latest.values():
                if z.get("side")!="resistance" or z.get("validated") not in {"True","true","1"} or z.get("tradeable") not in {"True","true","1"}:continue
                if z.get("structure_level_kind") not in {"","PARENT_ZONE"}:continue
                if z.get("role_state") not in {"RESISTANCE_CANDIDATE","RESISTANCE_CONFIRMED","ACTIVE",""}:continue
                if z.get("strength") not in {"MAJOR","MEANINGFUL"}:continue
                valid.append(z)
            dominant=[z for z in valid if z.get("strength")=="MAJOR"]
            if not dominant and valid:
                dominant=[max(valid,key=lambda z:(float(z.get("dominance_score") or 0),
                    z.get("timeframe")=="WEEKLY",str(z.get("zone_id"))))]
            dominant_ids={str(z.get("zone_id")) for z in dominant}
            friction_candidates=[{"zone_id":z.get("zone_id"),"lower_bound":float(z["lower_bound"]),
                "upper_bound":float(z["upper_bound"]),"dominance_score":float(z.get("dominance_score") or 0)}
                for z in valid if str(z.get("zone_id")) not in dominant_ids]
            for z in dominant:
                price=float(z["lower_bound"])-.10*atr
                targets.append({"price":price,"timeframe":z.get("timeframe"),"classification":"DOMINANT","target_id":z.get("zone_id"),
                    "dominance_score":float(z.get("dominance_score") or 0),"strength":z.get("strength")})
            emitted=False
            for hpos,(global_i,hbar) in enumerate(items):
                if global_i+1>=len(hrows): continue
                # A proximity check avoids repeatedly evaluating the same H1
                # prefix when price is remote from the daily rising support.
                projected=float(line["projected_level"])+float(line["slope_per_session"])
                if float(hbar["low"])>projected+float(line["tolerance"])*1.5: continue
                prefix_h=hrows[:global_i+1]
                nextbar=hrows[global_i+1]
                candidate=trendline_pullback_candidate(daily_prefix=prefix,hourly_prefix=prefix_h,
                    entry_bar=nextbar,target_candidates=targets,config=cfg,decision_date=day,regime_snapshot=state,
                    friction_candidates=friction_candidates)
                if candidate["status"]=="ENTRY_READY":
                    episode_id=lifecycle.consume(day)
                    result={"confirmation_type":"TRENDLINE_PULLBACK","confirmation_features":{"volume":{"passed":True,"features":candidate["volume_features"],"same_time_relative_volume":candidate["h1_relative_volume"]}},
                            "plan":{"entry":candidate["entry"],"stop":candidate["stop"],"target":candidate["target"],
                                   "target_timeframe":candidate["target_type"],"planned_reward_r":candidate["planned_rr"],
                                   "target_reference":candidate["target_type"],"daily_support_zone_id":None}}
                    stamp=_dt(candidate["entry_timestamp"])
                    symbol=symbols.get((sid,day),sid)
                    ready.append({"date":day,"setup_signal_date":day,"security_id":sid,"symbol":symbol,
                        "state":"ENTRY_READY","setup_type":"TRENDLINE_PULLBACK","setup_family":"TRENDLINE_PULLBACK",
                        "setup_instance_id":f"{sid}-{episode_id}-{candidate['confirmation_timestamp']}",
                        "setup_event_id":episode_id,"result":result,"plan":result["plan"],
                        "confirmation_timestamp":_dt(candidate["confirmation_timestamp"]),"executable_timestamp":stamp,
                        "regime":state["regime"],"regime_evidence":state["diagnostic_evidence"],
                        "previous_regime":state.get("previous_regime"),"regime_start_date":state.get("regime_start_date"),
                        "primary_geometry":"INCLINED","secondary_geometry":"HORIZONTAL",
                        "trendline":candidate["trendline"],"trendline_id":candidate["trendline"]["trendline_id"],"channel_id":candidate["channel_id"],
                        "horizontal_zone_id":None,"ema50_value":state["ema"]["ema50"]["value"],
                        "ema50_slope":state["ema"]["ema50"]["normalized_slope"],"ema200_value":state["ema"]["ema200"]["value"],
                        "ema200_slope":state["ema"]["ema200"]["normalized_slope"],
                        "ema50_state":candidate["ema50_state"],"ema200_state":candidate["ema200_state"],
                        "confluence_state":candidate["confluence_state"],"h1_confirmation_type":candidate["h1_confirmation_type"],
                        "volume_state":candidate["volume_state"],"reason_code":candidate["reason_code"]})
                    emitted=True; break
                diagnostics.append({"date":day,"security_id":sid,"symbol":symbols.get((sid,day),sid),
                    "setup_family":"TRENDLINE_PULLBACK","regime":state["regime"],"trendline_id":line["trendline_id"],
                    "status":candidate["status"],"reason_code":candidate["reason_code"],
                    "detail":candidate.get("missing_evidence"),"primary_geometry":state["primary_geometry"],
                    "ema50_state":state.get("ema",{}).get("ema50",{}).get("state"),"ema200_state":state.get("ema",{}).get("ema200",{}).get("state")})
            if not emitted and not items:
                diagnostics.append({"date":day,"security_id":sid,"symbol":symbols.get((sid,day),sid),"setup_family":"TRENDLINE_PULLBACK",
                    "regime":state["regime"],"status":"NO_H1_BARS","reason_code":"NO_H1_TRIGGER_DATA"})
    return ready,diagnostics


def replay(*,config_path:Path=CONFIG,output_dir:Path=OUTPUT,signal_path:Path=V2_SIGNALS,
           data_root:Path=DATA_ROOT,daily_manifest:Path=MANIFEST_PATH,
           hourly_manifest:Path=HOURLY_MANIFEST)->dict[str,Any]:
    raw=yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if raw.get("version")!="top_down_swing_v3": raise ValueError("V3 configuration version mismatch")
    period=raw.get("research_period",{})
    if _date(period.get("start"))!=START or _date(period.get("end"))!=END:
        raise ValueError("V3 replay is fixed to the frozen 2023-2024 research period")
    daily,_=load_daily_dataset_through(universe_id=PIT_TRADABLE,start_date=date(2022,7,1),end_date=END,
                                       data_root=data_root,manifest_path=daily_manifest)
    hourly,_=load_hourly_dataset_through(universe_id=PIT_TRADABLE,start_date=date(2022,10,1),end_date=END,
        data_root=data_root,manifest_path=hourly_manifest,daily_manifest_path=daily_manifest)
    provider=UniverseProvider(); symbols_by_day={}
    for day in sorted({str(r["date"])[:10] for group in daily.values() for r in group if START.isoformat()<=str(r["date"])[:10]<=END.isoformat()}):
        symbols_by_day[day]={m.security_id:m.symbol for m in provider.members_on(date.fromisoformat(day),PIT_TRADABLE)}
    symbols={}
    # Build a one-pass state history per security. For each decision date, the
    # daily feature prefix ends on the previous completed session.
    regime_rows=[]; regime_by_sid:dict[str,dict[str,dict[str,Any]]]={}
    for sid,rows in daily.items():
        series=classify_regime_series(rows,start_index=1)
        lookup={x["decision_date"]:x for x in series}; regime_by_sid[sid]=lookup
        for day,state in lookup.items():
            if START.isoformat()<=day<=END.isoformat():
                regime_rows.append({"date":day,"security_id":sid,"symbol":symbols_by_day.get(day,{}).get(sid,sid),**state})
    zone_path=PROJECT_ROOT/"data/processed/backtests/top_down_swing_v2_daily_structure/v2_daily_zones.csv"
    zone_rows=_read(zone_path) if zone_path.exists() else []
    horizontal=[]; rejected=[]
    for raw_sig in _read(signal_path):
        if raw_sig.get("state")!="ENTRY_READY": continue
        day=str(raw_sig.get("setup_signal_date") or raw_sig.get("date"))[:10]
        d=date.fromisoformat(day)
        if not START<=d<=END: continue
        sid=raw_sig["security_id"]; rows=daily.get(sid,()); idx=next((i for i,r in enumerate(rows) if str(r["date"])[:10]>=day),len(rows))
        prefix=rows[:idx]
        sig=_parse_signal(raw_sig)
        annotated=annotate_v2_signal(sig,prefix,regime_snapshot=regime_by_sid.get(sid,{}).get(day))
        if annotated["v3_status"]=="ENTRY_READY": horizontal.append(annotated)
        else: rejected.append(annotated)
    v3cfg=V3Config(minimum_structural_rr=float(raw.get("setups",{}).get("minimum_structural_rr",1.5)),
        support_near_atr=float(raw.get("setups",{}).get("support_near_atr",.75)),
        channel_lower_region_max=float(raw.get("setups",{}).get("channel_lower_region_max",.40)),
        stop_buffer_atr=float(raw.get("setups",{}).get("stop_buffer_atr",.25)),
        friction_distance_atr=float(raw.get("setups",{}).get("friction_distance_atr",.5)),
        clearance_buffer_atr=float(raw.get("setups",{}).get("clearance_buffer_atr",.1)),
        volume_lookback=int(raw.get("setups",{}).get("volume_lookback",20)))
    # Symbol labels follow point-in-time membership, including corporate-name
    # transitions, while identity and candle storage remain stable by security ID.
    symbols={(sid,day):symbol for day,mapping in symbols_by_day.items() for sid,symbol in mapping.items()}
    trend_ready,trend_diagnostics=_trendline_signals(daily=daily,hourly=hourly,regimes=regime_by_sid,
        zone_rows=zone_rows,symbols=symbols,cfg=v3cfg)
    # V3-family lifecycle keys are per confirming H1 event; no daily reset can
    # replay a consumed event. Multiple entries in one security remain subject
    # to the portfolio open-position and setup-id constraints.
    ready=horizontal+trend_ready
    ids=[str(x["setup_instance_id"]) for x in ready]
    if len(ids)!=len(set(ids)): raise ValueError("V3 replay generated duplicate setup instance IDs")
    for sig in ready:
        if sig.get("setup_family")=="TRENDLINE_PULLBACK":
            if float(sig["plan"]["stop"])>=float(sig["plan"]["entry"]): raise ValueError("V3 trendline stop must be below entry")
            if float(sig["plan"]["target"])<=float(sig["plan"]["entry"]): raise ValueError("V3 target must exceed entry")
            if float(sig["plan"]["planned_reward_r"])<1.5: raise ValueError("V3 RR below configured minimum")
    output_dir.mkdir(parents=True,exist_ok=True)
    _write(output_dir/"v3_regime_daily.csv",regime_rows)
    _write(output_dir/"v3_entry_ready_events.csv",ready)
    _write(output_dir/"v3_regime_rejections.csv",rejected)
    _write(output_dir/"v3_trendline_diagnostics.csv",trend_diagnostics)
    by_regime=defaultdict(int)
    by_geometry=defaultdict(int)
    by_family=defaultdict(int)
    for item in ready:
        by_regime[item.get("regime","UNKNOWN")]+=1; by_geometry[item.get("primary_geometry","UNKNOWN")]+=1
        by_family[item.get("setup_family","UNKNOWN")]+=1
    summary={"strategy_version":"top_down_swing_v3","period":{"start":START.isoformat(),"end":END.isoformat()},
        "2025_2026_loaded":False,"daily_and_hourly_data_cutoff":END.isoformat(),"daily_warmup_start":"2022-07-01",
        "hourly_warmup_start":"2022-10-01","source_horizontal_candidates":len(horizontal)+len(rejected),
        "v3_ready_signals":len(ready),"v3_rejected_horizontal_candidates":len(rejected),
        "trendline_pullback_signals":len(trend_ready),"trendline_diagnostic_rows":len(trend_diagnostics),
        "signals_by_regime":dict(by_regime),"signals_by_geometry":dict(by_geometry),"signals_by_family":dict(by_family),
        "source_v2_signal_path":str(signal_path),"output":str(output_dir),
        "limitations":["Horizontal V3 candidates are read from the saved frozen V2 signal artifact; V2 signal-generation replay was not rerun.",
            "Trendline pullbacks use established causal rising channels and dominant targets from prior saved V2 zone snapshots; manual examples are diagnostic, not forced." ]}
    (output_dir/"v3_replay_summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True,default=str)+"\n",encoding="utf-8")
    return {"summary":summary,"signals":ready,"regimes":regime_rows,"rejections":rejected,"trendline_diagnostics":trend_diagnostics}


def main()->None:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--config",type=Path,default=CONFIG); p.add_argument("--output-directory",type=Path,default=OUTPUT)
    p.add_argument("--signals",type=Path,default=V2_SIGNALS); p.add_argument("--data-root",type=Path,default=DATA_ROOT)
    p.add_argument("--daily-manifest",type=Path,default=MANIFEST_PATH); p.add_argument("--hourly-manifest",type=Path,default=HOURLY_MANIFEST)
    args=p.parse_args(); result=replay(config_path=args.config,output_dir=args.output_directory,signal_path=args.signals,
        data_root=args.data_root,daily_manifest=args.daily_manifest,hourly_manifest=args.hourly_manifest)
    print(json.dumps(result["summary"],indent=2,default=str))


if __name__=="__main__": main()
