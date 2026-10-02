"""V3 portfolio replay with and without one-way structural profit protection."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import asdict
from datetime import date,datetime
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

import yaml

from src.backtest.models import CostModel
from src.backtest.top_down_swing_portfolio import _cost_config, _metrics
from src.backtest.top_down_swing_v3_replay import END, OUTPUT as REPLAY_OUTPUT, START, _read, _write
from src.configuration import PROJECT_ROOT
from src.strategy.trade_management_v3 import initial_protection, stop_execution, update_protection

OUTPUT=PROJECT_ROOT/"data/processed/backtests/top_down_swing_v3/portfolio"
CONFIG=PROJECT_ROOT/"config/top_down_swing_v3.yaml"


def _stamp(x:Any)->datetime:
    return x if isinstance(x,datetime) else datetime.fromisoformat(str(x))


def _signals(path:Path)->list[dict[str,Any]]:
    out=[]
    for r in _read(path):
        if r.get("state")!="ENTRY_READY": continue
        plan=json.loads(r.get("plan") or "{}")
        if not plan: continue
        out.append({**r,"plan":plan,"result":json.loads(r.get("result") or "{}"),
            "confirmation_timestamp":_stamp(r["confirmation_timestamp"]),
            "executable_timestamp":_stamp(r["executable_timestamp"])})
    return out


def _entry_only(signals:Sequence[Mapping[str,Any]],bars:Mapping[str,Sequence[Mapping[str,Any]]],costs:CostModel,
                *,initial:float,risk:float,max_positions:int,max_notional_fraction:float,
                total_risk_fraction:float,ranking_key=None)->dict[str,Any]:
    """Run the same V3 portfolio controls with original structural brackets."""
    return _protected_run(signals,bars,costs,initial=initial,risk_fraction=risk,max_positions=max_positions,
        max_notional_fraction=max_notional_fraction,total_risk_fraction=total_risk_fraction,
        enable_protection=False,ranking_key=ranking_key)


def _protected_run(signals:Sequence[Mapping[str,Any]],bars:Mapping[str,Sequence[Mapping[str,Any]]],costs:CostModel,
                   *,initial:float,risk_fraction:float,max_positions:int,max_notional_fraction:float,
                   total_risk_fraction:float,enable_protection:bool,ranking_key=None)->dict[str,Any]:
    bars_by_sid={sid:sorted((dict(x) for x in rows),key=lambda r:_stamp(r["date"])) for sid,rows in bars.items()}
    at_time:dict[datetime,list[tuple[str,int,dict[str,Any]]]]=defaultdict(list)
    for sid,rows in bars_by_sid.items():
        for i,row in enumerate(rows): at_time[_stamp(row["date"])].append((sid,i,row))
    sig_at:dict[datetime,list[dict[str,Any]]]=defaultdict(list)
    for s in signals: sig_at[s["executable_timestamp"]].append(dict(s))
    times=sorted(set(at_time)|set(sig_at)); cash=float(initial); positions={}; marks={}; ledger=[]; eq=[]; events=[]
    ids=set(); rejected=defaultdict(int); dp_paid=set(); peak=initial

    def close(sid:str,stamp:datetime,raw:float,reason:str,ambiguous:bool=False)->None:
        nonlocal cash
        p=positions.pop(sid); fill=costs.fill_price(raw,side="sell"); qty=p["qty"]
        gross=(fill-p["entry"])*qty; dpkey=(sid,stamp.date()); include=dpkey not in dp_paid
        exit_cost=costs.calculate(fill*qty,side="sell",include_dp_charge=include)
        if include: dp_paid.add(dpkey)
        cash+=fill*qty-exit_cost.total; total=p["entry_cost"]+exit_cost.total; net=gross-total
        realized_r=(fill-p["entry"])/p["initial_risk_share"]
        ledger.append({**p["base"],"status":"CLOSED","exit_timestamp":stamp.isoformat(),"exit_price":fill,
            "raw_exit_price":raw,"exit_reason":reason,"ambiguous_intrabar_stop_first":ambiguous,
            "gross_pnl":gross,"net_pnl":net,"entry_costs":p["entry_cost"],"exit_costs":exit_cost.total,
            "transaction_costs":total,"entry_cost_breakdown":p["entry_breakdown"],"exit_cost_breakdown":exit_cost.as_dict(),
            "gross_R":gross/p["initial_risk_cash"],"net_R":net/p["initial_risk_cash"],"realized_R":realized_r,
            "mfe_R":max(p["protection"].mfe_r,(p["mfe_price"]-p["entry"])/p["initial_risk_share"]),"mae_R":p["mae_R"],"holding_hourly_bars":p["held_bars"],
            "holding_hours":(stamp-_stamp(p["base"]["entry_timestamp"])).total_seconds()/3600,
            "holding_sessions":len(p["sessions"]),"protection":asdict(p["protection"]),"final_current_stop":p["stop"],
            "stop_move_history":p["stop_history"],
            "intended_protected_R":p["protection"].r_floor,"realized_R_after_gap":realized_r if reason.startswith("GAP") else None})
        events.append({"timestamp":stamp.isoformat(),"event":"EXIT","security_id":sid,"reason":reason,"cash_after":cash})

    for stamp in times:
        current=sorted(at_time.get(stamp,[]),key=lambda x:x[0]); by_sid={sid:(i,b) for sid,i,b in current}
        held_at_open=set(positions)
        # Apply only stop updates calculated from previously completed H1 bars.
        for sid in sorted(held_at_open):
            if sid not in by_sid: continue
            idx,bar=by_sid[sid]; p=positions[sid]
            op,hi,lo=float(bar["open"]),float(bar["high"]),float(bar["low"])
            p["held_bars"]+=1; p["sessions"].add(stamp.date())
            exited,fill,kind=stop_execution(op,lo,p["stop"])
            target_gap=op>=p["target"]
            if op<=p["stop"]:
                p["mae_R"]=min(p["mae_R"],(op-p["entry"])/p["initial_risk_share"])
                close(sid,stamp,op,"GAP_PROTECTIVE_STOP" if p["stop"]>p["original_stop"] else "GAP_STOP")
            elif target_gap:
                p["mfe_price"]=max(p["mfe_price"],op)
                close(sid,stamp,op,"GAP_TARGET")
            else:
                p["mae_R"]=min(p["mae_R"],(lo-p["entry"])/p["initial_risk_share"])
                p["mfe_price"]=max(p["mfe_price"],hi)
                if exited:
                    trailing=(p["protection"].structural_reference is not None and p["protection"].r_floor_price is not None and
                              p["stop"]>p["protection"].r_floor_price+1e-9)
                    close(sid,stamp,float(fill),"TRAILING_STRUCTURAL_STOP" if trailing else "PROTECTIVE_STOP" if p["stop"]>p["original_stop"] else "ORIGINAL_STOP",
                          ambiguous=hi>=p["target"])
                    continue
                if hi>=p["target"]:
                    close(sid,stamp,p["target"],"TARGET")
                    continue
                round_trip_cost=(p["entry_cost"]+costs.calculate(p["entry"]*p["qty"],side="sell",include_dp_charge=True).total)/max(p["qty"],1)
                p["protection"]=update_protection(p["protection"],entry=p["entry"],high_since_entry=p["mfe_price"],
                    timestamp=_stamp(bars_by_sid[sid][idx+1]["date"]).isoformat() if idx+1<len(bars_by_sid[sid]) else stamp.isoformat(),
                    confirmed_h1_rows=bars_by_sid[sid][p["entry_bar_index"]:idx+1],
                    h1_atr=sum(float(x["high"])-float(x["low"]) for x in bars_by_sid[sid][max(0,idx-13):idx+1])/min(14,idx+1),
                    estimated_round_trip_cost_per_share=round_trip_cost if enable_protection else 0.0) if enable_protection else p["protection"]
                if p["protection"].current_stop>p["stop"]+1e-12:
                    p["stop_history"].append({"timestamp":p["protection"].stop_move_timestamp,"previous_stop":p["stop"],
                        "new_stop":p["protection"].current_stop,"mfe_r":p["protection"].mfe_r,
                        "milestone_r":p["protection"].milestone_r,"reason":p["protection"].stop_move_reason,
                        "structural_reference":p["protection"].structural_reference,"r_floor":p["protection"].r_floor,
                        "r_floor_price":p["protection"].r_floor_price})
                p["stop"]=p["protection"].current_stop
        group=sorted(sig_at.get(stamp,[]),key=ranking_key or
            (lambda s:(-float(s["plan"].get("planned_reward_r") or 0),str(s["security_id"]),str(s["setup_instance_id"]))))
        for s in group:
            sid=str(s["security_id"]); uid=str(s["setup_instance_id"]); plan=s["plan"]
            if uid in ids: rejected["DUPLICATE_SETUP_BLOCKED"]+=1; continue
            ids.add(uid)
            if sid not in by_sid: rejected["MISSING_EXECUTABLE_HOURLY_BAR"]+=1; continue
            if sid in positions: rejected["ALREADY_OPEN_POSITION"]+=1; continue
            entry_bar_index=by_sid[sid][0]
            op=float(by_sid[sid][1]["open"]); stop=float(plan["stop"]); target=float(plan["target"])
            if stop>=op or target<=op: rejected["INVALID_ENTRY_GEOMETRY"]+=1; continue
            if len(positions)>=max_positions: rejected["MAX_OPEN_POSITIONS"]+=1; continue
            equity=cash+sum(p["qty"]*marks.get(key,p["entry"]) for key,p in positions.items())
            fill=costs.fill_price(op,side="buy"); share_risk=fill-stop
            rr=(target-fill)/share_risk if share_risk>0 else 0.0
            if rr<1.5: rejected["ACTUAL_RR_BELOW_MINIMUM"]+=1; continue
            risk_budget=equity*risk_fraction
            aggregate_room=max(0,equity*total_risk_fraction-sum(p["qty"]*max(0,p["entry"]-p["stop"]) for p in positions.values()))
            max_notional=equity*max_notional_fraction
            qty=min(math.floor(risk_budget/share_risk),math.floor(cash/fill),math.floor(max_notional/fill),
                    math.floor(aggregate_room/share_risk))
            while qty>0 and fill*qty+costs.calculate(fill*qty,side="buy").total>cash: qty-=1
            if qty<=0: rejected["CAPITAL_OR_AGGREGATE_RISK_LIMIT"]+=1; continue
            buy_cost=costs.calculate(fill*qty,side="buy")
            cash-=fill*qty+buy_cost.total
            initial_share_risk=fill-stop; initial_risk_cash=qty*initial_share_risk
            state=initial_protection(fill,stop)
            base={"strategy_version":"top_down_swing_v3","setup_instance_id":uid,"setup_event_id":s.get("setup_event_id"),
                  "security_id":sid,"symbol":s.get("symbol"),"setup_family":s.get("setup_family"),"signal_date":s.get("setup_signal_date",s.get("date")),
                  "regime":s.get("regime"),"primary_geometry":s.get("primary_geometry"),"secondary_geometry":s.get("secondary_geometry"),
                  "ema_confluence":s.get("confluence_state"),"entry_timestamp":stamp.isoformat(),"entry_raw_open":op,"entry_price":fill,
                  "regime_evidence":s.get("regime_evidence"),"regime_start_date":s.get("regime_start_date"),
                  "trendline":s.get("trendline"),"trendline_id":s.get("trendline_id"),"channel_id":s.get("channel_id"),
                  "horizontal_zone_id":s.get("horizontal_zone_id"),"ema50_state":s.get("ema50_state"),"ema200_state":s.get("ema200_state"),
                  "h1_confirmation_type":s.get("h1_confirmation_type"),"volume_state":s.get("volume_state"),
                  "original_stop":stop,"stop_price":stop,"target_price":target,"target_type":plan.get("target_timeframe"),
                  "planned_rr":plan.get("planned_reward_r"),"actual_rr":rr,"quantity":qty,"position_notional":fill*qty,
                  "initial_risk_cash":initial_risk_cash,"risk_budget":risk_budget,"risk_utilization":initial_risk_cash/risk_budget if risk_budget else 0,
                  "transaction_cost_scenario":costs.scenario,"entry_costs":buy_cost.total,"entry_cost_breakdown":buy_cost.as_dict(),
                  "mfe_R":0.0,"mae_R":0.0,"trade_management_enabled":enable_protection}
            positions[sid]={"base":base,"entry":fill,"qty":qty,"stop":stop,"original_stop":stop,"target":target,
                "entry_cost":buy_cost.total,"entry_breakdown":buy_cost.as_dict(),"initial_risk_cash":initial_risk_cash,
                "initial_risk_share":initial_share_risk,"protection":state,"held_bars":0,"sessions":{stamp.date()},
                "mfe_price":fill,"mae_R":0.0,"stop_history":[],"entry_bar_index":entry_bar_index}
            events.append({"timestamp":stamp.isoformat(),"event":"ENTRY","security_id":sid,"setup_instance_id":uid,"quantity":qty,"cash_after":cash})
        for sid,_,bar in current: marks[sid]=float(bar["close"])
        eq_value=cash+sum(p["qty"]*marks.get(sid,p["entry"]) for sid,p in positions.items()); peak=max(peak,eq_value)
        eq.append({"timestamp":stamp.isoformat(),"cash":cash,"equity":eq_value,"available_cash":cash,
            "open_positions":len(positions),"drawdown_pct":(eq_value/peak-1)*100 if peak else 0.0,
            "market_value_open_positions":sum(p["qty"]*marks.get(sid,p["entry"]) for sid,p in positions.items()),
            "open_structural_risk":sum(p["qty"]*max(0,p["entry"]-p["stop"]) for p in positions.values())})
    last=times[-1] if times else datetime.combine(END,datetime.min.time())
    for sid,p in positions.items():
        mark=marks.get(sid,p["entry"]); unreal=(mark-p["entry"])*p["qty"]
        ledger.append({**p["base"],"status":"OPEN","exit_timestamp":"","exit_price":"","exit_reason":"OPEN_AT_PERIOD_END",
            "gross_pnl":"","net_pnl":"","transaction_costs":p["entry_cost"],"exit_costs":0.0,"gross_R":"","net_R":"",
            "holding_hourly_bars":p["held_bars"],"holding_sessions":len(p["sessions"]),"mfe_R":p["protection"].mfe_r,
            "mae_R":p["mae_R"],"unrealized_gross_pnl":unreal,"mark_timestamp":last.isoformat(),"mark_price":mark,
            "final_current_stop":p["stop"],"protection":asdict(p["protection"]),"stop_move_history":p["stop_history"]})
    # Daily close marks only; the final hourly observation of each session is
    # the MTM point, preserving the source engine's time order.
    daily={}
    for r in eq:
        if date.fromisoformat(r["timestamp"][:10])>=START: daily[r["timestamp"][:10]]=r
    eq=[daily[d] for d in sorted(daily)]
    metrics=_metrics(ledger,eq,initial)
    deployments=[float(r.get("market_value_open_positions",0))/float(r.get("equity",initial)) if float(r.get("equity",initial)) else 0.0 for r in eq]
    position_days={str(n):sum(int(r.get("open_positions",0))==n for r in eq) for n in range(max_positions+1)}
    metrics.update({"average_deployment_pct":sum(deployments)/len(deployments) if deployments else 0.0,
        "median_deployment_pct":median(deployments) if deployments else 0.0,
        "days_deployment_at_least_75_pct":sum(x>=.75 for x in deployments),
        "days_deployment_at_least_90_pct":sum(x>=.90 for x in deployments),
        "average_open_positions":sum(int(r.get("open_positions",0)) for r in eq)/len(eq) if eq else 0.0,
        "position_day_count":position_days,"average_idle_cash":sum(float(r.get("cash",0)) for r in eq)/len(eq) if eq else initial,
        "blocked_signals_by_reason":dict(rejected),"average_open_structural_risk":sum(float(r.get("open_structural_risk",0)) for r in eq)/len(eq) if eq else 0.0})
    metrics.update({"max_open_positions_observed":max((int(r["open_positions"]) for r in eq),default=0),
        "negative_cash_observed":any(float(r["cash"])<-1e-6 for r in eq),"protection_enabled":enable_protection,
        "rejections":dict(rejected),"protected_exit_count":sum(t.get("exit_reason")=="PROTECTIVE_STOP" for t in ledger),
        "gap_protected_exit_count":sum(t.get("exit_reason")=="GAP_PROTECTIVE_STOP" for t in ledger),
        "structural_trail_exit_count":sum(t.get("exit_reason")=="TRAILING_STRUCTURAL_STOP" for t in ledger)})
    return {"trades":ledger,"equity_curve":eq,"events":events,"metrics":metrics,"rejections":dict(rejected)}


def _year_breakdown(trades:Sequence[Mapping[str,Any]],equity:Sequence[Mapping[str,Any]])->list[dict[str,Any]]:
    out=[]; drawdown_by_year:dict[int,list[float]]=defaultdict(list); peak=100000.0
    # Use the same daily marked equity/high-water-mark convention as _metrics.
    # The saved curve's drawdown_pct is intraday path context and can differ.
    for row in equity:
        year=int(str(row.get("timestamp",row.get("date","")))[:4])
        value=float(row["equity"]); peak=max(peak,value)
        drawdown_by_year[year].append((value/peak-1)*100 if peak else 0.0)
    for year in (2023,2024):
        group=[t for t in trades if t.get("status")=="CLOSED" and str(t.get("entry_timestamp",""))[:4]==str(year)]
        values=[float(t["net_pnl"]) for t in group]; gross=[float(t["gross_pnl"]) for t in group]
        wins=sum(x>0 for x in values); gp=sum(x for x in values if x>0); gl=abs(sum(x for x in values if x<0))
        gross_wins=sum(x for x in gross if x>0); gross_losses=abs(sum(x for x in gross if x<0))
        year_marks=[r for r in equity if str(r.get("timestamp",r.get("date","")))[:4]==str(year)]
        before=[r for r in equity if str(r.get("timestamp",r.get("date","")))[:4]<str(year)]
        start_equity=float(before[-1]["equity"]) if before else 100000.0
        out.append({"year":year,"trades":len(group),"win_rate":wins/len(group) if group else 0,
            "net_pf":gp/gl if gl else (None if gp else 0),"gross_pnl":sum(gross),"net_pnl":sum(values),
            "gross_pf":gross_wins/gross_losses if gross_losses else (None if gross_wins else 0),
            "gross_expectancy":sum(gross)/len(group) if group else 0.0,
            "net_expectancy":sum(values)/len(group) if group else 0.0,
            "return":float(year_marks[-1]["equity"])/start_equity-1 if year_marks and start_equity else 0.0,
            "costs":sum(float(t["transaction_costs"]) for t in group),
            "max_dd_pct":min(drawdown_by_year.get(year,()),default=0.0)})
    return out


def _distribution(values:Sequence[float])->dict[str,Any]:
    if not values:return {"count":0,"mean":None,"median":None,"p10":None,"p90":None}
    v=sorted(values)
    def q(p:float)->float:return v[min(len(v)-1,int((len(v)-1)*p))]
    return {"count":len(v),"mean":sum(v)/len(v),"median":(v[(len(v)-1)//2]+v[len(v)//2])/2,
            "p10":q(.10),"p90":q(.90)}


def _breakdown(trades:Sequence[Mapping[str,Any]],field:str)->dict[str,dict[str,Any]]:
    result={}
    for label in sorted({str(t.get(field) or "UNKNOWN") for t in trades if t.get("status")=="CLOSED"}):
        group=[t for t in trades if t.get("status")=="CLOSED" and str(t.get(field) or "UNKNOWN")==label]
        net=[float(t["net_pnl"]) for t in group]; gross=[float(t["gross_pnl"]) for t in group]
        wins=sum(x>0 for x in net); gp=sum(x for x in net if x>0); gl=abs(sum(x for x in net if x<0))
        gross_wins=sum(x for x in gross if x>0); gross_losses=abs(sum(x for x in gross if x<0))
        result[label]={"trades":len(group),"win_rate":wins/len(group) if group else 0.0,
            "net_pf":gp/gl if gl else (None if gp else 0.0),"expectancy_R":sum(float(t.get("net_R") or 0) for t in group)/len(group) if group else 0.0,
            "gross_pf":gross_wins/gross_losses if gross_losses else (None if gross_wins else 0.0),
            "gross_expectancy":sum(gross)/len(group) if group else 0.0,
            "net_expectancy":sum(net)/len(group) if group else 0.0,
            "gross_pnl":sum(gross),"net_pnl":sum(net),"transaction_costs":sum(float(t.get("transaction_costs",0)) for t in group),
            "MFE_R":_distribution([float(t.get("mfe_R") or 0) for t in group]),
            "MAE_R":_distribution([float(t.get("mae_R") or 0) for t in group])}
    return result


def _protection_counterfactual(trades:Sequence[Mapping[str,Any]],bars:Mapping[str,Sequence[Mapping[str,Any]]])->dict[str,Any]:
    protected=[t for t in trades if t.get("status")=="CLOSED" and t.get("exit_reason") in
               {"PROTECTIVE_STOP","TRAILING_STRUCTURAL_STOP","GAP_PROTECTIVE_STOP"}]
    eventual_stop=eventual_target=unresolved=0; details=[]
    for trade in protected:
        sid=str(trade["security_id"]); exit_at=_stamp(trade["exit_timestamp"]); original=float(trade["original_stop"]); target=float(trade["target_price"])
        terminal="UNRESOLVED"
        for bar in bars.get(sid,()):
            stamp=_stamp(bar["date"])
            if stamp<=exit_at:continue
            op,hi,lo=float(bar["open"]),float(bar["high"]),float(bar["low"])
            if op<=original:terminal="ORIGINAL_STOP";break
            if op>=target:terminal="TARGET";break
            if lo<=original:terminal="ORIGINAL_STOP";break
            if hi>=target:terminal="TARGET";break
        eventual_stop+=terminal=="ORIGINAL_STOP"; eventual_target+=terminal=="TARGET"; unresolved+=terminal=="UNRESOLVED"
        details.append({"setup_instance_id":trade["setup_instance_id"],"actual_exit_reason":trade["exit_reason"],
            "actual_realized_R":trade.get("realized_R"),"counterfactual_terminal_event":terminal,"hindsight_only":True})
    return {"protected_stop_exits":len(protected),
        "protective_stop_exits":sum(t.get("exit_reason")=="PROTECTIVE_STOP" for t in protected),
        "gap_protective_stop_exits":sum(t.get("exit_reason")=="GAP_PROTECTIVE_STOP" for t in protected),
        "structural_trail_exits":sum(t.get("exit_reason")=="TRAILING_STRUCTURAL_STOP" for t in protected),
        "trades_saved_from_eventual_original_stop":eventual_stop,
        "protected_winners_cut_before_eventual_target":eventual_target,"unresolved_to_2024_cutoff":unresolved,
        "average_realized_R_on_protected_exits":sum(float(t.get("realized_R") or 0) for t in protected)/len(protected) if protected else 0.0,
        "counterfactual_is_hindsight_only":True,"trade_details":details}


def _compact_metrics(data:Mapping[str,Any],signal_count:int)->dict[str,Any]:
    trades=data["trades"]; m=data["metrics"]; closed=[t for t in trades if t.get("status")=="CLOSED"]
    gross=[float(t["gross_pnl"]) for t in closed]; net=[float(t["net_pnl"]) for t in closed]
    gp=sum(x for x in gross if x>0); gl=abs(sum(x for x in gross if x<0))
    risk_util=[float(t.get("risk_utilization") or 0.0) for t in trades]
    return {"signals":signal_count,"executed_trades":m["executed_trade_count"],"closed_trades":len(closed),
        "open_trades":sum(t.get("status")=="OPEN" for t in trades),"blocked_trades":sum(data.get("rejections",{}).values()),
        "win_rate":m["win_rate"],"gross_pf":gp/gl if gl else (None if gp else 0.0),"net_pf":m["profit_factor"],
        "gross_expectancy":sum(gross)/len(gross) if gross else 0.0,"net_expectancy":m["net_expectancy"],
        "expectancy_R":m["expectancy_R"],"gross_pnl":sum(gross),"net_pnl":sum(net),
        "return":m["net_return_mtm"],"max_drawdown_pct":m["daily_max_drawdown_pct"],
        "average_holding_sessions":m["average_holding_sessions"],"median_holding_sessions":m["median_holding_sessions"],
        "transaction_costs":m["transaction_costs"],"average_deployment_pct":m["average_deployment_pct"],
        "median_deployment_pct":m["median_deployment_pct"],"days_in_period":len(data["equity_curve"]),
        "days_deployment_at_least_75_pct":m["days_deployment_at_least_75_pct"],
        "days_deployment_at_least_90_pct":m["days_deployment_at_least_90_pct"],
        "position_day_count":m["position_day_count"],"average_idle_cash":m["average_idle_cash"],
        "blocked_signals_by_reason":m["blocked_signals_by_reason"],
        "average_risk_utilization":sum(risk_util)/len(risk_util) if risk_util else 0.0,
        "median_risk_utilization":median(risk_util) if risk_util else 0.0,
        "average_open_positions":m["average_open_positions"],
        "max_open_positions":m["max_open_positions_observed"],"negative_cash":m["negative_cash_observed"]}


def run(*,signal_path:Path=REPLAY_OUTPUT/"v3_entry_ready_events.csv",data_path:Path|None=None,
       config_path:Path=CONFIG,output_dir:Path=OUTPUT)->dict[str,Any]:
    # Hourly data are loaded via the same manifested, cutoff-enforcing loader as
    # V2's replay; caller may inject a mapping in unit tests.
    if data_path is None:
        from src.backtest.data import load_hourly_dataset_through
        from src.data.hourly import HOURLY_MANIFEST
        from src.data.market_data import DATA_ROOT,MANIFEST_PATH
        from src.data.universe.provider import PIT_TRADABLE
        from datetime import date as dt_date
        bars,_=load_hourly_dataset_through(universe_id=PIT_TRADABLE,start_date=dt_date(2022,10,1),end_date=END,
            data_root=DATA_ROOT,manifest_path=HOURLY_MANIFEST,daily_manifest_path=MANIFEST_PATH)
    else:
        raise ValueError("data_path injection is intentionally unsupported; pass an in-memory adapter in tests")
    raw=yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}; pcfg=raw.get("portfolio",{})
    signals=_signals(signal_path); costs=_cost_config(PROJECT_ROOT/"config/settings.yaml")
    initial=float(pcfg.get("initial_capital",100000)); risk=float(pcfg.get("risk_fraction",.0075)); maxpos=int(pcfg.get("max_open_positions",4))
    max_notional=float(pcfg.get("max_single_position_notional_fraction",.35))
    total_risk=float(pcfg.get("max_total_open_risk_fraction",.035))
    entry=_entry_only(signals,bars,costs,initial=initial,risk=risk,max_positions=maxpos,
        max_notional_fraction=max_notional,total_risk_fraction=total_risk)
    full=_protected_run(signals,bars,costs,initial=initial,risk_fraction=risk,max_positions=maxpos,
        max_notional_fraction=max_notional,total_risk_fraction=total_risk,enable_protection=True)
    output_dir.mkdir(parents=True,exist_ok=True)
    for arm,data in (("entry_only",entry),("full",full)):
        _write(output_dir/f"v3_{arm}_trades.csv",data["trades"]); _write(output_dir/f"v3_{arm}_equity_curve.csv",data["equity_curve"])
    v2_path=PROJECT_ROOT/"data/processed/backtests/top_down_swing_v2_daily_structure/portfolio_management/control_no_time_summary.json"
    v2_all=json.loads(v2_path.read_text(encoding="utf-8")) if v2_path.is_file() else {}
    v2_fields=("signal_count","executed_trades","closed_trades","open_positions","win_rate","gross_profit_factor",
        "net_profit_factor","gross_expectancy","net_expectancy","gross_pnl_closed","net_pnl_closed","gross_return",
        "net_return_mtm","max_drawdown_pct","average_holding_sessions","median_holding_sessions","transaction_costs",
        "average_deployment_pct","median_deployment_pct","average_open_positions","blocked_trades","days_deployment_at_least_75_pct",
        "days_deployment_at_least_90_pct")
    v2_saved={k:v2_all[k] for k in v2_fields if k in v2_all}
    trade_types={"setup_family":"setup_family","regime":"regime","primary_geometry":"primary_geometry","ema_confluence":"ema_confluence"}
    family_reports={arm:_breakdown(data["trades"],"setup_family") for arm,data in (("entry_only",entry),("full",full))}
    diagnostic_reports={arm:{key:_breakdown(data["trades"],field) for key,field in trade_types.items() if key!="setup_family"}
                        for arm,data in (("entry_only",entry),("full",full))}
    protection_cf=_protection_counterfactual(full["trades"],bars)
    summary={"strategy_version":"top_down_swing_v3","period":{"start":START.isoformat(),"end":END.isoformat()},
        "2025_2026_loaded":False,"signal_count":len(signals),
        "v3_entry_only":_compact_metrics(entry,len(signals)),"v3_full":_compact_metrics(full,len(signals)),
        "v3_entry_only_by_year":_year_breakdown(entry["trades"],entry["equity_curve"]),
        "v3_full_by_year":_year_breakdown(full["trades"],full["equity_curve"]),
        "v3_entry_only_by_setup_family":family_reports["entry_only"],"v3_full_by_setup_family":family_reports["full"],
        "v3_entry_only_diagnostics":diagnostic_reports["entry_only"],"v3_full_diagnostics":diagnostic_reports["full"],
        "v2_frozen_control_no_time_baseline":v2_saved,
        "comparison":{"V2_frozen_control_no_time":v2_saved,
            "V3_entry_only":_compact_metrics(entry,len(signals)),"V3_full":_compact_metrics(full,len(signals))},
        "profit_protection_analysis":protection_cf,
        "signals":len(signals),"executed_entry_only":entry["metrics"]["executed_trade_count"],
        "executed_full":full["metrics"]["executed_trade_count"],
        "protection_delta":{"net_pnl":full["metrics"]["net_pnl"]-entry["metrics"]["net_pnl"],
            "net_pf":full["metrics"]["profit_factor"]-entry["metrics"]["profit_factor"] if full["metrics"]["profit_factor"] is not None and entry["metrics"]["profit_factor"] is not None else None,
            "return":full["metrics"]["net_return_mtm"]-entry["metrics"]["net_return_mtm"],
            "max_drawdown_pct":full["metrics"]["daily_max_drawdown_pct"]-entry["metrics"]["daily_max_drawdown_pct"]},
        "limitations":["V2 baseline is referenced from saved control_no_time artifacts; this runner does not rerun V2.",
            "V3 entry-only uses the existing fixed-bracket hourly engine; V3 full applies next-bar one-way stop protection."]}
    (output_dir/"v3_portfolio_summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True,default=str)+"\n",encoding="utf-8")
    return {"summary":summary,"entry_only":entry,"full":full}


def main()->None:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--signals",type=Path,default=REPLAY_OUTPUT/"v3_entry_ready_events.csv");p.add_argument("--config",type=Path,default=CONFIG);p.add_argument("--output-directory",type=Path,default=OUTPUT)
    a=p.parse_args(); result=run(signal_path=a.signals,config_path=a.config,output_dir=a.output_directory); print(json.dumps(result["summary"].get("comparison"),indent=2,default=str))


if __name__=="__main__":main()
