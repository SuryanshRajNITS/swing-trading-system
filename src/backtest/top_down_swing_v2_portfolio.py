"""Portfolio execution and diagnostics for the separate V2 signal replay."""
from __future__ import annotations
import argparse,csv,json
from datetime import date,datetime
from pathlib import Path
from statistics import median
from typing import Any,Mapping,Sequence
from zoneinfo import ZoneInfo
import yaml
from src.configuration import PROJECT_ROOT
from src.data.market_data import DATA_ROOT,MANIFEST_PATH
from src.data.hourly import HOURLY_MANIFEST
from src.data.universe.provider import PIT_TRADABLE,UniverseProvider
from src.backtest.data import load_daily_dataset_through,load_hourly_dataset_through
from src.backtest.models import CostModel
from src.backtest.top_down_swing_portfolio import _cost_config,_metrics,_run_portfolio
from src.backtest.top_down_swing_v2_replay import START,END,CONFIG,OUTPUT,_csv

OUT=OUTPUT; SIGNALS=OUTPUT/"v2_entry_ready_events.csv"; TZ=ZoneInfo("Asia/Kolkata")
V1_TRADES=PROJECT_ROOT/"data"/"processed"/"backtests"/"top_down_swing_v1_sniper"/"patient_sniper_1_trades.csv"
MANUAL=[("HINDALCO","2023-10-09"),("ONGC","2023-10-27"),("APOLLOHOSP","2023-11-03"),("EICHERMOT","2023-10-04"),("TATACONSUM","2023-09-29")]

def _stamp(x:Any)->datetime:
    d=x if isinstance(x,datetime) else datetime.fromisoformat(str(x)); return d.replace(tzinfo=TZ) if d.tzinfo is None else d.astimezone(TZ)
def _rows(path:Path)->list[dict[str,str]]:
    with path.open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def x_date(signal:Mapping[str,Any])->str:return str(signal.get("setup_signal_date") or signal.get("date") or "")[:10]
def _rank(s:Mapping[str,Any])->tuple:
    p=s["plan"]; result=s["result"]
    location={"INSIDE_RECENT_DAILY_SUPPORT":0,"NEAR_RECENT_DAILY_SUPPORT":1}.get(str(s.get("daily_location_state")),2)
    target_strength={"MAJOR":0,"MEANINGFUL":1}.get(str(p.get("target_zone_strength")),2)
    target_tf={"WEEKLY":0,"DAILY":1}.get(str(p.get("target_timeframe")),2)
    confirm={"CLEARANCE":0,"RECLAIM":1,"SUPPORT_REACTION":2}.get(result.get("confirmation_type"),3)
    volume=0 if p.get("volume_confirmation_passed") else 1
    return (location,target_strength,target_tf,-float(p.get("planned_reward_r") or 0),confirm,volume,str(s["security_id"]),str(s["setup_instance_id"]))
def _read_signals(path:Path)->list[dict[str,Any]]:
    signals=[]
    for r in _rows(path):
        result=json.loads(r.get("result") or "{}"); plan=result.get("plan") or {}
        if not result or not plan: continue
        signals.append({**r,"result":result,"plan":plan,"support":json.loads(r.get("support") or "null"),"resistance":json.loads(r.get("resistance") or "null"),"daily_location_state":r.get("daily_location_state"),"setup_instance_id":r["setup_instance_id"],
            "setup_signal_date":r.get("setup_signal_date",r.get("date")),"symbol":r.get("symbol"),
            "executable_timestamp":_stamp(result["entry_timestamp"]),"confirmation_timestamp":_stamp(result["confirmation_timestamp"]),
            "weekly_selection_path":"V2_DAILY_STRUCTURE"})
    return signals
def _pf(values:Sequence[float])->float|None:
    wins=sum(x for x in values if x>0); losses=abs(sum(x for x in values if x<0)); return wins/losses if losses else (None if wins else 0.0)
def _metrics2(trades:Sequence[Mapping[str,Any]],initial:float,portfolio_metrics:Mapping[str,Any])->dict[str,Any]:
    closed=[t for t in trades if t.get("status")=="CLOSED"]; gross=[float(t["gross_pnl"]) for t in closed]; net=[float(t["net_pnl"]) for t in closed]
    months=24; ret=sum(net)/initial if initial else 0
    return {"initial_capital":initial,"trade_count":len(closed),"open_positions":sum(t.get("status")=="OPEN" for t in trades),
      "win_rate":sum(x>0 for x in net)/len(net) if net else 0,"gross_expectancy":sum(gross)/len(gross) if gross else 0,
      "net_expectancy":sum(net)/len(net) if net else 0,"gross_profit_factor":_pf(gross),"net_profit_factor":_pf(net),
      "gross_pnl":sum(gross),"net_pnl":sum(net),"gross_return":sum(gross)/initial if initial else 0,
      "net_return":float(portfolio_metrics.get("net_return_mtm",ret)),"gross_return":float(portfolio_metrics.get("gross_return_before_costs",sum(gross)/initial if initial else 0)),
      "ending_equity_mtm":portfolio_metrics.get("ending_equity_mtm"),"trades_per_month":len(closed)/months,"transaction_costs":sum(float(t.get("transaction_costs",0)) for t in trades),
      "dp_costs":sum(float((t.get("entry_cost_breakdown") or {}).get("dp_charge",0))+float((t.get("exit_cost_breakdown") or {}).get("dp_charge",0)) for t in trades),
      "average_holding_sessions":sum(float(t.get("holding_sessions",0)) for t in closed)/len(closed) if closed else 0,
      "median_holding_sessions":median([float(t.get("holding_sessions",0)) for t in closed]) if closed else 0,
      "average_planned_structural_rr":sum(float(t.get("planned_rr",0)) for t in trades)/len(trades) if trades else 0,
      "median_planned_structural_rr":median([float(t.get("planned_rr",0)) for t in trades]) if trades else 0,
      "average_stop_distance_pct":sum(float(t.get("stop_distance_pct",0)) for t in trades)/len(trades) if trades else 0,
      "median_stop_distance_pct":median([float(t.get("stop_distance_pct",0)) for t in trades]) if trades else 0,
      "max_drawdown_pct":min((float(r.get("drawdown_pct",0)) for r in _last_equity),default=0)}
_last_equity=[]

def _mfe(trades:Sequence[Mapping[str,Any]],signals:Sequence[Mapping[str,Any]],daily:Mapping[str,Sequence[Mapping[str,Any]]])->list[dict[str,Any]]:
    byid={s["setup_instance_id"]:s for s in signals}; out=[]
    for t in trades:
        s=byid.get(t.get("setup_instance_id")); bars=daily.get(str(t.get("security_id")),[])
        if not s: continue
        entry_day=str(t["entry_timestamp"])[:10]; start=next((i for i,b in enumerate(bars) if str(b["date"])[:10]>entry_day),len(bars))
        row={"trade_id":t["trade_id"],"security_id":t["security_id"],"entry_timestamp":t["entry_timestamp"],
             "mfe_before_exit_pct":t.get("mfe_pct"),"mae_before_exit_pct":t.get("mae_pct")}
        for n in (5,10,20):
            h=bars[start:start+n]; high=max((float(x["high"]) for x in h),default=None)
            row[f"mfe_next_{n}_sessions_pct"]=(high/float(t["entry_price"])-1)*100 if high is not None else None
        row["target_capture_ratio_20s"]=(float(t.get("gross_pnl",0))/(float(t["quantity"])*(max((float(x["high"]) for x in bars[start:start+20]),default=float(t["entry_price"]))-float(t["entry_price"])))) if bars[start:start+20] and float(t.get("gross_pnl") or 0)>0 else None
        support=s.get("support") if isinstance(s.get("support"),Mapping) else {}
        support_low=float(support.get("lower_bound",s["plan"].get("daily_support_lower",0)))
        support_high=float(support.get("upper_bound",s["plan"].get("daily_support_upper",0)))
        exit_day=str(t.get("exit_timestamp") or END.isoformat())[:10]
        window=[b for b in bars if entry_day < str(b["date"])[:10] <= exit_day]
        retest_i=next((i for i,b in enumerate(window) if float(b["low"])<=support_high and float(b["low"])>float(t["stop_price"])),None)
        row["trade_worked_after_zone_retest"]=bool(retest_i is not None and
            max((float(b["high"]) for b in window[retest_i+1:]),default=float(t["entry_price"]))>float(t["entry_price"]))
        row["stop_distance_pct"]=s["plan"].get("stop_distance_pct"); row["stop_distance_atr"]=s["plan"].get("stop_distance_atr")
        row["entry_to_zone_distance"]=s["plan"].get("entry_to_support"); row["zone_width_pct"]=s["plan"].get("zone_width_pct"); row["buffer_below_support"]=s["plan"].get("buffer_below_support")
        out.append(row)
    return out

def run(*,signal_path:Path=SIGNALS,output_dir:Path=OUT,data_root:Path=DATA_ROOT,daily_manifest:Path=MANIFEST_PATH,
       hourly_manifest:Path=HOURLY_MANIFEST,config_path:Path=CONFIG,v1_trades_path:Path=V1_TRADES)->dict[str,Any]:
    raw=yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}; sigs=_read_signals(signal_path)
    sod_path=output_dir/"v2_sod_watchlist_diagnostics.csv"
    sod_rows=_rows(sod_path) if sod_path.exists() else []
    for row in sod_rows: row.setdefault("recent_window_sessions_used",row.get("recent_window_sessions"))
    if sod_rows: _csv(sod_path,sod_rows)
    sod_lookup={(r.get("sod_date"),r.get("security_id")):r for r in sod_rows}
    for s in sigs:
        sod=sod_lookup.get((s.get("setup_signal_date"),s.get("security_id")),{})
        s["daily_location_state"]=sod.get("recent_daily_location_state",s.get("daily_location_state"))
        s["recent_window_sessions_used"]=sod.get("recent_window_sessions")
    daily,_=load_daily_dataset_through(universe_id=PIT_TRADABLE,start_date=date(2022,7,1),end_date=END,data_root=data_root,manifest_path=daily_manifest)
    bars,_=load_hourly_dataset_through(universe_id=PIT_TRADABLE,start_date=date(2022,10,1),end_date=END,data_root=data_root,manifest_path=hourly_manifest,daily_manifest_path=daily_manifest)
    costs=_cost_config(PROJECT_ROOT/"config"/"settings.yaml"); portfolio=raw.get("portfolio",{}); capitals=portfolio.get("capital_sensitivity",[10000,25000,50000,100000,200000])
    runs={}; sensitivities=[]; headline_trades=[]; headline_equity=[]
    for capital in capitals:
        result=_run_portfolio(sigs,bars,costs,initial_capital=float(capital),risk_fraction=float(portfolio.get("risk_fraction",.01)),max_positions=1,period_start=START,rank_key=_rank)
        metrics=result["metrics"]; trades=result["trades"]
        sensitivities.append({"capital":capital,"trades":metrics["trade_count"],"win_rate":metrics["win_rate"],"gross_pf":_pf([float(t["gross_pnl"]) for t in trades if t.get("status")=="CLOSED"]),"net_pf":metrics["profit_factor"],"gross_return":metrics["gross_return_before_costs"],"net_return":metrics["net_return_mtm"],"transaction_costs":metrics["transaction_costs"],"drawdown_pct":metrics["daily_max_drawdown_pct"]})
        runs[str(capital)]=result
        if float(capital)==100000: headline_trades=trades;headline_equity=result["equity_curve"]
    if not headline_trades and 100000 not in capitals: raise ValueError("Primary V2 capital 100000 is missing from capital_sensitivity.")
    _last_equity[:]=headline_equity; metrics=_metrics2(headline_trades,100000,runs["100000"]["metrics"])
    by_signal={s["setup_instance_id"]:s for s in sigs}
    for t in headline_trades:
        signal=by_signal.get(t.get("setup_instance_id"),{}); plan=signal.get("plan",{})
        t["strategy_version"]="top_down_swing_v2_daily_structure"
        t["planned_rr"]=plan.get("planned_reward_r",t.get("planned_rr"))
        t["stop_distance_pct"]=plan.get("stop_distance_pct")
        t["stop_distance_atr"]=plan.get("stop_distance_atr")
        t["daily_support_zone_id"]=plan.get("daily_support_zone_id")
        t["daily_support_lower"]=plan.get("daily_support_lower")
        t["daily_support_upper"]=plan.get("daily_support_upper")
        t["target_type"]=plan.get("target_timeframe",t.get("target_type"))
    for t in headline_trades: t["strategy_version"]="top_down_swing_v2_daily_structure"
    # Recompute after attaching plan metadata so stop-distance summary fields
    # reflect the signal plan instead of silently defaulting to zero.
    metrics=_metrics2(headline_trades,100000,runs["100000"]["metrics"])
    output_dir.mkdir(parents=True,exist_ok=True); _csv(output_dir/"v2_trades_100k.csv",headline_trades); _csv(output_dir/"v2_equity_curve_100k.csv",headline_equity);_csv(output_dir/"v2_capital_sensitivity.csv",sensitivities)
    mfe=_mfe(headline_trades,sigs,daily);_csv(output_dir/"v2_big_rally_diagnostics.csv",mfe);_csv(output_dir/"v2_stop_diagnostics.csv",mfe)
    by_year=[]
    for year in (2023,2024):
        ts=[t for t in headline_trades if str(t.get("entry_timestamp",""))[:4]==str(year) and t.get("status")=="CLOSED"]
        ye=[r for r in headline_equity if str(r.get("timestamp",""))[:4]==str(year)]
        by_year.append({"year":year,"trade_count":len(ts),"gross_pf":_pf([float(t["gross_pnl"]) for t in ts]),"net_pf":_pf([float(t["net_pnl"]) for t in ts]),"gross_expectancy":sum(float(t["gross_pnl"]) for t in ts)/len(ts) if ts else 0,"net_expectancy":sum(float(t["net_pnl"]) for t in ts)/len(ts) if ts else 0,"net_pnl":sum(float(t["net_pnl"]) for t in ts),"return_contribution":sum(float(t["net_pnl"]) for t in ts)/100000,"drawdown_contribution_pct":min((float(r.get("drawdown_pct",0)) for r in ye),default=0)})
    _csv(output_dir/"v2_year_breakdown.csv",by_year)
    _csv(output_dir/"v2_confirmation_breakdown.csv",[{"confirmation_type":k,"trade_count":sum(t.get("confirmation_archetype")==k for t in headline_trades)} for k in sorted({str(t.get("confirmation_archetype")) for t in headline_trades})])
    _csv(output_dir/"v2_target_breakdown.csv",[{"target_timeframe":k,"trade_count":sum(t.get("target_type")==k for t in headline_trades)} for k in sorted({str(t.get("target_type")) for t in headline_trades})])
    comparison=[]
    v1=list(csv.DictReader(v1_trades_path.open(encoding="utf-8-sig",newline=""))) if v1_trades_path.exists() else []
    sod_rows=_rows(output_dir/"v2_sod_watchlist_diagnostics.csv") if (output_dir/"v2_sod_watchlist_diagnostics.csv").exists() else []
    reject_rows=_rows(output_dir/"v2_rejected_candidates.csv") if (output_dir/"v2_rejected_candidates.csv").exists() else []
    for sym,day in MANUAL:
        candidates=[s for s in sigs if s.get("symbol")==sym and abs((date.fromisoformat(x_date(s))-date.fromisoformat(day)).days)<=10]
        nearest=min(candidates,key=lambda x:abs((date.fromisoformat(str(x_date(x)))-date.fromisoformat(day)).days),default=None)
        sod_candidates=[x for x in sod_rows if x.get("symbol")==sym and abs((date.fromisoformat(str(x.get("sod_date")))-date.fromisoformat(day)).days)<=2]
        sod=min(sod_candidates,key=lambda x:abs((date.fromisoformat(str(x.get("sod_date")))-date.fromisoformat(day)).days),default={})
        rej_candidates=[x for x in reject_rows if x.get("symbol")==sym and abs((date.fromisoformat(str(x.get("date")))-date.fromisoformat(day)).days)<=10]
        rejection=min(rej_candidates,key=lambda x:abs((date.fromisoformat(str(x.get("date")))-date.fromisoformat(day)).days),default={})
        prior=[t for t in v1 if t.get("symbol")==sym and abs((date.fromisoformat(str(t.get("signal_date",t.get("entry_timestamp",""))[:10]))-date.fromisoformat(day)).days)<=10]
        old=min(prior,key=lambda t:abs((date.fromisoformat(str(t.get("signal_date",t.get("entry_timestamp",""))[:10]))-date.fromisoformat(day)).days),default={})
        comparison.append({"symbol":sym,"reference_date":day,"v1_entry":old.get("entry_price"),"v1_stop":old.get("stop_price"),"v1_target":old.get("target_price"),"v1_target_timeframe":old.get("target_type"),"v1_planned_rr":old.get("planned_rr"),
          "v2_sod_date":sod.get("sod_date"),"v2_location_state":sod.get("recent_daily_location_state"),"v2_interesting_today":sod.get("interesting_today"),
          "v2_daily_support_lower":sod.get("nearest_recent_support_lower"),"v2_daily_support_upper":sod.get("nearest_recent_support_upper"),
          "v2_daily_resistance_lower":sod.get("nearest_recent_resistance_lower"),"v2_daily_resistance_upper":sod.get("nearest_recent_resistance_upper"),
          "v2_confirmation":nearest.get("result",{}).get("confirmation_type") if nearest else None,"v2_entry":nearest.get("plan",{}).get("entry") if nearest else None,
          "v2_stop":nearest.get("plan",{}).get("stop") if nearest else None,"v2_target":nearest.get("plan",{}).get("target") if nearest else None,
          "v2_target_timeframe":nearest.get("plan",{}).get("target_timeframe") if nearest else None,"v2_planned_rr":nearest.get("plan",{}).get("planned_reward_r") if nearest else None,
          "v2_rejection_reason":rejection.get("reason_code"),
          "mechanical_difference":"No entry: SOD map did not mark the stock interesting or H1/volume/structural gates rejected it." if nearest is None else "V2 uses the recent daily support thesis, daily stop, structural resistance target and confirmed H1 entry."})
    _csv(output_dir/"manual_trade_comparisons.csv",comparison)
    bpcl=[s for s in sigs if s.get("symbol")=="BPCL" and str(s.get("setup_signal_date",""))[:7]=="2023-09"]
    bpcl_audit={"reference":"BPCL September 2023 manual diagnostic; no prices hard-coded","ready_events":bpcl,
       "daily_zone_rows":_rows_if_exists(output_dir/"v2_daily_zones.csv","BPCL","2023-09"),
       "sod_recent_daily_map":[r for r in sod_rows if r.get("symbol")=="BPCL" and str(r.get("sod_date",""))[:7]=="2023-09"],
       "window_candles":[r for r in _rows(output_dir/"bpcl_recent_daily_windows.csv") if str(r.get("sod_date",""))[:7]=="2023-09"] if (output_dir/"bpcl_recent_daily_windows.csv").exists() else []}
    (output_dir/"manual_bpcl_audit.json").write_text(json.dumps(bpcl_audit,indent=2,sort_keys=True,ensure_ascii=True,default=str)+"\n",encoding="utf-8")
    v1_100k={}
    if V1_TRADES.exists():
        sensitivity_path=V1_TRADES.parent/"capital_sensitivity.csv"
        if sensitivity_path.exists():
            match=next((r for r in _rows(sensitivity_path) if r.get("mode")=="patient_sniper_1" and float(r.get("starting_capital",0))==100000),None)
            if match:
                v1_100k={"trades":match.get("trades"),"win_rate":match.get("win_rate"),"gross_pf":match.get("gross_pf"),"net_pf":match.get("net_pf"),"gross_return":float(match.get("gross_pnl",0))/100000,"net_return":float(match.get("return_pct",0))/100,"max_drawdown_pct":match.get("max_drawdown_pct"),"transaction_costs":match.get("transaction_costs"),"dp_costs":match.get("dp_charges")}
    comparison_rows=[{"strategy":"V1_PATIENT_SNIPER_1_AT_100K","source":"saved 2023-2024 V1 sensitivity artifact","metrics":v1_100k},
                     {"strategy":"V2_DAILY_STRUCTURE_AT_100K","source":"V2 replay","metrics":metrics}]
    _csv(output_dir/"v1_vs_v2_comparison.csv",comparison_rows)
    summary={"strategy_version":"top_down_swing_v2_daily_structure","data_freeze_version":"NIFTY50_PIT_2023_2024_V1","period":{"start":START.isoformat(),"end":END.isoformat()},"2025_2026_loaded":False,"portfolio_metrics":metrics,"v1_patient_sniper_1_100k_saved_metrics":v1_100k,"year_breakdown":by_year,"capital_sensitivity":sensitivities,"cost_scenario":costs.scenario,"risk_fraction":.01,"max_concurrent_positions":1,"exit_semantics":"shared conservative hourly OHLC engine: gap then intrabar stop/target, stop-first ambiguity, no same-entry-bar exit; frozen stop/target","diagnostics":{"mfe_horizons_daily_sessions":[5,10,20],"mfe_is_not_used_for_strategy_decisions":True,"MFE_trade_count":len(mfe)},"rejections":runs["100000"]["rejections"]}
    (output_dir/"v2_summary_100k.json").write_text(json.dumps(summary,indent=2,sort_keys=True,ensure_ascii=True,default=str)+"\n",encoding="utf-8")
    print(json.dumps({"portfolio_metrics":metrics,"year_breakdown":by_year,"capital_sensitivity":sensitivities,"output":str(output_dir)},indent=2,default=str))
    return {"summary":summary,"trades":headline_trades,"sensitivity":sensitivities}
def _rows_if_exists(path:Path,symbol:str,prefix:str)->list[dict[str,str]]:
    return [r for r in _rows(path) if r.get("symbol")==symbol and str(r.get("date",""))[:7]==prefix] if path.exists() else []
def main()->None:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--signals",type=Path,default=SIGNALS);p.add_argument("--output-directory",type=Path,default=OUT);p.add_argument("--config",type=Path,default=CONFIG)
    p.add_argument("--data-root",type=Path,default=DATA_ROOT);p.add_argument("--daily-manifest",type=Path,default=MANIFEST_PATH);p.add_argument("--hourly-manifest",type=Path,default=HOURLY_MANIFEST);p.add_argument("--v1-trades",type=Path,default=V1_TRADES)
    a=p.parse_args();run(signal_path=a.signals,output_dir=a.output_directory,data_root=a.data_root,daily_manifest=a.daily_manifest,hourly_manifest=a.hourly_manifest,config_path=a.config,v1_trades_path=a.v1_trades)
if __name__=="__main__":main()
