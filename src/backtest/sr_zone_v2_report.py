"""Compare saved sr_zone_v2 runs with the frozen sr_baseline_v1 artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from src.backtest.data import load_daily_dataset_through
from src.backtest.sr_zone_v2_run import END, START
from src.configuration import PROJECT_ROOT
from src.data.universe.provider import PIT_TRADABLE
from src.strategy.sr_baseline import SRBaselineConfig, cluster_zones, find_confirmed_swings
from src.strategy.sr_zone_v2 import ZoneConfig, _atr, detect_zones

V1 = PROJECT_ROOT / "data/processed/backtests/sr_baseline_v1"
V2 = PROJECT_ROOT / "data/processed/backtests/sr_zone_v2"
CASES = (("support_test",False),("breakout_retest",False),("both",False),("both",True))


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns=sorted({key for row in rows for key in row})
    with path.open("w",encoding="utf-8",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=columns); writer.writeheader(); writer.writerows(rows)


def generate_report(*, manifest_path: Path = PROJECT_ROOT/"data/processed/market_data_manifest.csv",
                    data_root: Path = PROJECT_ROOT/"data/processed/market_data") -> dict[str, Any]:
    yearly=[]
    daily,_=load_daily_dataset_through(universe_id=PIT_TRADABLE,start_date=START,end_date=END,
                                       data_root=data_root,manifest_path=manifest_path)
    for year in (2023,2024):
        per_sec=[]
        for sid,rows in daily.items():
            prefix=[row for row in rows if str(row["date"])[:4] <= str(year)]
            if not prefix or str(prefix[-1]["date"])[:4] != str(year):
                continue
            zones=detect_zones(prefix,security_id=sid,config=ZoneConfig(),include_lifecycle=True)
            overlaps=sum(1 for i,a in enumerate(zones) for b in zones[i+1:]
                         if a.side==b.side and a.lower_bound<=b.upper_bound and b.lower_bound<=a.upper_bound)
            price=float(prefix[-1]["close"])
            v1cfg=SRBaselineConfig()
            pivots=find_confirmed_swings(prefix,left=v1cfg.swing_left,right=v1cfg.swing_right)
            v1zones=[z for z in cluster_zones(pivots,tolerance_pct=v1cfg.zone_tolerance_pct)
                     if z.pivot_count>=v1cfg.minimum_zone_pivots
                     and len(prefix)-1-z.last_pivot_index<=v1cfg.maximum_zone_age_sessions]
            v1overlaps=sum(1 for i,a in enumerate(v1zones) for b in v1zones[i+1:]
                           if a.kind==b.kind and a.lower<=b.upper and b.lower<=a.upper)
            atr_now=_atr(prefix,len(prefix)-1,14)
            per_sec.append({"year":year,"security_id":sid,"zone_count":len(zones),
                            "overlap_pairs":overlaps,"merged_zone_count":sum(bool(z.merged_ids) for z in zones),
                            "average_width_pct":sum((z.upper_bound-z.lower_bound)/z.center for z in zones)/len(zones) if zones else 0,
                            "average_width_atr":sum((z.upper_bound-z.lower_bound)/atr_now for z in zones)/len(zones) if zones and atr_now else 0,
                            "v1_zone_count":len(v1zones),"v1_overlap_pairs":v1overlaps,
                            "v1_average_width_pct":sum((z.upper-z.lower)/z.center for z in v1zones)/len(v1zones) if v1zones else 0,
                            "touch_episodes":sum(len(z.touch_episodes) for z in zones),
                            "broken_zones":sum(z.broken for z in zones),
                            "signal_zones":0})
        yearly.extend(per_sec)
    comparison=[]; volume=[]
    signal_ids: dict[tuple[str,str,int],set[str]]={}
    bucket_specs=[("<0.8",None,.8),("0.8-1.0",.8,1.0),("1.0-1.2",1.0,1.2),
                  ("1.2-1.5",1.2,1.5),(">1.5",1.5,None)]
    for setup,weekly in CASES:
        name=f"{setup}_r2_weekly-{int(weekly)}_current_zerodha_0bps.json"
        old,new=_read(V1/name),_read(V2/name)
        for version,report in (("sr_baseline_v1",old),("sr_zone_v2",new)):
            for decision in report.get("diagnostics",[]):
                zid=decision.get("zone_id")
                if decision.get("accepted") and zid:
                    signal_ids.setdefault((version,str(decision.get("security_id")),
                                           int(str(decision.get("signal_date",""))[:4])),set()).add(str(zid))
            for subtype,data in report["setup_performance"].items():
                if (setup=="support_test" and subtype!="support_test") or (setup=="breakout_retest" and subtype!="breakout_retest"):
                    continue
                trades=[t for t in report.get("trades",[]) if t.get("metadata",{}).get("setup_type")==subtype]
                gross=sum(float(t.get("gross_pnl",0)) for t in trades)
                net=sum(float(t.get("net_pnl",0)) for t in trades)
                winners=sum(max(0,float(t.get("net_pnl",0))) for t in trades)
                losers=-sum(min(0,float(t.get("net_pnl",0))) for t in trades)
                comparison.append({"case":name,"strategy":version,"setup_type":subtype,
                                   "trades":data["trades"],"gross_expectancy":gross/len(trades) if trades else 0,
                                   "net_expectancy":net/len(trades) if trades else 0,
                                   "profit_factor":winners/losers if losers else None,
                                   "turnover":sum(float(t.get("entry_price",0))*int(t.get("quantity",0))+
                                                  float(t.get("exit_price",0))*int(t.get("quantity",0)) for t in trades),
                                   "transaction_costs":data["transaction_costs"],
                                   "trading_net_return":report["metrics"]["trading_net_return"]})
        if setup=="breakout_retest" and not weekly:
            trades=[t for t in new.get("trades",[]) if t.get("metadata",{}).get("setup_type")=="breakout_retest"]
            for label,low,high in bucket_specs:
                selected=[]
                for trade in trades:
                    ratio=trade.get("metadata",{}).get("breakout_volume_ratio")
                    above_lower=(low is None or (ratio>low if label==">1.5" else ratio>=low)) if ratio is not None else False
                    if above_lower and (high is None or ratio<high):
                        selected.append(trade)
                volume.append({"bucket":label,"trades":len(selected),
                               "gross_expectancy":sum(float(t["gross_pnl"]) for t in selected)/len(selected) if selected else None,
                               "net_expectancy":sum(float(t["net_pnl"]) for t in selected)/len(selected) if selected else None,
                               "profit_factor":sum(max(0,float(t["net_pnl"])) for t in selected)/
                                                -sum(min(0,float(t["net_pnl"])) for t in selected)
                                                if any(float(t["net_pnl"])<0 for t in selected) else None})
    for row in yearly:
        year=int(row["year"]); sid=str(row["security_id"])
        row["v1_signal_zones"]=len(signal_ids.get(("sr_baseline_v1",sid,year),set()))
        row["v2_signal_zones"]=len(signal_ids.get(("sr_zone_v2",sid,year),set()))
    V2.mkdir(parents=True,exist_ok=True)
    _write_csv(V2/"zone_quality_yearly.csv",yearly)
    _write_csv(V2/"v1_v2_comparison.csv",comparison)
    _write_csv(V2/"breakout_volume_buckets.csv",volume)
    result={"period":[START.isoformat(),END.isoformat()],"comparisons":comparison,
            "zone_quality_by_stock_year":yearly,"breakout_volume_buckets":volume,
            "v1_reference_zone_count":1236,"v1_reference_overlap_pairs":337,
            "costs":"CURRENT_ZERODHA_EQUITY_DELIVERY_2026 assumptions, 0 bps slippage; not historical fees",
            "fixed_operating_costs_excluded_from_trade_quality":True}
    (V2/"research_report.json").write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False),encoding="utf-8")
    summary=[]
    for setup,weekly in CASES:
        artifact=f"{setup}_r2_weekly-{int(weekly)}_current_zerodha_0bps.json"
        report=_read(V2/artifact)
        summary.append({"artifact":artifact,"strategy_version":"sr_zone_v2","setup":setup,
                        "weekly_filter":weekly,"metrics":report["metrics"],
                        "setup_performance":report["setup_performance"]})
    (V2/"comparison_summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True,allow_nan=False),encoding="utf-8")
    return result


if __name__=="__main__":
    report=generate_report()
    print(json.dumps({"comparison_rows":len(report["comparisons"]),
                      "volume_buckets":report["breakout_volume_buckets"]},indent=2))
