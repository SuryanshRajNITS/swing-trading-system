"""Run the sr_zone_v2 plus weekly stock-selection experiment on 2023-2024 only."""

from __future__ import annotations

import argparse
import csv
from datetime import date
import json
from pathlib import Path
from typing import Any

import yaml

from src.backtest.data import BacktestDataError, load_daily_dataset_through
from src.backtest.engine import BacktestEngine
from src.backtest.models import BacktestConfig, CostModel, OperatingCostModel
from src.backtest.strategy_run import _jsonable, build_research_report
from src.backtest.sr_zone_v2_run import END, START
from src.configuration import DEFAULT_SETTINGS_PATH, PROJECT_ROOT, load_settings
from src.data.universe.provider import PIT_TRADABLE
from src.strategy.sr_baseline import SRBaselineConfig
from src.strategy.sr_zone_v2 import SRZoneV2Strategy, ZoneConfig
from src.strategy.weekly_selection import WeeklySelectionConfig, WeeklySelectionStrategy


def load_nifty_benchmark(path: str|Path, *, start: date= date(2022,7,1), end: date=END) -> list[dict[str,Any]]:
    """Read caller-supplied official NSE Indices CSV; reject missing or malformed data."""
    source=Path(path)
    if not source.is_file():
        raise ValueError(f"Required official NIFTY 50 benchmark CSV is missing: {source}")
    with source.open("r",encoding="utf-8-sig",newline="") as handle:
        reader=csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("NIFTY benchmark CSV has no header row.")
        normalized={name.strip().casefold().replace(" ","_"):name for name in reader.fieldnames}
        date_key=next((normalized[key] for key in ("date","index_date","index_date_") if key in normalized),None)
        close_key=next((normalized[key] for key in ("close","closing_index_value","close_index_value") if key in normalized),None)
        if date_key is None or close_key is None:
            raise ValueError("NIFTY CSV must include a date field and closing index value field.")
        def get_key(*names):
            return next((normalized[name] for name in names if name in normalized),None)
        open_key,high_key,low_key=get_key("open","open_index_value"),get_key("high","high_index_value"),get_key("low","low_index_value")
        rows=[]
        for raw in reader:
            raw_day=str(raw.get(date_key," ")).strip()
            if not raw_day:
                continue
            try:
                day=date.fromisoformat(raw_day[:10])
            except ValueError:
                from datetime import datetime
                try: day=datetime.strptime(raw_day,"%d-%b-%Y").date()
                except ValueError: raise ValueError(f"Invalid date in official NIFTY CSV: {raw_day!r}.") from None
            if day>end:
                raise ValueError("NIFTY benchmark file extends past 2024-12-31; provide the bounded 2022-2024 slice.")
            if day<start:
                continue
            close=float(str(raw[close_key]).replace(",",""))
            op=float(str(raw[open_key]).replace(",","")) if open_key and raw.get(open_key) else close
            hi=float(str(raw[high_key]).replace(",","")) if high_key and raw.get(high_key) else max(op,close)
            lo=float(str(raw[low_key]).replace(",","")) if low_key and raw.get(low_key) else min(op,close)
            rows.append({"date":day.isoformat(),"open":op,"high":hi,"low":lo,"close":close,"volume":0})
    rows.sort(key=lambda row:row["date"])
    if len(rows)<100:
        raise ValueError("NIFTY benchmark CSV must provide daily warmup and evaluation coverage (at least 100 rows).")
    if len({row["date"] for row in rows})!=len(rows):
        raise ValueError("NIFTY benchmark CSV contains duplicate dates.")
    # The provided annual files begin on the first 2023 trading day. Early
    # evaluation dates therefore remain neutral until the configured 26-week
    # minimum history has accumulated; no benchmark returns are fabricated.
    if rows[0]["date"]>date(2023,1,3).isoformat() or rows[-1]["date"]<end.replace(day=24).isoformat():
        raise ValueError("NIFTY benchmark must span 2023-01-03 through the final 2024 evaluation sessions.")
    return rows


def run_research(*, benchmark_path: str|Path|None=None,
                 settings_path: str|Path=DEFAULT_SETTINGS_PATH,
                 manifest_path: str|Path=PROJECT_ROOT/"data/processed/market_data_manifest.csv",
                 data_root: str|Path=PROJECT_ROOT/"data/processed/market_data",
                 output_directory: str|Path=PROJECT_ROOT/"data/processed/backtests/weekly_selection") -> tuple[dict[str,Any],Path]:
    app=load_settings(settings_path)
    raw=yaml.safe_load(Path(settings_path).read_text(encoding="utf-8"))
    bt=raw.get("backtest",{})
    benchmark_settings=bt.get("weekly_selection",{})
    configured_path=benchmark_path or benchmark_settings.get("benchmark_path")
    if not configured_path:
        raise ValueError("Set backtest.weekly_selection.benchmark_path or pass --benchmark.")
    benchmark=load_nifty_benchmark(configured_path)
    base_raw=bt.get("strategy",{})
    allowed=set(SRBaselineConfig.__dataclass_fields__)
    daily_config=SRBaselineConfig(**{key:value for key,value in base_raw.items() if key in allowed})
    zone_raw=bt.get("strategy_zone_v2",{})
    zone_allowed=set(ZoneConfig.__dataclass_fields__)
    zone_values={key:value for key,value in zone_raw.items() if key in zone_allowed}
    zone_values.setdefault("swing_left",daily_config.swing_left)
    zone_values.setdefault("swing_right",daily_config.swing_right)
    zone_values.setdefault("minimum_pivots",daily_config.minimum_zone_pivots)
    zone_values.setdefault("maximum_age",daily_config.maximum_zone_age_sessions)
    zone_config=ZoneConfig(**zone_values)
    selection_allowed=set(WeeklySelectionConfig.__dataclass_fields__)
    selection_raw=benchmark_settings
    unknown=set(selection_raw)-selection_allowed-{"version","benchmark_path"}
    if unknown: raise ValueError(f"Unknown weekly-selection settings: {', '.join(sorted(unknown))}.")
    selection_config=WeeklySelectionConfig(**{k:v for k,v in selection_raw.items() if k in selection_allowed})
    costs_raw=bt.get("transaction_costs",{})
    costs=CostModel(brokerage_rate=float(costs_raw.get("brokerage_rate",0)),
                    stt_buy_rate=float(costs_raw.get("stt_buy_rate",.001)),
                    stt_sell_rate=float(costs_raw.get("stt_sell_rate",.001)),
                    nse_transaction_rate=float(costs_raw.get("nse_transaction_rate",.0000307)),
                    gst_rate=float(costs_raw.get("gst_rate",.18)),sebi_rate=float(costs_raw.get("sebi_rate",.000001)),
                    stamp_buy_rate=float(costs_raw.get("stamp_buy_rate",.00015)),
                    ipft_rate=float(costs_raw.get("ipft_rate",.000000001)),
                    dp_charge_per_sell=float(costs_raw.get("dp_charge_per_sell",15.34)),slippage_bps=0,
                    scenario=str(costs_raw.get("scenario","CURRENT_ZERODHA_EQUITY_DELIVERY_2026"))+"_SLIPPAGE_0BPS",
                    verification_label=str(costs_raw.get("verification_label","CURRENT_ZERODHA_EQUITY_DELIVERY_2026")),
                    verified_on=str(costs_raw.get("verified_on","")),provenance=str(costs_raw.get("provenance","")))
    op_raw=bt.get("operating_costs",{})
    operating=OperatingCostModel(float(op_raw.get("kite_connect_monthly",500)),float(op_raw.get("hosting_monthly",0)),
                                 float(op_raw.get("other_monthly",0)),bool(op_raw.get("apply_to_economic_results",True)))
    daily,dataset_version=load_daily_dataset_through(universe_id=PIT_TRADABLE,start_date=START,end_date=END,
                                                       data_root=data_root,manifest_path=manifest_path)
    config=BacktestConfig(PIT_TRADABLE,START,END,initial_capital=app.initial_capital,
                          max_risk_per_trade=app.max_risk_per_trade,max_open_positions=int(bt.get("max_open_positions",10)),
                          timezone_name=app.timezone,costs=costs,operating_costs=operating)
    daily_strategy=SRZoneV2Strategy(config=daily_config,setup="both",target_r=2.0,weekly_filter=False,
                                    evaluation_start=START,zone_config=zone_config)
    selection=WeeklySelectionStrategy(daily_strategy=daily_strategy,benchmark_daily=benchmark,config=selection_config)
    result=BacktestEngine(config=config).run(strategy=selection,daily_data=daily,dataset_version=dataset_version)
    report=build_research_report(result,selection)
    report["research_controls"]={"period":[START.isoformat(),END.isoformat()],"strategy_version":selection.version,
                                 "daily_strategy":"sr_zone_v2","setup":"both","target_r":2.0,
                                 "simple_weekly_filter":False,"cost_scenario":costs.scenario,"slippage_bps":0,
                                 "outside_period_data_parsed":False,"benchmark_file":str(configured_path),
                                 "benchmark_source":"User-provided official NIFTY 50 historical PR CSV files, combined for 2023 and 2024",
                                 "benchmark_source_files":[
                                     "NIFTY 50_Historical_PR_01012023to31122023.csv",
                                     "NIFTY 50_Historical_PR_01012024to31122024.csv"],
                                 "benchmark_sha256":selection.parameters["benchmark_sha256"],
                                 "benchmark_history_starts":"2023-01-02; weekly eligibility remains neutral until minimum history is available",
                                 "benchmark_source_required":"Official NSE Indices NIFTY 50 daily index CSV"}
    report["selection_decisions"]=selection.decisions
    report["selection_signal_decisions"]=selection.signal_decisions
    report["weekly_selection_snapshots"]=selection.weekly_snapshots
    output=Path(output_directory); output.mkdir(parents=True,exist_ok=True)
    path=output/"sr_zone_v2_weekly_selection_r1_2023_2024.json"
    temp=path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(_jsonable(report),indent=2,sort_keys=True,allow_nan=False),encoding="utf-8")
    temp.replace(path)
    return report,path


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark",type=Path,help="Official bounded NIFTY 50 daily index CSV.")
    parser.add_argument("--settings",type=Path,default=DEFAULT_SETTINGS_PATH)
    parser.add_argument("--manifest",type=Path,default=PROJECT_ROOT/"data/processed/market_data_manifest.csv")
    parser.add_argument("--data-root",type=Path,default=PROJECT_ROOT/"data/processed/market_data")
    parser.add_argument("--output-directory",type=Path,default=PROJECT_ROOT/"data/processed/backtests/weekly_selection")
    args=parser.parse_args()
    try:
        report,path=run_research(benchmark_path=args.benchmark,settings_path=args.settings,
                                 manifest_path=args.manifest,data_root=args.data_root,output_directory=args.output_directory)
    except (ValueError,OSError,BacktestDataError) as error: parser.error(str(error))
    print(json.dumps({"artifact":str(path),"period":report["research_controls"]["period"],
                      "trades":report["metrics"]["number_of_trades"],
                      "weekly_snapshots":len(report["weekly_selection_snapshots"])},indent=2))


if __name__=="__main__": main()
