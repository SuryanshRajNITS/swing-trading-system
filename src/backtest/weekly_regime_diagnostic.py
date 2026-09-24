"""Classify saved 2023-2024 weekly-selection trades by causal NIFTY regime."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import date
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from src.backtest.sr_zone_v2_run import END
from src.backtest.weekly_selection_run import load_nifty_benchmark
from src.configuration import DEFAULT_SETTINGS_PATH, PROJECT_ROOT
from src.strategy.weekly_regime import WeeklyRegimeConfig, weekly_regime


DEFAULT_BACKTEST=PROJECT_ROOT/"data/processed/backtests/weekly_selection/sr_zone_v2_weekly_selection_r1_2023_2024.json"
DEFAULT_BENCHMARK=PROJECT_ROOT/"data/raw/reference/nifty50_daily.csv"
DEFAULT_OUTPUT=PROJECT_ROOT/"data/processed/backtests/weekly_selection/weekly_regime_diagnostic_2023_2024.json"


def _aggregate(trades: Sequence[Mapping[str,Any]]) -> dict[str,Any]:
    count=len(trades)
    net=[float(trade["net_pnl"]) for trade in trades]
    gross=[float(trade["gross_pnl"]) for trade in trades]
    gains=sum(value for value in net if value>0)
    losses=-sum(value for value in net if value<0)
    return {
        "trade_count":count,
        "win_rate":sum(value>0 for value in net)/count if count else 0.0,
        "gross_expectancy":sum(gross)/count if count else 0.0,
        "net_expectancy":sum(net)/count if count else 0.0,
        "profit_factor":gains/losses if losses else (None if gains else 0.0),
        "stop_rate":sum(trade.get("exit_reason")=="STOP" for trade in trades)/count if count else 0.0,
        "target_rate":sum(trade.get("exit_reason")=="TARGET" for trade in trades)/count if count else 0.0,
    }


def classify_saved_trades(*, backtest_path: str|Path=DEFAULT_BACKTEST,
                          benchmark_path: str|Path=DEFAULT_BENCHMARK,
                          settings_path: str|Path=DEFAULT_SETTINGS_PATH) -> dict[str,Any]:
    """Read existing artifacts only; this function never starts a backtest."""
    artifact_path=Path(backtest_path)
    if not artifact_path.is_file():
        raise ValueError(f"Saved weekly-selection backtest not found: {artifact_path}")
    report=json.loads(artifact_path.read_text(encoding="utf-8"))
    controls=report.get("research_controls",{})
    period=controls.get("period",[])
    if len(period)!=2 or date.fromisoformat(period[1])>END:
        raise ValueError("Diagnostic accepts only saved backtests ending no later than 2024-12-31.")
    trades=report.get("trades")
    if not isinstance(trades,list):
        raise ValueError("Saved report does not contain a trade ledger.")
    trade_years={str(trade.get("signal_timestamp",""))[:4] for trade in trades}
    if trade_years-{"2023","2024"}:
        raise ValueError("Diagnostic accepts trades from 2023 and 2024 only.")

    settings=yaml.safe_load(Path(settings_path).read_text(encoding="utf-8")) or {}
    config=WeeklyRegimeConfig(**settings.get("backtest",{}).get("weekly_regime",{}))
    nifty=load_nifty_benchmark(benchmark_path)
    regime_history=[weekly_regime(nifty_daily=nifty,as_of=date.fromisoformat(row["date"]),config=config)
                    for row in nifty]
    by_date={row["as_of"]:row for row in regime_history}

    classified=[]
    for trade in trades:
        signal_time=str(trade.get("signal_timestamp",""))
        signal_day=signal_time[:10]
        regime=by_date.get(signal_day)
        if regime is None:
            eligible_dates=[day for day in by_date if day<=signal_day]
            if not eligible_dates:
                raise ValueError(f"No NIFTY data on or before signal date {signal_day}.")
            regime=by_date[max(eligible_dates)]
        classified.append({
            "trade_id":trade.get("trade_id"),"symbol":trade.get("symbol"),
            "security_id":trade.get("security_id"),"setup_type":trade.get("metadata",{}).get("setup_type"),
            "signal_timestamp":signal_time,"entry_timestamp":trade.get("entry_timestamp"),
            "exit_timestamp":trade.get("exit_timestamp"),"exit_reason":trade.get("exit_reason"),
            "gross_pnl":trade.get("gross_pnl"),"net_pnl":trade.get("net_pnl"),
            "regime":regime["regime"],"reason_codes":regime["reason_codes"],
            "component_values":regime["component_values"],
            "regime_as_of":regime["as_of"],"last_completed_week":regime["last_completed_week"],
        })

    states=("BULLISH","NEUTRAL","WEAK")
    metrics={}
    for state in states:
        subset=[trade for trade in classified if trade["regime"]==state]
        metrics[state]={"all_years":_aggregate(subset),
                        "reason_code_counts":dict(sorted(Counter(
                            code for trade in subset for code in trade["reason_codes"]).items())),
                        "by_year":{
            year:_aggregate([trade for trade in subset if trade["signal_timestamp"][:4]==year])
            for year in ("2023","2024")}}
    benchmark_payload=json.dumps(nifty,sort_keys=True,separators=(",",":"),allow_nan=False)
    return {
        "diagnostic":"weekly_nifty_regime_at_signal_time",
        "backtest_artifact":str(artifact_path),"benchmark_file":str(benchmark_path),
        "benchmark_sha256":hashlib.sha256(benchmark_payload.encode("utf-8")).hexdigest(),
        "evaluation_period":period,"regime_config":asdict(config),
        "weekly_regime_history":regime_history,
        "metrics_by_regime":metrics,"trade_classifications":classified,
        "notes":[
            "Regime diagnostics only; trades were not filtered or changed.",
            "Each as-of date excludes the full calendar week containing that date.",
            "Per-year grouping uses signal date. Profit factor uses net trade P&L.",
            "Only completed trades in the saved ledger are summarized.",
        ],
    }


def write_report(report: Mapping[str,Any], output_path: str|Path=DEFAULT_OUTPUT) -> Path:
    path=Path(output_path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False),encoding="utf-8")
    temp.replace(path)
    return path


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backtest",type=Path,default=DEFAULT_BACKTEST,
                        help="Existing saved weekly-selection backtest JSON; no backtest is run.")
    parser.add_argument("--benchmark",type=Path,default=DEFAULT_BENCHMARK,
                        help="Daily NIFTY 50 historical index CSV.")
    parser.add_argument("--settings",type=Path,default=DEFAULT_SETTINGS_PATH)
    parser.add_argument("--output",type=Path,default=DEFAULT_OUTPUT)
    args=parser.parse_args()
    result=classify_saved_trades(backtest_path=args.backtest,benchmark_path=args.benchmark,
                                 settings_path=args.settings)
    output=write_report(result,args.output)
    print(json.dumps({"output":str(output),"trades":len(result["trade_classifications"]),
                      "regime_counts":{name:row["all_years"]["trade_count"]
                                       for name,row in result["metrics_by_regime"].items()},
                      "regime_dates":len(result["weekly_regime_history"])},indent=2))


if __name__=="__main__":
    main()
