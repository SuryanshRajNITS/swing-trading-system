"""Build an A/B/C comparison and weekly-feature outcome summary."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import fmean, median
from typing import Any, Mapping, Sequence

from src.configuration import PROJECT_ROOT


FEATURES = (
    "structure_score", "momentum_score", "relative_strength_score",
    "relative_strength_4w", "relative_strength_13w", "resistance_distance_atr",
    "support_distance_atr", "trend_efficiency_13w", "weekly_volatility_13w",
    "strong_bullish_close_fraction_8w", "directional_move_4w_atr",
    "weekly_slope_pct_per_week",
)
SETUPS = ("support_test", "breakout_retest")


def _trade_metrics(trades: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    net=[float(t["net_pnl"]) for t in trades]
    gross=[float(t["gross_pnl"]) for t in trades]
    positives=sum(value for value in net if value>0)
    negatives=-sum(value for value in net if value<0)
    count=len(trades)
    exits=[str(t.get("exit_reason","")) for t in trades]
    turnover=sum(float(t["quantity"])*(float(t["entry_price"])+float(t["exit_price"])) for t in trades)
    return {
        "trades":count,
        "gross_expectancy":fmean(gross) if count else 0.0,
        "net_expectancy":fmean(net) if count else 0.0,
        "profit_factor":positives/negatives if negatives else (None if positives else 0.0),
        "win_rate":sum(value>0 for value in net)/count if count else 0.0,
        "stop_rate":sum(reason=="STOP" for reason in exits)/count if count else 0.0,
        "target_rate":sum(reason=="TARGET" for reason in exits)/count if count else 0.0,
        "transaction_costs":sum(float(t.get("costs",0)) for t in trades),
        "turnover":turnover,
        "net_pnl":sum(net),
    }


def _features_by_outcome(trades: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    winners=[]
    losers=[]
    for trade in trades:
        features=trade.get("metadata",{}).get("weekly_selection")
        if not isinstance(features,Mapping):
            continue
        (winners if float(trade["net_pnl"])>0 else losers).append(features)
    result={}
    for feature in FEATURES:
        values_w=[float(row[feature]) for row in winners if row.get(feature) is not None]
        values_l=[float(row[feature]) for row in losers if row.get(feature) is not None]
        if not values_w and not values_l:
            continue
        result[feature]={
            "winner_count":len(values_w),"winner_mean":fmean(values_w) if values_w else None,
            "winner_median":median(values_w) if values_w else None,
            "loser_count":len(values_l),"loser_mean":fmean(values_l) if values_l else None,
            "loser_median":median(values_l) if values_l else None,
            "mean_difference_winner_minus_loser":(fmean(values_w)-fmean(values_l)
                                                    if values_w and values_l else None),
        }
    return result


def build_comparison(*, baseline_daily_path: str|Path, baseline_simple_weekly_path: str|Path,
                     selection_path: str|Path) -> dict[str, Any]:
    paths={"A_daily_only":Path(baseline_daily_path),"B_simple_weekly_filter":Path(baseline_simple_weekly_path),
           "C_weekly_selection":Path(selection_path)}
    reports={name:json.loads(path.read_text(encoding="utf-8")) for name,path in paths.items()}
    comparison={}
    for name,report in reports.items():
        trades=report.get("trades",[])
        metrics=dict(report["metrics"])
        rows_by_setup={setup:[trade for trade in trades
                              if trade.get("metadata",{}).get("setup_type")==setup] for setup in SETUPS}
        comparison[name]={
            "artifact":str(paths[name]),
            "period":report.get("research_controls",{}).get("period"),
            "metrics":metrics,
            "average_trades_per_month":metrics.get("number_of_trades",0)/24,
            "turnover":_trade_metrics(trades)["turnover"],
            "setup_performance":{"combined":_trade_metrics(trades),
                                 **{setup:_trade_metrics(rows) for setup,rows in rows_by_setup.items()}},
        }
        # The engine includes fees on open positions in transaction_costs_total;
        # the closed-trade ledger intentionally does not invent exits for them.
        combined=comparison[name]["setup_performance"]["combined"]
        combined["transaction_costs"]=metrics.get("transaction_costs_total",combined["transaction_costs"])
        combined["turnover"]+=sum(float(row["quantity"])*float(row["entry_price"])
                                  for row in report.get("open_positions_at_end",[]))
        if name=="C_weekly_selection":
            snapshots=report.get("weekly_selection_snapshots",[])
            fractions=[float(row["retained_fraction"]) for row in snapshots]
            comparison[name]["selection"]={
                "weekly_snapshots":len(snapshots),
                "mean_retained_fraction":fmean(fractions) if fractions else 0.0,
                "median_retained_fraction":median(fractions) if fractions else 0.0,
                "mean_pit_universe_count":fmean([int(row["pit_universe_count"]) for row in snapshots]) if snapshots else 0.0,
                "mean_eligible_count":fmean([int(row["eligible_count"]) for row in snapshots]) if snapshots else 0.0,
            }
            comparison[name]["weekly_feature_outcomes"]=_features_by_outcome(trades)
    return {"comparison":comparison,
            "interpretation":"Winner/loser feature differences are descriptive associations, not causal evidence.",
            "scope":"2023-01-01 through 2024-12-31; combined support and breakout-retest; 2R; current delivery cost assumptions; 0 bps slippage."}


def write_comparison(report: Mapping[str,Any], output_dir: str|Path) -> tuple[Path,Path]:
    output=Path(output_dir); output.mkdir(parents=True,exist_ok=True)
    json_path=output/"sr_zone_v2_weekly_selection_comparison_2023_2024.json"
    csv_path=output/"sr_zone_v2_weekly_selection_comparison_2023_2024.csv"
    json_path.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False),encoding="utf-8")
    with csv_path.open("w",encoding="utf-8",newline="") as handle:
        writer=csv.writer(handle)
        writer.writerow(("arm","setup","trades","gross_expectancy","net_expectancy","profit_factor",
                         "win_rate","stop_rate","target_rate","transaction_costs","turnover",
                         "trading_net_return","economic_net_return","average_trades_per_month"))
        for arm,values in report["comparison"].items():
            metrics=values["metrics"]
            for setup,stats in values["setup_performance"].items():
                writer.writerow((arm,setup,stats["trades"],stats["gross_expectancy"],stats["net_expectancy"],
                    stats["profit_factor"],stats["win_rate"],stats["stop_rate"],stats["target_rate"],
                    stats["transaction_costs"],stats["turnover"],metrics.get("trading_net_return"),
                    metrics.get("economic_net_return"),values["average_trades_per_month"]))
    return json_path,csv_path


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    default_dir=PROJECT_ROOT/"data/processed/backtests"
    parser.add_argument("--baseline-dir",type=Path,default=default_dir/"sr_zone_v2")
    parser.add_argument("--selection",type=Path,default=default_dir/"weekly_selection"/"sr_zone_v2_weekly_selection_r1_2023_2024.json")
    parser.add_argument("--output",type=Path,default=default_dir/"weekly_selection")
    args=parser.parse_args()
    report=build_comparison(
        baseline_daily_path=args.baseline_dir/"both_r2_weekly-0_current_zerodha_0bps.json",
        baseline_simple_weekly_path=args.baseline_dir/"both_r2_weekly-1_current_zerodha_0bps.json",
        selection_path=args.selection)
    paths=write_comparison(report,args.output)
    print(json.dumps({"reports":[str(path) for path in paths],
                      "arms":list(report["comparison"])},indent=2))


if __name__=="__main__":
    main()
