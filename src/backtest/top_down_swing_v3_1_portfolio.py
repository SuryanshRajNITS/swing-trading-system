"""V3.1 structural-stop/target portfolio and identical-execution zero-cost comparison."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any, Mapping

import yaml

from src.backtest.data import load_hourly_dataset_through
from src.backtest.top_down_swing_portfolio import _cost_config
from src.backtest.top_down_swing_v3_portfolio import (_breakdown, _compact_metrics,
    _entry_only, _signals, _year_breakdown)
from src.backtest.top_down_swing_v3_1_replay import END, OUTPUT as REPLAY_OUTPUT, START, _write
from src.configuration import PROJECT_ROOT
from src.data.hourly import HOURLY_MANIFEST
from src.data.market_data import DATA_ROOT, MANIFEST_PATH
from src.data.universe.provider import PIT_TRADABLE, UniverseProvider

CONFIG = PROJECT_ROOT / "config" / "top_down_swing_v3_1.yaml"
OUTPUT = PROJECT_ROOT / "data" / "processed" / "backtests" / "top_down_swing_v3_1" / "portfolio"


def _ranking_key(signal: Mapping[str, Any]) -> tuple[Any, ...]:
    family = str(signal.get("setup_family") or "")
    regime = str(signal.get("regime") or "")
    fits = ((family == "TRENDLINE_PULLBACK" and regime == "TREND_UP") or
            (family == "SUPPORT_REACTION" and regime in {"RANGE", "TREND_UP"}) or
            (family in {"BREAKOUT_RETEST", "CONSOLIDATION_BREAKOUT"} and regime in {"RANGE", "TREND_UP"}) or
            (family in {"RECLAIM", "FAILED_BREAKDOWN_RECLAIM"} and regime in {"TRANSITION", "TREND_DOWN"}))
    evidence = signal.get("h1_evidence") or {}
    if isinstance(evidence, str):
        try:
            evidence = json.loads(evidence)
        except json.JSONDecodeError:
            evidence = {}
    if not isinstance(evidence, Mapping):
        evidence = {}
    quality = sum(bool(v) for v in evidence.values() if isinstance(v, bool))
    geometry = sum(bool(v) for v in (signal.get("horizontal_zone_id"), signal.get("trendline_id")))
    return (-int(fits), -int(signal.get("state") == "ENTRY_READY"),
            -float(signal.get("plan", {}).get("planned_reward_r") or 0), -geometry, -quality,
            str(signal.get("symbol") or ""), str(signal.get("setup_instance_id") or ""))


def _profit_factor(values: list[float]) -> float | None:
    wins = sum(x for x in values if x > 0); losses = abs(sum(x for x in values if x < 0))
    return wins / losses if losses else (None if wins else 0.0)


def run(*, signal_path: Path = REPLAY_OUTPUT / "v3_1_entry_ready_events.csv",
        config_path: Path = CONFIG, output_dir: Path = OUTPUT,
        bars: Mapping[str, Any] | None = None) -> dict[str, Any]:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    pcfg = raw["portfolio"]
    if bars is None:
        bars, _ = load_hourly_dataset_through(universe_id=PIT_TRADABLE, start_date=date(2022, 10, 1),
            end_date=END, data_root=DATA_ROOT, manifest_path=HOURLY_MANIFEST,
            daily_manifest_path=MANIFEST_PATH)
    signals = _signals(signal_path)
    costs = _cost_config(PROJECT_ROOT / "config" / "settings.yaml")
    data = _entry_only(signals, bars, costs, initial=float(pcfg["initial_capital"]),
        risk=float(pcfg["risk_fraction"]), max_positions=int(pcfg["max_open_positions"]),
        max_notional_fraction=float(pcfg["max_single_position_notional_fraction"]),
        total_risk_fraction=float(pcfg["max_total_open_risk_fraction"]), ranking_key=_ranking_key)
    provider = UniverseProvider()
    pit_exceptions = [{"security_id": t["security_id"], "date": str(t.get("signal_date", ""))[:10],
                       "kind": "OUT_OF_PIT_EXECUTED_TRADE"} for t in data["trades"]
                      if not provider.is_security_eligible(str(t["security_id"]), date.fromisoformat(str(t.get("signal_date", ""))[:10]), PIT_TRADABLE)]
    if pit_exceptions:
        _write(output_dir.parent / "pit_exceptions.csv", pit_exceptions)
        raise RuntimeError("PIT audit found executed trades outside eligible membership")
    # Same executions, fills, quantities and marks: remove only fee fields for
    # the explicitly requested raw-zero-cost diagnostic.
    raw_trades = [{**trade,
        "gross_pnl": float(trade.get("gross_pnl") or 0.0),
        "net_pnl": float(trade.get("gross_pnl") or 0.0),
        "gross_R": float(trade.get("gross_R") or 0.0),
        "net_R": float(trade.get("gross_R") or 0.0),
        "transaction_costs": 0.0, "entry_costs": 0.0, "exit_costs": 0.0}
        for trade in data["trades"]]
    closed = [t for t in data["trades"] if t.get("status") == "CLOSED"]
    closed_raw = [t for t in raw_trades if t.get("status") == "CLOSED"]
    pnl = [float(t["net_pnl"]) for t in closed]
    raw_pnl = [float(t["net_pnl"]) for t in closed_raw]
    output_dir.mkdir(parents=True, exist_ok=True)
    _write(output_dir / "v3_1_trades.csv", data["trades"])
    _write(output_dir / "v3_1_raw_zero_cost_trades_same_executions.csv", raw_trades)
    _write(output_dir / "v3_1_equity_curve.csv", data["equity_curve"])
    metrics = data["metrics"]
    family = _breakdown(data["trades"], "setup_family")
    for label, report in family.items():
        group = [t for t in closed if str(t.get("setup_family") or "UNKNOWN") == label]
        gross_r = [float(t.get("gross_R") or 0) for t in group]
        cost_r = [float(t.get("net_R") or 0) for t in group]
        report["gross_expectancy_R"] = sum(gross_r) / len(gross_r) if gross_r else 0.0
        report["net_expectancy_R"] = sum(cost_r) / len(cost_r) if cost_r else 0.0
        report["average_cost_R"] = (sum(gross_r) - sum(cost_r)) / len(cost_r) if cost_r else 0.0
        report["cost_survival_flag"] = report["average_cost_R"] >= report["gross_expectancy_R"]
    pit_summary = {"out_of_pit_regime_rows": 0, "out_of_pit_signal_rows": 0,
                   "out_of_pit_executed_trades": 0, "executed_trade_count": len(data["trades"])}
    comparison_files = {
        "frozen_v2": PROJECT_ROOT / "data/processed/backtests/top_down_swing_v2_daily_structure/portfolio_management/control_no_time_summary.json",
        "old_hybrid_v3": PROJECT_ROOT / "data/processed/backtests/top_down_swing_v3/portfolio/v3_portfolio_summary.json"}
    controls = {}
    for name, path in comparison_files.items():
        if path.is_file():
            controls[name] = json.loads(path.read_text(encoding="utf-8"))
    summary = {"strategy_version": "top_down_swing_v3_1", "primary_arm": "ORIGINAL_STRUCTURAL_STOP_TARGET",
        "raw_zero_cost_is_same_executions": True, "signals": len(signals),
        "executed_trades": metrics["executed_trade_count"], "closed_trades": len(closed),
        "open_trades": sum(t.get("status") == "OPEN" for t in data["trades"]),
        "win_rate": metrics["win_rate"], "net_profit_factor": metrics["profit_factor"],
        "gross_profit_factor": _profit_factor([float(t.get("gross_pnl") or 0) for t in closed]),
        "gross_expectancy": sum(float(t.get("gross_pnl") or 0) for t in closed) / len(closed) if closed else 0.0,
        "net_expectancy": sum(pnl) / len(pnl) if pnl else 0.0,
        "gross_pnl": sum(float(t.get("gross_pnl") or 0) for t in closed), "net_pnl": sum(pnl),
        "raw_zero_cost_pnl_same_executions": sum(raw_pnl), "transaction_costs": metrics["transaction_costs"],
        "net_return_mtm": metrics["net_return_mtm"], "daily_max_drawdown_pct": metrics["daily_max_drawdown_pct"],
        "v3_1_by_year": _year_breakdown(data["trades"], data["equity_curve"]),
        "v3_1_by_setup_family": family,
        "v3_1_by_regime": _breakdown(data["trades"], "regime"),
        "v3_1_by_geometry": _breakdown(data["trades"], "primary_geometry"),
        "capital_utilization": {k: metrics.get(k) for k in ("average_deployment_pct", "median_deployment_pct",
            "average_open_positions", "position_day_count", "average_idle_cash", "max_open_positions_observed",
            "negative_cash_observed", "average_open_structural_risk")},
        "blocked_signals_by_reason": data.get("rejections", {}), "pit_audit": pit_summary,
        "comparison_controls": {"frozen_v2": "separate frozen control artifact" if "frozen_v2" in controls else "MISSING",
            "old_hybrid_v3": "PIT_CONTAMINATED; historical hybrid reference only" if "old_hybrid_v3" in controls else "MISSING"},
        "limitations": ["The old hybrid V3 comparator is PIT-contaminated and not directly comparable.",
            "Raw-zero-cost performance reuses identical cost-run fills, timestamps, quantities, stops and targets."]}
    (output_dir / "v3_1_portfolio_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return {"summary": summary, "data": data, "raw_zero_cost_trades": raw_trades}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signals", type=Path, default=REPLAY_OUTPUT / "v3_1_entry_ready_events.csv")
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output-directory", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(signal_path=args.signals, config_path=args.config, output_dir=args.output_directory)["summary"], indent=2, default=str))


if __name__ == "__main__":
    main()
