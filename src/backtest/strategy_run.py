"""Run the declared sr_baseline_v1 variants on the 2023-2024 development window only."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import fields, replace
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from src.backtest.data import BacktestDataError, load_daily_dataset_through
from src.backtest.engine import BacktestEngine
from src.backtest.models import BacktestConfig, CostModel, OperatingCostModel
from src.configuration import DEFAULT_SETTINGS_PATH, PROJECT_ROOT, load_settings
from src.data.universe.provider import PIT_TRADABLE
from src.strategy.sr_baseline import SRBaselineConfig, SRBaselineStrategy


DEVELOPMENT_START = date(2023, 1, 1)
DEVELOPMENT_END = date(2024, 12, 31)
TARGET_VARIANTS = (1.5, 2.0, 2.5, 3.0)


def _jsonable(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return {item.name: _jsonable(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, (date,)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _load_config(raw: Mapping[str, Any]) -> SRBaselineConfig:
    allowed = {field for field in SRBaselineConfig.__dataclass_fields__}
    unknown = set(raw) - allowed - {"version", "target_r_variants", "weekly_filter_default"}
    if unknown:
        raise ValueError(f"Unknown strategy settings: {', '.join(sorted(unknown))}")
    values = {key: value for key, value in raw.items() if key in allowed}
    if raw.get("version", "sr_baseline_v1") != "sr_baseline_v1":
        raise ValueError("This runner only implements strategy version sr_baseline_v1.")
    values["strategy_version"] = "sr_baseline_v1"
    return SRBaselineConfig(**values)


def _monthly_returns(curve: Sequence[Mapping[str, Any]], starting_equity: float) -> list[dict[str, Any]]:
    month_end: dict[str, float] = {}
    for row in curve:
        month_end[str(row["date"])[:7]] = float(row["equity"])
    result = []
    prior = starting_equity
    for month, equity in sorted(month_end.items()):
        result.append({"month": month, "ending_equity": equity,
                       "return": equity / prior - 1 if prior else 0.0})
        prior = equity
    return result


def _trade_summary(trades: Sequence[Any]) -> dict[str, Any]:
    net = [float(trade.net_pnl) for trade in trades]
    return {"trades": len(trades), "net_pnl": sum(net),
            "win_rate": sum(value > 0 for value in net) / len(net) if net else 0.0,
            "average_R": sum(trade.R_multiple for trade in trades) / len(trades) if trades else 0.0,
            "average_holding_sessions": sum(trade.holding_period for trade in trades) / len(trades) if trades else 0.0,
            "transaction_costs": sum(trade.costs for trade in trades)}


def build_research_report(result: Any, strategy: SRBaselineStrategy) -> dict[str, Any]:
    decisions = [dict(row) for row in strategy.diagnostics]
    events = list(result.events)
    # Reconcile signals that passed setup logic with portfolio/risk/cash rejections at next open.
    accepted = [item for item in decisions if item.get("accepted")]
    for event in events:
        if event.event_type == "SIGNAL_EXPIRED" and event.security_id:
            signal_date = event.timestamp.date().isoformat()
            expired = [item for item in accepted if item["security_id"] == event.security_id
                       and item["signal_date"] == signal_date and item.get("accepted")]
            for decision in expired:
                decision["accepted"] = False
                decision["rejection_reason"] = "NO_NEXT_SESSION_IN_WINDOW"
            continue
        if event.event_type == "POSITION_ENTRY" and event.security_id:
            execution_date = event.timestamp.date().isoformat()
            pending_entry = [item for item in accepted if item["security_id"] == event.security_id
                             and item["signal_date"] < execution_date and item.get("entry_price") is None]
            if pending_entry:
                decision = max(pending_entry, key=lambda item: item["signal_date"])
                fill = float(event.details["fill_price"])
                decision["entry_price"] = fill
                decision["proposed_entry"] = fill
                decision["execution_timestamp"] = event.timestamp.isoformat()
                decision["quantity"] = int(event.details["quantity"])
                decision["risk_distance"] = fill-float(decision["stop"])
                decision["planned_risk_rupees"] = float(event.details["planned_risk"])
            continue
        if event.event_type != "ENTRY_REJECTED" or not event.security_id:
            continue
        event_date = event.timestamp.date().isoformat()
        pending = [item for item in accepted if item["security_id"] == event.security_id
                   and item["signal_date"] < event_date and item["accepted"]]
        if pending:
            decision = max(pending, key=lambda item: item["signal_date"])
            decision["accepted"] = False
            decision["rejection_reason"] = event.details.get("reason", "entry_rejected")
    by_setup: dict[str, list[Any]] = defaultdict(list)
    for trade in result.trades:
        setup_type = trade.metadata.get("setup_type", "unknown")
        by_setup[setup_type].append(trade)
    rejection_counts = Counter(item.get("rejection_reason") for item in decisions if not item.get("accepted"))
    unmatched_engine_rejections = []
    for event in events:
        if event.event_type != "ENTRY_REJECTED" or not event.security_id:
            continue
        reason = event.details.get("reason", "unknown")
        event_date = event.timestamp.date().isoformat()
        matched = any(item.get("security_id") == event.security_id
                      and item.get("signal_date", "") < event_date
                      and item.get("rejection_reason") == reason for item in decisions)
        if not matched:
            unmatched_engine_rejections.append(reason)
    rejection_counts.update(unmatched_engine_rejections)
    setup_counts = Counter(item.get("setup_type", "unknown") for item in decisions)
    exit_counts = Counter(trade.exit_reason for trade in result.trades)
    r_values = [trade.R_multiple for trade in result.trades]
    holding = [trade.holding_period for trade in result.trades]
    largest = lambda reverse: [
        {"security_id": trade.security_id, "symbol": trade.symbol, "setup_type": trade.metadata.get("setup_type"),
         "entry_date": trade.entry_timestamp.date().isoformat(), "exit_date": trade.exit_timestamp.date().isoformat(),
         "net_pnl": trade.net_pnl, "R_multiple": trade.R_multiple, "exit_reason": trade.exit_reason}
        for trade in sorted(result.trades, key=lambda item: item.net_pnl, reverse=reverse)[:10]
    ]
    for setup_name in ("support_test", "breakout_retest"):
        by_setup.setdefault(setup_name, [])
    return {
        "metadata": _jsonable(result.metadata),
        "metrics": dict(result.metrics),
        "setup_performance": {key: _trade_summary(value) for key, value in sorted(by_setup.items())},
        "setup_decision_counts": dict(sorted(setup_counts.items())),
        "rejection_reason_counts": dict(sorted(rejection_counts.items())),
        "exit_reason_counts": dict(sorted(exit_counts.items())),
        "monthly_returns": _monthly_returns(result.equity_curve, result.metadata.starting_capital),
        "R_distribution": r_values,
        "holding_period_distribution_sessions": holding,
        "largest_winners": largest(True), "largest_losers": largest(False),
        "diagnostics": decisions,
        "trades": [_jsonable(trade) for trade in result.trades],
        "equity_curve": [_jsonable(row) for row in result.equity_curve],
        "open_positions_at_end": _jsonable(result.open_positions),
        "events": [_jsonable(event) for event in events],
    }


def run_research(
    *, setup: str, target_r: float, weekly_filter: bool | None, slippage_bps: float,
    cost_scenario: str = "current_zerodha", settings_path: str | Path = DEFAULT_SETTINGS_PATH,
    manifest_path: str | Path = PROJECT_ROOT / "data/processed/market_data_manifest.csv",
    data_root: str | Path = PROJECT_ROOT / "data/processed/market_data",
    output_directory: str | Path = PROJECT_ROOT / "data/processed/backtests/sr_baseline_v1",
) -> tuple[dict[str, Any], Path]:
    if setup not in {"support_test", "breakout_retest", "both"}:
        raise ValueError("Unknown setup selection.")
    if slippage_bps < 0:
        raise ValueError("slippage_bps must be non-negative.")
    if cost_scenario not in {"current_zerodha", "raw_zero"}:
        raise ValueError("cost_scenario must be current_zerodha or raw_zero.")

    app = load_settings(settings_path)
    with Path(settings_path).open("r", encoding="utf-8") as handle:
        raw_settings = yaml.safe_load(handle)
    backtest_settings = raw_settings.get("backtest", {})
    strategy_settings = backtest_settings.get("strategy", {})
    target_variants = tuple(float(value) for value in strategy_settings.get("target_r_variants", TARGET_VARIANTS))
    if target_r not in target_variants:
        raise ValueError(f"target_r must be one of the configured variants: {target_variants}.")
    if weekly_filter is None:
        weekly_filter = bool(strategy_settings.get("weekly_filter_default", False))
    strategy_config = _load_config(strategy_settings)
    transaction_settings = backtest_settings.get("transaction_costs", {})
    if cost_scenario == "raw_zero":
        costs = replace(CostModel.raw_zero(), slippage_bps=slippage_bps,
                        scenario=f"RAW_ZERO_COST_SLIPPAGE_{slippage_bps}BPS")
        operating = OperatingCostModel(0.0, 0.0, 0.0, False)
    else:
        costs = CostModel(
            brokerage_rate=float(transaction_settings.get("brokerage_rate", 0.0)),
            stt_buy_rate=float(transaction_settings.get("stt_buy_rate", 0.001)),
            stt_sell_rate=float(transaction_settings.get("stt_sell_rate", 0.001)),
            nse_transaction_rate=float(transaction_settings.get("nse_transaction_rate", 0.0000307)),
            gst_rate=float(transaction_settings.get("gst_rate", 0.18)),
            sebi_rate=float(transaction_settings.get("sebi_rate", 0.000001)),
            stamp_buy_rate=float(transaction_settings.get("stamp_buy_rate", 0.00015)),
            ipft_rate=float(transaction_settings.get("ipft_rate", 0.000000001)),
            dp_charge_per_sell=float(transaction_settings.get("dp_charge_per_sell", 15.34)),
            slippage_bps=slippage_bps,
            scenario=f"{transaction_settings.get('scenario', 'CURRENT_ZERODHA_EQUITY_DELIVERY_2026')}_SLIPPAGE_{slippage_bps}BPS",
            verification_label=str(transaction_settings.get("verification_label", "CURRENT_ZERODHA_EQUITY_DELIVERY_2026")),
            verified_on=str(transaction_settings.get("verified_on", "")),
            provenance=str(transaction_settings.get("provenance", "")),
        )
        operating_settings = backtest_settings.get("operating_costs", {})
        operating = OperatingCostModel(
            float(operating_settings.get("kite_connect_monthly", 500)),
            float(operating_settings.get("hosting_monthly", 0)),
            float(operating_settings.get("other_monthly", 0)),
            bool(operating_settings.get("apply_to_economic_results", True)),
        )
    daily, dataset_version = load_daily_dataset_through(
        universe_id=PIT_TRADABLE, start_date=DEVELOPMENT_START, end_date=DEVELOPMENT_END,
        data_root=data_root, manifest_path=manifest_path,
    )
    config = BacktestConfig(
        universe_id=PIT_TRADABLE, start_date=DEVELOPMENT_START, end_date=DEVELOPMENT_END,
        initial_capital=app.initial_capital, max_risk_per_trade=app.max_risk_per_trade,
        max_open_positions=int(backtest_settings.get("max_open_positions", 10)),
        timezone_name=app.timezone, costs=costs, operating_costs=operating,
    )
    strategy = SRBaselineStrategy(config=strategy_config, setup=setup, target_r=target_r,
                                  weekly_filter=weekly_filter, evaluation_start=DEVELOPMENT_START)
    result = BacktestEngine(config=config).run(strategy=strategy, daily_data=daily,
                                               dataset_version=dataset_version)
    report = build_research_report(result, strategy)
    report["research_controls"] = {
        "period": [DEVELOPMENT_START.isoformat(), DEVELOPMENT_END.isoformat()],
        "strategy_version": strategy.version,
        "setup": setup, "target_r": target_r, "weekly_filter": weekly_filter,
        "cost_scenario": costs.scenario, "slippage_bps": slippage_bps,
        "outside_period_data_parsed": False,
    }
    out_dir = Path(output_directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{setup}_r{target_r:g}_weekly-{int(weekly_filter)}_{cost_scenario}_{slippage_bps:g}bps.json"
    path = out_dir / name
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_jsonable(report), indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    temporary.replace(path)
    return report, path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setup", choices=("support_test", "breakout_retest", "both"), required=True)
    parser.add_argument("--target-r", type=float, default=2.0)
    parser.add_argument("--weekly-filter", choices=("off", "on"))
    parser.add_argument("--slippage-bps", type=float, default=0.0)
    parser.add_argument("--cost-scenario", choices=("current_zerodha", "raw_zero"), default="current_zerodha")
    parser.add_argument("--settings", type=Path, default=DEFAULT_SETTINGS_PATH)
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "data/processed/market_data_manifest.csv")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data/processed/market_data")
    parser.add_argument("--output-directory", type=Path, default=PROJECT_ROOT / "data/processed/backtests/sr_baseline_v1")
    args = parser.parse_args()
    try:
        report, path = run_research(
            setup=args.setup, target_r=args.target_r,
            weekly_filter=(args.weekly_filter == "on") if args.weekly_filter is not None else None,
            slippage_bps=args.slippage_bps, cost_scenario=args.cost_scenario,
            settings_path=args.settings, manifest_path=args.manifest,
            data_root=args.data_root, output_directory=args.output_directory,
        )
    except (ValueError, OSError, BacktestDataError) as error:
        parser.error(str(error))
    print(json.dumps({"result_file": str(path), "period": report["research_controls"]["period"],
                      "strategy": "sr_baseline_v1", "setup": args.setup, "target_r": args.target_r,
                      "weekly_filter": report["research_controls"]["weekly_filter"], "slippage_bps": args.slippage_bps,
                      "cost_scenario": args.cost_scenario, "metrics": report["metrics"],
                      "setup_performance": report["setup_performance"]}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
