"""Run the four registered weekly-selection exit ablation arms."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
from datetime import date
import json
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

import yaml

from src.backtest.data import BacktestDataError, load_daily_dataset_through
from src.backtest.engine import BacktestEngine, BacktestResult
from src.backtest.models import BacktestConfig, CostModel, OperatingCostModel
from src.backtest.sr_zone_v2_run import END, START
from src.backtest.weekly_selection_run import load_nifty_benchmark
from src.configuration import DEFAULT_SETTINGS_PATH, PROJECT_ROOT, load_settings
from src.data.universe.provider import PIT_TRADABLE
from src.decision.interfaces import EntrySignal, ExitSignal
from src.strategy.sr_baseline import SRBaselineConfig
from src.strategy.sr_zone_v2 import SRZoneV2Strategy, ZoneConfig
from src.strategy.structural_exit import (StructuralExitConfig,
                                           active_resistance_zones_asof,
                                           build_structural_exit_plan, daily_atr)
from src.strategy.weekly_selection import WeeklySelectionConfig, WeeklySelectionStrategy


ARMS = ("A_BASELINE", "B_TIME_EXIT", "C_STRUCTURAL_STOP_TIME", "D_FULL_STRUCTURAL")
FIXED_TARGET_R = 2.0
MAX_HOLDING_SESSIONS = 15
MINIMUM_ACTUAL_RR = 1.5


class AblationArmStrategy:
    """Apply a registered exit arm around an otherwise identical C-arm selector."""

    def __init__(self, *, arm: str, signal_strategy: Any,
                 structural_config: StructuralExitConfig) -> None:
        if arm not in ARMS:
            raise ValueError(f"Unknown exit-ablation arm: {arm}.")
        self.arm = arm
        self.signal_strategy = signal_strategy
        self.structural_config = structural_config
        self.name = f"weekly_selection_exit_ablation_{arm.lower()}"
        self.version = f"weekly_selection_exit_ablation_v1:{arm}"
        self.parameters = {"arm": arm, "fixed_target_r": FIXED_TARGET_R,
                           "maximum_holding_sessions": MAX_HOLDING_SESSIONS,
                           "minimum_actual_reward_risk": MINIMUM_ACTUAL_RR,
                           "structural_config": asdict(structural_config),
                           "signal_strategy": dict(signal_strategy.parameters)}
        self.entry_decisions: list[dict[str, Any]] = []
        self.time_exit_decisions: list[dict[str, Any]] = []
        self.rejection_counts: Counter[str] = Counter()
        self.zone_config = self._find_zone_config()

    def _find_zone_config(self) -> ZoneConfig:
        current = self.signal_strategy
        while current is not None:
            config = getattr(current, "zone_config", None)
            if config is not None:
                return config
            current = getattr(current, "daily_strategy", None)
        raise ValueError("Exit ablation requires the wrapped sr_zone_v2 zone configuration.")

    @property
    def diagnostics(self):
        return self.signal_strategy.diagnostics

    def _structural_plan(self, signal: EntrySignal, context: Any, *, full: bool) -> dict[str, Any]:
        rows = context.daily_history.get(signal.security_id, ())
        atr = daily_atr(rows, self.zone_config.atr_period)
        resistances = active_resistance_zones_asof(
            daily_rows=rows, security_id=signal.security_id,
            as_of=context.session_date, zone_config=self.zone_config)
        config = replace(self.structural_config,
                         minimum_reward_risk=MINIMUM_ACTUAL_RR,
                         maximum_holding_sessions=MAX_HOLDING_SESSIONS,
                         no_target_policy="REJECT")
        return build_structural_exit_plan(
            setup_type=str(signal.metadata.get("setup_type", "")),
            signal_metadata=signal.metadata, active_resistances=resistances,
            atr=atr, config=config,
            # D evaluates the RR using the next-session executable fill.
            enforce_minimum_reward_risk=not full)

    def generate_entries(self, context: Any):
        source_signals = tuple(self.signal_strategy.generate_entries(context))
        if self.arm == "A_BASELINE":
            return tuple(replace(signal, metadata={**dict(signal.metadata),
                                                   "target_r": FIXED_TARGET_R})
                         for signal in source_signals)

        accepted: list[EntrySignal] = []
        for signal in source_signals:
            metadata = {**dict(signal.metadata), "target_r": FIXED_TARGET_R}
            if self.arm == "B_TIME_EXIT":
                metadata["maximum_holding_sessions"] = MAX_HOLDING_SESSIONS
                accepted.append(replace(signal, metadata=metadata))
                continue

            is_full = self.arm == "D_FULL_STRUCTURAL"
            plan = self._structural_plan(signal, context, full=is_full)
            decision = {"date": context.session_date.isoformat(),
                        "security_id": signal.security_id, "symbol": signal.symbol,
                        "setup_type": signal.metadata.get("setup_type"), **plan}
            self.entry_decisions.append(decision)
            stop_valid = (plan.get("stop_price") is not None
                          and plan.get("stop_distance_percent", 0) > 0)
            if not stop_valid:
                reason = str(plan.get("reason_code", "INVALID_STRUCTURAL_STOP"))
                self.rejection_counts[reason] += 1
                self._record_diagnostic(signal, context, reason)
                continue

            if is_full and not plan.get("accepted", False):
                reason = str(plan["reason_code"])
                self.rejection_counts[reason] += 1
                self._record_diagnostic(signal, context, reason)
                continue

            metadata.update({"exit_ablation_arm": self.arm,
                             "maximum_holding_sessions": MAX_HOLDING_SESSIONS,
                             "structural_exit": {key: value for key, value in plan.items()
                                                 if key not in {"accepted", "reason_code"}}})
            if is_full:
                metadata["target_r"] = None
                metadata["actual_minimum_reward_risk"] = MINIMUM_ACTUAL_RR
                target_price = float(plan["target_price"])
            else:
                # BacktestEngine derives this target from the actual executable fill.
                metadata["target_r"] = FIXED_TARGET_R
                target_price = None
            accepted.append(replace(signal, stop_price=float(plan["stop_price"]),
                                    target_price=target_price, metadata=metadata,
                                    strategy_version=self.version,
                                    reason_codes=tuple(signal.reason_codes)+(self.arm,)))
        return tuple(accepted)

    def _record_diagnostic(self, signal: EntrySignal, context: Any, reason: str) -> None:
        setup = str(signal.metadata.get("setup_type", "unknown"))
        for row in reversed(self.diagnostics):
            if (row.get("security_id") == signal.security_id
                    and row.get("signal_date") == context.session_date.isoformat()
                    and row.get("setup_type") == setup and row.get("accepted")):
                row.update({"accepted": False, "rejection_reason": reason,
                            "reason_code": reason, "exit_ablation_arm": self.arm})
                return

    def generate_exits(self, context: Any):
        exits = list(getattr(self.signal_strategy, "generate_exits", lambda _: ())(context))
        scheduled = {signal.security_id for signal in exits}
        if self.arm == "A_BASELINE" or context.next_session_open is None:
            return tuple(exits)
        for security_id, position in context.open_positions.items():
            if security_id in scheduled:
                continue
            entry_day = date.fromisoformat(str(position["entry_timestamp"])[:10])
            rows = context.daily_history.get(security_id, ())
            held = sum(entry_day <= date.fromisoformat(str(row["date"])[:10])
                       <= context.session_date for row in rows)
            if held < MAX_HOLDING_SESSIONS:
                continue
            details = {"holding_sessions": held,
                       "maximum_holding_sessions": MAX_HOLDING_SESSIONS,
                       "exit_price_basis": "NEXT_SESSION_OPEN"}
            self.time_exit_decisions.append({"date": context.session_date.isoformat(),
                                             "security_id": security_id, **details})
            exits.append(ExitSignal(security_id=security_id,
                                    signal_timestamp=context.timestamp,
                                    execution_timestamp=context.next_session_open,
                                    reason_code="TIME_EXIT", metadata=details,
                                    strategy_version=self.version))
        return tuple(exits)


def _summary(trades: Sequence[Any], starting_capital: float,
             *, transaction_costs: float | None = None) -> dict[str, Any]:
    net = [float(trade.net_pnl) for trade in trades]
    gross = [float(trade.gross_pnl) for trade in trades]
    gains = sum(value for value in net if value > 0)
    losses = -sum(value for value in net if value < 0)
    exits = Counter(str(trade.exit_reason) for trade in trades)
    time_net = [float(trade.net_pnl) for trade in trades if trade.exit_reason == "TIME_EXIT"]
    all_exit_count = len(trades)
    return {
        "trade_count": all_exit_count,
        "win_rate": sum(value > 0 for value in net)/all_exit_count if all_exit_count else 0.0,
        "gross_expectancy": sum(gross)/all_exit_count if all_exit_count else 0.0,
        "net_expectancy": sum(net)/all_exit_count if all_exit_count else 0.0,
        "profit_factor": gains/losses if losses else (None if gains else 0.0),
        "trading_return": sum(net)/starting_capital if starting_capital else 0.0,
        "transaction_costs": (sum(float(trade.costs) for trade in trades)
                              if transaction_costs is None else transaction_costs),
        "average_holding_sessions": (sum(int(trade.holding_period) for trade in trades)/all_exit_count
                                     if all_exit_count else 0.0),
        "median_holding_sessions": median(int(trade.holding_period) for trade in trades) if trades else 0.0,
        "stop_count": exits.get("STOP", 0),
        "target_count": exits.get("TARGET", 0),
        "time_exit_count": exits.get("TIME_EXIT", 0),
        "other_exit_count": all_exit_count-sum(exits.get(reason, 0)
                                                for reason in ("STOP", "TARGET", "TIME_EXIT")),
        "exit_counts": dict(sorted(exits.items())),
        "time_exit": {
            "count": len(time_net),
            "profitable_count": sum(value > 0 for value in time_net),
            "losing_count": sum(value <= 0 for value in time_net),
            "average_net_pnl": sum(time_net)/len(time_net) if time_net else 0.0,
            "average_R_at_exit": (sum(float(trade.R_multiple) for trade in trades
                                       if trade.exit_reason == "TIME_EXIT")/len(time_net)
                                   if time_net else 0.0),
        },
    }


def _year_summary(result: BacktestResult, year: int) -> dict[str, Any]:
    trades = [trade for trade in result.trades if trade.exit_timestamp.year == year]
    curve = list(result.equity_curve)
    year_rows = [row for row in curve if str(row["date"])[:4] == str(year)]
    previous = [row for row in curve if str(row["date"])[:4] < str(year)]
    start_equity = (float(previous[-1]["equity"]) if previous
                    else float(result.metadata.starting_capital))
    end_equity = float(year_rows[-1]["equity"]) if year_rows else start_equity
    summary = _summary(trades, start_equity)
    summary["trading_return"] = end_equity/start_equity-1 if start_equity else 0.0
    entry_costs = sum(float(event.details.get("costs", 0.0)) for event in result.events
                      if event.timestamp.year == year and event.event_type == "POSITION_ENTRY")
    exit_costs = sum(float(trade.cost_breakdown.get("exit", 0.0))
                     +float(trade.cost_breakdown.get("exit_slippage", 0.0))
                     for trade in trades if trade.exit_timestamp.year == year)
    summary["transaction_costs"] = entry_costs+exit_costs
    return summary


def _arm_report(result: BacktestResult, strategy: AblationArmStrategy) -> dict[str, Any]:
    trades = list(result.trades)
    initial = float(result.metadata.starting_capital)
    by_setup = {}
    for setup, label in (("support_test", "SUPPORT"),
                         ("breakout_retest", "BREAKOUT_RETEST")):
        by_setup[label] = _summary([trade for trade in trades
                                   if trade.metadata.get("setup_type") == setup], initial)
    event_rejections = Counter(
        str(event.details.get("reason_code", event.details.get("reason", "UNKNOWN")))
        for event in result.events
        if event.event_type == "ENTRY_REJECTED"
        and (str(event.details.get("reason_code", "")).startswith("ACTUAL_")
             or str(event.details.get("reason", "")).startswith("ACTUAL_")))
    structural_rejections = Counter(strategy.rejection_counts)
    all_rejections = structural_rejections + event_rejections
    if strategy.arm == "D_FULL_STRUCTURAL":
        for reason in ("NO_STRUCTURAL_TARGET", "INSUFFICIENT_STRUCTURAL_RR",
                       "ACTUAL_RR_BELOW_MINIMUM"):
            all_rejections.setdefault(reason, 0)
    overall = _summary(trades, initial,
                       transaction_costs=float(result.metrics.get("transaction_costs_total", 0.0)))
    overall["trading_return"] = float(result.metrics.get("trading_net_return", 0.0))
    return {
        "arm": strategy.arm,
        "overall": overall,
        "by_year": {str(year): _year_summary(result, year) for year in (2023, 2024)},
        "by_setup": by_setup,
        "time_exit": _summary([trade for trade in trades if trade.exit_reason == "TIME_EXIT"], initial)["time_exit"],
        "structural_entry_rejections": dict(sorted(all_rejections.items())),
        "entry_signal_count": sum(event.event_type == "ENTRY_SIGNAL" for event in result.events),
        "position_entry_count": sum(event.event_type == "POSITION_ENTRY" for event in result.events),
        "time_exit_decision_count": len(strategy.time_exit_decisions),
        "trading_ending_equity": float(result.metrics.get("trading_ending_equity", initial)),
        "transaction_costs_total": float(result.metrics.get("transaction_costs_total", 0.0)),
        "open_position_count_at_end": len(result.open_positions),
    }


def _deltas(reports: Mapping[str, Any]) -> dict[str, Any]:
    measures = ("profit_factor", "net_expectancy", "trading_return", "average_holding_sessions")
    pairs = (("B_vs_A", "B_TIME_EXIT", "A_BASELINE"),
             ("C_vs_B", "C_STRUCTURAL_STOP_TIME", "B_TIME_EXIT"),
             ("D_vs_C", "D_FULL_STRUCTURAL", "C_STRUCTURAL_STOP_TIME"))
    output = {}
    for label, newer, older in pairs:
        output[label] = {metric: ((None if reports[newer]["overall"][metric] is None
                                   or reports[older]["overall"][metric] is None
                                   else reports[newer]["overall"][metric]-reports[older]["overall"][metric]))
                         for metric in measures}
    return output


def run_ablation(*, benchmark_path: str | Path | None = None,
                 settings_path: str | Path = DEFAULT_SETTINGS_PATH,
                 manifest_path: str | Path = PROJECT_ROOT/"data/processed/market_data_manifest.csv",
                 data_root: str | Path = PROJECT_ROOT/"data/processed/market_data",
                 output_directory: str | Path = PROJECT_ROOT/"data/processed/backtests/weekly_selection") -> tuple[dict[str, Any], Path]:
    app = load_settings(settings_path)
    raw = yaml.safe_load(Path(settings_path).read_text(encoding="utf-8"))
    bt = raw.get("backtest", {})
    weekly_raw = bt.get("weekly_selection", {})
    configured_benchmark = benchmark_path or weekly_raw.get("benchmark_path")
    if not configured_benchmark:
        raise ValueError("Set backtest.weekly_selection.benchmark_path or pass --benchmark.")
    benchmark = load_nifty_benchmark(configured_benchmark)

    base_raw = bt.get("strategy", {})
    daily_config = SRBaselineConfig(**{key: value for key, value in base_raw.items()
                                      if key in SRBaselineConfig.__dataclass_fields__})
    zone_raw = bt.get("strategy_zone_v2", {})
    zone_values = {key: value for key, value in zone_raw.items()
                   if key in ZoneConfig.__dataclass_fields__}
    zone_values.setdefault("swing_left", daily_config.swing_left)
    zone_values.setdefault("swing_right", daily_config.swing_right)
    zone_values.setdefault("minimum_pivots", daily_config.minimum_zone_pivots)
    zone_values.setdefault("maximum_age", daily_config.maximum_zone_age_sessions)
    zone_config = ZoneConfig(**zone_values)
    weekly_allowed = set(WeeklySelectionConfig.__dataclass_fields__)
    unknown = set(weekly_raw)-weekly_allowed-{"version", "benchmark_path"}
    if unknown:
        raise ValueError(f"Unknown weekly-selection settings: {', '.join(sorted(unknown))}.")
    weekly_config = WeeklySelectionConfig(**{key: value for key, value in weekly_raw.items()
                                             if key in weekly_allowed})
    structural_raw = bt.get("structural_exit", {})
    structural_allowed = set(StructuralExitConfig.__dataclass_fields__)
    unknown_structural = set(structural_raw)-structural_allowed-{"default_mode"}
    if unknown_structural:
        raise ValueError(f"Unknown structural-exit settings: {', '.join(sorted(unknown_structural))}.")
    structural_config = StructuralExitConfig(**{key: value for key, value in structural_raw.items()
                                                  if key in structural_allowed})

    costs_raw = bt.get("transaction_costs", {})
    costs = CostModel(brokerage_rate=float(costs_raw.get("brokerage_rate", 0)),
                      stt_buy_rate=float(costs_raw.get("stt_buy_rate", .001)),
                      stt_sell_rate=float(costs_raw.get("stt_sell_rate", .001)),
                      nse_transaction_rate=float(costs_raw.get("nse_transaction_rate", .0000307)),
                      gst_rate=float(costs_raw.get("gst_rate", .18)),
                      sebi_rate=float(costs_raw.get("sebi_rate", .000001)),
                      stamp_buy_rate=float(costs_raw.get("stamp_buy_rate", .00015)),
                      ipft_rate=float(costs_raw.get("ipft_rate", .000000001)),
                      dp_charge_per_sell=float(costs_raw.get("dp_charge_per_sell", 15.34)),
                      slippage_bps=0,
                      scenario=str(costs_raw.get("scenario", "CURRENT_ZERODHA_EQUITY_DELIVERY_2026"))+"_SLIPPAGE_0BPS",
                      verification_label=str(costs_raw.get("verification_label", "CURRENT_ZERODHA_EQUITY_DELIVERY_2026")),
                      verified_on=str(costs_raw.get("verified_on", "")),
                      provenance=str(costs_raw.get("provenance", "")))
    operating_raw = bt.get("operating_costs", {})
    operating = OperatingCostModel(float(operating_raw.get("kite_connect_monthly", 500)),
                                  float(operating_raw.get("hosting_monthly", 0)),
                                  float(operating_raw.get("other_monthly", 0)),
                                  bool(operating_raw.get("apply_to_economic_results", True)))
    daily, dataset_version = load_daily_dataset_through(
        universe_id=PIT_TRADABLE, start_date=START, end_date=END,
        data_root=data_root, manifest_path=manifest_path)
    backtest_config = BacktestConfig(
        PIT_TRADABLE, START, END, initial_capital=app.initial_capital,
        max_risk_per_trade=0.01,
        max_open_positions=int(bt.get("max_open_positions", 10)),
        timezone_name=app.timezone, costs=costs, operating_costs=operating)

    results: dict[str, Any] = {}
    for arm in ARMS:
        daily_strategy = SRZoneV2Strategy(config=daily_config, setup="both", target_r=2.0,
                                          weekly_filter=False, evaluation_start=START,
                                          zone_config=zone_config)
        selection = WeeklySelectionStrategy(daily_strategy=daily_strategy,
                                            benchmark_daily=benchmark, config=weekly_config)
        strategy = AblationArmStrategy(arm=arm, signal_strategy=selection,
                                       structural_config=structural_config)
        result = BacktestEngine(config=backtest_config).run(
            strategy=strategy, daily_data=daily, dataset_version=dataset_version)
        results[arm] = {"result": result, "strategy": strategy,
                        "report": _arm_report(result, strategy),
                        "benchmark_sha256": selection.parameters["benchmark_sha256"]}

    reports = {arm: item["report"] for arm, item in results.items()}
    report = {
        "experiment": "WEEKLY_SELECTION_EXIT_ABLATION_2023_2024",
        "arms": reports,
        "deltas": _deltas(reports),
        "structural_rejection_reasons_D": reports["D_FULL_STRUCTURAL"]["structural_entry_rejections"],
        "research_controls": {
            "period": [START.isoformat(), END.isoformat()],
            "universe": PIT_TRADABLE,
            "starting_capital": app.initial_capital,
            "risk_per_trade": 0.01,
            "max_open_positions": int(bt.get("max_open_positions", 10)),
            "transaction_cost_scenario": costs.scenario,
            "slippage_bps": 0,
            "weekly_selection_config": asdict(weekly_config),
            "strategy_version": "sr_zone_v2_weekly_selection_v1",
            "entry_setup_logic_shared": True,
            "dataset_version": dataset_version,
            "fixed_target_r_A_B_C": FIXED_TARGET_R,
            "maximum_holding_sessions_B_C_D": MAX_HOLDING_SESSIONS,
            "minimum_actual_reward_risk_D": MINIMUM_ACTUAL_RR,
            "benchmark_sha256": results["A_BASELINE"]["benchmark_sha256"],
            "year_trade_attribution": "exit calendar year; yearly trading return is mark-to-market equity return",
            "arms": list(ARMS),
        },
        "event_ordering": [
            "Prior-close exits, including TIME_EXIT, execute at the next session open.",
            "Existing-position opening gaps through stop or target are resolved before new entries.",
            "D actual fill, stop risk, target reward, and RR are checked before opening the position.",
            "Daily stop/target checks follow entry; stop wins when both are touched in one bar.",
            "After daily-bar exits at close, the 15th completed holding session may schedule TIME_EXIT for the next open.",
        ],
    }
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    artifact = output/"weekly_selection_exit_ablation_2023_2024.json"
    temp = artifact.with_suffix(".json.tmp")
    temp.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    temp.replace(artifact)
    return report, artifact


def _display(report: Mapping[str, Any]) -> str:
    columns = ("trades", "win%", "grossExp", "netExp", "PF", "return%", "costs",
               "avgHold", "medHold", "STOP", "TARGET", "TIME_EXIT", "other")

    def row(label: str, stats: Mapping[str, Any]) -> str:
        pf = "n/a" if stats["profit_factor"] is None else f"{stats['profit_factor']:.3f}"
        values = (str(stats["trade_count"]), f"{stats['win_rate']*100:.1f}",
                  f"{stats['gross_expectancy']:.2f}", f"{stats['net_expectancy']:.2f}", pf,
                  f"{stats['trading_return']*100:.2f}", f"{stats['transaction_costs']:.2f}",
                  f"{stats['average_holding_sessions']:.2f}", f"{stats['median_holding_sessions']:.1f}",
                  str(stats["stop_count"]), str(stats["target_count"]),
                  str(stats["time_exit_count"]), str(stats["other_exit_count"]))
        return f"{label:<26} " + " ".join(f"{value:>9}" for value in values)

    lines = ["OVERALL", "arm                       " + " ".join(f"{name:>9}" for name in columns)]
    for arm in ARMS:
        lines.append(row(arm, report["arms"][arm]["overall"]))
    for year in ("2023", "2024"):
        lines += ["", year, "arm                       " + " ".join(f"{name:>9}" for name in columns)]
        for arm in ARMS:
            lines.append(row(arm, report["arms"][arm]["by_year"][year]))
    lines += ["", "TIME_EXIT DETAILS (overall by arm)"]
    for arm in ARMS:
        lines.append(f"{arm}: {json.dumps(report['arms'][arm]['time_exit'], sort_keys=True)}")
    for year in ("2023", "2024"):
        lines.append(f"TIME_EXIT DETAILS {year}")
        for arm in ARMS:
            lines.append(f"{arm}: {json.dumps(report['arms'][arm]['by_year'][year]['time_exit'], sort_keys=True)}")
    lines += ["", "SETUP SPLIT (overall by arm)"]
    for arm in ARMS:
        lines.append(f"{arm}: {json.dumps(report['arms'][arm]['by_setup'], sort_keys=True)}")
    lines += ["", "D STRUCTURAL ENTRY REJECTIONS",
              json.dumps(report["structural_rejection_reasons_D"], indent=2, sort_keys=True),
              "", "FACTUAL DELTAS (later arm minus earlier arm)"]
    for label, values in report["deltas"].items():
        lines.append(f"{label}: {json.dumps(values, sort_keys=True)}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, help="Official bounded NIFTY 50 daily index CSV.")
    parser.add_argument("--settings", type=Path, default=DEFAULT_SETTINGS_PATH)
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT/"data/processed/market_data_manifest.csv")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT/"data/processed/market_data")
    parser.add_argument("--output-directory", type=Path, default=PROJECT_ROOT/"data/processed/backtests/weekly_selection")
    args = parser.parse_args()
    try:
        report, artifact = run_ablation(benchmark_path=args.benchmark, settings_path=args.settings,
                                        manifest_path=args.manifest, data_root=args.data_root,
                                        output_directory=args.output_directory)
    except (ValueError, OSError, BacktestDataError) as error:
        parser.error(str(error))
    print(_display(report))
    print(f"\nArtifact: {artifact}")


if __name__ == "__main__":
    main()


__all__ = ["ARMS", "AblationArmStrategy", "run_ablation"]
