"""CLI for deliberately selected research phases and the engine smoke strategy."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml

from src.backtest.data import BacktestDataError, load_daily_dataset
from src.backtest.engine import BacktestEngine
from src.backtest.models import BacktestConfig, CostModel, OperatingCostModel
from src.backtest.smoke import SmokeStrategy
from src.configuration import DEFAULT_SETTINGS_PATH, PROJECT_ROOT, load_settings
from src.data.universe.provider import PIT_TRADABLE


RESEARCH_PHASES = {
    "development": (date(2023, 1, 1), date(2024, 12, 31)),
    "validation": (date(2025, 1, 1), date(2025, 12, 31)),
    "final_oos": (date(2026, 1, 1), date(2026, 9, 22)),
    "stress": (date(2020, 1, 1), date(2022, 12, 31)),
}


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def persist_result(result: Any, *, output_directory: str | Path) -> Path:
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / f"{result.metadata.run_id}.json"
    temporary = output.with_suffix(".json.tmp")
    payload = {
        "metadata": _jsonable(result.metadata),
        "metrics": _jsonable(result.metrics),
        "trades": _jsonable(result.trades),
        "events": _jsonable(result.events),
        "equity_curve": _jsonable(result.equity_curve),
        "remaining_cash": result.remaining_cash,
        "reserved_capital": result.reserved_capital,
        "realized_pnl": result.realized_pnl,
        "unrealized_pnl": result.unrealized_pnl,
        "open_positions": _jsonable(result.open_positions),
    }
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def _cost_model(raw: dict[str, Any]) -> CostModel:
    return CostModel(
        brokerage_rate=float(raw.get("brokerage_rate", 0.0)),
        stt_buy_rate=float(raw.get("stt_buy_rate", 0.0)),
        stt_sell_rate=float(raw.get("stt_sell_rate", 0.0)),
        nse_transaction_rate=float(raw.get("nse_transaction_rate", 0.0)),
        gst_rate=float(raw.get("gst_rate", 0.0)),
        sebi_rate=float(raw.get("sebi_rate", 0.0)),
        stamp_buy_rate=float(raw.get("stamp_buy_rate", 0.0)),
        ipft_rate=float(raw.get("ipft_rate", 0.0)),
        dp_charge_per_sell=float(raw.get("dp_charge_per_sell", 0.0)),
        slippage_bps=float(raw.get("slippage_bps", 0.0)),
        scenario=str(raw.get("scenario", "CUSTOM")),
        verification_label=str(raw.get("verification_label", "CUSTOM_UNVERIFIED")),
        verified_on=str(raw.get("verified_on", "")),
        provenance=str(raw.get("provenance", "")),
    )


def _operating_cost_model(raw: dict[str, Any]) -> OperatingCostModel:
    return OperatingCostModel(
        kite_connect_monthly=float(raw.get("kite_connect_monthly", 500.0)),
        hosting_monthly=float(raw.get("hosting_monthly", 0.0)),
        other_monthly=float(raw.get("other_monthly", 0.0)),
        apply_to_economic_results=bool(raw.get("apply_to_economic_results", True)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=tuple(RESEARCH_PHASES), required=True,
                        help="Select exactly one research phase; other periods are not run automatically.")
    parser.add_argument("--strategy", choices=("smoke",), required=True)
    parser.add_argument("--cost-scenario", choices=("current_zerodha", "raw_zero"),
                        default="current_zerodha",
                        help="Current Zerodha delivery costs are the default; raw_zero explicitly disables all costs.")
    parser.add_argument("--universe", default=PIT_TRADABLE)
    parser.add_argument("--start", type=date.fromisoformat)
    parser.add_argument("--end", type=date.fromisoformat)
    parser.add_argument("--settings", type=Path, default=DEFAULT_SETTINGS_PATH)
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "data/processed/market_data_manifest.csv")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data/processed/market_data")
    parser.add_argument("--output-directory", type=Path, default=PROJECT_ROOT / "data/processed/backtests")
    args = parser.parse_args()
    if (args.start is None) != (args.end is None):
        parser.error("--start and --end must be provided together.")
    default_start, default_end = RESEARCH_PHASES[args.phase]
    start_date, end_date = (args.start, args.end) if args.start else (default_start, default_end)
    if start_date > end_date:
        parser.error("--start must not be later than --end.")

    try:
        app_settings = load_settings(args.settings)
        with Path(args.settings).open("r", encoding="utf-8") as handle:
            raw_settings = yaml.safe_load(handle)
        bt_settings = raw_settings.get("backtest", {})
        if args.cost_scenario == "raw_zero":
            costs = CostModel.raw_zero()
            operating_costs = OperatingCostModel(0.0, 0.0, 0.0, True)
        else:
            costs = _cost_model(bt_settings.get("transaction_costs", {}))
            operating_costs = _operating_cost_model(bt_settings.get("operating_costs", {}))
        daily, dataset_version = load_daily_dataset(
            universe_id=args.universe, start_date=start_date, end_date=end_date,
            data_root=args.data_root, manifest_path=args.manifest,
        )
        config = BacktestConfig(
            universe_id=args.universe, start_date=start_date, end_date=end_date,
            initial_capital=app_settings.initial_capital,
            max_risk_per_trade=app_settings.max_risk_per_trade,
            max_open_positions=int(bt_settings.get("max_open_positions", 10)),
            timezone_name=app_settings.timezone, costs=costs,
            operating_costs=operating_costs,
        )
        strategy = SmokeStrategy(target_r=float(bt_settings.get("smoke_target_r", 2.0)))
        result = BacktestEngine(config=config).run(
            strategy=strategy, daily_data=daily, dataset_version=dataset_version,
        )
        output = persist_result(result, output_directory=args.output_directory)
    except (ValueError, OSError, BacktestDataError) as error:
        parser.error(str(error))
    print(json.dumps({"run_id": result.metadata.run_id, "phase": args.phase,
                      "strategy": strategy.name, "period": [start_date.isoformat(), end_date.isoformat()],
                      "cost_scenario": costs.scenario,
                      "metrics": result.metrics, "trades": len(result.trades),
                      "open_positions": len(result.open_positions),
                      "cash": result.remaining_cash, "realized_pnl": result.realized_pnl,
                      "unrealized_pnl": result.unrealized_pnl,
                      "reserved_capital": result.reserved_capital,
                      "result_file": str(output)},
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
