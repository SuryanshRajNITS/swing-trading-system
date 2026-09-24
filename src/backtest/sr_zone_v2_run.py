"""Pre-registered sr_zone_v2 comparisons on the 2023-2024 development window."""

from __future__ import annotations

from dataclasses import fields, replace
from datetime import date
import json
from pathlib import Path
from typing import Any

import yaml

from src.backtest.data import BacktestDataError, load_daily_dataset_through
from src.backtest.engine import BacktestEngine
from src.backtest.models import BacktestConfig, CostModel, OperatingCostModel
from src.backtest.strategy_run import build_research_report, _jsonable
from src.configuration import DEFAULT_SETTINGS_PATH, PROJECT_ROOT, load_settings
from src.data.universe.provider import PIT_TRADABLE
from src.strategy.sr_baseline import SRBaselineConfig
from src.strategy.sr_zone_v2 import SRZoneV2Strategy, ZoneConfig

START, END = date(2023, 1, 1), date(2024, 12, 31)


def run_research(*, setup: str, weekly_filter: bool = False, slippage_bps: float = 0,
                 settings_path: str | Path = DEFAULT_SETTINGS_PATH,
                 manifest_path: str | Path = PROJECT_ROOT / "data/processed/market_data_manifest.csv",
                 data_root: str | Path = PROJECT_ROOT / "data/processed/market_data",
                 output_directory: str | Path = PROJECT_ROOT / "data/processed/backtests/sr_zone_v2") -> tuple[dict[str, Any], Path]:
    if setup not in {"support_test", "breakout_retest", "both"} or slippage_bps < 0:
        raise ValueError("Invalid setup or negative slippage.")
    app = load_settings(settings_path)
    raw = yaml.safe_load(Path(settings_path).read_text(encoding="utf-8"))
    bt = raw.get("backtest", {})
    st = bt.get("strategy", {})
    allowed = set(SRBaselineConfig.__dataclass_fields__)
    cfg = SRBaselineConfig(**{k:v for k,v in st.items() if k in allowed})
    zraw=bt.get("strategy_zone_v2",{})
    if zraw.get("version","sr_zone_v2") != "sr_zone_v2":
        raise ValueError("This runner only implements strategy version sr_zone_v2.")
    zallowed=set(ZoneConfig.__dataclass_fields__)
    unknown=set(zraw)-zallowed-{"version"}
    if unknown:
        raise ValueError(f"Unknown sr_zone_v2 settings: {', '.join(sorted(unknown))}.")
    zone_values={k:v for k,v in zraw.items() if k in zallowed}
    zone_values.setdefault("swing_left",cfg.swing_left)
    zone_values.setdefault("swing_right",cfg.swing_right)
    zone_values.setdefault("minimum_pivots",cfg.minimum_zone_pivots)
    zone_values.setdefault("maximum_age",cfg.maximum_zone_age_sessions)
    zcfg=ZoneConfig(**zone_values)
    cs = bt.get("transaction_costs", {})
    costs = CostModel(
        brokerage_rate=float(cs.get("brokerage_rate",0)), stt_buy_rate=float(cs.get("stt_buy_rate",.001)),
        stt_sell_rate=float(cs.get("stt_sell_rate",.001)), nse_transaction_rate=float(cs.get("nse_transaction_rate",.0000307)),
        gst_rate=float(cs.get("gst_rate",.18)), sebi_rate=float(cs.get("sebi_rate",.000001)),
        stamp_buy_rate=float(cs.get("stamp_buy_rate",.00015)), ipft_rate=float(cs.get("ipft_rate",.000000001)),
        dp_charge_per_sell=float(cs.get("dp_charge_per_sell",15.34)), slippage_bps=slippage_bps,
        scenario=f"{cs.get('scenario','CURRENT_ZERODHA_EQUITY_DELIVERY_2026')}_SLIPPAGE_{slippage_bps:g}BPS",
        verification_label=str(cs.get("verification_label","CURRENT_ZERODHA_EQUITY_DELIVERY_2026")),
        verified_on=str(cs.get("verified_on","")), provenance=str(cs.get("provenance","")))
    op = bt.get("operating_costs", {})
    operating = OperatingCostModel(float(op.get("kite_connect_monthly",500)), float(op.get("hosting_monthly",0)),
                                   float(op.get("other_monthly",0)), bool(op.get("apply_to_economic_results",True)))
    daily, version = load_daily_dataset_through(universe_id=PIT_TRADABLE, start_date=START, end_date=END,
                                                 data_root=data_root, manifest_path=manifest_path)
    backtest = BacktestConfig(PIT_TRADABLE, START, END, initial_capital=app.initial_capital,
                              max_risk_per_trade=app.max_risk_per_trade,
                              max_open_positions=int(bt.get("max_open_positions",10)),
                              timezone_name=app.timezone, costs=costs, operating_costs=operating)
    strategy = SRZoneV2Strategy(config=cfg, setup=setup, target_r=2.0,
                                weekly_filter=weekly_filter, evaluation_start=START,
                                zone_config=zcfg)
    result = BacktestEngine(config=backtest).run(strategy=strategy, daily_data=daily, dataset_version=version)
    report = build_research_report(result, strategy)
    report["research_controls"] = {"period":[START.isoformat(),END.isoformat()],"strategy_version":strategy.version,
                                   "setup":setup,"target_r":2.0,"weekly_filter":weekly_filter,
                                   "cost_scenario":costs.scenario,"slippage_bps":slippage_bps,
                                   "outside_period_data_parsed":False}
    report["zone_quality"] = {k:{key:value for key,value in v.items() if key != "zones"}
                              for k,v in strategy.zone_snapshots.items()}
    out = Path(output_directory); out.mkdir(parents=True, exist_ok=True)
    path = out / f"{setup}_r2_weekly-{int(weekly_filter)}_current_zerodha_{slippage_bps:g}bps.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(_jsonable(report),indent=2,sort_keys=True,allow_nan=False),encoding="utf-8"); tmp.replace(path)
    return report,path


def run_pre_registered(*, slippage_bps: float = 0, **kwargs: Any) -> dict[str, Any]:
    cases = [("support_test",False),("breakout_retest",False),("both",False),("both",True)]
    summaries=[]
    for setup,weekly in cases:
        report,path=run_research(setup=setup,weekly_filter=weekly,slippage_bps=slippage_bps,**kwargs)
        summaries.append({"artifact":path.name,"strategy_version":"sr_zone_v2","setup":setup,
                         "weekly_filter":weekly,"metrics":report["metrics"],
                         "setup_performance":report["setup_performance"]})
    out=Path(kwargs.get("output_directory",PROJECT_ROOT/"data/processed/backtests/sr_zone_v2"))
    (out/"comparison_summary.json").write_text(json.dumps(summaries,indent=2,sort_keys=True,allow_nan=False),encoding="utf-8")
    return {"runs":summaries,"period":[START.isoformat(),END.isoformat()]}


if __name__ == "__main__":
    print(json.dumps(run_pre_registered(),indent=2,allow_nan=False))
