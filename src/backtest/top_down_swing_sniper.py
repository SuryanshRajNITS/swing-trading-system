"""Corrected 2023-2024 hourly-zone replay and ex-ante patient-sniper study."""
from __future__ import annotations

import argparse
import csv
from datetime import date
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

import yaml

from src.backtest import top_down_swing_portfolio as portfolio
from src.backtest.data import BacktestDataError, load_hourly_dataset_through
from src.backtest.top_down_swing_replay import START, END, replay
from src.configuration import PROJECT_ROOT
from src.data.hourly import HOURLY_MANIFEST
from src.data.market_data import DATA_ROOT, MANIFEST_PATH
from src.data.universe.provider import PIT_TRADABLE

OUTPUT = PROJECT_ROOT / "data/processed/backtests/top_down_swing_v1_sniper"
OLD_TRACE = PROJECT_ROOT / "data/processed/backtests/top_down_swing_v1/strategy_trace_2023_2024.csv"
OLD_REPORT = PROJECT_ROOT / "data/processed/backtests/top_down_swing_v1/top_down_swing_v1_replay_2023_2024.json"
CONFIG = PROJECT_ROOT / "config/top_down_swing_v1_sniper.yaml"


def _number(value: Any) -> float | None:
    try:
        n = float(value)
        return n if math.isfinite(n) else None
    except (TypeError, ValueError):
        return None


def _json(value: str | None) -> dict[str, Any]:
    try:
        item = json.loads(value or "{}")
        return item if isinstance(item, dict) else {}
    except (TypeError, json.JSONDecodeError):
        return {}


def classify_candidate(signal: Mapping[str, Any], minimum_rr: float = 1.5) -> dict[str, Any]:
    """Outcome-blind classification from information present by executable time."""
    plan = signal.get("plan") or {}
    result = signal.get("result") or {}
    features = result.get("confirmation_features") or {}
    confirmation = result.get("confirmation_type")
    zones = (features.get("held_support_zone_eligibility") or features.get("support_zone_eligibility") or
             features.get("involved_zones") or [])
    eligible = [z for z in zones if z.get("eligible_for_support_hold") or
        (z.get("role_state") in {"ACTIVE", "SUPPORT_CONFIRMED", "RESISTANCE_CONFIRMED"} and
         z.get("strength") in {"MEANINGFUL", "MAJOR"} and
         (not z.get("broken") or z.get("role_state") in {"SUPPORT_CONFIRMED", "RESISTANCE_CONFIRMED"}))]
    role_flip = any(z.get("role_state") == "SUPPORT_CONFIRMED" and
                    (z.get("original_side") or z.get("role_before")) == "resistance" for z in eligible)
    setup = signal.get("setup_zone") or {}
    location = signal.get("structural_location_category") or signal.get("weekly_location") or "UNKNOWN"
    entry, stop, target = (_number(plan.get(k)) for k in ("entry", "stop", "target"))
    rr = _number(plan.get("planned_reward_r"))
    valid_plan = entry is not None and stop is not None and target is not None and stop < entry < target and rr is not None and rr >= minimum_rr
    target_type = str(plan.get("target_timeframe") or plan.get("target_reference") or "")
    structural_target = target_type in {"DAILY", "WEEKLY", "HOURLY"} and plan.get("target_zone_strength") in {"MEANINGFUL", "MAJOR"}
    archetypes = {"SUPPORT_HOLD_CONFIRMATION", "RECLAIM_CONFIRMATION", "BREAKOUT_RETEST_CONFIRMATION",
                  "CONSOLIDATION_BREAKOUT_CONFIRMATION", "HIGHER_LOW_HIGHER_HIGH_CONFIRMATION"}
    location_good = location in {"SUPPORT_LOCATION_ACTIONABLE", "APPROACHING_SUPPORT", "APPROACHING_RESISTANCE", "BREAKOUT_RETEST"} or setup.get("strength") in {"MEANINGFUL", "MAJOR"}
    volume_reasons = []
    if confirmation in {"BREAKOUT_RETEST_CONFIRMATION", "CONSOLIDATION_BREAKOUT_CONFIRMATION"}:
        if features.get("expansion_vs_prior_mean"): volume_reasons.append("BREAKOUT_VOLUME_ABOVE_PRIOR_MEAN")
        if features.get("same_time_expansion"): volume_reasons.append("BREAKOUT_VOLUME_ABOVE_SAME_TIME_BASELINE")
    elif confirmation == "SUPPORT_HOLD_CONFIRMATION":
        if features.get("same_time_contraction"): volume_reasons.append("SELLING_VOLUME_CONTRACTION_AT_SUPPORT")
        if features.get("expansion_vs_prior_mean") or features.get("same_time_expansion"):
            volume_reasons.append("BULLISH_SUPPORT_REACTION_PARTICIPATION")
    elif confirmation == "RECLAIM_CONFIRMATION":
        if features.get("expansion_vs_prior_mean") or features.get("same_time_expansion"):
            volume_reasons.append("RECLAIM_WITH_INCREASED_PARTICIPATION")
    volume_support = bool(volume_reasons)
    hostile_value = features.get("high_volume_bearish_acceptance_below_support")
    hostile = hostile_value is True
    weekly = signal.get("weekly_structure") or {}
    weekly_aligned = bool(weekly.get("higher_high") and weekly.get("higher_low"))
    support_or_retest = confirmation in {"SUPPORT_HOLD_CONFIRMATION", "RECLAIM_CONFIRMATION", "BREAKOUT_RETEST_CONFIRMATION", "CONSOLIDATION_BREAKOUT_CONFIRMATION"}
    reasons: list[str] = []
    if not valid_plan:
        tier = "REJECTED"; reasons.append("INVALID_PLAN_OR_RR_BELOW_MINIMUM")
    elif not location_good or confirmation not in archetypes:
        tier = "TIER_C_WEAK"; reasons.append("MARGINAL_LOCATION_OR_CONFIRMATION")
    elif hostile:
        tier = "TIER_C_WEAK"; reasons.append("HOSTILE_BEARISH_VOLUME_EVIDENCE")
    elif not support_or_retest:
        tier = "TIER_C_WEAK"; reasons.append("NO_STRONG_STRUCTURAL_CONFIRMATION")
    else:
        supporting = []
        supporting.extend(volume_reasons)
        if weekly_aligned: supporting.append("WEEKLY_HH_HL_ALIGNMENT")
        if role_flip: supporting.append("CONFIRMED_ROLE_FLIP")
        if features.get("higher_low_to_higher_high"): supporting.append("HOURLY_HL_TO_HH")
        if setup.get("strength") in {"MEANINGFUL", "MAJOR"} and confirmation in {"SUPPORT_HOLD_CONFIRMATION", "RECLAIM_CONFIRMATION"}:
            supporting.append("MEANINGFUL_SUPPORT_REACTION")
        if supporting and structural_target:
            tier = "TIER_A_ROBUST"; reasons.extend(supporting)
        elif not structural_target:
            tier = "TIER_B_VALID" if location_good else "TIER_C_WEAK"
            reasons.append("FALLBACK_OR_NO_MEANINGFUL_TARGET_LIMITS_TIER")
        else:
            tier = "TIER_B_VALID"; reasons.append("VALID_WITHOUT_EXTRA_CONFLUENCE")
    loc_rank = 3 if role_flip else 2 if setup.get("strength") == "MAJOR" else 1 if location_good else 0
    t_rank = {"WEEKLY": 3, "DAILY": 2, "HOURLY": 1}.get(target_type, 0)
    c_rank = {"RECLAIM_CONFIRMATION": 5, "BREAKOUT_RETEST_CONFIRMATION": 4, "CONSOLIDATION_BREAKOUT_CONFIRMATION": 3,
              "SUPPORT_HOLD_CONFIRMATION": 2, "HIGHER_LOW_HIGHER_HIGH_CONFIRMATION": 1}.get(confirmation, 0)
    return {"quality_tier": tier, "tier_reasons": reasons,
        "role_at_confirmation": ",".join(sorted({str(z.get("role_state")) for z in zones if z.get("role_state")})),
        "zone_strength": ",".join(sorted({str(z.get("strength")) for z in zones if z.get("strength")})),
        "role_flip_confirmed": role_flip, "location_quality_rank": loc_rank, "structural_rr": rr,
        "target_quality_rank": t_rank,
        "target_strength_rank": {"MAJOR": 2, "MEANINGFUL": 1}.get(str(plan.get("target_zone_strength")), 0),
        "confirmation_quality_rank": c_rank,
        "volume_support": volume_support, "volume_support_reasons": volume_reasons,
        "hostile_volume_status": ("PRESENT" if hostile_value is True else
                                  "ABSENT" if hostile_value is False else "NOT_AVAILABLE_IN_FROZEN_FEATURES"),
        "weekly_hh_hl_alignment": weekly_aligned,
        "structural_target": structural_target, "confirmation_type": confirmation,
        "volume_features": features}


def _tier_rank(s: Mapping[str, Any]) -> tuple[Any, ...]:
    q = s["quality"]
    return ({"TIER_A_ROBUST": 0, "TIER_B_VALID": 1, "TIER_C_WEAK": 2, "REJECTED": 3}.get(s["tier"], 4),
        -q["location_quality_rank"], -(q["structural_rr"] or 0), -q["target_quality_rank"],
        -q.get("target_strength_rank", 0),
        -q["confirmation_quality_rank"], -int(q["volume_support"]), str(s["security_id"]), str(s["setup_instance_id"]))


def _sniper_candidates(signals: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Tier C remains in the candidate ledger but cannot consume sniper capital."""
    return [s for s in signals if s.get("tier") in {"TIER_A_ROBUST", "TIER_B_VALID"}]


def _read(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _write(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    portfolio._write_csv(path, rows)


def _pf(values: Sequence[float]) -> float | None:
    profit = sum(max(0, x) for x in values); loss = abs(sum(min(0, x) for x in values))
    return profit/loss if loss else (None if profit else 0.0)


def _tier_metrics(signals: Sequence[Mapping[str, Any]], trades: Sequence[Mapping[str, Any]], tier: str) -> dict[str, Any]:
    candidates = [s for s in signals if s["tier"] == tier]
    closed = [t for t in trades if t.get("status") == "CLOSED" and t.get("quality_tier") == tier]
    n = len(closed); gp = sum(max(0, float(t["gross_pnl"])) for t in closed); gl = abs(sum(min(0, float(t["gross_pnl"])) for t in closed))
    np = sum(max(0, float(t["net_pnl"])) for t in closed); nl = abs(sum(min(0, float(t["net_pnl"])) for t in closed))
    return {"tier": tier, "candidate_count": len(candidates), "executed_count": sum(t.get("quality_tier") == tier for t in trades),
        "closed_count": n, "gross_profit_factor": gp/gl if gl else None, "net_profit_factor": np/nl if nl else None,
        "win_rate": sum(float(t["net_pnl"]) > 0 for t in closed)/n if n else 0,
        "gross_expectancy": sum(float(t["gross_pnl"]) for t in closed)/n if n else 0,
        "net_expectancy": sum(float(t["net_pnl"]) for t in closed)/n if n else 0,
        "average_holding_period_sessions": sum(float(t.get("holding_sessions", 0)) for t in closed)/n if n else 0,
        "max_adverse_behavior_pct": min((float(t.get("mae_pct", 0)) for t in closed), default=0)}


def _with_tiers(trades: Sequence[Mapping[str, Any]], signals: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    lookup = {s["setup_instance_id"]: s for s in signals}; result = []
    for item in trades:
        row = dict(item); source = lookup.get(row.get("setup_instance_id"), {})
        qty = float(row.get("quantity") or 0); entry = _number(row.get("entry_price")); stop = _number(row.get("stop_price"))
        actual_risk = qty*(entry-stop) if entry is not None and stop is not None else 0
        row["quality_tier"] = source.get("tier")
        row["gross_structural_R"] = float(row["gross_pnl"])/actual_risk if actual_risk and row.get("gross_pnl") not in (None, "") else None
        risk_rupees = float(row.get("initial_risk_rupees") or 0)
        row["net_account_R"] = float(row["net_pnl"])/risk_rupees if risk_rupees and row.get("net_pnl") not in (None, "") else None
        row["max_adverse_behavior_pct"] = row.get("mae_pct")
        result.append(row)
    return result


def _costs(trades: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    keys = ("brokerage", "stt", "exchange_charges", "sebi_charges", "ipft_charges", "gst", "stamp_duty", "dp_charge")
    totals = {k: 0.0 for k in keys}; allin=[]; dp=[]; values=[]; gross=[]; closed=0
    for t in trades:
        closed += t.get("status") == "CLOSED"
        combined = {k: sum(float((t.get(part) or {}).get(k, 0) or 0) for part in ("entry_cost_breakdown", "exit_cost_breakdown")) for k in keys}
        for k in keys: totals[k] += combined[k]
        allin.append(sum(combined.values()))
        if t.get("status") == "CLOSED": dp.append(combined["dp_charge"])
        values.append(float(t.get("position_value", 0) or 0)); gross.append(float(t.get("gross_pnl", 0) or 0))
    total = sum(totals.values()); gp = sum(max(0, x) for x in gross)
    totals.update({"total": total, "dp_cost_per_closed_trade": sum(dp)/closed if closed else 0,
        "median_all_in_cost_per_trade": median(allin) if allin else 0,
        "cost_as_pct_position_value": total/sum(values) if sum(values) else 0,
        "cost_as_pct_gross_profit": total/gp if gp else None,
        "costs_reconcile": math.isclose(total, sum(float(t.get("transaction_costs", 0) or 0) for t in trades), abs_tol=1e-6)})
    return totals


def _frequency(trades: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    monthly = {f"{y}-{m:02d}": 0 for y in (2023, 2024) for m in range(1, 13)}
    for t in trades:
        if t.get("entry_timestamp"):
            key = str(t["entry_timestamp"])[:7]
            if key in monthly: monthly[key] += 1
    return {"trades_per_year": {str(y): sum(v for k,v in monthly.items() if k.startswith(str(y))) for y in (2023,2024)},
        "trades_per_month": sum(monthly.values())/24, "median_trades_per_month": median(monthly.values()),
        "months_0": sum(v == 0 for v in monthly.values()), "months_1_2": sum(1 <= v <= 2 for v in monthly.values()),
        "months_3_5": sum(3 <= v <= 5 for v in monthly.values()), "months_6_plus": sum(v >= 6 for v in monthly.values()),
        "monthly_counts": monthly}


def _support_hold_delta(old_rows: Sequence[Mapping[str, Any]], new_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    old_holds = [r for r in old_rows if _json(r.get("hourly_result")).get("confirmation_type") == "SUPPORT_HOLD_CONFIRMATION"]
    new_ids = {r.get("setup_instance_id") for r in new_rows
               if _json(r.get("hourly_result")).get("confirmation_type") == "SUPPORT_HOLD_CONFIRMATION"}
    old_by_id = {r.get("setup_instance_id"): r for r in old_holds if r.get("setup_instance_id")}
    removed = set(old_by_id) - new_ids
    causes = {"broken_or_invalid_role": 0, "minor_zone": 0,
              "legacy_hl_hh_label": 0, "not_reconstructable": 0}
    for setup_id in removed:
        result = _json(old_by_id[setup_id].get("hourly_result"))
        features = result.get("confirmation_features") or {}
        zones = features.get("held_support_zone_eligibility") or features.get("involved_zones") or []
        reason_codes = set(result.get("reason_codes") or [])
        if any(z.get("role_state") in {"RESISTANCE_CANDIDATE", "RESISTANCE_CONFIRMED"} or
               (z.get("broken") and z.get("role_state") != "SUPPORT_CONFIRMED") for z in zones):
            causes["broken_or_invalid_role"] += 1
        elif any(z.get("strength") == "MINOR" for z in zones):
            causes["minor_zone"] += 1
        elif "HOURLY_HIGHER_LOW_TO_HIGHER_HIGH" in reason_codes:
            causes["legacy_hl_hh_label"] += 1
        else:
            causes["not_reconstructable"] += 1
    before = len(set(old_by_id))
    after = len(new_ids)
    added = len(new_ids-set(old_by_id))
    return {"before": before, "after": after, "removed": len(removed), "added": added,
            "net_delta": after-before, "causes": causes,
            "cause_counts_reconcile": sum(causes.values()) == len(removed)}


def run(*, trace: Path = OLD_TRACE, replay_report: Path = OLD_REPORT, output_dir: Path = OUTPUT,
        data_root: Path = DATA_ROOT, daily_manifest: Path = MANIFEST_PATH,
        hourly_manifest: Path = HOURLY_MANIFEST, config_path: Path = CONFIG,
        reuse_corrected: bool = False) -> dict[str, Any]:
    old_report = json.loads(replay_report.read_text(encoding="utf-8"))
    if old_report.get("strategy_version") != "top_down_swing_v1" or old_report.get("2025_2026_loaded", False):
        raise ValueError("Input must be the frozen top_down_swing_v1 2023-2024 replay.")
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}; sniper = cfg.get("patient_sniper", {})
    min_rr = float(sniper.get("minimum_structural_rr", 1.5)); risk = float(sniper.get("risk_budget_pct", .01))
    deployment = float(sniper.get("preferred_capital_deployment_pct", .8))
    capitals = [float(x) for x in sniper.get("capital_sensitivity", [10000,25000,50000,100000,200000])]
    maxima = [int(x) for x in sniper.get("max_concurrent_positions", [1,2])]
    if set(maxima) != {1,2} or not 0 < risk <= 1 or not 0 < deployment <= 1 or min_rr <= 0:
        raise ValueError("Config must define max positions 1 and 2 and valid risk/deployment/RR values.")
    old_rows = _read(trace)
    corrected_dir = output_dir/"corrected"
    if reuse_corrected:
        new_trace = corrected_dir/"strategy_trace_2023_2024.csv"
        corrected_report_path = corrected_dir/"top_down_swing_v1_replay_2023_2024.json"
        if not new_trace.is_file() or not corrected_report_path.is_file():
            raise ValueError("--reuse-corrected requires completed corrected replay artifacts.")
        corrected = {"trace_path":new_trace,"report_path":corrected_report_path}
    else:
        corrected = replay(start=START, end=END, config_path=config_path, output_dir=corrected_dir,
            data_root=data_root, daily_manifest=daily_manifest, hourly_manifest=hourly_manifest, max_days=None)
        new_trace = Path(corrected["trace_path"])
    new_trace = Path(corrected["trace_path"]); new_rows = _read(new_trace)
    signals = portfolio._load_signals(new_trace, START, END)
    for s in signals:
        for key in ("setup_zone", "weekly_structure", "daily_structure", "structural_location_components",
                    "hourly_volume", "daily_volume", "weekly_selection_components", "nifty_regime_components"):
            if isinstance(s.get(key), str):
                s[key] = _json(s[key])
        s["quality"] = classify_candidate(s, min_rr); s["tier"] = s["quality"]["quality_tier"]
    bars, data_hash = load_hourly_dataset_through(universe_id=PIT_TRADABLE, start_date=date(2022,10,1), end_date=END,
        data_root=data_root, manifest_path=hourly_manifest, daily_manifest_path=daily_manifest)
    cost_model = portfolio._cost_config(PROJECT_ROOT/"config/settings.yaml")
    modes = [("corrected_baseline", 10, None, portfolio._rank, signals)]
    eligible = _sniper_candidates(signals)
    modes.extend((f"patient_sniper_{n}", n, deployment, _tier_rank, eligible) for n in maxima)
    headline={}; sensitivity=[]; trade_sets={}; raw_runs={}
    for mode, max_pos, deploy, rank_key, selected in modes:
        for capital in capitals:
            raw = portfolio._run_portfolio(selected, bars, cost_model, initial_capital=capital,
                risk_fraction=risk, max_positions=max_pos, period_start=START, rank_key=rank_key,
                preferred_deployment_pct=deploy)
            trades = _with_tiers(raw["trades"], signals); metrics=raw["metrics"]; costs=_costs(trades)
            dep=[float(t.get("capital_deployed_pct",0)) for t in trades]; util=[float(t.get("risk_budget_utilization",0)) for t in trades]
            row={"mode":mode,"starting_capital":capital,"ending_equity":metrics["ending_equity_mtm"],
                "return_pct":metrics["net_return_mtm"]*100,"max_drawdown_pct":metrics["daily_max_drawdown_pct"],
                "trades":metrics["trade_count"],"win_rate":metrics["win_rate"],
                "gross_pf":_pf([float(t["gross_pnl"]) for t in trades if t.get("status")=="CLOSED"]),
                "net_pf":metrics["profit_factor"],"gross_pnl":metrics["gross_pnl"],"transaction_costs":metrics["transaction_costs"],
                "dp_charges":costs["dp_charge"],"non_dp_costs":costs["total"]-costs["dp_charge"],
                "costs_pct_gross_profit":costs["cost_as_pct_gross_profit"],"costs_pct_position_value":costs["cost_as_pct_position_value"],
                "average_capital_deployed_pct":sum(dep)/len(dep) if dep else 0,"median_capital_deployed_pct":median(dep) if dep else 0,
                "average_actual_risk_pct":sum(float(t.get("actual_risk_pct",0)) for t in trades)/len(trades) if trades else 0,
                "median_actual_risk_pct":median([float(t.get("actual_risk_pct",0)) for t in trades]) if trades else 0,
                "risk_budget_utilization":sum(util)/len(util) if util else 0,
                "median_risk_budget_utilization":median(util) if util else 0,
                "zero_quantity_signals":raw["rejections"].get("ZERO_QUANTITY_CAPITAL_CONSTRAINT",0),
                "cash_blocked_signals":raw["rejections"].get("INSUFFICIENT_CASH",0),"rejections":raw["rejections"]}
            sensitivity.append(row)
            if capital == capitals[0]:
                trade_sets[mode]=trades; raw_runs[mode]=raw
                headline[mode]={"portfolio_metrics":metrics,"cost_breakdown":costs,"frequency":_frequency(trades),
                    "rejections":raw["rejections"],"initial_capital":capital,"risk_fraction":risk,
                    "max_positions":max_pos,"preferred_deployment_pct":deploy,
                    "average_capital_deployed_pct":sum(float(t.get("capital_deployed_pct",0)) for t in trades)/len(trades) if trades else 0,
                    "median_capital_deployed_pct":median([float(t.get("capital_deployed_pct",0)) for t in trades]) if trades else 0,
                    "average_actual_risk_pct":sum(float(t.get("actual_risk_pct",0)) for t in trades)/len(trades) if trades else 0,
                    "median_actual_risk_pct":median([float(t.get("actual_risk_pct",0)) for t in trades]) if trades else 0,
                    "average_risk_budget_utilization":sum(float(t.get("risk_budget_utilization",0)) for t in trades)/len(trades) if trades else 0,
                    "median_risk_budget_utilization":median([float(t.get("risk_budget_utilization",0)) for t in trades]) if trades else 0}
    tier_names=("TIER_A_ROBUST","TIER_B_VALID","TIER_C_WEAK")
    tiers_by_mode={mode:[_tier_metrics(signals,trade_sets[mode],tier) for tier in tier_names]
                   for mode in trade_sets}
    tier_rows=[{"mode":mode,**row} for mode,rows in tiers_by_mode.items() for row in rows]
    hold_delta = _support_hold_delta(old_rows, new_rows)
    before_ready=sum(r.get("state")=="ENTRY_READY_EVENT" for r in old_rows); after_ready=len(signals)
    summary={"strategy_version":"top_down_swing_v1","iteration":"H1_ROLE_ELIGIBILITY_AND_PATIENT_SNIPER_V1",
        "period":{"start":START.isoformat(),"end":END.isoformat()},"2025_2026_loaded":False,
        "old_ready_events":before_ready,"corrected_ready_events":after_ready,"ready_delta":after_ready-before_ready,
        "support_hold_before":hold_delta["before"],"support_hold_after":hold_delta["after"],
        "support_hold_removed":hold_delta["removed"],"support_hold_added":hold_delta["added"],
        "support_hold_net_delta":hold_delta["net_delta"],
        "support_hold_reason_diagnostics":hold_delta["causes"],"support_hold_reason_counts_reconcile":hold_delta["cause_counts_reconcile"],
        "tier_counts":{t:sum(s["tier"]==t for s in signals) for t in ("TIER_A_ROBUST","TIER_B_VALID","TIER_C_WEAK","REJECTED")},
        "hourly_prefix_sha256":data_hash,"trace":str(new_trace),"replay_report":str(corrected["report_path"]),
        "tier_rules":{"REJECTED":"invalid stop-entry-target or planned RR below configured minimum",
          "TIER_A_ROBUST":"actionable structural location, >= minimum RR, recognized H1 archetype, meaningful structural target, no explicit hostile-volume flag, plus causal volume, weekly HH/HL, role flip, HL->HH, or meaningful support evidence",
          "TIER_B_VALID":"clean location, valid H1 archetype, >= minimum RR; lacks Tier A confluence; fallback target remains at most B",
          "TIER_C_WEAK":"other technically ready candidates with marginal location/confirmation or hostile evidence",
          "uses_future_outcomes":False,"hostile_volume_limitation":"Frozen hourly confirmation features do not expose a dedicated high-volume bearish acceptance-below-support flag; NOT_AVAILABLE is not treated as verified absence."}}
    output_dir.mkdir(parents=True,exist_ok=True)
    _write(output_dir/"capital_sensitivity.csv",sensitivity); _write(output_dir/"quality_tier_breakdown.csv",tier_rows)
    _write(output_dir/"trade_frequency.csv",[{"mode":k,**v["frequency"]} for k,v in headline.items()])
    _write(output_dir/"cost_breakdown.csv",[{"mode":k,**v["cost_breakdown"]} for k,v in headline.items()])
    # Candidate-level confirmation evidence for all corrected executable signals.
    diagnostics=[]
    for s in signals:
        f=s["result"].get("confirmation_features") or {}; q=s["quality"]
        zs=f.get("held_support_zone_eligibility") or f.get("support_zone_eligibility") or f.get("involved_zones") or []
        diagnostics.append({"setup_instance_id":s["setup_instance_id"],"signal_date":s.get("setup_signal_date"),
            "security_id":s["security_id"],"symbol":s.get("symbol"),"setup_type":s.get("setup_type"),
            "confirmation_type":q["confirmation_type"],"tier":s["tier"],"zone_role":q["role_at_confirmation"],
            "zone_timeframe":",".join(sorted({str(z.get("timeframe","HOURLY")) for z in zs})),
            "zone_strength":q["zone_strength"],"touch_episodes":max((int(z.get("touch_episodes",z.get("touch_episode_count",0))) for z in zs),default=0),
            "reaction_strength_atr":max((_number(z.get("mean_reaction_atr")) or 0 for z in zs),default=0),
            "zone_recency_date":max((str(z.get("last_update") or "") for z in zs),default=""),
            "higher_low":f.get("higher_low"),"higher_high":f.get("higher_high"),
            "higher_low_to_higher_high":f.get("higher_low_to_higher_high"),"failed_downside_attempts":f.get("failed_downside_attempts"),
            "consolidation":f.get("consolidation"),"bullish_confirmation_close":f.get("bullish_close"),
            "volume_support":q["volume_support"],"volume_support_reasons":q["volume_support_reasons"],
            "hostile_bearish_volume_evidence":q["hostile_volume_status"],
            "volume_features":q["volume_features"],
            "target_timeframe":s["plan"].get("target_timeframe"),"target_strength":s["plan"].get("target_zone_strength"),
            "structural_rr":q["structural_rr"],"volume_support_reasons":q["volume_support_reasons"],
            "weekly_selection_path":s.get("weekly_selection_path"),"weekly_selection_components":s.get("weekly_selection_components"),
            "weekly_structure":s.get("weekly_structure"),"daily_structure":s.get("daily_structure"),
            "nifty_regime":s.get("nifty_regime"),"structural_location_category":s.get("structural_location_category"),
            "structural_position_fraction":s.get("structural_range_position"),"target_zone_id":s["plan"].get("target_zone_id"),
            "overhead_friction_present":s["plan"].get("overhead_friction_present"),"ema_context":s["plan"].get("ema_context"),
            "structural_obstacles":s["plan"].get("obstacles"),
            "tier_reasons":q["tier_reasons"]})
    _write(output_dir/"confirmation_diagnostics.csv",diagnostics)
    for mode,filename in (("corrected_baseline","corrected_baseline_trades.csv"),("patient_sniper_1","patient_sniper_1_trades.csv"),("patient_sniper_2","patient_sniper_2_trades.csv")):
        _write(output_dir/filename,trade_sets[mode]); _write(output_dir/(mode+"_equity_curve.csv"),raw_runs[mode]["equity_curve"])
        payload={**headline[mode],"strategy_mode":mode,"tier_outcomes":tiers_by_mode[mode],"2025_2026_loaded":False}
        (output_dir/(mode+"_summary.json")).write_text(json.dumps(payload,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")
    audit=sorted(diagnostics,key=lambda r:(r["tier"],r["signal_date"] or "",r["security_id"]))[:250]
    _write(output_dir/"manual_audit_sample.csv",audit)
    (output_dir/"corrected_signal_summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")
    result={"signal_summary":summary,"modes":headline,"capital_sensitivity":sensitivity,"tiers_by_mode":tiers_by_mode,"output_directory":str(output_dir)}
    print(json.dumps(result,indent=2,default=str,allow_nan=False)); return result


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--trace",type=Path,default=OLD_TRACE)
    p.add_argument("--replay-report",type=Path,default=OLD_REPORT); p.add_argument("--output-directory",type=Path,default=OUTPUT)
    p.add_argument("--data-root",type=Path,default=DATA_ROOT); p.add_argument("--daily-manifest",type=Path,default=MANIFEST_PATH)
    p.add_argument("--hourly-manifest",type=Path,default=HOURLY_MANIFEST); p.add_argument("--config",type=Path,default=CONFIG)
    p.add_argument("--reuse-corrected",action="store_true",help="reuse a completed corrected replay in the output directory")
    a=p.parse_args()
    try: run(trace=a.trace,replay_report=a.replay_report,output_dir=a.output_directory,data_root=a.data_root,
             daily_manifest=a.daily_manifest,hourly_manifest=a.hourly_manifest,config_path=a.config,
             reuse_corrected=a.reuse_corrected)
    except (ValueError,OSError,BacktestDataError) as e: p.error(str(e))


if __name__=="__main__": main()
