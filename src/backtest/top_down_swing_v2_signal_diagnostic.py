"""Independent signal-level outcome diagnostics for the frozen V2 replay."""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path
from statistics import mean, median
from typing import Any, Mapping, Sequence

import yaml

from src.backtest.data import load_hourly_dataset_through
from src.backtest.models import CostModel
from src.backtest.top_down_swing_portfolio import _cost_config, _run_portfolio
from src.backtest.top_down_swing_v2_portfolio import _read_signals
from src.backtest.top_down_swing_v2_replay import END, OUTPUT, START, _csv
from src.configuration import PROJECT_ROOT
from src.data.hourly import HOURLY_MANIFEST
from src.data.market_data import DATA_ROOT, MANIFEST_PATH
from src.data.universe.provider import PIT_TRADABLE

OUT = OUTPUT / "signal_level_diagnostic"
SIGNALS = OUTPUT / "v2_entry_ready_events.csv"
CONFIG = PROJECT_ROOT / "config" / "top_down_swing_v2.yaml"
IDENTITY_EXCLUSIONS = PROJECT_ROOT / "data" / "processed" / "hourly" / "price_integrity" / "invalid_hourly_identities.csv"
INITIAL_CAPITAL = 100_000.0
RISK_FRACTION = 0.01
RANDOM_SEED = 20260927
FROZEN_EXECUTION_FIELDS = (
    "setup_event_id", "setup_instance_id", "symbol", "entry_timestamp", "entry_price",
    "entry_raw_open", "stop_price", "target_price", "quantity", "exit_timestamp",
    "exit_reason", "exit_price", "raw_exit_price", "gap_handling", "holding_sessions",
)


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _json(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return None
    return value


def _finite_values(rows: Sequence[Mapping[str, Any]], key: str) -> list[float]:
    return [float(row[key]) for row in rows if row.get(key) not in (None, "")]


def _pf(values: Sequence[float]) -> float | None:
    gains = sum(value for value in values if value > 0)
    losses = abs(sum(value for value in values if value < 0))
    return gains / losses if losses else (None if gains else 0.0)


def _distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "p25": None, "median": None, "mean": None, "p75": None, "max": None}
    ordered = sorted(values)
    def percentile(p: float) -> float:
        index = (len(ordered) - 1) * p
        lo, hi = int(index), min(int(index) + 1, len(ordered) - 1)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)
    return {"count": len(values), "min": ordered[0], "p25": percentile(.25), "median": median(ordered),
            "mean": mean(ordered), "p75": percentile(.75), "max": ordered[-1]}


def accounting_views(canonical: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build cost and raw views from one already-sized, already-executed trade.

    The raw view removes explicit turnover charges only. Execution fills (including
    slippage), quantity, and exit decisions are copied from the canonical trade.
    """
    current = dict(canonical)
    raw = dict(canonical)
    is_closed = canonical.get("status") == "CLOSED"
    gross = float(canonical["gross_pnl"]) if canonical.get("gross_pnl") not in (None, "") else None
    explicit_costs = float(canonical.get("transaction_costs", 0.0) or 0.0)
    if is_closed:
        if gross is None:
            raise ValueError(f"Closed canonical trade lacks gross P&L: {canonical.get('setup_instance_id')}")
        current_net = float(canonical["net_pnl"])
        if abs(current_net - (gross - explicit_costs)) > 1e-7:
            raise ValueError(f"Canonical cost reconciliation failed for {canonical.get('setup_instance_id')}")
        raw["zero_cost_net_pnl"] = gross
        raw["net_pnl"] = gross
    else:
        if canonical.get("status") != "OPEN":
            raise ValueError(f"Unexpected canonical trade status: {canonical.get('status')}")
        current["net_pnl"] = None
        raw["zero_cost_net_pnl"] = None
        raw["net_pnl"] = None
        raw["zero_cost_unrealized_gross_pnl"] = canonical.get("unrealized_gross_pnl")
        if canonical.get("unrealized_gross_pnl") != raw["zero_cost_unrealized_gross_pnl"]:
            raise ValueError(f"Open unrealized P&L changed for {canonical.get('setup_instance_id')}")
    raw["transaction_costs"] = 0.0
    raw["entry_costs"] = 0.0
    raw["exit_costs"] = 0.0
    raw["entry_cost_breakdown"] = {}
    raw["exit_cost_breakdown"] = {}
    raw["zero_cost_transaction_costs"] = 0.0
    raw["zero_cost_entry_costs"] = 0.0
    raw["zero_cost_exit_costs"] = 0.0
    raw["zero_cost_entry_cost_breakdown"] = {}
    raw["zero_cost_exit_cost_breakdown"] = {}
    raw["zero_cost_scenario"] = "RAW_ZERO_EXPLICIT_COSTS_SAME_FROZEN_EXECUTION"
    current["current_cost_net_pnl"] = current.get("net_pnl")
    current["current_cost_transaction_costs"] = explicit_costs
    current["current_cost_scenario"] = canonical.get("transaction_cost_scenario")
    for field in FROZEN_EXECUTION_FIELDS:
        if current.get(field) != raw.get(field):
            raise ValueError(f"Frozen execution field changed in accounting view: {field}")
    if current["gross_pnl"] != raw["gross_pnl"]:
        raise ValueError("Gross P&L changed across accounting views.")
    if is_closed and abs(float(raw["zero_cost_net_pnl"]) - float(current["gross_pnl"])) > 1e-9:
        raise ValueError("Raw-zero explicit-cost net P&L must equal canonical gross P&L.")
    if not is_closed and current.get("unrealized_gross_pnl") != raw.get("zero_cost_unrealized_gross_pnl"):
        raise ValueError("Open unrealized P&L changed across accounting views.")
    return current, raw


def _metrics(trades: Sequence[Mapping[str, Any]], *, zero_cost: bool = False) -> dict[str, Any]:
    pnl_key = "zero_cost_net_pnl" if zero_cost else "net_pnl"
    closed = [row for row in trades if row.get("status") == "CLOSED"]
    net = [float(row[pnl_key]) for row in closed]
    gross = [float(row["gross_pnl"]) for row in closed]
    realized_r = _finite_values(closed, "gross_R")
    planned_r = _finite_values(trades, "planned_rr")
    holding = _finite_values(closed, "holding_sessions")
    gross_abs = sum(abs(value) for value in gross)
    costs = 0.0 if zero_cost else sum(float(row.get("transaction_costs", 0) or 0) for row in trades)
    closed_costs = 0.0 if zero_cost else sum(float(row.get("transaction_costs", 0) or 0)
                                             for row in closed)
    open_costs = costs - closed_costs
    cost_components: Counter[str] = Counter()
    if not zero_cost:
        for row in trades:
            for side in ("entry_cost_breakdown", "exit_cost_breakdown"):
                for component, value in (row.get(side) or {}).items():
                    if component != "total":
                        cost_components[component] += float(value or 0)
    open_unrealized = sum(float(row.get("unrealized_gross_pnl", 0) or 0)
                          for row in trades if row.get("status") == "OPEN")
    reasons = Counter(str(row.get("exit_reason") or "OPEN") for row in trades)
    return {
        "signals": len(trades), "closed": len(closed), "open_at_period_end": len(trades) - len(closed),
        "wins": sum(value > 0 for value in net), "losses": sum(value < 0 for value in net),
        "win_rate": sum(value > 0 for value in net) / len(closed) if closed else None,
        "loss_rate": sum(value < 0 for value in net) / len(closed) if closed else None,
        "gross_profit_factor": _pf(gross), "net_profit_factor": _pf(net),
        "gross_expectancy_per_trade": mean(gross) if gross else None,
        "net_expectancy_per_trade": mean(net) if net else None,
        "gross_pnl": sum(gross), "net_pnl": sum(net),
        "open_unrealized_gross_pnl": open_unrealized,
        "gross_return_on_independent_100k_risk_budget": sum(gross) / INITIAL_CAPITAL,
        "net_return_on_independent_100k_risk_budget": sum(net) / INITIAL_CAPITAL,
        "mean_realized_R": mean(realized_r) if realized_r else None,
        "median_realized_R": median(realized_r) if realized_r else None,
        "mean_planned_R": mean(planned_r) if planned_r else None,
        "median_planned_R": median(planned_r) if planned_r else None,
        "mean_holding_sessions": mean(holding) if holding else None,
        "median_holding_sessions": median(holding) if holding else None,
        "mae_pct_distribution": _distribution(_finite_values(trades, "mae_pct")),
        "mfe_pct_distribution": _distribution(_finite_values(trades, "mfe_pct")),
        "target_hits": reasons["TARGET"], "stop_hits": reasons["STOP"] + reasons["AMBIGUOUS_INTRABAR_STOP_FIRST"],
        "ambiguous_stop_first": reasons["AMBIGUOUS_INTRABAR_STOP_FIRST"],
        "gap_target_hits": reasons["GAP_TARGET"], "gap_stop_hits": reasons["GAP_STOP"],
        "thesis_invalidation_exits": reasons["THESIS_INVALIDATION"], "exit_reasons": dict(sorted(reasons.items())),
        "transaction_costs": costs, "cost_as_pct_of_gross_absolute_pnl": costs / gross_abs if gross_abs else None,
        "closed_trade_costs": closed_costs, "open_position_entry_costs": open_costs,
        "cost_components": dict(sorted(cost_components.items())),
        "dp_charges": 0.0 if zero_cost else sum(float(((row.get("exit_cost_breakdown") or {}).get("dp_charge", 0)) or 0) for row in trades),
        "zero_cost": zero_cost,
    }


def _stratified_sample(trades: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rng = random.Random(RANDOM_SEED)
    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in trades:
        if row.get("setup_family") not in {"SUPPORT_REACTION", "BREAKOUT_RETEST", "CONSOLIDATION_BREAKOUT"}:
            continue
        year = str(row.get("signal_date", ""))[:4]
        result_bucket = "WIN" if float(row.get("net_pnl", 0) or 0) > 0 else "NON_WIN"
        groups[(str(row.get("setup_family")), year, result_bucket)].append(row)
    for values in groups.values():
        rng.shuffle(values)
    output: list[dict[str, Any]] = []
    for family in ("SUPPORT_REACTION", "BREAKOUT_RETEST", "CONSOLIDATION_BREAKOUT"):
        selected: list[Mapping[str, Any]] = []
        keys = sorted(key for key in groups if key[0] == family)
        # Rotate through year/outcome strata so a populated stratum cannot dominate.
        while len(selected) < min(5, sum(len(groups[key]) for key in keys)):
            moved = False
            for key in keys:
                if groups[key] and len(selected) < 5:
                    selected.append(groups[key].pop())
                    moved = True
            if not moved:
                break
        output.extend(dict(row) for row in selected)
    return output


def _rejection_artifacts(output_dir: Path, trades: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    reject_path = OUTPUT / "v2_rejected_candidates.csv"
    trace_path = OUTPUT / "v2_signal_trace.csv"
    summary_path = OUTPUT / "v2_signal_summary.json"
    rejects = _rows(reject_path)
    trace = _rows(trace_path)
    replay_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rejects:
        grouped[str(row.get("reason_code") or "UNKNOWN")].append(row)
    funnel_rows: list[dict[str, Any]] = []
    for reason, occurrences in sorted(grouped.items()):
        unique = set()
        by_year: dict[str, dict[str, int]] = defaultdict(lambda: {"occurrences": 0, "unique_interactions": 0})
        symbols, families = Counter(), Counter()
        seen_by_year: dict[str, set[tuple[str, ...]]] = defaultdict(set)
        for row in occurrences:
            year = str(row.get("date", ""))[:4]
            boundary = row.get("required_zone_bound") or row.get("daily_zone_lower_bound") or row.get("zone_lower_bound") or ""
            key = (row.get("security_id", ""), row.get("setup_family", ""), row.get("date", ""), str(boundary))
            unique.add(key)
            seen_by_year[year].add(key)
            by_year[year]["occurrences"] += 1
            symbols[str(row.get("symbol") or "UNKNOWN")] += 1
            families[str(row.get("setup_family") or "UNKNOWN")] += 1
        for year, keys in seen_by_year.items():
            by_year[year]["unique_interactions"] = len(keys)
        funnel_rows.append({"reason_code": reason, "occurrences": len(occurrences),
                            "unique_interactions_approx": len(unique),
                            "top_symbols": json.dumps(symbols.most_common(10)),
                            "setup_families": json.dumps(families.most_common()),
                            "year_split": json.dumps(by_year, sort_keys=True)})
    _csv(output_dir / "rejection_funnel.csv", funnel_rows)
    family_candidates: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
    state_counts = Counter(row.get("state", "") for row in trace)
    weekly_rows = []
    for row in trace:
        if row.get("state") == "SETUP_ACTIVE" and row.get("setup_family"):
            family_candidates[str(row["setup_family"])].add((row.get("security_id", ""), row.get("date", ""), row.get("setup_event_id", "")))
        if row.get("state") == "WEEKLY_SELECTION_REJECTED":
            selector = _json(row.get("weekly_selection")) or {}
            try:
                reasons = _json(row.get("reason_codes")) or []
            except TypeError:
                reasons = []
            weekly_rows.append({"date": row.get("date"), "year": str(row.get("date", ""))[:4],
                                "security_id": row.get("security_id"), "symbol": row.get("symbol"),
                                "selector_status": selector.get("status"), "selector_reason": selector.get("reason"),
                                "reason_codes": reasons})
    _csv(output_dir / "weekly_selector_rejected_observations.csv", weekly_rows)
    _csv(output_dir / "setup_family_candidate_counts.csv", [
        {"setup_family": family, "setup_active_security_date_event_count": len(keys),
         "entry_ready_count": sum(row.get("setup_family") == family for row in trades)}
        for family, keys in sorted(family_candidates.items())])
    selector_years = Counter(str(row["year"]) for row in weekly_rows)
    selector_symbols = Counter(str(row["symbol"]) for row in weekly_rows)
    selector_reasons = Counter(str(row["selector_reason"] or ",".join(row["reason_codes"]) or "UNKNOWN") for row in weekly_rows)
    eich = [row for row in weekly_rows if row.get("symbol") == "EICHERMOT" and row.get("date") == "2023-11-06"]
    weekly_diag = {
        "status": "OBSERVED_REJECTIONS_ONLY_COUNTERFACTUAL_NOT_AVAILABLE_FROM_SAVED_ARTIFACTS",
        "weekly_selector_rejected_observations": len(weekly_rows),
        "unique_security_dates": len({(row.get("security_id"), row.get("date")) for row in weekly_rows}),
        "by_year": dict(sorted(selector_years.items())),
        "top_symbols": selector_symbols.most_common(20),
        "reasons": dict(sorted(selector_reasons.items())),
        "otherwise_structurally_valid_setup_count": None,
        "eventual_hypothetical_outcomes_without_weekly_gate": None,
        "reason_unavailable": "The saved causal replay exits at the weekly selector before daily structure/H1 setup evaluation for rejected dates. No no-gate candidate/outcome artifact exists. Generating one requires another signal-generation replay, which this task forbids.",
        "eichermot_2023_11_06": eich,
    }
    (output_dir / "weekly_selector_diagnostic.json").write_text(
        json.dumps(weekly_diag, indent=2, sort_keys=True, ensure_ascii=True, default=str) + "\n", encoding="utf-8")
    funnel_summary = {
        "stage_occurrences": {
            "daily_observations": replay_summary.get("counts", {}).get("daily_observations"),
            "weekly_selector_rejected": replay_summary.get("counts", {}).get("WEEKLY_SELECTION_REJECTED"),
            "watchlist_candidates": replay_summary.get("counts", {}).get("watchlist_candidates"),
            "setup_active_trace_rows": state_counts.get("SETUP_ACTIVE", 0),
            "h1_confirmation_count": replay_summary.get("counts", {}).get("hourly_confirmation_count"),
            "entry_ready": len(trades),
        },
        "rejection_reason_count": len(funnel_rows),
        "rejection_occurrences": sum(row["occurrences"] for row in funnel_rows),
        "unique_interactions_are_approximated_by_security_family_date_boundary": True,
        "weekly_selector_counterfactual": weekly_diag,
    }
    (output_dir / "rejection_funnel_summary.json").write_text(
        json.dumps(funnel_summary, indent=2, sort_keys=True, ensure_ascii=True, default=str) + "\n", encoding="utf-8")
    return funnel_summary


def run(*, signals_path: Path = SIGNALS, output_dir: Path = OUT, config_path: Path = CONFIG,
        data_root: Path = DATA_ROOT, daily_manifest: Path = MANIFEST_PATH,
        hourly_manifest: Path = HOURLY_MANIFEST) -> dict[str, Any]:
    raw_rows = _rows(signals_path)
    signals = _read_signals(signals_path)
    ids = [str(row["setup_instance_id"]) for row in signals]
    duplicates = sorted({setup_id for setup_id in ids if ids.count(setup_id) > 1})
    if duplicates:
        raise ValueError(f"Duplicate setup_instance_id values in ENTRY_READY source: {duplicates}")
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    strategy_version = str(cfg.get("version") or "")
    if strategy_version != "top_down_swing_v2_daily_structure":
        raise ValueError(f"Signal diagnostic requires the frozen V2 config version; got {strategy_version!r}.")
    bars, hourly_hash = load_hourly_dataset_through(
        universe_id=PIT_TRADABLE, start_date=date(2022, 10, 1), end_date=END,
        data_root=data_root, manifest_path=hourly_manifest, daily_manifest_path=daily_manifest,
        identity_exclusions_path=IDENTITY_EXCLUSIONS)
    current_costs = _cost_config(PROJECT_ROOT / "config" / "settings.yaml")
    trades: list[dict[str, Any]] = []
    for signal in signals:
        sid = str(signal["security_id"])
        entry_stamp = signal["executable_timestamp"]
        security_bars = [bar for bar in bars.get(sid, [])
                         if datetime.fromisoformat(str(bar["date"])) >= entry_stamp]
        if not security_bars or datetime.fromisoformat(str(security_bars[0]["date"])) != entry_stamp:
            raise ValueError(f"ENTRY_READY signal lacks exact next H1 execution bar: {signal['setup_instance_id']}")
        args = dict(initial_capital=INITIAL_CAPITAL, risk_fraction=RISK_FRACTION,
                    max_positions=1, period_start=START)
        current = _run_portfolio([signal], {sid: security_bars}, current_costs, **args)["trades"]
        if len(current) != 1:
            raise ValueError(f"Independent signal did not enter exactly once: {signal['setup_instance_id']}")
        current[0]["setup_event_id"] = signal.get("setup_event_id")
        live, no_cost = accounting_views(current[0])
        # Resolve the unslipped provider exit price for an auditable raw-price field.
        exit_stamp = live.get("exit_timestamp") or live.get("mark_timestamp")
        if exit_stamp:
            exit_bar = next((bar for bar in security_bars if str(bar["date"]) == exit_stamp), None)
            if exit_bar is None:
                raise ValueError(f"Exit timestamp is absent from hourly data: {signal['setup_instance_id']}")
            if live.get("status") == "OPEN":
                raw_exit = float(exit_bar["close"])
            elif str(live.get("exit_reason", "")).startswith("GAP_"):
                raw_exit = float(exit_bar["open"])
            elif live.get("exit_reason") in {"STOP", "AMBIGUOUS_INTRABAR_STOP_FIRST"}:
                raw_exit = float(live["stop_price"])
            elif live.get("exit_reason") == "TARGET":
                raw_exit = float(live["target_price"])
            else:
                raise ValueError(f"Unrecognized exit reason for raw price: {live.get('exit_reason')}")
            live["raw_exit_price"] = no_cost["raw_exit_price"] = raw_exit
            live["gap_handling"] = no_cost["gap_handling"] = str(live.get("exit_reason", "")).startswith("GAP_")
            # Re-check after enriching the canonical trade with raw exit details.
            for field in FROZEN_EXECUTION_FIELDS:
                if live.get(field) != no_cost.get(field):
                    raise ValueError(f"Execution invariant failed for {signal['setup_instance_id']}: {field}")
        else:
            live["raw_exit_price"] = no_cost["raw_exit_price"] = None
            live["gap_handling"] = no_cost["gap_handling"] = False
        plan = signal["plan"]
        result = signal["result"]
        live["zero_cost_net_pnl"] = no_cost.get("zero_cost_net_pnl")
        live["zero_cost_transaction_costs"] = no_cost.get("zero_cost_transaction_costs")
        live["raw_zero_cost_gross_pnl"] = no_cost.get("gross_pnl")
        live["gross_pnl_current_cost_view"] = live.get("gross_pnl")
        live["gross_pnl_raw_zero_cost_view"] = no_cost.get("gross_pnl")
        live["execution_invariants_passed"] = all(
            live.get(field) == no_cost.get(field) for field in FROZEN_EXECUTION_FIELDS
        )
        live["explicit_cost_difference_reconciles"] = (
            live.get("status") != "CLOSED"
            or abs(float(live["gross_pnl"]) - float(live["net_pnl"]) - float(live.get("transaction_costs", 0))) < 1e-7
        )
        live["setup_family"] = signal.get("setup_family") or signal.get("setup_type")
        # The shared legacy portfolio executor stamps its rows as V1. The V2
        # config is authoritative for provenance; this changes metadata only.
        live["strategy_version"] = strategy_version
        live["setup_type"] = signal.get("setup_type")
        live["setup_instance_id"] = signal["setup_instance_id"]
        live["setup_event_id"] = signal.get("setup_event_id")
        live["signal_date"] = signal.get("setup_signal_date")
        live["planned_rr"] = plan.get("planned_reward_r")
        live["target_timeframe"] = plan.get("target_timeframe")
        live["target_classification"] = plan.get("target_classification")
        live["intermediate_friction_count"] = len(plan.get("intermediate_daily_friction") or []) + len(plan.get("friction") or [])
        role_zone = signal.get("support") if isinstance(signal.get("support"), Mapping) else None
        if role_zone is None:
            resistance_zones = signal.get("resistance")
            if isinstance(resistance_zones, Sequence) and not isinstance(resistance_zones, (str, bytes)):
                role_zone = next((zone for zone in resistance_zones if isinstance(zone, Mapping)), None)
            elif isinstance(resistance_zones, Mapping):
                role_zone = resistance_zones
        live["zone_role"] = role_zone.get("role_state") if isinstance(role_zone, Mapping) else None
        live["nifty_regime"] = signal.get("nifty_regime")
        live["weekly_structure"] = _json(signal.get("weekly_structure"))
        weekly = live["weekly_structure"] if isinstance(live["weekly_structure"], Mapping) else {}
        if weekly.get("higher_high") is True and weekly.get("higher_low") is True:
            live["weekly_context"] = "BULLISH_HH_HL"
        elif weekly.get("higher_high") is False and weekly.get("higher_low") is False:
            live["weekly_context"] = "NON_BULLISH_LH_LL"
        elif weekly.get("higher_high") is not None or weekly.get("higher_low") is not None:
            live["weekly_context"] = "MIXED_OR_PARTIAL"
        else:
            live["weekly_context"] = "UNAVAILABLE"
        live["ema_context"] = plan.get("ema_context")
        live["retest_subtype"] = signal.get("retest_subtype") or (result.get("breakout_retest_provenance", {}).get("retest", {}).get("retest_subtype") if isinstance(result.get("breakout_retest_provenance"), Mapping) else None)
        live["confirmation_timestamp"] = signal["confirmation_timestamp"].isoformat()
        live["setup_signal_date"] = signal.get("setup_signal_date")
        live["confirmation_type"] = result.get("confirmation_type")
        live["volume_confirmation_required"] = plan.get("volume_confirmation_required")
        live["volume_confirmation_passed"] = plan.get("volume_confirmation_passed")
        live["volume_confirmation_type"] = plan.get("volume_confirmation_type")
        live["entry"] = plan.get("entry")
        live["stop"] = plan.get("stop")
        live["target"] = plan.get("target")
        live["structural_map"] = signal.get("daily_zones")
        live["support_zone"] = signal.get("support")
        live["resistance_zones"] = signal.get("resistance")
        live["confirmation_features"] = result.get("confirmation_features")
        live["setup_lifecycle"] = signal.get("setup_lifecycle")
        trades.append(live)

    family_names = ("SUPPORT_REACTION", "BREAKOUT_RETEST", "CONSOLIDATION_BREAKOUT",
                    "DIRECT_BREAKOUT", "RECLAIM", "FAILED_BREAKDOWN_RECLAIM",
                    "HL_HH_CONFIRMATION", "FALSE_BREAKOUT_RECLAIM")
    group_specs = {"year": lambda row: str(row.get("signal_date", ""))[:4],
                   "setup_family": lambda row: str(row.get("setup_family") or "UNKNOWN"),
                   "target_type": lambda row: str(row.get("target_timeframe") or "UNAVAILABLE"),
                   "regime": lambda row: str(row.get("nifty_regime") or "UNAVAILABLE"),
                   "weekly_context": lambda row: str(row.get("weekly_context") or "UNAVAILABLE"),
                   "zone_role": lambda row: str(row.get("zone_role") or "UNAVAILABLE"),
                   "retest_subtype": lambda row: str(row.get("retest_subtype") or "NOT_APPLICABLE"),
                   "target_classification": lambda row: str(row.get("target_classification") or "UNAVAILABLE"),
                   "ema_context": lambda row: json.dumps(row.get("ema_context"), sort_keys=True, default=str) if row.get("ema_context") is not None else "UNAVAILABLE",
                   "intermediate_friction_count": lambda row: str(row.get("intermediate_friction_count", "UNAVAILABLE")),
                   "confirmation_type": lambda row: str(row.get("confirmation_type") or "UNAVAILABLE")}
    breakdowns: dict[str, dict[str, Any]] = {}
    for label, key_fn in group_specs.items():
        buckets: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for trade in trades:
            buckets[key_fn(trade)].append(trade)
        if label == "setup_family":
            for family in family_names:
                buckets.setdefault(family, [])
        breakdowns[label] = {key: {"count": len(rows), "current_costs": _metrics(rows), "zero_cost": _metrics(rows, zero_cost=True)}
                             for key, rows in sorted(buckets.items())}

    repeat_groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in trades:
        repeat_groups[(str(row["security_id"]), str(row["setup_family"]), str(row.get("setup_event_id")))].append(row)
    repeated = []
    for key, members in sorted(repeat_groups.items()):
        if len(members) < 2:
            continue
        ordered = sorted(members, key=lambda row: str(row.get("signal_date")))
        for order, member in enumerate(ordered, start=1):
            member["parent_event_signal_order"] = order
            member["later_signal_after_same_parent_event"] = order > 1
        repeated.append({"security_id": key[0], "setup_family": key[1], "parent_setup_event_id": key[2],
                         "entry_ready_count": len(members), "signal_dates": [row.get("signal_date") for row in ordered],
                         "setup_instance_ids": [row["setup_instance_id"] for row in ordered],
                         "entry_timestamps": [row.get("entry_timestamp") for row in ordered],
                         "classification": "LATER_FRESH_EPISODES_RETAINED; inspect deterministic reset/revisit evidence"})
    repeated_keys = {(item["security_id"], item["setup_family"], item["parent_setup_event_id"]) for item in repeated}
    for trade in trades:
        key = (str(trade["security_id"]), str(trade["setup_family"]), str(trade.get("setup_event_id")))
        if key not in repeated_keys:
            trade["parent_event_signal_order"] = 1
            trade["later_signal_after_same_parent_event"] = False
    raw_ready = [{**row, "thesis_counting_status": "UNIQUE_SETUP_INSTANCE"} for row in raw_rows]
    sample = _stratified_sample(trades)
    wrong_versions = sorted({str(row.get("strategy_version")) for row in trades
                             if row.get("strategy_version") != strategy_version})
    if wrong_versions:
        raise ValueError(f"V2 outcome ledger contains wrong strategy_version values: {wrong_versions}")
    output_dir.mkdir(parents=True, exist_ok=True)
    _csv(output_dir / "raw_entry_ready_signals.csv", raw_ready)
    _csv(output_dir / "unique_thesis_signals.csv", raw_ready)
    _csv(output_dir / "independent_trade_outcomes.csv", trades)
    _csv(output_dir / "random_manual_audit_sample.csv", sample)
    context_rows = []
    for field, buckets in breakdowns.items():
        for key, value in buckets.items():
            context_rows.append({"dimension": field, "bucket": key, "signals": value["count"],
                                 "closed": value["current_costs"]["closed"],
                                 "win_rate": value["current_costs"]["win_rate"],
                                 "gross_pf": value["current_costs"]["gross_profit_factor"],
                                 "net_pf": value["current_costs"]["net_profit_factor"],
                                 "gross_pnl": value["current_costs"]["gross_pnl"],
                                 "net_pnl": value["current_costs"]["net_pnl"]})
    _csv(output_dir / "structural_context_breakdown.csv", context_rows)
    rejection_summary = _rejection_artifacts(output_dir, trades)
    summary = {
        "strategy_version": cfg.get("version"), "research_period": {"start": START.isoformat(), "end": END.isoformat()},
        "2025_2026_loaded": False, "universe": PIT_TRADABLE, "hourly_prefix_sha256": hourly_hash,
        "sizing_convention": "Each signal is independently sized from a reset 100000 INR equity balance at 1% risk; no portfolio slots/cash occupancy; same quantity and execution compared in raw-zero-cost and current-cost scenarios.",
        "current_cost_scenario": current_costs.scenario,
        "raw_zero_cost_scenario": "RAW_ZERO_EXPLICIT_COSTS_SAME_FROZEN_EXECUTION",
        "canonical_execution_invariant_passed": True,
        "raw_entry_ready_rows": len(raw_rows), "unique_setup_instance_theses": len({r["setup_instance_id"] for r in raw_ready}),
        "duplicate_rows_suppressed": 0, "repeated_parent_setup_event_groups": repeated,
        "repeated_parent_event_later_fresh_episode_count": sum(max(0, group["entry_ready_count"] - 1) for group in repeated),
        "all_signals_current_costs": _metrics(trades), "all_signals_zero_cost_same_executions": _metrics(trades, zero_cost=True),
        "breakdowns": breakdowns, "manual_audit_sample_seed": RANDOM_SEED,
        "rejection_funnel": rejection_summary,
        "manual_audit_sample_count": len(sample), "manual_audit_sample_ids": [r["setup_instance_id"] for r in sample],
        "all_129_execution_invariants_passed": all(bool(row.get("execution_invariants_passed")) for row in trades),
        "all_closed_cost_reconciliations_passed": all(bool(row.get("explicit_cost_difference_reconciles")) for row in trades),
        "portfolio_slots_ignored": True, "strategy_logic_modified": False,
        "important_limitations": ["This is a signal-level independent-trade diagnostic, not an account equity curve or compounded portfolio.",
          "Thesis invalidation exits are not implemented by the current shared V2 stop/target portfolio execution engine.",
          "Setup family candidate counts and weekly-selector no-gate counterfactual require separate replay instrumentation."],
    }
    (output_dir / "signal_level_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=True, default=str) + "\n", encoding="utf-8")
    (output_dir / "DIAGNOSTIC_RESOLVED.json").write_text(json.dumps({
        "status": "COMPLETED_WITH_CANONICAL_SAME_EXECUTION_ACCOUNTING",
        "historical_stop_artifact": "DIAGNOSTIC_STOPPED.json",
        "strategy_version": strategy_version,
        "signals": len(trades),
        "closed": summary["all_signals_current_costs"]["closed"],
        "open": summary["all_signals_current_costs"]["open_at_period_end"],
        "all_execution_invariants_passed": summary["all_129_execution_invariants_passed"],
        "all_closed_cost_reconciliations_passed": summary["all_closed_cost_reconciliations_passed"],
        "signal_generation_replay_rerun": False,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"signals": len(trades), "closed": summary["all_signals_current_costs"]["closed"],
                      "open": summary["all_signals_current_costs"]["open_at_period_end"],
                      "gross_pnl": summary["all_signals_current_costs"]["gross_pnl"],
                      "net_pnl": summary["all_signals_current_costs"]["net_pnl"], "output": str(output_dir)}, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signals", type=Path, default=SIGNALS)
    parser.add_argument("--output-directory", type=Path, default=OUT)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--daily-manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--hourly-manifest", type=Path, default=HOURLY_MANIFEST)
    args = parser.parse_args()
    run(signals_path=args.signals, output_dir=args.output_directory, config_path=args.config,
        data_root=args.data_root, daily_manifest=args.daily_manifest, hourly_manifest=args.hourly_manifest)


if __name__ == "__main__":
    main()
