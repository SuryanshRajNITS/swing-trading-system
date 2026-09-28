"""Frozen-signal V2 portfolio construction and causal trade management study.

This module consumes the saved V2 ENTRY_READY artifact. It never invokes signal
generation. The current-cost portfolio is canonical; the zero-explicit-cost view
is a bookkeeping view over the same frozen fills, quantities, and exit events.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, time
from pathlib import Path
from statistics import mean, median
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

import yaml

from src.backtest.data import load_hourly_dataset_through
from src.backtest.top_down_swing_portfolio import _cost_config
from src.backtest.top_down_swing_v2_portfolio import SIGNALS, _read_signals
from src.backtest.top_down_swing_v2_replay import END, OUTPUT, START, _csv
from src.configuration import PROJECT_ROOT
from src.data.hourly import HOURLY_MANIFEST
from src.data.market_data import DATA_ROOT, MANIFEST_PATH
from src.data.universe.provider import PIT_TRADABLE

TZ = ZoneInfo("Asia/Kolkata")
CONFIG = PROJECT_ROOT / "config" / "top_down_swing_v2_portfolio.yaml"
OUT = OUTPUT / "portfolio_management"
SIGNAL_SUMMARY = OUTPUT / "v2_signal_summary.json"


@dataclass(frozen=True)
class FamilyEdge:
    closed_trades: int
    expectancy_r: float | None
    net_profit_factor: float | None
    classification: str
    multiplier: float
    net_r_stddev: float | None = None
    positive_net_r_fraction: float | None = None


def family_edge(closed_trades: Sequence[Mapping[str, Any]], *, enabled: bool = True,
                thresholds: Mapping[str, Any] | None = None) -> FamilyEdge:
    """Compute family sizing only from already-closed trades supplied by caller."""
    cfg = dict(thresholds or {})
    n = len(closed_trades)
    if not enabled or n < 10:
        rs = [float(t["net_R"]) for t in closed_trades]
        return FamilyEdge(n, None if not n else mean(rs),
                          None if not n else _profit_factor([float(t["net_pnl"]) for t in closed_trades]),
                          "INSUFFICIENT_HISTORY" if enabled else "FIXED_WEIGHT", 1.0,
                          _stddev(rs), sum(x > 0 for x in rs) / n if n else None)
    rs = [float(t["net_R"]) for t in closed_trades]
    pnl = [float(t["net_pnl"]) for t in closed_trades]
    exp_r, pf = mean(rs), _profit_factor(pnl)
    full = n >= int(cfg.get("full_confidence_closed_trades", 20))
    if exp_r > float(cfg.get("high_expectancy_r", 0.15)) and pf is not None and pf >= float(cfg.get("high_net_profit_factor", 1.25)):
        return FamilyEdge(n, exp_r, pf, "HIGH_CONFIDENCE_EDGE", 1.5 if full else 1.25,
                          _stddev(rs), sum(x > 0 for x in rs) / n)
    if exp_r < float(cfg.get("weak_expectancy_r", -0.10)) and pf is not None and pf <= float(cfg.get("weak_net_profit_factor", 0.85)):
        return FamilyEdge(n, exp_r, pf, "WEAK_EDGE", 0.5 if full else 0.75,
                          _stddev(rs), sum(x > 0 for x in rs) / n)
    return FamilyEdge(n, exp_r, pf, "NORMAL", 1.0, _stddev(rs), sum(x > 0 for x in rs) / n)


def stagnation_decision(*, completed_sessions: int, mfe_r: float,
                        current_r: float, target_hit: bool = False,
                        early_check: int = 18, exit_session: int = 20,
                        max_hold: int = 25) -> tuple[str | None, bool]:
    """Return (scheduled reason, stagnating flag) using only current/past values."""
    stagnating = (completed_sessions >= early_check and not target_hit
                  and mfe_r < 1.0 and current_r < 0.5)
    if completed_sessions >= max_hold:
        return "TIME_MAX_HOLD_EXIT", stagnating
    if completed_sessions >= exit_session and stagnating:
        return "TIME_STAGNATION_EXIT", True
    return None, stagnating


def size_position(*, equity: float, cash: float, entry_price: float,
                  structural_stop: float, family_multiplier: float,
                  open_risk: float, cfg: Mapping[str, Any], costs: Any) -> dict[str, Any]:
    """Return integer size and component caps. Structural stop is read-only."""
    if entry_price <= structural_stop or structural_stop <= 0 or equity <= 0:
        return {"quantity": 0, "reason": "INVALID_ENTRY_OR_STRUCTURAL_STOP"}
    mult = min(float(cfg["family_weighting"]["maximum_multiplier"]),
               max(float(cfg["family_weighting"]["minimum_multiplier"]), float(family_multiplier)))
    risk_fraction = float(cfg["base_risk_fraction"]) * mult
    max_trade_fraction = float(cfg["base_risk_fraction"]) * float(cfg["family_weighting"]["maximum_multiplier"])
    min_trade_fraction = float(cfg["base_risk_fraction"]) * float(cfg["family_weighting"]["minimum_multiplier"])
    risk_fraction = min(max_trade_fraction, max(min_trade_fraction, risk_fraction))
    risk_per_share = entry_price - structural_stop
    allowed_risk = equity * risk_fraction
    slot_notional = equity / float(cfg["base_slot_notional_divisor"]) * mult
    notional_cap = equity * float(cfg["max_single_position_notional_fraction"])
    desired_notional = min(slot_notional, notional_cap)
    total_risk_capacity = max(0.0, equity * float(cfg["max_total_open_risk_fraction"]) - open_risk)
    qty_by_notional = math.floor(desired_notional / entry_price)
    qty_by_risk = math.floor(allowed_risk / risk_per_share)
    qty_by_aggregate_risk = math.floor(total_risk_capacity / risk_per_share)
    qty_by_cash = math.floor(max(0.0, cash) / entry_price)
    limits = {"NOTIONAL": qty_by_notional, "PER_TRADE_RISK": qty_by_risk,
              "AGGREGATE_RISK": qty_by_aggregate_risk, "CASH": qty_by_cash}
    pre_fee_quantity = min(limits.values())
    binding = [name for name, value in limits.items() if value == pre_fee_quantity]
    quantity = pre_fee_quantity
    while quantity > 0:
        entry_cost = costs.calculate(entry_price * quantity, side="buy").total
        if entry_price * quantity + entry_cost <= cash + 1e-9:
            break
        quantity -= 1
    if quantity < pre_fee_quantity:
        binding.append("CASH_AFFORDABILITY_AFTER_FEES")
    return {"quantity": quantity,
            "reason": None if quantity else "PORTFOLIO_SIZING_ZERO_QUANTITY",
            "family_multiplier": mult, "risk_fraction": risk_fraction,
            "risk_budget": allowed_risk, "risk_per_share": risk_per_share,
            "planned_risk": quantity * risk_per_share,
            "desired_notional": desired_notional,
            "notional_cap": notional_cap,
            "position_notional": quantity * entry_price,
            "quantity_by_notional": qty_by_notional,
            "quantity_by_risk": qty_by_risk,
            "quantity_by_aggregate_risk": qty_by_aggregate_risk,
            "quantity_by_cash": qty_by_cash,
            "sizing_binding_constraints": binding,
            "aggregate_risk_capacity": total_risk_capacity}


def _profit_factor(values: Sequence[float]) -> float | None:
    profit = sum(x for x in values if x > 0)
    loss = -sum(x for x in values if x < 0)
    if loss == 0:
        return math.inf if profit else 0.0
    return profit / loss


def _stddev(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return 0.0 if values else None
    avg = mean(values)
    return math.sqrt(sum((x - avg) ** 2 for x in values) / (len(values) - 1))


def _stamp(value: Any) -> datetime:
    dt = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
    return dt.replace(tzinfo=TZ) if dt.tzinfo is None else dt.astimezone(TZ)


def _family(signal: Mapping[str, Any]) -> str:
    return str(signal.get("setup_family") or signal.get("setup_type") or "UNKNOWN")


def _priority(signal: Mapping[str, Any], multiplier: float) -> tuple[Any, ...]:
    plan = signal["plan"]
    result = signal.get("result") or {}
    zones = signal.get("daily_zones") or []
    if isinstance(zones, str):
        try:
            zones = json.loads(zones)
        except json.JSONDecodeError:
            zones = []
    structural_quality = max((float(z.get("dominance_score") or z.get("evidence_score") or 0)
                              for z in zones if isinstance(z, Mapping)), default=0.0)
    if not structural_quality:
        candidates = [signal.get("support")]
        resistance = signal.get("resistance") or []
        candidates.extend(resistance if isinstance(resistance, list) else [resistance])
        structural_quality = max((float(z.get("dominance_score") or z.get("evidence_score") or 0)
                                  for z in candidates if isinstance(z, Mapping)), default=0.0)
    features = result.get("confirmation_features") or {}
    if isinstance(features, str):
        try:
            features = json.loads(features)
        except json.JSONDecodeError:
            features = {}
    volume = features.get("volume") or {}
    if isinstance(volume, Mapping):
        volume_score = float(volume.get("relative_volume") or volume.get("same_time_relative_volume") or 0)
        volume_passed = bool(volume.get("passed"))
    else:
        volume_score, volume_passed = 0.0, False
    confirmation = result.get("confirmation_type") or ""
    return (-multiplier, -float(plan.get("planned_reward_r") or 0), -structural_quality,
            -int(volume_passed), -volume_score, str(confirmation),
            str(signal.get("symbol") or ""), str(signal["setup_instance_id"]))


def _percentile(values: Sequence[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    i = (len(ordered) - 1) * p
    lo, hi = math.floor(i), math.ceil(i)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (i - lo)


def _metrics(trades: Sequence[Mapping[str, Any]], curve: Sequence[Mapping[str, Any]],
             initial: float, run: Mapping[str, Any]) -> dict[str, Any]:
    closed = [t for t in trades if t.get("status") == "CLOSED"]
    gross, net = [float(t["gross_pnl"]) for t in closed], [float(t["net_pnl"]) for t in closed]
    deployments = [float(row.get("gross_exposure_pct") or 0) for row in curve]
    open_risks = [float(row.get("open_structural_risk") or 0) for row in curve]
    positions = [int(row.get("open_positions") or 0) for row in curve]
    equity_last = float(curve[-1]["equity"]) if curve else initial
    current_costs = sum(float(t.get("transaction_costs") or 0) for t in trades)
    returns = [float(row.get("drawdown_pct") or 0) for row in curve]
    days = len(curve)
    days_by_count = {str(i): (sum(n == i for n in positions) / days if days else 0.0)
                     for i in range(5)}
    idle = Counter(str(row.get("idle_reason") or "") for row in curve)
    idle_cash_by_reason: dict[str, float] = defaultdict(float)
    for row in curve:
        idle_cash_by_reason[str(row.get("idle_reason") or "") ] += float(row.get("cash") or 0)
    lost_risk = [max(0.0, float(t.get("planned_risk") or 0) - float(t.get("gross_pnl") or 0))
                 for t in closed if t.get("exit_reason") == "GAP_STOP"
                 and float(t.get("gross_pnl") or 0) < -float(t.get("planned_risk") or 0)]
    sizing_shortfall: dict[str, float] = defaultdict(float)
    for trade in trades:
        unfilled_notional = max(0.0, float(trade.get("desired_notional") or 0)
                                - float(trade.get("position_notional") or 0))
        for reason in trade.get("sizing_binding_constraints", []):
            sizing_shortfall[reason] += unfilled_notional / max(1, len(trade.get("sizing_binding_constraints", [])))
    return {"initial_capital": initial,
            "executed_trades": len(trades), "closed_trades": len(closed),
            "open_positions": sum(t.get("status") == "OPEN" for t in trades),
            "blocked_trades": sum(int(v) for v in run["rejections"].values()),
            "blocked_by_reason": run["rejections"],
            "gross_pnl_closed": sum(gross), "net_pnl_closed": sum(net),
            "gross_return": (equity_last - initial + current_costs) / initial if initial else 0,
            "net_return_mtm": (equity_last / initial - 1) if initial else 0,
            "ending_equity_mtm": equity_last,
            "gross_profit_factor": _profit_factor(gross), "net_profit_factor": _profit_factor(net),
            "gross_expectancy": mean(gross) if gross else 0.0,
            "net_expectancy": mean(net) if net else 0.0,
            "win_rate": sum(x > 0 for x in net) / len(net) if net else 0.0,
            "max_drawdown_pct": min(returns, default=0.0),
            "transaction_costs": current_costs,
            "average_holding_sessions": mean([float(t.get("holding_sessions") or 0) for t in closed]) if closed else 0.0,
            "median_holding_sessions": median([float(t.get("holding_sessions") or 0) for t in closed]) if closed else 0.0,
            "average_deployment_pct": mean(deployments) if deployments else 0.0,
            "median_deployment_pct": median(deployments) if deployments else 0.0,
            "deployment_p25_pct": _percentile(deployments, .25), "deployment_p75_pct": _percentile(deployments, .75),
            "maximum_deployment_pct": max(deployments, default=0.0),
            "average_idle_cash": mean([float(row.get("cash") or 0) for row in curve]) if curve else 0.0,
            "position_day_fraction_by_count": days_by_count,
            "average_open_positions": mean(positions) if positions else 0.0,
            "maximum_open_positions": max(positions, default=0),
            "days_deployment_at_least_75_pct": sum(x >= .75 for x in deployments) / days if days else 0.0,
            "days_deployment_at_least_90_pct": sum(x >= .90 for x in deployments) / days if days else 0.0,
            "average_open_structural_risk": mean(open_risks) if open_risks else 0.0,
            "maximum_open_structural_risk": max(open_risks, default=0.0),
            "average_open_structural_risk_pct": mean([risk / float(row["equity"]) if float(row["equity"]) else 0
                                                       for risk, row in zip(open_risks, curve)]) if curve else 0.0,
            "average_planned_risk_per_entry": mean([float(t.get("planned_risk") or 0) for t in trades]) if trades else 0.0,
            "median_planned_risk_per_entry": median([float(t.get("planned_risk") or 0) for t in trades]) if trades else 0.0,
            "maximum_planned_risk_per_entry": max((float(t.get("planned_risk") or 0) for t in trades), default=0),
            "average_planned_risk_pct_equity": mean([float(t.get("planned_risk_pct_equity") or 0) for t in trades]) if trades else 0.0,
            "average_family_weighted_risk_per_entry": mean([float(t.get("planned_risk") or 0) for t in trades]) if trades else 0.0,
            "unused_desired_notional_by_binding_constraint": dict(sizing_shortfall),
            "average_realized_loss_to_planned_risk": mean([
                abs(float(t.get("gross_pnl") or 0)) / float(t.get("planned_risk") or 1)
                for t in closed if float(t.get("gross_pnl") or 0) < 0]) if any(float(t.get("gross_pnl") or 0) < 0 for t in closed) else 0.0,
            "gap_stop_count": sum(t.get("exit_reason") == "GAP_STOP" for t in closed),
            "gap_loss_beyond_planned_risk_count": len(lost_risk),
            "gap_loss_beyond_planned_risk_amount": sum(lost_risk),
            "capital_idle_day_classification": dict(idle),
            "idle_cash_day_sum_by_reason": dict(idle_cash_by_reason),
            "average_idle_cash_by_reason": {key: value / max(1, idle[key]) for key, value in idle_cash_by_reason.items()},
            "raw_zero_explicit_cost_same_executions": {
                "gross_pnl": sum(float(t.get("gross_pnl") or 0) for t in closed),
                "net_pnl": sum(float(t.get("gross_pnl") or 0) for t in closed),
                "gross_profit_factor": _profit_factor(gross),
                "net_profit_factor": _profit_factor(gross),
                "net_expectancy": mean(gross) if gross else 0.0,
                "ending_equity_mtm": equity_last + current_costs,
                "net_return_mtm": (equity_last + current_costs) / initial - 1 if initial else 0.0,
                "explicit_transaction_costs": 0.0,
            },
            "time_stagnation_exit_count": sum(t.get("exit_reason") == "TIME_STAGNATION_EXIT" for t in closed),
            "time_max_hold_exit_count": sum(t.get("exit_reason") == "TIME_MAX_HOLD_EXIT" for t in closed),
            "cost_scenario": "CURRENT_ZERODHA_EQUITY_DELIVERY_2026"}


def _counterfactual_exit(signal_bars: Sequence[Mapping[str, Any]], from_stamp: datetime,
                         entry: float, stop: float, target: float, quantity: int,
                         entry_cost: float, costs: Any) -> dict[str, Any]:
    for bar in signal_bars:
        stamp = _stamp(bar["date"])
        if stamp < from_stamp:
            continue
        op, hi, lo = float(bar["open"]), float(bar["high"]), float(bar["low"])
        if op <= stop:
            raw, reason = op, "GAP_STOP"
        elif op >= target:
            raw, reason = op, "GAP_TARGET"
        elif lo <= stop:
            raw, reason = stop, "STOP"
        elif hi >= target:
            raw, reason = target, "TARGET"
        else:
            continue
        fill = costs.fill_price(raw, side="sell")
        sell_cost = costs.calculate(fill * quantity, side="sell", include_dp_charge=True).total
        gross = (fill - entry) * quantity
        return {"label": "COUNTERFACTUAL_ONLY_ORIGINAL_STOP_TARGET_CONTINUED",
                "exit_timestamp": stamp.isoformat(), "exit_reason": reason, "exit_price": fill,
                "gross_pnl": gross, "net_pnl": gross - entry_cost - sell_cost}
    last = signal_bars[-1] if signal_bars else None
    if last is None:
        return {"label": "COUNTERFACTUAL_ONLY_ORIGINAL_STOP_TARGET_CONTINUED", "exit_reason": "NO_REMAINING_BARS"}
    mark = float(last["close"])
    mark_cost = costs.calculate(mark * quantity, side="sell", include_dp_charge=True).total
    gross = (mark - entry) * quantity
    return {"label": "COUNTERFACTUAL_ONLY_ORIGINAL_STOP_TARGET_CONTINUED",
            "exit_timestamp": str(last["date"]), "exit_reason": "OPEN_AT_PERIOD_END", "exit_price": mark,
            "gross_pnl": gross, "net_pnl": gross - entry_cost - mark_cost}


def run_portfolio(signals: Sequence[Mapping[str, Any]], bars: Mapping[str, Sequence[Mapping[str, Any]]],
                  costs: Any, cfg: Mapping[str, Any], *, weighted: bool,
                  time_exits: bool) -> dict[str, Any]:
    """Run one deterministic portfolio arm on saved signal plans and H1 bars."""
    initial = float(cfg["initial_capital"])
    max_positions = int(cfg["max_open_positions"])
    by_time: dict[datetime, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    sid_bars: dict[str, list[Mapping[str, Any]]] = {str(sid): sorted(seq, key=lambda b: _stamp(b["date"]))
                                                    for sid, seq in bars.items()}
    last_stamp_by_day: dict[tuple[str, date], datetime] = {}
    index_by_stamp: dict[str, dict[datetime, int]] = {}
    for sid, seq in sid_bars.items():
        index_by_stamp[sid] = {}
        for i, bar in enumerate(seq):
            stamp = _stamp(bar["date"])
            if stamp.date() < START or stamp.date() > END:
                continue
            if sid in by_time[stamp]:
                raise ValueError(f"Duplicate hourly timestamp for {sid} at {stamp}.")
            by_time[stamp][sid] = bar
            index_by_stamp[sid][stamp] = i
            key = (sid, stamp.date())
            last_stamp_by_day[key] = max(stamp, last_stamp_by_day.get(key, stamp))
    signal_by_time: dict[datetime, list[Mapping[str, Any]]] = defaultdict(list)
    for signal in signals:
        stamp = _stamp(signal["executable_timestamp"])
        if stamp.date() < START or stamp.date() > END:
            raise ValueError(f"Saved signal falls outside frozen period: {signal['setup_instance_id']}.")
        seq = sid_bars.get(str(signal["security_id"]), [])
        i = index_by_stamp.get(str(signal["security_id"]), {}).get(stamp)
        if i is None:
            raise ValueError(f"Saved V2 signal lacks exact executable H1 candle: {signal['setup_instance_id']}.")
        if _stamp(seq[i]["date"]) != stamp:
            raise ValueError("Executable bar index/timestamp mismatch.")
        signal_by_time[stamp].append(signal)

    cash, positions, marks = initial, {}, {}
    ledger: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    rejection_counts: Counter[str] = Counter()
    rejection_rows: list[dict[str, Any]] = []
    family_closed: dict[str, list[dict[str, Any]]] = defaultdict(list)
    family_decisions: list[dict[str, Any]] = []
    executed_ids: set[str] = set()
    cumulative_costs = 0.0
    peak_equity = initial
    stamps_by_date: dict[date, list[datetime]] = defaultdict(list)
    for stamp in sorted(by_time):
        stamps_by_date[stamp.date()].append(stamp)

    def close_position(sid: str, stamp: datetime, raw_price: float, reason: str,
                       *, gap: bool = False, time_exit: bool = False) -> None:
        nonlocal cash, cumulative_costs
        pos = positions.pop(sid)
        fill = costs.fill_price(raw_price, side="sell")
        exit_costs = costs.calculate(fill * pos["quantity"], side="sell", include_dp_charge=True)
        gross = (fill - pos["entry_fill"]) * pos["quantity"]
        total_costs = pos["entry_cost"] + exit_costs.total
        net = gross - total_costs
        cash += fill * pos["quantity"] - exit_costs.total
        cumulative_costs += exit_costs.total
        row = {**pos["base"], "status": "CLOSED", "exit_timestamp": stamp.isoformat(),
               "exit_price": fill, "raw_exit_price": raw_price, "exit_reason": reason,
               "gap_handling": gap, "gross_pnl": gross, "net_pnl": net,
               "entry_costs": pos["entry_cost"], "exit_costs": exit_costs.total,
               "transaction_costs": total_costs, "exit_cost_breakdown": exit_costs.as_dict(),
               "gross_R": gross / pos["planned_risk"] if pos["planned_risk"] else 0,
               "net_R": net / pos["planned_risk"] if pos["planned_risk"] else 0,
               "holding_sessions": len(pos["held_dates"]), "holding_hourly_bars": pos["held_bars"],
               "mfe_R": pos["mfe_R"], "mae_R": pos["mae_R"],
               "completed_sessions": len(pos["held_dates"]),
               "target_reached": reason in {"TARGET", "GAP_TARGET"},
               "current_thesis_state": "CLOSED_" + reason,
               "exit_cost_breakdown": exit_costs.as_dict()}
        if time_exit:
            seq = sid_bars[sid]
            row["counterfactual_original_bracket"] = _counterfactual_exit(
                seq, stamp, pos["entry_fill"], pos["stop"], pos["target"], pos["quantity"],
                pos["entry_cost"], costs)
        ledger.append(row)
        pos["closed_trade"] = row
        # Family samples update at timestamp end so same-timestamp entries cannot
        # use a trade which did not close strictly before their decision time.
        pos["family_for_update"] = pos["family"]
        events.append({"timestamp": stamp.isoformat(), "event": "EXIT", "setup_instance_id": row["setup_instance_id"],
                       "security_id": sid, "reason": reason, "gross_pnl": gross, "net_pnl": net,
                       "released_notional": pos["entry_fill"] * pos["quantity"]})
        deferred_family_updates.append(pos)

    deferred_family_updates: list[dict[str, Any]] = []
    for stamp in sorted(by_time):
        current = by_time[stamp]
        # At a date transition, the prior date is complete. For the calendar's
        # explicit provider grid, a security's last observed bar is session close.
        snapshot = {fam: family_edge(rows, enabled=weighted, thresholds=cfg["family_weighting"])
                    for fam, rows in family_closed.items()}
        family_multiplier = lambda fam: snapshot.get(fam, FamilyEdge(0, None, None,
                                       "INSUFFICIENT_HISTORY" if weighted else "FIXED_WEIGHT", 1.0)).multiplier

        # Update marks at the tradable open, then honor gap barriers before a
        # scheduled next-open time exit.
        for sid in sorted(set(positions) & set(current)):
            bar = current[sid]
            op = float(bar["open"])
            marks[sid] = op
            pos = positions[sid]
            if op <= pos["stop"]:
                close_position(sid, stamp, op, "GAP_STOP", gap=True)
            elif op >= pos["target"]:
                close_position(sid, stamp, op, "GAP_TARGET", gap=True)
            elif pos.get("scheduled_exit"):
                close_position(sid, stamp, op, str(pos["scheduled_exit"]), time_exit=True)

        candidates = sorted(signal_by_time.get(stamp, []),
                            key=lambda s: _priority(s, family_multiplier(_family(s))))
        for rank, signal in enumerate(candidates, 1):
            sid, setup_id = str(signal["security_id"]), str(signal["setup_instance_id"])
            family = _family(signal)
            mult = family_multiplier(family)
            priority = _priority(signal, mult)
            reason = None
            if setup_id in executed_ids:
                reason = "DUPLICATE_SETUP_BLOCKED"
            elif sid in positions:
                reason = "SAME_SECURITY_ALREADY_OPEN"
            elif len(positions) >= max_positions:
                reason = "MAX_OPEN_POSITIONS"
            bar = current.get(sid)
            if bar is None:
                reason = reason or "MISSING_EXECUTABLE_HOURLY_BAR"
            if reason:
                rejection_counts[reason] += 1
                rejection_rows.append({"timestamp": stamp.isoformat(), "setup_instance_id": setup_id,
                                       "security_id": sid, "symbol": signal.get("symbol"),
                                       "reason": reason, "priority_rank": rank,
                                       "priority_key": list(priority), "family_multiplier": mult})
                events.append({"timestamp": stamp.isoformat(), "event": "REJECT", "setup_instance_id": setup_id,
                               "security_id": sid, "reason": reason, "priority_rank": rank})
                continue
            plan = signal["plan"]
            open_price = float(bar["open"])
            stop, target = float(plan["stop"]), float(plan["target"])
            if not math.isclose(float(plan.get("entry", open_price)), open_price, rel_tol=0.0, abs_tol=0.01):
                raise ValueError(f"Saved V2 entry plan differs from executable hourly open for {setup_id}.")
            if _stamp(signal["confirmation_timestamp"]) >= stamp:
                raise ValueError(f"V2 entry is not later than confirmation for {setup_id}.")
            if stop >= open_price:
                reason = "ENTRY_GAP_BELOW_STOP"
            elif target <= open_price:
                reason = "ENTRY_GAP_ABOVE_TARGET"
            if reason:
                rejection_counts[reason] += 1
                rejection_rows.append({"timestamp": stamp.isoformat(), "setup_instance_id": setup_id,
                                       "security_id": sid, "symbol": signal.get("symbol"),
                                       "reason": reason, "priority_rank": rank, "priority_key": list(priority)})
                continue
            open_equity = cash + sum(int(p["quantity"]) * marks.get(k, float(p["entry_fill"]))
                                     for k, p in positions.items())
            open_risk = sum(float(p["planned_risk"]) for p in positions.values())
            entry_fill = costs.fill_price(open_price, side="buy")
            sizing = size_position(equity=open_equity, cash=cash, entry_price=entry_fill,
                                   structural_stop=stop, family_multiplier=mult,
                                   open_risk=open_risk, cfg=cfg, costs=costs)
            if sizing["quantity"] <= 0:
                reason = sizing["reason"]
                rejection_counts[reason] += 1
                rejection_rows.append({"timestamp": stamp.isoformat(), "setup_instance_id": setup_id,
                                       "security_id": sid, "symbol": signal.get("symbol"),
                                       "reason": reason, "priority_rank": rank, "sizing": sizing})
                continue
            qty = int(sizing["quantity"])
            entry_breakdown = costs.calculate(entry_fill * qty, side="buy")
            cash -= entry_fill * qty + entry_breakdown.total
            if cash < -1e-8:
                raise AssertionError("Negative cash after buy affordability check.")
            cumulative_costs += entry_breakdown.total
            planned_risk = qty * (entry_fill - stop)
            edge = snapshot.get(family, FamilyEdge(0, None, None,
                                "INSUFFICIENT_HISTORY" if weighted else "FIXED_WEIGHT", 1.0))
            zone_quality = -priority[2]
            base = {"setup_instance_id": setup_id, "setup_event_id": signal.get("setup_event_id"),
                    "strategy_version": "top_down_swing_v2_daily_structure",
                    "symbol": signal.get("symbol"), "security_id": sid, "setup_family": family,
                    "signal_date": signal.get("setup_signal_date"), "entry_timestamp": stamp.isoformat(),
                    "entry_price": entry_fill, "entry_raw_open": open_price, "quantity": qty,
                    "stop_price": stop, "target_price": target, "target_type": plan.get("target_timeframe"),
                    "entry_costs": entry_breakdown.total, "entry_cost_breakdown": entry_breakdown.as_dict(),
                    "planned_rr": float(plan.get("planned_reward_r") or 0), "family_multiplier": mult,
                    "family_edge_class": edge.classification, "family_closed_trades_before": edge.closed_trades,
                    "family_expectancy_R_before": edge.expectancy_r, "family_net_pf_before": edge.net_profit_factor,
                    "family_net_R_stddev_before": edge.net_r_stddev,
                    "family_positive_net_R_fraction_before": edge.positive_net_r_fraction,
                    "family_prior_closed_trade_ids": [t["setup_instance_id"] for t in family_closed.get(family, [])],
                    "family_prior_closed_exit_timestamps": [t["exit_timestamp"] for t in family_closed.get(family, [])],
                    "family_priority_rank": rank, "dominant_zone_quality_score": zone_quality,
                    "equity_before_entry": open_equity, "cash_before_entry": cash + entry_fill * qty + entry_breakdown.total,
                    "position_notional": entry_fill * qty, "capital_deployed_pct": entry_fill * qty / open_equity,
                    "desired_notional": sizing["desired_notional"],
                    "sizing_binding_constraints": sizing["sizing_binding_constraints"],
                    "risk_budget": sizing["risk_budget"], "planned_risk": planned_risk,
                    "planned_risk_pct_equity": planned_risk / open_equity,
                    "quantity_by_notional": sizing["quantity_by_notional"],
                    "quantity_by_risk": sizing["quantity_by_risk"],
                    "quantity_by_cash_before_fee_adjustment": sizing["quantity_by_cash"],
                    "quantity_by_aggregate_risk": sizing["quantity_by_aggregate_risk"],
                    "aggregate_risk_capacity_before_entry": sizing["aggregate_risk_capacity"],
                    "target_reached": False, "current_thesis_state": "OPEN",
                    "per_share_structural_risk": entry_fill - stop,
                    "transaction_cost_scenario": costs.scenario,
                    "volume_supported_confirmation": bool(plan.get("volume_confirmation_passed")),
                    "confirmation_type": (signal.get("result") or {}).get("confirmation_type"),
                    "max_open_positions_configured": max_positions,
                    "actual_stop_immutable": True}
            if base["strategy_version"] != "top_down_swing_v2_daily_structure":
                raise AssertionError("V2 strategy_version provenance mismatch.")
            position = {"base": base, "family": family, "quantity": qty,
                        "entry_fill": entry_fill, "entry_cost": entry_breakdown.total,
                        "stop": stop, "target": target, "planned_risk": planned_risk,
                        "planned_risk_pct_equity": planned_risk / open_equity,
                        "held_bars": 0, "held_dates": set(), "mfe_R": 0.0, "mae_R": 0.0,
                        "max_high": entry_fill, "min_low": entry_fill,
                        "scheduled_exit": None, "stagnating": False}
            positions[sid] = position
            if entry_fill * qty > open_equity * float(cfg["max_single_position_notional_fraction"]) + 1e-6:
                raise AssertionError("Single-position notional cap exceeded.")
            max_trade_risk = open_equity * float(cfg["base_risk_fraction"]) * float(cfg["family_weighting"]["maximum_multiplier"])
            if planned_risk > max_trade_risk + 1e-6:
                raise AssertionError("Family-weighted per-trade risk cap exceeded.")
            marks[sid] = entry_fill
            executed_ids.add(setup_id)
            family_decisions.append({**base, "timestamp": stamp.isoformat(), "weighted_arm": weighted})
            events.append({"timestamp": stamp.isoformat(), "event": "ENTRY", "setup_instance_id": setup_id,
                           "security_id": sid, "quantity": qty, "notional": entry_fill * qty,
                           "planned_risk": planned_risk, "cash_after": cash, "family_multiplier": mult,
                           "priority_rank": rank})

        # Use bar extrema for held trades after opening decisions; same-bar
        # entry candles can contribute MFE/MAE at their close, never to entry.
        for sid in sorted(set(positions) & set(current)):
            bar = current[sid]
            pos = positions[sid]
            op, hi, lo = float(bar["open"]), float(bar["high"]), float(bar["low"])
            pos["held_bars"] += 1
            pos["held_dates"].add(stamp.date())
            pos["max_high"] = max(pos["max_high"], hi)
            pos["min_low"] = min(pos["min_low"], lo)
            rps = pos["entry_fill"] - pos["stop"]
            pos["mfe_R"] = max(0.0, (pos["max_high"] - pos["entry_fill"]) / rps)
            pos["mae_R"] = min(0.0, (pos["min_low"] - pos["entry_fill"]) / rps)
            # Newly entered trades cannot stop/target on their entry candle.
            if pos["base"]["entry_timestamp"] != stamp.isoformat():
                stop_hit, target_hit = lo <= pos["stop"], hi >= pos["target"]
                if stop_hit:
                    close_position(sid, stamp, pos["stop"],
                                   "AMBIGUOUS_INTRABAR_STOP_FIRST" if target_hit else "STOP")
                elif target_hit:
                    close_position(sid, stamp, pos["target"], "TARGET")
            if sid in positions:
                marks[sid] = float(bar["close"])

        # Finalize each observed session at its final provider candle. This is
        # causal: only that session's complete OHLC is consulted here.
        for sid in sorted(set(positions) & set(current)):
            if last_stamp_by_day.get((sid, stamp.date())) != stamp:
                continue
            pos = positions[sid]
            if not time_exits:
                continue
            current_r = (marks[sid] - pos["entry_fill"]) / (pos["entry_fill"] - pos["stop"])
            management = cfg["trade_management"]
            scheduled, stagnating = stagnation_decision(
                completed_sessions=len(pos["held_dates"]), mfe_r=pos["mfe_R"], current_r=current_r,
                early_check=int(management["early_stagnation_check_session"]),
                exit_session=int(management["stagnation_exit_session"]),
                max_hold=int(management["absolute_max_hold_sessions"]))
            pos["stagnating"] = stagnating
            events.append({"timestamp": stamp.isoformat(), "event": "TIME_CHECK",
                           "setup_instance_id": pos["base"]["setup_instance_id"],
                           "completed_sessions": len(pos["held_dates"]), "current_R": current_r,
                           "MFE_R": pos["mfe_R"], "MAE_R": pos["mae_R"],
                           "stagnating": stagnating, "scheduled_exit": scheduled})
            if scheduled and not pos.get("scheduled_exit"):
                pos["scheduled_exit"] = scheduled
                pos["base"]["time_exit_scheduled_at"] = stamp.isoformat()
                pos["base"]["time_exit_completed_sessions"] = len(pos["held_dates"])

        if stamp == stamps_by_date[stamp.date()][-1] and stamp.date() >= START:
            market_value = sum(p["quantity"] * marks.get(sid, p["entry_fill"]) for sid, p in positions.items())
            equity = cash + market_value
            if equity < -1e-8 or cash < -1e-8:
                raise AssertionError("Portfolio equity or cash became negative.")
            peak_equity = max(peak_equity, equity)
            open_risk = sum(p["planned_risk"] for p in positions.values())
            if len(positions) > max_positions:
                raise AssertionError("Maximum concurrent positions exceeded.")
            if open_risk > equity * float(cfg["max_total_open_risk_fraction"]) + 1e-6:
                raise AssertionError("Aggregate planned open risk cap exceeded.")
            exposure = market_value / equity if equity else 0.0
            candidates_today = [s for ts, group in signal_by_time.items() if ts.date() == stamp.date() for s in group]
            day_rejects = [r for r in rejection_rows if r["timestamp"][:10] == stamp.date().isoformat()]
            if any(r["reason"] in {"MAX_OPEN_POSITIONS", "SAME_SECURITY_ALREADY_OPEN"} for r in day_rejects):
                idle_reason = "SLOT_CONSTRAINT"
            elif any(r["reason"] == "PORTFOLIO_SIZING_ZERO_QUANTITY" for r in day_rejects):
                idle_reason = "RISK_CASH_OR_NOTIONAL_CONSTRAINT"
            elif not candidates_today:
                idle_reason = "INSUFFICIENT_VALID_SIGNALS"
            else:
                idle_reason = "AVAILABLE_UNUSED_CAPACITY"
            curves.append({"date": stamp.date().isoformat(), "timestamp": stamp.isoformat(), "cash": cash,
                           "market_value": market_value, "equity": equity, "gross_exposure_pct": exposure,
                           "cash_pct": cash / equity if equity else 0.0,
                           "open_positions": len(positions), "open_structural_risk": open_risk,
                           "open_structural_risk_pct": open_risk / equity if equity else 0.0,
                           "remaining_risk_capacity": max(0.0, equity * float(cfg["max_total_open_risk_fraction"]) - open_risk),
                           "remaining_position_slots": max_positions - len(positions),
                           "cumulative_transaction_costs": cumulative_costs,
                           "unrealized_gross_pnl": sum((marks.get(sid, p["entry_fill"]) - p["entry_fill"]) * p["quantity"]
                                                        for sid, p in positions.items()),
                           "idle_reason": idle_reason,
                           "drawdown_pct": (equity / peak_equity - 1) * 100 if peak_equity else 0.0})

        # The closed family records become usable only after all entries at this
        # timestamp, so no same-timestamp outcome can leak into sizing.
        for update in deferred_family_updates:
            closed = update.pop("closed_trade", None)
            family = update.pop("family_for_update", None)
            if closed is not None and family is not None:
                family_closed[family].append(closed)
        deferred_family_updates.clear()

    last_stamp = max((max((_stamp(b["date"]) for b in seq), default=datetime.min.replace(tzinfo=TZ))
                      for seq in sid_bars.values()), default=None)
    if last_stamp is None:
        raise ValueError("No hourly bars were loaded for the saved-signal portfolio.")
    for sid, pos in sorted(positions.items()):
        mark = marks.get(sid, pos["entry_fill"])
        liquidation = costs.calculate(mark * pos["quantity"], side="sell", include_dp_charge=True).total
        ledger.append({**pos["base"], "status": "OPEN", "exit_timestamp": "", "exit_price": "",
                       "exit_reason": "OPEN_AT_PERIOD_END", "mark_timestamp": last_stamp.isoformat(),
                       "raw_exit_price": mark, "gap_handling": False,
                       "mark_price": mark, "unrealized_gross_pnl": (mark - pos["entry_fill"]) * pos["quantity"],
                       "estimated_liquidation_value_after_exit_cost": mark * pos["quantity"] - liquidation,
                       "transaction_costs": pos["entry_cost"], "gross_pnl": "", "net_pnl": "",
                       "exit_costs": 0.0, "holding_sessions": len(pos["held_dates"]),
                       "holding_hourly_bars": pos["held_bars"], "mfe_R": pos["mfe_R"], "mae_R": pos["mae_R"],
                       "completed_sessions": len(pos["held_dates"]),
                       "time_exit_scheduled": pos.get("scheduled_exit"),
                       "stagnating": pos.get("stagnating", False),
                       "current_thesis_state": "STAGNATING" if pos.get("stagnating") else "ACTIVE",
                       "target_reached": False})
    for row in curves:
        explicit_costs_paid = float(row.get("cumulative_transaction_costs") or 0)
        row["raw_zero_cost_cash_same_execution"] = float(row["cash"]) + explicit_costs_paid
        row["raw_zero_cost_equity_same_execution"] = float(row["equity"]) + explicit_costs_paid
        row["raw_zero_cost_net_pnl_same_execution"] = row["raw_zero_cost_equity_same_execution"] - initial
    summary = _metrics(ledger, curves, initial, {"rejections": dict(sorted(rejection_counts.items()))})
    summary.update({"weighted_family_sizing": weighted, "time_exits_enabled": time_exits,
                   "max_concurrent_positions_configured": max_positions,
                   "zero_cost_same_execution_invariants_passed": _zero_cost_invariants(ledger),
                   "2025_2026_loaded": False, "start": START.isoformat(), "end": END.isoformat(),
                   "signal_count": len(signals), "unique_signal_ids": len({s["setup_instance_id"] for s in signals}),
                   "simultaneous_signal_priority_rule": ["family_multiplier_desc", "planned_RR_desc",
                      "dominant_zone_quality_desc", "confirmation_quality_desc", "symbol_asc", "setup_instance_id_asc"],
                   "family_weight_history": family_decisions,
                   "capital_idle_due_insufficient_valid_signals": sum(r["idle_reason"] == "INSUFFICIENT_VALID_SIGNALS" for r in curves),
                   "capital_idle_due_risk_constraints": sum(r["idle_reason"] == "RISK_CASH_OR_NOTIONAL_CONSTRAINT" for r in curves),
                   "capital_idle_due_slot_constraints": sum(r["idle_reason"] == "SLOT_CONSTRAINT" for r in curves)})
    return {"summary": summary, "trades": ledger, "equity_curve": curves,
            "events": events, "rejections": rejection_rows,
            "family_decisions": family_decisions}


def _zero_cost_invariants(trades: Sequence[Mapping[str, Any]]) -> bool:
    required = ("setup_instance_id", "setup_event_id", "symbol", "entry_timestamp", "entry_price",
                "stop_price", "target_price", "quantity", "exit_timestamp", "exit_reason",
                "raw_exit_price", "gap_handling", "holding_sessions")
    for t in trades:
        if t.get("strategy_version") != "top_down_swing_v2_daily_structure":
            return False
        current_execution = {field: t.get(field) for field in required}
        if t.get("status") == "CLOSED":
            raw_gross = float(t["gross_pnl"])
            t["gross_pnl_current_cost_view"] = raw_gross
            t["gross_pnl_raw_zero_cost_view"] = raw_gross
            t["raw_zero_cost_net_pnl"] = raw_gross
            t["raw_zero_cost_transaction_costs"] = 0.0
            raw_execution = {**current_execution, "gross_pnl": raw_gross,
                             "net_pnl": raw_gross, "explicit_transaction_costs": 0.0}
            if t["gross_pnl_current_cost_view"] != t["gross_pnl_raw_zero_cost_view"]:
                return False
            if abs(raw_gross - float(t["net_pnl"]) - float(t["transaction_costs"])) > 1e-6:
                return False
        else:
            t["raw_zero_cost_net_pnl"] = t.get("unrealized_gross_pnl")
            t["raw_zero_cost_transaction_costs"] = 0.0
            t["gross_pnl_current_cost_view"] = t.get("unrealized_gross_pnl")
            t["gross_pnl_raw_zero_cost_view"] = t.get("unrealized_gross_pnl")
            raw_execution = {**current_execution, "gross_pnl": t.get("unrealized_gross_pnl"),
                             "net_pnl": t.get("unrealized_gross_pnl"), "explicit_transaction_costs": 0.0}
        if any(current_execution[key] != raw_execution[key] for key in required):
            return False
        t["raw_zero_cost_execution"] = raw_execution
        if any(field not in t for field in required):
            return False
    return True


def _portfolio_breakdowns(trades: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    closed = [t for t in trades if t.get("status") == "CLOSED"]
    out: dict[str, Any] = {}
    for family in sorted({str(t.get("setup_family")) for t in trades}):
        rows = [t for t in trades if str(t.get("setup_family")) == family]
        done = [t for t in rows if t.get("status") == "CLOSED"]
        nets = [float(t["net_pnl"]) for t in done]
        out[family] = {"signals": sum(1 for t in trades if str(t.get("setup_family")) == family),
                       "executed": len(rows), "blocked": None,
                       "notional_deployed_total": sum(float(t.get("position_notional") or 0) for t in rows),
                       "planned_risk_deployed_total": sum(float(t.get("planned_risk") or 0) for t in rows),
                       "gross_pnl": sum(float(t.get("gross_pnl") or 0) for t in done),
                       "net_pnl": sum(nets), "net_pf": _profit_factor(nets),
                       "net_expectancy_R": mean([float(t.get("net_R") or 0) for t in done]) if done else 0,
                       "win_rate": sum(v > 0 for v in nets) / len(nets) if nets else 0,
                       "average_multiplier": mean([float(t.get("family_multiplier") or 1) for t in rows]) if rows else 1,
                       "average_position_notional": mean([float(t.get("position_notional") or 0) for t in rows]) if rows else 0,
                       "average_allocation_pct": mean([float(t.get("capital_deployed_pct") or 0) for t in rows]) if rows else 0,
                       "average_planned_risk": mean([float(t.get("planned_risk") or 0) for t in rows]) if rows else 0,
                       "average_holding_sessions": mean([float(t.get("holding_sessions") or 0) for t in done]) if done else 0,
                       "time_exit_count": sum(t.get("exit_reason") in {"TIME_STAGNATION_EXIT", "TIME_MAX_HOLD_EXIT"} for t in done),
                       "closed_trades": len(done)}
    return out


def _year_breakdown(trades: Sequence[Mapping[str, Any]], curve: Sequence[Mapping[str, Any]], initial: float) -> list[dict[str, Any]]:
    out = []
    prior_equity = initial
    for year in (2023, 2024):
        daily = [r for r in curve if str(r["date"]).startswith(str(year))]
        closed = [t for t in trades if t.get("status") == "CLOSED" and str(t["exit_timestamp"]).startswith(str(year))]
        year_end = float(daily[-1]["equity"]) if daily else prior_equity
        nets = [float(t["net_pnl"]) for t in closed]
        gross = [float(t["gross_pnl"]) for t in closed]
        out.append({"year": year, "closed_trades": len(closed), "return": year_end / prior_equity - 1 if prior_equity else 0,
                    "gross_pnl": sum(gross), "net_pnl": sum(nets), "gross_pf": _profit_factor(gross),
                    "net_pf": _profit_factor(nets), "win_rate": sum(x > 0 for x in nets) / len(nets) if nets else 0,
                    "gross_expectancy": mean(gross) if gross else 0, "net_expectancy": mean(nets) if nets else 0,
                    "max_drawdown_pct": min((float(r["drawdown_pct"]) for r in daily), default=0),
                    "average_deployment_pct": mean([float(r["gross_exposure_pct"]) for r in daily]) if daily else 0})
        if daily:
            prior_equity = year_end
    return out


def _time_exit_audit(with_time: Mapping[str, Any], without_time: Mapping[str, Any]) -> dict[str, Any]:
    timed = [t for t in with_time["trades"] if t.get("exit_reason") in {"TIME_STAGNATION_EXIT", "TIME_MAX_HOLD_EXIT"}]
    control_ids = {t["setup_instance_id"] for t in without_time["trades"]}
    actual_ids = {t["setup_instance_id"] for t in with_time["trades"]}
    admitted = sorted(actual_ids - control_ids)
    releases = [float(t.get("position_notional") or 0) for t in timed]
    trading_dates = [date.fromisoformat(str(row["date"])) for row in with_time["equity_curve"]]
    slot_days = 0
    for t in timed:
        c = (t.get("counterfactual_original_bracket") or {}).get("exit_timestamp")
        if c:
            actual_day = date.fromisoformat(str(t["exit_timestamp"])[:10])
            counterfactual_day = date.fromisoformat(str(c)[:10])
            slot_days += sum(actual_day < day <= counterfactual_day for day in trading_dates)
    return {"time_stagnation_exit_count": sum(t.get("exit_reason") == "TIME_STAGNATION_EXIT" for t in timed),
            "time_max_hold_exit_count": sum(t.get("exit_reason") == "TIME_MAX_HOLD_EXIT" for t in timed),
            "average_capital_released": mean(releases) if releases else 0.0,
            "slot_days_released": slot_days,
            "time_exit_gross_pnl_total": sum(float(t.get("gross_pnl") or 0) for t in timed),
            "time_exit_net_pnl_total": sum(float(t.get("net_pnl") or 0) for t in timed),
            "time_exit_net_expectancy": mean([float(t.get("net_pnl") or 0) for t in timed]) if timed else 0.0,
            "subsequent_signal_ids_admitted_only_with_time_exit": admitted,
            "counterfactual_original_stop_target_results": [
                {"setup_instance_id": t["setup_instance_id"],
                 "time_exit_reason": t["exit_reason"], "time_exit_net_pnl": t.get("net_pnl"),
                 **(t.get("counterfactual_original_bracket") or {})} for t in timed]}


def _family_audit(result: Mapping[str, Any], all_signals: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    trades = result["trades"]
    grouped = _portfolio_breakdowns(trades)
    totals = Counter(_family(s) for s in all_signals)
    executed = Counter(str(t.get("setup_family")) for t in trades)
    for family in set(totals) | set(grouped):
        grouped.setdefault(family, {"signals": totals[family], "executed": 0})
        grouped[family]["signals"] = totals[family]
        grouped[family]["blocked"] = totals[family] - executed[family]
        grouped[family]["executed"] = executed[family]
    decisions = result["family_decisions"]
    grouped["_multiplier_history"] = [{k: d[k] for k in ("setup_instance_id", "timestamp", "setup_family",
        "family_multiplier", "family_edge_class", "family_closed_trades_before",
        "family_expectancy_R_before", "family_net_pf_before", "family_net_R_stddev_before",
        "family_positive_net_R_fraction_before", "family_prior_closed_trade_ids",
        "family_prior_closed_exit_timestamps", "planned_risk") if k in d} for d in decisions]
    return grouped


def run(*, signals_path: Path = SIGNALS, output_dir: Path = OUT,
        config_path: Path = CONFIG, data_root: Path = DATA_ROOT,
        daily_manifest: Path = MANIFEST_PATH, hourly_manifest: Path = HOURLY_MANIFEST) -> dict[str, Any]:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if cfg.get("version") != "top_down_swing_v2_portfolio_management_1":
        raise ValueError("Unexpected V2 portfolio-management configuration version.")
    saved_summary = json.loads(SIGNAL_SUMMARY.read_text(encoding="utf-8"))
    if saved_summary.get("strategy_version") != "top_down_swing_v2_daily_structure" or saved_summary.get("2025_2026_loaded"):
        raise ValueError("Saved signal artifact is not the frozen V2 2023-2024 replay.")
    signals = _read_signals(signals_path)
    if len(signals) != 129 or len({s["setup_instance_id"] for s in signals}) != 129:
        raise ValueError(f"Expected 129 distinct saved V2 setup instances, received {len(signals)}.")
    bars, hourly_hash = load_hourly_dataset_through(universe_id=PIT_TRADABLE,
        start_date=date(2022, 10, 1), end_date=END, data_root=data_root,
        manifest_path=hourly_manifest, daily_manifest_path=daily_manifest)
    expected_hash = saved_summary.get("data_hashes", {}).get("hourly_prefix_sha256")
    if expected_hash and hourly_hash != expected_hash:
        raise ValueError("Loaded hourly prefix does not match saved V2 replay provenance.")
    costs = _cost_config(PROJECT_ROOT / "config" / "settings.yaml")
    arms = {}
    for label, weighted, use_time in (("control_no_time", False, False),
                                      ("control_with_time", False, True),
                                      ("weighted_no_time", True, False),
                                      ("weighted_with_time", True, True)):
        arms[label] = run_portfolio(signals, bars, costs, cfg, weighted=weighted, time_exits=use_time)
        if not arms[label]["summary"]["zero_cost_same_execution_invariants_passed"]:
            raise RuntimeError(f"STOP: execution/cost comparator invariant failed for {label}.")
        if any(t["strategy_version"] != "top_down_swing_v2_daily_structure" for t in arms[label]["trades"]):
            raise RuntimeError(f"STOP: V2 provenance mismatch in {label}.")
    old = json.loads((OUTPUT / "v2_summary_100k.json").read_text(encoding="utf-8"))
    old_metrics = old["portfolio_metrics"]
    for label, result in arms.items():
        result["summary"]["year_breakdown"] = _year_breakdown(result["trades"], result["equity_curve"], float(cfg["initial_capital"]))
        allocation = _family_audit(result, signals)
        result["summary"]["family_allocation_audit"] = allocation
        result["summary"]["setup_family_contribution"] = {k: v for k, v in allocation.items() if not k.startswith("_")}
    for weighted_prefix in ("control", "weighted"):
        arms[f"{weighted_prefix}_with_time"]["summary"]["time_exit_audit"] = _time_exit_audit(
            arms[f"{weighted_prefix}_with_time"], arms[f"{weighted_prefix}_no_time"])
    comparison = {
        "old_baseline_saved_only": {"source": str(OUTPUT / "v2_summary_100k.json"),
             "executed_trades": int(old_metrics.get("trade_count") or 0) + int(old_metrics.get("open_positions") or 0),
             "closed_trades": old_metrics.get("trade_count"),
             "open_positions": old_metrics.get("open_positions"),
             "blocked_trades": None,
             "net_return": old_metrics.get("net_return"), "gross_return": old_metrics.get("gross_return"),
             "net_return_mtm": old_metrics.get("net_return"),
             "net_profit_factor": old_metrics.get("net_profit_factor"),
             "gross_profit_factor": old_metrics.get("gross_profit_factor"),
             "net_expectancy": old_metrics.get("net_expectancy"),
             "gross_expectancy": old_metrics.get("gross_expectancy"),
             "win_rate": old_metrics.get("win_rate"),
             "max_drawdown_pct": old_metrics.get("max_drawdown_pct"),
             "transaction_costs": old_metrics.get("transaction_costs"),
             "average_holding_sessions": old_metrics.get("average_holding_sessions")},
        "control_no_time": arms["control_no_time"]["summary"],
        "control_with_time": arms["control_with_time"]["summary"],
        "weighted_no_time": arms["weighted_no_time"]["summary"],
        "weighted_with_time": arms["weighted_with_time"]["summary"],
        "hourly_prefix_sha256": hourly_hash, "2025_2026_loaded": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for label, result in arms.items():
        _csv(output_dir / f"{label}_trades.csv", result["trades"])
        _csv(output_dir / f"{label}_equity_curve.csv", result["equity_curve"])
        _csv(output_dir / f"{label}_events.csv", result["events"])
        _csv(output_dir / f"{label}_rejections.csv", result["rejections"])
        _csv(output_dir / f"{label}_family_allocation.csv", result["summary"]["family_allocation_audit"].get("_multiplier_history", []))
        (output_dir / f"{label}_summary.json").write_text(json.dumps(result["summary"], indent=2, sort_keys=True, allow_nan=False, default=str) + "\n", encoding="utf-8")
    (output_dir / "portfolio_comparison.json").write_text(json.dumps(comparison, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n", encoding="utf-8")
    (output_dir / "portfolio_comparison.csv").write_text("arm,executed,closed,blocked,net_return,net_pf,win_rate,max_drawdown_pct,average_deployment_pct\n" + "".join(
        f"{name},{arms[name]['summary']['executed_trades']},{arms[name]['summary']['closed_trades']},{arms[name]['summary']['blocked_trades']},{arms[name]['summary']['net_return_mtm']},{arms[name]['summary']['net_profit_factor']},{arms[name]['summary']['win_rate']},{arms[name]['summary']['max_drawdown_pct']},{arms[name]['summary']['average_deployment_pct']}\n"
        for name in arms), encoding="utf-8")
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signals", type=Path, default=SIGNALS)
    parser.add_argument("--output-directory", type=Path, default=OUT)
    args = parser.parse_args()
    report = run(signals_path=args.signals, output_dir=args.output_directory)
    print(json.dumps({name: {k: result.get(k) for k in ("executed_trades", "closed_trades", "blocked_trades",
          "net_return_mtm", "net_profit_factor", "win_rate", "max_drawdown_pct", "average_deployment_pct",
          "time_stagnation_exit_count", "time_max_hold_exit_count")}
          for name, result in report.items() if isinstance(result, Mapping)}, indent=2, default=str))


if __name__ == "__main__":
    main()
