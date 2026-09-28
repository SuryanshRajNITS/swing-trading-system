"""Portfolio-level execution replay for frozen top_down_swing_v1 ready events."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import asdict
from datetime import date, datetime
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from zoneinfo import ZoneInfo

import yaml

from src.backtest.data import BacktestDataError, load_hourly_dataset_through
from src.backtest.models import CostModel
from src.configuration import PROJECT_ROOT
from src.data.hourly import HOURLY_MANIFEST, HOURLY_QUARANTINE_PATH
from src.data.market_data import DATA_ROOT, MANIFEST_PATH
from src.data.universe.provider import PIT_TRADABLE

START, END = date(2023, 1, 1), date(2024, 12, 31)
TZ = ZoneInfo("Asia/Kolkata")
SIGNAL_DIR = PROJECT_ROOT / "data" / "processed" / "backtests" / "top_down_swing_v1"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "backtests" / "top_down_swing_v1_portfolio"


def _stamp(value: Any) -> datetime:
    result = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
    return result.replace(tzinfo=TZ) if result.tzinfo is None else result.astimezone(TZ)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns or ["empty"], extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True, ensure_ascii=True)
                             if isinstance(value, (dict, list, tuple)) else value
                             for key, value in row.items()})


def _load_signals(trace_path: Path, start: date, end: date) -> list[dict[str, Any]]:
    signals = []
    for row in _read_csv(trace_path):
        if row.get("state") != "ENTRY_READY_EVENT":
            continue
        signal_day = date.fromisoformat(row["date"][:10])
        if not start <= signal_day <= end:
            continue
        result = json.loads(row["hourly_result"] or "{}")
        plan = result.get("plan") or {}
        if not result.get("entry_timestamp") or not result.get("confirmation_timestamp"):
            raise ValueError(f"Ready event lacks frozen timestamps: {row.get('setup_instance_id')}")
        signal = {**row, "result": result, "plan": plan,
                  "executable_timestamp": _stamp(result["entry_timestamp"]),
                  "confirmation_timestamp": _stamp(result["confirmation_timestamp"])}
        for key in ("entry", "stop", "target"):
            if plan.get(key) is None or not math.isfinite(float(plan[key])):
                raise ValueError(f"Ready event has invalid {key}: {row.get('setup_instance_id')}")
        signals.append(signal)
    if not signals:
        raise ValueError("No saved ENTRY_READY_EVENT signals overlap the requested period.")
    return signals


def _rank(signal: Mapping[str, Any]) -> tuple[float, int, float, str, str]:
    plan = signal["plan"]
    target_type = str(plan.get("target_timeframe") or plan.get("target_reference") or "")
    # Real structural targets outrank fixed 2R fallback; higher timeframe first.
    timeframe_rank = {"WEEKLY": 3, "DAILY": 2, "HOURLY": 1}.get(target_type, 0)
    strength = {"MAJOR": 3, "MEANINGFUL": 2, "MINOR": 1}.get(str(plan.get("target_zone_strength")), 0)
    return (-float(plan.get("planned_reward_r") or 0), -timeframe_rank, -strength,
            str(signal["security_id"]), str(signal["setup_instance_id"]))


def _cost_config(path: Path) -> CostModel:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    costs = raw.get("backtest", {}).get("transaction_costs", {})
    allowed = {key: costs[key] for key in CostModel.__dataclass_fields__ if key in costs}
    return CostModel(**allowed)


def _metrics(trades: Sequence[Mapping[str, Any]], equity_rows: Sequence[Mapping[str, Any]],
             initial: float) -> dict[str, Any]:
    closed = [r for r in trades if r["status"] == "CLOSED"]
    gross = [float(r["gross_pnl"]) for r in closed]
    net = [float(r["net_pnl"]) for r in closed]
    winners = [x for x in net if x > 0]
    losers = [x for x in net if x < 0]
    pf = sum(winners) / abs(sum(losers)) if losers else (None if winners else 0.0)
    last = equity_rows[-1] if equity_rows else {"equity": initial, "cash": initial}
    closed_net = sum(net)
    open_unrealized = sum(float(r.get("unrealized_gross_pnl", 0)) for r in trades if r["status"] == "OPEN")
    total_costs = sum(float(r.get("transaction_costs", 0)) for r in trades)
    ending_equity_gross = float(last["equity"]) + total_costs
    net_r = [float(r["net_R"]) for r in closed if r.get("net_R") not in (None, "")]
    gross_profit = sum(x for x in gross if x > 0)
    gross_loss = abs(sum(x for x in gross if x < 0))
    drawdowns = []
    peak_value = initial
    max_drawdown_rupees = 0.0
    current_dd_days = 0
    maximum_dd_days = 0
    for row in equity_rows:
        peak_value = max(peak_value, float(row["equity"]))
        drawdown_rupees = peak_value-float(row["equity"])
        drawdowns.append((float(row["equity"])/peak_value-1)*100 if peak_value else 0.0)
        max_drawdown_rupees = max(max_drawdown_rupees, drawdown_rupees)
        current_dd_days = current_dd_days + 1 if drawdown_rupees > 1e-9 else 0
        maximum_dd_days = max(maximum_dd_days, current_dd_days)
    durations = [float(r.get("holding_hours", 0)) for r in closed]
    session_durations = [float(r.get("holding_sessions", 0)) for r in closed]
    sorted_net = sorted(net)
    median_net = (sorted_net[(len(sorted_net)-1)//2] + sorted_net[len(sorted_net)//2])/2 if sorted_net else 0.0
    return {"initial_capital": initial, "executed_trade_count": len(trades), "trade_count": len(closed), "open_positions": sum(r["status"] == "OPEN" for r in trades),
            "win_rate": sum(x > 0 for x in net) / len(net) if net else 0.0,
            "gross_expectancy": sum(gross) / len(gross) if gross else 0.0,
            "net_expectancy": sum(net) / len(net) if net else 0.0,
            "expectancy_R": sum(net_r)/len(net_r) if net_r else 0.0,
            "average_net_trade": sum(net)/len(net) if net else 0.0,
            "median_net_trade": median_net,
            "winning_trades": len(winners), "losing_trades": len(losers),
            "gross_profit": gross_profit, "gross_loss": gross_loss,
            "average_winner_R": sum(float(r["net_R"]) for r in closed if float(r["net_pnl"]) > 0)/len(winners) if winners else 0.0,
            "average_loser_R": sum(float(r["net_R"]) for r in closed if float(r["net_pnl"]) < 0)/len(losers) if losers else 0.0,
            "median_R": (sorted(net_r)[(len(net_r)-1)//2] + sorted(net_r)[len(net_r)//2])/2 if net_r else 0.0,
            "profit_factor": pf, "gross_pnl": sum(gross), "net_pnl": closed_net,
            "transaction_costs": total_costs,
            "closed_trading_return": closed_net / initial,
            "gross_return_before_costs": ending_equity_gross/initial-1,
            "ending_equity_mtm": float(last["equity"]),
            "ending_equity_gross_mtm": ending_equity_gross,
            "ending_cash": float(last["cash"]),
            "open_unrealized_gross_pnl": open_unrealized,
            "average_holding_hours": sum(durations)/len(durations) if durations else 0.0,
            "median_holding_hours": ((sorted(durations)[(len(durations)-1)//2]+sorted(durations)[len(durations)//2])/2
                                     if durations else 0.0),
            "average_holding_sessions": sum(session_durations)/len(session_durations) if session_durations else 0.0,
            "median_holding_sessions": ((sorted(session_durations)[(len(session_durations)-1)//2]+
                sorted(session_durations)[len(session_durations)//2])/2 if session_durations else 0.0),
            "best_trade_net_pnl": max(net) if net else 0.0,
            "worst_trade_net_pnl": min(net) if net else 0.0,
            "best_trade": (max(closed, key=lambda r: float(r["net_pnl"])) if closed else None),
            "worst_trade": (min(closed, key=lambda r: float(r["net_pnl"])) if closed else None),
            "accounting_check_initial_plus_closed_net_plus_open_gross_less_entry_cost":
                initial + closed_net + open_unrealized - sum(float(r.get("entry_costs", 0)) for r in trades if r["status"] == "OPEN"),
            "daily_max_drawdown_pct": min(drawdowns, default=0.0),
            "daily_max_drawdown_rupees": max_drawdown_rupees,
            "longest_drawdown_duration_days": maximum_dd_days,
            "net_return_mtm": float(last["equity"])/initial-1 if initial else 0.0}


def _run_portfolio(signals: Sequence[Mapping[str, Any]], bars: Mapping[str, Sequence[Mapping[str, Any]]],
                   costs: CostModel, *, initial_capital: float = 10_000.0,
                   risk_fraction: float = .01, max_positions: int = 10,
                   period_start: date = START, rank_key: Callable[[Mapping[str, Any]], Any] = _rank,
                   preferred_deployment_pct: float | None = None) -> dict[str, Any]:
    if preferred_deployment_pct is not None and not 0 < preferred_deployment_pct <= 1:
        raise ValueError("preferred_deployment_pct must be in (0, 1].")
    bars_by_sid: dict[str, list[dict[str, Any]]] = {}
    bars_at_time: dict[datetime, list[tuple[str, int, dict[str, Any]]]] = defaultdict(list)
    for sid, rows in bars.items():
        ordered = sorted((dict(row) for row in rows), key=lambda r: _stamp(r["date"]))
        bars_by_sid[sid] = ordered
        for index, bar in enumerate(ordered):
            bars_at_time[_stamp(bar["date"])].append((sid, index, bar))
    signal_by_time: dict[datetime, list[dict[str, Any]]] = defaultdict(list)
    for signal in signals:
        signal_by_time[signal["executable_timestamp"]].append(dict(signal))
    available_times = sorted(set(bars_at_time) | set(signal_by_time))
    if not available_times:
        raise ValueError("No hourly timestamps available.")
    cash = float(initial_capital)
    cumulative_costs = 0.0
    positions: dict[str, dict[str, Any]] = {}
    dp_charged: set[tuple[str, date]] = set()
    marks: dict[str, float] = {}
    ledger: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    equity_rows: list[dict[str, Any]] = []
    executed_ids: set[str] = set()
    rejected_counts: dict[str, int] = defaultdict(int)
    sizing_counts: dict[str, int] = defaultdict(int)
    simultaneous_groups: list[dict[str, Any]] = []
    peak = initial_capital

    def close(sid: str, stamp: datetime, price: float, reason: str, ambiguous: bool = False) -> None:
        nonlocal cash, cumulative_costs
        position = positions.pop(sid)
        raw = float(price)
        exit_fill = costs.fill_price(raw, side="sell")
        gross = (exit_fill - position["entry_fill"]) * position["quantity"]
        dp_key = (sid, stamp.date())
        include_dp = dp_key not in dp_charged
        exit_breakdown = costs.calculate(exit_fill * position["quantity"], side="sell", include_dp_charge=include_dp)
        exit_cost = exit_breakdown.total
        if include_dp:
            dp_charged.add(dp_key)
        entry_cost = position["entry_cost"]
        cash += exit_fill * position["quantity"] - exit_cost
        total_cost = entry_cost + exit_cost
        cumulative_costs += exit_cost
        net = gross - total_cost
        ledger.append({**position["base"], "status": "CLOSED", "exit_timestamp": stamp.isoformat(),
            "exit_price": exit_fill, "exit_reason": reason,
            "ambiguous_intrabar_stop_first": ambiguous,
            "gross_pnl": gross, "entry_costs": entry_cost, "exit_costs": exit_cost,
            "entry_cost_breakdown": position["entry_cost_breakdown"],
            "exit_cost_breakdown": exit_breakdown.as_dict(),
            "transaction_costs": total_cost, "net_pnl": net,
            "gross_R": gross / position["initial_risk"] if position["initial_risk"] else None,
            "net_R": net / position["initial_risk"] if position["initial_risk"] else None,
            "holding_hourly_bars": position["held_bars"],
            "holding_hours": (_stamp(stamp)-_stamp(position["base"]["entry_timestamp"])).total_seconds()/3600,
            "holding_sessions": len(position["held_sessions"]),
            "mae_pct": position["mae_pct"], "mfe_pct": position["mfe_pct"]})
        event_rows.append({"timestamp": stamp.isoformat(), "event": "EXIT", "security_id": sid,
                           "reason": reason, "cash_after": cash})

    for stamp in available_times:
        current_bars = sorted(bars_at_time.get(stamp, []), key=lambda x: x[0])
        # Keep prior marks through open-time decisions; current closes are known only later.
        by_sid = {sid: (index, bar) for sid, index, bar in current_bars}
        for sid in sorted(list(positions)):
            if sid not in by_sid:
                continue
            index, bar = by_sid[sid]
            pos = positions[sid]
            pos["held_bars"] += 1
            pos["held_sessions"].add(stamp.date())
            op, hi, lo = float(bar["open"]), float(bar["high"]), float(bar["low"])
            pos["mae_pct"] = min(pos["mae_pct"], (lo / pos["entry_fill"] - 1) * 100)
            pos["mfe_pct"] = max(pos["mfe_pct"], (hi / pos["entry_fill"] - 1) * 100)
            if op <= pos["stop"]:
                close(sid, stamp, op, "GAP_STOP")
            elif op >= pos["target"]:
                close(sid, stamp, op, "GAP_TARGET")
        # Same-time signals are allocated as a single deterministic group.
        group = sorted(signal_by_time.get(stamp, []), key=rank_key)
        if len(group) > 1:
            simultaneous_groups.append({"timestamp": stamp.isoformat(), "setup_instance_ids": [s["setup_instance_id"] for s in group],
                "priority": [{"setup_instance_id": s["setup_instance_id"], "planned_rr": s["plan"].get("planned_reward_r"),
                              "target_timeframe": s["plan"].get("target_timeframe"),
                              "target_zone_strength": s["plan"].get("target_zone_strength")} for s in group]})
        for signal in group:
            sid, setup_id = str(signal["security_id"]), str(signal["setup_instance_id"])
            if setup_id in executed_ids:
                rejected_counts["DUPLICATE_SETUP_BLOCKED"] += 1
                event_rows.append({"timestamp": stamp.isoformat(), "event": "REJECT", "setup_instance_id": setup_id,
                                   "security_id": sid, "reason": "DUPLICATE_SETUP_BLOCKED"})
                continue
            executed_ids.add(setup_id)
            bar_info = by_sid.get(sid)
            if bar_info is None or bar_info[1].get("date") is None:
                rejected_counts["MISSING_EXECUTABLE_HOURLY_BAR"] += 1
                event_rows.append({"timestamp": stamp.isoformat(), "event": "REJECT", "setup_instance_id": setup_id,
                                   "security_id": sid, "reason": "MISSING_EXECUTABLE_HOURLY_BAR"})
                continue
            _, entry_bar = bar_info
            if _stamp(entry_bar["date"]) <= signal["confirmation_timestamp"]:
                raise ValueError(f"Execution is not after confirmation for {setup_id}.")
            if sid in positions:
                rejected_counts["ALREADY_OPEN_POSITION"] += 1
                reason = "CAPITAL_ALLOCATED_TO_PRIOR_SIGNAL" if len(group) > 1 else "ALREADY_OPEN_POSITION"
                event_rows.append({"timestamp": stamp.isoformat(), "event": "REJECT", "setup_instance_id": setup_id,
                                   "security_id": sid, "reason": reason})
                continue
            op = float(entry_bar["open"])
            plan = signal["plan"]
            planned_entry = float(plan["entry"])
            if not math.isclose(op, planned_entry, rel_tol=0.0, abs_tol=0.01):
                raise ValueError(f"Saved entry plan differs from next hourly open for {setup_id}: {planned_entry} vs {op}.")
            stop, target = float(plan["stop"]), float(plan["target"])
            if stop >= op:
                rejected_counts["ENTRY_GAP_BELOW_STOP"] += 1
                event_rows.append({"timestamp": stamp.isoformat(), "event": "REJECT", "setup_instance_id": setup_id,
                                   "security_id": sid, "reason": "ENTRY_GAP_BELOW_STOP"})
                continue
            if target <= op:
                rejected_counts["ENTRY_GAP_ABOVE_TARGET"] += 1
                event_rows.append({"timestamp": stamp.isoformat(), "event": "REJECT", "setup_instance_id": setup_id,
                                   "security_id": sid, "reason": "ENTRY_GAP_ABOVE_TARGET"})
                continue
            if len(positions) >= max_positions:
                rejected_counts["MAX_OPEN_POSITIONS"] += 1
                reason = "CAPITAL_ALLOCATED_TO_PRIOR_SIGNAL" if len(group) > 1 else "MAX_OPEN_POSITIONS"
                event_rows.append({"timestamp": stamp.isoformat(), "event": "REJECT", "setup_instance_id": setup_id,
                                   "security_id": sid, "reason": reason})
                continue
            equity_before = cash + sum(p["quantity"] * marks.get(key, p["entry_fill"]) for key, p in positions.items())
            risk_budget = equity_before * risk_fraction
            entry_fill = costs.fill_price(op, side="buy")
            per_share_risk = entry_fill - stop
            if per_share_risk <= 0:
                rejected_counts["INVALID_ACTUAL_RISK"] += 1
                continue
            qty_risk = math.floor(risk_budget / per_share_risk)
            qty_cash = math.floor(cash / entry_fill)
            qty = min(qty_risk, qty_cash)
            desired_value = cash if preferred_deployment_pct is None else cash * preferred_deployment_pct
            qty_desired = math.floor(desired_value / entry_fill)
            if preferred_deployment_pct is not None:
                qty = min(qty, qty_desired)
            # Affordability includes buy charges; reduce safely until it fits.
            while qty > 0:
                entry_cost = costs.calculate(entry_fill * qty, side="buy").total
                if entry_fill * qty + entry_cost <= cash + 1e-9:
                    break
                qty -= 1
            if qty <= 0:
                underlying = ("INSUFFICIENT_CASH" if qty_cash <= 0 < qty_risk
                              else "ZERO_QUANTITY_CAPITAL_CONSTRAINT")
                reason = ("CAPITAL_ALLOCATED_TO_PRIOR_SIGNAL" if len(group) > 1 and group.index(signal) > 0
                          else underlying)
                rejected_counts[reason] += 1
                event_rows.append({"timestamp": stamp.isoformat(), "event": "REJECT", "setup_instance_id": setup_id,
                                   "security_id": sid, "reason": reason, "detail_reason": underlying,
                                   "reason_codes": [reason, underlying] if reason != underlying else [reason]})
                continue
            if qty < qty_risk:
                sizing_counts["QUANTITY_CAPPED_BY_AVAILABLE_CASH"] += 1
            entry_breakdown = costs.calculate(entry_fill * qty, side="buy")
            buy_cost = entry_breakdown.total
            cash -= entry_fill * qty + buy_cost
            cumulative_costs += buy_cost
            risk_rupees = qty * per_share_risk
            metadata = {"setup_family": signal.get("setup_type"),
                "confirmation_archetype": signal["result"].get("confirmation_type"),
                "weekly_selection_path": signal.get("weekly_selection_path"),
                "target_type": plan.get("target_timeframe"), "target_reference": plan.get("target_reference"),
                "planned_rr": plan.get("planned_reward_r"),
                "volume_relative": (signal["result"].get("confirmation_features") or {}).get("same_time_relative_volume"),
                "ema_context": plan.get("ema_context")}
            base = {"setup_instance_id": setup_id, "strategy_version": "top_down_swing_v1",
                "trade_id": setup_id, "setup_type": signal.get("setup_type"),
                "security_id": sid, "symbol": signal.get("symbol"), "setup_family": metadata["setup_family"],
                "confirmation_archetype": metadata["confirmation_archetype"],
                "weekly_selection_path": metadata["weekly_selection_path"],
                "signal_date": signal.get("setup_signal_date"), "confirmation_timestamp": signal["confirmation_timestamp"].isoformat(),
                "entry_timestamp": stamp.isoformat(), "entry_raw_open": op, "entry_price": entry_fill,
                "entry_slippage_cost": (entry_fill-op)*qty, "stop_price": stop, "target_price": target,
                "target_type": metadata["target_type"], "target_reference": metadata["target_reference"],
                "planned_rr": metadata["planned_rr"], "actual_entry_rr": (target-entry_fill)/(entry_fill-stop),
                "quantity": qty, "equity_before_entry": equity_before, "risk_budget": risk_budget,
                "initial_risk_rupees": risk_rupees, "available_cash_after_entry": cash,
                "position_value": entry_fill*qty,
                "capital_deployed_pct": entry_fill*qty/equity_before if equity_before else 0.0,
                "intended_risk_pct": risk_fraction,
                "actual_risk_pct": risk_rupees/equity_before if equity_before else 0.0,
                "risk_budget_utilization": risk_rupees/risk_budget if risk_budget else 0.0,
                "entry_costs": buy_cost, "transaction_cost_scenario": costs.scenario,
                "entry_cost_breakdown": entry_breakdown.as_dict(),
                "confirmation_volume_relative": metadata["volume_relative"],
                "volume_supported_confirmation": bool((signal["result"].get("confirmation_features") or {}).get("expansion_vs_prior_mean")
                    or (signal["result"].get("confirmation_features") or {}).get("same_time_expansion")),
                "ema_confluence": bool(plan.get("ema_sr_confluence")),
                "overhead_friction_present": bool(plan.get("overhead_friction_present")),
                "stop_zone_id": plan.get("stop_zone_id"), "target_zone_id": plan.get("target_zone_id"),
                "daily_ema50": signal.get("daily_ema50"), "daily_ema200": signal.get("daily_ema200")}
            positions[sid] = {"base": base, "entry_fill": entry_fill, "quantity": qty,
                "entry_cost": buy_cost, "entry_cost_breakdown": entry_breakdown.as_dict(),
                "initial_risk": risk_rupees, "stop": stop, "target": target,
                "held_bars": 0, "held_sessions": {stamp.date()}, "mae_pct": 0.0, "mfe_pct": 0.0}
            event_rows.append({"timestamp": stamp.isoformat(), "event": "ENTRY", "setup_instance_id": setup_id,
                               "security_id": sid, "quantity": qty, "cash_after": cash})
        # Intrabar stop/target tests are for positions held before this bar only.
        entered_now = {e.get("security_id") for e in event_rows if e.get("timestamp") == stamp.isoformat() and e.get("event") == "ENTRY"}
        for sid in sorted(list(positions)):
            if sid not in by_sid or sid in entered_now:
                continue
            _, bar = by_sid[sid]
            pos = positions[sid]
            lo, hi = float(bar["low"]), float(bar["high"])
            stop_hit, target_hit = lo <= pos["stop"], hi >= pos["target"]
            if stop_hit:
                close(sid, stamp, pos["stop"], "STOP", ambiguous=target_hit)
                if target_hit:
                    ledger[-1]["exit_reason"] = "AMBIGUOUS_INTRABAR_STOP_FIRST"
            elif target_hit:
                close(sid, stamp, pos["target"], "TARGET")
        # Update close marks after all bar decisions; no future mark is used.
        for sid, _, bar in current_bars:
            marks[sid] = float(bar["close"])
        equity = cash + sum(p["quantity"] * marks.get(sid, p["entry_fill"]) for sid, p in positions.items())
        peak = max(peak, equity)
        market_value = sum(p["quantity"] * marks.get(sid, p["entry_fill"]) for sid, p in positions.items())
        equity_rows.append({"timestamp": stamp.isoformat(), "cash": cash, "market_value_open_positions": market_value, "equity": equity,
            "available_cash": cash, "open_positions": len(positions),
            "cumulative_transaction_costs": cumulative_costs,
            "realized_net_pnl": sum(float(r["net_pnl"]) for r in ledger),
            "unrealized_gross_pnl": sum((marks.get(sid, p["entry_fill"])-p["entry_fill"])*p["quantity"]
                                         for sid, p in positions.items()),
            "drawdown_pct": (equity/peak-1)*100 if peak else 0.0})

    last_stamp = available_times[-1]
    for sid, pos in sorted(positions.items()):
        mark = marks.get(sid, pos["entry_fill"])
        unrealized = (mark-pos["entry_fill"])*pos["quantity"]
        ledger.append({**pos["base"], "status": "OPEN", "exit_timestamp": "", "exit_price": "",
            "exit_reason": "OPEN_AT_PERIOD_END", "gross_pnl": "", "entry_costs": pos["entry_cost"],
            "exit_costs": 0.0, "transaction_costs": pos["entry_cost"], "net_pnl": "",
            "gross_R": "", "net_R": "", "holding_hourly_bars": pos["held_bars"],
            "holding_hours": (_stamp(last_stamp)-_stamp(pos["base"]["entry_timestamp"])).total_seconds()/3600,
            "holding_sessions": len(pos["held_sessions"]),
            "mae_pct": pos["mae_pct"], "mfe_pct": pos["mfe_pct"],
            "mark_timestamp": last_stamp.isoformat(), "mark_price": mark,
            "unrealized_gross_pnl": unrealized,
            "estimated_liquidation_value_after_exit_cost": mark*pos["quantity"] -
                costs.calculate(mark*pos["quantity"], side="sell", include_dp_charge=True).total})
    daily_equity: dict[str, dict[str, Any]] = {}
    for row in equity_rows:
        if date.fromisoformat(row["timestamp"][:10]) >= period_start:
            daily_equity[row["timestamp"][:10]] = {"date": row["timestamp"][:10], **row}
    equity_rows = [daily_equity[day] for day in sorted(daily_equity)]
    return {"trades": ledger, "equity_curve": equity_rows, "events": event_rows,
            "simultaneous_groups": simultaneous_groups, "rejections": dict(sorted(rejected_counts.items())),
            "sizing_constraints": dict(sorted(sizing_counts.items())),
            "metrics": _metrics(ledger, equity_rows, initial_capital), "end_timestamp": last_stamp.isoformat()}


def _breakdowns(trades: Sequence[Mapping[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in trades:
        if row.get("status") != "CLOSED":
            continue
        value = row.get(key)
        label = "UNKNOWN" if value is None or value == "" else str(value)
        bucket = result.setdefault(label, {"trade_count": 0, "wins": 0, "gross_pnl": 0.0,
            "net_pnl": 0.0, "gross_profit": 0.0, "gross_loss": 0.0,
            "costs": 0.0, "net_R_values": []})
        bucket["trade_count"] += 1
        bucket["wins"] += float(row["net_pnl"]) > 0
        bucket["gross_pnl"] += float(row["gross_pnl"])
        bucket["net_pnl"] += float(row["net_pnl"])
        bucket["costs"] += float(row["transaction_costs"])
        bucket["gross_profit"] += max(0.0, float(row["gross_pnl"]))
        bucket["gross_loss"] += min(0.0, float(row["gross_pnl"]))
        bucket.setdefault("net_profit", 0.0)
        bucket.setdefault("net_loss", 0.0)
        bucket["net_profit"] += max(0.0, float(row["net_pnl"]))
        bucket["net_loss"] += min(0.0, float(row["net_pnl"]))
        if row.get("net_R") not in (None, ""):
            bucket["net_R_values"].append(float(row["net_R"]))
    for bucket in result.values():
        n = bucket["trade_count"]
        bucket["win_rate"] = bucket["wins"] / n if n else 0.0
        bucket["gross_expectancy"] = bucket["gross_pnl"] / n if n else 0.0
        bucket["net_expectancy"] = bucket["net_pnl"] / n if n else 0.0
        bucket["expectancy_R"] = sum(bucket["net_R_values"])/n if n else 0.0
        bucket["average_R"] = bucket["expectancy_R"]
        bucket["profit_factor"] = bucket["net_profit"]/abs(bucket["net_loss"]) if bucket["net_loss"] else None
        bucket["gross_profit_factor"] = bucket["gross_profit"]/abs(bucket["gross_loss"]) if bucket["gross_loss"] else None
        bucket.pop("net_R_values")
    return result


def run(*, signal_trace: str | Path = SIGNAL_DIR / "strategy_trace_2023_2024.csv",
        replay_report: str | Path = SIGNAL_DIR / "top_down_swing_v1_replay_2023_2024.json",
        output_dir: str | Path = OUTPUT_DIR, data_root: str | Path = DATA_ROOT,
        daily_manifest: str | Path = MANIFEST_PATH, hourly_manifest: str | Path = HOURLY_MANIFEST,
        start: date = START, end: date = END) -> dict[str, Any]:
    if start < START or end > END or start > end:
        raise ValueError("Portfolio replay is restricted to the frozen 2023-2024 period.")
    report = json.loads(Path(replay_report).read_text(encoding="utf-8"))
    if report.get("strategy_version") != "top_down_swing_v1" or report.get("2025_2026_loaded", False):
        raise ValueError("Replay report is not a valid frozen top_down_swing_v1 source.")
    if report.get("data_hashes", {}).get("daily_prefix_sha256") is None:
        raise ValueError("Replay report lacks frozen input hashes.")
    signals = _load_signals(Path(signal_trace), start, end)
    expected_ready = report.get("transition_counts", {}).get("ENTRY_READY_EVENT")
    if start == START and end == END and expected_ready is not None and int(expected_ready) != len(signals):
        raise BacktestDataError("Saved ready-event trace count differs from its frozen replay report.")
    bars, hourly_hash = load_hourly_dataset_through(universe_id=PIT_TRADABLE,
        start_date=date(2022, 10, 1), end_date=end, data_root=data_root,
        manifest_path=hourly_manifest, daily_manifest_path=daily_manifest)
    missing = sorted({str(s["security_id"]) for s in signals} - set(bars))
    if missing:
        raise BacktestDataError(f"ENTRY_READY signals lack eligible hourly data: {missing}")
    expected_hash = report.get("data_hashes", {}).get("hourly_prefix_sha256")
    if start == START and end == END and hourly_hash != expected_hash:
        raise BacktestDataError("Loaded hourly prefix hash differs from saved frozen strategy replay.")
    costs = _cost_config(PROJECT_ROOT / "config" / "settings.yaml")
    raw = _run_portfolio(signals, bars, costs, period_start=start)
    repeated = _run_portfolio(signals, bars, costs, period_start=start)
    deterministic = raw == repeated
    if not deterministic:
        raise RuntimeError("Repeated portfolio simulations were not deterministic.")
    # The raw comparator removes costs from the exact same execution ledger.
    # Quantities, fills, exits, and timing therefore remain identical.
    zero_trades = []
    for item in raw["trades"]:
        row = dict(item)
        if row["status"] == "CLOSED":
            row["transaction_costs"] = 0.0
            row["entry_costs"] = 0.0
            row["exit_costs"] = 0.0
            row["net_pnl"] = row["gross_pnl"]
            row["net_R"] = row["gross_R"]
        else:
            row["transaction_costs"] = 0.0
            row["entry_costs"] = 0.0
            row["estimated_liquidation_value_after_exit_cost"] = float(row.get("mark_price", 0))*int(row["quantity"])
        row["transaction_cost_scenario"] = "RAW_ZERO_COST"
        zero_trades.append(row)
    zero_curve = []
    for item in raw["equity_curve"]:
        row = dict(item)
        row["cash"] = float(row["cash"]) + float(row["cumulative_transaction_costs"])
        row["available_cash"] = row["cash"]
        row["equity"] = float(row["equity"]) + float(row["cumulative_transaction_costs"])
        zero_curve.append(row)
    zero = {**raw, "trades": zero_trades, "equity_curve": zero_curve,
            "metrics": _metrics(zero_trades, zero_curve, 10000.0)}
    base = Path(output_dir)
    base.mkdir(parents=True, exist_ok=True)
    _write_csv(base / "trades_2023_2024.csv", raw["trades"])
    _write_csv(base / "equity_curve_2023_2024.csv", raw["equity_curve"])
    _write_csv(base / "trades_raw_zero_costs_2023_2024.csv", zero["trades"])
    _write_csv(base / "equity_curve_raw_zero_costs_2023_2024.csv", zero["equity_curve"])
    _write_csv(base / "portfolio_events_2023_2024.csv", raw["events"])
    _write_csv(base / "simultaneous_signal_groups_2023_2024.csv", raw["simultaneous_groups"])
    _write_csv(base / "skipped_signals_2023_2024.csv",
               [r for r in raw["events"] if r.get("event") == "REJECT"])
    # Stable, auditable review population: 20 wins, 20 losses, all gap/ambiguous exits, 10 cash skips if present.
    closed = [r for r in raw["trades"] if r["status"] == "CLOSED"]
    wins = sorted((r for r in closed if float(r["net_pnl"]) > 0), key=lambda r: r["setup_instance_id"])
    losses = sorted((r for r in closed if float(r["net_pnl"]) <= 0), key=lambda r: r["setup_instance_id"])
    review = wins[:20] + losses[:20] + [r for r in closed if r["exit_reason"] in
        {"GAP_STOP", "AMBIGUOUS_INTRABAR_STOP_FIRST"}]
    signal_lookup = {s["setup_instance_id"]: s for s in signals}
    reject_rows = [r for r in raw["events"] if r.get("event") == "REJECT" and r.get("reason") in
        {"ZERO_QUANTITY_CAPITAL_CONSTRAINT", "INSUFFICIENT_CASH", "CAPITAL_ALLOCATED_TO_PRIOR_SIGNAL"}]
    for event in reject_rows[:10]:
        source = signal_lookup.get(event.get("setup_instance_id"), {})
        plan = source.get("plan", {})
        review.append({**event, "symbol": source.get("symbol"), "setup_type": source.get("setup_type"),
            "confirmation_type": (source.get("result") or {}).get("confirmation_type"),
            "entry_timestamp": source.get("executable_timestamp"), "entry_price": plan.get("entry"),
            "stop_price": plan.get("stop"), "target_price": plan.get("target"),
            "target_type": plan.get("target_timeframe"), "planned_rr": plan.get("planned_reward_r"),
            "weekly_selection_path": source.get("weekly_selection_path"),
            "confirmation_volume_relative": ((source.get("result") or {}).get("confirmation_features") or {}).get("same_time_relative_volume"),
            "ema_context": plan.get("ema_context"), "stop_zone_id": plan.get("stop_zone_id"),
            "target_zone_id": plan.get("target_zone_id")})
    unique_review = {str(r.get("setup_instance_id") or f"{r.get('timestamp')}|{r.get('security_id')}|{r.get('reason')}"): r for r in review}
    _write_csv(base / "manual_trade_audit_2023_2024.csv", list(unique_review.values()))
    summary = {"strategy_version": "top_down_swing_v1", "data_freeze_version": "NIFTY50_PIT_2023_2024_V1",
        "period": {"start": start.isoformat(), "end": end.isoformat()}, "signals_source": str(signal_trace),
        "signal_events": len(signals), "signal_trace_sha256": hashlib.sha256(Path(signal_trace).read_bytes()).hexdigest(),
        "data_hashes": {"hourly_prefix_sha256": hourly_hash,
            "frozen_replay_hourly_sha256": expected_hash, "frozen_replay_daily_sha256": report["data_hashes"]["daily_prefix_sha256"]},
        "assumptions": {"capital": 10000, "risk_fraction_current_equity": .01, "long_only_cash_no_leverage": True,
            "no_pyramiding": True, "same_bar_stop_target": "stop first; marked ambiguous",
            "current_costs_are_current_assumptions_not_historical": True,
            "target_stop_frozen_at_entry": True, "trailing_time_ema_exits": False,
            "2025_2026_loaded": False},
        "current_cost_scenario": raw["metrics"], "raw_zero_cost_scenario": zero["metrics"],
        "current_cost_model": asdict(costs),
        "cost_impact": {"transaction_drag": raw["metrics"]["transaction_costs"],
            "net_return_difference_percentage_points": (raw["metrics"]["net_return_mtm"]-
                zero["metrics"]["net_return_mtm"])*100,
            "profit_factor_difference": ((raw["metrics"]["profit_factor"] or 0)-
                (zero["metrics"]["profit_factor"] or 0)),
            "expectancy_difference_rupees": raw["metrics"]["net_expectancy"]-zero["metrics"]["net_expectancy"]},
        "rejections": raw["rejections"], "simultaneous_group_count": len(raw["simultaneous_groups"]),
        "breakdowns": {key: _breakdowns(raw["trades"], key) for key in
            ("setup_family", "confirmation_archetype", "weekly_selection_path", "target_type",
             "volume_supported_confirmation", "ema_confluence", "overhead_friction_present")},
        "signal_counts_by_setup_family": {name: sum(s.get("setup_type") == name for s in signals)
            for name in ("BREAKOUT_RETEST", "SUPPORT_REACTION")},
        "planned_rr_buckets": _breakdowns(
            [{**r, "planned_rr_bucket": ("<1.50" if float(r.get("planned_rr") or 0) < 1.5 else
                "1.50-1.74" if float(r.get("planned_rr") or 0) < 1.75 else
                "1.75-1.99" if float(r.get("planned_rr") or 0) < 2.0 else
                "2.00-2.49" if float(r.get("planned_rr") or 0) < 2.5 else "2.50+")}
             for r in raw["trades"]], "planned_rr_bucket"),
        "descriptive_volume_ema": {
            "volume_relative_mean": (sum(float(r["confirmation_volume_relative"]) for r in raw["trades"]
                if r.get("status") == "CLOSED" and r.get("confirmation_volume_relative") not in (None, "")) /
                max(1, sum(r.get("status") == "CLOSED" and r.get("confirmation_volume_relative") not in (None, "")
                           for r in raw["trades"]))),
            "daily_ema50_present_closed_trades": sum(r.get("status") == "CLOSED" and r.get("daily_ema50") not in (None, "") for r in raw["trades"]),
            "daily_ema200_present_closed_trades": sum(r.get("status") == "CLOSED" and r.get("daily_ema200") not in (None, "") for r in raw["trades"]),
        },
        "causal_audits": {"all_executions_after_confirmation": all(
            _stamp(r["entry_timestamp"]) > _stamp(r["confirmation_timestamp"]) for r in raw["trades"]),
            "saved_frozen_replay_is_signal_source": True,
            "signal_replay_did_not_use_signal_day_hourly_bars": bool(report.get("daily_signal_hourly_data_excluded")),
            "signal_replay_loaded_no_2025_2026": not report.get("2025_2026_loaded", False),
            "all_exits_after_entries": all(r.get("status") != "CLOSED" or
                _stamp(r["exit_timestamp"]) > _stamp(r["entry_timestamp"]) for r in raw["trades"]),
            "all_executed_longs_have_stop_below_entry_and_target_above": all(
                float(r["stop_price"]) < float(r["entry_price"]) < float(r["target_price"])
                for r in raw["trades"]),
            "all_quantities_are_positive_integers": all(int(r["quantity"]) > 0 and float(r["quantity"]).is_integer()
                for r in raw["trades"]),
            "costs_non_negative": all(float(r.get("transaction_costs", 0)) >= 0 for r in raw["trades"]),
            "cash_never_negative": min((float(r["cash"]) for r in raw["equity_curve"]), default=10000) >= -1e-8,
            "no_leverage": min((float(r["cash"]) for r in raw["equity_curve"]), default=10000) >= -1e-8,
            "no_duplicate_setup_instance_ids_in_signal_source": len({s["setup_instance_id"] for s in signals}) == len(signals),
            "deterministic_repeated_run": deterministic,
            "no_2025_2026_loaded": not report.get("2025_2026_loaded", False),
            "known_hourly_exceptions_preserved": report.get("known_hourly_exceptions", {})},
        "skipped_signal_counts": raw["rejections"], "sizing_constraints": raw["sizing_constraints"],
        "exit_reason_counts": {reason: sum(r.get("status") == "CLOSED" and r.get("exit_reason") == reason for r in raw["trades"])
            for reason in sorted({str(r.get("exit_reason")) for r in raw["trades"] if r.get("status") == "CLOSED"})},
        "paths": {"trade_ledger": str(base / "trades_2023_2024.csv"),
            "raw_zero_trade_ledger": str(base / "trades_raw_zero_costs_2023_2024.csv"),
            "equity_curve": str(base / "equity_curve_2023_2024.csv"),
            "skipped_signals": str(base / "skipped_signals_2023_2024.csv"),
            "simultaneous_groups": str(base / "simultaneous_signal_groups_2023_2024.csv"),
            "performance_breakdowns": str(base / "performance_breakdowns_2023_2024.json"),
            "manual_audit": str(base / "manual_trade_audit_2023_2024.csv")}}
    # Metrics sliced from the same event stream, without rerunning the strategy.
    yearly = {}
    for year in (2023, 2024):
        year_text = str(year)
        curve = [r for r in raw["equity_curve"] if r["timestamp"][:4] == year_text]
        prior = [r for r in raw["equity_curve"] if r["timestamp"][:4] < year_text]
        start_equity = float(prior[-1]["equity"]) if prior else 10000.0
        end_equity = float(curve[-1]["equity"]) if curve else start_equity
        exits = [r for r in raw["trades"] if r.get("status") == "CLOSED" and r.get("exit_timestamp", "")[:4] == year_text]
        raw_exits = [r for r in zero["trades"] if r.get("status") == "CLOSED" and r.get("exit_timestamp", "")[:4] == year_text]
        yearly[year_text] = {"calendar_starting_equity": start_equity,
            "calendar_ending_equity_mtm": end_equity,
            "calendar_equity_return": end_equity / start_equity - 1 if start_equity else 0.0,
            "return_contribution_from_closed_trades": sum(float(r["net_pnl"]) for r in exits)/10000.0 if exits else 0.0,
            "closed_trade_metrics_by_exit_year_current_costs": _metrics(exits, [], start_equity),
            "closed_trade_metrics_by_exit_year_raw_zero_costs": _metrics(raw_exits, [], start_equity),
            "entries": sum(r.get("entry_timestamp", "")[:4] == year_text for r in raw["trades"]),
            "exits": len(exits), "transaction_costs_on_entries": sum(float(r.get("entry_costs", 0))
                for r in raw["trades"] if r.get("entry_timestamp", "")[:4] == year_text),
            "transaction_costs_on_exits": sum(float(r.get("exit_costs", 0)) for r in exits)}
    summary["by_entry_year"] = yearly
    summary["end_of_period_open_positions"] = [r for r in raw["trades"] if r.get("status") == "OPEN"]
    summary["setup_family_signal_and_execution_counts"] = {family: {
        "signals": sum(s.get("setup_type") == family for s in signals),
        "executed_entries": sum(r.get("setup_family") == family for r in raw["trades"])}
        for family in ("BREAKOUT_RETEST", "SUPPORT_REACTION")}
    summary_path = base / "portfolio_summary_2023_2024.json"
    breakdown_path = base / "performance_breakdowns_2023_2024.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")
    breakdown_path.write_text(json.dumps({"breakdowns": summary["breakdowns"],
        "planned_rr_buckets": summary["planned_rr_buckets"],
        "signal_counts_by_setup_family": summary["signal_counts_by_setup_family"]},
        indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")
    compact_metrics = ("executed_trade_count", "trade_count", "open_positions", "win_rate",
        "net_expectancy", "expectancy_R", "profit_factor", "gross_pnl", "net_pnl",
        "transaction_costs", "ending_equity_mtm", "net_return_mtm", "daily_max_drawdown_pct")
    print(json.dumps({"current_costs": {k: raw["metrics"][k] for k in compact_metrics},
                      "raw_zero_costs": {k: zero["metrics"][k] for k in compact_metrics},
                      "rejections": raw["rejections"], "simultaneous_groups": len(raw["simultaneous_groups"]),
                      "summary": str(summary_path)}, indent=2, allow_nan=False))
    return {"summary": summary, "current": raw, "raw_zero": zero}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, default=SIGNAL_DIR / "strategy_trace_2023_2024.csv")
    parser.add_argument("--replay-report", type=Path, default=SIGNAL_DIR / "top_down_swing_v1_replay_2023_2024.json")
    parser.add_argument("--output-directory", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--daily-manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--hourly-manifest", type=Path, default=HOURLY_MANIFEST)
    parser.add_argument("--start", type=date.fromisoformat, default=START)
    parser.add_argument("--end", type=date.fromisoformat, default=END)
    args = parser.parse_args()
    try:
        run(signal_trace=args.trace, replay_report=args.replay_report, output_dir=args.output_directory,
            data_root=args.data_root, daily_manifest=args.daily_manifest, hourly_manifest=args.hourly_manifest,
            start=args.start, end=args.end)
    except (ValueError, OSError, BacktestDataError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
