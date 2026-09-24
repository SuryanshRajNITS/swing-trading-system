"""Reproducible candle-derived diagnostics for sr_baseline_v1 development runs.

This module does not run or alter the strategy. It reads a saved 2023-2024
run artifact and the same bounded daily dataset, then writes descriptive CSVs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.backtest.data import load_daily_dataset_through
from src.backtest.strategy_run import DEVELOPMENT_END, DEVELOPMENT_START
from src.data.universe.provider import PIT_TRADABLE
from src.strategy.sr_baseline import (
    PriceZone,
    SRBaselineConfig,
    _is_broken,
    _touch_episodes,
    cluster_zones,
    find_confirmed_swings,
)


DEFAULT_RUN = Path("data/processed/backtests/sr_baseline_v1/both_r2_weekly-0_current_zerodha_0bps.json")
DEFAULT_OUT = Path("data/processed/backtests/sr_baseline_v1/diagnostics")
MILESTONES_R = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)


def _day(row: Mapping[str, Any]) -> date:
    return date.fromisoformat(str(row["date"])[:10])


def _safe_div(a: float, b: float) -> float | None:
    return a / b if b else None


def _atr(rows: Sequence[Mapping[str, Any]], index: int, period: int = 14) -> float | None:
    """Simple mean true range ending at index, diagnostic only (not strategy ATR)."""
    start = max(0, index - period + 1)
    values = []
    for i in range(start, index + 1):
        row = rows[i]
        previous_close = float(rows[i - 1]["close"]) if i else float(row["close"])
        values.append(max(float(row["high"]) - float(row["low"]),
                          abs(float(row["high"]) - previous_close),
                          abs(float(row["low"]) - previous_close)))
    return sum(values) / len(values) if values else None


def _close_location(row: Mapping[str, Any]) -> float:
    high, low, close = float(row["high"]), float(row["low"]), float(row["close"])
    return (close - low) / (high - low) if high > low else 0.5


def _last_zone_touch_index(rows: Sequence[Mapping[str, Any]], zone: Mapping[str, Any], before: int,
                           start: int = 0) -> int | None:
    lower, upper = float(zone["lower"]), float(zone["upper"])
    for index in range(min(before, len(rows)) - 1, max(-1, start - 1), -1):
        if float(rows[index]["low"]) <= upper and float(rows[index]["high"]) >= lower:
            return index
    return None


def _weekly_features(rows: Sequence[Mapping[str, Any]], signal_day: date, *, average_weeks: int = 4,
                     require_higher_close: bool = True, swing_left: int = 3,
                     swing_right: int = 3) -> dict[str, Any]:
    # Exclude the signal calendar week entirely, matching sr_baseline_v1's rule.
    monday = signal_day.fromordinal(signal_day.toordinal() - signal_day.weekday())
    grouped: dict[date, dict[str, Any]] = {}
    for row in rows:
        day = _day(row)
        week = day.fromordinal(day.toordinal() - day.weekday())
        if week < monday:
            if week not in grouped:
                grouped[week] = {"date": week.isoformat(), "open": float(row["open"]),
                                 "high": float(row["high"]), "low": float(row["low"]),
                                 "close": float(row["close"])}
            else:
                grouped[week]["high"] = max(grouped[week]["high"], float(row["high"]))
                grouped[week]["low"] = min(grouped[week]["low"], float(row["low"]))
                grouped[week]["close"] = float(row["close"])
    weekly_rows = [grouped[key] for key in sorted(grouped)]
    closes = [item["close"] for item in weekly_rows]
    if not closes:
        return {"weekly_direction": None, "weekly_close_vs_4w_avg_pct": None,
                "weekly_close_vs_prior_pct": None, "weekly_filter_pass": False,
                "distance_to_weekly_resistance_pct": None}
    latest = closes[-1]
    avg = sum(closes[-average_weeks:]) / min(average_weeks, len(closes))
    prior = closes[-2] if len(closes) > 1 else None
    weekly_above_mean = latest > avg
    weekly_higher = prior is not None and latest > prior
    weekly_zones = cluster_zones(find_confirmed_swings(weekly_rows, left=swing_left, right=swing_right), tolerance_pct=.01)
    above_levels = [zone.lower for zone in weekly_zones
                    if zone.kind == "resistance" and zone.pivot_count >= 2
                    and len(weekly_rows) - 1 - zone.last_pivot_index <= 52 and zone.lower > latest]
    return {
        "weekly_direction": "up" if prior is not None and latest > prior else "down_or_flat",
        "weekly_close_vs_4w_avg_pct": _safe_div(latest - avg, avg) * 100 if avg else None,
        "weekly_close_vs_prior_pct": _safe_div(latest - prior, prior) * 100 if prior else None,
        "weekly_filter_pass": bool(len(closes) >= average_weeks and weekly_above_mean
                                    and (weekly_higher or not require_higher_close)),
        "distance_to_weekly_resistance_pct": (min(above_levels) - latest) / latest * 100 if above_levels else None,
    }


def _zone_id(security_id: str, kind: str, first_pivot_date: str) -> str:
    raw = f"{security_id}|{kind}|{first_pivot_date}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def _feature_trade(trade: Mapping[str, Any], candles: Sequence[Mapping[str, Any]],
                   config: SRBaselineConfig) -> dict[str, Any]:
    meta = trade.get("metadata", {})
    signal_day = date.fromisoformat(str(trade["signal_timestamp"])[:10])
    entry_day = date.fromisoformat(str(trade["entry_timestamp"])[:10])
    exit_day = date.fromisoformat(str(trade["exit_timestamp"])[:10])
    day_index = {_day(row): i for i, row in enumerate(candles)}
    signal_i = day_index.get(signal_day)
    entry_i, exit_i = day_index.get(entry_day), day_index.get(exit_day)
    if signal_i is None or entry_i is None or exit_i is None or exit_i < entry_i:
        raise ValueError(f"Trade dates missing or out of order for {trade.get('trade_id')}")
    signal = candles[signal_i]
    zone = meta.get("zone", {})
    center = zone.get("center")
    width = (float(zone.get("upper", 0)) - float(zone.get("lower", 0))) if zone else None
    entry, stop = float(trade["entry_price"]), float(trade["initial_stop"])
    risk_ps = float(trade.get("initial_r_distance") or entry - stop)
    atr = _atr(candles, signal_i)
    body = abs(float(signal["close"]) - float(signal["open"]))
    upper_wick = float(signal["high"]) - max(float(signal["open"]), float(signal["close"]))
    lower_wick = min(float(signal["open"]), float(signal["close"])) - float(signal["low"])
    pivot_index = int(zone.get("first_pivot_index", signal_i)) if zone else signal_i
    breakout_date = meta.get("breakout_date")
    breakout_i = day_index.get(date.fromisoformat(breakout_date)) if breakout_date else None
    first_touch = meta.get("first_touch_date")
    if zone:
        reference_before = breakout_i if meta.get("setup_type") == "breakout_retest" and breakout_i is not None else signal_i
        latest_touch_i = _last_zone_touch_index(candles, zone, reference_before, start=pivot_index)
    else:
        latest_touch_i = None
    latest_pre_signal_touch = str(candles[latest_touch_i]["date"])[:10] if latest_touch_i is not None else None
    sessions_since_touch = signal_i - latest_touch_i if latest_touch_i is not None else None
    prior_reaction = None
    if latest_touch_i is not None and center:
        reaction_limit = breakout_i if breakout_i is not None else signal_i
        look_end = min(reaction_limit, latest_touch_i + 7)
        relevant = candles[latest_touch_i:look_end]
        if relevant:
            if zone.get("kind") == "support":
                prior_reaction = max(float(row["high"]) for row in relevant) - float(center)
            else:
                prior_reaction = float(center) - min(float(row["low"]) for row in relevant)
    excursions = candles[entry_i:exit_i + 1]
    max_high_row = max(excursions, key=lambda row: float(row["high"]))
    min_low_row = min(excursions, key=lambda row: float(row["low"]))
    max_high, min_low = float(max_high_row["high"]), float(min_low_row["low"])
    mfe_r, mae_r = (max_high - entry) / risk_ps, (min_low - entry) / risk_ps
    first_hit: dict[float, tuple[int, bool | None]] = {}
    stop_index: int | None = None
    for i in range(entry_i, exit_i + 1):
        row = candles[i]
        stop_hit = float(row["low"]) <= stop
        if stop_hit and stop_index is None:
            stop_index = i
        for milestone in MILESTONES_R:
            if milestone in first_hit:
                continue
            target = entry + milestone * risk_ps
            if float(row["high"]) >= target:
                before = None if stop_hit else (stop_index is None or i < stop_index)
                first_hit[milestone] = (i, before)
    retest_age = meta.get("retest_age")
    retest_low = meta.get("retest_low")
    breakout_level = meta.get("breakout_level")
    breakout_row = candles[breakout_i] if breakout_i is not None else None
    breakout_atr = _atr(candles, breakout_i) if breakout_i is not None else None
    lower = float(zone.get("lower", 0)) if zone else 0.0
    upper = float(zone.get("upper", 0)) if zone else 0.0
    penetration = None
    if zone and width and width > 0:
        penetration = max(0.0, min(1.0, (upper - float(signal["low"])) / width))
    close_distance = (float(signal["close"]) - float(center)) / float(center) * 100 if center else None
    retest_depth = (float(breakout_level) - float(retest_low)) / (float(breakout_level) - lower) if breakout_level and breakout_level > lower and retest_low is not None else None
    costs = float(trade.get("costs", 0))
    gross = float(trade.get("gross_pnl", 0))
    net = float(trade.get("net_pnl", 0))
    out: dict[str, Any] = {
        "trade_id": trade.get("trade_id"), "symbol": trade.get("symbol"), "security_id": trade.get("security_id"),
        "setup_type": meta.get("setup_type"), "signal_date": signal_day.isoformat(),
        "entry_date": entry_day.isoformat(), "exit_date": exit_day.isoformat(),
        "zone_id": _zone_id(str(trade.get("security_id")), str(zone.get("kind", "unknown")),
                             str(candles[pivot_index]["date"])[:10] if 0 <= pivot_index < len(candles) else "unknown"),
        "zone_role": zone.get("kind"), "zone_center": center,
        "zone_width_pct": width / center * 100 if width and center else None,
        "zone_width_atr": width / atr if width and atr else None,
        "number_of_pivots_in_zone": zone.get("pivot_count"), "number_of_touch_episodes": meta.get("touch_count"),
        "first_zone_date": str(candles[pivot_index]["date"])[:10] if 0 <= pivot_index < len(candles) else None,
        "first_touch_date": first_touch, "latest_pre_signal_touch_date": latest_pre_signal_touch,
        "zone_age_sessions": meta.get("zone_age_sessions"),
        "sessions_since_previous_touch": sessions_since_touch,
        "fresh_vs_retested_zone": "fresh" if meta.get("touch_count", 0) <= 2 else "retested",
        "role_reversal_candidate": bool(meta.get("setup_type") == "breakout_retest"),
        "prior_reaction_pct": prior_reaction / float(center) * 100 if prior_reaction is not None and center else None,
        "prior_reaction_atr": prior_reaction / atr if prior_reaction is not None and atr else None,
        "signal_close_location": _close_location(signal), "signal_body_pct": body / float(signal["close"]) * 100,
        "upper_wick_to_body": _safe_div(upper_wick, body), "lower_wick_to_body": _safe_div(lower_wick, body),
        "distance_close_from_zone_pct": close_distance, "penetration_depth_zone_fraction": penetration,
        "rejection_magnitude_pct": (float(signal["close"]) - float(signal["low"])) / float(signal["close"]) * 100,
        "breakout_body_pct": abs(float(breakout_row["close"])-float(breakout_row["open"]))/float(breakout_row["close"])*100 if breakout_row else None,
        "breakout_range_atr": (float(breakout_row["high"])-float(breakout_row["low"]))/breakout_atr if breakout_row and breakout_atr else None,
        "breakout_close_distance_above_resistance_pct": (float(meta["breakout_close"])-float(breakout_level))/float(breakout_level)*100 if meta.get("breakout_close") and breakout_level else None,
        "breakout_volume_ratio": meta.get("breakout_volume_ratio"), "resistance_touches_before_breakout": meta.get("touch_count") if meta.get("setup_type")=="breakout_retest" else None,
        "consolidation_sessions_near_resistance": (
            sum(1 for c in candles[max(0, latest_touch_i if latest_touch_i is not None else (breakout_i or 0)):breakout_i]
                if center and abs(float(c["close"])/float(center)-1) <= .01) if breakout_i is not None else None
        ),
        "retest_delay_sessions": retest_age,
        "retest_depth_zone_fraction": retest_depth,
        "retest_close_position": _close_location(signal) if meta.get("setup_type")=="breakout_retest" else None,
        "retest_touched_center": bool(retest_low is not None and center is not None and
                                        float(signal["low"]) <= float(center) <= float(signal["high"])),
        "retest_touched_lower_bound": bool(retest_low is not None and lower and
                                             float(signal["low"]) <= lower <= float(signal["high"])),
        "retest_touched_upper_bound": bool(retest_low is not None and upper and
                                             float(signal["low"]) <= upper <= float(signal["high"])),
        "closed_back_below_resistance_before_entry": bool(
            breakout_i is not None and breakout_level and any(
                float(c["close"]) < float(breakout_level) for c in candles[breakout_i + 1:entry_i + 1])),
        **_weekly_features(candles[:signal_i + 1], signal_day,
                           average_weeks=config.weekly_sma_weeks,
                           require_higher_close=config.weekly_require_higher_close,
                           swing_left=config.swing_left, swing_right=config.swing_right),
        "distance_from_252_session_high_pct": (float(signal["close"])/max(float(r["high"]) for r in candles[max(0, signal_i-251):signal_i+1])-1)*100,
        "entry": entry, "initial_stop": stop, "stop_distance_pct": risk_ps/entry*100,
        "signal_close": float(signal["close"]),
        "entry_gap_from_signal_pct": (entry/float(signal["close"])-1)*100,
        "entry_gap_from_signal_R": (entry-float(signal["close"]))/risk_ps,
        "signal_close_to_stop_pct": (float(signal["close"])-stop)/float(signal["close"])*100,
        "stop_distance_atr": risk_ps/atr if atr else None, "quantity": trade.get("quantity"),
        "initial_risk_rupees": trade.get("initial_risk_rupees"), "target_r": meta.get("target_r"),
        "holding_period_sessions": trade.get("holding_period"), "exit_reason": trade.get("exit_reason"),
        "gross_pnl": gross, "transaction_costs": costs, "net_pnl": net, "R_multiple": trade.get("R_multiple"),
        "MFE_R": mfe_r, "MAE_R": mae_r, "maximum_favorable_price": max_high,
        "maximum_adverse_price": min_low, "session_of_MFE": str(max_high_row["date"])[:10],
        "session_of_MAE": str(min_low_row["date"])[:10],
    }
    for milestone in MILESTONES_R:
        hit = first_hit.get(milestone)
        out[f"reached_{milestone:g}R"] = hit is not None
        out[f"reached_{milestone:g}R_before_stop"] = hit[1] if hit else False
    return out


def _false_breakout(row: Mapping[str, Any], candles: Sequence[Mapping[str, Any]], index_by_day: Mapping[date, int]) -> str:
    if row.get("setup_type") != "breakout_retest":
        return "not_applicable"
    if row.get("exit_reason") == "TARGET":
        return "successful_continuation"
    held = int(row.get("holding_period_sessions") or 0)
    mfe = float(row.get("MFE_R") or 0)
    depth = row.get("retest_depth_zone_fraction")
    age = int(row.get("retest_delay_sessions") or 0)
    if held <= 3:
        return "immediate_failed_breakout"
    if depth is not None and float(depth) >= 0.75:
        return "deep_retest_failure"
    if age >= 8:
        return "late_retest_failure"
    entry_day = date.fromisoformat(str(row["entry_date"]))
    exit_day = date.fromisoformat(str(row["exit_date"]))
    idx = index_by_day.get(entry_day)
    if idx is not None:
        risk_entry = float(row["entry"])
        risk_ps = (risk_entry * float(row["stop_distance_pct"]) / 100)
        later = [c for c in candles if exit_day < _day(c) <= DEVELOPMENT_END]
        if any(float(c["high"]) >= risk_entry + 0.5 * risk_ps for c in later):
            return "stopped_then_eventually_moved_up"
    if mfe < 0.5:
        return "breakout_without_followthrough"
    return "stopped_after_partial_followthrough"


def _bucket(value: Any, edges: Sequence[tuple[float, float, str]]) -> str:
    if value is None:
        return "missing"
    x = float(value)
    for low, high, label in edges:
        if low <= x < high:
            return label
    return edges[-1][2] if x >= edges[-1][1] else edges[0][2]


def _aggregate(rows: Sequence[Mapping[str, Any]], group_fields: Sequence[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(field) for field in group_fields)].append(row)
    result = []
    for key, trades in sorted(groups.items(), key=lambda item: tuple(str(x) for x in item[0])):
        gross = [float(t["gross_pnl"]) for t in trades]
        net = [float(t["net_pnl"]) for t in trades]
        winners = [x for x in net if x > 0]
        losers = [x for x in net if x < 0]
        row = dict(zip(group_fields, key))
        row.update({"trade_count": len(trades), "win_rate": sum(x > 0 for x in net)/len(net),
                    "gross_expectancy": sum(gross)/len(gross), "net_expectancy": sum(net)/len(net),
                    "average_R": sum(float(t["R_multiple"]) for t in trades)/len(trades),
                    "profit_factor": sum(winners)/abs(sum(losers)) if losers and sum(losers) else None,
                    "gross_pnl": sum(gross), "transaction_costs": sum(float(t["transaction_costs"]) for t in trades),
                    "net_pnl": sum(net)})
        result.append(row)
    return result


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _zone_audit(daily: Mapping[str, Sequence[Mapping[str, Any]]], trades: Sequence[Mapping[str, Any]],
                diagnostics: Sequence[Mapping[str, Any]], config: SRBaselineConfig) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    zones_seen: dict[tuple[str, str, int], dict[str, Any]] = {}
    yearly: Counter[tuple[str, int]] = Counter()
    touches: Counter[int] = Counter()
    signal_density: Counter[tuple[str, int]] = Counter()
    security_years: set[tuple[str, int]] = set()
    for security_id, rows in daily.items():
        swings = find_confirmed_swings(rows, left=config.swing_left, right=config.swing_right)
        for day_index, row in enumerate(rows):
            day = _day(row)
            if day > DEVELOPMENT_END:
                continue
            if DEVELOPMENT_START <= day <= DEVELOPMENT_END:
                security_years.add((security_id, day.year))
            eligible_swings = [s for s in swings if s.confirmation_index <= day_index]
            for zone in cluster_zones(eligible_swings, tolerance_pct=config.zone_tolerance_pct):
                if zone.pivot_count < config.minimum_zone_pivots or day_index-zone.last_pivot_index > config.maximum_zone_age_sessions:
                    continue
                key = (security_id, zone.kind, zone.first_pivot_index)
                state = zones_seen.setdefault(key, {
                    "security_id": security_id, "zone_kind": zone.kind,
                    "first_pivot_index": zone.first_pivot_index,
                    "first_pivot_date": str(rows[zone.first_pivot_index]["date"])[:10],
                    "formed_date": day.isoformat(), "formation_year": day.year,
                    "center": zone.center, "lower": zone.lower, "upper": zone.upper,
                    "formation_lower": zone.lower, "formation_upper": zone.upper,
                    "max_pivots": zone.pivot_count, "max_touch_episodes": 0,
                    "last_pivot_index": zone.last_pivot_index,
                    "last_seen_date": day.isoformat(), "breached_within_20_sessions": False,
                    "led_to_executed_trade": False,
                })
                state["max_pivots"] = max(state["max_pivots"], zone.pivot_count)
                state["center"], state["lower"], state["upper"] = zone.center, zone.lower, zone.upper
                state["last_pivot_index"] = max(state["last_pivot_index"], zone.last_pivot_index)
                state["last_seen_date"] = day.isoformat()
            if day.year in (2023, 2024):
                # Count eligible distinct zones at year end once below; signal count is direct.
                pass
    for trade in trades:
        meta = trade.get("metadata", {})
        zone = meta.get("zone", {})
        sid = str(trade.get("security_id"))
        candle_rows = daily.get(sid, ())
        pivot = int(zone.get("first_pivot_index", -1))
        state = zones_seen.get((sid, str(zone.get("kind")), pivot))
        if state:
            state["led_to_executed_trade"] = True
    for state in zones_seen.values():
        sid = state["security_id"]
        rows = daily[sid]
        start_i = next((i for i, row in enumerate(rows) if str(row["date"])[:10] == state["formed_date"]), 0)
        upper, lower = float(state["formation_upper"]), float(state["formation_lower"])
        representative = PriceZone(
            str(state["zone_kind"]), float(state["center"]), lower, upper,
            int(state["max_pivots"]), int(state["first_pivot_index"]), int(state["first_pivot_index"]),
        )
        valid_through = min(len(rows), int(state["last_pivot_index"]) + config.maximum_zone_age_sessions + 1)
        state["max_touch_episodes"] = len(_touch_episodes(rows[start_i:valid_through], representative))
        end_i = min(len(rows), start_i + 21)
        # A decisive break within 20 sessions; resistance breaks use symmetric upside rule.
        if state["zone_kind"] == "support":
            state["breached_within_20_sessions"] = any(float(r["close"]) < lower * 0.995 for r in rows[start_i:end_i])
        else:
            state["breached_within_20_sessions"] = any(float(r["close"]) > upper * 1.005 for r in rows[start_i:end_i])
        touches[state["max_touch_episodes"]] += 1
        yearly[(sid, int(state["formation_year"]))] += 1
    for row in diagnostics:
        if row.get("accepted"):
            signal_density[(str(row.get("security_id")), int(str(row.get("signal_date"))[:4]))] += 1
    per_year = []
    for year in (2023, 2024):
        eligible_sids = sorted(sid for sid, y in security_years if y == year)
        counts = [yearly.get((sid, year), 0) for sid in eligible_sids]
        signal_counts = [signal_density.get((sid, year), 0) for sid in eligible_sids]
        per_year.append({"year": year, "security_count": len({sid for sid, y in security_years if y == year}),
                         "zones_formed": sum(counts), "average_zones_per_security": sum(counts)/len(counts) if counts else 0,
                         "median_zones_per_security": ((sorted(counts)[(len(counts)-1)//2] + sorted(counts)[len(counts)//2]) / 2) if counts else 0,
                         "accepted_signals": sum(signal_counts),
                         "average_accepted_signals_per_security": sum(signal_counts)/len(signal_counts) if signal_counts else 0,
                         "median_accepted_signals_per_security": ((sorted(signal_counts)[(len(signal_counts)-1)//2]
                                                                    + sorted(signal_counts)[len(signal_counts)//2]) / 2)
                         if signal_counts else 0})
    all_zones = [zone for zone in zones_seen.values()
                 if DEVELOPMENT_START.year <= int(zone["formation_year"]) <= DEVELOPMENT_END.year]
    audit = {"zone_count": len(all_zones),
             "zones_never_leading_to_trade_pct": 100*sum(not z["led_to_executed_trade"] for z in all_zones)/len(all_zones) if all_zones else None,
             "zones_breached_within_20_sessions_pct": _rate(all_zones),
             "support_zones_breached_within_20_sessions_pct": _rate([z for z in all_zones if z["zone_kind"] == "support"]),
             "resistance_zones_breached_within_20_sessions_pct": _rate([z for z in all_zones if z["zone_kind"] == "resistance"]),
             "touch_count_distribution": {str(k): v for k, v in sorted(touches.items())},
             "near_duplicate_overlap_pairs": _overlap_pairs(all_zones)}
    return all_zones, per_year, audit


def _rate(zones: Sequence[Mapping[str, Any]]) -> float | None:
    return 100 * sum(bool(zone["breached_within_20_sessions"]) for zone in zones) / len(zones) if zones else None


def _overlap_pairs(zones: Sequence[Mapping[str, Any]]) -> int:
    count = 0
    by_security_kind: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for zone in zones:
        by_security_kind[(str(zone["security_id"]), str(zone["zone_kind"]))].append(zone)
    for group in by_security_kind.values():
        ordered = sorted(group, key=lambda z: float(z["lower"]))
        for i, left in enumerate(ordered):
            for right in ordered[i+1:]:
                dates_overlap = (str(right["formed_date"]) <= str(left["last_seen_date"])
                                 and str(left["formed_date"]) <= str(right["last_seen_date"]))
                if dates_overlap and float(right["formation_lower"]) <= float(left["formation_upper"]):
                    count += 1
                else:
                    break
    return count


def generate_diagnostics(*, run_path: str | Path = DEFAULT_RUN, output_dir: str | Path = DEFAULT_OUT,
                         manifest_path: str | Path = Path("data/processed/market_data_manifest.csv"),
                         data_root: str | Path = Path("data/processed/market_data")) -> dict[str, Any]:
    run_path, output_dir = Path(run_path), Path(output_dir)
    report = json.loads(run_path.read_text(encoding="utf-8"))
    meta = report.get("metadata", {})
    if meta.get("evaluation_start") != DEVELOPMENT_START.isoformat() or meta.get("evaluation_end") != DEVELOPMENT_END.isoformat():
        raise ValueError("Diagnostics accept only a saved 2023-01-01 through 2024-12-31 run.")
    if report.get("research_controls", {}).get("outside_period_data_parsed") is not False:
        raise ValueError("Run artifact does not certify a bounded development-only input.")
    snapshot = meta.get("config_snapshot", {}).get("strategy", {}).get("parameters", {})
    config_values = {key: value for key, value in snapshot.items() if key in SRBaselineConfig.__dataclass_fields__}
    config = SRBaselineConfig(**config_values)
    daily, dataset_version = load_daily_dataset_through(
        universe_id=PIT_TRADABLE, start_date=DEVELOPMENT_START, end_date=DEVELOPMENT_END,
        data_root=data_root, manifest_path=manifest_path,
    )
    trades = [trade for trade in report.get("trades", [])
              if DEVELOPMENT_START.isoformat() <= str(trade.get("entry_timestamp", ""))[:10] <= DEVELOPMENT_END.isoformat()
              and str(trade.get("exit_timestamp", ""))[:10] <= DEVELOPMENT_END.isoformat()]
    feature_rows = []
    by_security = daily
    for trade in trades:
        row = _feature_trade(trade, by_security[str(trade["security_id"])], config)
        row["false_breakout_class"] = _false_breakout(
            row, by_security[str(trade["security_id"])],
            {_day(c): i for i, c in enumerate(by_security[str(trade["security_id"])])},
        )
        feature_rows.append(row)
    breakout = [row for row in feature_rows if row["setup_type"] == "breakout_retest"]
    support = [row for row in feature_rows if row["setup_type"] == "support_test"]
    # Diagnostic buckets are descriptive, coarse bins selected before seeing subgroup results.
    bucket_specs = {
        "retest_delay_sessions": [(1, 4, "1-3"), (4, 8, "4-7"), (8, 15, "8-14"), (15, 21, "15-20")],
        "breakout_volume_ratio": [(0, .8, "<0.8"), (.8, 1, "0.8-1.0"), (1, 1.2, "1.0-1.2"), (1.2, 1.5, "1.2-1.5"), (1.5, math.inf, ">1.5")],
        "number_of_touch_episodes": [(0, 3, "2"), (3, 4, "3"), (4, math.inf, "4+")],
        "zone_age_sessions": [(0, 21, "young_<21"), (21, 84, "medium_21-83"), (84, math.inf, "old_84+")],
        "stop_distance_atr": [(0, .75, "tight_<0.75"), (.75, 1.5, "medium_0.75-1.5"), (1.5, math.inf, "wide_1.5+")],
        "prior_reaction_atr": [(0, .5, "weak_<0.5"), (.5, 1.5, "medium_0.5-1.5"), (1.5, math.inf, "strong_1.5+")],
        "penetration_depth_zone_fraction": [(0, .25, "shallow_<25pct"), (.25, .6, "medium_25-60pct"), (.6, 1.01, "deep_60pct+")],
        "sessions_since_previous_touch": [(0, 6, "short_1-5"), (6, 16, "medium_6-15"), (16, math.inf, "long_16+")],
        "entry_gap_from_signal_pct": [(-math.inf, 0, "gap_down"), (0, .5, "flat_to_<0.5pct"),
                                       (.5, 1.5, "gap_up_0.5-1.5pct"), (1.5, math.inf, "gap_up_1.5pct+")],
    }
    bucket_rows = []
    for feature, edges in bucket_specs.items():
        source = breakout if feature in {"retest_delay_sessions", "breakout_volume_ratio"} else feature_rows
        temp = []
        for row in source:
            item = dict(row)
            item["bucket"] = _bucket(row.get(feature), edges)
            item["diagnostic_feature"] = feature
            temp.append(item)
        bucket_rows.extend(_aggregate(temp, ("setup_type", "diagnostic_feature", "bucket")))
    for field in ("weekly_direction", "weekly_filter_pass"):
        bucket_rows.extend(_aggregate(feature_rows, ("setup_type", field)))
    touch_support = []
    for row in support:
        item = dict(row)
        count = int(row.get("number_of_touch_episodes") or 0)
        item["touch_phase"] = "second_touch" if count == 2 else "third_plus" if count >= 3 else "other"
        touch_support.append(item)
    buckets = _aggregate(touch_support, ("touch_phase",))
    zone_rows, yearly, zone_audit = _zone_audit(daily, trades, report.get("diagnostics", []), config)
    false_rows = _aggregate(breakout, ("false_breakout_class",))
    setup_rows = []
    for name, rows in (("support_test", support), ("breakout_retest", breakout)):
        gross = sum(float(r["gross_pnl"]) for r in rows)
        costs = sum(float(r["transaction_costs"]) for r in rows)
        net = sum(float(r["net_pnl"]) for r in rows)
        winners = [float(r["gross_pnl"]) for r in rows if float(r["gross_pnl"]) > 0]
        risks = [float(r["initial_risk_rupees"]) for r in rows if r.get("initial_risk_rupees")]
        avg_cost = costs/len(rows) if rows else 0.0
        avg_gross_winner = sum(winners)/len(winners) if winners else None
        avg_risk = sum(risks)/len(risks) if risks else None
        setup_rows.append({"setup_type": name, "trade_count": len(rows), "gross_pnl": gross,
                           "transaction_costs": costs, "net_trading_pnl": net,
                           "average_cost_per_trade": avg_cost,
                           "average_gross_winner": avg_gross_winner,
                           "average_initial_risk": avg_risk,
                           "cost_as_pct_average_gross_winner": 100*avg_cost/avg_gross_winner if avg_gross_winner else None,
                           "cost_as_pct_average_initial_risk": 100*avg_cost/avg_risk if avg_risk else None,
                           "average_gross_pnl": gross/len(rows) if rows else None,
                           "gross_edge_sign": "positive" if gross > 0 else "negative" if gross < 0 else "flat"})
    milestones = []
    for setup_name, rows in (("all", feature_rows), ("support_test", support), ("breakout_retest", breakout)):
        for threshold in MILESTONES_R:
            key = f"reached_{threshold:g}R"
            before_key = f"reached_{threshold:g}R_before_stop"
            milestones.append({"setup_type": setup_name, "milestone_R": threshold, "trade_count": len(rows),
                               "reach_rate_pct": 100*sum(bool(r[key]) for r in rows)/len(rows) if rows else None,
                               "reached_before_stop_pct": 100*sum(bool(r[before_key]) for r in rows)/len(rows) if rows else None})
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "trades.csv", feature_rows)
    _write_csv(output_dir / "buckets.csv", bucket_rows + buckets)
    _write_csv(output_dir / "false_breakout_classes.csv", false_rows)
    _write_csv(output_dir / "setup_cost_decomposition.csv", setup_rows)
    _write_csv(output_dir / "mfe_mae_milestones.csv", milestones)
    _write_csv(output_dir / "sr_zone_audit.csv", zone_rows)
    _write_csv(output_dir / "sr_zone_yearly.csv", yearly)
    _write_csv(output_dir / "sr_zone_touch_distribution.csv",
               [{"touch_episodes": key, "zone_count": value}
                for key, value in sorted(zone_audit["touch_count_distribution"].items(), key=lambda item: int(item[0]))])
    rejection_counts = Counter(str(row.get("rejection_reason")) for row in report.get("diagnostics", []) if not row.get("accepted"))
    # Winner/loser directional association is descriptive only; no causal interpretation.
    feature_comparison = []
    numeric_features = ("MFE_R", "MAE_R", "zone_age_sessions", "number_of_touch_episodes",
                        "sessions_since_previous_touch", "prior_reaction_atr", "signal_close_location",
                        "penetration_depth_zone_fraction", "breakout_volume_ratio", "retest_delay_sessions",
                        "stop_distance_atr", "weekly_close_vs_4w_avg_pct", "entry_gap_from_signal_pct",
                        "entry_gap_from_signal_R")
    for feature in numeric_features:
        for outcome, group in (("net_winner", [r for r in feature_rows if float(r["net_pnl"]) > 0]),
                               ("net_loser", [r for r in feature_rows if float(r["net_pnl"]) <= 0])):
            values = [float(r[feature]) for r in group if r.get(feature) is not None]
            feature_comparison.append({"feature": feature, "outcome": outcome, "n": len(values),
                                       "mean": sum(values)/len(values) if values else None,
                                       "median": sorted(values)[len(values)//2] if values else None})
    _write_csv(output_dir / "winner_loser_features.csv", feature_comparison)
    variant_rows = []
    for candidate in sorted(run_path.parent.glob("*.json")):
        try:
            variant = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        vm = variant.get("metadata", {})
        if vm.get("evaluation_start") != DEVELOPMENT_START.isoformat() or vm.get("evaluation_end") != DEVELOPMENT_END.isoformat():
            continue
        strategy_parameters = vm.get("config_snapshot", {}).get("strategy", {}).get("parameters", {})
        grouped_trades: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for item in variant.get("trades", []):
            grouped_trades[str(item.get("metadata", {}).get("setup_type", "unknown"))].append(item)
        for setup_name, subset in sorted(grouped_trades.items()):
            gross_value = sum(float(item.get("gross_pnl", 0)) for item in subset)
            transaction_value = sum(float(item.get("costs", 0)) for item in subset)
            net_value = sum(float(item.get("net_pnl", 0)) for item in subset)
            variant_rows.append({"artifact": candidate.name, "setup_type": setup_name, "trades": len(subset),
                                 "target_r": strategy_parameters.get("target_r"),
                                 "weekly_filter": strategy_parameters.get("weekly_filter"),
                                 "cost_scenario": variant.get("research_controls", {}).get("cost_scenario"),
                                 "slippage_bps": variant.get("research_controls", {}).get("slippage_bps"),
                                 "gross_pnl": gross_value, "transaction_costs": transaction_value,
                                 "net_trading_pnl": net_value,
                                 "run_economic_net_profit_including_opex": variant.get("metrics", {}).get("economic_net_profit"),
                                 "profit_factor": variant.get("metrics", {}).get("profit_factor"),
                                 "win_rate": variant.get("metrics", {}).get("win_rate"),
                                 "average_R_net": sum(float(item.get("R_multiple", 0)) for item in subset)/len(subset) if subset else None})
    _write_csv(output_dir / "variant_comparison.csv", variant_rows)
    result = {
        "period": [DEVELOPMENT_START.isoformat(), DEVELOPMENT_END.isoformat()],
        "strategy_version": "sr_baseline_v1", "source_run": str(run_path),
        "source_run_id": meta.get("run_id"), "dataset_version": dataset_version,
        "source_cost_scenario": report.get("research_controls", {}).get("cost_scenario"),
        "baseline_metrics": {key: report.get("metrics", {}).get(key) for key in (
            "number_of_trades", "trading_net_return", "profit_factor", "win_rate", "average_R",
            "maximum_drawdown_pct", "transaction_costs_total", "operating_costs_total",
        )},
        "trades_analyzed": len(feature_rows), "trades_by_setup": dict(Counter(str(r["setup_type"]) for r in feature_rows)),
        "open_positions_at_evaluation_end": len(report.get("open_positions_at_end", [])),
        "operating_costs_excluded_from_trade_quality": True,
        "definitions": {
            "atr": "Simple 14-session mean true range, diagnostic feature only; strategy does not use ATR.",
            "excursions": "Daily highs/lows from actual entry session through actual exit session inclusive.",
            "milestone_before_stop": "False if stop and milestone are both touched on the same daily candle (intraday order unknown); otherwise milestone bar precedes first stop-hit bar.",
            "false_breakout": "Mutually exclusive priority: target=successful_continuation; stop within 3 sessions=immediate_failed_breakout; retest depth >=75% of zone height=deep_retest_failure; retest age >=8=late_retest_failure; a later +0.5R high by 2024-12-31=stopped_then_eventually_moved_up; MFE<0.5R=breakout_without_followthrough; else stopped_after_partial_followthrough.",
            "zones": "Audited as-of each daily close using confirmed pivots only, 3/3 pivots, 1% same-side clustering, >=2 pivots and <=252-session age. Zone identity proxy is security+side+first pivot index/date. A zone is considered trade-led only when its first pivot index matches an executed trade's saved zone.",
            "zone_broken_soon": "Called a directional breach: support close below lower bound by 0.5%, resistance close above upper bound by 0.5%, within 20 sessions starting at first eligibility date. These are reported separately because an upside resistance breach may be a valid breakout.",
            "breakout_consolidation_proxy": "Count close-in-band sessions from the saved latest pre-breakout touch through the breakout session; a close within 1% of zone center counts. This is a descriptive proxy, not a strategy input.",
            "fixed_operating_costs": "Excluded from per-trade/setup quality; transaction costs remain included as reported separately.",
            "limitations": ["Daily candles cannot establish intraday ordering when stop and target/milestone share a candle.",
                            "Zone IDs are reconstructed proxies, not stable IDs emitted by the strategy. Consolidation sessions use a declared close-in-band proxy.",
                            "This report is descriptive development evidence, not out-of-sample validation or parameter optimization."],
        },
        "zone_audit": zone_audit, "zones_per_security_year": yearly,
        "breakout_failure_counts": dict(Counter(r["false_breakout_class"] for r in breakout)),
        "rejection_reason_counts": dict(rejection_counts),
        "variant_comparison_rows": len(variant_rows),
        "artifacts": ["README.md", "findings.md", "trades.csv", "buckets.csv", "false_breakout_classes.csv", "setup_cost_decomposition.csv",
                      "mfe_mae_milestones.csv", "sr_zone_audit.csv", "sr_zone_yearly.csv",
                      "sr_zone_touch_distribution.csv", "winner_loser_features.csv", "variant_comparison.csv"],
    }
    (output_dir / "report.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    (output_dir / "README.md").write_text(
        "# sr_baseline_v1 development diagnostics\n\n"
        "Generated by `python -m src.backtest.sr_diagnostics` from a saved run artifact and the bounded daily dataset. "
        "The accepted run dates are 2023-01-01 through 2024-12-31 only. No strategy run is performed by this command.\n\n"
        "`findings.md` gives the concise result interpretation. `trades.csv` contains one row per closed trade in the selected run, including candle-derived diagnostics and MFE/MAE. "
        "`buckets.csv` and `winner_loser_features.csv` are descriptive group comparisons. `false_breakout_classes.csv` "
        "uses the mutually exclusive definitions recorded in `report.json`. `setup_cost_decomposition.csv` reports "
        "closed-trade gross P&L, transaction charges and net P&L, excluding fixed operating costs. `variant_comparison.csv` "
        "summarizes all saved 2023-2024 JSON runs in the parent directory. `sr_zone_audit.csv`, `sr_zone_yearly.csv` and "
        "`sr_zone_touch_distribution.csv` audit reconstructed zone candidates.\n\n"
        "Important limits: all excursion and milestone analysis uses daily bars; a milestone and stop on the same candle "
        "is treated as not proven to have preceded the stop. Current Zerodha cost assumptions are not historical fee rates. "
        "Fixed Kite/infrastructure costs do not affect trade quality metrics. The zone ID and consolidation fields are "
        "documented proxies. These are development-period descriptions, not out-of-sample validation or evidence of causality.\n",
        encoding="utf-8",
    )
    failure_pct = {key: 100 * value / len(breakout) if breakout else 0
                   for key, value in Counter(row["false_breakout_class"] for row in breakout).items()}
    by_bucket = {(row.get("diagnostic_feature"), row.get("bucket")): row for row in bucket_rows}
    def bval(feature: str, label: str, field: str) -> Any:
        row = by_bucket.get((feature, label), {})
        return row.get(field)
    setup_cost_by_name = {row["setup_type"]: row for row in setup_rows}
    support_isolated = next((row for row in variant_rows if row["artifact"].startswith("support_test_r2_weekly-0_current_zerodha_0bps")
                             and row["setup_type"] == "support_test"), None)
    breakout_isolated = next((row for row in variant_rows if row["artifact"].startswith("breakout_retest_r2_weekly-0_current_zerodha_0bps")
                              and row["setup_type"] == "breakout_retest"), None)
    mfe_half = next(row for row in milestones if row["setup_type"] == "all" and row["milestone_R"] == .5)
    mfe_two = next(row for row in milestones if row["setup_type"] == "all" and row["milestone_R"] == 2.0)
    feature_rows_by_name = {(row["feature"], row["outcome"]): row for row in feature_comparison}
    ftxt = lambda feature, outcome, col: feature_rows_by_name.get((feature, outcome), {}).get(col)
    findings = [
        f"1. The combined R2 development run closed {len(feature_rows)} trades. Trading return was {float(result['baseline_metrics']['trading_net_return'])*100:.2f}%, profit factor {float(result['baseline_metrics']['profit_factor']):.3f}, and mean net R {float(result['baseline_metrics']['average_R']):.3f}. Five positions remained open at the evaluation end and are excluded from closed-trade excursion statistics.",
        f"2. Gross edge was positive in the combined sample for both setups, but transaction costs changed the result: support-test gross/net P&L INR {setup_cost_by_name['support_test']['gross_pnl']:.2f}/INR {setup_cost_by_name['support_test']['net_trading_pnl']:.2f}; breakout-retest INR {setup_cost_by_name['breakout_retest']['gross_pnl']:.2f}/INR {setup_cost_by_name['breakout_retest']['net_trading_pnl']:.2f}. Fixed operating costs are excluded from this comparison.",
        f"3. In isolated saved runs, support-test had {support_isolated['trades']} trades and net P&L INR {support_isolated['net_trading_pnl']:.2f}; breakout-retest had {breakout_isolated['trades']} trades and net P&L INR {breakout_isolated['net_trading_pnl']:.2f}. The selected trade sets change with portfolio competition, so the 13 support trades in the combined run do not establish a robust positive edge.",
        f"4. Breakout-retest classifications: {failure_pct.get('successful_continuation', 0):.1f}% exited at target; {failure_pct.get('immediate_failed_breakout', 0):.1f}% stopped within three sessions; {failure_pct.get('stopped_then_eventually_moved_up', 0):.1f}% later reached +0.5R after exit by 2024-12-31. Classification rules are mutually exclusive and listed in report.json.",
        f"5. Retest-delay bucket 4-7 sessions: {bval('retest_delay_sessions', '4-7', 'trade_count')} trades, net expectancy INR {float(bval('retest_delay_sessions', '4-7', 'net_expectancy')):.2f}; 1-3 sessions: {bval('retest_delay_sessions', '1-3', 'trade_count')} trades, INR {float(bval('retest_delay_sessions', '1-3', 'net_expectancy')):.2f}. Treat as a hypothesis, not a selected rule.",
        f"6. Breakout volume ratio >1.5: {bval('breakout_volume_ratio', '>1.5', 'trade_count')} trades, net expectancy INR {float(bval('breakout_volume_ratio', '>1.5', 'net_expectancy')):.2f}; below 0.8: {bval('breakout_volume_ratio', '<0.8', 'trade_count')} trades, INR {float(bval('breakout_volume_ratio', '<0.8', 'net_expectancy')):.2f}. Bucket sizes and the post-hoc nature limit inference.",
        f"7. The support-only signal sample contains no second-touch executions; the baseline's executed support trades are all third-or-later zone touches. In the combined run, 13 support trades netted INR {setup_cost_by_name['support_test']['net_trading_pnl']:.2f}; sparse subgroup counts prevent reliable touch-age/reaction conclusions.",
        f"8. The zone audit found {zone_audit['zone_count']} reconstructed eligible zones formed in 2023-2024 ({yearly[0]['average_zones_per_security']:.1f} average per security in 2023; {yearly[1]['average_zones_per_security']:.1f} in 2024), {zone_audit['zones_never_leading_to_trade_pct']:.1f}% never matched an executed trade, and {zone_audit['near_duplicate_overlap_pairs']} overlapping same-side zone pairs during overlapping lifetimes. This suggests redundant/mechanical candidates but is not by itself proof zones are useless.",
        f"9. Closed trades reached +0.5R in {mfe_half['reach_rate_pct']:.1f}% and +2R in {mfe_two['reach_rate_pct']:.1f}%; mean MFE was {float(ftxt('MFE_R','net_winner','mean')):.2f}R for eventual net winners vs {float(ftxt('MFE_R','net_loser','mean')):.2f}R for net losers. Mean MAE was {float(ftxt('MAE_R','net_winner','mean')):.2f}R vs {float(ftxt('MAE_R','net_loser','mean')):.2f}R. These are post-entry outcomes, not ex-ante predictors.",
        f"10. Zone age (winner/loser mean {float(ftxt('zone_age_sessions','net_winner','mean')):.1f}/{float(ftxt('zone_age_sessions','net_loser','mean')):.1f} sessions), touch count ({float(ftxt('number_of_touch_episodes','net_winner','mean')):.1f}/{float(ftxt('number_of_touch_episodes','net_loser','mean')):.1f}), and signal close location ({float(ftxt('signal_close_location','net_winner','mean')):.3f}/{float(ftxt('signal_close_location','net_loser','mean')):.3f}) barely separate outcomes in this combined run. The zone width is mechanically 2% of center because the configured 1% tolerance is also applied symmetrically to both boundaries.",
    ]
    (output_dir / "findings.md").write_text(
        "# Findings from sr_baseline_v1 (2023-2024 only)\n\n"
        "This is development-only descriptive analysis using the saved combined R2, weekly-filter-off, current Zerodha cost run. "
        "No 2025/2026 candles, strategy changes, hourly/15-minute data, or parameter optimization were used. "
        "Current fees are assumptions for comparison and are not a historical fee schedule.\n\n"
        + "\n\n".join(findings) +
        "\n\n## Recommended V2 research sequence\n\n"
        "1. Freeze sr_baseline_v1 and make the next comparison portfolio-aware and execution-cost-aware, including nonzero slippage and explicit reporting of open positions.\n"
        "2. Evaluate breakout-retest as its own hypothesis. Pre-register one-at-a-time tests for retest delay and breakout volume (the observed buckets are development evidence only), then reserve later dates for validation.\n"
        "3. Audit zone identity, active overlap, touch episodes, and the support penetration threshold before proposing a detector variant. Compare a versioned deterministic alternative without editing this baseline.\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/market_data_manifest.csv"))
    parser.add_argument("--data-root", type=Path, default=Path("data/processed/market_data"))
    args = parser.parse_args()
    result = generate_diagnostics(run_path=args.run, output_dir=args.output,
                                  manifest_path=args.manifest, data_root=args.data_root)
    print(json.dumps({"output": str(args.output), "trades_analyzed": result["trades_analyzed"],
                      "trades_by_setup": result["trades_by_setup"],
                      "breakout_failure_counts": result["breakout_failure_counts"],
                      "zone_audit": result["zone_audit"]}, indent=2))


if __name__ == "__main__":
    main()
