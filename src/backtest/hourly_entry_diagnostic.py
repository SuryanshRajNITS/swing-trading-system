"""Causal descriptive replay of saved weekly-selection entries with weekly, daily and 1H S/R.

This module does not generate or alter trades. Every hourly feature is observed
after its native provider bar, beginning on the session after the daily signal.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

from src.backtest.data import BacktestDataError, load_daily_dataset_through, load_hourly_dataset_through
from src.configuration import PROJECT_ROOT
from src.data.hourly import SESSION_OVERRIDES
from src.data.market_data import DATA_ROOT, MANIFEST_PATH
from src.data.universe.provider import PIT_TRADABLE
from src.data.weekly import resample_daily_to_weekly
from src.strategy.sr_baseline import SwingPoint, find_confirmed_swings
from src.strategy.sr_zone_v2 import Zone, ZoneConfig, detect_zones


TZ = ZoneInfo("Asia/Kolkata")
TRADE_PATH = PROJECT_ROOT / "data/processed/backtests/weekly_selection/trade_audit_A_baseline_2023_2024.csv"
SAMPLE_PATH = PROJECT_ROOT / "data/processed/backtests/weekly_selection/trade_audit_sample_2023_2024.csv"
OUTPUT_DIR = PROJECT_ROOT / "data/processed/backtests/hourly_entry"
START = date(2023, 1, 1)
END = date(2024, 12, 31)


@dataclass(frozen=True)
class TimeframeConfig:
    swing_left: int
    swing_right: int
    atr_period: int
    atr_half_width: float
    minimum_width_pct: float
    maximum_width_pct: float
    cluster_distance_atr: float
    minimum_pivots: int
    maximum_age: int

    def zone_config(self) -> ZoneConfig:
        return ZoneConfig(
            swing_left=self.swing_left, swing_right=self.swing_right,
            atr_period=self.atr_period, atr_half_width=self.atr_half_width,
            minimum_width_pct=self.minimum_width_pct, maximum_width_pct=self.maximum_width_pct,
            cluster_distance_atr=self.cluster_distance_atr, minimum_pivots=self.minimum_pivots,
            maximum_age=self.maximum_age,
        )


# Separate structural configurations; hourly defaults are not copied from daily.
WEEKLY_CONFIG = TimeframeConfig(2, 2, 14, 0.5, 0.003, 0.02, 0.75, 2, 104)
DAILY_CONFIG = TimeframeConfig(3, 3, 14, 0.5, 0.0025, 0.01, 0.75, 2, 252)
HOURLY_CONFIG = TimeframeConfig(3, 3, 14, 0.5, 0.0025, 0.015, 0.75, 2, 130)
CONFIGS = {"weekly": WEEKLY_CONFIG, "daily": DAILY_CONFIG, "hourly": HOURLY_CONFIG}


def _day(value: Any) -> date:
    return date.fromisoformat(str(value)[:10])


def _stamp(row: Mapping[str, Any]) -> datetime:
    value = row["date"]
    value = datetime.fromisoformat(str(value)) if not isinstance(value, datetime) else value
    return value.replace(tzinfo=TZ) if value.tzinfo is None else value.astimezone(TZ)


def _atr(rows: Sequence[Mapping[str, Any]], index: int, period: int = 14) -> float:
    if not rows:
        return 0.0
    start = max(0, index - period + 1)
    ranges = []
    for i in range(start, index + 1):
        high, low = float(rows[i]["high"]), float(rows[i]["low"])
        prior = float(rows[i - 1]["close"]) if i else float(rows[i]["close"])
        ranges.append(max(high - low, abs(high - prior), abs(low - prior)))
    return sum(ranges) / len(ranges)


def _windowed_zone_snapshot(
    rows: Sequence[Mapping[str, Any]], *, security_id: str, index: int,
    config: TimeframeConfig, confirmed_swings: Sequence[SwingPoint] | None = None,
) -> tuple[Zone, ...]:
    """Detect zones on a bounded as-of prefix; pivots count only after confirmation."""
    if index < 0 or index >= len(rows):
        return ()
    right = config.swing_right
    offset = max(0, index - config.maximum_age - right)
    prefix = rows[offset:index + 1]
    full_swings = list(confirmed_swings) if confirmed_swings is not None else find_confirmed_swings(
        rows[:index + 1], left=config.swing_left, right=right)
    swings = [SwingPoint(s.kind, s.pivot_index - offset, s.confirmation_index - offset, s.price)
              for s in full_swings if s.pivot_index >= offset and s.confirmation_index <= index]
    # Lifecycle is computed on the same as-of prefix, so any break/retest/reaction
    # metadata is historical at this timestamp and cannot inspect later bars.
    return detect_zones(prefix, security_id=security_id, config=config.zone_config(),
                        include_lifecycle=True, confirmed_swings=swings)


def _nearest(zones: Sequence[Zone], price: float, side: str) -> Zone | None:
    candidates = [z for z in zones if z.side == side and
                  (z.upper_bound <= price if side == "support" else z.lower_bound >= price)]
    if not candidates:
        # Price inside a zone counts as the nearest zone of that side.
        candidates = [z for z in zones if z.side == side and z.lower_bound <= price <= z.upper_bound]
    if not candidates:
        return None
    return min(candidates, key=lambda z: (max(0.0, z.lower_bound - price, price - z.upper_bound), z.zone_id))


def _zone_record(zone: Zone | None, price: float, atr: float) -> dict[str, Any]:
    if zone is None:
        return {"zone_id": None, "lower": None, "upper": None, "center": None,
                "distance": None, "distance_pct": None, "distance_atr": None,
                "overlap": False, "pivot_count": 0, "first_pivot": None, "last_pivot": None,
                "touch_episode_count": 0, "reaction_count": 0, "broken": False,
                "role_flip": None, "freshness": None}
    distance = max(0.0, zone.lower_bound - price, price - zone.upper_bound)
    return {"zone_id": zone.zone_id, "lower": zone.lower_bound, "upper": zone.upper_bound,
            "center": zone.center, "distance": distance,
            "distance_pct": distance / price if price else None,
            "distance_atr": distance / atr if atr else None,
            "overlap": zone.lower_bound <= price <= zone.upper_bound,
            "pivot_count": len(zone.pivots), "first_pivot": zone.first_pivot,
            "last_pivot": zone.last_update, "touch_episode_count": len(zone.touch_episodes),
            "reaction_count": len(zone.reactions), "broken": zone.broken,
            "role_flip": zone.role_flip, "freshness": zone.freshness}


def _zone_payload(zone: Zone, asof: date | None = None) -> dict[str, Any]:
    """Compact, auditable lifecycle detail avoids repeating full event histories per bar."""
    episodes = list(zone.touch_episodes)
    reactions = list(zone.reactions)
    return {"zone_id": zone.zone_id, "security_id": zone.security_id, "side": zone.side,
        "lower_bound": zone.lower_bound, "upper_bound": zone.upper_bound, "center": zone.center,
        "creation_date": zone.creation_date, "first_pivot": zone.first_pivot,
        "last_update": zone.last_update, "pivots": list(zone.pivots),
        "age_calendar_days": ((asof - _day(zone.creation_date)).days if asof else None),
        "touch_episode_count": len(episodes),
        "recent_touch_episodes": [asdict(x) for x in episodes[-3:]],
        "reaction_count": len(reactions),
        "max_reaction_atr": max((float(x.get("move_atr", 0)) for x in reactions), default=None),
        "broken": zone.broken, "role_flip": zone.role_flip, "freshness": zone.freshness,
        "merged_ids": list(zone.merged_ids)}


def _pair_confluence(a: Mapping[str, Any], b: Mapping[str, Any], atr: float, price: float) -> dict[str, Any]:
    if a.get("lower") is None or b.get("lower") is None:
        return {"nearby": False, "overlap": False, "zone_gap": None,
                "zone_gap_pct": None, "zone_gap_atr": None}
    overlap = max(float(a["lower"]), float(b["lower"])) <= min(float(a["upper"]), float(b["upper"]))
    gap = max(0.0, float(a["lower"]) - float(b["upper"]), float(b["lower"]) - float(a["upper"]))
    # Nearby is a fixed transparent diagnostic convention: zone gap <= 0.5 current 1H ATR.
    return {"nearby": overlap or gap <= 0.5 * atr, "overlap": overlap,
            "zone_gap": gap, "zone_gap_pct": gap / price if price else None,
            "zone_gap_atr": gap / atr if atr else None}


def _hourly_zone_behavior(
    zones: Sequence[Zone], row: Mapping[str, Any], previous_close: float | None,
) -> dict[str, bool]:
    high, low, close = (float(row[key]) for key in ("high", "low", "close"))
    supports = [z for z in zones if z.side == "support"]
    resistances = [z for z in zones if z.side == "resistance"]
    support_test = any(low <= z.upper_bound and high >= z.lower_bound for z in supports)
    support_reaction = any(low <= z.upper_bound and high >= z.lower_bound and close >= z.upper_bound
                           for z in supports)
    support_break = any(previous_close is not None and previous_close >= z.lower_bound
                        and close < z.lower_bound for z in supports)
    resistance_break = any(previous_close is not None and previous_close <= z.upper_bound
                           and close > z.upper_bound for z in resistances)
    resistance_retest = any(previous_close is not None and previous_close > z.upper_bound
                            and low <= z.upper_bound and close >= z.center for z in resistances)
    return {"support_test": support_test, "support_reaction": support_reaction,
            "support_break": support_break, "resistance_break": resistance_break,
            "resistance_retest": resistance_retest}


def _timeframe_context(
    rows: Sequence[Mapping[str, Any]], *, security_id: str, asof_index: int,
    price: float, atr: float, config: TimeframeConfig,
) -> tuple[dict[str, Any], tuple[Zone, ...]]:
    zones = _windowed_zone_snapshot(rows, security_id=security_id, index=asof_index, config=config)
    asof = _day(rows[asof_index]["date"]) if asof_index >= 0 else None
    return _context_for_zones(zones, price=price, atr=atr, asof=asof)


def _context_for_zones(zones: Sequence[Zone], *, price: float, atr: float,
                       asof: date | None = None) -> tuple[dict[str, Any], tuple[Zone, ...]]:
    support = _zone_record(_nearest(zones, price, "support"), price, atr)
    resistance = _zone_record(_nearest(zones, price, "resistance"), price, atr)
    return {"support": support, "resistance": resistance,
            "zone_count": len(zones), "zones": [_zone_payload(z, asof) for z in zones]}, zones


def _confirmed_structure(rows: Sequence[Mapping[str, Any]], swings: Sequence[SwingPoint], index: int) -> dict[str, Any]:
    known = [s for s in swings if s.confirmation_index <= index]
    lows = [s for s in known if s.kind == "support"][-2:]
    highs = [s for s in known if s.kind == "resistance"][-2:]
    higher_low = len(lows) == 2 and lows[-1].price > lows[-2].price
    lower_low = len(lows) == 2 and lows[-1].price < lows[-2].price
    higher_high = len(highs) == 2 and highs[-1].price > highs[-2].price
    lower_high = len(highs) == 2 and highs[-1].price < highs[-2].price
    hl_then_hh = (higher_low and higher_high and
                  lows[-1].confirmation_index < highs[-1].confirmation_index)
    state = "BULLISH" if higher_low and higher_high else "DETERIORATING" if lower_low and lower_high else "NEUTRAL"
    return {"higher_low_formed": higher_low, "lower_low_formed": lower_low,
            "higher_high_formed": higher_high, "lower_high_formed": lower_high,
            "higher_low_then_higher_high": hl_then_hh, "structure_state": state,
            "confirmed_swing_high_count": sum(s.kind == "resistance" for s in known),
            "confirmed_swing_low_count": sum(s.kind == "support" for s in known),
            "recent_confirmed_swing_highs": [{"pivot_index": s.pivot_index, "confirmation_index": s.confirmation_index,
                                               "price": s.price} for s in [x for x in known if x.kind == "resistance"][-6:]],
            "recent_confirmed_swing_lows": [{"pivot_index": s.pivot_index, "confirmation_index": s.confirmation_index,
                                              "price": s.price} for s in [x for x in known if x.kind == "support"][-6:]]}


def _special_type(day: date) -> str:
    return str(SESSION_OVERRIDES.get(day, {}).get("session_type", "NORMAL_SESSION"))


def _provider_starts(day: date) -> list[time] | None:
    kind = _special_type(day)
    if kind in {"NORMAL_SESSION", "SPECIAL_REGULAR_SESSION"}:
        # Kite may return the final partial 15:15 bar, but it is optional.
        return [time(h, 15) for h in range(9, 15)]
    if kind == "SPECIAL_SPLIT_SESSION":
        return [time(9, 15), time(11, 15), time(12, 15)]
    if kind == "MUHURAT":
        return None  # Do not infer provider coverage from exchange continuous windows.
    return []


def _bar_available_at(row: Mapping[str, Any]) -> datetime:
    start = _stamp(row)
    if start.time().replace(tzinfo=None) == time(15, 15):
        return start.replace(hour=15, minute=30)
    # Explicit split-session Kite 60minute semantics: native timestamps are retained.
    if start.date() in {date(2024, 3, 2), date(2024, 5, 18)}:
        return {time(9, 15): start.replace(hour=10, minute=0),
                time(11, 15): start.replace(hour=12, minute=15),
                time(12, 15): start.replace(hour=12, minute=30)}.get(start.time().replace(tzinfo=None), start + timedelta(hours=1))
    return start + timedelta(hours=1)


def _hourly_volume_ratio(rows: Sequence[Mapping[str, Any]], index: int, lookback_sessions: int = 20) -> float | None:
    current = rows[index]
    if _special_type(_stamp(current).date()) != "NORMAL_SESSION":
        return None
    tod = _stamp(current).time().replace(tzinfo=None)
    prior = [float(r["volume"]) for r in rows[:index]
             if _stamp(r).time().replace(tzinfo=None) == tod
             and _special_type(_stamp(r).date()) == "NORMAL_SESSION"][-lookback_sessions:]
    baseline = sum(prior) / len(prior) if prior else 0.0
    return float(current["volume"]) / baseline if baseline else None


def _bearish_rejection_at_resistance(
    row: Mapping[str, Any], close_location: float, zones: Sequence[Zone],
) -> bool:
    candle_shape = float(row["close"]) < float(row["open"]) and close_location <= 0.25
    return candle_shape and any(
        z.side == "resistance" and float(row["high"]) >= z.lower_bound
        and float(row["low"]) <= z.upper_bound and float(row["close"]) < z.lower_bound
        for z in zones)


def _six_bar_contraction_context(
    prefix: Sequence[Mapping[str, Any]],
) -> tuple[float | None, float | None, float | None, float | None]:
    """Return completed pre-current 6-bar range, prior 6-bar range and bounds."""
    recent = prefix[-7:-1]
    previous = prefix[-13:-7]
    recent_range = (max(float(x["high"]) for x in recent) - min(float(x["low"]) for x in recent)
                    if len(recent) == 6 else None)
    previous_range = (max(float(x["high"]) for x in previous) - min(float(x["low"]) for x in previous)
                      if len(previous) == 6 else None)
    prior_high = max((float(x["high"]) for x in recent), default=None)
    prior_low = min((float(x["low"]) for x in recent), default=None)
    return recent_range, previous_range, prior_high, prior_low


def _hourly_features(
    rows: Sequence[Mapping[str, Any]], index: int, *, security_id: str,
    hourly_swings: Sequence[SwingPoint],
) -> dict[str, Any]:
    prefix = rows[:index + 1]
    row = rows[index]
    price, atr = float(row["close"]), _atr(prefix, len(prefix) - 1)
    zones = _windowed_zone_snapshot(prefix, security_id=security_id, index=len(prefix) - 1,
                                    config=HOURLY_CONFIG, confirmed_swings=hourly_swings)
    context, _ = _context_for_zones(zones, price=price, atr=atr)
    structure = _confirmed_structure(prefix, hourly_swings, index)
    support_zone = _nearest(zones, price, "support")
    resistance_zone = _nearest(zones, price, "resistance")
    previous_close = float(prefix[-2]["close"]) if len(prefix) > 1 else None
    zone_behavior = _hourly_zone_behavior(zones, row, previous_close)
    recent_range, prior_range, prior_high, prior_low = _six_bar_contraction_context(prefix)
    candle_range = max(float(row["high"]) - float(row["low"]), 1e-12)
    close_location = (price - float(row["low"])) / candle_range
    bearish_rejection_candle = float(row["close"]) < float(row["open"]) and close_location <= 0.25
    bearish_rejection_at_resistance = _bearish_rejection_at_resistance(row, close_location, zones)
    contraction = recent_range is not None and prior_range is not None and recent_range < prior_range
    breakout_up = prior_high is not None and float(row["close"]) > prior_high
    breakout_down = prior_low is not None and float(row["close"]) < prior_low
    return {"timestamp": _stamp(row).isoformat(), "known_after_timestamp": _bar_available_at(row).isoformat(),
            "session_type": row.get("session_type", _special_type(_stamp(row).date())),
            "open": float(row["open"]), "high": float(row["high"]), "low": float(row["low"]),
            "close": price, "volume": int(float(row["volume"])), "atr_1h": atr,
            "volume_ratio_same_time_of_day": _hourly_volume_ratio(rows, index),
            "close_location": close_location,
            "bearish_rejection_candle": bearish_rejection_candle,
            "bearish_rejection_at_resistance": bearish_rejection_at_resistance,
            "hourly_sr": {"support": context["support"], "resistance": context["resistance"],
                          "zone_count": context["zone_count"], "zones": context["zones"]},
            "hourly_support_test": zone_behavior["support_test"],
            "hourly_support_reaction": zone_behavior["support_reaction"],
            "hourly_support_break": zone_behavior["support_break"],
            "hourly_resistance_break": zone_behavior["resistance_break"],
            "hourly_resistance_retest": zone_behavior["resistance_retest"],
            "trapped_between_hourly_sr": bool(support_zone and resistance_zone and
                                                support_zone.upper_bound < price < resistance_zone.lower_bound),
            "recent_6_bar_range": recent_range, "recent_range_atr": recent_range / atr if atr and recent_range is not None else None,
            "previous_6_bar_range": prior_range,
            "range_contraction": contraction,
            "local_consolidation_breakout_up": breakout_up,
            "local_consolidation_breakout_down": breakout_down,
            "structure": structure}


def _daily_sessions(daily_data: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[date]:
    return sorted({_day(row["date"]) for rows in daily_data.values() for row in rows})


def _find_index(rows: Sequence[Mapping[str, Any]], cutoff: date) -> int:
    result = -1
    for index, row in enumerate(rows):
        if _day(row["date"]) <= cutoff:
            result = index
        else:
            break
    return result


def _window_days(signal_day: date, sessions: Sequence[date], count: int = 3) -> list[date]:
    return [day for day in sessions if day > signal_day][:count]


def _outcome(row: Mapping[str, Any]) -> str:
    reason = str(row.get("exit_reason", "")).upper()
    if reason == "TARGET":
        return "TARGET"
    if reason == "STOP":
        return "STOP"
    if str(row.get("record_status", "")).upper() == "OPEN_AT_END" or not reason:
        return "OPEN"
    return "OTHER"


def _extension_bucket(value: float | None, entry: float | None,
                      lower: float | None, upper: float | None) -> str:
    """Fixed descriptive bins only; these are not decision thresholds."""
    if value is None:
        return "UNAVAILABLE"
    if entry is not None and lower is not None and entry < lower:
        return "BELOW_SETUP_ZONE"
    if entry is not None and upper is not None and entry <= upper:
        return "WITHIN_SETUP_ZONE"
    if value < 0.01:
        return "0_TO_1_PCT"
    if value < 0.02:
        return "1_TO_2_PCT"
    return "AT_LEAST_2_PCT"


def _missing_coverage(
    security_id: str, days: Sequence[date], hourly_data: Mapping[str, Sequence[Mapping[str, Any]]],
    quarantines: Sequence[Mapping[str, Any]],
) -> tuple[str, list[str], list[str]]:
    if security_id not in hourly_data:
        return "HOURLY_DATA_UNAVAILABLE", [], []
    actual = {_stamp(r) for r in hourly_data[security_id]}
    expected: list[str] = []
    unknown_special: list[str] = []
    for day in days:
        starts = _provider_starts(day)
        if starts is None:
            if not any(stamp.date() == day for stamp in actual):
                unknown_special.append(f"{day.isoformat()}|SPECIAL_SESSION_PROVIDER_GRID_UNVERIFIED")
            continue
        expected.extend(datetime.combine(day, tod, tzinfo=TZ).isoformat() for tod in starts)
    absent = [stamp for stamp in expected if datetime.fromisoformat(stamp) not in actual]
    absent.extend(unknown_special)
    quarantine_keys = {(_day(q["timestamp"]), _stamp({"date": q["timestamp"]}).time().replace(tzinfo=None))
                       for q in quarantines if q.get("security_id") == security_id}
    quarantined = [stamp for stamp in absent if "|" not in stamp and (datetime.fromisoformat(stamp).date(),
                   datetime.fromisoformat(stamp).time().replace(tzinfo=None)) in quarantine_keys]
    return ("PARTIAL_HOURLY_DATA" if absent else "FULL_HOURLY_COVERAGE", absent, quarantined)


def _daily_weekly_context(
    security_id: str, trade: Mapping[str, Any], daily_data: Mapping[str, Sequence[Mapping[str, Any]]],
    asof: datetime, price: float | None = None, trading_sessions: Sequence[date] | None = None,
    cache: dict[str, Any] | None = None,
) -> dict[str, Any]:
    all_daily = list(daily_data.get(security_id, ()))
    signal_day = _day(trade["signal_date"])
    cache = cache if cache is not None else {}
    if "weekly_rows" not in cache:
        daily_prefix_end = _find_index(all_daily, signal_day)
        daily_prefix = all_daily[:daily_prefix_end + 1] if daily_prefix_end >= 0 else []
        weekly_rows = resample_daily_to_weekly(all_daily, trading_sessions=trading_sessions)
        # The zone detector's common candle interface uses ``date``. This date is
        # when the weekly aggregate became available, not the Monday label.
        weekly_rows = [{**row, "date": row["completion_at"]} for row in weekly_rows]
        daily_atr = _atr(daily_prefix, len(daily_prefix) - 1) if daily_prefix else 0.0
        daily_zones = (_windowed_zone_snapshot(daily_prefix, security_id=security_id,
            index=len(daily_prefix)-1, config=DAILY_CONFIG) if daily_prefix else ())
        cache.update({"daily_prefix": daily_prefix, "weekly_rows": weekly_rows,
                      "daily_atr": daily_atr, "daily_zones": daily_zones,
                      "weekly_zones_by_count": {}})
    daily_prefix = cache["daily_prefix"]
    weekly_rows = cache["weekly_rows"]
    completed_weekly = [row for row in weekly_rows
                        if datetime.fromisoformat(row["completion_at"]).astimezone(TZ) <= asof]
    price = price if price is not None else float(trade["actual_entry_price"] or trade["signal_close"])
    daily_index = len(daily_prefix) - 1
    weekly_atr = _atr(completed_weekly, len(completed_weekly) - 1) if completed_weekly else 0.0
    daily_ctx, daily_zones = _context_for_zones(cache["daily_zones"], price=price,
        atr=cache["daily_atr"], asof=asof.astimezone(TZ).date()) if daily_prefix else ({}, ())
    weekly_count = len(completed_weekly)
    if weekly_count not in cache["weekly_zones_by_count"]:
        cache["weekly_zones_by_count"][weekly_count] = (
            _windowed_zone_snapshot(completed_weekly, security_id=security_id,
                index=weekly_count-1, config=WEEKLY_CONFIG) if weekly_count else ())
    weekly_zones = cache["weekly_zones_by_count"][weekly_count]
    weekly_ctx, weekly_zones = _context_for_zones(weekly_zones, price=price,
        atr=weekly_atr, asof=asof.astimezone(TZ).date()) if completed_weekly else ({}, ())
    return {"weekly": weekly_ctx, "daily": daily_ctx, "daily_zones": daily_zones,
            "weekly_zones": weekly_zones, "daily_prefix": daily_prefix,
            "completed_weekly": completed_weekly, "daily_setup_zone": {
                "lower": _float_or_none(trade.get("setup_zone_lower")),
                "center": _float_or_none(trade.get("setup_zone_center")),
                "upper": _float_or_none(trade.get("setup_zone_upper")),
                "type": trade.get("setup_zone_type"), "zone_id": trade.get("setup_zone_id")}}


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except (ValueError, TypeError):
        return None


def _confluence_context(
    weekly: Mapping[str, Any], daily: Mapping[str, Any], hourly: Mapping[str, Any],
    atr: float, price: float,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for side in ("support", "resistance"):
        zones = [weekly.get(side, {}), daily.get(side, {}), hourly.get(side, {})]
        pairs = {"weekly_daily": _pair_confluence(zones[0], zones[1], atr, price),
                 "weekly_hourly": _pair_confluence(zones[0], zones[2], atr, price),
                 "daily_hourly": _pair_confluence(zones[1], zones[2], atr, price)}
        # Count each timeframe only when its zone is present and is nearby another timeframe's zone.
        result[side] = {"pairs": pairs,
                        "confluence_count": sum(bool(info["nearby"]) for info in pairs.values()),
                        "three_timeframe_confluence": all(info["nearby"] for info in pairs.values())}
    return result


def _summarize_trade(
    trade: Mapping[str, Any], *, daily_data: Mapping[str, Sequence[Mapping[str, Any]]],
    hourly_data: Mapping[str, Sequence[Mapping[str, Any]]], sessions: Sequence[date],
    quarantines: Sequence[Mapping[str, Any]],
    invalid_hourly_identities: Mapping[str, str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    sid = str(trade["security_id"])
    signal_day = _day(trade["signal_date"])
    next_days = _window_days(signal_day, sessions)
    exit_raw = str(trade.get("exit_timestamp") or "")
    exit_stamp = datetime.fromisoformat(exit_raw) if exit_raw else None
    if exit_stamp and exit_stamp.tzinfo is None:
        exit_stamp = exit_stamp.replace(tzinfo=TZ)
    # Entire signal-date hourly data is excluded from all feature history. The
    # signal is daily-close based; the first included bar starts next session.
    raw_hours = [r for r in hourly_data.get(sid, ()) if _stamp(r).date() != signal_day]
    window = [(i, row) for i, row in enumerate(raw_hours)
              if next_days and _stamp(row).date() in next_days
              and (exit_stamp is None or _stamp(row) <= exit_stamp)]
    relevant_days = [d for d in next_days if exit_stamp is None or d <= exit_stamp.date()]
    coverage, absent, quarantined = _missing_coverage(sid, relevant_days, hourly_data, quarantines)
    if next_days and not window and coverage != "HOURLY_DATA_UNAVAILABLE":
        coverage = "PARTIAL_HOURLY_DATA"
    if not next_days:
        coverage = "HOURLY_DATA_UNAVAILABLE"
    entry_raw = str(trade.get("entry_timestamp") or "")
    entry_asof = datetime.fromisoformat(entry_raw) if entry_raw else datetime.combine(signal_day, time(15, 30), tzinfo=TZ)
    if entry_asof.tzinfo is None:
        entry_asof = entry_asof.replace(tzinfo=TZ)
    entry_price = _float_or_none(trade.get("actual_entry_price")) or float(trade.get("signal_close") or 0)
    mtf_cache: dict[str, Any] = {}
    mtf = _daily_weekly_context(sid, trade, daily_data, entry_asof,
                                entry_price, sessions, mtf_cache)
    daily_setup = mtf["daily_setup_zone"]
    dlow, dupper = daily_setup["lower"], daily_setup["upper"]
    full_swings = find_confirmed_swings(raw_hours, left=HOURLY_CONFIG.swing_left,
                                        right=HOURLY_CONFIG.swing_right) if raw_hours else []
    pre_entry = [r for r in raw_hours if _stamp(r) < entry_asof]
    if pre_entry:
        pre_entry_atr = _atr(pre_entry, len(pre_entry) - 1)
        pre_entry_ctx, _ = _timeframe_context(pre_entry, security_id=sid,
            asof_index=len(pre_entry)-1, price=entry_price, atr=pre_entry_atr, config=HOURLY_CONFIG)
    else:
        pre_entry_ctx = {}
    hourly_sequence: list[dict[str, Any]] = []
    features = []
    zone_lost_during_window = False
    for index, bar in window:
        snapshot = _hourly_features(raw_hours, index, security_id=sid, hourly_swings=full_swings)
        price, atr = snapshot["close"], snapshot["atr_1h"]
        mtf_at_bar = _daily_weekly_context(sid, trade, daily_data, _bar_available_at(bar),
                                           price, sessions, mtf_cache)
        snapshot["weekly_sr"] = {"support": mtf_at_bar["weekly"].get("support"),
                                 "resistance": mtf_at_bar["weekly"].get("resistance")}
        snapshot["daily_sr"] = {"support": mtf_at_bar["daily"].get("support"),
                                "resistance": mtf_at_bar["daily"].get("resistance")}
        snapshot["multi_timeframe_confluence"] = _confluence_context(
            mtf_at_bar["weekly"], mtf_at_bar["daily"], snapshot["hourly_sr"], atr, price)
        snapshot["daily_setup_zone"] = daily_setup
        snapshot["distance_from_daily_zone_pct"] = ((price - dupper) / dupper if dupper else None)
        snapshot["distance_from_daily_zone_atr"] = ((price - dupper) / atr if dupper and atr else None)
        snapshot["trades_near_daily_setup_zone"] = bool(dlow is not None and dupper is not None
            and float(bar["low"]) <= dupper and float(bar["high"]) >= dlow)
        snapshot["daily_zone_lost"] = bool(dlow is not None and float(bar["close"]) < dlow)
        snapshot["daily_zone_reclaimed"] = bool(zone_lost_during_window and dupper is not None and price > dupper)
        zone_lost_during_window = zone_lost_during_window or snapshot["daily_zone_lost"]
        features.append(snapshot)
        hourly_sequence.append(snapshot)
    first = features[0] if features else None
    weekly = mtf["weekly"]
    daily = mtf["daily"]
    first_hourly = {"support": pre_entry_ctx.get("support", {}),
                    "resistance": pre_entry_ctx.get("resistance", {}),
                    "zone_count": pre_entry_ctx.get("zone_count", 0),
                    "zones": pre_entry_ctx.get("zones", [])}
    entry_atr = _atr(pre_entry, len(pre_entry)-1) if pre_entry else 0.0
    first_confluence = _confluence_context(weekly, daily, first_hourly, entry_atr, entry_price) if pre_entry else {}
    def any_feature(path: tuple[str, ...]) -> bool:
        for item in features:
            value: Any = item
            for key in path:
                value = value.get(key, {}) if isinstance(value, Mapping) else {}
            if value is True:
                return True
        return False
    consecutive_above = 0
    max_consecutive_above = 0
    for item in features:
        if dupper is not None and item["close"] > dupper:
            consecutive_above += 1
            max_consecutive_above = max(max_consecutive_above, consecutive_above)
        else:
            consecutive_above = 0
    first_day_open = next((float(r["open"]) for _, r in window
                           if _stamp(r).time().replace(tzinfo=None) == time(9, 15)), None)
    entry_price = _float_or_none(trade.get("actual_entry_price"))
    # This is the executable baseline fill's extension from the saved daily
    # setup-zone ceiling. Hourly 09:15 open is not the baseline fill price.
    entry_extension_pct = (entry_price - dupper) / dupper if entry_price is not None and dupper else None
    outcome = _outcome(trade)
    compact = {
        "trade_id": trade.get("trade_id"), "security_id": sid, "symbol": trade.get("symbol"),
        "setup_type": trade.get("setup_type"), "signal_date": signal_day.isoformat(),
        "signal_close": _float_or_none(trade.get("signal_close")),
        "baseline_entry_timestamp": trade.get("entry_timestamp"),
        "baseline_next_session_open_fill": entry_price,
        "daily_stop": _float_or_none(trade.get("stop_price_at_entry")),
        "daily_target": _float_or_none(trade.get("target_price_at_entry")),
        "actual_outcome": outcome, "exit_reason": trade.get("exit_reason"),
        "entry_session_open_extension_from_daily_zone_pct": entry_extension_pct,
        "entry_extension_bucket": _extension_bucket(entry_extension_pct, entry_price, dlow, dupper),
        "entry_extension_atr": ((entry_price - dupper) / first["atr_1h"] if entry_price is not None and dupper and first and first["atr_1h"] else None),
        "hourly_identity_status": (invalid_hourly_identities or {}).get(sid, "PRICE_CONSISTENT_OR_NOT_FLAGGED"),
        "hourly_coverage_status": coverage, "window_session_dates": [x.isoformat() for x in next_days],
        "expected_missing_hourly_timestamps": absent,
        "missing_timestamps_due_to_quarantine": quarantined,
        "hourly_bar_count": len(features),
        "weekly_sr_at_entry": {"support": weekly.get("support"), "resistance": weekly.get("resistance")},
        "daily_sr_at_entry": {"support": daily.get("support"), "resistance": daily.get("resistance")},
        "daily_setup_zone": daily_setup,
        "hourly_sr_at_entry": first_hourly,
        "confluence_at_entry": first_confluence,
        "support_confluence_count": first_confluence.get("support", {}).get("confluence_count"),
        "resistance_confluence_count": first_confluence.get("resistance", {}).get("confluence_count"),
        "hourly_support_near_entry": bool(pre_entry and first_hourly.get("support", {}).get("distance_atr") is not None
                                              and first_hourly["support"]["distance_atr"] <= 1.0),
        "hourly_resistance_near_entry": bool(pre_entry and first_hourly.get("resistance", {}).get("distance_atr") is not None
                                                 and first_hourly["resistance"]["distance_atr"] <= 1.0),
        "higher_low_formed": any_feature(("structure", "higher_low_formed")),
        "lower_low_formed": any_feature(("structure", "lower_low_formed")),
        "higher_high_formed": any_feature(("structure", "higher_high_formed")),
        "lower_high_formed": any_feature(("structure", "lower_high_formed")),
        "higher_low_then_higher_high": any_feature(("structure", "higher_low_then_higher_high")),
        "short_term_structure": features[-1]["structure"]["structure_state"] if features else "UNAVAILABLE",
        "daily_zone_held": max_consecutive_above >= 2,
        "daily_zone_reclaimed": any(item["daily_zone_reclaimed"] for item in features),
        "daily_zone_lost": any(item["daily_zone_lost"] for item in features),
        "immediate_loss_of_daily_zone": any(item["daily_zone_lost"] for item in features[:3]),
        "hourly_support_tested": any(item["hourly_support_test"] for item in features),
        "hourly_support_reacted": any(item["hourly_support_reaction"] for item in features),
        "hourly_support_broken": any(item["hourly_support_break"] for item in features),
        "hourly_resistance_broken": any(item["hourly_resistance_break"] for item in features),
        "hourly_resistance_retested": any(item["hourly_resistance_retest"] for item in features),
        "trapped_between_hourly_sr": any(item["trapped_between_hourly_sr"] for item in features),
        "immediate_rejection": bool(features and features[0]["bearish_rejection_at_resistance"]),
        "bearish_rejection_seen": any(item["bearish_rejection_at_resistance"] for item in features),
        "bearish_rejection_candle_seen": any(item["bearish_rejection_candle"] for item in features),
        "break_nearby_hourly_support": any(item["hourly_support_break"] for item in features),
        "consolidation_breakout": any(item["local_consolidation_breakout_up"] or item["local_consolidation_breakout_down"] for item in features),
        "range_contraction_seen": any(item["range_contraction"] and
            (item["local_consolidation_breakout_up"] or item["local_consolidation_breakout_down"])
            for item in features),
        "range_contraction_observed": any(item["range_contraction"] for item in features),
        "outcome_class": outcome,
    }
    detail = {"trade": compact, "timeframe_configs": {key: asdict(value) for key, value in CONFIGS.items()},
              "hourly_sequence": hourly_sequence,
              "weekly_zones": mtf["weekly"].get("zones", []), "daily_zones": mtf["daily"].get("zones", [])}
    return compact, detail


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=True, sort_keys=True)
                             if isinstance(value, (dict, list, tuple)) else value for key, value in row.items()})


def _load_quarantines(path: Path, start: date, end: date) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    return [r for r in _read_csv(path) if start <= _day(r.get("timestamp", "1900-01-01")) <= end]


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    feature_rows = [r for r in rows if r.get("hourly_identity_status") != "HOURLY_IDENTITY_PRICE_MISMATCH"
                    and r.get("hourly_coverage_status") != "HOURLY_DATA_UNAVAILABLE"]
    result["feature_sample_trade_count"] = len(feature_rows)
    result["excluded_unavailable_or_invalid_identity_trade_count"] = len(rows) - len(feature_rows)
    features = ("higher_low_formed", "higher_low_then_higher_high", "daily_zone_held",
                "daily_zone_reclaimed", "immediate_rejection", "immediate_loss_of_daily_zone",
                "hourly_support_near_entry", "hourly_resistance_near_entry",
                "hourly_support_tested", "hourly_support_reacted", "hourly_support_broken",
                "hourly_resistance_broken",
                "hourly_resistance_retested", "consolidation_breakout",
                "bearish_rejection_seen", "bearish_rejection_candle_seen",
                "range_contraction_seen", "range_contraction_observed")
    for feature in features:
        categories = {}
        for value in (False, True):
            for outcome in ("TARGET", "STOP", "OPEN", "OTHER"):
                count = sum(bool(r.get(feature)) is value and r.get("outcome_class") == outcome for r in feature_rows)
                categories[f"{str(value).lower()}_{outcome.lower()}"] = count
        result[feature] = categories
    for key in ("support_confluence_count", "resistance_confluence_count"):
        buckets = {}
        for r in feature_rows:
            bucket = str(r.get(key))
            buckets.setdefault(bucket, {x: 0 for x in ("TARGET", "STOP", "OPEN", "OTHER")})
            buckets[bucket][str(r.get("outcome_class"))] += 1
        result[key] = buckets
    result["entry_extension_bucket"] = {}
    for r in feature_rows:
        bucket = str(r.get("entry_extension_bucket", "UNAVAILABLE"))
        result["entry_extension_bucket"].setdefault(bucket, {x: 0 for x in ("TARGET", "STOP", "OPEN", "OTHER")})
        result["entry_extension_bucket"][bucket][str(r.get("outcome_class"))] += 1
    for key in ("hourly_coverage_status",):
        result[key] = {value: sum(r.get(key) == value for r in rows)
                       for value in ("FULL_HOURLY_COVERAGE", "PARTIAL_HOURLY_DATA", "HOURLY_DATA_UNAVAILABLE")}
    return result


def _console(rows: Sequence[Mapping[str, Any]]) -> None:
    coverage = _summary(rows)["hourly_coverage_status"]
    print(f"total baseline trades: {len(rows)}")
    print(f"trades with full hourly coverage: {coverage['FULL_HOURLY_COVERAGE']}")
    print(f"partial coverage: {coverage['PARTIAL_HOURLY_DATA']}")
    print(f"unavailable coverage: {coverage['HOURLY_DATA_UNAVAILABLE']}")
    summary = _summary(rows)
    print(f"feature-evaluable trades: {summary['feature_sample_trade_count']}")
    print(f"excluded from feature counts (missing/invalid hourly identity): {summary['excluded_unavailable_or_invalid_identity_trade_count']}")
    print("\nTARGET vs STOP descriptive counts (feature false/true):")
    for feature, values in summary.items():
        if feature in {"hourly_coverage_status", "feature_sample_trade_count",
                       "excluded_unavailable_or_invalid_identity_trade_count"}:
            continue
        if feature.endswith("_confluence_count"):
            print(feature)
            for bucket, counts in sorted(values.items()):
                print(f"  {bucket}: TARGET={counts['TARGET']} STOP={counts['STOP']} OPEN={counts['OPEN']}")
        elif feature == "entry_extension_bucket":
            print(feature)
            for bucket, counts in sorted(values.items()):
                print(f"  {bucket}: TARGET={counts['TARGET']} STOP={counts['STOP']} OPEN={counts['OPEN']}")
        else:
            a, b = values["true_target"], values["true_stop"]
            print(f"  {feature}: TARGET={a} STOP={b}; false TARGET={values['false_target']} STOP={values['false_stop']}")


def run_diagnostic(
    *, trade_path: str | Path = TRADE_PATH, sample_path: str | Path = SAMPLE_PATH,
    output_dir: str | Path = OUTPUT_DIR, start: date = START, end: date = END,
    data_root: str | Path = DATA_ROOT, daily_manifest: str | Path = MANIFEST_PATH,
    hourly_manifest: str | Path | None = None, quarantine_path: str | Path | None = None,
    identity_exclusions_path: str | Path | None = None,
) -> dict[str, Any]:
    if start < START or end > END or start > end:
        raise ValueError("This diagnostic is bounded to 2023-01-01 through 2024-12-31.")
    trades = [row for row in _read_csv(trade_path) if start <= _day(row["signal_date"]) <= end]
    if not trades:
        raise ValueError(f"No saved A_BASELINE trades in the requested interval: {trade_path}")
    daily_data, daily_hash = load_daily_dataset_through(universe_id=PIT_TRADABLE,
        start_date=date(2022, 7, 1), end_date=end, data_root=data_root, manifest_path=daily_manifest)
    from src.data.hourly_price_integrity import EXCLUSIONS_CSV
    exclusions_file = Path(identity_exclusions_path or EXCLUSIONS_CSV)
    identity_rows = _read_csv(exclusions_file) if exclusions_file.is_file() else []
    invalid_identities = {r["security_id"]: r.get("identity_status", "HOURLY_IDENTITY_PRICE_MISMATCH")
                          for r in identity_rows
                          if r.get("identity_status") == "HOURLY_IDENTITY_PRICE_MISMATCH"}
    hourly_data, hourly_hash = load_hourly_dataset_through(universe_id=PIT_TRADABLE,
        start_date=start, end_date=end, data_root=data_root, manifest_path=hourly_manifest,
        daily_manifest_path=daily_manifest, identity_exclusions_path=exclusions_file)
    from src.data.hourly import HOURLY_QUARANTINE_PATH, HOURLY_MANIFEST
    from src.data.universe.provider import UniverseProvider
    quarantines = _load_quarantines(Path(quarantine_path or HOURLY_QUARANTINE_PATH), start, end)
    sessions = _daily_sessions(daily_data)
    expected_pit_ids = {m.security_id for m in UniverseProvider().members_over(start, end, PIT_TRADABLE) if m.tradable}
    compact, details = [], []
    for trade in sorted(trades, key=lambda r: (r["signal_date"], r["security_id"], r["trade_id"])):
        row, detail = _summarize_trade(trade, daily_data=daily_data, hourly_data=hourly_data,
                                       sessions=sessions, quarantines=quarantines,
                                       invalid_hourly_identities=invalid_identities)
        compact.append(row)
        details.append(detail)
    output = Path(output_dir)
    csv_path = output / "hourly_entry_diagnostic_2023_2024.csv"
    json_path = output / "hourly_entry_diagnostic_2023_2024.json"
    sample_output = output / "hourly_entry_diagnostic_sample_2023_2024.csv"
    sample_ids = {r["trade_id"] for r in _read_csv(sample_path)} if Path(sample_path).is_file() else set()
    sample = [r for r in compact if r["trade_id"] in sample_ids]
    if sample_ids and len(sample) != len(sample_ids):
        missing = sorted(sample_ids - {r["trade_id"] for r in sample})
        raise ValueError(f"Saved audit sample contains trade IDs absent from baseline artifact: {missing[:5]}")
    _write_csv(csv_path, compact)
    _write_csv(sample_output, sample)
    payload = {"diagnostic_version": "hourly_entry_diagnostic_v1", "period": {"start": start.isoformat(), "end": end.isoformat()},
        "universe_id": PIT_TRADABLE, "trade_source": str(Path(trade_path)),
        "hourly_source": "Kite Connect normalized 60minute data; provider timestamps preserved",
        "data_hashes": {"daily_prefix_sha256": daily_hash, "hourly_prefix_sha256": hourly_hash},
        "timezone": "Asia/Kolkata", "signal_day_hourly_bars_excluded": True,
        "window_sessions_after_signal": 3, "hourly_bar_features_are_available_after_provider_bar": True,
        "configs": {k: asdict(v) for k, v in CONFIGS.items()},
        "feature_definitions": {
            "pivot": "unique local high/low with symmetric left/right bars; usable at right-bar confirmation; retain count and latest six confirmed pivots of each side per snapshot",
            "zone": "ATR half-width around confirmed same-side pivot clusters; merged only when same-side intervals overlap",
            "hourly_config": "left/right=3 bars; ATR=14 bars; 2 pivots; cluster distance=0.75 ATR; width clamp 0.25%-1.5%; age=130 bars",
            "weekly_config": "left/right=2 weekly bars; ATR=14; 2 pivots; width clamp 0.3%-2%; age=104 weeks",
            "daily_config": "left/right=3 daily sessions; ATR=14; 2 pivots; width clamp 0.25%-1%; age=252 sessions",
            "zone_lifecycle": "touch/reaction/break/role-flip/freshness recomputed on each as-of prefix only",
            "near_support_or_resistance": "nearest zone within 1.0 current 1H ATR at baseline fill",
            "cross_timeframe_nearby": "same-side zones overlap or gap is at most 0.5 current 1H ATR; confluence count is the number (0-3) of nearby W-D, W-1H, and D-1H pairs",
            "daily_zone_held": "at least two consecutive post-entry 1H closes above saved daily setup-zone upper bound",
            "daily_zone_reclaimed": "a post-entry 1H close below saved daily setup-zone lower bound is followed by a close above its upper bound",
            "structure_state": "BULLISH=latest confirmed swing low and high both higher; DETERIORATING=both lower; otherwise NEUTRAL",
            "higher_low_then_higher_high": "latest higher low confirmation precedes latest higher high confirmation",
            "bearish_rejection_candle_seen": "red hourly candle closing in bottom quartile; candle-shape diagnostic only, without requiring a resistance test",
            "bearish_rejection_seen": "bearish rejection candle whose range intersects a causally known hourly resistance zone and closes below its lower bound",
            "consolidation": "six completed bars before the current bar have a narrower high-low range than the prior six completed bars; range_contraction_seen additionally requires the current bar to close beyond the compressed range",
            "volume_ratio": "bar volume divided by mean volume for up to 20 prior normal sessions at the same native start time; special sessions excluded",
            "entry_extension_buckets": "fixed descriptive labels below setup zone, within zone, 0-1%, 1-2%, and >=2% above zone; not selection cutoffs",
            "missing_bar_coverage": "checks six required regular-session Kite starts; 15:15 final partial bar is retained when present but optional",
        },
        "coverage_summary": _summary(compact)["hourly_coverage_status"],
        "pit_data_coverage": {"expected_security_ids": len(expected_pit_ids),
            "securities_with_hourly_data": len(hourly_data),
            "unavailable_security_ids": sorted(expected_pit_ids - set(hourly_data)),
            "excluded_identity_mismatch_security_ids": sorted(invalid_identities),
            "quarantined_provider_candles_in_period": len(quarantines)},
        "descriptive_summary": _summary(compact), "trades": details}
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True, default=str) + "\n", encoding="utf-8")
    _console(compact)
    print(f"\nCSV: {csv_path}")
    print(f"JSON: {json_path}")
    print(f"Manual sample CSV: {sample_output}")
    return {"trades": compact, "details": details, "csv_path": csv_path,
            "json_path": json_path, "sample_path": sample_output, "payload": payload}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trades", type=Path, default=TRADE_PATH)
    parser.add_argument("--sample", type=Path, default=SAMPLE_PATH)
    parser.add_argument("--output-directory", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--daily-manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--hourly-manifest", type=Path, default=None)
    parser.add_argument("--quarantine", type=Path, default=None)
    parser.add_argument("--identity-exclusions", type=Path, default=None)
    parser.add_argument("--start", type=date.fromisoformat, default=START)
    parser.add_argument("--end", type=date.fromisoformat, default=END)
    args = parser.parse_args()
    try:
        run_diagnostic(trade_path=args.trades, sample_path=args.sample,
                       output_dir=args.output_directory, data_root=args.data_root,
                       daily_manifest=args.daily_manifest, hourly_manifest=args.hourly_manifest,
                       quarantine_path=args.quarantine, identity_exclusions_path=args.identity_exclusions,
                       start=args.start, end=args.end)
    except (ValueError, OSError, BacktestDataError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
