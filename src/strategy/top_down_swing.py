"""Deterministic top-down swing-entry research skeleton.

The module describes candidates and causal entry confirmations. It does not
place orders, alter frozen data, or claim that any configured rule is profitable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import date, datetime
from math import floor
from statistics import fmean
from typing import Any, Mapping, Sequence

from src.strategy.sr_baseline import SwingPoint, find_confirmed_swings
from src.strategy.sr_zone_v2 import TouchEpisode, Zone, ZoneConfig, detect_zones


@dataclass(frozen=True)
class TopDownConfig:
    version: str = "top_down_swing_v1"
    weekly_approach_distance_atr: float = 1.5
    mid_range_lower_fraction: float = 0.35
    mid_range_upper_fraction: float = 0.65
    daily_approach_distance_atr: float = 1.0
    support_reaction_close_location: float = 0.60
    breakout_retest_max_sessions: int = 20
    confirmation_expiry_sessions: int = 10
    consolidation_bars: int = 6
    stop_buffer_atr: float = 0.15
    minimum_room_r: float = 1.5
    initial_capital: float = 10_000.0
    risk_fraction: float = 0.01
    max_open_positions: int = 10
    zone_acceptance_closes: int = 2
    zone_break_buffer_atr: float = 0.10
    meaningful_touch_episodes: int = 2
    major_touch_episodes: int = 4
    meaningful_reaction_atr: float = 0.50
    major_reaction_atr: float = 1.00
    meaningful_recency_bars: int = 130
    major_recency_bars: int = 65
    meaningful_pivots: int = 2
    actionable_support_distance_atr: float = 1.5
    actionable_resistance_distance_atr: float = 1.0
    context_min_completed_weeks: int = 26
    context_min_rs_weeks: int = 13
    no_resistance_fallback_r: float = 2.0
    ema_confluence_distance_atr: float = 0.25
    weekly_zone: ZoneConfig = ZoneConfig(swing_left=2, swing_right=2, minimum_pivots=2,
                                          maximum_age=104, minimum_width_pct=0.003,
                                          maximum_width_pct=0.02)
    daily_zone: ZoneConfig = ZoneConfig(swing_left=3, swing_right=3, minimum_pivots=2,
                                         maximum_age=252)
    hourly_zone: ZoneConfig = ZoneConfig(swing_left=3, swing_right=3, minimum_pivots=2,
                                          maximum_age=130)

    def __post_init__(self) -> None:
        if not 0 <= self.mid_range_lower_fraction < self.mid_range_upper_fraction <= 1:
            raise ValueError("Mid-range fractions must be ordered within [0, 1].")
        if min(self.weekly_approach_distance_atr, self.daily_approach_distance_atr,
               self.stop_buffer_atr, self.minimum_room_r, self.ema_confluence_distance_atr) < 0:
            raise ValueError("Distance and reward/risk thresholds must not be negative.")
        if not 0 < self.risk_fraction <= 1 or self.initial_capital <= 0:
            raise ValueError("Risk fraction and initial capital must be positive.")
        if min(self.confirmation_expiry_sessions, self.breakout_retest_max_sessions,
               self.consolidation_bars, self.max_open_positions, self.zone_acceptance_closes,
               self.meaningful_touch_episodes, self.major_touch_episodes,
               self.meaningful_pivots, self.context_min_completed_weeks,
               self.context_min_rs_weeks) < 1:
            raise ValueError("Session limits and lookbacks must be positive.")


def atr(rows: Sequence[Mapping[str, Any]], period: int = 14) -> float:
    sample = rows[-period:]
    values = []
    offset = len(rows) - len(sample)
    for j, row in enumerate(sample):
        i = offset + j
        prior = float(rows[i - 1]["close"]) if i else float(row["close"])
        high, low = float(row["high"]), float(row["low"])
        values.append(max(high - low, abs(high - prior), abs(low - prior)))
    return fmean(values) if values else 0.0


def ema(rows: Sequence[Mapping[str, Any]], period: int) -> float | None:
    """Return an EMA using only rows supplied (callers pass an as-of prefix)."""
    if len(rows) < period:
        return None
    alpha = 2.0 / (period + 1)
    value = fmean(float(row["close"]) for row in rows[:period])
    for row in rows[period:]:
        value = alpha * float(row["close"]) + (1 - alpha) * value
    return value


def _zones(rows: Sequence[Mapping[str, Any]], *, security_id: str,
           config: ZoneConfig) -> tuple[Zone, ...]:
    # A zone outside its configured maximum age cannot be actionable. Retain
    # ATR and pivot-confirmation context around that active window only.
    lookback = config.maximum_age + config.atr_period + max(config.swing_left, config.swing_right)
    bounded_rows = rows[-lookback:]
    return detect_zones(bounded_rows, security_id=security_id, config=config,
                        include_lifecycle=False)


def _lifecycle_snapshot(zones: Sequence[Zone], rows: Sequence[Mapping[str, Any]],
                        config: ZoneConfig, strategy_config: TopDownConfig | None = None) -> tuple[Zone, ...]:
    """Compute causal touch, strength, sustained-break and retest-confirmation state."""
    cfg = strategy_config or TopDownConfig()
    date_index = {str(row["date"])[:10]: i for i, row in enumerate(rows)}
    true_ranges: list[float] = []
    atr_values: list[float] = []
    rolling_total = 0.0
    for i, row in enumerate(rows):
        previous_close = float(rows[i-1]["close"]) if i else float(row["close"])
        true_ranges.append(max(float(row["high"])-float(row["low"]),
                               abs(float(row["high"])-previous_close),
                               abs(float(row["low"])-previous_close)))
        rolling_total += true_ranges[-1]
        if i >= config.atr_period:
            rolling_total -= true_ranges[i-config.atr_period]
        atr_values.append(rolling_total/min(i+1, config.atr_period))
    result = []
    for zone in zones:
        start = date_index.get(zone.first_pivot, 0)
        active_start = None
        episodes = []
        breach_index = None
        accepted_closes = 0
        reaction_values: list[float] = []
        touch_indices: list[int] = []
        wick_reclaim = False
        wick_rejection = False
        for index in range(start, len(rows)):
            row = rows[index]
            overlap = float(row["low"]) <= zone.upper_bound and float(row["high"]) >= zone.lower_bound
            close = float(row["close"])
            if overlap and active_start is None:
                active_start = index
                touch_indices.append(index)
            elif active_start is not None:
                departure = atr_values[index] * config.departure_atr
                if close < zone.lower_bound-departure or close > zone.upper_bound+departure:
                    end = index-1
                    episodes.append(TouchEpisode(active_start, end,
                        None if not episodes else active_start-episodes[-1].end_index-1,
                        end-active_start+1))
                    active_start = None
            threshold = atr_values[index] * cfg.zone_break_buffer_atr
            wick_reclaim = wick_reclaim or (zone.side == "support" and
                float(row["low"]) < zone.lower_bound and close >= zone.lower_bound)
            wick_rejection = wick_rejection or (zone.side == "resistance" and
                float(row["high"]) > zone.upper_bound and close <= zone.upper_bound)
            breached = close < zone.lower_bound-threshold if zone.side == "support" else close > zone.upper_bound+threshold
            accepted_closes = accepted_closes + 1 if breached else 0
            if accepted_closes >= cfg.zone_acceptance_closes and breach_index is None:
                breach_index = index
        if active_start is not None:
            episodes.append(TouchEpisode(active_start, len(rows)-1,
                None if not episodes else active_start-episodes[-1].end_index-1,
                len(rows)-active_start))
        for episode_index, episode in enumerate(episodes):
            start_i = episode.end_index + 1
            end_i = min(len(rows), start_i + config.reaction_horizon)
            if episode_index + 1 < len(episodes):
                end_i = min(end_i, episodes[episode_index+1].start_index)
            response = rows[start_i:end_i]
            scale = atr_values[episode.end_index] if episode.end_index >= 0 else 0
            if response and scale > 0:
                move = (max(float(r["high"]) for r in response)-zone.upper_bound if zone.side == "support"
                        else zone.lower_bound-min(float(r["low"]) for r in response))
                if move > 0:
                    reaction_values.append(move/scale)
        flip = None
        role_state = "ACTIVE"
        if breach_index is not None:
            later = rows[breach_index+1:]
            role_state = "SUPPORT_CANDIDATE" if zone.side == "resistance" else "RESISTANCE_CANDIDATE"
            flip = "ACCEPTANCE_ABOVE_RESISTANCE" if zone.side == "resistance" else "ACCEPTANCE_BELOW_SUPPORT"
            if zone.side == "resistance":
                if any(float(r["low"]) <= zone.upper_bound and float(r["close"]) >= zone.upper_bound
                       for r in later):
                    flip, role_state = "resistance_to_support_candidate", "SUPPORT_CONFIRMED"
            else:
                accepted_above = 0
                accepted_below = 0
                resistance_confirmed = False
                for i, r in enumerate(later):
                    close_i = float(r["close"])
                    accepted_below = accepted_below + 1 if close_i < zone.lower_bound else 0
                    accepted_above = accepted_above + 1 if close_i > zone.upper_bound else 0
                    if not resistance_confirmed and float(r["high"]) >= zone.lower_bound and close_i <= zone.lower_bound:
                        flip, role_state = "support_to_resistance_candidate", "RESISTANCE_CONFIRMED"
                        resistance_confirmed = True
                    elif resistance_confirmed and accepted_above >= cfg.zone_acceptance_closes and any(
                            float(x["low"]) <= zone.upper_bound and float(x["close"]) >= zone.upper_bound
                            for x in later[i+1:]):
                        flip, role_state = "support_reclaimed", "SUPPORT_CONFIRMED"
                        break
        recency = (len(rows)-1-touch_indices[-1]) if touch_indices else len(rows)
        if breach_index is None and wick_reclaim:
            flip = "WICK_BELOW_SUPPORT_THEN_RECLAIMED"
        elif breach_index is None and wick_rejection:
            flip = "WICK_ABOVE_RESISTANCE_THEN_REJECTED"
        reactions = reaction_values
        mean_reaction = fmean(reactions) if reactions else 0.0
        touch_count = len(episodes)
        strength = "MINOR"
        # Strength needs distinct visits, observable reaction and recent evidence.
        if (len(zone.pivots) >= cfg.meaningful_pivots and touch_count >= cfg.meaningful_touch_episodes and
                mean_reaction >= cfg.meaningful_reaction_atr and recency <= cfg.meaningful_recency_bars):
            strength = "MEANINGFUL"
        if (touch_count >= cfg.major_touch_episodes and mean_reaction >= cfg.major_reaction_atr and
                recency <= cfg.major_recency_bars):
            strength = "MAJOR"
        freshness = ("FRESH" if not episodes else "FIRST_RETEST" if len(episodes) == 1 else
                     "SECOND_RETEST" if len(episodes) == 2 else "HEAVILY_TESTED")
        effective_side = ("support" if role_state == "SUPPORT_CONFIRMED" else
                          "resistance" if role_state == "RESISTANCE_CONFIRMED" else zone.side)
        result.append(replace(zone, side=effective_side, touch_episodes=tuple(episodes), broken=breach_index is not None,
                              role_flip=flip, freshness=freshness,
                              original_side=zone.original_side or zone.side,
                              role_state=role_state, acceptance_close_count=accepted_closes,
                              reaction_count=len(reactions), mean_reaction_atr=mean_reaction,
                              last_touch_index=touch_indices[-1] if touch_indices else None,
                              strength=strength))
    return tuple(result)


def nearest(zones: Sequence[Zone], price: float, side: str) -> Zone | None:
    candidates = [z for z in zones if z.side == side and
                  (not z.broken or z.role_state == ("SUPPORT_CONFIRMED" if side == "support" else "RESISTANCE_CONFIRMED")) and z.strength != "MINOR" and
                  (z.center <= price if side == "support" else z.center >= price)]
    return min(candidates, key=lambda z: (abs(price - z.center), z.zone_id), default=None)


def _zone_values(zones: Sequence[Zone], price: float, current_atr: float) -> dict[str, Any]:
    support, resistance = nearest(zones, price, "support"), nearest(zones, price, "resistance")
    return {
        "zone_count": len(zones),
        "support": _zone_dict(support, price, current_atr),
        "resistance": _zone_dict(resistance, price, current_atr),
    }


def _zone_dict(zone: Zone | None, price: float, current_atr: float) -> dict[str, Any] | None:
    if zone is None:
        return None
    gap = max(0.0, zone.lower_bound - price, price - zone.upper_bound)
    return {"zone_id": zone.zone_id, "lower": zone.lower_bound,
            "upper": zone.upper_bound, "center": zone.center,
            "distance": gap, "distance_atr": gap / current_atr if current_atr else None,
            "pivot_count": len(zone.pivots), "last_update": zone.last_update,
            "broken": zone.broken, "role_flip": zone.role_flip,
            "freshness": zone.freshness, "touch_episode_count": len(zone.touch_episodes),
            "strength": zone.strength, "reaction_count": zone.reaction_count,
            "mean_reaction_atr": zone.mean_reaction_atr, "role_state": zone.role_state,
            "original_side": zone.original_side}


def volume_features(rows: Sequence[Mapping[str, Any]], *, lookback: int = 20) -> dict[str, Any]:
    current = float(rows[-1]["volume"]) if rows else 0.0
    prior = [float(row["volume"]) for row in rows[-lookback-1:-1]]
    average = fmean(prior) if prior else None
    same_time_average = None
    same_time_ratio = None
    if rows:
        stamp = str(rows[-1].get("date", ""))
        current_day, _, clock = stamp.partition("T")
        candidates = []
        for row in rows[:-1]:
            row_stamp = str(row.get("date", ""))
            row_day, _, row_clock = row_stamp.partition("T")
            if (row_day != current_day and row_clock[:5] == clock[:5]
                    and str(row.get("session_type", "NORMAL_SESSION")) == "NORMAL_SESSION"):
                candidates.append(float(row["volume"]))
        candidates = candidates[-lookback:]
        if candidates:
            same_time_average = fmean(candidates)
            same_time_ratio = current / same_time_average if same_time_average else None
    return {"volume": current, "prior_mean_volume": average,
            "relative_volume": current / average if average else None,
            "expansion_vs_prior_mean": current > average if average else None,
            "same_time_mean_volume": same_time_average,
            "same_time_relative_volume": same_time_ratio,
            "same_time_expansion": same_time_ratio > 1 if same_time_ratio is not None else None,
            "same_time_contraction": same_time_ratio < 1 if same_time_ratio is not None else None}


def _weekly_location(weekly_rows: Sequence[Mapping[str, Any]], zones: Sequence[Zone],
                     close: float, weekly_atr: float, cfg: TopDownConfig) -> tuple[str, list[str], dict[str, Any]]:
    supports = [z for z in zones if z.side == "support" and (not z.broken or z.role_state == "SUPPORT_CONFIRMED") and
                z.strength in {"MEANINGFUL", "MAJOR"} and z.center < close]
    resistances = [z for z in zones if z.side == "resistance" and (not z.broken or z.role_state == "RESISTANCE_CONFIRMED") and
                   z.strength in {"MEANINGFUL", "MAJOR"} and z.center > close]
    support = min(supports, key=lambda z: close - z.center, default=None)
    resistance = min(resistances, key=lambda z: z.center - close, default=None)
    support_distance = (close - support.center) / weekly_atr if support and weekly_atr else None
    resistance_distance = (resistance.center - close) / weekly_atr if resistance and weekly_atr else None
    reasons: list[str] = []
    component = {"nearest_support": _zone_dict(support, close, weekly_atr),
                 "nearest_resistance": _zone_dict(resistance, close, weekly_atr),
                 "support_distance_atr": support_distance,
                 "resistance_distance_atr": resistance_distance}
    if not weekly_rows:
        return "UNAVAILABLE", ["WEEKLY_HISTORY_UNAVAILABLE"], component
    highest = max(float(x["high"]) for x in weekly_rows[-26:])
    lowest = min(float(x["low"]) for x in weekly_rows[-26:])
    span = highest - lowest
    location = (close - lowest) / span if span else 0.5
    component.update({"range_26w_high": highest, "range_26w_low": lowest,
                      "range_location_fraction": location})
    if (support_distance is not None and support_distance <= cfg.actionable_support_distance_atr):
        reasons.append("WEEKLY_SUPPORT_PROXIMITY")
        return "APPROACHING_SUPPORT", reasons, component
    broken_resistances = [z for z in zones if z.original_side == "resistance" and
                          z.role_state == "SUPPORT_CONFIRMED" and close > z.upper_bound]
    if broken_resistances:
        reasons.append("WEEKLY_BREAKOUT_CONTEXT")
        component["broken_resistance"] = _zone_dict(
            max(broken_resistances, key=lambda z: z.upper_bound), close, weekly_atr)
        return "BREAKOUT_CONTEXT", reasons, component
    if resistance_distance is not None and resistance_distance <= cfg.actionable_resistance_distance_atr:
        reasons.append("WEEKLY_RESISTANCE_PROXIMITY")
        return "APPROACHING_RESISTANCE", reasons, component
    if support is not None and resistance is not None:
        stop = support.lower_bound-cfg.stop_buffer_atr*weekly_atr
        risk = close-stop
        reward = resistance.lower_bound-close
        rr = reward/risk if risk > 0 else 0
        component.update({"structural_risk": risk, "structural_reward": reward,
                          "structural_rr": rr, "range_26w_position_diagnostic_only": location})
        if rr >= cfg.minimum_room_r:
            return "STRUCTURAL_LOCATION_ACTIONABLE", reasons, component
        reasons.extend(["STRUCTURAL_RR_UNATTRACTIVE", "WEEKLY_STRONG_MID_RANGE"])
        return "MID_RANGE", reasons, component
    component["range_26w_position_diagnostic_only"] = location
    if support is not None:
        return "APPROACHING_SUPPORT", ["WEEKLY_SUPPORT_PROXIMITY"], component
    if resistance is not None:
        return "APPROACHING_RESISTANCE", ["WEEKLY_RESISTANCE_PROXIMITY"], component
    return "NO_MEANINGFUL_WEEKLY_STRUCTURE", ["NO_MEANINGFUL_WEEKLY_STRUCTURE"], component


def structural_location(zones: Sequence[Zone], price: float, current_atr: float,
                        cfg: TopDownConfig) -> tuple[str, dict[str, Any]]:
    """Classify location from nearest enclosing meaningful S/R; range position is excluded."""
    supports = [z for z in zones if z.side == "support" and (not z.broken or z.role_state == "SUPPORT_CONFIRMED") and
                z.strength in {"MEANINGFUL", "MAJOR"} and z.center < price]
    resistances = [z for z in zones if z.side == "resistance" and (not z.broken or z.role_state == "RESISTANCE_CONFIRMED") and
                   z.strength in {"MEANINGFUL", "MAJOR"} and z.center > price]
    support = max(supports, key=lambda z:z.upper_bound, default=None)
    resistance = min(resistances, key=lambda z:z.lower_bound, default=None)
    sd = (price-support.upper_bound)/current_atr if support and current_atr else None
    rd = (resistance.lower_bound-price)/current_atr if resistance and current_atr else None
    stop = support.lower_bound-cfg.stop_buffer_atr*current_atr if support and current_atr else None
    risk = price-stop if stop is not None else None
    reward = resistance.lower_bound-price if resistance else None
    rr = reward/risk if reward is not None and risk and risk > 0 else None
    values = {"nearest_support": _zone_dict(support, price, current_atr),
              "nearest_resistance": _zone_dict(resistance, price, current_atr),
              "support_distance_atr": sd, "resistance_distance_atr": rd,
              "structural_stop": stop, "structural_risk": risk,
              "structural_reward": reward, "structural_rr": rr}
    if support and sd is not None and sd <= cfg.actionable_support_distance_atr and rr is not None and rr >= cfg.minimum_room_r:
        return "SUPPORT_LOCATION_ACTIONABLE", values
    if resistance and rd is not None and rd <= cfg.actionable_resistance_distance_atr and (rr is None or rr < cfg.minimum_room_r):
        return "NEAR_RESISTANCE_POOR_RR", values
    if support and resistance:
        return ("STRUCTURAL_LOCATION_ACTIONABLE" if rr is not None and rr >= cfg.minimum_room_r
                else "MID_RANGE_NOT_ACTIONABLE"), values
    return "NO_MEANINGFUL_ENCLOSING_STRUCTURE", values


def evaluate_daily_candidate(*, security_id: str, symbol: str,
                             daily_rows: Sequence[Mapping[str, Any]],
                             weekly_rows: Sequence[Mapping[str, Any]],
                             nifty_weekly_regime: Mapping[str, Any],
                             weekly_selection: Mapping[str, Any] | None = None,
                             config: TopDownConfig | None = None) -> dict[str, Any]:
    """Evaluate one completed daily bar using only the supplied historical prefixes."""
    cfg = config or TopDownConfig()
    if not daily_rows:
        return {"state": "UNINTERESTING", "reason_codes": ["DAILY_HISTORY_UNAVAILABLE"]}
    daily = daily_rows[-1]
    day = str(daily["date"])[:10]
    close = float(daily["close"])
    d_atr = atr(daily_rows)
    w_atr = atr(weekly_rows)
    daily_zones = _zones(daily_rows, security_id=security_id, config=cfg.daily_zone)
    weekly_zones = _zones(weekly_rows, security_id=security_id, config=cfg.weekly_zone)
    daily_zones = _lifecycle_snapshot(daily_zones, daily_rows, cfg.daily_zone, cfg)
    weekly_zones = _lifecycle_snapshot(weekly_zones, weekly_rows, cfg.weekly_zone, cfg)
    broken_daily_supports = [z for z in daily_zones if z.original_side == "support" and z.broken and
                             z.role_state != "SUPPORT_CONFIRMED"]
    broken_anchor = max(broken_daily_supports, key=lambda z:z.upper_bound, default=None)
    next_support_watch = (max([z for z in daily_zones if z.side == "support" and (not z.broken or z.role_state == "SUPPORT_CONFIRMED") and
        z.strength in {"MEANINGFUL", "MAJOR"} and broken_anchor and z.upper_bound < broken_anchor.lower_bound],
        key=lambda z:z.upper_bound, default=None) if broken_anchor else None)
    daily_context = _zone_values(daily_zones, close, d_atr)
    daily_location, daily_location_values = structural_location(daily_zones, close, d_atr, cfg)
    weekly_context = _zone_values(weekly_zones, float(weekly_rows[-1]["close"]), w_atr) if weekly_rows else {
        "zone_count": 0, "support": None, "resistance": None}
    location, location_reasons, location_values = _weekly_location(weekly_rows, weekly_zones,
                                                                   float(weekly_rows[-1]["close"]) if weekly_rows else close,
                                                                   w_atr, cfg)
    reasons = list(location_reasons)
    if next_support_watch:
        reasons.append("NEXT_SUPPORT_WATCH_ZONE")
    for timeframe, context in (("WEEKLY", weekly_context), ("DAILY", daily_context)):
        for side in ("support", "resistance"):
            zone = context.get(side)
            if zone and zone.get("broken"):
                reasons.append(f"{timeframe}_{side.upper()}_ZONE_BROKEN")
            if zone and zone.get("role_flip"):
                reasons.append(f"{timeframe}_{zone['role_flip'].upper()}")
    regime = dict(nifty_weekly_regime)
    # Market regime is a soft context: reported but never alone rejects an otherwise valid setup.
    selection = dict(weekly_selection or {})
    nifty_ready = int(regime.get("component_values", {}).get("completed_weeks", 0)) >= cfg.context_min_completed_weeks
    weekly_context_ready = (selection.get("features_available") is True and
        int(selection.get("completed_weeks", 0)) >= cfg.context_min_completed_weeks and
        selection.get("relative_strength_4w") is not None and
        selection.get("relative_strength_13w") is not None)
    near_weekly = location in {"APPROACHING_SUPPORT", "APPROACHING_RESISTANCE",
                               "STRUCTURAL_LOCATION_ACTIONABLE", "BREAKOUT_CONTEXT"}
    structural_location_value = (daily_location if daily_location in {
        "SUPPORT_LOCATION_ACTIONABLE", "STRUCTURAL_LOCATION_ACTIONABLE",
        "NEAR_RESISTANCE_POOR_RR", "MID_RANGE_NOT_ACTIONABLE"} else location)
    weekly_interesting = (near_weekly or structural_location_value in {
        "SUPPORT_LOCATION_ACTIONABLE", "STRUCTURAL_LOCATION_ACTIONABLE"})
    weekly_structure = {"higher_high": selection.get("higher_high"),
                        "higher_low": selection.get("higher_low"),
                        "structure_score": selection.get("structure_score")}
    weak_weekly_structure = (selection.get("features_available") is True
                             and not selection.get("higher_high")
                             and not selection.get("higher_low")
                             and selection.get("status") == "rejected")
    weekly_support_exception = location == "APPROACHING_SUPPORT"
    weekly_breakout_exception = (location == "BREAKOUT_CONTEXT" and
        selection.get("reason") == "nearby_weekly_resistance_breakout_exception_only")
    selector_pass = (selection.get("status") in {"eligible", "neutral"})
    exception_location = any(z.strength in {"MEANINGFUL", "MAJOR"} and (not z.broken or z.role_state == "SUPPORT_CONFIRMED") and z.side == "support"
                             and z.lower_bound <= close <= z.upper_bound + cfg.actionable_support_distance_atr*d_atr
                             for z in daily_zones + weekly_zones)
    zone_context_conflict = ((location == "APPROACHING_SUPPORT" and daily_location == "NEAR_RESISTANCE_POOR_RR") or
        (location == "APPROACHING_RESISTANCE" and daily_location == "SUPPORT_LOCATION_ACTIONABLE"))
    if not nifty_ready:
        reasons.append("NIFTY_CONTEXT_NOT_READY")
        state = "UNINTERESTING"
    elif not weekly_context_ready:
        reasons.append("WEEKLY_STOCK_CONTEXT_NOT_READY")
        state = "UNINTERESTING"
    elif zone_context_conflict:
        reasons.append("AMBIGUOUS_CURRENT_ZONE_CONTEXT")
        state = "UNINTERESTING"
    elif not weekly_interesting:
        if structural_location_value == "NEAR_RESISTANCE_POOR_RR":
            reasons.append("NEAR_RESISTANCE_POOR_RR")
        elif structural_location_value == "MID_RANGE_NOT_ACTIONABLE" or location == "MID_RANGE":
            reasons.extend(["MID_RANGE_NOT_ACTIONABLE", "STRUCTURAL_RR_UNATTRACTIVE"])
        else:
            reasons.append("NO_ACTIONABLE_STRUCTURAL_LOCATION")
        state = "MID_RANGE_NOT_ACTIONABLE" if structural_location_value == "MID_RANGE_NOT_ACTIONABLE" or location == "MID_RANGE" else "UNINTERESTING"
    elif selector_pass:
        reasons.append("MOMENTUM_SELECTION")
        state = "WATCHLIST"
    elif exception_location:
        reasons.append("STRUCTURAL_LOCATION_EXCEPTION")
        state = "WATCHLIST"
    else:
        reasons.append("WEEKLY_SELECTOR_GATE_FAILED")
        state = "UNINTERESTING"

    daily_supports = [z for z in daily_zones if z.side == "support" and (not z.broken or z.role_state == "SUPPORT_CONFIRMED") and
                      z.strength in {"MEANINGFUL", "MAJOR"} and
                      z.lower_bound <= float(daily["high"]) and z.upper_bound >= float(daily["low"])]
    support_candidates = []
    span = float(daily["high"]) - float(daily["low"])
    close_location = (close - float(daily["low"])) / span if span > 0 else 0.5
    for zone in daily_supports:
        penetrated = (zone.upper_bound - float(daily["low"])) / max(zone.upper_bound-zone.lower_bound, 1e-12)
        if close >= zone.upper_bound and close_location >= cfg.support_reaction_close_location:
            support_candidates.append((zone, penetrated))

    breakout_candidates = []
    max_age = cfg.breakout_retest_max_sessions
    for zone in daily_zones:
        if zone.side != "support" or zone.original_side != "resistance" or zone.role_state != "SUPPORT_CONFIRMED":
            continue
        prior_above = next((i for i in range(len(daily_rows)-2, max(-1, len(daily_rows)-max_age-2), -1)
                            if float(daily_rows[i]["close"]) > zone.upper_bound), None)
        if prior_above is None:
            continue
        if float(daily["low"]) <= zone.upper_bound and close >= zone.center:
            breakout_candidates.append(zone)

    setup = None
    setup_zone: Zone | None = None
    if len(breakout_candidates) == 1:
        setup, setup_zone = "BREAKOUT_RETEST", breakout_candidates[0]
    elif len(support_candidates) == 1:
        setup, (setup_zone, _) = "SUPPORT_REACTION", support_candidates[0]
    elif len(breakout_candidates) > 1 or len(support_candidates) > 1:
        reasons.append("AMBIGUOUS_CURRENT_ZONE_CONTEXT")
        state = "UNINTERESTING"
    if setup:
        reasons.append("DAILY_SETUP_BREAKOUT_RETEST" if setup == "BREAKOUT_RETEST"
                       else "DAILY_SETUP_SUPPORT_REACTION")
        if state in {"MID_RANGE_NOT_ACTIONABLE", "UNINTERESTING"}:
            reasons.append("WEEKLY_LOCATION_GATE_FAILED")
            setup = None
            setup_zone = None
        else:
            state = "WAITING_1H_CONFIRMATION"
    elif state == "WATCHLIST":
        daily_nearby = any(
            isinstance(daily_context.get(side), dict)
            and daily_context[side].get("distance_atr") is not None
            and daily_context[side]["distance_atr"] <= cfg.daily_approach_distance_atr
            for side in ("support", "resistance"))
        if location in {"APPROACHING_SUPPORT", "APPROACHING_RESISTANCE"} or daily_nearby:
            state = "APPROACHING_ZONE"

    daily_swings = find_confirmed_swings(daily_rows, left=cfg.daily_zone.swing_left,
                                         right=cfg.daily_zone.swing_right)
    lows = [s.price for s in daily_swings if s.kind == "support" and s.confirmation_index < len(daily_rows)]
    highs = [s.price for s in daily_swings if s.kind == "resistance" and s.confirmation_index < len(daily_rows)]
    output = {
        "date": day, "security_id": security_id, "symbol": symbol,
        "state": state, "setup_type": setup, "reason_codes": sorted(set(reasons)),
        "nifty_regime": regime.get("regime", "NEUTRAL"),
        "nifty_regime_component_values": regime.get("component_values", {}),
        "nifty_regime_reasons": regime.get("reason_codes", []),
        "weekly_selector_status": selection.get("status"),
        "weekly_selector_score": selection.get("total_score"),
        "weekly_selection_components": {key: selection.get(key) for key in (
            "completed_weeks", "relative_strength_4w", "relative_strength_13w",
            "return_4w", "return_13w", "weekly_slope_pct_per_week", "trend_efficiency_13w",
            "higher_high", "higher_low", "weekly_volatility_13w", "status", "reason")},
        "weekly_filter_pass": selector_pass,
        "NIFTY_CONTEXT_READY": nifty_ready,
        "WEEKLY_STOCK_CONTEXT_READY": weekly_context_ready,
        "weekly_selection_path": ("MOMENTUM_SELECTION" if selector_pass else
                                   "STRUCTURAL_LOCATION_EXCEPTION" if exception_location else None),
        "weekly_location": location, "weekly_location_components": location_values,
        "daily_location": daily_location, "daily_location_components": daily_location_values,
        "structural_range_position": daily_location,
        "weekly_structure": {"higher_high": len(highs) >= 2 and highs[-1] > highs[-2],
                             "higher_low": len(lows) >= 2 and lows[-1] > lows[-2],
                             "selector_higher_high": weekly_structure["higher_high"],
                             "selector_higher_low": weekly_structure["higher_low"],
                             "selector_structure_score": weekly_structure["structure_score"],
                             "confirmed_swing_highs": len(highs), "confirmed_swing_lows": len(lows)},
        "weekly_zones": weekly_context, "daily_zones": daily_context,
        "daily_atr": d_atr, "weekly_atr": w_atr,
        "daily_ema50": ema(daily_rows, 50), "daily_ema200": ema(daily_rows, 200),
        "weekly_ema50": ema(weekly_rows, 50), "weekly_ema200": ema(weekly_rows, 200),
        "ema_context": {
            "daily_close_above_ema50": close > ema(daily_rows, 50) if ema(daily_rows, 50) is not None else None,
            "daily_close_above_ema200": close > ema(daily_rows, 200) if ema(daily_rows, 200) is not None else None,
            "weekly_close_above_ema50": float(weekly_rows[-1]["close"]) > ema(weekly_rows, 50)
                if weekly_rows and ema(weekly_rows, 50) is not None else None,
            "weekly_close_above_ema200": float(weekly_rows[-1]["close"]) > ema(weekly_rows, 200)
                if weekly_rows and ema(weekly_rows, 200) is not None else None,
        },
        "weekly_volume": volume_features(weekly_rows),
        "daily_volume": volume_features(daily_rows),
        "daily_bar": {key: float(daily[key]) for key in ("open", "high", "low", "close")},
        "setup_zone": _zone_dict(setup_zone, close, d_atr) if setup_zone else None,
        "current_zone_state": setup_zone.role_state if setup_zone else None,
        "role_flip_state": setup_zone.role_flip if setup_zone else None,
        "next_support_watch_zone": _zone_dict(next_support_watch, close, d_atr),
        "support_reaction_penetration_fraction": support_candidates[0][1] if len(support_candidates) == 1 else None,
        "daily_close_location": close_location,
    }
    return output


def _hourly_pattern(rows: Sequence[Mapping[str, Any]], zones: Sequence[Zone],
                    setup_zone: Mapping[str, Any] | None, consolidation_bars: int,
                    require_stronger: bool = False, structure_lookback: int = 130) -> tuple[str | None, list[str], dict[str, Any]]:
    if not rows:
        return None, ["NO_COMPLETED_HOURLY_BARS"], {}
    current = rows[-1]
    prior = rows[-2] if len(rows) >= 2 else None
    close = float(current["close"])
    supports = [z for z in zones if z.side == "support"]
    resistances = [z for z in zones if z.side == "resistance"]
    values: dict[str, Any] = {"bar_timestamp": str(current["date"]),
                              "bar_volume": float(current["volume"]),
                              **volume_features(rows), "hourly_ema50": ema(rows, 50),
                              "hourly_ema200": ema(rows, 200),
                              "hourly_close_above_ema50": close > ema(rows, 50) if ema(rows, 50) is not None else None,
                              "hourly_close_above_ema200": close > ema(rows, 200) if ema(rows, 200) is not None else None}
    # Hourly structure is local execution context; the configured zone age bounds
    # it and avoids repeatedly scanning the entire historical prefix.
    swing_rows = rows[-structure_lookback:]
    swings = find_confirmed_swings(swing_rows, left=2, right=2)
    lows = [x.price for x in swings if x.kind == "support"]
    highs = [x.price for x in swings if x.kind == "resistance"]
    higher_low = len(lows) >= 2 and lows[-1] > lows[-2]
    higher_high = len(highs) >= 2 and highs[-1] > highs[-2]
    hl_to_hh = higher_low and higher_high
    bearish_failures = 0
    start_failure = max(1, len(rows)-6)
    for i in range(start_failure, len(rows)):
        local_floor = min(float(x["low"]) for x in rows[max(0, i-3):i])
        if float(rows[i]["low"]) <= local_floor and float(rows[i]["close"]) > local_floor:
            bearish_failures += 1
    consolidation = len(rows) >= consolidation_bars and (
        max(float(x["high"]) for x in rows[-consolidation_bars:])-
        min(float(x["low"]) for x in rows[-consolidation_bars:]) <= atr(rows, 14))
    values.update({"higher_low": higher_low, "higher_high": higher_high,
                   "higher_low_to_higher_high": hl_to_hh,
                   "failed_downside_attempts": bearish_failures,
                   "consolidation": consolidation,
                   "bullish_close": close > float(current["open"]),
                   "involved_zones": [{"zone_id": z.zone_id, "role_before": z.original_side,
                       "role_state": z.role_state, "role_flip": z.role_flip,
                       "strength": z.strength, "timeframe": "HOURLY"} for z in supports+resistances]})
    held = [z for z in supports if float(current["low"]) <= z.upper_bound and
            float(current["high"]) >= z.lower_bound and close >= z.upper_bound and close > float(current["open"])]
    strong_sequence = bool(held) and close > float(current["open"]) and higher_low
    reclaim_hold = bool(prior and any(float(prior["close"]) < z.lower_bound and
        float(current["low"]) <= z.upper_bound and close >= z.upper_bound for z in supports+resistances))
    if require_stronger and not (strong_sequence or hl_to_hh or
            reclaim_hold or
            (consolidation and bearish_failures >= 2 and prior and close > float(prior["high"]))):
        values["strong_confirmation_required"] = True
        return None, ["STRUCTURAL_EXCEPTION_REQUIRES_STRONGER_H1_CONFIRMATION"], values
    # Archetype tests are ordered and use only completed bars through current.
    if prior:
        reclaimed = [z for z in supports + resistances
                     if float(prior["close"]) < z.lower_bound and close > z.upper_bound]
        if reclaimed:
            return "RECLAIM_CONFIRMATION", ["HOURLY_ZONE_RECLAIMED"], values
    if held:
        return "SUPPORT_HOLD_CONFIRMATION", ["HOURLY_SUPPORT_HELD_WITH_BULLISH_CLOSE"], values
    if hl_to_hh:
        return "SUPPORT_HOLD_CONFIRMATION", ["HOURLY_HIGHER_LOW_TO_HIGHER_HIGH"], values
    if len(rows) >= consolidation_bars + 1:
        base = rows[-consolidation_bars-1:-1]
        prior_base = rows[-2*consolidation_bars-1:-consolidation_bars-1]
        if len(prior_base) == consolidation_bars:
            base_range = max(float(x["high"]) for x in base) - min(float(x["low"]) for x in base)
            prior_range = max(float(x["high"]) for x in prior_base) - min(float(x["low"]) for x in prior_base)
            resistance = max(float(x["high"]) for x in base)
            values.update({"consolidation_range": base_range, "prior_range": prior_range,
                           "consolidation_compressed": base_range < prior_range,
                           "breakout_level": resistance})
            if base_range < prior_range and close > resistance:
                return "CONSOLIDATION_BREAKOUT_CONFIRMATION", ["HOURLY_COMPRESSED_RANGE_BROKEN_UP"], values
    if prior:
        retested = [z for z in resistances if float(prior["close"]) > z.upper_bound and
                    float(current["low"]) <= z.upper_bound and close >= z.center]
        if retested:
            return "BREAKOUT_RETEST_CONFIRMATION", ["HOURLY_RESISTANCE_RETEST_HELD"], values
    # A daily setup level can serve as the breakout/retest reference when local H1
    # pivots have not yet produced a two-pivot resistance zone.
    if prior and setup_zone:
        upper = float(setup_zone["upper"])
        center = float(setup_zone["center"])
        if float(prior["close"]) > upper and float(current["low"]) <= upper and close >= center:
            return "BREAKOUT_RETEST_CONFIRMATION", ["HOURLY_DAILY_SETUP_ZONE_RETEST_HELD"], values
    return None, [], values


def structural_plan(*, entry: float, confirmation_close: float,
                    weekly_zones: Sequence[Zone], daily_zones: Sequence[Zone],
                    hourly_zones: Sequence[Zone], setup_zone: Mapping[str, Any] | None,
                    daily_atr: float, hourly_atr: float, cfg: TopDownConfig,
                    ema_obstacles: Mapping[str, float | None] | None = None) -> dict[str, Any]:
    """Build an auditable structural plan; EMAs/minor zones are friction only."""
    meaningful = {"MEANINGFUL", "MAJOR"}
    ema_values = {name: float(value) for name, value in (ema_obstacles or {}).items()
                  if value is not None}
    confluence_tolerance = cfg.ema_confluence_distance_atr * max(daily_atr, hourly_atr, 1e-12)
    all_zones = list(weekly_zones) + list(daily_zones) + list(hourly_zones)
    confluence_ids = {z.zone_id for z in all_zones if z.strength in meaningful and
        any(abs(ema_value-z.center) <= confluence_tolerance for ema_value in ema_values.values())}
    references: list[tuple[str, float, Zone | None]] = []
    for timeframe, zones in (("HOURLY", hourly_zones), ("DAILY", daily_zones), ("WEEKLY", weekly_zones)):
        candidates = [z for z in zones if z.side == "support" and (not z.broken or z.role_state == "SUPPORT_CONFIRMED") and
                      z.strength in meaningful and z.lower_bound < entry]
        if timeframe == "HOURLY":
            candidates = [z for z in candidates if z.strength == "MAJOR" or z.reaction_count >= 4 or z.zone_id in confluence_ids]
        zone = max(candidates, key=lambda z: z.lower_bound, default=None)
        if zone:
            references.append((timeframe, zone.lower_bound, zone))
    if setup_zone and float(setup_zone["lower"]) < entry:
        references.append(("DAILY_SETUP", float(setup_zone["lower"]), None))
    selected = max(references, key=lambda item: item[1], default=None)
    if selected is None:
        return {"status": "SETUP_INVALIDATED", "reason_code": "NO_MEANINGFUL_STRUCTURAL_STOP",
                "stop": None, "stop_reference": None, "risk_per_share": None}
    timeframe, support_lower, zone = selected
    buffer_atr = hourly_atr or daily_atr
    stop = support_lower - cfg.stop_buffer_atr * buffer_atr
    risk = entry - stop
    if risk <= 0:
        return {"status": "SETUP_INVALIDATED", "reason_code": "NON_POSITIVE_STRUCTURAL_RISK",
                "stop": stop, "stop_reference": timeframe, "risk_per_share": risk}
    obstacles: list[tuple[str, float, Zone | None, str]] = []
    for tf, zones in (("HOURLY", hourly_zones), ("DAILY", daily_zones), ("WEEKLY", weekly_zones)):
        for z in zones:
            if z.side != "resistance" or (z.broken and z.role_state != "RESISTANCE_CONFIRMED") or z.lower_bound <= entry or z.strength not in meaningful:
                continue
            if tf == "HOURLY" and not (z.strength == "MAJOR" or z.reaction_count >= 4 or z.zone_id in confluence_ids):
                continue
            obstacles.append((tf, z.lower_bound, z, "PRIMARY_STRUCTURAL_OBSTACLE"))
    friction: list[tuple[str, float, Zone | None, str]] = []
    for tf, zones in (("HOURLY", hourly_zones), ("DAILY", daily_zones), ("WEEKLY", weekly_zones)):
        friction.extend((tf, z.lower_bound, z, "SECONDARY_FRICTION") for z in zones
                        if z.side == "resistance" and (not z.broken or z.role_state == "RESISTANCE_CONFIRMED") and z.lower_bound > entry
                        and (z.strength == "MINOR" or (tf == "HOURLY" and not (
                            z.strength in meaningful and (z.strength == "MAJOR" or
                            z.reaction_count >= 4 or z.zone_id in confluence_ids)))))
    friction.extend((name, float(value), None, "SECONDARY_FRICTION")
                    for name, value in (ema_obstacles or {}).items()
                    if value is not None and float(value) > entry)
    all_obstacles = obstacles + friction
    obstacle_rows = []
    for tf, level, z, classification in sorted(all_obstacles, key=lambda x: (x[1], x[0])):
        obstacle_rows.append({"type": "STRUCTURAL" if z else "EMA_FRICTION", "timeframe": tf,
            "price": level, "zone_strength": z.strength if z else None,
            "touch_count": len(z.touch_episodes) if z else 0,
            "reaction_count": z.reaction_count if z else 0,
            "distance_pct": (level-entry)/entry if entry else None,
            "distance_r": (level-entry)/risk if risk > 0 else None,
            "classification": classification, "zone_id": z.zone_id if z else None,
            "ema_confluence": bool(z and z.zone_id in confluence_ids)})
    primary = min(obstacles, key=lambda x: (x[1], {"DAILY": 0, "WEEKLY": 1, "HOURLY": 2}.get(x[0], 3)), default=None)
    if primary:
        target_ref, target, target_zone, _ = primary
        reward_r = (target-entry)/risk if risk > 0 else None
    else:
        target_ref, target_zone = "FIXED_2R_RESEARCH_FALLBACK", None
        target = entry + cfg.no_resistance_fallback_r * risk
        reward_r = cfg.no_resistance_fallback_r
    if primary and reward_r < cfg.minimum_room_r:
        status, reason = "ROOM_INSUFFICIENT", "STRUCTURAL_ROOM_BELOW_MINIMUM_R"
    else:
        status, reason = "ENTRY_READY", "STRUCTURAL_ROOM_ACCEPTABLE"
    risk_budget = cfg.initial_capital * cfg.risk_fraction
    quantity = min(floor(risk_budget/risk), floor(cfg.initial_capital/entry)) if entry > 0 else 0
    plan = {"status": status, "reason_code": reason, "entry": entry, "confirmation_close": confirmation_close,
            "stop": stop, "stop_reference": timeframe, "stop_zone_id": zone.zone_id if zone else None,
            "risk_per_share": risk, "risk_budget": risk_budget, "quantity": max(0, quantity),
            "planned_rupee_risk": max(0, quantity)*risk, "target": target,
            "target_reference": target_ref, "planned_reward_r": reward_r,
            "target_zone_id": target_zone.zone_id if target_zone else None,
            "target_zone_strength": target_zone.strength if target_zone else None,
            "target_timeframe": target_ref, "obstacles": obstacle_rows,
            "overhead_friction_present": bool(friction),
            "ema_context": {"ema50_distance_pct": {k: (v-entry)/entry if entry else None
                              for k, v in ema_values.items() if "EMA50" in k},
                            "ema200_distance_pct": {k: (v-entry)/entry if entry else None
                              for k, v in ema_values.items() if "EMA200" in k},
                            "ema50_near_support": any("EMA50" in k and any(z.side == "support" and z.zone_id in confluence_ids and
                                abs(v-z.center) <= confluence_tolerance for z in all_zones) for k,v in ema_values.items()),
                            "ema200_near_support": any("EMA200" in k and any(z.side == "support" and z.zone_id in confluence_ids and
                                abs(v-z.center) <= confluence_tolerance for z in all_zones) for k,v in ema_values.items()),
                            "ema50_near_resistance": any("EMA50" in k and any(z.side == "resistance" and z.zone_id in confluence_ids and
                                abs(v-z.center) <= confluence_tolerance for z in all_zones) for k,v in ema_values.items()),
                            "ema200_near_resistance": any("EMA200" in k and any(z.side == "resistance" and z.zone_id in confluence_ids and
                                abs(v-z.center) <= confluence_tolerance for z in all_zones) for k,v in ema_values.items()),
                            "ema_sr_confluence": sorted(confluence_ids)},
            "minimum_room_r": cfg.minimum_room_r, "position_value": max(0, quantity)*entry,
            "cash_limited": quantity == floor(cfg.initial_capital/entry) if entry > 0 else True}
    plan.update(plan["ema_context"])
    return plan


def confirm_hourly(*, hourly_rows: Sequence[Mapping[str, Any]], after_timestamp: datetime,
                   setup_zone: Mapping[str, Any] | None, weekly_zones: Sequence[Zone],
                   daily_zones: Sequence[Zone], hourly_config: ZoneConfig,
                   ema_obstacles: Mapping[str, float | None] | None = None,
                   require_stronger_confirmation: bool = False,
                   start_index: int = 0,
                   security_id: str = "hourly",
                   config: TopDownConfig | None = None) -> dict[str, Any]:
    """Find the first completed H1 confirmation after the daily signal.

    The actual entry is the next available provider bar's open. No same-bar or
    signal-day hourly data is eligible.
    """
    cfg = config or TopDownConfig()
    eligible = [i for i, row in enumerate(hourly_rows)
                if i >= start_index and _as_datetime(row["date"]) > after_timestamp]
    for index in eligible:
        prefix = hourly_rows[:index+1]
        zone_prefix = prefix[-hourly_config.maximum_age:]
        zones = _lifecycle_snapshot(_zones(zone_prefix, security_id=security_id, config=hourly_config),
                                    zone_prefix, hourly_config, cfg)
        pattern, reasons, feature_values = _hourly_pattern(prefix, zones, setup_zone,
                                                           cfg.consolidation_bars,
                                                           require_stronger_confirmation,
                                                           hourly_config.maximum_age)
        if not pattern:
            continue
        confirmation = prefix[-1]
        next_row = next((row for row in hourly_rows[index+1:]
                         if _as_datetime(row["date"]) > _as_datetime(confirmation["date"])), None)
        if next_row is None:
            return {"state": "WAITING_1H_CONFIRMATION", "reason_codes": ["AWAITING_NEXT_EXECUTABLE_HOURLY_BAR"],
                    "confirmation_type": pattern, "confirmation_timestamp": str(confirmation["date"]),
                    "confirmation_index": index,
                    "confirmation_features": feature_values}
        fill = float(next_row["open"])
        plan = structural_plan(entry=fill, confirmation_close=float(confirmation["close"]),
                               weekly_zones=weekly_zones, daily_zones=daily_zones,
                               hourly_zones=zones, setup_zone=setup_zone,
                               daily_atr=atr(prefix, 14), hourly_atr=atr(prefix, 14), cfg=cfg,
                               ema_obstacles={**dict(ema_obstacles or {}),
                                              "HOURLY_EMA50": ema(prefix, 50),
                                              "HOURLY_EMA200": ema(prefix, 200)})
        state = plan.pop("status")
        return {"state": state, "reason_codes": reasons + [plan["reason_code"]],
                "confirmation_type": pattern,
                "confirmation_index": index,
                "confirmation_timestamp": str(confirmation["date"]),
                "entry_timestamp": str(next_row["date"]), "entry_type": "NEXT_AVAILABLE_1H_OPEN",
                "hours_waited": (_as_datetime(next_row["date"])-after_timestamp).total_seconds()/3600,
                "confirmation_features": feature_values, "plan": plan}
    return {"state": "WAITING_1H_CONFIRMATION", "reason_codes": ["NO_1H_ARCHETYPE_YET"]}


def _as_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))


def config_as_dict(config: TopDownConfig | None = None) -> dict[str, Any]:
    return asdict(config or TopDownConfig())


__all__ = ["TopDownConfig", "atr", "ema", "volume_features", "evaluate_daily_candidate",
           "confirm_hourly", "structural_plan", "config_as_dict"]
