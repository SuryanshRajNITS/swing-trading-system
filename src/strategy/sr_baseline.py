"""Objective daily support/resistance swing baseline (sr_baseline_v1).

All signals are generated from the as-of daily history supplied by the event
engine. A swing is usable only after its right-hand confirmation bars exist.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from typing import Any, Mapping, Sequence

from src.decision.interfaces import EntrySignal, StrategyContext


@dataclass(frozen=True)
class SwingPoint:
    kind: str
    pivot_index: int
    confirmation_index: int
    price: float


@dataclass(frozen=True)
class PriceZone:
    kind: str
    center: float
    lower: float
    upper: float
    pivot_count: int
    first_pivot_index: int
    last_pivot_index: int


@dataclass(frozen=True)
class SRBaselineConfig:
    swing_left: int = 3
    swing_right: int = 3
    zone_tolerance_pct: float = 0.01
    minimum_zone_pivots: int = 2
    maximum_zone_age_sessions: int = 252
    minimum_touch_separation_sessions: int = 5
    support_min_close_location: float = 0.65
    support_max_penetration_pct: float = 1.0
    zone_break_close_buffer_pct: float = 0.005
    support_stop_buffer_pct: float = 0.002
    breakout_distance_pct: float = 0.002
    retest_tolerance_pct: float = 0.01
    maximum_retest_sessions: int = 20
    breakout_invalidation_pct: float = 0.01
    breakout_minimum_touches: int = 2
    breakout_min_close_location: float = 0.60
    breakout_stop_buffer_pct: float = 0.002
    volume_lookback_sessions: int = 20
    volume_ratio_threshold: float | None = None
    weekly_sma_weeks: int = 4
    weekly_require_higher_close: bool = True
    strategy_version: str = "sr_baseline_v1"

    def __post_init__(self) -> None:
        positive_ints = (self.swing_left, self.swing_right, self.minimum_zone_pivots,
                         self.maximum_zone_age_sessions, self.minimum_touch_separation_sessions,
                         self.maximum_retest_sessions, self.volume_lookback_sessions,
                         self.weekly_sma_weeks)
        if any(value < 1 for value in positive_ints):
            raise ValueError("Strategy lookbacks and counts must be positive integers.")
        fractions = (self.zone_tolerance_pct, self.support_min_close_location,
                     self.support_max_penetration_pct, self.zone_break_close_buffer_pct,
                     self.support_stop_buffer_pct, self.breakout_distance_pct,
                     self.retest_tolerance_pct, self.breakout_invalidation_pct,
                     self.breakout_min_close_location, self.breakout_stop_buffer_pct)
        if any(value < 0 or value > 1 for value in fractions):
            raise ValueError("Strategy percentages and close-location thresholds must be between 0 and 1.")
        if self.volume_ratio_threshold is not None and self.volume_ratio_threshold <= 0:
            raise ValueError("volume_ratio_threshold must be positive or null.")
        if not self.strategy_version:
            raise ValueError("strategy_version must not be empty.")


def _field(row: Mapping[str, Any], name: str) -> float:
    return float(row[name])


def find_confirmed_swings(
    rows: Sequence[Mapping[str, Any]], *, left: int, right: int,
) -> list[SwingPoint]:
    """Return unique local extrema, dated by the bar when confirmation became known."""
    if left < 1 or right < 1:
        raise ValueError("Swing windows must be positive.")
    result: list[SwingPoint] = []
    for pivot in range(left, len(rows) - right):
        high_window = [_field(rows[index], "high") for index in range(pivot-left, pivot+right+1)]
        low_window = [_field(rows[index], "low") for index in range(pivot-left, pivot+right+1)]
        pivot_high = _field(rows[pivot], "high")
        pivot_low = _field(rows[pivot], "low")
        if pivot_high == max(high_window) and high_window.count(pivot_high) == 1:
            result.append(SwingPoint("resistance", pivot, pivot + right, pivot_high))
        if pivot_low == min(low_window) and low_window.count(pivot_low) == 1:
            result.append(SwingPoint("support", pivot, pivot + right, pivot_low))
    return result


def cluster_zones(
    swings: Sequence[SwingPoint], *, tolerance_pct: float,
) -> list[PriceZone]:
    """Cluster same-side pivot prices in chronological order using a fixed-centre tolerance."""
    if tolerance_pct < 0 or tolerance_pct > 1:
        raise ValueError("tolerance_pct must be between zero and one.")
    buckets: list[list[SwingPoint]] = []
    bucket_sums: list[float] = []
    ordered = sorted(swings, key=lambda item: (item.kind, item.price, item.confirmation_index, item.pivot_index))
    for swing in ordered:
        bucket = buckets[-1] if buckets and buckets[-1][0].kind == swing.kind else None
        bucket_index = len(buckets)-1 if bucket is not None else -1
        center = bucket_sums[bucket_index] / len(bucket) if bucket else 0.0
        if bucket and swing.price-center <= center*tolerance_pct:
            bucket.append(swing)
            bucket_sums[bucket_index] += swing.price
        else:
            buckets.append([swing])
            bucket_sums.append(swing.price)
    zones: list[PriceZone] = []
    for bucket in buckets:
        center = sum(item.price for item in bucket) / len(bucket)
        zones.append(PriceZone(
            bucket[0].kind, center, center * (1-tolerance_pct), center * (1+tolerance_pct),
            len(bucket), min(item.pivot_index for item in bucket), max(item.pivot_index for item in bucket),
        ))
    return zones


def _touch_episodes(rows: Sequence[Mapping[str, Any]], zone: PriceZone) -> list[int]:
    """Return first bar indices of contiguous OHLC overlaps with a zone."""
    starts: list[int] = []
    inside = False
    for index, row in enumerate(rows):
        overlaps = _field(row, "low") <= zone.upper and _field(row, "high") >= zone.lower
        if overlaps and not inside:
            starts.append(index)
        inside = overlaps
    return starts


def _close_location(row: Mapping[str, Any]) -> float:
    high, low, close = (_field(row, key) for key in ("high", "low", "close"))
    return (close - low) / (high - low) if high > low else 0.5


def _weekly_filter(context: StrategyContext, security_id: str, config: SRBaselineConfig) -> bool:
    latest = context.latest_completed_weekly.get(security_id)
    if latest is None:
        return False
    # Weekly candles are intentionally derived only from daily rows available as of this close.
    current_week = date.fromordinal(context.session_date.toordinal() - context.session_date.weekday())
    # Exclude the signal date's calendar week entirely, even on Friday's close.
    weekly_closes: dict[date, float] = {}
    for row in context.daily_history.get(security_id, ()):
        day = date.fromisoformat(str(row["date"])[:10])
        week_start = date.fromordinal(day.toordinal()-day.weekday())
        if week_start < current_week:
            weekly_closes[week_start] = _field(row, "close")
    completed = [weekly_closes[key] for key in sorted(weekly_closes)]
    count = config.weekly_sma_weeks
    if len(completed) < count:
        return False
    closes = completed[-count:]
    reference_close = completed[-1]
    if reference_close <= sum(closes) / count:
        return False
    if config.weekly_require_higher_close and len(completed) > 1:
        return reference_close > completed[-2]
    return True


@dataclass(frozen=True)
class _BreakoutWatch:
    zone: PriceZone
    breakout_index: int
    breakout_date: str
    breakout_close: float
    breakout_volume_ratio: float | None
    touch_count: int
    first_touch_date: str
    most_recent_touch_date: str


class SRBaselineStrategy:
    """Daily support-test and resistance-breakout/retest research baseline."""

    name = "sr_baseline"
    version = "sr_baseline_v1"

    def __init__(
        self, *, config: SRBaselineConfig | None = None, setup: str = "both",
        target_r: float = 2.0, weekly_filter: bool = False,
        evaluation_start: date = date(2023, 1, 1),
    ) -> None:
        if setup not in {"support_test", "breakout_retest", "both"}:
            raise ValueError("setup must be support_test, breakout_retest, or both")
        if target_r not in {1.5, 2.0, 2.5, 3.0}:
            raise ValueError("target_r must be one of 1.5, 2.0, 2.5, or 3.0")
        self.config = config or SRBaselineConfig()
        self.setup = setup
        self.target_r = target_r
        self.weekly_filter = weekly_filter
        self.evaluation_start = evaluation_start
        self.parameters = {**asdict(self.config), "setup": setup, "target_r": target_r,
                           "weekly_filter": weekly_filter, "evaluation_start": evaluation_start.isoformat()}
        self.diagnostics: list[dict[str, Any]] = []
        self._breakout_watches: dict[str, list[_BreakoutWatch]] = {}
        self._processed_dates: dict[str, str] = {}
        self._swing_cache: dict[str, tuple[tuple[str, ...], list[SwingPoint]]] = {}

    def _confirmed_swings(self, security_id: str, rows: Sequence[Mapping[str, Any]]) -> list[SwingPoint]:
        dates = tuple(str(row["date"]) for row in rows)
        cached = self._swing_cache.get(security_id)
        if cached is None or len(dates) < len(cached[0]) or dates[:len(cached[0])] != cached[0]:
            swings = find_confirmed_swings(rows, left=self.config.swing_left, right=self.config.swing_right)
            self._swing_cache[security_id] = (dates, swings)
            return swings
        cached_dates, cached_swings = cached
        swings = list(cached_swings)
        for confirmation in range(max(len(cached_dates), self.config.swing_left+self.config.swing_right), len(rows)):
            pivot = confirmation-self.config.swing_right
            start, end = pivot-self.config.swing_left, pivot+self.config.swing_right+1
            high_values = [_field(rows[index], "high") for index in range(start, end)]
            low_values = [_field(rows[index], "low") for index in range(start, end)]
            high, low = _field(rows[pivot], "high"), _field(rows[pivot], "low")
            if high == max(high_values) and high_values.count(high) == 1:
                swings.append(SwingPoint("resistance", pivot, confirmation, high))
            if low == min(low_values) and low_values.count(low) == 1:
                swings.append(SwingPoint("support", pivot, confirmation, low))
        self._swing_cache[security_id] = (dates, swings)
        return swings

    def _zones(self, rows: Sequence[Mapping[str, Any]], security_id: str = "") -> tuple[list[PriceZone], list[PriceZone]]:
        confirmed = self._confirmed_swings(security_id, rows) if security_id else find_confirmed_swings(
            rows, left=self.config.swing_left, right=self.config.swing_right)
        zones = cluster_zones(confirmed, tolerance_pct=self.config.zone_tolerance_pct)
        fresh = [zone for zone in zones if len(rows)-1-zone.last_pivot_index <= self.config.maximum_zone_age_sessions]
        return ([zone for zone in fresh if zone.kind == "support"],
                [zone for zone in fresh if zone.kind == "resistance"])

    def _support_signal(self, security_id: str, symbol: str, rows: Sequence[Mapping[str, Any]], context: StrategyContext) -> EntrySignal | None:
        supports, _ = self._zones(rows[:-1], security_id)
        current = rows[-1]
        choices = []
        for zone in supports:
            if zone.pivot_count < self.config.minimum_zone_pivots:
                continue
            if not (_field(current, "low") <= zone.upper and _field(current, "high") >= zone.lower):
                continue
            if len(rows) > 1 and _field(rows[-2], "low") <= zone.upper and _field(rows[-2], "high") >= zone.lower:
                continue
            if _is_broken(rows[:-1], zone, self.config.zone_break_close_buffer_pct):
                self._diagnostic("support_test", context, security_id, symbol, zone,
                                 "SUPPORT_BROKEN", rows, zone_broken=True, zone_state="BROKEN",
                                 touch_count=len(_touch_episodes(rows[:-1], zone)),
                                 zone_age_sessions=len(rows)-1-zone.last_pivot_index,
                                 distance_from_current_price=abs(_field(current, "close")-zone.center))
                continue
            episodes = _touch_episodes(rows[:-1], zone)
            # Include today's revisit as one new episode only when yesterday was outside.
            previous_start = episodes[-1] if episodes else None
            days_since = len(rows)-1-previous_start if previous_start is not None else None
            if previous_start is None or days_since is None or days_since < self.config.minimum_touch_separation_sessions:
                continue
            reaction_window = rows[previous_start:min(len(rows)-1, previous_start+6)]
            if not reaction_window or max(_field(item, "high") for item in reaction_window) <= zone.upper:
                continue
            if _field(current, "close") < zone.lower:
                self._diagnostic("support_test", context, security_id, symbol, zone, "SUPPORT_BROKEN", rows)
                continue
            penetration = max(0.0, min(1.0, (zone.upper-_field(current, "low"))/(zone.upper-zone.lower)))
            close_location = _close_location(current)
            if penetration > self.config.support_max_penetration_pct or _field(current, "close") < zone.center or close_location < self.config.support_min_close_location:
                self._diagnostic("support_test", context, security_id, symbol, zone, "WEAK_SUPPORT_REJECTION", rows,
                                 penetration=penetration, rejection_strength=close_location, days_since_previous_touch=days_since)
                continue
            stop = zone.lower - zone.center*self.config.support_stop_buffer_pct
            choices.append((zone, days_since, penetration, close_location, stop))
        if not choices:
            return None
        episodes_by_zone = {choice[0]: _touch_episodes(rows, choice[0]) for choice in choices}
        if len(choices) > 1:
            for candidate_zone, days_since, penetration, strength, _ in choices:
                self._diagnostic("support_test", context, security_id, symbol, candidate_zone,
                                 "MULTIPLE_SUPPORT_ZONE_CONFLICT", rows,
                                 touch_count=len(episodes_by_zone[candidate_zone]), days_since_previous_touch=days_since,
                                 penetration=penetration, rejection_strength=strength,
                                 zone_broken=False, zone_state="AMBIGUOUS",
                                 zone_age_sessions=len(rows)-1-candidate_zone.last_pivot_index,
                                 distance_from_current_price=abs(_field(current, "close")-candidate_zone.center))
            return None
        zone, days_since, penetration, strength, stop = choices[0]
        episodes = episodes_by_zone[zone]
        return self._signal("support_test", security_id, symbol, context, zone, stop,
                            touch_count=len(episodes),
                            first_touch_date=str(rows[episodes[0]]["date"])[:10],
                            most_recent_touch_date=str(rows[episodes[-1]]["date"])[:10],
                            zone_age_sessions=len(rows)-1-zone.last_pivot_index,
                            zone_broken=False, zone_state="RETESTED",
                            distance_from_current_price=abs(_field(current, "close")-zone.center),
                            days_since_previous_touch=days_since, penetration_into_zone=penetration,
                            rejection_strength=strength, signal_close=_field(current, "close"),
                            proposed_entry=_field(current, "close"), risk_distance=_field(current, "close")-stop)

    def _advance_breakouts(self, security_id: str, rows: Sequence[Mapping[str, Any]], *,
                           context: StrategyContext, symbol: str, collect_signal: bool) -> EntrySignal | None:
        """Replay unseen bars once, preserving each resistance price at breakout time."""
        previous_date = self._processed_dates.get(security_id)
        dates = [str(row["date"])[:10] for row in rows]
        if previous_date is None:
            start = 0
        else:
            try:
                start = dates.index(previous_date) + 1
            except ValueError:
                self._breakout_watches.pop(security_id, None)
                start = 0
        watches = self._breakout_watches.setdefault(security_id, [])
        signals: list[EntrySignal] = []
        for index in range(start, len(rows)):
            prefix = rows[:index+1]
            row = rows[index]
            is_current = index == len(rows)-1
            # First resolve watches created on an earlier bar; retest cannot be same-bar as breakout.
            for watch in list(watches):
                age = index-watch.breakout_index
                if age > self.config.maximum_retest_sessions:
                    if is_current:
                        self._diagnostic("breakout_retest", context, security_id, symbol, watch.zone,
                                         "RETEST_TIMEOUT", rows, breakout_date=watch.breakout_date,
                                         breakout_level=watch.zone.upper, retest_age=age,
                                         touch_count=watch.touch_count, first_touch_date=watch.first_touch_date,
                                         most_recent_touch_date=watch.most_recent_touch_date,
                                         zone_broken=False, zone_state="EXPIRED",
                                         zone_age_sessions=len(prefix)-1-watch.zone.last_pivot_index,
                                         distance_from_current_price=abs(_field(row, "close")-watch.zone.center))
                    watches.remove(watch)
                    continue
                invalid = _field(row, "close") < watch.zone.lower*(1-self.config.breakout_invalidation_pct)
                if invalid:
                    if is_current:
                        self._diagnostic("breakout_retest", context, security_id, symbol, watch.zone,
                                         "BREAKOUT_INVALIDATED", rows, breakout_date=watch.breakout_date,
                                         breakout_level=watch.zone.upper, retest_age=age,
                                         touch_count=watch.touch_count, first_touch_date=watch.first_touch_date,
                                         most_recent_touch_date=watch.most_recent_touch_date,
                                         zone_broken=True, zone_state="BROKEN",
                                         zone_age_sessions=len(prefix)-1-watch.zone.last_pivot_index,
                                         distance_from_current_price=abs(_field(row, "close")-watch.zone.center))
                    watches.remove(watch)
                    continue
                overlap = _field(row, "low") <= watch.zone.upper*(1+self.config.retest_tolerance_pct)
                holding = (_field(row, "close") >= watch.zone.lower*(1-self.config.breakout_invalidation_pct)
                           and _field(row, "close") >= watch.zone.center
                           and _close_location(row) >= self.config.breakout_min_close_location)
                if age >= 1 and overlap and holding and is_current and collect_signal:
                    stop = min(watch.zone.lower, _field(row, "low")) - watch.zone.center*self.config.breakout_stop_buffer_pct
                    signals.append(self._signal("breakout_retest", security_id, symbol, context, watch.zone, stop,
                                          touch_count=watch.touch_count, breakout_date=watch.breakout_date,
                                          breakout_level=watch.zone.upper, breakout_close=watch.breakout_close,
                                          breakout_volume_ratio=watch.breakout_volume_ratio,
                                          first_touch_date=watch.first_touch_date,
                                          most_recent_touch_date=watch.most_recent_touch_date,
                                          zone_age_sessions=len(prefix)-1-watch.zone.last_pivot_index,
                                          zone_broken=False, zone_state="RETESTED",
                                          distance_from_current_price=abs(_field(row, "close")-watch.zone.center),
                                          retest_date=dates[index], retest_age=age,
                                          retest_low=_field(row, "low"), signal_close=_field(row, "close"),
                                          proposed_entry=_field(row, "close"), risk_distance=_field(row, "close")-stop))
                    watches.remove(watch)
                elif age >= 1 and overlap and is_current and collect_signal:
                    self._diagnostic("breakout_retest", context, security_id, symbol, watch.zone,
                                     "RETEST_REJECTED", rows, breakout_date=watch.breakout_date,
                                     breakout_level=watch.zone.upper, retest_age=age,
                                     retest_low=_field(row, "low"), touch_count=watch.touch_count,
                                     first_touch_date=watch.first_touch_date,
                                     most_recent_touch_date=watch.most_recent_touch_date,
                                     zone_broken=False, zone_state="RETESTING",
                                     zone_age_sessions=len(prefix)-1-watch.zone.last_pivot_index,
                                     distance_from_current_price=abs(_field(row, "close")-watch.zone.center))
            # Determine breakout from resistance known at that bar's close.
            if index >= self.config.swing_left + self.config.swing_right + 1:
                _, resistances = self._zones(prefix, security_id)
                for zone in resistances:
                    if zone.pivot_count < self.config.breakout_minimum_touches:
                        continue
                    episodes = _touch_episodes(prefix, zone)
                    if len(episodes) < self.config.breakout_minimum_touches:
                        continue
                    if not _field(row, "close") > zone.upper*(1+self.config.breakout_distance_pct):
                        continue
                    if index and _field(rows[index-1], "close") > zone.upper*(1+self.config.breakout_distance_pct):
                        continue
                    if any(watch.breakout_index == index and watch.zone == zone for watch in watches):
                        continue
                    volume_ratio = _volume_ratio(prefix, self.config.volume_lookback_sessions)
                    if self.config.volume_ratio_threshold is not None and (
                        volume_ratio is None or volume_ratio < self.config.volume_ratio_threshold
                    ):
                        if is_current:
                            self._diagnostic("breakout_retest", context, security_id, symbol, zone,
                                             "BREAKOUT_VOLUME_REJECTED", rows, breakout_level=zone.upper,
                                             volume_ratio=volume_ratio)
                        continue
                    watches.append(_BreakoutWatch(zone, index, dates[index], _field(row, "close"),
                                                  volume_ratio, len(episodes),
                                                  dates[episodes[0]], dates[episodes[-1]]))
            self._processed_dates[security_id] = dates[index]
        if len(signals) > 1:
            for decision in self.diagnostics:
                if (decision.get("security_id") == security_id
                        and decision.get("signal_date") == context.session_date.isoformat()
                        and decision.get("setup_type") == "breakout_retest"
                        and decision.get("accepted")):
                    decision["accepted"] = False
                    decision["rejection_reason"] = "MULTIPLE_RESISTANCE_ZONE_CONFLICT"
            return None
        return signals[0] if signals else None

    def generate_entries(self, context: StrategyContext) -> Sequence[EntrySignal]:
        if context.next_session_open is None or context.session_date < self.evaluation_start:
            return ()
        candidates: list[EntrySignal] = []
        for security in context.eligible:
            security_id, symbol = security.security_id, security.symbol
            rows = context.daily_history.get(security_id, ())
            if len(rows) < self.config.swing_left + self.config.swing_right + 2 or security_id in context.open_security_ids:
                continue
            signals: list[EntrySignal] = []
            if self.setup in {"support_test", "both"}:
                support_signal = self._support_signal(security_id, symbol, rows, context)
                if support_signal:
                    signals.append(support_signal)
            if self.setup in {"breakout_retest", "both"}:
                breakout_signal = self._advance_breakouts(security_id, rows, context=context,
                                                           symbol=symbol, collect_signal=True)
                if breakout_signal:
                    signals.append(breakout_signal)
            else:
                # Keep breakout state current even when running support-only so diagnostics/repeated runs agree.
                self._advance_breakouts(security_id, rows, context=context, symbol=symbol, collect_signal=False)
            weekly_ok = not self.weekly_filter or _weekly_filter(context, security_id, self.config)
            if signals and not weekly_ok:
                for signal in signals:
                    self._reject_signal_decision(signal, context, "WEEKLY_FILTER_REJECTED",
                                                 weekly_filter_state=False)
                continue
            if len(signals) > 1:
                for signal in signals:
                    self._reject_signal_decision(signal, context, "MULTIPLE_SETUP_CONFLICT",
                                                 weekly_filter_state=weekly_ok)
                continue
            candidates.extend(signals)
        return tuple(candidates)

    def _reject_signal_decision(self, signal: EntrySignal, context: StrategyContext,
                                reason: str, *, weekly_filter_state: bool | None) -> None:
        setup_type = str(signal.metadata["setup_type"])
        for decision in reversed(self.diagnostics):
            if (decision.get("security_id") == signal.security_id
                    and decision.get("signal_date") == context.session_date.isoformat()
                    and decision.get("setup_type") == setup_type
                    and decision.get("accepted")):
                decision["accepted"] = False
                decision["rejection_reason"] = reason
                decision["reason_code"] = reason
                decision["weekly_filter_state"] = weekly_filter_state
                return

    def _signal(self, setup: str, security_id: str, symbol: str, context: StrategyContext,
                zone: PriceZone, stop: float, **details: Any) -> EntrySignal:
        metadata = {"setup_type": setup, "zone": asdict(zone), **details,
                    "weekly_filter_enabled": self.weekly_filter,
                    "weekly_filter_state": _weekly_filter(context, security_id, self.config) if self.weekly_filter else None,
                    "target_r": self.target_r}
        diagnostic = {"setup_type": setup, "security_id": security_id, "symbol": symbol,
                      "signal_date": context.session_date.isoformat(), "zone": asdict(zone),
                      "touch_count": details.get("touch_count"),
                      "first_touch_date": details.get("first_touch_date"),
                      "most_recent_touch_date": details.get("most_recent_touch_date"),
                      "zone_age_sessions": details.get("zone_age_sessions"),
                      "zone_broken": details.get("zone_broken"),
                      "zone_state": details.get("zone_state"),
                      "distance_from_current_price": details.get("distance_from_current_price"),
                      "breakout_level": details.get("breakout_level"),
                      "retest_date": details.get("retest_date"),
                      "volume_ratio": details.get("breakout_volume_ratio"),
                      "weekly_filter_state": metadata["weekly_filter_state"],
                      "signal_close": details.get("signal_close"),
                      "proposed_entry": details.get("proposed_entry"), "entry_price": None,
                      "execution_timestamp": None, "quantity": None, "stop": stop,
                      "risk_distance": details.get("risk_distance"), "target_r": self.target_r,
                      "accepted": True, "rejection_reason": None,
                      "reason_code": setup.upper()}
        self.diagnostics.append(diagnostic)
        return EntrySignal(security_id, symbol, context.timestamp, context.next_session_open,
                           stop, None, (setup.upper(),), metadata, self.version)

    def _diagnostic(self, setup: str, context: StrategyContext, security_id: str, symbol: str,
                    zone: PriceZone, reason: str, rows: Sequence[Mapping[str, Any]], **extra: Any) -> None:
        if context.session_date < self.evaluation_start:
            return
        self.diagnostics.append({"setup_type": setup, "security_id": security_id, "symbol": symbol,
                                 "signal_date": context.session_date.isoformat(), "zone": asdict(zone),
                                 "touch_count": extra.get("touch_count"),
                                 "first_touch_date": extra.get("first_touch_date"),
                                 "most_recent_touch_date": extra.get("most_recent_touch_date"),
                                 "zone_age_sessions": extra.get("zone_age_sessions"),
                                 "zone_broken": extra.get("zone_broken"),
                                 "zone_state": extra.get("zone_state"),
                                 "distance_from_current_price": extra.get("distance_from_current_price"),
                                 "breakout_level": extra.get("breakout_level"),
                                 "retest_date": extra.get("retest_date"),
                                 "volume_ratio": extra.get("volume_ratio"),
                                 "weekly_filter_state": extra.get("weekly_filter_state"),
                                 "signal_close": _field(rows[-1], "close") if rows else None,
                                 "proposed_entry": None, "entry_price": None,
                                 "execution_timestamp": None, "quantity": None,
                                 "stop": None, "risk_distance": None, "target_r": self.target_r,
                                 "accepted": False, "rejection_reason": reason, "reason_code": reason,
                                 **extra})

    def generate_exits(self, context: StrategyContext) -> Sequence[Any]:
        return ()


def _is_broken(rows: Sequence[Mapping[str, Any]], zone: PriceZone, buffer_pct: float) -> bool:
    # A close below the zone lower edge plus buffer decisively breaks support/resistance.
    return any(_field(row, "close") < zone.lower*(1-buffer_pct) for row in rows[zone.first_pivot_index:])


def _volume_ratio(rows: Sequence[Mapping[str, Any]], lookback: int) -> float | None:
    if len(rows) < lookback + 1:
        return None
    baseline = sum(_field(row, "volume") for row in rows[-lookback-1:-1]) / lookback
    return _field(rows[-1], "volume") / baseline if baseline > 0 else None
