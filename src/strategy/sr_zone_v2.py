"""Causal ATR-width support/resistance zones for the sr_zone_v2 experiment.

This detector is deliberately separate from sr_baseline_v1. Zone snapshots are
recomputed from the rows available at each as-of date, so later bars cannot
change earlier decisions or reaction statistics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
from typing import Any, Mapping, Sequence

from src.strategy.sr_baseline import (PriceZone, SwingPoint, _close_location, _field,
                                      _is_broken, _touch_episodes, find_confirmed_swings)


@dataclass(frozen=True)
class ZoneConfig:
    swing_left: int = 3
    swing_right: int = 3
    atr_period: int = 14
    atr_half_width: float = 0.5
    minimum_width_pct: float = 0.0025
    maximum_width_pct: float = 0.01
    cluster_distance_atr: float = 0.75
    minimum_pivots: int = 2
    maximum_age: int = 252
    departure_atr: float = 0.5
    reaction_horizon: int = 5
    break_atr: float = 0.5

    def __post_init__(self) -> None:
        if min(self.swing_left, self.swing_right, self.atr_period, self.minimum_pivots,
               self.maximum_age, self.reaction_horizon) < 1:
            raise ValueError("Zone lookbacks and counts must be positive.")
        if not 0 < self.minimum_width_pct <= self.maximum_width_pct <= 0.1:
            raise ValueError("Zone width guards must satisfy 0 < min <= max <= 10%.")
        if min(self.atr_half_width, self.cluster_distance_atr, self.departure_atr, self.break_atr) <= 0:
            raise ValueError("ATR multipliers must be positive.")


@dataclass(frozen=True)
class TouchEpisode:
    start_index: int
    end_index: int
    separation_from_previous: int | None
    duration: int


@dataclass(frozen=True)
class Zone:
    zone_id: str
    security_id: str
    side: str
    lower_bound: float
    upper_bound: float
    center: float
    creation_date: str
    first_pivot: str
    last_update: str
    pivots: tuple[str, ...]
    touch_episodes: tuple[TouchEpisode, ...]
    reactions: tuple[Mapping[str, float], ...]
    broken: bool
    role_flip: str | None
    freshness: str
    merged_ids: tuple[str, ...] = ()
    original_side: str | None = None
    role_state: str = "ACTIVE"
    acceptance_close_count: int = 0
    reaction_count: int = 0
    mean_reaction_atr: float | None = None
    last_touch_index: int | None = None
    strength: str = "MINOR"

    def as_price_zone(self, first_index: int, last_index: int) -> PriceZone:
        return PriceZone(self.side, self.center, self.lower_bound, self.upper_bound,
                         len(self.pivots), first_index, last_index)


def _f(row: Mapping[str, Any], key: str) -> float:
    return float(row[key])


def _atr(rows: Sequence[Mapping[str, Any]], index: int, period: int) -> float:
    start = max(0, index-period+1)
    ranges = []
    for i in range(start, index+1):
        high, low = _f(rows[i], "high"), _f(rows[i], "low")
        previous = _f(rows[i-1], "close") if i else _f(rows[i], "close")
        ranges.append(max(high-low, abs(high-previous), abs(low-previous)))
    return sum(ranges)/len(ranges)


def _episodes(rows: Sequence[Mapping[str, Any]], lower: float, upper: float,
              departure_atr: float, atr_period: int, offset: int = 0) -> list[TouchEpisode]:
    result: list[TouchEpisode] = []
    active_start: int | None = None
    previous_end: int | None = None
    for i, row in enumerate(rows):
        overlap = _f(row, "low") <= upper and _f(row, "high") >= lower
        if active_start is None and overlap:
            active_start = i
        elif active_start is not None:
            close = _f(row, "close")
            departure = _atr(rows,i,atr_period)*departure_atr
            if close < lower-departure or close > upper+departure:
                end = i-1
                result.append(TouchEpisode(active_start, end,
                                           None if previous_end is None else active_start-previous_end-1,
                                           end-active_start+1))
                previous_end, active_start = end, None
    if active_start is not None:
        end = len(rows)-1
        result.append(TouchEpisode(active_start, end,
                                   None if previous_end is None else active_start-previous_end-1,
                                   end-active_start+1))
    return [TouchEpisode(x.start_index+offset,x.end_index+offset,
                         x.separation_from_previous,x.duration) for x in result]


def detect_zones(rows: Sequence[Mapping[str, Any]], *, security_id: str,
                 config: ZoneConfig | None = None, include_lifecycle: bool = True,
                 confirmed_swings: Sequence[SwingPoint] | None = None,
                 merge_overlaps: bool = True) -> tuple[Zone, ...]:
    """Build deterministically merged zones from confirmed pivots and an as-of prefix."""
    cfg = config or ZoneConfig()
    swings = list(confirmed_swings) if confirmed_swings is not None else find_confirmed_swings(
        rows, left=cfg.swing_left, right=cfg.swing_right)
    clusters: list[list[SwingPoint]] = []
    for swing in sorted(swings, key=lambda x: (x.kind, x.confirmation_index, x.pivot_index)):
        matches = []
        for cluster in clusters:
            if cluster[0].kind != swing.kind:
                continue
            center = sum(item.price for item in cluster)/len(cluster)
            volatility = _atr(rows, swing.confirmation_index, cfg.atr_period)
            if abs(swing.price-center) <= max(volatility*cfg.cluster_distance_atr, center*cfg.minimum_width_pct):
                matches.append(cluster)
        if matches:
            target = min(matches, key=lambda c: (c[0].confirmation_index, c[0].pivot_index))
            target.append(swing)
            # Deterministically absorb all now-connected same-side clusters.
            for extra in matches:
                if extra is not target:
                    target.extend(extra)
                    clusters.remove(extra)
        else:
            clusters.append([swing])

    candidates: list[tuple[Zone, int, int]] = []
    for cluster in clusters:
        if len(cluster) < cfg.minimum_pivots:
            continue
        cluster.sort(key=lambda x: (x.pivot_index, x.confirmation_index))
        first, last = cluster[0], cluster[-1]
        center = sum(x.price for x in cluster)/len(cluster)
        atr = _atr(rows, last.confirmation_index, cfg.atr_period)
        half = min(max(atr*cfg.atr_half_width, center*cfg.minimum_width_pct), center*cfg.maximum_width_pct)
        # Width guards apply to the final zone around the clustered-pivot center,
        # so a wide pivot cloud cannot bypass the ATR percentage cap.
        low = center-half
        high = center+half
        side = cluster[0].kind
        pivots = tuple(str(rows[x.pivot_index]["date"])[:10] for x in cluster)
        stable_seed = f"{security_id}|{side}|{pivots[0]}"
        zone_id = "SZ2-"+hashlib.sha256(stable_seed.encode("utf-8")).hexdigest()[:16]
        candidates.append((Zone(zone_id, security_id, side, low, high, center,
                                str(rows[first.confirmation_index]["date"])[:10], pivots[0],
                                str(rows[last.confirmation_index]["date"])[:10], pivots,
                                (), (), False, None, "FRESH"), first.pivot_index, last.pivot_index))

    # Merge overlapping same-side geometry. The oldest zone keeps its ID; other IDs
    # are recorded as absorbed aliases. Merely adjacent intervals do not merge.
    merged: list[tuple[Zone, int, int]] = []
    if not merge_overlaps:
        merged = candidates.copy()
    for item in ([] if not merge_overlaps else sorted(candidates, key=lambda x: (x[0].side, x[1], x[0].zone_id))):
        zone, first_i, last_i = item
        hits = [j for j, (old, _, _) in enumerate(merged)
                if old.side == zone.side and zone.lower_bound <= old.upper_bound and old.lower_bound <= zone.upper_bound]
        if not hits:
            merged.append(item)
            continue
        j = hits[0]
        old, old_first, old_last = merged[j]
        ids = tuple(sorted(set(old.merged_ids+zone.merged_ids+(zone.zone_id,))))
        pivots = tuple(sorted(set(old.pivots+zone.pivots)))
        merged_center=(old.center*len(old.pivots)+zone.center*len(zone.pivots))/(len(old.pivots)+len(zone.pivots))
        merged_half=merged_center*cfg.maximum_width_pct
        combined = Zone(old.zone_id, old.security_id, old.side,
                        max(min(old.lower_bound, zone.lower_bound),merged_center-merged_half),
                        min(max(old.upper_bound, zone.upper_bound),merged_center+merged_half),
                        merged_center,
                        min(old.creation_date, zone.creation_date), pivots[0], max(old.last_update, zone.last_update),
                        pivots, (), (), False, None, "FRESH", ids)
        merged[j] = (combined, min(old_first, first_i), max(old_last, last_i))
        for other_j in reversed(hits[1:]):
            other, other_first, other_last = merged.pop(other_j)
            merged_center=(combined.center*len(combined.pivots)+other.center*len(other.pivots))/(len(combined.pivots)+len(other.pivots))
            merged_half=merged_center*cfg.maximum_width_pct
            merged[j] = (Zone(combined.zone_id, combined.security_id, combined.side,
                              max(min(combined.lower_bound, other.lower_bound),merged_center-merged_half),
                              min(max(combined.upper_bound, other.upper_bound),merged_center+merged_half),
                              merged_center, min(combined.creation_date, other.creation_date), combined.first_pivot,
                              max(combined.last_update, other.last_update),
                              tuple(sorted(set(combined.pivots+other.pivots))), (), (), False, None, "FRESH",
                              tuple(sorted(set(combined.merged_ids+other.merged_ids+(other.zone_id,))))),
                         min(old_first, other_first), max(old_last, other_last))

    result = []
    for zone, first_i, last_i in merged:
        if len(rows)-1-last_i > cfg.maximum_age:
            continue
        if not include_lifecycle:
            result.append(zone)
            continue
        creation_i = next((i for i,row in enumerate(rows)
                           if str(row["date"])[:10] == zone.creation_date), len(rows)-1)
        episode_rows = rows[creation_i:]
        episodes = _episodes(episode_rows, zone.lower_bound, zone.upper_bound,
                             cfg.departure_atr,cfg.atr_period,creation_i)
        reactions = []
        for episode in episodes:
            end = episode.end_index
            ready = end+cfg.reaction_horizon
            if ready >= len(rows):
                continue
            future = rows[end+1:ready+1]
            move = max((_f(x,"high")-zone.upper_bound for x in future), default=0.0) if zone.side == "support" else max((zone.lower_bound-_f(x,"low") for x in future), default=0.0)
            basis = max(_atr(rows, end, cfg.atr_period), 1e-12)
            reactions.append({"available_index": float(ready), "max_move": move,
                              "move_atr": move/basis, "move_pct": move/max(zone.center,1e-12),
                              "close_away": abs(_f(future[-1],"close")-zone.center)})
        breach = any((_f(row,"close") < zone.lower_bound-_atr(rows,i,cfg.atr_period)*cfg.break_atr)
                     if zone.side == "support" else
                     (_f(row,"close") > zone.upper_bound+_atr(rows,i,cfg.atr_period)*cfg.break_atr)
                     for i,row in enumerate(rows[first_i:], start=first_i))
        flip = None
        if zone.side == "resistance" and breach:
            breach_i = next(i for i,row in enumerate(rows[first_i:], start=first_i)
                            if _f(row,"close") > zone.upper_bound+_atr(rows,i,cfg.atr_period)*cfg.break_atr)
            if any(_f(row,"low") <= zone.upper_bound and _f(row,"close") >= zone.center for row in rows[breach_i+1:]):
                flip = "resistance_to_support_candidate"
        if zone.side == "support" and breach:
            flip = "support_to_resistance_candidate"
        freshness = ("FRESH" if len(episodes)==0 else "FIRST_RETEST" if len(episodes)==1 else
                     "SECOND_RETEST" if len(episodes)==2 else "HEAVILY_TESTED")
        result.append(Zone(zone.zone_id, zone.security_id, zone.side, zone.lower_bound, zone.upper_bound,
                           zone.center, zone.creation_date, zone.first_pivot, zone.last_update, zone.pivots,
                           tuple(episodes), tuple(reactions), breach, flip, freshness, zone.merged_ids))
    return tuple(sorted(result, key=lambda x: (x.side, x.center, x.zone_id)))


def penetration_depth(*, upper: float, lower: float, low: float, atr: float,
                      close_location: float) -> dict[str, float]:
    width = max(upper-lower, 1e-12)
    depth = upper-low
    return {"zone_fraction": depth/width, "atr": depth/max(atr, 1e-12),
            "close_location": close_location}


def zone_quality(zones: Sequence[Zone], *, asof_index: int) -> dict[str, Any]:
    """Small deterministic aggregate for run-level detector quality summaries."""
    widths = [z.upper_bound-z.lower_bound for z in zones]
    return {"zone_count": len(zones), "overlap_pairs": sum(
                1 for i,a in enumerate(zones) for b in zones[i+1:]
                if a.side == b.side and a.lower_bound <= b.upper_bound and b.lower_bound <= a.upper_bound),
            "merged_zones": sum(bool(z.merged_ids) for z in zones),
            "average_width": sum(widths)/len(widths) if widths else 0.0,
            "average_touch_episodes": sum(len(z.touch_episodes) for z in zones)/len(zones) if zones else 0.0,
            "broken_zones": sum(z.broken for z in zones),
            "asof_index": asof_index}


from src.strategy.sr_baseline import SRBaselineConfig, SRBaselineStrategy


class SRZoneV2Strategy(SRBaselineStrategy):
    """V1 entry/exit framework consuming the independent V2 zone detector."""

    name = "sr_zone"
    version = "sr_zone_v2"

    def __init__(self, *, zone_config: ZoneConfig | None = None, **kwargs: Any) -> None:
        base = kwargs.pop("config", None) or SRBaselineConfig()
        # Preserve explicitly configured V1 entry/exit thresholds. The detector
        # geometry and identity are versioned independently.
        super().__init__(config=base, **kwargs)
        self.zone_config = zone_config or ZoneConfig(swing_left=base.swing_left,
                                                      swing_right=base.swing_right,
                                                      minimum_pivots=base.minimum_zone_pivots,
                                                      maximum_age=base.maximum_zone_age_sessions)
        self._v2_zones: dict[tuple[str, str], Zone] = {}
        self.zone_snapshots: dict[str, dict[str, Any]] = {}
        self._zone_cache: dict[str, tuple[int, str, tuple[list[PriceZone], list[PriceZone]]]] = {}
        self._geometry_cache: dict[str, tuple[tuple[tuple[str,int,float], ...], tuple[list[PriceZone],list[PriceZone]]]] = {}
        self._lifecycle_cache: dict[tuple[str,str], tuple[Zone,...]] = {}

    def _zones(self, rows: Sequence[Mapping[str, Any]], security_id: str = "") -> tuple[list[PriceZone], list[PriceZone]]:
        if not rows or not security_id:
            return ([], [])
        cache_key = (len(rows), str(rows[-1]["date"])[:10])
        cached = self._zone_cache.get(security_id)
        if cached is not None and cached[:2] == cache_key:
            return cached[2]
        swings = self._confirmed_swings(security_id, rows)
        signature = tuple((s.kind,s.pivot_index,s.price) for s in swings)
        geometry = self._geometry_cache.get(security_id)
        if geometry is not None and geometry[0] == signature:
            result = geometry[1]
            age_filtered = tuple(zone for side in result for zone in side
                                 if len(rows)-1-zone.last_pivot_index <= self.zone_config.maximum_age)
            supports=[z for z in age_filtered if z.kind=="support"]
            resistances=[z for z in age_filtered if z.kind=="resistance"]
            self._zone_cache[security_id]=(*cache_key,(supports,resistances))
            return supports,resistances
        zones = detect_zones(rows, security_id=security_id, config=self.zone_config,
                             include_lifecycle=False, confirmed_swings=swings) if security_id else ()
        current: dict[tuple[str, str], Zone] = {}
        mapped = []
        index_by_date = {str(row["date"])[:10]: i for i, row in enumerate(rows)}
        for item in zones:
            indices = [index_by_date[pivot] for pivot in item.pivots if pivot in index_by_date]
            price_zone = item.as_price_zone(min(indices), max(indices))
            current[(security_id, item.zone_id)] = item
            mapped.append(price_zone)
        self._v2_zones.update(current)
        self.zone_snapshots[security_id] = {
            "count": len(zones),
            "overlap_pairs": zone_quality(zones, asof_index=len(rows)-1)["overlap_pairs"],
            "merged_zone_count": sum(bool(z.merged_ids) for z in zones),
            "zones": zones,
        }
        result = ([p for p in mapped if p.kind == "support"],
                  [p for p in mapped if p.kind == "resistance"])
        self._zone_cache[security_id] = (*cache_key, result)
        self._geometry_cache[security_id]=(signature,result)
        return result

    def _support_signal(self, security_id: str, symbol: str, rows: Sequence[Mapping[str, Any]], context: Any):
        supports, _ = self._zones(rows[:-1], security_id)
        current = rows[-1]
        candidates = []
        for zone in supports:
            if zone.pivot_count < self.config.minimum_zone_pivots:
                continue
            if not (_field(current,"low") <= zone.upper and _field(current,"high") >= zone.lower):
                continue
            if len(rows)>1 and _field(rows[-2],"low") <= zone.upper and _field(rows[-2],"high") >= zone.lower:
                continue
            if _is_broken(rows[:-1],zone,self.config.zone_break_close_buffer_pct):
                self._diagnostic("support_test",context,security_id,symbol,zone,"SUPPORT_BROKEN",rows,
                                 zone_broken=True,zone_state="BROKEN")
                continue
            episodes=_touch_episodes(rows[:-1],zone)
            prior=episodes[-1] if episodes else None
            gap=len(rows)-1-prior if prior is not None else None
            if prior is None or gap is None or gap < self.config.minimum_touch_separation_sessions:
                continue
            reaction=rows[prior:min(len(rows)-1,prior+6)]
            if not reaction or max(_field(x,"high") for x in reaction) <= zone.upper:
                continue
            close_location=_close_location(current)
            depth=penetration_depth(upper=zone.upper,lower=zone.lower,low=_field(current,"low"),
                                    atr=_atr(rows,len(rows)-1,self.zone_config.atr_period),
                                    close_location=close_location)
            if (_field(current,"close") < zone.lower or depth["zone_fraction"] > self.config.support_max_penetration_pct
                    or _field(current,"close") < zone.center or close_location < self.config.support_min_close_location):
                self._diagnostic("support_test",context,security_id,symbol,zone,"WEAK_SUPPORT_REJECTION",rows,
                                 penetration=depth["zone_fraction"],penetration_atr=depth["atr"],
                                 rejection_strength=close_location)
                continue
            stop=zone.lower-zone.center*self.config.support_stop_buffer_pct
            candidates.append((zone,gap,depth,close_location,stop,episodes))
        if len(candidates)!=1:
            if len(candidates)>1:
                for zone,*_ in candidates:
                    self._diagnostic("support_test",context,security_id,symbol,zone,
                                     "MULTIPLE_SUPPORT_ZONE_CONFLICT",rows,zone_state="AMBIGUOUS")
            return None
        zone,gap,depth,strength,stop,episodes=candidates[0]
        return self._signal("support_test",security_id,symbol,context,zone,stop,
                            touch_count=len(episodes),first_touch_date=str(rows[episodes[0]]["date"])[:10],
                            most_recent_touch_date=str(rows[episodes[-1]]["date"])[:10],
                            zone_age_sessions=len(rows)-1-zone.last_pivot_index,zone_broken=False,
                            zone_state="RETESTED",days_since_previous_touch=gap,
                            penetration_into_zone=depth["zone_fraction"],penetration_atr=depth["atr"],
                            rejection_strength=strength,signal_close=_field(current,"close"),
                            proposed_entry=_field(current,"close"),risk_distance=_field(current,"close")-stop)

    def generate_entries(self, context: Any):
        # A support-only comparison does not need to replay breakout watches;
        # that replay is diagnostic-only and has no effect on its trade ledger.
        if self.setup != "support_test":
            return super().generate_entries(context)
        if context.next_session_open is None or context.session_date < self.evaluation_start:
            return ()
        signals = []
        for security in context.eligible:
            sid, symbol = security.security_id, security.symbol
            rows = context.daily_history.get(sid, ())
            if len(rows) < self.config.swing_left+self.config.swing_right+2 or sid in context.open_security_ids:
                continue
            signal = self._support_signal(sid,symbol,rows,context)
            weekly_ok = not self.weekly_filter or _weekly_filter(context,sid,self.config)
            if signal and not weekly_ok:
                self._reject_signal_decision(signal,context,"WEEKLY_FILTER_REJECTED",weekly_filter_state=False)
            elif signal:
                signals.append(signal)
        return tuple(signals)

    def _signal(self, setup: str, security_id: str, symbol: str, context: Any,
                zone: PriceZone, stop: float, **details: Any):
        from dataclasses import replace
        signal = super()._signal(setup, security_id, symbol, context, zone, stop, **details)
        candidate = next((item for item in self._v2_zones.values()
                          if item.security_id == security_id and item.side == zone.kind
                          and abs(item.center-zone.center) < 1e-9
                          and abs(item.lower_bound-zone.lower) < 1e-9), None)
        if candidate is None:
            return signal
        asof_rows = context.daily_history.get(security_id, ())
        if asof_rows:
            life_key=(security_id,str(asof_rows[-1]["date"])[:10])
            enriched=self._lifecycle_cache.get(life_key)
            if enriched is None:
                enriched=detect_zones(asof_rows,security_id=security_id,config=self.zone_config,
                                      include_lifecycle=True)
                self._lifecycle_cache[life_key]=enriched
            candidate = next((item for item in enriched if item.zone_id==candidate.zone_id),candidate)
        extra = {"zone_id": candidate.zone_id, "freshness": candidate.freshness,
                 "zone_touch_episodes": len(candidate.touch_episodes),
                 "zone_broken_state": candidate.broken, "role_flip": candidate.role_flip,
                 "merged_zone_ids": candidate.merged_ids}
        if setup == "breakout_retest":
            history = context.daily_history.get(security_id, ())
            breakout_day = str(details.get("breakout_date", ""))[:10]
            index = next((i for i,row in enumerate(history) if str(row["date"])[:10] == breakout_day), None)
            if index is not None:
                candle = history[index]
                atr = _atr(history,index,self.zone_config.atr_period)
                span = max(_field(candle,"high")-_field(candle,"low"),1e-12)
                breakout_key=(security_id,breakout_day)
                higher=self._lifecycle_cache.get(breakout_key)
                if higher is None:
                    higher=detect_zones(history[:index+1],security_id=security_id,
                                        config=self.zone_config,include_lifecycle=False)
                    self._lifecycle_cache[breakout_key]=higher
                higher = [z for z in higher if z.side=="resistance" and z.lower_bound>zone.upper]
                nearest = min((z.lower_bound-_field(candle,"close") for z in higher),default=None)
                start_day = str(details.get("first_touch_date",breakout_day))[:10]
                start_idx = next((i for i,row in enumerate(history[:index+1])
                                  if str(row["date"])[:10] == start_day),index)
                extra.update({"breakout_distance_atr": (_field(candle,"close")-zone.upper)/max(atr,1e-12),
                              "breakout_body_range": abs(_field(candle,"close")-_field(candle,"open"))/span,
                              "breakout_close_location": (_field(candle,"close")-_field(candle,"low"))/span,
                              "breakout_consolidation_sessions": sum(
                                  zone.lower <= _field(row,"close") <= zone.upper
                                  for row in history[start_idx:index+1]),
                              "resistance_touch_episodes": details.get("touch_count"),
                              "nearest_higher_resistance_distance": nearest})
        metadata = {**dict(signal.metadata), **extra}
        self.diagnostics[-1].update(extra)
        return replace(signal, metadata=metadata)


__all__ = ["Zone", "ZoneConfig", "TouchEpisode", "detect_zones", "penetration_depth", "zone_quality"]
