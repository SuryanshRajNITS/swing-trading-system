"""Daily-structure-first V2 rules, isolated from the frozen V1 strategy."""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from math import isfinite
from statistics import fmean, median
from typing import Any, Mapping, MutableMapping, Sequence

from src.strategy.sr_zone_v2 import Zone, ZoneConfig, detect_zones
from src.strategy.top_down_swing import _hourly_pattern, _lifecycle_snapshot, _zones, atr, ema, volume_features


@dataclass(frozen=True)
class V2Zone(Zone):
    """V2-only structural evidence; the shared Zone payload remains unchanged."""
    validated: bool = True
    tradeable: bool = True
    hovering_bars: int = 0
    failed_crosses: int = 0
    evidence_score: float = 0.0
    setup_event_id: str | None = None
    validation_reasons: tuple[str, ...] = ()
    tradeability_reasons: tuple[str, ...] = ()
    role_transition_count: int = 0
    two_sided_history: bool = False
    persistence_sessions: int = 0
    dominance_score: float = 0.0
    dominance_reasons: tuple[str, ...] = ()
    role_transitions: tuple[Mapping[str, Any], ...] = ()
    parent_zone_id: str | None = None
    structure_level_kind: str = "PARENT_ZONE"
    nested_boundaries: tuple[Zone, ...] = ()
    boundary_price: float | None = None


def _as_v2_zone(zone: Zone) -> V2Zone:
    return V2Zone(**{item.name:getattr(zone,item.name) for item in fields(Zone)})


def _v2_evidence(zone: Zone, name: str, default: Any) -> Any:
    return getattr(zone,name,default)


def _dominance_score(zone: Zone) -> tuple[float, tuple[str, ...]]:
    """Explainable evidence-only ranking; distance is deliberately not part of the score."""
    episodes=len(zone.touch_episodes)
    reactions=len(zone.reactions)
    reaction=max(float(zone.mean_reaction_atr or 0.0),0.0)
    hovering=int(_v2_evidence(zone,"hovering_bars",0))
    failed=int(_v2_evidence(zone,"failed_crosses",0))
    flips=int(_v2_evidence(zone,"role_transition_count",0))
    persistent=int(_v2_evidence(zone,"persistence_sessions",0))
    merged=len(zone.merged_ids)
    score=(3.0*min(episodes,20)+2.0*min(reactions,20)+min(reaction,3.0)+
           0.2*min(hovering,50)+0.5*min(failed,20)+4.0*min(flips,5)+
           0.04*min(persistent,250)+1.5*int(_v2_evidence(zone,"two_sided_history",False))+merged)
    reasons=[]
    if episodes: reasons.append(f"{episodes}_separated_interaction_episodes")
    if reactions: reasons.append(f"{reactions}_measured_reactions")
    if hovering: reasons.append(f"{hovering}_hovering_sessions")
    if failed: reasons.append(f"{failed}_failed_cross_episodes")
    if flips: reasons.append(f"{flips}_causal_role_transitions")
    if persistent: reasons.append(f"{persistent}_sessions_in_memory")
    if _v2_evidence(zone,"two_sided_history",False): reasons.append("support_resistance_role_history")
    if merged: reasons.append(f"{merged}_merged_source_zones")
    return score,tuple(reasons)


def _priority_key(zone: Zone, price: float) -> tuple[float, int, float, str]:
    strength_rank={"MAJOR":2,"MEANINGFUL":1,"MINOR":0}.get(zone.strength,0)
    score=float(_v2_evidence(zone,"dominance_score",0.0))
    if score<=0:
        score=float(_v2_evidence(zone,"evidence_score",0.0))
    return (-score,
            -strength_rank,abs(float(zone.center)-price),zone.zone_id)


def _v2_current_role(zone: Zone, rows: Sequence[Mapping[str, Any]], zone_cfg: ZoneConfig,
                    acceptance_closes: int = 2, break_buffer_atr: float = 0.10,
                    transitions: list[dict[str, Any]] | None = None) -> tuple[str, str | None, bool]:
    """Replay one zone's role transitions using only its as-of price prefix."""
    original=zone.original_side or zone.side
    role="ACTIVE"; flip=None; broken=False
    beyond=0; accepted_at=None
    start=next((i for i,row in enumerate(rows) if str(row["date"])[:10]>=zone.first_pivot),0)
    for i in range(start,len(rows)):
        prior_role=role
        row=rows[i]
        previous=float(rows[i-1]["close"]) if i else float(row["close"])
        close=float(row["close"]); high=float(row["high"]); low=float(row["low"])
        start_atr=max(0,i-zone_cfg.atr_period+1)
        true_ranges=[]
        for j in range(start_atr,i+1):
            previous_close=float(rows[j-1]["close"]) if j else float(rows[j]["close"])
            high_j=float(rows[j]["high"]); low_j=float(rows[j]["low"])
            true_ranges.append(max(high_j-low_j,abs(high_j-previous_close),abs(low_j-previous_close)))
        volatility=max(fmean(true_ranges),1e-12)
        buffer=break_buffer_atr*volatility
        if original=="support":
            if role in {"ACTIVE","SUPPORT_CONFIRMED"}:
                broken_close=close < zone.lower_bound-buffer
                beyond=beyond+1 if broken_close else 0
                if beyond>=acceptance_closes:
                    role="RESISTANCE_CANDIDATE"; flip="ACCEPTANCE_BELOW_SUPPORT"
                    broken=True; beyond=0; accepted_at=None
            elif role=="RESISTANCE_CANDIDATE":
                if accepted_at is not None and low<=zone.upper_bound and close>=zone.upper_bound:
                    role="SUPPORT_CONFIRMED"; flip="support_reclaimed"; accepted_at=None
                    continue
                accepted=close > zone.upper_bound+buffer
                beyond=beyond+1 if accepted else 0
                if beyond>=acceptance_closes:
                    accepted_at=i; beyond=0
        else:
            if role in {"ACTIVE","RESISTANCE_CONFIRMED"}:
                broken_close=close > zone.upper_bound+buffer
                beyond=beyond+1 if broken_close else 0
                if beyond>=acceptance_closes:
                    role="BROKEN_RESISTANCE_WAITING_HOLD"; flip="ACCEPTANCE_ABOVE_RESISTANCE"
                    broken=True; beyond=0; accepted_at=i
            elif role in {"SUPPORT_CANDIDATE", "BROKEN_RESISTANCE_WAITING_HOLD"}:
                if accepted_at is not None and low<=zone.upper_bound and close>=zone.upper_bound:
                    role="SUPPORT_CONFIRMED"; flip="resistance_to_support_candidate"; accepted_at=None
                    continue
                accepted=close > zone.upper_bound+buffer
                beyond=beyond+1 if accepted else 0
                if beyond>=acceptance_closes:
                    accepted_at=i; beyond=0
        # A confirmed role can be invalidated by a new accepted break.
        if role=="SUPPORT_CONFIRMED" and close < zone.lower_bound-buffer:
            beyond+=1
            if beyond>=acceptance_closes:
                role="RESISTANCE_CANDIDATE"; flip="SUPPORT_REBROKEN"; broken=True
                beyond=0; accepted_at=None
        elif role=="RESISTANCE_CONFIRMED" and close > zone.upper_bound+buffer:
            beyond+=1
            if beyond>=acceptance_closes:
                role="SUPPORT_CANDIDATE"; flip="RESISTANCE_REBROKEN"; broken=True
                beyond=0; accepted_at=None
        # Evidence for a failed single-bar crossing is a validation feature.
        if original=="resistance" and high>zone.upper_bound and close<=zone.upper_bound and previous<=zone.upper_bound:
            flip=flip or "WICK_ABOVE_RESISTANCE_THEN_REJECTED"
        if original=="support" and low<zone.lower_bound and close>=zone.lower_bound and previous>=zone.lower_bound:
            flip=flip or "WICK_BELOW_SUPPORT_THEN_RECLAIMED"
        if transitions is not None and role!=prior_role:
            transitions.append({"date":str(row["date"])[:10],"from_role":prior_role,
                "to_role":role,"reason_code":flip})
    return role,flip,broken


@dataclass(frozen=True)
class V2Config:
    recent_daily_structure_min_sessions: int = 45
    recent_daily_structure_max_sessions: int = 60
    structural_memory_sessions: int = 250
    support_watch_distance_atr: float = 1.0
    stop_buffer_atr: float = 0.25
    target_front_run_buffer_atr: float = 0.10
    minimum_structural_rr: float = 1.5
    friction_distance_atr: float = 0.5
    clearance_buffer_atr: float = 0.10
    confirmation_expiry_sessions: int = 10
    consolidation_bars: int = 6
    breakout_acceptance_closes: int = 2
    ema_rejection_lookback_bars: int = 20
    ema_min_rejections: int = 3
    volume_lookback: int = 20
    meaningful_touch_episodes: int = 2
    major_touch_episodes: int = 4
    meaningful_reaction_atr: float = 0.5
    major_reaction_atr: float = 1.0
    meaningful_recency_bars: int = 130
    major_recency_bars: int = 65
    zone_break_buffer_atr: float = 0.10
    zone_acceptance_closes: int = 2
    direct_breakout_min_body_fraction: float = 0.50
    direct_breakout_min_close_location: float = 0.75
    direct_breakout_min_range_ratio: float = 1.0
    daily_zone: ZoneConfig = ZoneConfig(swing_left=3, swing_right=3, minimum_pivots=2, maximum_age=252)
    weekly_zone: ZoneConfig = ZoneConfig(swing_left=2, swing_right=2, minimum_pivots=2, maximum_age=104,
                                         minimum_width_pct=.003, maximum_width_pct=.02)
    hourly_zone: ZoneConfig = ZoneConfig(swing_left=3, swing_right=3, minimum_pivots=2, maximum_age=130)

    def __post_init__(self) -> None:
        if min(self.support_watch_distance_atr, self.stop_buffer_atr, self.target_front_run_buffer_atr,
               self.minimum_structural_rr, self.friction_distance_atr, self.clearance_buffer_atr,
               self.zone_break_buffer_atr) < 0:
            raise ValueError("V2 distances and RR thresholds must be nonnegative.")
        if min(self.confirmation_expiry_sessions, self.consolidation_bars, self.breakout_acceptance_closes,
               self.zone_acceptance_closes,
               self.volume_lookback, self.ema_min_rejections,
               self.recent_daily_structure_min_sessions,self.recent_daily_structure_max_sessions) < 1:
            raise ValueError("V2 lookbacks and confirmation windows must be positive.")
        if self.recent_daily_structure_max_sessions < self.recent_daily_structure_min_sessions:
            raise ValueError("Adaptive daily structure max window must be >= minimum window.")
        if self.structural_memory_sessions < self.recent_daily_structure_min_sessions:
            raise ValueError("Structural memory must cover at least the local daily context window.")
        if not 0 < self.direct_breakout_min_body_fraction <= 1 or not 0 < self.direct_breakout_min_close_location <= 1:
            raise ValueError("Direct-breakout candle-quality fractions must be in (0, 1].")
        if self.direct_breakout_min_range_ratio < 1:
            raise ValueError("Direct-breakout range expansion ratio must be at least 1.")


def enriched_zones(rows: Sequence[Mapping[str, Any]], security_id: str, zone_cfg: ZoneConfig,
                   cfg: V2Config) -> tuple[Zone, ...]:
    raw = tuple(_as_v2_zone(zone) for zone in _zones(rows, security_id=security_id, config=zone_cfg))
    # The V1 lifecycle helper is a shared causal zone-evidence utility; V2 uses
    # its own thresholds and all rows supplied are strictly as-of.
    lifecycle_cfg = type("LifecycleConfig", (), {
        "zone_break_buffer_atr": cfg.zone_break_buffer_atr,
        "zone_acceptance_closes": cfg.zone_acceptance_closes,
        "meaningful_pivots": 2, "meaningful_touch_episodes": cfg.meaningful_touch_episodes,
        "major_touch_episodes": cfg.major_touch_episodes,
        "meaningful_reaction_atr": cfg.meaningful_reaction_atr,
        "major_reaction_atr": cfg.major_reaction_atr,
        # V2 levels live in a bounded structural memory. Calendar/session age
        # limits the source history, but does not demote evidence within it.
        "meaningful_recency_bars": len(rows) + 1,
        "major_recency_bars": len(rows) + 1,
    })()
    def measure(source: Sequence[Zone]) -> tuple[V2Zone, ...]:
      lifecycle=_lifecycle_snapshot(source, rows, zone_cfg, lifecycle_cfg)
      measured=[]
      for zone in lifecycle:
        role_transitions=[]
        role_state,role_flip,currently_broken=_v2_current_role(zone,rows,zone_cfg,
            cfg.zone_acceptance_closes,cfg.zone_break_buffer_atr,role_transitions)
        effective_side=("support" if role_state in {"SUPPORT_CONFIRMED","SUPPORT_CANDIDATE"} else
                        "resistance" if role_state in {"RESISTANCE_CONFIRMED","RESISTANCE_CANDIDATE"} else
                        zone.original_side or zone.side)
        zone=replace(zone,side=effective_side,role_state=role_state,role_flip=role_flip,
                     broken=currently_broken,role_transition_count=len(role_transitions),
                     role_transitions=tuple(role_transitions),
                     two_sided_history=any(item["to_role"] in
                         ({"RESISTANCE_CANDIDATE","RESISTANCE_CONFIRMED"} if
                          (zone.original_side or zone.side)=="support" else
                          {"SUPPORT_CANDIDATE","SUPPORT_CONFIRMED"}) for item in role_transitions),
                     persistence_sessions=max(0,len(rows)-1-next((i for i,r in enumerate(rows)
                         if str(r["date"])[:10]>=zone.first_pivot),0)))
        reactions=[]
        for episode in zone.touch_episodes:
            end=episode.end_index; available=end+zone_cfg.reaction_horizon
            if available>=len(rows):
                continue
            response=rows[end+1:available+1]
            move=(max(float(r["high"]) for r in response)-zone.upper_bound if zone.side=="support"
                  else zone.lower_bound-min(float(r["low"]) for r in response))
            if move>0:
                scale=max(atr(rows[:end+1],zone_cfg.atr_period),1e-12)
                reactions.append({"available_index":available,"max_move":move,"move_atr":move/scale})
        hovering_bars=sum(ep.duration for ep in zone.touch_episodes)
        failed_crosses=0; previous_failed_cross=False
        for row in rows:
            if zone.original_side == "resistance":
                failed_cross=(float(row["high"]) > zone.upper_bound and
                              float(row["close"]) <= zone.upper_bound)
            else:
                failed_cross=(float(row["low"]) < zone.lower_bound and
                              float(row["close"]) >= zone.lower_bound)
            failed_crosses += int(failed_cross and not previous_failed_cross)
            previous_failed_cross=failed_cross
        mean_reaction=max((float(x["move_atr"]) for x in reactions),default=0.0)
        # A first meaningful reaction validates a level but is not executable.
        # Hovering, failed crossing attempts, and an accepted role flip are
        # separate observable validation evidence, not guessed chart labels.
        validation_reasons=[]
        if mean_reaction >= cfg.meaningful_reaction_atr:
            validation_reasons.append("MEANINGFUL_POST_TOUCH_REACTION")
        if hovering_bars >= zone_cfg.reaction_horizon:
            validation_reasons.append("PROLONGED_HOVER_OR_CONSOLIDATION")
        if failed_crosses > 0:
            validation_reasons.append("FAILED_CROSS_ATTEMPT")
        if zone.role_state in {"SUPPORT_CONFIRMED","RESISTANCE_CONFIRMED"}:
            validation_reasons.append("ACCEPTED_ROLE_FLIP")
        validated=bool(validation_reasons)
        revisits=len(zone.touch_episodes)
        # A confirmed role flip itself includes the later acceptance/hold
        # revisit; ordinary first reactions still need a separate episode.
        tradeability_reasons=[]
        if revisits >= 2:
            tradeability_reasons.append("SEPARATE_VALIDATED_REVISIT")
        if zone.role_state in {"SUPPORT_CONFIRMED","RESISTANCE_CONFIRMED"}:
            tradeability_reasons.append("ACCEPTED_ROLE_FLIP_AND_HOLD")
        tradeable=validated and bool(tradeability_reasons)
        zone=replace(zone,hovering_bars=hovering_bars,failed_crosses=failed_crosses)
        dominance_score,dominance_reasons=_dominance_score(zone)
        strength="MINOR"
        if validated and len(zone.pivots) >= zone_cfg.minimum_pivots:
            strength="MEANINGFUL"
        if revisits >= cfg.major_touch_episodes and mean_reaction >= cfg.major_reaction_atr:
            strength="MAJOR"
        evidence_score=(len(zone.pivots)+revisits+len(reactions)+hovering_bars/max(zone_cfg.reaction_horizon,1)+
                        failed_crosses+int(zone.role_state in {"SUPPORT_CONFIRMED","RESISTANCE_CONFIRMED"}))
        event_date=(str(rows[zone.touch_episodes[-1].start_index]["date"])[:10]
                    if zone.touch_episodes else None)
        measured.append(replace(zone,reactions=tuple(reactions),validated=validated,
            tradeable=tradeable,hovering_bars=hovering_bars,failed_crosses=failed_crosses,
            evidence_score=evidence_score,strength=strength,
            setup_event_id=f"{zone.zone_id}:{event_date}" if event_date else None,
            validation_reasons=tuple(validation_reasons),tradeability_reasons=tuple(tradeability_reasons),
            dominance_score=dominance_score,dominance_reasons=dominance_reasons))
      return tuple(measured)

    parents=measure(raw)
    # Keep the merged persistent band as the macro map, while retaining only
    # independently validated component bands as nested breakout boundaries.
    components=tuple(_as_v2_zone(z) for z in detect_zones(
        rows[-(zone_cfg.maximum_age+zone_cfg.atr_period+max(zone_cfg.swing_left,zone_cfg.swing_right)):],
        security_id=security_id,config=zone_cfg,include_lifecycle=False,merge_overlaps=False))
    measured_components={z.zone_id:z for z in measure(components)}
    children=[]
    for parent in parents:
        aliases=set(parent.merged_ids)
        aliases.add(parent.zone_id)
        for component_id in sorted(aliases-{parent.zone_id}):
            child=measured_components.get(component_id)
            if child is None or not child.validated or child.strength not in {"MEANINGFUL","MAJOR"}:
                continue
            parent_width=parent.upper_bound-parent.lower_bound
            child_width=child.upper_bound-child.lower_bound
            if child_width>=parent_width*.95:
                continue
            if child.upper_bound<parent.lower_bound or child.lower_bound>parent.upper_bound:
                continue
            children.append(replace(child,parent_zone_id=parent.zone_id,
                                    structure_level_kind="INNER_BOUNDARY"))
        if parent.side=="resistance" and parent.original_side=="resistance":
            # A persistent broad resistance can contain a separately tested
            # lower edge. Model the edge as a price band one parent-width below
            # the parent, then retain it only when its own causal history has
            # repeated interaction evidence. This does not manufacture levels
            # from arbitrary pivots and preserves the macro parent unchanged.
            width=parent.upper_bound-parent.lower_bound
            edge=replace(parent,zone_id=f"{parent.zone_id}-LOWER-EDGE",
                lower_bound=parent.lower_bound-width,upper_bound=parent.lower_bound,
                center=parent.lower_bound-width/2,merged_ids=(),touch_episodes=(),reactions=(),
                role_state="ACTIVE",role_flip=None,broken=False,role_transition_count=0,
                role_transitions=(),nested_boundaries=(),structure_level_kind="INNER_BOUNDARY",
                parent_zone_id=parent.zone_id,boundary_price=parent.lower_bound)
            measured_edge=measure((edge,))[0]
            repeated_evidence=(len(measured_edge.touch_episodes)>=2 or measured_edge.failed_crosses>=2
                or measured_edge.role_transition_count>0)
            if measured_edge.validated and repeated_evidence:
                children.append(replace(measured_edge,parent_zone_id=parent.zone_id,
                    structure_level_kind="INNER_BOUNDARY",boundary_price=parent.lower_bound))
    parents=tuple(replace(parent,nested_boundaries=tuple(child for child in children
        if child.parent_zone_id==parent.zone_id)) for parent in parents)
    return tuple((*parents,*children))


def zone_row(zone: Zone, rows: Sequence[Mapping[str, Any]], *, index_space_matches: bool = True) -> dict[str, Any]:
    last = zone.last_touch_index
    matching_index_space=index_space_matches and (last is None or 0<=last<len(rows))
    return {"zone_id": zone.zone_id, "security_id": zone.security_id,
        "side": zone.side, "lower_bound": zone.lower_bound, "upper_bound": zone.upper_bound,
        "center": zone.center, "touch_episodes": len(zone.touch_episodes),
        "reaction_count": zone.reaction_count, "mean_reaction_atr": zone.mean_reaction_atr,
        "maximum_reaction_atr": max((float(x.get("move_atr", 0)) for x in zone.reactions), default=None),
        "last_touch_date": str(rows[last]["date"])[:10] if last is not None and matching_index_space else None,
        # Zone indices are relative to the sequence used to build the Zone.
        # Callers sometimes render an older-memory zone against the short
        # recent window; report an unavailable age instead of a negative or
        # misleading value in that case. Lifecycle/strength uses its own
        # correctly aligned source rows and is unaffected by this diagnostic.
        "age": len(rows)-1-last if last is not None and matching_index_space else (len(rows) if last is None and index_space_matches else None),
        "age_reference_mismatch": not matching_index_space,
        "role_state": zone.role_state, "role_flip_history": zone.role_flip,
        "strength": zone.strength, "pivots": zone.pivots, "merged_ids": zone.merged_ids,
        "broken": zone.broken,"validated":_v2_evidence(zone,"validated",True),
        "tradeable":_v2_evidence(zone,"tradeable",True),
        "hovering_bars":_v2_evidence(zone,"hovering_bars",0),
        "failed_crosses":_v2_evidence(zone,"failed_crosses",0),
        "evidence_score":_v2_evidence(zone,"evidence_score",0.0),
        "validation_reasons":_v2_evidence(zone,"validation_reasons",()),
        "tradeability_reasons":_v2_evidence(zone,"tradeability_reasons",()),
        "role_transition_count":_v2_evidence(zone,"role_transition_count",0),
        "two_sided_history":_v2_evidence(zone,"two_sided_history",False),
        "persistence_sessions":_v2_evidence(zone,"persistence_sessions",0),
        "dominance_score":_v2_evidence(zone,"dominance_score",0.0),
        "dominance_reasons":_v2_evidence(zone,"dominance_reasons",()),
        "role_transitions":_v2_evidence(zone,"role_transitions",()),
        "parent_zone_id":_v2_evidence(zone,"parent_zone_id",None),
        "structure_level_kind":_v2_evidence(zone,"structure_level_kind","PARENT_ZONE"),
        "boundary_price":_v2_evidence(zone,"boundary_price",None),
        "nested_boundary_ids":[z.zone_id for z in _v2_evidence(zone,"nested_boundaries",())]}


def _priority_rows(zones: Sequence[Zone], price: float, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    ordered=sorted(zones,key=lambda z:_priority_key(z,price))
    result=[]
    for index,zone in enumerate(ordered):
        result.append({**zone_row(zone,rows),"priority_rank":index+1,
            "priority_class":"DOMINANT_STRUCTURAL_ZONE" if index==0 else "LOCAL_SECONDARY_ZONE",
            "priority_reason_codes":_v2_evidence(zone,"dominance_reasons",())})
    return result


def _support_deterioration(zone: Zone, rows: Sequence[Mapping[str, Any]], ema50: float | None,
                           ema200: float | None) -> tuple[bool, tuple[str, ...]]:
    """Require local support deterioration plus EMA loss; EMAs never gate unrelated setups."""
    recent=rows[-3:]
    closes_below_upper=sum(float(row["close"])<zone.upper_bound for row in recent)
    if closes_below_upper<2 or ema50 is None:
        return False,()
    current=float(rows[-1]["close"])
    if current>=ema50:
        return False,()
    reasons=["TWO_OF_THREE_CLOSES_BELOW_SUPPORT_UPPER"]
    reasons.append("CLOSE_BELOW_EMA50")
    if ema200 is not None:
        if current>=ema200:
            return False,()
        reasons.append("CLOSE_BELOW_EMA200")
    else:
        reasons.append("EMA200_UNAVAILABLE_EMA50_USED_WITH_LOCAL_SUPPORT_LOSS")
    return True,tuple(reasons)


def recent_daily_window(rows: Sequence[Mapping[str, Any]], sod_date: str, lookback: int = 30) -> list[Mapping[str, Any]]:
    """Return at most N completed daily bars strictly before the SOD date."""
    if lookback < 1:
        raise ValueError("recent daily lookback must be positive")
    return [r for r in rows if str(r["date"])[:10] < sod_date][-lookback:]


def is_h1_watchlisted(location: Mapping[str, Any]) -> bool:
    """The SOD gate: only a current recent-support location activates H1."""
    return bool(location.get("interesting_today")) and location.get("state") in {
        "NEAR_RECENT_DAILY_SUPPORT", "INSIDE_RECENT_DAILY_SUPPORT"}


def adaptive_recent_structure(rows: Sequence[Mapping[str, Any]], *, sod_date: str, security_id: str,
                              cfg: V2Config) -> dict[str, Any]:
    """Return a persistent long-history map and a shorter local context window."""
    completed=[r for r in rows if str(r["date"])[:10]<sod_date]
    memory=completed[-cfg.structural_memory_sessions:]
    zones=enriched_zones(memory,security_id,cfg.daily_zone,cfg) if memory else ()
    maximum=min(cfg.recent_daily_structure_max_sessions,len(completed))
    minimum=min(cfg.recent_daily_structure_min_sessions,maximum)
    local=completed[-maximum:] if maximum else []
    context_sufficient=False; used=maximum
    if maximum:
        for count in range(minimum,maximum+1):
            candidate=completed[-count:]
            price=float(candidate[-1]["close"]); volatility=max(atr(candidate,14),1e-12)
            established=[z for z in zones if _v2_evidence(z,"validated",True) and z.strength in {"MEANINGFUL","MAJOR"}]
            local_support=any(z.side=="support" and z.lower_bound<=price and
                z.upper_bound>=price-cfg.support_watch_distance_atr*volatility for z in established)
            local_resistance=any(z.side=="resistance" and z.lower_bound>price for z in established)
            relevant_flip=any(z.role_state in {"SUPPORT_CONFIRMED","RESISTANCE_CONFIRMED"} and
                z.lower_bound-volatility<=price<=z.upper_bound+volatility for z in established)
            if (local_support and local_resistance) or relevant_flip:
                local=candidate; used=count; context_sufficient=True; break
    return {"rows":local,"zones":zones,"structure_rows":memory,"sessions_used":len(local),
            "extension_reason":("INITIAL_LOCAL_CONTEXT_SUFFICIENT" if used==minimum else "EXTENDED_LOCAL_CONTEXT")
                if context_sufficient else "MAX_LOCAL_CONTEXT_REACHED_OR_INSUFFICIENT_HISTORY",
            "structure_sufficient":bool(memory and any(_v2_evidence(z,"validated",True) for z in zones)),
            "local_context_sufficient":context_sufficient,"memory_sessions":len(memory)}


def daily_location(rows: Sequence[Mapping[str, Any]], zones: Sequence[Zone], cfg: V2Config,
                   older_zones: Sequence[Zone] = (), *,
                   structure_rows: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    if not rows:
        return {"state": "NO_MEANINGFUL_DAILY_STRUCTURE", "reason_codes": ["NO_DAILY_DATA"]}
    structure_rows=structure_rows or rows
    def audit_row(zone: Zone) -> dict[str, Any]:
        return zone_row(zone,structure_rows,index_space_matches=zone in zones or zone in older_zones)
    bar = rows[-1]; close = float(bar["close"]); low = float(bar["low"]); vol = max(atr(rows, 14), 1e-12)
    ema50=ema(structure_rows,50); ema200=ema(structure_rows,200)
    meaningful=[z for z in (*zones,*older_zones) if z.strength in {"MEANINGFUL","MAJOR"} and
                _v2_evidence(z,"validated",True)]
    relevant=[z for z in meaningful if max(z.lower_bound-close,close-z.upper_bound,0.0)<=2.5*vol]
    priority_rows=_priority_rows(relevant,close,structure_rows)
    priority_context={"dominant_zone":priority_rows[0] if priority_rows else None,
        "secondary_zones":priority_rows[1:],"ranking_method":"STRUCTURAL_EVIDENCE_THEN_STRENGTH_THEN_DISTANCE",
        "distance_is_tiebreaker_only":True}
    direct_resistance_watches=[z for z in meaningful if z.side=="resistance" and z.original_side=="resistance" and
        z.role_state in {"ACTIVE","RESISTANCE_CANDIDATE"} and close<=z.upper_bound and
        z.lower_bound-close<=2.5*vol and
        (int(_v2_evidence(z,"hovering_bars",0))>=cfg.daily_zone.reaction_horizon or
         (len(z.touch_episodes)>=cfg.meaningful_touch_episodes and int(_v2_evidence(z,"failed_crosses",0))>0))]
    broken_supports=[z for z in meaningful if z.original_side=="support" and
        (z.role_state=="RESISTANCE_CANDIDATE" or (z.broken and z.side!="support")) and
        max(z.lower_bound-close,close-z.upper_bound,0.0)<=cfg.support_watch_distance_atr*vol]
    supports=[z for z in meaningful if z.side=="support" and _v2_evidence(z,"tradeable",True) and
        (not z.broken or z.role_state=="SUPPORT_CONFIRMED") and close>=z.lower_bound and
        z.upper_bound>=close-cfg.support_watch_distance_atr*vol and
        any(float(r["close"])>z.upper_bound for r in rows[max(0,len(rows)-11):-1])]
    relevant_broken=[z for z in broken_supports if z.lower_bound-close<=cfg.support_watch_distance_atr*vol]
    # A breached local zone cannot automatically override a stronger active
    # structural support below it. Rank both by recorded evidence first.
    role_candidates=[*supports,*relevant_broken]
    role_candidates.sort(key=lambda z:_priority_key(z,close))
    support=(role_candidates[0] if role_candidates and role_candidates[0] not in relevant_broken else None)
    broken=(role_candidates[0] if role_candidates and role_candidates[0] in relevant_broken else None)
    pending_resistance_reclaims=[z for z in meaningful if z.original_side=="resistance" and
        z.role_state=="RESISTANCE_CANDIDATE" and close>=z.lower_bound and
        max(z.lower_bound-close,close-z.upper_bound,0.0)<=2.5*vol]
    if pending_resistance_reclaims:
        pending=min(pending_resistance_reclaims,key=lambda z:_priority_key(z,close))
        if support is None or _priority_key(pending,close)<_priority_key(support,close):
            return {"state":"WAIT_FOR_RECLAIM",
                "reason_codes":["DOMINANT_BROKEN_RESISTANCE","WAIT_FOR_RECLAIM"],
                "interesting_today":False,"daily_zones":[audit_row(z) for z in zones],
                "daily_atr":vol,"broken_zone":pending,
                "activation_state":"BROKEN_SUPPORT_RECLAIM_WATCH",
                "current_reference_price":close,"structure_provenance":"PERSISTENT_DAILY_STRUCTURE",
                "direct_breakout_candidates":direct_resistance_watches,
                **priority_context}
    if broken is not None:
        return {"state":"WAIT_FOR_RECLAIM","reason_codes":["SUPPORT_DAMAGED","BROKEN_SUPPORT","WAIT_FOR_RECLAIM"],
            "interesting_today":False,"daily_zones":[audit_row(z) for z in zones],
            "daily_atr":vol,"broken_zone":broken,"activation_state":"BROKEN_SUPPORT_RECLAIM_WATCH",
            "current_reference_price":close,"structure_provenance":"PERSISTENT_DAILY_STRUCTURE",
            "direct_breakout_candidates":direct_resistance_watches,
            **priority_context}
    previous_close=float(rows[-2]["close"]) if len(rows)>1 else close
    if support is None:
        just_broken=[z for z in meaningful if z.original_side=="resistance" and
            z.role_state in {"BROKEN_RESISTANCE_WAITING_HOLD","SUPPORT_CANDIDATE","SUPPORT_CONFIRMED"} and
            previous_close<=z.upper_bound<close]
        if just_broken:
            broken=min(just_broken,key=lambda z:_priority_key(z,close))
            hover=sum(1 for r in rows[-min(cfg.recent_daily_structure_max_sessions,len(rows)):-1]
                if float(r["high"])>=broken.lower_bound-.5*vol and float(r["low"])<=broken.upper_bound+.5*vol)
            return {"state":"BREAKOUT_ABOVE_RECENT_DAILY_RESISTANCE",
                "reason_codes":["BREAKOUT_ABOVE_RECENT_DAILY_RESISTANCE"],"interesting_today":False,
                "daily_atr":vol,"current_reference_price":close,"nearest_resistance":broken,
                "resistance_hovering_bars":hover,"resistance_provenance":"PERSISTENT_DAILY_STRUCTURE",
                "nearest_resistance_row":audit_row(broken),"daily_zones":[audit_row(z) for z in zones],
                "structure_provenance":"PERSISTENT_DAILY_STRUCTURE",**priority_context}
        flip_candidates=[z for z in meaningful if z.original_side=="resistance" and
            z.role_state=="SUPPORT_CONFIRMED" and
            max(z.lower_bound-close,close-z.upper_bound,0.0)<=cfg.support_watch_distance_atr*vol]
        if flip_candidates:
            candidate=min(flip_candidates,key=lambda z:_priority_key(z,close))
            return {"state":"RETESTING_BROKEN_RESISTANCE","reason_codes":["RETESTING_BROKEN_RESISTANCE"],
                "interesting_today":False,"daily_atr":vol,"current_reference_price":close,
                "nearest_resistance":candidate,"resistance":candidate,
                "nearest_resistance_row":audit_row(candidate),
                "resistance_provenance":"PERSISTENT_DAILY_STRUCTURE",
                "daily_zones":[audit_row(z) for z in zones],
                "structure_provenance":"PERSISTENT_DAILY_STRUCTURE",**priority_context}
    resistances=sorted((z for z in meaningful if z.side=="resistance" and
        z.role_state not in {"SUPPORT_CANDIDATE","SUPPORT_CONFIRMED"} and
        close<=z.upper_bound and z.lower_bound-close<=cfg.support_watch_distance_atr*vol),
        key=lambda z:_priority_key(z,close))
    resistance=resistances[0] if resistances else None
    if support is None:
        if resistance and resistance.lower_bound-close<=cfg.support_watch_distance_atr*vol:
            hover_limit=min(cfg.recent_daily_structure_max_sessions,len(rows))
            local=rows[-hover_limit:]
            hover=sum(1 for r in local if float(r["high"])>=resistance.lower_bound-.5*vol and
                      float(r["low"])<=resistance.upper_bound+.5*vol)
            state="NEAR_DAILY_RESISTANCE_BREAKOUT_WATCH"
            if close>resistance.upper_bound and previous_close<=resistance.upper_bound:
                state="BREAKOUT_ABOVE_RECENT_DAILY_RESISTANCE"
            return {"state":state,"reason_codes":[state],"interesting_today":False,"daily_atr":vol,
                "current_reference_price":close,"nearest_resistance":resistance,
                "resistance":resistance,
                "nearest_resistance_row":audit_row(resistance),
                "resistance_provenance":"PERSISTENT_DAILY_STRUCTURE","resistance_hovering_bars":hover,
                "daily_zones":[audit_row(z) for z in zones],
                "structure_provenance":"PERSISTENT_DAILY_STRUCTURE",**priority_context}
        # A broken resistance is support only after the lifecycle records an
        # accepted break and a subsequent hold/retest.
        flips=[z for z in meaningful if z.original_side=="resistance" and z.role_state=="SUPPORT_CONFIRMED"
               and max(z.lower_bound-close,close-z.upper_bound,0.0)<=cfg.support_watch_distance_atr*vol]
        if flips:
            candidate=min(flips,key=lambda z:_priority_key(z,close))
            if _v2_evidence(candidate,"tradeable",True):
                support=candidate
        if support is None:
            near_support=[z for z in meaningful if z.side=="support" and _v2_evidence(z,"tradeable",True) and
                z.lower_bound<=close and z.upper_bound>=close-cfg.support_watch_distance_atr*vol]
            near_support.sort(key=lambda z:_priority_key(z,close))
            damaged=near_support[0] if near_support else None
            if damaged:
                deteriorated,damage_reasons=_support_deterioration(damaged,rows,ema50,ema200)
                if deteriorated:
                    return {"state":"WAIT_FOR_RECLAIM",
                        "reason_codes":["SUPPORT_DAMAGED",*damage_reasons,"WAIT_FOR_RECLAIM"],
                        "interesting_today":False,"daily_atr":vol,"ema50":ema50,"ema200":ema200,
                        "ema200_available":ema200 is not None,"current_reference_price":close,
                        "broken_zone":damaged,"activation_state":"BROKEN_SUPPORT_RECLAIM_WATCH",
                        "support_damage_reasons":damage_reasons,"daily_zones":[audit_row(z) for z in zones],
                        "direct_breakout_candidates":direct_resistance_watches,
                        "structure_provenance":"PERSISTENT_DAILY_STRUCTURE",**priority_context}
            state="MID_RECENT_DAILY_RANGE" if len(meaningful)>=2 else "NO_CLEAR_RECENT_DAILY_STRUCTURE"
            return {"state":state,"reason_codes":[state],"interesting_today":False,
                "daily_zones":[audit_row(z) for z in zones],
                "daily_atr":vol,"ema50":ema50,"ema200":ema200,
                "structure_provenance":"PERSISTENT_DAILY_STRUCTURE",**priority_context}
    gap=close-support.upper_bound
    inside=support.lower_bound<=close<=support.upper_bound
    state="INSIDE_RECENT_DAILY_SUPPORT" if inside else "NEAR_RECENT_DAILY_SUPPORT"
    if support.original_side=="resistance" and support.role_state=="SUPPORT_CONFIRMED":
        state="RETESTING_BROKEN_RESISTANCE"
    deteriorated,damage_reasons=_support_deterioration(support,rows,ema50,ema200)
    if deteriorated:
        return {"state":"WAIT_FOR_RECLAIM",
            "reason_codes":["SUPPORT_DAMAGED",*damage_reasons,"WAIT_FOR_RECLAIM"],
            "interesting_today":False,"daily_atr":vol,"ema50":ema50,"ema200":ema200,
            "ema200_available":ema200 is not None,"current_reference_price":close,
            "broken_zone":support,"activation_state":"BROKEN_SUPPORT_RECLAIM_WATCH",
            "support_damage_reasons":damage_reasons,"daily_zones":[audit_row(z) for z in zones],
            "structure_provenance":"PERSISTENT_DAILY_STRUCTURE",**priority_context}
    stop=support.lower_bound-cfg.stop_buffer_atr*vol
    target_info=structural_target(close,stop,resistance,(),vol,cfg)
    return {"state":state,"reason_codes":[state,"DAILY_SUPPORT_TESTING" if low<=support.upper_bound else "DAILY_SUPPORT_APPROACHING"],
        "daily_atr":vol,"interesting_today":True,"interesting_today_reason":"VALIDATED_PERSISTENT_SUPPORT_REVISIT",
        "current_reference_price":close,"support_distance_pct":gap/close if close else None,
        "resistance_distance_pct":((resistance.lower_bound-close)/close if resistance and close else None),
        "resistance_distance_atr":((resistance.lower_bound-close)/vol if resistance else None),
        "support":support,"support_row":audit_row(support),
        "resistance":resistance,"support_provenance":"PERSISTENT_DAILY_STRUCTURE",
        "resistance_provenance":"PERSISTENT_DAILY_STRUCTURE" if resistance else None,
        "resistance_row":audit_row(resistance) if resistance else None,
        "target_info":target_info,"stop":stop,"daily_zones":[audit_row(z) for z in zones],
        "support_distance_atr":gap/vol,"structure_provenance":"PERSISTENT_DAILY_STRUCTURE",
        "stop_reference":stop,"dominant_zone_context":priority_context,**priority_context}


def daily_setup_activations(location: Mapping[str, Any], cfg: V2Config) -> list[dict[str, Any]]:
    """Activate monitoring families from daily location; H1 still supplies the trigger."""
    state=str(location.get("state") or "")
    zone=location.get("support") or location.get("broken_zone") or location.get("nearest_resistance")
    family=None
    if state in {"NEAR_RECENT_DAILY_SUPPORT","INSIDE_RECENT_DAILY_SUPPORT"} and zone:
        family="SUPPORT_REACTION"
    elif state=="RETESTING_BROKEN_RESISTANCE" and zone:
        family="BREAKOUT_RETEST"
    elif state in {"NEAR_DAILY_RESISTANCE_BREAKOUT_WATCH","BREAKOUT_ABOVE_RECENT_DAILY_RESISTANCE"} and zone:
        family=("CONSOLIDATION_BREAKOUT" if int(location.get("resistance_hovering_bars",0)) >=
                cfg.daily_zone.reaction_horizon else "BREAKOUT_RETEST")
    elif state=="BROKEN_SUPPORT_RECLAIM_WATCH" and zone:
        family="RECLAIM"
    elif state in {"DAILY_SUPPORT_BROKEN","WAIT_FOR_RECLAIM"} and zone and location.get("activation_state")=="BROKEN_SUPPORT_RECLAIM_WATCH":
        long_respected=(zone.side=="resistance" and zone.original_side=="resistance" and
            int(_v2_evidence(zone,"hovering_bars",0))>=cfg.daily_zone.reaction_horizon and
            _v2_evidence(zone,"validated",True))
        family="CONSOLIDATION_BREAKOUT" if long_respected else "RECLAIM"
    activations=[]
    if family and zone:
        current=float(location.get("current_reference_price") or zone.center)
        if family in {"BREAKOUT_RETEST","CONSOLIDATION_BREAKOUT"}:
            nested=[candidate for candidate in _v2_evidence(zone,"nested_boundaries",())
                if candidate.side=="resistance" and (candidate.original_side or candidate.side)=="resistance"
                and _v2_evidence(candidate,"validated",False)
                and candidate.lower_bound<=current<=zone.upper_bound]
            if nested:
                zone=min(nested,key=lambda candidate:(abs(current-candidate.upper_bound),
                    -float(_v2_evidence(candidate,"dominance_score",0.0)),candidate.zone_id))
        activations.append({"family":family,"zone":zone,
            "setup_event_id":_v2_evidence(zone,"setup_event_id",None) or f"{zone.zone_id}:UNRECORDED_INTERACTION"})
    # A damaged-support/reclaim watch does not suppress a distinct, established
    # resistance-clearance thesis. The H1 path still requires decisive close,
    # volume, friction clearance and structural RR before it can become a trade.
    for candidate in location.get("direct_breakout_candidates", ()):
        if any(item["family"]=="CONSOLIDATION_BREAKOUT" and item["zone"].zone_id==candidate.zone_id
               for item in activations):
            continue
        activations.append({"family":"CONSOLIDATION_BREAKOUT","zone":candidate,
            "setup_event_id":_v2_evidence(candidate,"setup_event_id",None) or
                f"{candidate.zone_id}:DIRECT_BREAKOUT_INTERACTION"})
    breakout=[item for item in activations if item["family"]=="CONSOLIDATION_BREAKOUT"]
    other=[item for item in activations if item["family"]!="CONSOLIDATION_BREAKOUT"]
    # Overlapping historical bands represent one active price area. Keep the
    # structurally strongest member so one interaction cannot fan out into
    # duplicate direct-breakout watches for near-identical zones.
    clusters=[]
    for item in sorted(breakout,key=lambda x:(x["zone"].lower_bound,x["zone"].upper_bound,x["zone"].zone_id)):
        zone=item["zone"]
        if clusters and zone.lower_bound<=max(x["zone"].upper_bound for x in clusters[-1]):
            clusters[-1].append(item)
        else:
            clusters.append([item])
    for cluster in clusters:
        other.append(min(cluster,key=lambda x:(
            0 if _v2_evidence(x["zone"],"structure_level_kind","PARENT_ZONE")=="INNER_BOUNDARY" else 1,
            _priority_key(x["zone"],float(location.get("current_reference_price") or x["zone"].center)))))
    return other


def structural_target(entry: float, stop: float, daily_resistance: Zone | Sequence[Zone] | None,
                      weekly_zones: Sequence[Zone], daily_atr: float, cfg: V2Config) -> dict[str, Any]:
    candidates = []
    daily_candidates=([daily_resistance] if isinstance(daily_resistance,Zone) else list(daily_resistance or ()))
    for timeframe, zones in (("DAILY", daily_candidates),
                             ("WEEKLY", list(weekly_zones))):
        for zone in zones:
            if zone is None or zone.side != "resistance" or zone.strength not in {"MEANINGFUL", "MAJOR"}:
                continue
            # Nested breakout boundaries refine setup lifecycle. Their parent
            # remains the structural target/friction object, avoiding a second
            # synthetic target at the parent edge.
            if _v2_evidence(zone,"structure_level_kind","PARENT_ZONE")=="INNER_BOUNDARY":
                continue
            if zone.broken and zone.role_state not in {"RESISTANCE_CONFIRMED","RESISTANCE_CANDIDATE"}:
                continue
            price = zone.lower_bound-cfg.target_front_run_buffer_atr*daily_atr
            if price > entry:
                candidates.append((price, timeframe, zone))
    if not candidates:
        return {"status": "REJECT", "reason_code": "NO_MEANINGFUL_STRUCTURAL_TARGET"}
    # Structural rank comes only from evidence. Price distance chooses the next
    # target after rank is established; it never promotes a weak nearby level.
    parent=list(range(len(candidates)))
    def root(index: int) -> int:
        while parent[index]!=index:
            parent[index]=parent[parent[index]]; index=parent[index]
        return index
    for i,( _,tf_a,a) in enumerate(candidates):
        for j in range(i+1,len(candidates)):
            _,tf_b,b=candidates[j]
            if (_v2_evidence(a,"structure_level_kind","PARENT_ZONE")!="PARENT_ZONE" or
                    _v2_evidence(b,"structure_level_kind","PARENT_ZONE")!="PARENT_ZONE"):
                continue
            overlaps=a.lower_bound<=b.upper_bound and b.lower_bound<=a.upper_bound
            if overlaps and tf_a!=tf_b:
                parent[root(j)]=root(i)
    groups: dict[int,list[int]]={}
    for i in range(len(candidates)):
        groups.setdefault(root(i),[]).append(i)
    group_evidence={key:(max(float(_v2_evidence(candidates[i][2],"dominance_score",0.0)) for i in indices),
        any(candidates[i][2].strength=="MAJOR" for i in indices),
        len({candidates[i][1] for i in indices})>1) for key,indices in groups.items()}
    parent_groups=[key for key,indices in groups.items() if any(
        _v2_evidence(candidates[i][2],"structure_level_kind","PARENT_ZONE")=="PARENT_ZONE"
        for i in indices)]
    dominant_groups=[key for key,(score,major,multitimeframe) in group_evidence.items()
                     if key in parent_groups and (major or multitimeframe)]
    if not dominant_groups:
        # A sole/top evidence-backed zone is still the dominant structural
        # reference when no major or multi-timeframe confluence exists.
        if not parent_groups:
            return {"status":"REJECT","reason_code":"NO_DOMINANT_PARENT_STRUCTURAL_TARGET",
                "structural_resistance_ladder":[]}
        dominant_groups=[max(parent_groups,key=lambda key:(group_evidence[key][0],
            max(candidates[i][2].strength=="MEANINGFUL" for i in groups[key]),-key))]
    dominant_indices={i for key in dominant_groups for i in groups[key]}
    ladder=[]
    for i,(price,timeframe,candidate_zone) in enumerate(candidates):
        if _v2_evidence(candidate_zone,"structure_level_kind","PARENT_ZONE")=="INNER_BOUNDARY":
            classification="INTERMEDIATE_FRICTION"
        elif i in dominant_indices:
            classification="DOMINANT"
        elif candidate_zone.strength=="MEANINGFUL":
            classification="SECONDARY"
        else:
            classification="LOCAL_FRICTION"
        ladder.append({"zone_id":candidate_zone.zone_id,"timeframe":timeframe,
            "lower_bound":candidate_zone.lower_bound,"upper_bound":candidate_zone.upper_bound,
            "target_price":price,"strength":candidate_zone.strength,"dominance_score":_v2_evidence(candidate_zone,"dominance_score",0.0),
            "classification":classification,"structural_reasons":_v2_evidence(candidate_zone,"dominance_reasons",()),
            "distance_from_entry":price-entry,"distance_atr":(price-entry)/max(daily_atr,1e-12),
            "distance_r":(price-entry)/(entry-stop) if entry>stop else None})
    eligible=[candidates[i] for i in dominant_indices
        if _v2_evidence(candidates[i][2],"structure_level_kind","PARENT_ZONE")=="PARENT_ZONE"]
    target,timeframe,zone=min(eligible,key=lambda x:(x[0],0 if x[1]=="DAILY" else 1,x[2].zone_id))
    risk = entry-stop; reward = target-entry; rr = reward/risk if risk > 0 else None
    return {"status": "READY" if rr is not None and rr >= cfg.minimum_structural_rr else "REJECT",
        "reason_code": "STRUCTURAL_TARGET_VALID" if rr is not None and rr >= cfg.minimum_structural_rr else "RR_INSUFFICIENT",
        "target": target, "target_timeframe": timeframe, "target_zone_id": zone.zone_id,
        "target_zone_strength": zone.strength, "target_zone_lower": zone.lower_bound,
        "target_zone_upper": zone.upper_bound, "risk": risk, "reward": reward,
        "planned_structural_rr": rr,"target_classification":"DOMINANT",
        "target_dominance_score":_v2_evidence(zone,"dominance_score",0.0),
        "target_dominance_reasons":_v2_evidence(zone,"dominance_reasons",()),
        "structural_resistance_ladder":ladder,
        "intermediate_structural_friction":[item for item in ladder if item["classification"] in {"SECONDARY","INTERMEDIATE_FRICTION"}
            and item["target_price"]<target]}


def _volume_confirmation(rows: Sequence[Mapping[str, Any]], archetype: str, cfg: V2Config) -> dict[str, Any]:
    features = volume_features(rows, lookback=cfg.volume_lookback)
    bullish = float(rows[-1]["close"]) > float(rows[-1]["open"])
    expansion = features.get("same_time_expansion") is True or features.get("expansion_vs_prior_mean") is True
    contraction = features.get("same_time_contraction") is True
    if archetype in {"CLEARANCE", "BREAKOUT", "CONSOLIDATION_BREAKOUT"}:
        passed = expansion; kind = "BREAKOUT_CLEARANCE_VOLUME"
        code = "BREAKOUT_CLEARANCE_VOLUME_CONFIRMED" if passed else "BREAKOUT_CLEARANCE_VOLUME_INSUFFICIENT"
    elif archetype == "RECLAIM":
        passed = expansion and bullish; kind = "RECLAIM_PARTICIPATION"
        code = "RECLAIM_VOLUME_CONFIRMED" if passed else "VOLUME_CONFIRMATION_MISSING"
    else:
        passed = bullish and (contraction or expansion); kind = "SUPPLY_EXHAUSTION_PLUS_BUYER_RESPONSE"
        if contraction and bullish:
            code = "SUPPORT_SELLING_VOLUME_CONTRACTING"
        elif expansion and bullish:
            code = "SUPPORT_BULLISH_RESPONSE_VOLUME"
        else:
            code = "VOLUME_CONFIRMATION_MISSING"
    hostile = not bullish and expansion and float(rows[-1]["close"]) < float(rows[-1]["open"])
    if hostile:
        passed = False; code = "HOSTILE_BEARISH_VOLUME_AT_DAILY_SUPPORT"
    return {"required": True, "type": kind, "features": features,
            "passed": bool(passed), "reason_code": code, "hostile": hostile}


def _historical_breakout_acceptance(hourly_rows: Sequence[Mapping[str, Any]], zone: Zone,
                                    cfg: V2Config, end_exclusive: int) -> dict[str, Any] | None:
    """Find a causal, volume-confirmed two-close H1 breakout before a retest watch."""
    first_date=str(zone.creation_date)[:10]
    count=0; candidate=None; accepted_closes=[]; accepted_snapshot=None; accepted=False
    for index in range(0,min(end_exclusive,len(hourly_rows))):
        bar=hourly_rows[index]
        if str(bar["date"])[:10]<first_date:
            continue
        prefix=hourly_rows[:index+1]
        close=float(bar["close"])
        # A close below the former resistance's lower edge invalidates the
        # prior acceptance. A later breakout must establish acceptance anew.
        if close<zone.lower_bound:
            count=0; candidate=None; accepted_closes=[]; accepted_snapshot=None; accepted=False
            continue
        if accepted:
            continue
        h_atr=max(atr(prefix,14),1e-12)
        buffer=cfg.clearance_buffer_atr*h_atr
        volume=volume_features(prefix,lookback=cfg.volume_lookback)
        expansion=volume.get("same_time_expansion") is True or volume.get("expansion_vs_prior_mean") is True
        bullish=close>float(bar["open"])
        passed=close>zone.upper_bound+buffer and bullish and expansion
        if not passed:
            count=0; candidate=None; accepted_closes=[]
            continue
        detail={"timestamp":str(bar["date"]),"close":close,"resistance_upper_bound":zone.upper_bound,
            "close_above_resistance":close-zone.upper_bound,"acceptance_buffer":buffer,
            "close_above_buffered_threshold":close-(zone.upper_bound+buffer),
            "volume_features":volume,"volume_passed":bool(expansion and bullish)}
        if count==0:
            candidate=detail
        accepted_closes.append(detail); count+=1
        if count>=cfg.breakout_acceptance_closes:
            accepted=True
            accepted_snapshot={"breakout_zone":{"zone_id":zone.zone_id,"lower_bound":zone.lower_bound,
                    "upper_bound":zone.upper_bound,"original_side":zone.original_side},
                "breakout_timestamp":candidate["timestamp"],"breakout_close":candidate["close"],
                "close_above_resistance":candidate["close_above_resistance"],
                "acceptance_buffer":candidate["acceptance_buffer"],
                "breakout_volume_features":candidate["volume_features"],
                "breakout_volume_passed":candidate["volume_passed"],
                "acceptance_closes":accepted_closes,"acceptance_state_after_breakout":"ACCEPTED",
                "accepted_timestamp":str(bar["date"]),"breakout_index":index-len(accepted_closes)+1,
                "accepted_index":index}
    return accepted_snapshot


def _strong_ema_rejections(rows: Sequence[Mapping[str, Any]], period: int, cfg: V2Config) -> int:
    recent = rows[-cfg.ema_rejection_lookback_bars:]
    # EMA evidence is only strong after repeated bars traded at/above and closed below it.
    count = 0
    for index in range(max(0, len(rows)-len(recent)), len(rows)):
        prefix = rows[:index+1]; value = ema(prefix, period)
        if value is not None and float(rows[index]["high"]) >= value and float(rows[index]["close"]) < value:
            count += 1
    return count


def _direct_breakout_features(rows: Sequence[Mapping[str, Any]], zone: Zone, h_atr: float,
                              cfg: V2Config) -> dict[str, Any]:
    """Causal direct clearance test for an established, long-respected resistance."""
    bar=rows[-1]
    high=float(bar["high"]); low=float(bar["low"]); open_price=float(bar["open"]); close=float(bar["close"])
    span=max(high-low,1e-12)
    body_fraction=max(close-open_price,0.0)/span
    close_location=(close-low)/span
    n=cfg.consolidation_bars
    prior=rows[-n-1:-1]
    prior_ranges=[float(x["high"])-float(x["low"]) for x in prior]
    prior_median=median(prior_ranges) if prior_ranges else None
    range_ratio=span/max(float(prior_median or 0.0),1e-12) if prior_median is not None else None
    buffer=cfg.clearance_buffer_atr*h_atr
    hovering=int(_v2_evidence(zone,"hovering_bars",0))
    repeated=(len(zone.touch_episodes)>=cfg.meaningful_touch_episodes and
              int(_v2_evidence(zone,"failed_crosses",0))>0)
    respected=(zone.side=="resistance" and zone.strength in {"MEANINGFUL","MAJOR"} and
        _v2_evidence(zone,"validated",True) and
        (hovering>=cfg.daily_zone.reaction_horizon or repeated))
    vol_features=volume_features(rows,lookback=cfg.volume_lookback)
    expansion=(vol_features.get("same_time_expansion") is True or
               vol_features.get("expansion_vs_prior_mean") is True)
    checks={
        "established_resistance":respected,
        "long_respect_sessions":hovering,
        "repeated_failed_cross_evidence":repeated,
        "close_clearly_above_zone":close>zone.upper_bound+buffer,
        "bullish_candle":close>open_price,
        "body_fraction":body_fraction,
        "body_fraction_passed":body_fraction>=cfg.direct_breakout_min_body_fraction,
        "close_location":close_location,
        "close_location_passed":close_location>=cfg.direct_breakout_min_close_location,
        "range_expansion_ratio":range_ratio,
        "range_expansion_passed":range_ratio is not None and range_ratio>=cfg.direct_breakout_min_range_ratio,
        "volume_features":vol_features,
        "volume_expansion_passed":expansion,
        "buffer":buffer,
        "zone_upper_bound":zone.upper_bound,
        "confirmation_close":close,
    }
    checks["passed"]=all((checks["established_resistance"],checks["close_clearly_above_zone"],
        checks["bullish_candle"],checks["body_fraction_passed"],checks["close_location_passed"],
        checks["range_expansion_passed"],checks["volume_expansion_passed"]))
    return checks


def hourly_entry(*, hourly_rows: Sequence[Mapping[str, Any]], start_index: int,
                 support: Zone, daily_stop: float, daily_resistance: Zone | Sequence[Zone] | None,
                 weekly_zones: Sequence[Zone], daily_atr: float, security_id: str,
                 cfg: V2Config, end_index: int | None = None,
                 setup_family: str = "SUPPORT_REACTION",
                 setup_state: MutableMapping[str, Any] | None = None) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    rejected: list[dict[str, Any]] = []
    state=setup_state if setup_state is not None else {}
    friction_required_seen = bool(state.get("friction_required_seen",False))
    friction_history: list[dict[str, Any]] = state.setdefault("friction_history",[])
    clearance_history: dict[str, Any] | None = state.get("clearance_history")
    pre_clearance_rr: float | None = state.get("pre_clearance_rr")
    accepted_breakout=bool(state.get("accepted_breakout",False))
    breakout_acceptance_count=int(state.get("breakout_acceptance_count",0))
    breakout_acceptance_index=int(state.get("breakout_acceptance_index",-1))
    breakout_provenance=state.get("breakout_provenance")
    breakout_candidate=state.get("breakout_candidate")
    retest_candidate=state.get("retest_candidate")
    if (setup_family=="BREAKOUT_RETEST" and support.original_side=="resistance"
            and not state.get("breakout_history_scanned")):
        state["breakout_history_scanned"]=True
        historical=_historical_breakout_acceptance(hourly_rows,support,cfg,start_index)
        if historical:
            accepted_breakout=True; breakout_provenance=historical
            breakout_acceptance_count=cfg.breakout_acceptance_closes
            breakout_acceptance_index=int(historical["accepted_index"])
            breakout_candidate=None

    def persist_state() -> None:
        state.update({"friction_required_seen":friction_required_seen,
            "friction_history":friction_history,"clearance_history":clearance_history,
            "pre_clearance_rr":pre_clearance_rr,"accepted_breakout":accepted_breakout,
            "breakout_acceptance_count":breakout_acceptance_count,
            "breakout_acceptance_index":breakout_acceptance_index,
            "breakout_provenance":breakout_provenance,"breakout_candidate":breakout_candidate,
            "retest_candidate":retest_candidate})
    last_index = min(len(hourly_rows)-1, end_index if end_index is not None else len(hourly_rows)-1)
    for i in range(start_index, last_index):
        prefix = hourly_rows[:i+1]; bar = prefix[-1]; price = float(bar["close"])
        if setup_family not in {"BREAKOUT_RETEST","CONSOLIDATION_BREAKOUT"} and float(bar["low"]) > support.upper_bound + cfg.support_watch_distance_atr*daily_atr:
            continue
        h_atr = max(atr(prefix, 14), 1e-12)
        hourly_zones = enriched_zones(prefix[-cfg.hourly_zone.maximum_age:], security_id, cfg.hourly_zone, cfg)
        raw_volume = volume_features(prefix, lookback=cfg.volume_lookback)
        if (setup_family in {"SUPPORT_REACTION","HL_HH_CONFIRMATION","FAILED_BREAKDOWN_RECLAIM"}
                and float(bar["low"]) <= support.upper_bound and float(bar["close"]) < float(bar["open"])
                and float(bar["close"]) < support.lower_bound
                and (raw_volume.get("same_time_expansion") is True or raw_volume.get("expansion_vs_prior_mean") is True)):
            rejected.append({"reason_code":"HOSTILE_BEARISH_VOLUME_AT_DAILY_SUPPORT","timestamp":bar["date"],"volume":raw_volume})
            continue
        if price < support.lower_bound:
            if setup_family=="BREAKOUT_RETEST" and support.original_side=="resistance" and accepted_breakout:
                accepted_breakout=False; breakout_acceptance_count=0; breakout_acceptance_index=-1
                breakout_provenance=None; breakout_candidate=None
                rejected.append({"reason_code":"SETUP_INVALIDATED","invalidation":"BREAKOUT_ACCEPTANCE_INVALIDATED",
                    "timestamp":bar["date"],"close":price,"zone_lower_bound":support.lower_bound})
                break
            may_wait_below_unbroken_resistance=(setup_family in {"BREAKOUT_RETEST","CONSOLIDATION_BREAKOUT"}
                and support.side=="resistance" and
                support.role_state in {"ACTIVE","RESISTANCE_CANDIDATE","SUPPORT_CANDIDATE",
                    "BROKEN_RESISTANCE_WAITING_HOLD"} and not accepted_breakout)
            if setup_family not in {"RECLAIM","FAILED_BREAKDOWN_RECLAIM"} and not may_wait_below_unbroken_resistance:
                rejected.append({"reason_code":"DAILY_SUPPORT_BROKEN","timestamp":bar["date"],
                    "hourly_close":price,"daily_zone_lower_bound":support.lower_bound,
                    "daily_zone_upper_bound":support.upper_bound,"role_state":support.role_state})
                break
        overhead = [z for z in hourly_zones if z.side == "resistance" and z.strength in {"MEANINGFUL", "MAJOR"}
                    and z.lower_bound >= price and (not z.broken or z.role_state == "RESISTANCE_CONFIRMED")]
        overhead.sort(key=lambda z: (z.lower_bound, z.zone_id))
        friction: list[dict[str, Any]] = []
        resistance = overhead[0] if overhead else None
        if resistance and resistance.lower_bound-price <= cfg.friction_distance_atr*h_atr:
            friction.append({"type": "HOURLY_RESISTANCE", "price": resistance.lower_bound,
                "zone_lower":resistance.lower_bound,"zone_upper":resistance.upper_bound,
                "zone_id": resistance.zone_id, "distance_pct": (resistance.lower_bound-price)/price,
                "distance_atr": (resistance.lower_bound-price)/h_atr,
                "distance_r": (resistance.lower_bound-price)/(price-daily_stop) if price>daily_stop else None,
                "strong": resistance.strength in {"MEANINGFUL", "MAJOR"}})
        for period in (50, 200):
            value = ema(prefix, period); rejects = _strong_ema_rejections(prefix, period, cfg)
            if value is not None and value >= price and (value-price) <= cfg.friction_distance_atr*h_atr:
                strong = rejects >= cfg.ema_min_rejections
                if any(abs(value-z.center) <= .25*h_atr for z in overhead):
                    strong = True
                friction.append({"type": f"EMA{period}", "price": value, "distance_pct": (value-price)/price,
                    "distance_atr": (value-price)/h_atr, "distance_r": (value-price)/(price-daily_stop),
                    "strong": strong, "rejection_count": rejects})
        strong = [f for f in friction if f["strong"]]
        if any(f["type"].startswith("EMA") for f in strong) and any(f["type"]=="HOURLY_RESISTANCE" for f in strong):
            for item in strong: item["confluence"] = True
            confluence_price=max(float(f["price"]) for f in strong)
            friction.append({"type":"HOURLY_SR_EMA_CONFLUENCE","price":confluence_price,
                "distance_pct":(confluence_price-price)/price,"distance_atr":(confluence_price-price)/h_atr,
                "distance_r":(confluence_price-price)/(price-daily_stop) if price>daily_stop else None,"strong":True})
            strong.append(friction[-1])
        for item in friction:
            key=(item.get("type"),item.get("zone_id"),round(float(item.get("price",0)),6))
            existing=next((x for x in friction_history if (x.get("type"),x.get("zone_id"),round(float(x.get("price",0)),6))==key),None)
            if existing is None:
                friction_history.append({**item,"first_seen":str(bar["date"]),"last_seen":str(bar["date"])})
            else:
                existing["last_seen"]=str(bar["date"]); existing["strong"]=bool(existing.get("strong") or item.get("strong"))
        friction_required_seen = friction_required_seen or bool(strong)
        is_bullish = float(bar["close"]) > float(bar["open"])
        prior = prefix[-2] if len(prefix)>1 else None
        archetype = None; archetype_family=None
        direct_breakout_evidence=None
        direct_zone_watch=(setup_family in {"RECLAIM","CONSOLIDATION_BREAKOUT"} and
            support.side=="resistance" and
            support.role_state in {"ACTIVE","RESISTANCE_CANDIDATE","BROKEN_RESISTANCE_WAITING_HOLD"})
        if direct_zone_watch:
            direct_breakout_evidence=_direct_breakout_features(prefix,support,h_atr,cfg)
            if direct_breakout_evidence["passed"]:
                archetype="CONSOLIDATION_BREAKOUT"
                archetype_family="CONSOLIDATION_BREAKOUT"
            elif float(bar["high"])>support.upper_bound:
                rejected.append({"reason_code":"DIRECT_BREAKOUT_QUALIFICATION_NOT_MET",
                    "timestamp":str(bar["date"]),"direct_breakout_evidence":direct_breakout_evidence})
                persist_state()
                continue
        if archetype is not None:
            pass
        elif setup_family=="BREAKOUT_RETEST":
            if not accepted_breakout and support.original_side=="resistance":
                threshold=support.upper_bound+cfg.clearance_buffer_atr*h_atr
                expansion=raw_volume.get("same_time_expansion") is True or raw_volume.get("expansion_vs_prior_mean") is True
                if price>threshold and is_bullish and expansion:
                    if breakout_acceptance_count==0:
                        breakout_candidate={"breakout_zone":{"zone_id":support.zone_id,"lower_bound":support.lower_bound,
                            "upper_bound":support.upper_bound,"original_side":support.original_side},
                            "breakout_timestamp":str(bar["date"]),"breakout_close":price,
                            "close_above_resistance":price-support.upper_bound,
                            "acceptance_buffer":cfg.clearance_buffer_atr*h_atr,
                            "breakout_volume_features":raw_volume,"breakout_volume_passed":True,"acceptance_closes":[]}
                    breakout_acceptance_count+=1
                    breakout_candidate["acceptance_closes"].append({"timestamp":str(bar["date"]),"close":price,
                        "close_above_resistance":price-support.upper_bound,
                        "acceptance_buffer":cfg.clearance_buffer_atr*h_atr,
                        "close_above_buffered_threshold":price-threshold,
                        "volume_features":raw_volume,"volume_passed":True})
                    if breakout_acceptance_count>=cfg.breakout_acceptance_closes:
                        accepted_breakout=True; breakout_acceptance_index=i
                        breakout_provenance={**breakout_candidate,"acceptance_state_after_breakout":"ACCEPTED",
                            "accepted_timestamp":str(bar["date"]),
                            "breakout_index":i-len(breakout_candidate["acceptance_closes"])+1,"accepted_index":i}
                        breakout_candidate=None
                else:
                    breakout_acceptance_count=0
                    breakout_candidate=None
            if accepted_breakout and i>breakout_acceptance_index:
                bar_low=float(bar["low"]); bar_high=float(bar["high"])
                distance=max(0.0,support.upper_bound-bar_high,bar_low-support.upper_bound)
                distance_atr=distance/h_atr
                if retest_candidate is None and prior and float(prior["close"])>support.upper_bound and \
                        bar_low<=support.upper_bound+cfg.support_watch_distance_atr*h_atr:
                    retest_candidate={"approach_timestamp":str(bar["date"]),
                        "prior_bar_timestamp":str(prior["date"]),"prior_close":float(prior["close"]),
                        "approached_from_above":float(prior["close"])>support.upper_bound,
                        "minimum_price_during_retest":bar_low,"minimum_distance_to_zone":distance,
                        "minimum_distance_to_zone_atr":distance_atr,"touched_zone":bar_low<=support.upper_bound}
                elif retest_candidate is not None:
                    retest_candidate["minimum_price_during_retest"]=min(
                        float(retest_candidate["minimum_price_during_retest"]),bar_low)
                    retest_candidate["minimum_distance_to_zone"]=min(
                        float(retest_candidate["minimum_distance_to_zone"]),distance)
                    retest_candidate["minimum_distance_to_zone_atr"]=min(
                        float(retest_candidate["minimum_distance_to_zone_atr"]),distance_atr)
                    retest_candidate["touched_zone"]=bool(retest_candidate["touched_zone"] or bar_low<=support.upper_bound)
                if retest_candidate is not None and price>=support.center and is_bullish:
                    touched=bool(retest_candidate["touched_zone"])
                    subtype="BREAKOUT_RETEST_TOUCH" if touched else "BREAKOUT_RETEST_FRONT_RUN"
                    archetype="BREAKOUT_RETEST"
                    retest_evidence={"retest_subtype":subtype,
                        "former_resistance_lower":support.lower_bound,"former_resistance_upper":support.upper_bound,
                        "minimum_price_during_retest":retest_candidate["minimum_price_during_retest"],
                        "minimum_distance_to_zone":retest_candidate["minimum_distance_to_zone"],
                        "minimum_distance_to_zone_atr":retest_candidate["minimum_distance_to_zone_atr"],
                        "touched_zone":touched,"retest_timestamp":retest_candidate["approach_timestamp"],
                        "retest_response_timestamp":str(bar["date"]),
                        "retest_response_type":"BULLISH_CLOSE_HOLD_ABOVE_CENTER",
                        "approached_from_above":bool(retest_candidate.get("approached_from_above")),
                        "retest_tested_from_above":bool(touched and
                            retest_candidate.get("approached_from_above")),
                        "retest_held_or_reclaimed_center":price>=support.center,
                        "bullish_retest_close":is_bullish,
                        "retest_low":float(retest_candidate["minimum_price_during_retest"]),
                        "retest_close":price,
                        "retest_tolerance":cfg.support_watch_distance_atr*h_atr,
                        "approached_within_tolerance":True}
                elif retest_candidate is not None and price>support.upper_bound+cfg.support_watch_distance_atr*h_atr:
                    # Price left the approach area without a hold/response; a later move
                    # must establish a fresh approach before it can qualify.
                    retest_candidate=None
            if archetype is None:
                reference={"lower":support.lower_bound,"upper":support.upper_bound,"center":support.center}
                pattern,_,features=_hourly_pattern(prefix,hourly_zones,reference,cfg.consolidation_bars,
                                                   structure_lookback=cfg.hourly_zone.maximum_age)
                if pattern=="CONSOLIDATION_BREAKOUT_CONFIRMATION" and is_bullish and \
                        (support.original_side!="resistance" or accepted_breakout):
                    archetype="CONSOLIDATION_BREAKOUT"
        elif setup_family=="CONSOLIDATION_BREAKOUT":
            n=cfg.consolidation_bars
            if support.side!="resistance" and len(prefix)>=2*n+1:
                base=prefix[-n-1:-1]; prior_base=prefix[-2*n-1:-n-1]
                base_range=max(float(x["high"]) for x in base)-min(float(x["low"]) for x in base)
                prior_range=max(float(x["high"]) for x in prior_base)-min(float(x["low"]) for x in prior_base)
                if base_range<prior_range and price>max(float(x["high"]) for x in base) and is_bullish:
                    archetype="CONSOLIDATION_BREAKOUT"
        elif setup_family=="HL_HH_CONFIRMATION":
            reference={"lower":support.lower_bound,"upper":support.upper_bound,"center":support.center}
            pattern,_,features=_hourly_pattern(prefix,hourly_zones,reference,cfg.consolidation_bars,
                                               structure_lookback=cfg.hourly_zone.maximum_age)
            if pattern=="HIGHER_LOW_HIGHER_HIGH_CONFIRMATION" and is_bullish:
                archetype="HL_HH_CONFIRMATION"
            elif pattern=="SUPPORT_HOLD_CONFIRMATION" and is_bullish:
                archetype="HL_HH_CONFIRMATION"
        elif setup_family=="FAILED_BREAKDOWN_RECLAIM":
            if float(bar["low"])<support.lower_bound and price>support.upper_bound and is_bullish:
                archetype="RECLAIM"; archetype_family="FAILED_BREAKDOWN_RECLAIM"
            else:
                reference={"lower":support.lower_bound,"upper":support.upper_bound,"center":support.center}
                pattern,_,features=_hourly_pattern(prefix,hourly_zones,reference,cfg.consolidation_bars,
                                                   structure_lookback=cfg.hourly_zone.maximum_age)
                if int(features.get("failed_downside_attempts",0))>=2 and prior and price>float(prior["high"]) and is_bullish:
                    archetype="RECLAIM"
        else:
            archetype = "SUPPORT_REACTION" if is_bullish and float(bar["low"]) <= support.upper_bound else None
            if prior and float(prior["close"]) < support.lower_bound and price >= support.upper_bound and is_bullish:
                archetype = "RECLAIM"
                if float(bar["low"])<support.lower_bound:
                    archetype_family="FAILED_BREAKDOWN_RECLAIM"
            if setup_family=="RECLAIM" and prior and float(prior["close"])<support.lower_bound and price>=support.upper_bound and is_bullish:
                archetype="RECLAIM"
            if archetype is None:
                reference={"lower":support.lower_bound,"upper":support.upper_bound,"center":support.center}
                pattern,_,features=_hourly_pattern(prefix,hourly_zones,reference,cfg.consolidation_bars,
                                                   structure_lookback=cfg.hourly_zone.maximum_age)
                pattern_family={"RECLAIM_CONFIRMATION":"RECLAIM","SUPPORT_HOLD_CONFIRMATION":"SUPPORT_REACTION",
                    "HIGHER_LOW_HIGHER_HIGH_CONFIRMATION":"HL_HH_CONFIRMATION",
                    "CONSOLIDATION_BREAKOUT_CONFIRMATION":"CONSOLIDATION_BREAKOUT"}.get(pattern)
                if pattern_family and is_bullish and (setup_family in {"SUPPORT_REACTION","RECLAIM"} or pattern_family==setup_family):
                    archetype={"RECLAIM":"RECLAIM","SUPPORT_REACTION":"SUPPORT_REACTION",
                        "HL_HH_CONFIRMATION":"HL_HH_CONFIRMATION","CONSOLIDATION_BREAKOUT":"CONSOLIDATION_BREAKOUT",
                        "BREAKOUT_RETEST":"BREAKOUT_RETEST"}[pattern_family]
                    if pattern_family=="RECLAIM" and int(features.get("failed_downside_attempts",0))>=2:
                        archetype_family="FAILED_BREAKDOWN_RECLAIM"
        if pre_clearance_rr is None:
            prospective_entry=float(hourly_rows[i+1]["open"])
            preplan=structural_target(prospective_entry,daily_stop,daily_resistance,weekly_zones,daily_atr,cfg)
            pre_clearance_rr=preplan.get("planned_structural_rr")
        clearance = None
        if strong:
            candidate = min(strong, key=lambda x: (x["price"], x["type"]))
            threshold = float(candidate.get("zone_upper",candidate["price"]))+cfg.clearance_buffer_atr*h_atr
            volume = _volume_confirmation(prefix, "CLEARANCE", cfg)
            if price > threshold and is_bullish and volume["passed"]:
                clearance = {**candidate, "timestamp": str(bar["date"]), "close": price,
                             "volume_confirmed": True, "threshold": threshold}
                clearance_history = clearance
                archetype = None if setup_family=="BREAKOUT_RETEST" else "CLEARANCE"
                if setup_family=="BREAKOUT_RETEST":
                    continue
            else:
                rejected.append({"reason_code": "WAITING_FOR_HOURLY_FRICTION_CLEARANCE",
                    "friction": candidate, "timestamp": bar["date"],
                    "friction_types":sorted({str(f.get("type")) for f in strong}),
                    "volume_code": volume["reason_code"]})
                continue
        if not archetype:
            continue
        volume = _volume_confirmation(prefix, archetype, cfg)
        if not volume["passed"]:
            rejected.append({"reason_code": volume["reason_code"], "timestamp": bar["date"],
                             "volume": volume})
            continue
        family=(archetype_family or (archetype if archetype in {"RECLAIM","HL_HH_CONFIRMATION","CONSOLIDATION_BREAKOUT","BREAKOUT_RETEST"}
                else "SUPPORT_REACTION"))
        entry_bar = hourly_rows[i+1]
        entry = float(entry_bar["open"])
        direct_breakout=(family=="CONSOLIDATION_BREAKOUT" and support.side=="resistance")
        required_fill_bound=(support.upper_bound if direct_breakout or setup_family in {"RECLAIM","FAILED_BREAKDOWN_RECLAIM"}
                             else support.lower_bound)
        if entry<required_fill_bound:
            code=("DIRECT_BREAKOUT_GAPPED_BACK_BELOW_RESISTANCE" if direct_breakout else
                  "RECLAIM_NOT_HELD_AT_ENTRY" if setup_family in {"RECLAIM","FAILED_BREAKDOWN_RECLAIM"}
                  else "ENTRY_GAPPED_BELOW_STRUCTURAL_ZONE")
            rejected.append({"reason_code":code,"timestamp":entry_bar["date"],"entry_open":entry,
                "required_zone_bound":required_fill_bound,"daily_zone_lower_bound":support.lower_bound,
                "daily_zone_upper_bound":support.upper_bound,"role_state":support.role_state,
                "confirmation_timestamp":str(bar["date"]),"confirmation_type":archetype})
            continue
        target = structural_target(entry, daily_stop, daily_resistance, weekly_zones, daily_atr, cfg)
        if target["status"] != "READY":
            reason = "CLEARANCE_CONSUMED_STRUCTURAL_ROOM" if clearance else target["reason_code"]
            rejected.append({"reason_code": reason, "timestamp": entry_bar["date"],
                             "pre_clearance_structural_rr": None,
                             "post_clearance_structural_rr": target.get("planned_structural_rr")})
            continue
        plan = {"entry": entry, "stop": daily_stop, "target": target["target"],
            "target_timeframe": target["target_timeframe"], "target_zone_id": target["target_zone_id"],
            "target_zone_strength": target["target_zone_strength"],
            "target_classification":target.get("target_classification"),
            "structural_resistance_ladder":target.get("structural_resistance_ladder",[]),
            "intermediate_structural_friction":target.get("intermediate_structural_friction",[]),
            "daily_support_strength": support.strength,
            "planned_reward_r": target["planned_structural_rr"], "target_reference": target["target_timeframe"],
            "daily_support_zone_id": support.zone_id, "daily_support_lower": support.lower_bound,
            "daily_support_upper": support.upper_bound, "daily_resistance_zone_id": target["target_zone_id"] if target["target_timeframe"]=="DAILY" else None,
            "structural_risk": target["risk"], "structural_reward": target["reward"],
            "stop_distance_pct": (entry-daily_stop)/entry, "stop_distance_atr": (entry-daily_stop)/daily_atr,
            "entry_to_support": entry-support.upper_bound, "zone_width_pct": (support.upper_bound-support.lower_bound)/support.center,
            "buffer_below_support": support.lower_bound-daily_stop,
            "immediate_hourly_friction_present": bool(friction_history), "friction": friction_history,
            "friction_clearance_required": friction_required_seen, "friction_clearance": clearance_history,
            "friction_type": ",".join(sorted({str(x["type"]) for x in friction_history})),
            "friction_price": friction_history[0]["price"] if friction_history else None,
            "friction_distance_pct": friction_history[0].get("distance_pct") if friction_history else None,
            "friction_distance_atr": friction_history[0].get("distance_atr") if friction_history else None,
            "friction_distance_r": friction_history[0].get("distance_r") if friction_history else None,
            "friction_clearance_timestamp": clearance_history.get("timestamp") if clearance_history else None,
            "friction_clearance_price": clearance_history.get("close") if clearance_history else None,
            "friction_clearance_volume_confirmed": clearance_history.get("volume_confirmed") if clearance_history else False,
            "pre_clearance_structural_rr": pre_clearance_rr, "post_clearance_structural_rr": target["planned_structural_rr"],
            "volume_confirmation_required": True, "volume_confirmation_type": volume["type"],
            "volume_confirmation_features": volume["features"], "volume_confirmation_passed": volume["passed"]}
        if family=="BREAKOUT_RETEST":
            if not breakout_provenance or not breakout_provenance.get("breakout_volume_passed"):
                rejected.append({"reason_code":"ORIGINAL_BREAKOUT_NOT_VOLUME_CONFIRMED",
                    "timestamp":str(bar["date"]),"confirmation_type":archetype})
                persist_state()
                continue
            breakout_provenance={**breakout_provenance,
                "original_breakout_timestamp":breakout_provenance.get("breakout_timestamp"),
                "original_breakout_volume_confirmation":bool(breakout_provenance.get("breakout_volume_passed")),
                "retest":{**retest_evidence,
                "retest_volume_confirmation":{"type":volume["type"],"features":volume["features"],
                    "passed":bool(volume["passed"])},
                "volume_evidence":volume,"volume_passed":bool(volume["passed"])},
                "final_confirmation_timestamp":str(bar["date"]),
                "acceptance_state_after_breakout":"ACCEPTED"}
        persist_state()
        return ({"state": "ENTRY_READY", "reason_codes": ["DAILY_STRUCTURE_VALID", family, volume["reason_code"]],
                 "setup_family":family,
                 "confirmation_type": archetype, "confirmation_timestamp": str(bar["date"]),
                 "entry_timestamp": str(entry_bar["date"]), "entry_type": "NEXT_AVAILABLE_1H_OPEN",
                 "breakout_path":"DIRECT_STRUCTURAL_BREAKOUT" if direct_breakout else
                    "BREAKOUT_RETEST" if family=="BREAKOUT_RETEST" else None,
                 "direct_breakout_evidence":direct_breakout_evidence,
                 "breakout_retest_provenance":breakout_provenance if family=="BREAKOUT_RETEST" else None,
                 "confirmation_features": {"volume": volume, "friction": friction}, "plan": plan}, rejected)
    persist_state()
    return None, rejected
