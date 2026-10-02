"""V3 regime-aware wrapper and trendline-pullback setup evaluator.

Horizontal V2 signal plans are consumed without changing their structure or
execution semantics. This module only applies explicit V3 context rules and
provides a separate inclined-geometry setup path.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from src.strategy.structural_regime_v3 import classify_regime, trendlines
from src.strategy.top_down_swing import volume_features


@dataclass(frozen=True)
class V3Config:
    minimum_structural_rr: float = 1.5
    support_near_atr: float = 0.75
    channel_lower_region_max: float = 0.40
    trendline_tolerance_atr: float = 0.20
    stop_buffer_atr: float = 0.25
    friction_distance_atr: float = 0.50
    clearance_buffer_atr: float = 0.10
    volume_lookback: int = 20

    def __post_init__(self) -> None:
        if self.minimum_structural_rr < 1.5:
            raise ValueError("V3 minimum planned RR cannot be below 1.5")
        if not 0 < self.channel_lower_region_max <= 0.5:
            raise ValueError("channel lower-region threshold must be in (0, 0.5]")
        if min(self.support_near_atr,self.trendline_tolerance_atr,self.stop_buffer_atr,
               self.friction_distance_atr,self.clearance_buffer_atr) < 0:
            raise ValueError("V3 distance settings must be nonnegative")


@dataclass
class TrendlineSetupLifecycle:
    """One executable thesis per touch episode; revisits need a reset."""
    trendline_id: str
    episode: int = 1
    status: str = "WATCHING"
    consumed_on: str | None = None
    reset_reason: str | None = None

    def advance_day(self, *, trendline_id: str, close: float, support_level: float,
                    daily_atr: float, date_text: str) -> None:
        if trendline_id!=self.trendline_id:
            self.trendline_id=trendline_id; self.episode+=1; self.status="WATCHING"
            self.consumed_on=None; self.reset_reason="NEW_CONFIRMED_TRENDLINE_ANCHOR_PAIR"
        elif self.status=="CONSUMED" and close>=support_level+max(daily_atr,1e-12):
            self.episode+=1; self.status="WATCHING"; self.consumed_on=None
            self.reset_reason=f"PRICE_REESTABLISHED_ABOVE_SUPPORT_{date_text}"

    def consume(self, date_text: str) -> str:
        if self.status!="WATCHING": raise ValueError("A consumed trendline thesis cannot be consumed twice")
        self.status="CONSUMED"; self.consumed_on=date_text
        return f"{self.trendline_id}:EPISODE:{self.episode}"


def family_allowed(regime: str, family: str, *, reclaim_confirmed: bool = False,
                   unusually_strong: bool = False) -> tuple[bool,str]:
    """Regime-specific permissions; ambiguous is fail-closed by default."""
    if regime not in {"TREND_UP","TREND_DOWN","RANGE","TRANSITION","AMBIGUOUS"}:
        raise ValueError(f"Unknown V3 regime: {regime}")
    if family == "TRENDLINE_PULLBACK":
        return (regime == "TREND_UP", "TREND_UP_REQUIRED" if regime != "TREND_UP" else "INCLINED_SETUP_ALLOWED")
    if regime == "TREND_DOWN" and family == "SUPPORT_REACTION" and not reclaim_confirmed:
        return False,"DOWNTREND_SUPPORT_REQUIRES_RECLAIM"
    if regime == "TRANSITION" and family in {"SUPPORT_REACTION","TRENDLINE_PULLBACK"} and not reclaim_confirmed:
        return False,"TRANSITION_REQUIRES_RECLAIM_OR_ACCEPTANCE"
    if regime == "AMBIGUOUS" and not (reclaim_confirmed and unusually_strong):
        return False,"AMBIGUOUS_REQUIRES_UNUSUALLY_STRONG_EVIDENCE"
    return True,"REGIME_CONTEXT_ALLOWED"


def trendline_pullback_candidate(*, daily_prefix: Sequence[Mapping[str,Any]],
                                 hourly_prefix: Sequence[Mapping[str,Any]],
                                 entry_bar: Mapping[str,Any] | None,
                                 target_candidates: Sequence[Mapping[str,Any]],
                                 config: V3Config = V3Config(),
                                 decision_date: str | None = None,
                                 regime_snapshot: Mapping[str,Any] | None = None,
                                 friction_candidates: Sequence[Mapping[str,Any]] = ()) -> dict[str,Any]:
    """Build an executable TRENDLINE_PULLBACK only from completed causal inputs.

    `hourly_prefix` ends on the confirming H1 close. `entry_bar` is the next
    actual provider bar and is used only as the planned executable open.
    Target candidates are established non-H1 resistances, rank ordered by the
    caller's persistent structural evidence.
    """
    regime=dict(regime_snapshot) if regime_snapshot is not None else classify_regime(daily_prefix,as_of=decision_date)
    line=trendlines(daily_prefix).get("rising_support")
    if regime["regime"]!="TREND_UP" or not line or line.status not in {"ESTABLISHED","STRONG"}:
        return {"status":"REJECT","reason_code":"NO_ESTABLISHED_UPTREND_GEOMETRY","regime":regime,
                "trendline":line.as_dict() if line else None}
    if not hourly_prefix or entry_bar is None:
        return {"status":"REJECT","reason_code":"NO_CAUSAL_H1_CONFIRMATION_OR_NEXT_OPEN","regime":regime,
                "trendline":line.as_dict()}
    daily_atr=max(float(regime.get("atr",0) or 0), sum(float(r["high"])-float(r["low"]) for r in daily_prefix[-14:])/min(14,len(daily_prefix)),1e-9)
    last_close=float(daily_prefix[-1]["close"])
    rising=line.projected_level
    channel_high=line.channel_opposite_level
    if channel_high is None or channel_high<=rising:
        return {"status":"REJECT","reason_code":"NO_VALID_RISING_CHANNEL","regime":regime,"trendline":line.as_dict()}
    percentile=(last_close-rising)/(channel_high-rising)
    if percentile>config.channel_lower_region_max or last_close<rising-config.support_near_atr*daily_atr:
        return {"status":"REJECT","reason_code":"NOT_IN_LOWER_RISING_CHANNEL_REGION","channel_percentile":percentile,
                "regime":regime,"trendline":line.as_dict()}
    recent=hourly_prefix[-4:]
    touched=any(float(b["low"])<=rising+line.tolerance for b in recent)
    # Buyer control requires both a rejected downside attempt at the rising
    # structure and a close through the recent local lower-high boundary.
    # Volume/candle strength can confirm this transfer, but cannot replace it.
    touch_index=next((i for i in range(len(hourly_prefix)-1,-1,-1)
                      if float(hourly_prefix[i]["low"])<=rising+line.tolerance),None)
    response_rows=hourly_prefix[touch_index:] if touch_index is not None else []
    prior_response=hourly_prefix[max(0,(touch_index or 0)-3):(touch_index or 0)] if touch_index is not None else []
    failed_downside=bool(response_rows and any(float(b["close"])>=float(b["low"])+
        0.5*max(float(b["high"])-float(b["low"]),1e-12) for b in response_rows))
    local_lower_high=max((float(b["high"]) for b in prior_response),default=None)
    reclaim=bool(local_lower_high is not None and float(hourly_prefix[-1]["close"])>local_lower_high)
    confirmation=failed_downside and reclaim
    vfeatures=volume_features(hourly_prefix,lookback=config.volume_lookback)
    relative=float(vfeatures.get("same_time_relative_volume") or vfeatures.get("relative_volume") or 0.0)
    bullish=float(recent[-1]["close"])>float(recent[-1]["open"])
    volume_ok=bool(vfeatures.get("same_time_expansion") or vfeatures.get("expansion_vs_prior_mean") or
                    (vfeatures.get("same_time_contraction") and bullish))
    if not touched or not confirmation or not volume_ok:
        missing=[name for name,ok in (("TRENDLINE_TOUCH",touched),
                                      ("H1_FAILED_DOWNSIDE_REJECTION",failed_downside),
                                      ("H1_LOCAL_LOWER_HIGH_RECLAIM",reclaim),
                                      ("H1_VOLUME_CONFIRMATION",volume_ok)) if not ok]
        return {"status":"REJECT","reason_code":"TRENDLINE_PULLBACK_CONFIRMATION_INCOMPLETE",
        "missing_evidence":missing,"h1_relative_volume":relative,"volume_features":vfeatures,"regime":regime,"trendline":line.as_dict()}
    entry=float(entry_bar["open"])
    recent_low=min(float(b["low"]) for b in recent)
    stop=min(rising-line.tolerance,recent_low)-config.stop_buffer_atr*daily_atr
    if stop>=entry:
        return {"status":"REJECT","reason_code":"INVALID_TRENDLINE_STRUCTURAL_STOP","regime":regime,"trendline":line.as_dict()}
    current_h1_close=float(recent[-1]["close"])
    nearest_target=min((float(t["price"]) for t in target_candidates if t.get("classification")=="DOMINANT"),default=float("inf"))
    frictions=[z for z in friction_candidates if current_h1_close<float(z["lower_bound"])<=nearest_target and
               float(z["lower_bound"])-current_h1_close<=config.friction_distance_atr*daily_atr]
    uncleared=[z for z in frictions if current_h1_close<=float(z["upper_bound"])+config.clearance_buffer_atr*daily_atr or
               entry<=float(z["upper_bound"])+config.clearance_buffer_atr*daily_atr]
    if uncleared:
        return {"status":"REJECT","reason_code":"OVERHEAD_STRUCTURAL_FRICTION_REQUIRES_H1_CLEARANCE",
            "friction_zone_ids":[z.get("zone_id") for z in uncleared],"regime":regime,"trendline":line.as_dict()}
    # The first structurally dominant, already-ranked horizontal/weekly
    # obstacle is selected; H1 local highs are deliberately not candidates.
    targets=[t for t in target_candidates if float(t["price"])>entry and
             str(t.get("classification","DOMINANT"))=="DOMINANT" and str(t.get("timeframe")) in {"DAILY","WEEKLY","CHANNEL"}]
    if not targets:
        return {"status":"REJECT","reason_code":"NO_DOMINANT_STRUCTURAL_TARGET","regime":regime,"trendline":line.as_dict()}
    target_row=min(targets,key=lambda t:(float(t["price"]),0 if t.get("timeframe")=="CHANNEL" else 1,str(t.get("target_id",""))))
    target=float(target_row["price"]); rr=(target-entry)/(entry-stop)
    if rr<config.minimum_structural_rr:
        return {"status":"REJECT","reason_code":"STRUCTURAL_RR_BELOW_MINIMUM","planned_rr":rr,
                "regime":regime,"trendline":line.as_dict(),"target":dict(target_row)}
    e50_state=regime["ema"]["ema50"]["state"]; e200_state=regime["ema"]["ema200"]["state"]
    if e50_state=="SUPPORTIVE" and e200_state=="SUPPORTIVE": confluence="DUAL_EMA_CONFLUENCE"
    elif e50_state=="SUPPORTIVE": confluence="EMA50_SUPPORTIVE"
    elif e200_state=="SUPPORTIVE": confluence="EMA200_SUPPORTIVE"
    elif e50_state=="FRICTION" or e200_state=="FRICTION": confluence="EMA_FRICTION"
    elif e50_state=="RECLAIM_IN_PROGRESS" or e200_state=="RECLAIM_IN_PROGRESS": confluence="EMA_RECLAIM"
    else: confluence="NONE"
    return {"status":"ENTRY_READY","setup_family":"TRENDLINE_PULLBACK","regime":regime,
        "primary_geometry":"INCLINED","secondary_geometry":"HORIZONTAL","trendline":line.as_dict(),
        "channel_id":line.trendline_id+":CHANNEL","ema50_state":e50_state,
        "ema200_state":e200_state,"confluence_state":confluence,
        "h1_confirmation_type":"FAILED_DOWNSIDE_REJECTION_AND_LOCAL_LOWER_HIGH_RECLAIM",
        "h1_buyer_control":{"failed_downside_rejection":failed_downside,
            "reclaimed_local_lower_high":reclaim,"reclaimed_level":local_lower_high,
            "response_timestamp":str(hourly_prefix[-1]["date"])},
        "volume_state":"EXPANSION" if volume_ok else "NONE",
        "h1_relative_volume":relative,"volume_features":vfeatures,"confirmation_timestamp":str(recent[-1]["date"]),
        "entry_timestamp":str(entry_bar["date"]),"entry":entry,"stop":stop,"target":target,
        "target_type":target_row.get("timeframe"),"planned_rr":rr,"reason_code":"CAUSAL_TRENDLINE_PULLBACK_CONFIRMED"}


def annotate_v2_signal(signal: Mapping[str,Any], daily_prefix: Sequence[Mapping[str,Any]],
                       regime_snapshot: Mapping[str,Any] | None = None) -> dict[str,Any]:
    """Attach V3 causal evidence and gate a saved V2 horizontal ENTRY_READY."""
    signal_date=str(signal.get("setup_signal_date") or signal.get("date"))[:10]
    regime=dict(regime_snapshot) if regime_snapshot is not None else classify_regime(daily_prefix,as_of=signal_date)
    family=str(signal.get("setup_family") or signal.get("setup_type") or "")
    result=signal.get("result") or {}
    reclaim=str((result.get("confirmation_type") or "")).upper().find("RECLAIM")>=0
    volume_info=(result.get("confirmation_features") or {}).get("volume") or {}
    volume_passed=bool(volume_info.get("passed",result.get("volume_confirmation_passed",False)))
    ok,reason=family_allowed(regime["regime"],family,reclaim_confirmed=reclaim,
                            unusually_strong=volume_passed)
    ema=regime.get("ema",{})
    e50=ema.get("ema50",{}).get("state"); e200=ema.get("ema200",{}).get("state")
    if e50=="SUPPORTIVE" and e200=="SUPPORTIVE": confluence="DUAL_EMA_CONFLUENCE"
    elif e50=="SUPPORTIVE": confluence="EMA50_SUPPORTIVE"
    elif e200=="SUPPORTIVE": confluence="EMA200_SUPPORTIVE"
    elif e50=="FRICTION" or e200=="FRICTION": confluence="EMA_FRICTION"
    elif e50=="RECLAIM_IN_PROGRESS" or e200=="RECLAIM_IN_PROGRESS": confluence="EMA_RECLAIM"
    else: confluence="NONE"
    lines=regime.get("trendlines",{})
    plan=signal.get("plan") or {}
    rise=lines.get("rising_support") if isinstance(lines,Mapping) else None
    fall=lines.get("falling_resistance") if isinstance(lines,Mapping) else None
    diag=regime.get("diagnostic_evidence",{})
    return {**dict(signal),"strategy_version":"top_down_swing_v3","regime":regime["regime"],
        "regime_evidence":regime["evidence"],"regime_reason_codes":regime["reason_codes"],
        "regime_details":regime,"previous_regime":regime.get("previous_regime"),
        "regime_start_date":regime.get("regime_start_date"),"regime_transition_reason":regime.get("regime_transition_reason"),
        "primary_geometry":regime["primary_geometry"],"secondary_geometry":regime["secondary_geometry"],
        "horizontal_zone_id":plan.get("daily_support_zone_id") or plan.get("daily_resistance_zone_id"),
        "trendline_id":(rise or fall or {}).get("trendline_id") if isinstance((rise or fall),Mapping) else None,
        "rising_trendline_id":rise.get("trendline_id") if isinstance(rise,Mapping) else None,
        "falling_trendline_id":fall.get("trendline_id") if isinstance(fall,Mapping) else None,
        "trendline_strength":max(int((rise or {}).get("contact_count",0)),int((fall or {}).get("contact_count",0))),
        "channel_id":rise.get("trendline_id") if isinstance(rise,Mapping) else None,
        "range_upper":regime.get("range_upper"),"range_lower":regime.get("range_lower"),
        "range_strength":regime.get("range_strength"),"recent_significant_highs":regime.get("recent_significant_highs"),
        "recent_significant_lows":regime.get("recent_significant_lows"),"swing_sequence":regime.get("swing_sequence"),
        "ema50_value":ema.get("ema50",{}).get("value"),"ema50_slope":ema.get("ema50",{}).get("normalized_slope"),
        "ema200_value":ema.get("ema200",{}).get("value"),"ema200_slope":ema.get("ema200",{}).get("normalized_slope"),
        "price_vs_ema50":diag.get("price_vs_ema50"),"price_vs_ema200":diag.get("price_vs_ema200"),
        "ema50_state":e50,"ema200_state":e200,"confluence_state":confluence,
        "h1_confirmation_type":result.get("confirmation_type"),
        "volume_state":"CONFIRMED" if volume_passed else "NOT_CONFIRMED",
        "dominant_target_type":plan.get("target_timeframe"),"planned_r":plan.get("planned_reward_r"),
        "v3_status":"ENTRY_READY" if ok else "REJECTED","v3_reason":reason}
