"""Causal daily structural regime, trendline, channel, and EMA context for V3.

All public functions accept an already-censored completed-candle prefix. They do
not inspect or retain later rows; a caller must pass data through D-1 for SOD D.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from itertools import combinations
from math import isfinite
from typing import Any, Mapping, Sequence


REGIMES = {"TREND_UP", "TREND_DOWN", "RANGE", "TRANSITION", "AMBIGUOUS"}


@dataclass(frozen=True)
class Trendline:
    trendline_id: str
    direction: str
    start_index: int
    end_index: int
    start_date: str
    end_date: str
    anchor_prices: tuple[float, float]
    slope_per_session: float
    normalized_slope: float
    projected_level: float
    tolerance: float
    contact_indices: tuple[int, ...]
    contact_count: int
    length_sessions: int
    average_reaction_atr: float
    violations: int
    status: str
    channel_opposite_level: float | None = None
    channel_width: float | None = None
    channel_contact_count: int = 0
    slope_stability: str = "STABLE"
    strength_score: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _f(row: Mapping[str, Any], key: str) -> float:
    value = float(row[key])
    if not isfinite(value):
        raise ValueError(f"Non-finite daily {key}")
    return value


def _atr(rows: Sequence[Mapping[str, Any]], period: int = 14) -> float:
    if not rows:
        return 0.0
    ranges = []
    for i, row in enumerate(rows):
        previous = _f(rows[i - 1], "close") if i else _f(row, "close")
        ranges.append(max(_f(row, "high") - _f(row, "low"),
                          abs(_f(row, "high") - previous), abs(_f(row, "low") - previous)))
    return sum(ranges[-period:]) / min(period, len(ranges))


def confirmed_pivots(rows: Sequence[Mapping[str, Any]], *, side: str,
                     left: int = 3, right: int = 3, min_reaction_atr: float = 0.75) -> list[dict[str, Any]]:
    """Return pivots confirmed in this prefix; right-hand bars are confirmation bars."""
    if side not in {"high", "low"} or left < 1 or right < 1:
        raise ValueError("side must be high/low and pivot spans must be positive")
    key = side
    out = []
    for pivot in range(left, len(rows) - right):
        value = _f(rows[pivot], key)
        neighborhood = [_f(rows[j], key) for j in range(pivot-left, pivot+right+1) if j != pivot]
        if (side == "low" and value > min(neighborhood)) or (side == "high" and value < max(neighborhood)):
            continue
        future = rows[pivot+1:pivot+right+1]
        reaction = (max(_f(x, "high") for x in future) - value if side == "low"
                    else value - min(_f(x, "low") for x in future))
        scale = max(_atr(rows[max(0, pivot-13):pivot+1]), 1e-12)
        if reaction / scale >= min_reaction_atr:
            out.append({"index": pivot, "date": str(rows[pivot]["date"])[:10],
                        "price": value, "reaction_atr": reaction / scale,
                        "confirm_index": pivot + right})
    return out


def _line(rows: Sequence[Mapping[str, Any]], pivots: Sequence[Mapping[str, Any]],
          *, direction: str, tolerance_atr: float = 0.20,
          break_atr: float = 0.35,
          anchor_pair: tuple[Mapping[str,Any],Mapping[str,Any]] | None = None) -> Trendline | None:
    if len(pivots) < 2:
        return None
    # The most recent anchor pair is the active candidate; earlier contacts
    # validate it but never act as future anchors.
    a, b = anchor_pair or (pivots[-2],pivots[-1])
    if b["index"] <= a["index"]:
        return None
    slope = (float(b["price"]) - float(a["price"])) / (b["index"] - a["index"])
    if (direction == "RISING_SUPPORT" and slope <= 0) or (direction == "FALLING_RESISTANCE" and slope >= 0):
        return None
    current_i = len(rows) - 1
    projected = float(a["price"]) + slope * (current_i - int(a["index"]))
    scale = max(_atr(rows[-14:]), abs(_f(rows[-1], "close")) * 0.002, 1e-9)
    tol = max(tolerance_atr * scale, abs(projected) * 0.001)
    contacts = []
    violations = 0
    reactions = []
    for i in range(int(a["index"]), current_i + 1):
        level = float(a["price"]) + slope * (i - int(a["index"]))
        row = rows[i]
        relevant = _f(row, "low") if direction == "RISING_SUPPORT" else _f(row, "high")
        close = _f(row, "close")
        if abs(relevant - level) <= tol:
            if not contacts or i - contacts[-1] >= 2:
                contacts.append(i)
                reactions.append((max(0.0, _f(row, "high")-level) if direction == "RISING_SUPPORT"
                                  else max(0.0, level-_f(row, "low"))) / scale)
        # A wick is not a break; require two consecutive completed closes.
        broken = close < level - break_atr*scale if direction == "RISING_SUPPORT" else close > level + break_atr*scale
        if broken:
            violations += 1
    last_level = float(a["price"]) + slope * (current_i - int(a["index"]))
    prior_break = False
    for i in range(int(a["index"]), current_i):
        level = float(a["price"]) + slope * (i - int(a["index"]))
        bad = (_f(rows[i], "close") < level - break_atr*scale if direction == "RISING_SUPPORT"
               else _f(rows[i], "close") > level + break_atr*scale)
        next_level = float(a["price"]) + slope * (i+1-int(a["index"]))
        next_bad = (_f(rows[i+1], "close") < next_level - break_atr*scale if direction == "RISING_SUPPORT"
                    else _f(rows[i+1], "close") > next_level + break_atr*scale)
        prior_break |= bad and next_bad
    slope_stability="STEEP_LESS_STABLE" if abs(slope/scale)>0.5 else "STABLE"
    status = "BROKEN" if prior_break else "PRELIMINARY" if len(contacts) < 3 or slope_stability!="STABLE" else "ESTABLISHED" if len(contacts) == 3 else "STRONG"
    # Parallel channel is measured from same-direction opposite pivots. Use
    # median offset to avoid one extreme tail dominating channel width.
    opposite_key = "high" if direction == "RISING_SUPPORT" else "low"
    offsets = []
    for i, row in enumerate(rows[int(a["index"]):], start=int(a["index"])):
        offsets.append((_f(row, opposite_key) - (float(a["price"])+slope*(i-int(a["index"]))))
                       if direction == "RISING_SUPPORT" else
                       ((float(a["price"])+slope*(i-int(a["index"]))) - _f(row, opposite_key)))
    positive = sorted(x for x in offsets if x > tol)
    width = positive[len(positive)//2] if positive else None
    channel_level = last_level + width if width is not None and direction == "RISING_SUPPORT" else (
        last_level - width if width is not None else None)
    channel_contacts = sum(abs(x-width) <= tol for x in offsets) if width is not None else 0
    score=(min(len(contacts),8)*2.0+min(sum(reactions)/len(reactions) if reactions else 0,3.0)+
           min((current_i-int(a["index"])+1)/30,4.0)-min(abs(slope/scale),2.0))
    return Trendline(f"{direction}:{a['date']}:{b['date']}", direction, int(a["index"]), int(b["index"]),
        str(a["date"]), str(b["date"]), (float(a["price"]), float(b["price"])), slope,
        slope / scale, last_level, tol, tuple(contacts), len(contacts), current_i-int(a["index"])+1,
        sum(reactions)/len(reactions) if reactions else 0.0, violations, status, channel_level,
        width, channel_contacts,slope_stability,score)


def trendlines(rows: Sequence[Mapping[str, Any]], *, left: int = 3, right: int = 3,
               min_reaction_atr: float = 0.75) -> dict[str, Trendline | None]:
    lows = confirmed_pivots(rows, side="low", left=left, right=right, min_reaction_atr=min_reaction_atr)
    highs = confirmed_pivots(rows, side="high", left=left, right=right, min_reaction_atr=min_reaction_atr)
    def select(pivots:Sequence[Mapping[str,Any]],direction:str)->Trendline|None:
        candidates=[]
        recent=list(pivots[-6:])
        for pair in combinations(recent,2):
            candidate=_line(rows,pivots,direction=direction,anchor_pair=pair)
            if candidate and candidate.length_sessions>=10 and candidate.contact_count>=2:
                candidates.append(candidate)
        if not candidates:return None
        return max(candidates,key=lambda x:(x.strength_score,x.status=="BROKEN",x.contact_count,
            x.length_sessions,-abs(x.normalized_slope),x.end_index,x.trendline_id))
    return {"rising_support": select(lows,"RISING_SUPPORT"),
            "falling_resistance": select(highs,"FALLING_RESISTANCE")}


def ema_context(rows: Sequence[Mapping[str, Any]], *, periods: tuple[int, int] = (50, 200),
                lookback: int = 20, touch_atr: float = 0.35,
                interaction_rows: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Compute EMAs from full causal history; limit interaction counts to recent rows.

    `rows` must end at D-1. `interaction_rows` is the recent regime window and
    affects only EMA interaction/state counts, never the EMA values or slopes.
    """
    closes = [_f(r, "close") for r in rows]
    interactions = list(interaction_rows if interaction_rows is not None else rows[-lookback:])
    result: dict[str, Any] = {}
    atr = max(_atr(interactions[-14:]), 1e-12)
    for period in periods:
        values: list[float | None] = []
        if len(rows) >= period:
            alpha=2.0/(period+1)
            value=sum(closes[:period])/period
            series=[None]*(period-1)+[value]
            for close in closes[period:]:
                value=alpha*close+(1-alpha)*value
                series.append(value)
            values=series
        if not values:
            value, slope = None, None
        else:
            value = float(values[-1])
            old_value=values[max(0, len(values)-1-min(lookback, len(values)-1))]
            old = float(old_value) if old_value is not None else value
            slope = (value-old)/max(1,min(lookback,len(values)-1)) / atr if len(values)>1 else 0.0
        tail = interactions[-lookback:]
        indexed=[]
        offset=len(rows)-len(tail)
        for j,row in enumerate(tail):
            idx=offset+j
            indexed.append((idx,row,values[idx] if idx<len(values) else None))
        touch_count = sum(value is not None and _f(r,"low")-touch_atr*atr <= float(value) <= _f(r,"high")+touch_atr*atr
                          for _,r,value in indexed)
        above = sum(value is not None and _f(r,"close") > float(value) for _,r,value in indexed)
        recent_above = above >= max(1, len(tail)*0.65)
        recently_below = above <= len(tail)*0.35
        result[f"ema{period}"] = {"value": value, "normalized_slope": slope,
            "distance_atr": (_f(rows[-1],"close")-value)/atr if value is not None else None,
            "valid": value is not None,
            "price_vs_ema": ("ABOVE" if _f(rows[-1],"close")>value else "BELOW") if value is not None else "UNAVAILABLE",
            "touch_count": touch_count, "accepted_above": recent_above,
            "accepted_below": recently_below,
            "state": ("SUPPORTIVE" if slope is not None and slope>0 and recent_above and touch_count>=2
                      else "FRICTION" if slope is not None and slope<0 and recently_below and touch_count>=1
                      else "RECLAIM_IN_PROGRESS" if value is not None and _f(rows[-1],"close")>value and recently_below
                      else "NONE")}
    return result


def ema_structural_state(rows: Sequence[Mapping[str, Any]], ema_values: Mapping[str, Any],
                         *, hover_lookback: int = 6) -> str:
    """Summarize causal EMA interactions; EMAs remain context, never a trigger alone."""
    if hover_lookback < 2:
        raise ValueError("hover_lookback must be at least two sessions")
    e50=ema_values.get("ema50",{}); e200=ema_values.get("ema200",{})
    if not rows:
        return "NONE"
    close=_f(rows[-1],"close")
    previous_close=_f(rows[-2],"close") if len(rows)>1 else close
    atr=max(_atr(rows[-14:]),1e-12)
    def prior_ema(period:int)->float|None:
        prefix=rows[:-1]
        if len(prefix)<period:return None
        alpha=2.0/(period+1); value=sum(_f(r,"close") for r in prefix[:period])/period
        for r in prefix[period:]:value=alpha*_f(r,"close")+(1-alpha)*value
        return value
    cross_down=[]; cross_up=[]
    for period,current in ((50,e50),(200,e200)):
        prev=prior_ema(period); now=current.get("value")
        if prev is not None and now is not None:
            cross_down.append(previous_close>=prev and close<now)
            cross_up.append(previous_close<=prev and close>now)
    if any(cross_down):return "EMA_BREAKDOWN"
    if any(cross_up):return "EMA_RECLAIM"
    if e50.get("valid") and close>float(e50["value"]) and e50.get("touch_count",0)>=3:
        tail=rows[-(hover_lookback+1):-1]
        if len(tail)>=hover_lookback and close>max(_f(r,"high") for r in tail):
            return "EMA_HOVER_BREAKOUT"
    support50=e50.get("valid") and float(e50.get("normalized_slope") or 0)>0 and bool(e50.get("accepted_above")) and int(e50.get("touch_count",0))>=2
    support200=e200.get("valid") and float(e200.get("normalized_slope") or 0)>0 and bool(e200.get("accepted_above")) and int(e200.get("touch_count",0))>=2
    if support50 and support200:return "DUAL_EMA_SUPPORT"
    if support50:return "EMA50_SUPPORT"
    if support200:return "EMA200_SUPPORT"
    for value in (e50,e200):
        dist=value.get("distance_atr"); slope=value.get("normalized_slope")
        if value.get("valid") and dist is not None and slope is not None and -0.5<=float(dist)<0 and float(slope)<0 and int(value.get("touch_count",0))>=1:
            return "EMA_FRICTION"
    if any(value.get("state")=="RECLAIM_IN_PROGRESS" for value in (e50,e200)):
        return "EMA_RECLAIM"
    return "NONE"


def classify_regime(rows: Sequence[Mapping[str, Any]], *, as_of: str | None = None,
                    left: int = 3, right: int = 3, min_reaction_atr: float = 0.75,
                    previous_state: Mapping[str, Any] | None = None,
                    horizontal_zones: Sequence[Mapping[str, Any]] = ()) -> dict[str, Any]:
    """Causally classify current geometry, using recent context and hysteresis.

    Supply only completed candles and call in date order with the prior result
    to preserve state. Older horizontal market memory is an explicit secondary
    input and cannot dominate current 45-60 session geometry.
    """
    if not rows:
        return {"regime":"AMBIGUOUS","reason_codes":["NO_COMPLETED_DAILY_HISTORY"],"as_of":as_of,
                "primary_geometry":"NONE","secondary_geometry":"NONE","trendlines":{},"ema":{}}
    if as_of and any(str(r["date"])[:10] >= as_of[:10] for r in rows):
        raise ValueError("Regime input must end strictly before decision date")
    context=rows[-60:]
    highs=confirmed_pivots(context,side="high",left=left,right=right,min_reaction_atr=min_reaction_atr)
    lows=confirmed_pivots(context,side="low",left=left,right=right,min_reaction_atr=min_reaction_atr)
    up=(len(highs)>=2 and len(lows)>=2 and highs[-1]["price"]>highs[-2]["price"] and lows[-1]["price"]>lows[-2]["price"])
    down=(len(highs)>=2 and len(lows)>=2 and highs[-1]["price"]<highs[-2]["price"] and lows[-1]["price"]<lows[-2]["price"])
    lines=trendlines(context,left=left,right=right,min_reaction_atr=min_reaction_atr)
    ema_state=ema_context(rows, interaction_rows=context)
    ema_state["structural_state"]=ema_structural_state(rows,ema_state)
    rising,falling=lines["rising_support"],lines["falling_resistance"]
    up_line=bool(rising and rising.status in {"ESTABLISHED","STRONG"})
    down_line=bool(falling and falling.status in {"ESTABLISHED","STRONG"})
    e50,e200=ema_state["ema50"],ema_state["ema200"]
    up_ema=bool(e50["normalized_slope"] is not None and e50["normalized_slope"]>0 and
                (e200["normalized_slope"] is None or e200["normalized_slope"]>=-0.15))
    down_ema=bool(e50["normalized_slope"] is not None and e50["normalized_slope"]<0 and
                  (e200["normalized_slope"] is None or e200["normalized_slope"]<=0.15))
    close=_f(context[-1],"close"); tail=context[-20:]
    hi=max(_f(r,"high") for r in tail); lo=min(_f(r,"low") for r in tail)
    overlap=(hi-lo)/max(close,1e-12)>=0.04 and (hi-lo)/max(close,1e-12)<0.12
    near_boundary=sum(abs(_f(r,"high")-hi)/max(close,1e-12)<0.015 or abs(_f(r,"low")-lo)/max(close,1e-12)<0.015 for r in tail)
    drift=(close-_f(tail[0],"close"))/max(abs(close)*max(1,len(tail)-1),1e-12)
    recent_highs=[x for x in highs if x["index"]>=len(context)-45]
    recent_lows=[x for x in lows if x["index"]>=len(context)-45]
    similar_highs=len(recent_highs)>=2 and abs(recent_highs[-1]["price"]-recent_highs[-2]["price"])/max(close,1e-12)<=0.025
    similar_lows=len(recent_lows)>=2 and abs(recent_lows[-1]["price"]-recent_lows[-2]["price"])/max(close,1e-12)<=0.025
    zone_strength=sum(1 for z in horizontal_zones if int(z.get("reactions",z.get("touch_count",0)) or 0)>=2)
    horizontal_strength=min(1.0,(near_boundary/8+int(similar_highs)+int(similar_lows)+min(zone_strength,2)/2)/4)
    flat_ema=e50["normalized_slope"] is not None and abs(e50["normalized_slope"])<0.10
    crossed_ema=e50["touch_count"]>=3
    horizontal=(overlap and near_boundary>=4 and abs(drift)<0.00075 and similar_highs and similar_lows and
                horizontal_strength>=0.55 and (flat_ema or crossed_ema))
    up_score=int(up)+int(up_line)+int(up_ema)
    down_score=int(down)+int(down_line)+int(down_ema)
    previous_regime=str((previous_state or {}).get("regime") or "")
    previous_start=str((previous_state or {}).get("regime_start_date") or "")
    important_low=float((previous_state or {}).get("latest_important_low") or 0)
    important_high=float((previous_state or {}).get("latest_important_high") or 0)
    atr_value=max(_atr(context[-14:]),1e-12)
    failed_hl=important_low>0 and len(context)>=2 and all(_f(r,"close")<important_low-0.25*atr_value for r in context[-2:])
    failed_lh=important_high>0 and len(context)>=2 and all(_f(r,"close")>important_high+0.25*atr_value for r in context[-2:])
    line_break=bool((rising and rising.status=="BROKEN") or (falling and falling.status=="BROKEN"))
    ema_shift=bool(e50["normalized_slope"] is not None and
        ((previous_regime=="TREND_UP" and e50["normalized_slope"]<0 and e50["accepted_below"]) or
         (previous_regime=="TREND_DOWN" and e50["normalized_slope"]>0 and e50["accepted_above"])))
    changing=line_break or failed_hl or failed_lh or ema_shift
    material_conflict=(up and down_line) or (down and up_line) or (horizontal and (up_score>=2 or down_score>=2))
    raw=("AMBIGUOUS" if material_conflict else "TREND_UP" if up and (up_line or up_ema) else "TREND_DOWN" if down and (down_line or down_ema)
         else "RANGE" if horizontal and up_score<=1 and down_score<=1
         else "AMBIGUOUS" if up_score and down_score else "TRANSITION" if changing else "AMBIGUOUS")
    if previous_regime in {"TREND_UP","TREND_DOWN"}:
        opposite=(previous_regime=="TREND_UP" and raw=="TREND_DOWN") or (previous_regime=="TREND_DOWN" and raw=="TREND_UP")
        if changing or opposite or (raw=="RANGE" and horizontal): regime="TRANSITION"
        elif material_conflict: regime="AMBIGUOUS"
        elif raw=="AMBIGUOUS" and not (up_score and down_score): regime=previous_regime
        else: regime=raw
    elif previous_regime=="TRANSITION" and raw in {"AMBIGUOUS","TRANSITION"}:
        regime="TRANSITION"
    else: regime=raw
    reasons=[]
    if up: reasons.append("HIGHER_HIGH_AND_HIGHER_LOW")
    if down: reasons.append("LOWER_HIGH_AND_LOWER_LOW")
    if up_line: reasons.append("ESTABLISHED_RISING_SUPPORT")
    if down_line: reasons.append("ESTABLISHED_FALLING_RESISTANCE")
    if up_ema: reasons.append("EMA_CONTEXT_SUPPORTIVE")
    if down_ema: reasons.append("EMA_CONTEXT_BEARISH")
    if horizontal: reasons.append("REPEATED_HORIZONTAL_SWINGS_AND_RANGE_WIDTH")
    if material_conflict: reasons.append("CONFLICTING_DIRECTIONAL_AND_GEOMETRY_EVIDENCE")
    if flat_ema: reasons.append("FLAT_EMA50_SUPPORTS_RANGE_CONTEXT")
    if crossed_ema: reasons.append("REPEATED_EMA50_CROSSINGS")
    if changing: reasons.append("STRUCTURAL_BREAK_OR_ACCEPTANCE_CHANGE")
    if regime=="TREND_UP": primary,secondary="INCLINED",("HORIZONTAL" if horizontal_zones or horizontal else "NONE")
    elif regime=="TREND_DOWN": primary,secondary="INCLINED",("HORIZONTAL" if horizontal_zones or horizontal else "NONE")
    elif regime=="RANGE": primary,secondary="HORIZONTAL","NONE"
    elif regime=="TRANSITION": primary,secondary="MIXED","INCLINED" if rising or falling else "HORIZONTAL"
    else: primary,secondary="NONE","MIXED" if rising or falling or horizontal else "NONE"
    regime_start=previous_start if regime==previous_regime and previous_start else str(context[-1]["date"])[:10]
    seq=[]
    for i in range(1,min(len(highs),len(lows))):
        seq.append({"high":"HH" if highs[i]["price"]>highs[i-1]["price"] else "LH" if highs[i]["price"]<highs[i-1]["price"] else "EH",
                    "low":"HL" if lows[i]["price"]>lows[i-1]["price"] else "LL" if lows[i]["price"]<lows[i-1]["price"] else "EL",
                    "high_date":highs[i]["date"],"low_date":lows[i]["date"]})
    accelerating=False
    if len(lows)>=3:
        a=(lows[-2]["price"]-lows[-3]["price"])/max(1,lows[-2]["index"]-lows[-3]["index"])
        b=(lows[-1]["price"]-lows[-2]["price"])/max(1,lows[-1]["index"]-lows[-2]["index"])
        accelerating=a>0 and b>2*a
    return {"as_of":as_of or str(rows[-1]["date"])[:10],"input_through":str(rows[-1]["date"])[:10],
        "regime":regime,"candidate_regime":raw,"previous_regime":previous_regime or None,
        "regime_start_date":regime_start,"candidate_confidence":max(up_score,down_score,horizontal_strength),
        "regime_transition_reason":";".join(reasons) if regime!=previous_regime else None,
        "acceleration_state":"ACCELERATING_TREND" if accelerating else "NONE",
        "recent_significant_highs":highs[-5:],"recent_significant_lows":lows[-5:],"swing_sequence":seq,
        "range_upper":hi if horizontal else None,"range_lower":lo if horizontal else None,
        "range_strength":horizontal_strength if horizontal else 0.0,
        "reason_codes":reasons or ["INSUFFICIENT_OR_CONFLICTING_STRUCTURAL_EVIDENCE"],
        "evidence":{"higher_high_higher_low":up,"lower_high_lower_low":down,"horizontal_range":horizontal,
            "up_score":up_score,"down_score":down_score,"ema50_state":e50["state"],"ema200_state":e200["state"]},
        "diagnostic_evidence":{"bullish_swing_evidence":int(up),"bearish_swing_evidence":int(down),
            "horizontal_range_evidence":horizontal_strength,"rising_trendline_strength":rising.contact_count if rising else 0,
            "falling_trendline_strength":falling.contact_count if falling else 0,
            "ema50_slope_state":"RISING" if e50["normalized_slope"] is not None and e50["normalized_slope"]>0 else "FLAT_OR_FALLING",
            "ema200_slope_state":"RISING" if e200["normalized_slope"] is not None and e200["normalized_slope"]>0 else "FLAT_OR_FALLING",
            "price_vs_ema50":"ABOVE" if e50["distance_atr"] is not None and e50["distance_atr"]>0 else "BELOW_OR_UNAVAILABLE",
            "price_vs_ema200":"ABOVE" if e200["distance_atr"] is not None and e200["distance_atr"]>0 else "BELOW_OR_UNAVAILABLE",
            "horizontal_structure_strength":horizontal_strength,"recent_structure_break":changing,
            "weekly_context":"CONTEXT_ONLY"},
        "latest_important_low":float(lows[-1]["price"]) if lows else None,
        "latest_important_high":float(highs[-1]["price"]) if highs else None,
        "primary_geometry":primary,"secondary_geometry":secondary,
        "trendlines":{k:(v.as_dict() if v else None) for k,v in lines.items()},"ema":ema_state}


def classify_regime_series(rows: Sequence[Mapping[str, Any]], *, start_index: int = 1,
                           left: int = 3, right: int = 3,
                           min_reaction_atr: float = 0.75,
                           candidate_confirmation_sessions: int = 2) -> list[dict[str, Any]]:
    """Advance the regime state machine one decision date at a time.

    Each row's decision date is that candle's date and its information set ends
    on the preceding completed candle. Earlier output rows are immutable and
    never revised when later confirmation arrives.
    """
    if candidate_confirmation_sessions < 1:
        raise ValueError("candidate_confirmation_sessions must be positive")
    output=[]; prior: Mapping[str,Any] | None=None
    candidate_name: str | None=None; candidate_count=0
    for i in range(max(1,start_index),len(rows)):
        decision=str(rows[i]["date"])[:10]
        state=classify_regime(rows[:i],as_of=decision,left=left,right=right,
                              min_reaction_atr=min_reaction_atr,previous_state=prior)
        raw_candidate=str(state.get("candidate_regime") or state["regime"])
        previous_regime=str(prior.get("regime")) if prior else ""
        stable={"TREND_UP","TREND_DOWN","RANGE"}
        if raw_candidate in stable and raw_candidate!=previous_regime:
            if candidate_name==raw_candidate:
                candidate_count+=1
            else:
                candidate_name=raw_candidate; candidate_count=1
        else:
            candidate_name=None; candidate_count=0
        if prior and raw_candidate in stable and raw_candidate!=previous_regime:
            if state["regime"]=="TRANSITION" or previous_regime=="TRANSITION":
                if candidate_count<candidate_confirmation_sessions:
                    state["regime"]="TRANSITION"
                else:
                    state["regime"]=raw_candidate
                    state["regime_start_date"]=decision
                    state["regime_transition_reason"]="CANDIDATE_CONFIRMED_OVER_CONSECUTIVE_SESSIONS"
            elif candidate_count<candidate_confirmation_sessions:
                state["regime"]=previous_regime
            else:
                state["regime_start_date"]=decision
        state["candidate_regime"]=raw_candidate
        state["candidate_strength"]=state.get("candidate_confidence")
        state["candidate_sessions"]=(candidate_count if raw_candidate==candidate_name else 0)
        state["decision_date"]=decision
        output.append(state); prior=state
    return output
