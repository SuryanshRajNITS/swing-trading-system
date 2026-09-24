"""Causal, transparent weekly stock selection for the sr_zone_v2 research layer."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
import hashlib
import json
from math import sqrt
from statistics import fmean
from typing import Any, Mapping, Sequence

from dataclasses import replace

from src.strategy.sr_baseline import find_confirmed_swings
from src.strategy.sr_zone_v2 import ZoneConfig, detect_zones


@dataclass(frozen=True)
class WeeklySelectionConfig:
    minimum_completed_weeks: int = 26
    short_return_weeks: int = 4
    long_return_weeks: int = 13
    structure_minimum: int = 1
    momentum_minimum: int = 2
    relative_strength_minimum: int = 1
    total_score_minimum: int = 5
    strong_close_location: float = 0.65
    minimum_trend_efficiency: float = 0.15
    nearby_resistance_atr: float = 0.75
    nearby_support_atr: float = 1.5
    resistance_breakout_exception: bool = True

    def __post_init__(self) -> None:
        if min(self.minimum_completed_weeks,self.short_return_weeks,self.long_return_weeks) < 2:
            raise ValueError("Weekly lookbacks must be at least two weeks.")
        if not 0 <= self.strong_close_location <= 1:
            raise ValueError("strong_close_location must be between zero and one.")
        if min(self.minimum_trend_efficiency,self.nearby_resistance_atr,self.nearby_support_atr) < 0:
            raise ValueError("Weekly selection thresholds must not be negative.")


def completed_weekly_rows(daily_rows: Sequence[Mapping[str,Any]], as_of: date) -> list[dict[str,Any]]:
    """Aggregate daily candles but exclude the entire signal-date calendar week."""
    current_monday=date.fromordinal(as_of.toordinal()-as_of.weekday())
    grouped: dict[date,list[Mapping[str,Any]]]={}
    for row in daily_rows:
        day=date.fromisoformat(str(row["date"])[:10])
        monday=date.fromordinal(day.toordinal()-day.weekday())
        if monday < current_monday and day <= as_of:
            grouped.setdefault(monday,[]).append(row)
    result=[]
    for monday,rows in sorted(grouped.items()):
        rows=sorted(rows,key=lambda item:str(item["date"]))
        result.append({"date":str(rows[-1]["date"])[:10],"week_start":monday.isoformat(),
                       "open":float(rows[0]["open"]),"high":max(float(r["high"]) for r in rows),
                       "low":min(float(r["low"]) for r in rows),"close":float(rows[-1]["close"]),
                       "volume":sum(int(r["volume"]) for r in rows)})
    return result


def _weekly_atr(rows: Sequence[Mapping[str,Any]], period: int=14) -> float:
    recent=rows[-period:]
    ranges=[]
    offset=len(rows)-len(recent)
    for j,row in enumerate(recent):
        i=offset+j
        previous=float(rows[i-1]["close"]) if i else float(row["close"])
        high,low=float(row["high"]),float(row["low"])
        ranges.append(max(high-low,abs(high-previous),abs(low-previous)))
    return fmean(ranges) if ranges else 0.0


def _return(rows: Sequence[Mapping[str,Any]], weeks: int) -> float | None:
    if len(rows) <= weeks:
        return None
    base=float(rows[-weeks-1]["close"])
    return float(rows[-1]["close"])/base-1 if base else None


def _slope_pct(rows: Sequence[Mapping[str,Any]], window: int=13) -> float | None:
    values=[float(row["close"]) for row in rows[-window:]]
    if len(values)<2 or values[0]==0:
        return None
    mean_x=(len(values)-1)/2
    mean_y=fmean(values)
    slope=sum((i-mean_x)*(value-mean_y) for i,value in enumerate(values))/sum((i-mean_x)**2 for i in range(len(values)))
    return slope/values[0]


def _structure(rows: Sequence[Mapping[str,Any]]) -> tuple[int,bool|None,bool|None]:
    swings=find_confirmed_swings(rows,left=2,right=2)
    highs=[s.price for s in swings if s.kind=="resistance"]
    lows=[s.price for s in swings if s.kind=="support"]
    higher_high=highs[-1]>highs[-2] if len(highs)>=2 else None
    higher_low=lows[-1]>lows[-2] if len(lows)>=2 else None
    return int(higher_high is True)+int(higher_low is True),higher_high,higher_low


def nearest_weekly_zones(zones: Sequence[Any], price: float, atr: float) -> dict[str, float | None]:
    """Return distances to the nearest confirmed weekly zones around price."""
    supports=[zone for zone in zones if zone.side == "support" and zone.center < price]
    resistances=[zone for zone in zones if zone.side == "resistance" and zone.center > price]
    support=min(supports,key=lambda zone:price-zone.center,default=None)
    resistance=min(resistances,key=lambda zone:zone.center-price,default=None)
    return {
        "nearest_support": support.center if support else None,
        "support_distance_atr": (price-support.center)/atr if support and atr > 0 else None,
        "support_distance_pct": (price-support.center)/price if support and price else None,
        "nearest_resistance": resistance.center if resistance else None,
        "resistance_distance_atr": (resistance.center-price)/atr if resistance and atr > 0 else None,
        "resistance_distance_pct": (resistance.center-price)/price if resistance and price else None,
    }


def weekly_features(*, stock_daily: Sequence[Mapping[str,Any]], benchmark_daily: Sequence[Mapping[str,Any]],
                    as_of: date, security_id: str, config: WeeklySelectionConfig|None=None,
                    zone_config: ZoneConfig|None=None) -> dict[str,Any]:
    cfg=config or WeeklySelectionConfig()
    weeks=completed_weekly_rows(stock_daily,as_of)
    benchmark=completed_weekly_rows(benchmark_daily,as_of)
    if len(weeks)<cfg.minimum_completed_weeks:
        return {"completed_weeks":len(weeks),"features_available":False,"security_id":security_id}
    atr=_weekly_atr(weeks)
    short_ret=_return(weeks,cfg.short_return_weeks)
    long_ret=_return(weeks,cfg.long_return_weeks)
    benchmark_short=_return(benchmark,cfg.short_return_weeks)
    benchmark_long=_return(benchmark,cfg.long_return_weeks)
    rs_short=short_ret-benchmark_short if short_ret is not None and benchmark_short is not None else None
    rs_long=long_ret-benchmark_long if long_ret is not None and benchmark_long is not None else None
    structure_score,higher_high,higher_low=_structure(weeks)
    slope=_slope_pct(weeks)
    last=weeks[-1]
    strong=[]
    for row in weeks[-8:]:
        span=float(row["high"])-float(row["low"])
        close_location=(float(row["close"])-float(row["low"]))/span if span>0 else .5
        strong.append(float(row["close"])>float(row["open"]) and close_location>=cfg.strong_close_location)
    strong_fraction=sum(strong)/len(strong) if strong else 0.0
    directional_atr=(float(last["close"])-float(weeks[-cfg.short_return_weeks-1]["close"]))/atr if atr else None
    close_changes=[float(weeks[i]["close"])-float(weeks[i-1]["close"]) for i in range(max(1,len(weeks)-13),len(weeks))]
    path=sum(abs(x) for x in close_changes)
    efficiency=abs(sum(close_changes))/path if path else 0.0
    zones=detect_zones(weeks,security_id=security_id,
                       config=zone_config or ZoneConfig(swing_left=2,swing_right=2,maximum_age=104),
                       include_lifecycle=False)
    price=float(last["close"])
    zone_distances=nearest_weekly_zones(zones,price,atr)
    support_atr=zone_distances["support_distance_atr"]
    resistance_atr=zone_distances["resistance_distance_atr"]
    structure_score += int(slope is not None and slope>0)
    momentum_score=int(short_ret is not None and short_ret>0)+int(long_ret is not None and long_ret>0)
    momentum_score += int(strong_fraction>=.5)+int(directional_atr is not None and directional_atr>=.5)
    rs_score=int(rs_short is not None and rs_short>0)+int(rs_long is not None and rs_long>0)
    resistance_blocked=resistance_atr is not None and resistance_atr<cfg.nearby_resistance_atr
    zone_score=int(support_atr is not None and support_atr<=cfg.nearby_support_atr)-2*int(resistance_blocked)
    total=structure_score+momentum_score+rs_score+zone_score
    trend_clean=efficiency>=cfg.minimum_trend_efficiency
    if benchmark_short is None or benchmark_long is None:
        status="neutral"; reason="benchmark_history_insufficient"
    elif not trend_clean:
        status="rejected"; reason="weekly_trend_too_choppy"
    elif structure_score<cfg.structure_minimum or momentum_score<cfg.momentum_minimum or rs_score<cfg.relative_strength_minimum:
        status="rejected"; reason="weekly_component_gate_failed"
    elif resistance_blocked:
        status="neutral"; reason="nearby_weekly_resistance_breakout_exception_only"
    elif total<cfg.total_score_minimum:
        status="rejected"; reason="weekly_total_score_below_threshold"
    else:
        status="eligible"; reason="weekly_gates_passed"
    return {"security_id":security_id,"features_available":True,"as_of":as_of.isoformat(),
            "last_completed_week":last["week_start"],"completed_weeks":len(weeks),"price":price,"atr":atr,
            "higher_high":higher_high,"higher_low":higher_low,"weekly_slope_pct_per_week":slope,
            "structure_score":structure_score,"return_4w":short_ret,"return_13w":long_ret,
            "directional_move_4w_atr":directional_atr,"strong_bullish_close_fraction_8w":strong_fraction,
            "momentum_score":momentum_score,"nifty_return_4w":benchmark_short,"nifty_return_13w":benchmark_long,
            "relative_strength_4w":rs_short,"relative_strength_13w":rs_long,"relative_strength_score":rs_score,
            "trend_efficiency_13w":efficiency,"trend_clean":trend_clean,"weekly_volatility_13w":
            (sqrt(fmean(x*x for x in close_changes))/price if close_changes and price else None),
            **zone_distances,
            "resistance_blocked":bool(resistance_blocked),"zone_score":zone_score,"total_score":total,
            "status":status,"reason":reason,"score_components":{"structure":structure_score,
            "momentum":momentum_score,"relative_strength":rs_score,"weekly_zone_context":zone_score}}


class WeeklySelectionStrategy:
    """Filter PIT members before daily V2 logic and persist auditable decisions."""

    name="sr_zone_weekly_selection"
    version="sr_zone_v2_weekly_selection_v1"

    def __init__(self, *, daily_strategy: Any, benchmark_daily: Sequence[Mapping[str,Any]],
                 config: WeeklySelectionConfig|None=None) -> None:
        self.daily_strategy=daily_strategy
        self.benchmark_daily=tuple(benchmark_daily)
        self.config=config or WeeklySelectionConfig()
        self.version="sr_zone_v2_weekly_selection_v1"
        self.name="sr_zone_v2_weekly_selection"
        benchmark_payload=json.dumps(self.benchmark_daily,sort_keys=True,separators=(",",":"),allow_nan=False)
        self.parameters={"weekly_selection":asdict(self.config),"daily_strategy":dict(daily_strategy.parameters),
                         "benchmark":"official_nifty_50_index_daily",
                         "benchmark_sha256":hashlib.sha256(benchmark_payload.encode("utf-8")).hexdigest()}
        self.decisions: list[dict[str,Any]]=[]
        self.signal_decisions: list[dict[str,Any]]=[]
        self.weekly_snapshots: list[dict[str,Any]]=[]
        self._feature_cache: dict[str,tuple[str,dict[str,Any]]]={}

    @property
    def diagnostics(self):
        return self.daily_strategy.diagnostics

    def _features(self, sid: str, rows: Sequence[Mapping[str,Any]], as_of: date) -> dict[str,Any]:
        week_key=date.fromordinal(as_of.toordinal()-as_of.weekday()).isoformat()
        cached=self._feature_cache.get(sid)
        if cached and cached[0]==week_key:
            return cached[1]
        zone_config=self.daily_strategy.zone_config
        if zone_config is not None:
            zone_config=replace(zone_config,swing_left=2,swing_right=2,maximum_age=104)
        result=weekly_features(stock_daily=rows,benchmark_daily=self.benchmark_daily,
                               as_of=as_of,security_id=sid,config=self.config,
                               zone_config=zone_config)
        self._feature_cache[sid]=(week_key,result)
        return result

    def generate_entries(self, context: Any):
        decisions={}
        allowed=[]
        for security in context.eligible:
            sid=security.security_id
            features=self._features(sid,context.daily_history.get(sid,()),context.session_date)
            decisions[sid]=features
            self.decisions.append({"date":context.session_date.isoformat(),"security_id":sid,
                                    **features})
            resistance_only=(features.get("status")=="neutral" and
                             features.get("reason")=="nearby_weekly_resistance_breakout_exception_only")
            if features.get("status")=="eligible" or (
                    resistance_only and features.get("resistance_blocked") and
                    self.config.resistance_breakout_exception):
                allowed.append(security)
        enriched_context=replace(context,eligible=tuple(allowed))
        signals=self.daily_strategy.generate_entries(enriched_context)
        accepted=[]
        for signal in signals:
            feature=decisions.get(signal.security_id,{})
            exception=bool(feature.get("resistance_blocked") and
                           feature.get("status")=="neutral" and
                           feature.get("reason")=="nearby_weekly_resistance_breakout_exception_only" and
                           signal.metadata.get("setup_type")=="breakout_retest" and
                           self.config.resistance_breakout_exception)
            passed=feature.get("status")=="eligible" or exception
            row={"date":context.session_date.isoformat(),"security_id":signal.security_id,
                 "status":feature.get("status","neutral"),"reason":feature.get("reason"),
                 "score":feature.get("total_score"),"resistance_breakout_exception":exception,
                 **{k:v for k,v in feature.items() if k not in {"security_id","status","reason"}}}
            self.signal_decisions.append(row)
            if passed:
                accepted.append(replace(signal,metadata={**dict(signal.metadata),
                                                         "weekly_selection":feature,
                                                         "weekly_selection_exception":exception},
                                        strategy_version=self.version))
            else:
                for diagnostic in reversed(self.daily_strategy.diagnostics):
                    if diagnostic.get("security_id")==signal.security_id and diagnostic.get("signal_date")==context.session_date.isoformat() and diagnostic.get("accepted"):
                        diagnostic.update({"accepted":False,"rejection_reason":"WEEKLY_SELECTION_REJECTED",
                                           "reason_code":"WEEKLY_SELECTION_REJECTED"})
                        break
        # Log one PIT-universe selection snapshot per week or membership change.
        ids=tuple(sorted(security.security_id for security in context.eligible))
        week=date.fromordinal(context.session_date.toordinal()-context.session_date.weekday()).isoformat()
        signature=(week,ids)
        if not self.weekly_snapshots or self.weekly_snapshots[-1].get("signature")!=signature:
            rows=list(decisions.values())
            pass_ids=[sid for sid,item in decisions.items() if item.get("status")=="eligible"]
            self.weekly_snapshots.append({"week_start":week,"as_of":context.session_date.isoformat(),
                                          "pit_universe_count":len(ids),"eligible_count":len(pass_ids),
                                          "retained_fraction":len(pass_ids)/len(ids) if ids else 0.0,
                                          "eligible_security_ids":tuple(sorted(pass_ids)),
                                          "signature":signature})
        return tuple(accepted)

    def generate_exits(self, context: Any):
        return self.daily_strategy.generate_exits(context)


__all__=["WeeklySelectionConfig","WeeklySelectionStrategy","completed_weekly_rows",
         "nearest_weekly_zones","weekly_features"]
