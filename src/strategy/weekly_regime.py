"""Causal, explainable weekly NIFTY 50 regime classification."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from statistics import fmean
from typing import Any, Mapping, Sequence

from src.strategy.sr_baseline import find_confirmed_swings
from src.strategy.weekly_selection import completed_weekly_rows


@dataclass(frozen=True)
class WeeklyRegimeConfig:
    minimum_completed_weeks: int = 26
    close_trend_weeks: int = 13
    short_return_weeks: int = 4
    long_return_weeks: int = 13
    slope_weeks: int = 13
    atr_weeks: int = 14
    minimum_trend_efficiency: float = 0.15
    bullish_vote_threshold: int = 4
    weak_vote_threshold: int = 4
    minimum_return_4w: float = 0.0
    minimum_return_13w: float = 0.0
    minimum_slope_pct_per_week: float = 0.0
    minimum_directional_move_atr: float = 0.0

    def __post_init__(self) -> None:
        lookbacks=(self.minimum_completed_weeks,self.close_trend_weeks,self.short_return_weeks,
                   self.long_return_weeks,self.slope_weeks,self.atr_weeks)
        if min(lookbacks) < 2:
            raise ValueError("Weekly regime lookbacks must be at least two weeks.")
        if self.minimum_completed_weeks < max(self.close_trend_weeks,self.long_return_weeks+1,self.atr_weeks):
            raise ValueError("minimum_completed_weeks must cover the configured regime lookbacks.")
        if not 1 <= self.bullish_vote_threshold <= 6 or not 1 <= self.weak_vote_threshold <= 6:
            raise ValueError("Regime vote thresholds must be between one and six.")
        if min(self.minimum_trend_efficiency,self.minimum_return_4w,self.minimum_return_13w,
               self.minimum_slope_pct_per_week,self.minimum_directional_move_atr) < 0:
            raise ValueError("Weekly regime thresholds must not be negative.")


def _weekly_return(rows: Sequence[Mapping[str,Any]], periods: int) -> float | None:
    if len(rows) <= periods:
        return None
    initial=float(rows[-periods-1]["close"])
    return float(rows[-1]["close"])/initial-1.0 if initial else None


def _weekly_atr(rows: Sequence[Mapping[str,Any]], periods: int) -> float | None:
    if len(rows) < periods:
        return None
    sample=rows[-periods:]
    offset=len(rows)-len(sample)
    true_ranges=[]
    for j,row in enumerate(sample):
        i=offset+j
        previous=float(rows[i-1]["close"]) if i else float(row["close"])
        high,low=float(row["high"]),float(row["low"])
        true_ranges.append(max(high-low,abs(high-previous),abs(low-previous)))
    return fmean(true_ranges)


def _slope_pct_per_week(rows: Sequence[Mapping[str,Any]], periods: int) -> float | None:
    values=[float(row["close"]) for row in rows[-periods:]]
    if len(values)<2 or values[0]==0:
        return None
    mean_x=(len(values)-1)/2
    mean_y=fmean(values)
    denominator=sum((i-mean_x)**2 for i in range(len(values)))
    slope=sum((i-mean_x)*(value-mean_y) for i,value in enumerate(values))/denominator
    return slope/values[0]


def weekly_regime(*, nifty_daily: Sequence[Mapping[str,Any]], as_of: date,
                  config: WeeklyRegimeConfig|None=None) -> dict[str,Any]:
    """Classify the latest completed-week information available on ``as_of``.

    The entire calendar week containing ``as_of`` is excluded, even when the
    input series already contains later sessions from that week.
    """
    cfg=config or WeeklyRegimeConfig()
    rows=completed_weekly_rows(nifty_daily,as_of)
    component: dict[str,Any]={
        "completed_weeks":len(rows),"close":None,"sma_close":None,"close_above_sma":None,
        "return_4w":None,"return_13w":None,"close_slope_pct_per_week":None,
        "higher_high":None,"higher_low":None,"lower_high":None,"lower_low":None,
        "directional_move_4w_atr":None,"trend_efficiency_13w":None,
        "bullish_votes":0,"weak_votes":0,
    }
    last_week=rows[-1]["week_start"] if rows else None
    if len(rows)<cfg.minimum_completed_weeks:
        return {"as_of":as_of.isoformat(),"last_completed_week":last_week,"regime":"NEUTRAL",
                "component_values":component,"reason_codes":["INSUFFICIENT_COMPLETED_WEEK_HISTORY"]}

    closes=[float(row["close"]) for row in rows]
    price=closes[-1]
    sma=fmean(closes[-cfg.close_trend_weeks:])
    return_4w=_weekly_return(rows,cfg.short_return_weeks)
    return_13w=_weekly_return(rows,cfg.long_return_weeks)
    slope=_slope_pct_per_week(rows,cfg.slope_weeks)
    atr=_weekly_atr(rows,cfg.atr_weeks)
    prior=closes[-cfg.short_return_weeks-1]
    directional=(price-prior)/atr if atr else None
    changes=[closes[i]-closes[i-1] for i in range(max(1,len(closes)-13),len(closes))]
    path=sum(abs(value) for value in changes)
    efficiency=abs(sum(changes))/path if path else 0.0

    swings=find_confirmed_swings(rows,left=2,right=2)
    highs=[point.price for point in swings if point.kind=="resistance"]
    lows=[point.price for point in swings if point.kind=="support"]
    higher_high=highs[-1]>highs[-2] if len(highs)>=2 else None
    lower_high=highs[-1]<highs[-2] if len(highs)>=2 else None
    higher_low=lows[-1]>lows[-2] if len(lows)>=2 else None
    lower_low=lows[-1]<lows[-2] if len(lows)>=2 else None

    close_above=price>sma
    bullish_votes=sum((
        close_above,
        return_4w is not None and return_4w>cfg.minimum_return_4w,
        return_13w is not None and return_13w>cfg.minimum_return_13w,
        slope is not None and slope>cfg.minimum_slope_pct_per_week,
        higher_high is True or higher_low is True,
        directional is not None and directional>cfg.minimum_directional_move_atr,
    ))
    weak_votes=sum((
        not close_above,
        return_4w is not None and return_4w < -cfg.minimum_return_4w,
        return_13w is not None and return_13w < -cfg.minimum_return_13w,
        slope is not None and slope < -cfg.minimum_slope_pct_per_week,
        lower_high is True or lower_low is True,
        directional is not None and directional < -cfg.minimum_directional_move_atr,
    ))
    component.update({
        "close":price,"sma_close":sma,"close_above_sma":close_above,
        "return_4w":return_4w,"return_13w":return_13w,
        "close_slope_pct_per_week":slope,"higher_high":higher_high,"higher_low":higher_low,
        "lower_high":lower_high,"lower_low":lower_low,
        "directional_move_4w_atr":directional,"trend_efficiency_13w":efficiency,
        "bullish_votes":bullish_votes,"weak_votes":weak_votes,
    })
    if efficiency<cfg.minimum_trend_efficiency:
        regime="NEUTRAL"; reasons=["WEEKLY_TREND_CHOPPY"]
    elif return_13w is not None and return_13w>cfg.minimum_return_13w and bullish_votes>=cfg.bullish_vote_threshold:
        regime="BULLISH"; reasons=["BULLISH_VOTE_THRESHOLD_MET"]
    elif return_13w is not None and return_13w < -cfg.minimum_return_13w and weak_votes>=cfg.weak_vote_threshold:
        regime="WEAK"; reasons=["WEAK_VOTE_THRESHOLD_MET"]
    else:
        regime="NEUTRAL"; reasons=["BULLISH_AND_WEAK_CRITERIA_UNMET"]
    return {"as_of":as_of.isoformat(),"last_completed_week":last_week,"regime":regime,
            "component_values":component,"reason_codes":reasons}


__all__=["WeeklyRegimeConfig","weekly_regime"]
