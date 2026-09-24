from datetime import date, timedelta

from src.strategy.weekly_regime import WeeklyRegimeConfig, weekly_regime


def _daily_from_weekly(closes):
    rows=[]
    monday=date(2020,1,6)
    previous=float(closes[0])
    for close in closes:
        for offset in range(5):
            day_close=previous+(float(close)-previous)*(offset+1)/5
            day_open=previous+(float(close)-previous)*offset/5
            rows.append({"date":(monday+timedelta(days=offset)).isoformat(),
                         "open":day_open,"high":max(day_open,day_close)+1,
                         "low":min(day_open,day_close)-1,"close":day_close,"volume":1000})
        previous=float(close)
        monday+=timedelta(days=7)
    return rows


def test_regime_uses_only_completed_weeks_as_of_signal_date():
    daily=_daily_from_weekly([100+i*1.5 for i in range(50)])
    as_of=date.fromisoformat(daily[-3]["date"]) # Wednesday of the final calendar week.
    current_monday=as_of-timedelta(days=as_of.weekday())
    result=weekly_regime(nifty_daily=daily,as_of=as_of)
    assert date.fromisoformat(result["last_completed_week"])<current_monday
    assert result["component_values"]["close"]==daily[-6]["close"]


def test_bullish_classification_on_persistent_rising_weekly_trend():
    daily=_daily_from_weekly([100+i*1.5 for i in range(60)])
    result=weekly_regime(nifty_daily=daily,as_of=date.fromisoformat(daily[-3]["date"]))
    assert result["regime"]=="BULLISH"
    assert "BULLISH_VOTE_THRESHOLD_MET" in result["reason_codes"]
    assert result["component_values"]["bullish_votes"]>=4


def test_choppy_flat_market_is_neutral():
    daily=_daily_from_weekly([100+(i%2)*0.1 for i in range(60)])
    result=weekly_regime(nifty_daily=daily,as_of=date.fromisoformat(daily[-3]["date"]))
    assert result["regime"]=="NEUTRAL"
    assert result["reason_codes"]==["WEEKLY_TREND_CHOPPY"]


def test_weak_classification_on_persistent_falling_weekly_trend():
    daily=_daily_from_weekly([300-i*2 for i in range(60)])
    result=weekly_regime(nifty_daily=daily,as_of=date.fromisoformat(daily[-3]["date"]))
    assert result["regime"]=="WEAK"
    assert "WEAK_VOTE_THRESHOLD_MET" in result["reason_codes"]
    assert result["component_values"]["weak_votes"]>=4


def test_regime_output_is_deterministic():
    daily=_daily_from_weekly([100+i*0.8 for i in range(60)])
    as_of=date.fromisoformat(daily[-3]["date"])
    first=weekly_regime(nifty_daily=daily,as_of=as_of,config=WeeklyRegimeConfig())
    assert first==weekly_regime(nifty_daily=daily,as_of=as_of,config=WeeklyRegimeConfig())


def test_future_days_in_signal_week_cannot_change_regime():
    daily=_daily_from_weekly([100+i*1.2 for i in range(60)])
    as_of=date.fromisoformat(daily[-3]["date"])
    prefix=[row for row in daily if row["date"]<=as_of.isoformat()]
    baseline=weekly_regime(nifty_daily=prefix,as_of=as_of)
    with_future_days=weekly_regime(nifty_daily=daily,as_of=as_of)
    assert baseline==with_future_days
