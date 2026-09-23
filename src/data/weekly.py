"""Deterministic weekly candles derived from daily rows, with close-time gating."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import Any, Iterable
from zoneinfo import ZoneInfo

from src.data.market_data import validate_ohlcv


DEFAULT_CLOSE = time(15, 30)


def resample_daily_to_weekly(
    daily_rows: Iterable[dict[str, Any]],
    *,
    trading_sessions: Iterable[date] | None = None,
    timezone_name: str = "Asia/Kolkata",
    session_close: time = DEFAULT_CLOSE,
) -> list[dict[str, Any]]:
    """Aggregate available exchange sessions by Monday-Sunday calendar week.

    The session calendar may contain four, five, or another number of sessions in
    a week. OHLCV is aggregated from observed daily candles only; no rows are
    synthesized for holidays or no-trade dates.
    """
    zone = ZoneInfo(timezone_name)
    rows = validate_ohlcv(daily_rows, timezone_name=timezone_name)
    grouped: dict[date, list[dict[str, Any]]] = {}
    observed_sessions: set[date] = set()
    for row in rows:
        stamp = datetime.fromisoformat(row["date"]).astimezone(zone)
        day = stamp.date()
        week_start = day - timedelta(days=day.weekday())
        grouped.setdefault(week_start, []).append(row)
        observed_sessions.add(day)

    calendar = set(trading_sessions) if trading_sessions is not None else observed_sessions
    weekly: list[dict[str, Any]] = []
    for week_start, candles in sorted(grouped.items()):
        week_end = week_start + timedelta(days=6)
        sessions = [day for day in calendar if week_start <= day <= week_end]
        if not sessions:
            continue
        candles.sort(key=lambda row: row["date"])
        final_session = max(sessions)
        weekly.append({
            "week_start": week_start.isoformat(),
            "week_end": week_end.isoformat(),
            "last_session": final_session.isoformat(),
            "completion_at": datetime.combine(final_session, session_close, tzinfo=zone).isoformat(),
            "open": candles[0]["open"],
            "high": max(row["high"] for row in candles),
            "low": min(row["low"] for row in candles),
            "close": candles[-1]["close"],
            "volume": sum(row["volume"] for row in candles),
        })
    return weekly


def last_completed_weekly_candle(
    as_of_timestamp: datetime,
    weekly_candles: Iterable[dict[str, Any]],
    *,
    timezone_name: str = "Asia/Kolkata",
) -> dict[str, Any] | None:
    """Return only a weekly bar whose final session has closed as of the timestamp."""
    zone = ZoneInfo(timezone_name)
    as_of = as_of_timestamp.replace(tzinfo=zone) if as_of_timestamp.tzinfo is None else as_of_timestamp.astimezone(zone)
    completed = [
        candle for candle in weekly_candles
        if datetime.fromisoformat(candle["completion_at"]).astimezone(zone) <= as_of
    ]
    return max(completed, key=lambda candle: candle["week_start"]) if completed else None
