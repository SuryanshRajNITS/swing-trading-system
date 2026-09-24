from datetime import date

from src.data.hourly_price_integrity import _audit_security_days


def test_daily_aggregate_uses_first_open_extremes_and_last_close_with_strict_tolerance():
    day = date(2024, 1, 2)
    daily = {day: {"open": 100, "high": 105, "low": 98, "close": 103}}
    hourly = {day: [
        {"date": "2024-01-02T09:15:00+05:30", "open": 100, "high": 102, "low": 99, "close": 101},
        {"date": "2024-01-02T15:15:00+05:30", "open": 102, "high": 105, "low": 98, "close": 103.04},
    ]}
    result = _audit_security_days(daily, hourly, tolerance=0.05)
    assert result["days_compared"] == 1
    assert result["matched_days"] == 1
    assert result["mismatched_days"] == 0
    assert result["median_ratio"] == 103.04 / 103


def test_cross_timeframe_audit_reports_off_tolerance_deltas_without_adjustment():
    day = date(2023, 4, 17)
    daily = {day: {"open": 475, "high": 480, "low": 470, "close": 478}}
    hourly = {day: [
        {"date": "2023-04-17T09:15:00+05:30", "open": 324, "high": 326, "low": 320, "close": 322},
        {"date": "2023-04-17T15:15:00+05:30", "open": 324, "high": 325, "low": 321, "close": 325},
    ]}
    result = _audit_security_days(daily, hourly, tolerance=0.05)
    assert result["matched_days"] == 0
    assert result["mismatched_days"] == 1
    assert result["median_ratio"] < 0.7
    assert result["first_mismatch"]["hourly_aggregate"]["open"] == 324
    assert result["first_mismatch"]["deltas_hourly_minus_daily"]["open"] == -151.0
