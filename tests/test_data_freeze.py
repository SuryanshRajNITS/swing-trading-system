from datetime import date

from src.data.data_freeze import _cross_timeframe_statistics, _distribution, _p95


def test_p95_uses_deterministic_nearest_rank():
    assert _p95([4, 1, 3, 2]) == 4
    assert _p95([]) == 0


def test_discrepancy_distribution_uses_absolute_pct_and_strict_thresholds():
    result = _distribution([(0.1, 0.10), (0.2, 0.25), (0.3, 0.50), (0.4, 2.10)])
    assert result["session_count"] == 4
    assert result["median_absolute_difference_rupees"] == 0.25
    assert result["median_absolute_difference_pct"] == 0.375
    assert result["p95_absolute_difference_pct"] == 2.1
    assert result["sessions_over_0.25pct"] == 2
    assert result["sessions_over_0.50pct"] == 1
    assert result["sessions_over_1.00pct"] == 1
    assert result["sessions_over_2.00pct"] == 1


def test_scale_classification_does_not_call_single_ohlc_outlier_persistent_identity_divergence():
    day = date(2024, 1, 2)
    daily = {"SEC-1": [{"date": "2024-01-02", "open": 100, "high": 105,
                        "low": 95, "close": 100}]}
    hourly = {"SEC-1": [
        {"date": "2024-01-02T09:15:00+05:30", "open": 102, "high": 105,
         "low": 96, "close": 99},
        {"date": "2024-01-02T15:15:00+05:30", "open": 99, "high": 104,
         "low": 95, "close": 100},
    ]}
    stats, security_rows, breaches = _cross_timeframe_statistics(daily, hourly, {"SEC-1": "TEST"})
    assert stats["persistent_material_scale_divergence_security_ids"] == []
    assert stats["all_usable_series_classification"] == "PRICE_SCALE_VALIDATED_WITH_PROVIDER_OHLC_VARIANCE"
    assert security_rows[0]["classification"] == "PRICE_SCALE_VALIDATED_WITH_PROVIDER_OHLC_VARIANCE"
    assert stats["by_ohlc_field"]["open"]["sessions_over_1.00pct"] == 1
    assert breaches[0]["date"] == day.isoformat()
