from __future__ import annotations

from src.backtest.sr_diagnostics import _aggregate, _bucket, _false_breakout, _last_zone_touch_index


def test_diagnostic_buckets_are_fixed_and_missing_values_are_explicit():
    edges = [(1, 4, "1-3"), (4, 8, "4-7")]
    assert _bucket(3, edges) == "1-3"
    assert _bucket(4, edges) == "4-7"
    assert _bucket(None, edges) == "missing"


def test_bucket_aggregate_reconciles_gross_net_cost_and_r():
    rows = [
        {"setup_type": "support_test", "bucket": "x", "gross_pnl": 10, "transaction_costs": 2, "net_pnl": 8, "R_multiple": 1},
        {"setup_type": "support_test", "bucket": "x", "gross_pnl": -5, "transaction_costs": 1, "net_pnl": -6, "R_multiple": -1},
    ]
    row = _aggregate(rows, ("setup_type", "bucket"))[0]
    assert row["trade_count"] == 2
    assert row["gross_pnl"] == 5
    assert row["transaction_costs"] == 3
    assert row["net_pnl"] == 2
    assert row["net_expectancy"] == 1


def test_false_breakout_classification_uses_ordered_deterministic_rules():
    target = {"setup_type": "breakout_retest", "exit_reason": "TARGET"}
    assert _false_breakout(target, [], {}) == "successful_continuation"
    immediate = {"setup_type": "breakout_retest", "exit_reason": "STOP", "holding_period_sessions": 2}
    assert _false_breakout(immediate, [], {}) == "immediate_failed_breakout"
    deep = {"setup_type": "breakout_retest", "exit_reason": "STOP", "holding_period_sessions": 6,
            "retest_depth_zone_fraction": .8, "retest_delay_sessions": 3}
    assert _false_breakout(deep, [], {}) == "deep_retest_failure"
    residual = {"setup_type": "breakout_retest", "exit_reason": "STOP", "holding_period_sessions": 6,
                "retest_depth_zone_fraction": .3, "retest_delay_sessions": 3, "MFE_R": .8,
                "entry_date": "2024-01-02", "exit_date": "2024-01-10", "entry": 100,
                "stop_distance_pct": 1}
    assert _false_breakout(residual, [], {}) == "stopped_after_partial_followthrough"


def test_last_zone_touch_is_bounded_to_zone_lifetime_start():
    candles = [
        {"low": 99, "high": 101},  # Same price band before this pivot-defined zone existed.
        {"low": 105, "high": 106},
        {"low": 99, "high": 101},
        {"low": 102, "high": 104},
    ]
    zone = {"lower": 99.5, "upper": 100.5}
    assert _last_zone_touch_index(candles, zone, before=4, start=1) == 2
    assert _last_zone_touch_index(candles, zone, before=2, start=1) is None
