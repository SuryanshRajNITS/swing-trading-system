from copy import deepcopy

from src.backtest.trade_audit import (build_trade_audit, select_deterministic_sample,
                                      summarize_audit)


def _setup(zone_id="Z1"):
    return {"setup_type": "support_test", "signal_close": 10.0,
            "target_r": 2.0, "zone_id": zone_id,
            "zone": {"kind": "support", "lower": 9.0, "center": 10.0,
                     "upper": 11.0},
            "touch_count": 3, "freshness": "FRESH",
            "weekly_selection": {"total_score": 6, "momentum_score": 3,
                                 "relative_strength_4w": .1,
                                 "relative_strength_13w": .2,
                                 "trend_efficiency_13w": .3,
                                 "nearest_support": 8.0,
                                 "nearest_resistance": 14.0}}


def _artifact():
    metadata = _setup()
    stop = 9.0-10.0*.002
    risk = 12.0-stop
    target = 12.0+risk*2
    closed = {"trade_id": "CLOSED-1", "security_id": "S1", "symbol": "AAA",
              "signal_timestamp": "2024-01-02T15:30:00+05:30",
              "entry_timestamp": "2024-01-03T09:15:00+05:30",
              "entry_price": 12.0, "quantity": 2, "initial_stop": stop,
              "initial_r_distance": risk, "exit_timestamp": "2024-01-08T15:30:00+05:30",
              "exit_price": target, "exit_reason": "TARGET", "holding_period": 4,
              "gross_pnl": 12.0, "costs": 1.0, "net_pnl": 11.0,
              "R_multiple": 11.0/(risk*2), "metadata": metadata}
    open_metadata = {**_setup("Z2"), "security_id": "S2",
                     "signal_date": "2024-01-04",
                     "execution_timestamp": "2024-01-05T09:15:00+05:30",
                     "accepted": True, "entry_price": 12.0, "quantity": 1}
    open_position = {"security_id": "S2", "symbol": "BBB",
                     "entry_timestamp": "2024-01-05T09:15:00+05:30",
                     "entry_price": 12.0, "quantity": 1,
                     "initial_stop": stop, "target_price": target, "last_price": 13.0}
    return {
        "research_controls": {"period": ["2023-01-01", "2024-12-31"], "target_r": 2.0},
        "metadata": {"config_snapshot": {"strategy": {"parameters": {
            "daily_strategy": {"support_stop_buffer_pct": .002,
                               "breakout_stop_buffer_pct": .002}}}}},
        "trades": [closed], "open_positions_at_end": [open_position],
        "events": [{"event_type": "POSITION_ENTRY"}, {"event_type": "POSITION_ENTRY"}],
        "diagnostics": [open_metadata],
        "selection_signal_decisions": [{"security_id": "S2", "date": "2024-01-04",
                                         "score": 7, "momentum_score": 4,
                                         "relative_strength_4w": .2,
                                         "relative_strength_13w": .3,
                                         "trend_efficiency_13w": .4,
                                         "nearest_support": 8.0,
                                         "nearest_resistance": 15.0}],
    }


def _daily():
    return {
        "S1": [
            {"date": "2024-01-02", "open": 9.8, "high": 10.5, "low": 9.5,
             "close": 10.0, "volume": 100},
            {"date": "2024-01-03", "open": 12.0, "high": 13.0, "low": 10.5,
             "close": 12.5, "volume": 120},
        ],
        "S2": [
            {"date": "2024-01-04", "open": 10.0, "high": 10.6, "low": 9.4,
             "close": 10.0, "volume": 90},
            {"date": "2024-01-05", "open": 12.0, "high": 13.2, "low": 11.0,
             "close": 13.0, "volume": 150},
        ],
    }


def test_one_audit_row_is_exported_for_each_executed_entry_including_open_positions():
    rows = build_trade_audit(_artifact(), _daily())
    assert len(rows) == 2
    assert {row["record_status"] for row in rows} == {"CLOSED", "OPEN_AT_END"}
    assert summarize_audit(rows)["total_executed_trades"] == 2


def test_audit_uses_actual_entry_fill_and_exports_stop_and_fixed_target():
    row = build_trade_audit(_artifact(), _daily())[0]
    assert row["signal_close"] == 10.0
    assert row["actual_entry_price"] == 12.0
    assert row["stop_price_at_entry"] == 8.98
    assert row["target_type"] == "FIXED_R"
    assert row["target_R"] == 2.0
    assert row["target_price_at_entry"] == 18.04
    assert row["planned_risk_per_share"] == 12.0-8.98
    assert row["planned_reward_per_share"] == 18.04-12.0
    assert row["planned_RR"] == 2.0


def test_signal_and_entry_candles_and_setup_zone_intersection_are_exported():
    row = build_trade_audit(_artifact(), _daily())[0]
    assert (row["signal_day_open"], row["signal_day_high"], row["signal_day_low"],
            row["signal_day_close"], row["signal_day_volume"]) == (9.8, 10.5, 9.5, 10.0, 100)
    assert (row["entry_day_open"], row["entry_day_high"], row["entry_day_low"],
            row["entry_day_close"], row["entry_day_volume"]) == (12.0, 13.0, 10.5, 12.5, 120)
    assert row["entry_day_candle_intersects_setup_zone"] is True
    assert row["would_daily_low_touch_setup_zone_on_entry_day"] is True
    assert row["entry_above_setup_zone"] is True
    assert row["entry_inside_setup_zone"] is False
    assert row["entry_execution_model"] == "NEXT_SESSION_OPEN"


def test_deterministic_sample_and_export_do_not_mutate_saved_inputs():
    artifact = _artifact()
    original = deepcopy(artifact)
    rows = build_trade_audit(artifact, _daily())
    assert artifact == original
    assert select_deterministic_sample(rows) == select_deterministic_sample(rows)


def test_entry_day_candle_without_zone_intersection_is_false():
    daily = _daily()
    daily["S1"][1] = {**daily["S1"][1], "low": 11.2, "high": 13.0}
    row = build_trade_audit(_artifact(), daily)[0]
    assert row["entry_day_candle_intersects_setup_zone"] is False
    assert row["would_daily_low_touch_setup_zone_on_entry_day"] is False
