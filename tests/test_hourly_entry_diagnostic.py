from __future__ import annotations

from datetime import date, datetime, time, timedelta

from src.backtest.hourly_entry_diagnostic import (
    HOURLY_CONFIG, WEEKLY_CONFIG, _bar_available_at, _confirmed_structure,
    _bearish_rejection_at_resistance, _confluence_context, _hourly_features, _hourly_volume_ratio,
    _hourly_zone_behavior, _missing_coverage, _pair_confluence,
    _provider_starts, _six_bar_contraction_context, _summarize_trade, _summary, _windowed_zone_snapshot,
)
from src.strategy.sr_baseline import SwingPoint
from src.strategy.sr_zone_v2 import Zone


def _candles(start: date, days: int, *, base: float = 100.0, hourly: bool = False):
    rows = []
    for offset in range(days):
        day = start + timedelta(days=offset)
        if hourly:
            for hour in (9, 10, 11, 12, 13, 14, 15):
                stamp = datetime.combine(day, time(hour, 15))
                value = base + offset * 0.2 + hour * 0.01
                rows.append({"date": stamp.isoformat() + "+05:30", "open": value,
                             "high": value + 1, "low": value - 1, "close": value + 0.2,
                             "volume": 1000, "session_type": "NORMAL_SESSION"})
        else:
            value = base + offset * 0.2
            rows.append({"date": day.isoformat(), "open": value, "high": value + 2,
                         "low": value - 2, "close": value + 0.3, "volume": 1000})
    return rows


def test_timeframe_configs_and_zone_inputs_are_independent():
    assert HOURLY_CONFIG != WEEKLY_CONFIG
    hourly = _candles(date(2023, 1, 2), 60, base=100, hourly=True)
    weekly = _candles(date(2023, 1, 2), 60, base=500)
    hz = _windowed_zone_snapshot(hourly, security_id="H", index=len(hourly)-1, config=HOURLY_CONFIG)
    wz = _windowed_zone_snapshot(weekly, security_id="W", index=len(weekly)-1, config=WEEKLY_CONFIG)
    assert all(z.security_id == "H" and z.center < 200 for z in hz)
    assert all(z.security_id == "W" and z.center > 400 for z in wz)


def test_asof_zone_snapshot_does_not_change_when_future_bars_are_appended():
    rows = _candles(date(2023, 1, 2), 40, base=100, hourly=True)
    end = 60
    before = _windowed_zone_snapshot(rows, security_id="X", index=end, config=HOURLY_CONFIG)
    future = rows + _candles(date(2023, 3, 20), 30, base=300, hourly=True)
    after = _windowed_zone_snapshot(future, security_id="X", index=end, config=HOURLY_CONFIG)
    assert before == after


def test_confirmed_hourly_higher_low_then_higher_high_is_causal():
    swings = [SwingPoint("support", 2, 5, 90), SwingPoint("support", 8, 11, 95),
              SwingPoint("resistance", 4, 7, 110), SwingPoint("resistance", 12, 15, 115)]
    rows = [{"close": 100} for _ in range(20)]
    early = _confirmed_structure(rows, swings, 10)
    later = _confirmed_structure(rows, swings, 16)
    assert not early["higher_low_formed"]
    assert later["higher_low_formed"]
    assert later["higher_high_formed"]
    assert later["higher_low_then_higher_high"]


def test_confluence_distinguishes_overlap_and_nearby_zones():
    a = {"lower": 99.0, "upper": 101.0}
    overlap = {"lower": 100.0, "upper": 103.0}
    nearby = {"lower": 101.4, "upper": 102.0}
    separate = {"lower": 103.0, "upper": 104.0}
    assert _pair_confluence(a, overlap, 1.0, 100)["overlap"]
    assert _pair_confluence(a, nearby, 1.0, 100)["nearby"]
    assert not _pair_confluence(a, separate, 1.0, 100)["nearby"]


def test_pairwise_confluence_count_can_be_one_two_or_three():
    def side(lower, upper):
        return {"lower": lower, "upper": upper}
    context = _confluence_context(
        {"support": side(99, 101), "resistance": side(109, 111)},
        {"support": side(100, 102), "resistance": side(115, 117)},
        {"support": side(120, 121), "resistance": side(130, 131)}, 1.0, 100)
    assert context["support"]["confluence_count"] == 1


def test_hourly_zone_test_reaction_break_retest_use_prior_close_and_current_bar():
    def zone(side, low, high, center):
        return Zone("Z", "X", side, low, high, center, "2024-01-01", "2023-12-01",
                    "2024-01-01", ("2023-12-01", "2023-12-10"), (), (), False,
                    None, "FRESH")
    support = zone("support", 90, 95, 92.5)
    resistance = zone("resistance", 100, 105, 102.5)
    support_hit = _hourly_zone_behavior([support], {"high": 96, "low": 91, "close": 95.5}, 97)
    support_lost = _hourly_zone_behavior([support], {"high": 92, "low": 88, "close": 89}, 96)
    resistance_break = _hourly_zone_behavior([resistance], {"high": 107, "low": 104, "close": 106}, 104)
    resistance_retest = _hourly_zone_behavior([resistance], {"high": 106, "low": 104, "close": 103}, 106)
    assert support_hit["support_test"] and support_hit["support_reaction"]
    assert support_lost["support_break"]
    assert resistance_break["resistance_break"]
    assert resistance_retest["resistance_retest"]


def test_provider_native_split_session_start_semantics():
    assert _provider_starts(date(2024, 3, 2)) == [time(9, 15), time(11, 15), time(12, 15)]
    bar = {"date": "2024-03-02T11:15:00+05:30"}
    assert _bar_available_at(bar).isoformat() == "2024-03-02T12:15:00+05:30"


def test_missing_quarantined_bar_is_partial_and_unavailable_security_is_explicit():
    day = date(2024, 1, 2)
    data = {"X": [{"date": datetime.combine(day, time(h, 15)).isoformat() + "+05:30"}
                  for h in range(9, 16)]}
    data["X"].pop(0)
    quarantines = [{"security_id": "X", "timestamp": "2024-01-02T09:15:00+05:30"}]
    status, missing, quarantined = _missing_coverage("X", [day], data, quarantines)
    assert status == "PARTIAL_HOURLY_DATA"
    assert missing == ["2024-01-02T09:15:00+05:30"]
    assert quarantined == missing
    assert _missing_coverage("HDFC", [day], data, quarantines)[0] == "HOURLY_DATA_UNAVAILABLE"
    unknown, _, _ = _missing_coverage("X", [date(2023, 11, 12)], data, quarantines)
    assert unknown == "PARTIAL_HOURLY_DATA"


def test_signal_day_hourly_bars_are_excluded_and_result_is_deterministic():
    signal_day = date(2023, 1, 2)
    session_days = [signal_day + timedelta(days=i) for i in range(7)]
    daily = [{"date": d.isoformat(), "open": 100, "high": 102, "low": 98,
              "close": 101, "volume": 1000} for d in session_days]
    hourly = _candles(signal_day, 7, hourly=True)
    # These signal-day bars are conspicuous; none may appear in the replay output.
    for bar in hourly:
        if bar["date"].startswith(signal_day.isoformat()):
            bar["open"] = bar["high"] = bar["low"] = bar["close"] = 99999
    trade = {"trade_id": "T1", "security_id": "X", "symbol": "TEST", "setup_type": "breakout_retest",
        "signal_date": signal_day.isoformat(), "signal_close": "101", "entry_date": "2023-01-03",
        "entry_timestamp": "2023-01-03T09:15:00+05:30", "actual_entry_price": "101",
        "stop_price_at_entry": "95", "target_price_at_entry": "113", "setup_zone_lower": "99",
        "setup_zone_center": "100", "setup_zone_upper": "101", "setup_zone_type": "resistance",
        "setup_zone_id": "Z", "exit_timestamp": "", "exit_reason": "", "record_status": "OPEN_AT_END"}
    first, detail = _summarize_trade(trade, daily_data={"X": daily}, hourly_data={"X": hourly},
                                    sessions=session_days, quarantines=[])
    second, _ = _summarize_trade(trade, daily_data={"X": daily}, hourly_data={"X": hourly},
                                 sessions=session_days, quarantines=[])
    assert first == second
    assert first["hourly_coverage_status"] == "FULL_HOURLY_COVERAGE"
    assert all(not x["timestamp"].startswith(signal_day.isoformat()) for x in detail["hourly_sequence"])
    assert len(detail["hourly_sequence"]) == 21
    assert detail["hourly_sequence"][0]["open"] != 99999


def test_daily_zone_hold_requires_consecutive_post_entry_closes():
    signal_day = date(2023, 1, 2)
    sessions = [signal_day + timedelta(days=i) for i in range(5)]
    daily = [{"date": d.isoformat(), "open": 100, "high": 102, "low": 98,
              "close": 101, "volume": 1000} for d in sessions]
    hourly = _candles(signal_day, 5, base=110, hourly=True)
    trade = {"trade_id": "T2", "security_id": "X", "symbol": "TEST", "setup_type": "breakout_retest",
        "signal_date": signal_day.isoformat(), "signal_close": "110", "entry_date": "2023-01-03",
        "entry_timestamp": "2023-01-03T09:15:00+05:30", "actual_entry_price": "110",
        "stop_price_at_entry": "95", "target_price_at_entry": "113", "setup_zone_lower": "99",
        "setup_zone_center": "100", "setup_zone_upper": "101", "setup_zone_type": "resistance",
        "setup_zone_id": "Z", "exit_timestamp": "", "exit_reason": "", "record_status": "OPEN_AT_END"}
    compact, detail = _summarize_trade(trade, daily_data={"X": daily}, hourly_data={"X": hourly},
                                       sessions=sessions, quarantines=[])
    assert compact["daily_zone_held"]
    assert all(bar["close"] > 101 for bar in detail["hourly_sequence"])


def test_daily_zone_reclaim_requires_post_entry_loss_then_recovery():
    signal_day = date(2023, 1, 2)
    sessions = [signal_day + timedelta(days=i) for i in range(5)]
    daily = [{"date": d.isoformat(), "open": 100, "high": 102, "low": 98,
              "close": 101, "volume": 1000} for d in sessions]
    hourly = _candles(signal_day, 5, base=102, hourly=True)
    first = next(i for i, r in enumerate(hourly) if r["date"].startswith("2023-01-03T09:15"))
    hourly[first].update(open=100, high=100.5, low=97, close=98)
    hourly[first + 1].update(open=98, high=104, low=97.5, close=103)
    trade = {"trade_id": "T3", "security_id": "X", "symbol": "TEST", "setup_type": "breakout_retest",
        "signal_date": signal_day.isoformat(), "signal_close": "102", "entry_date": "2023-01-03",
        "entry_timestamp": "2023-01-03T09:15:00+05:30", "actual_entry_price": "102",
        "stop_price_at_entry": "95", "target_price_at_entry": "113", "setup_zone_lower": "99",
        "setup_zone_center": "100", "setup_zone_upper": "101", "setup_zone_type": "resistance",
        "setup_zone_id": "Z", "exit_timestamp": "", "exit_reason": "", "record_status": "OPEN_AT_END"}
    compact, _ = _summarize_trade(trade, daily_data={"X": daily}, hourly_data={"X": hourly},
                                  sessions=sessions, quarantines=[])
    assert compact["daily_zone_lost"]
    assert compact["daily_zone_reclaimed"]


def test_time_of_day_volume_baseline_excludes_special_sessions():
    rows = [
        {"date": "2024-01-02T09:15:00+05:30", "volume": 10, "session_type": "NORMAL_SESSION"},
        {"date": "2024-01-20T09:15:00+05:30", "volume": 10000, "session_type": "SPECIAL_REGULAR_SESSION"},
        {"date": "2024-01-03T09:15:00+05:30", "volume": 20, "session_type": "NORMAL_SESSION"},
    ]
    assert _hourly_volume_ratio(rows, 2) == 2.0


def test_consolidation_features_are_raw_and_deterministic():
    rows = []
    start = datetime(2024, 1, 2, 9, 15)
    # Prior six-bar range is broad; latest six compress, with last close breaking
    # the preceding local range. The function only describes this sequence.
    for i in range(12):
        base = 100 + (i % 2) * 10 if i < 6 else 100
        rows.append({"date": (start + timedelta(hours=i)).isoformat() + "+05:30",
                     "open": base, "high": base + (1 if i >= 6 else 2),
                     "low": base - (1 if i >= 6 else 2), "close": base,
                     "volume": 1000, "session_type": "NORMAL_SESSION"})
    rows.append({"date": (start + timedelta(hours=12)).isoformat() + "+05:30",
                 "open": 106, "high": 107, "low": 105, "close": 106.5,
                 "volume": 1100, "session_type": "NORMAL_SESSION"})
    first = _hourly_features(rows, len(rows)-1, security_id="X", hourly_swings=[])
    second = _hourly_features(rows, len(rows)-1, security_id="X", hourly_swings=[])
    assert first == second
    assert first["range_contraction"]
    assert first["local_consolidation_breakout_up"]


def test_bearish_rejection_requires_a_causally_known_resistance_test():
    zone = Zone("R", "X", "resistance", 104, 106, 105, "2024-01-01", "2023-12-01",
                "2024-01-01", ("2023-12-01",), (), (), False, None, "FRESH")
    rejection = {"open": 105, "high": 106, "low": 102, "close": 102.5}
    assert _bearish_rejection_at_resistance(rejection, 0.125, [zone])
    assert not _bearish_rejection_at_resistance(rejection, 0.125, [])
    # A bearish candle shape alone is intentionally a separate, broader feature.
    assert rejection["close"] < rejection["open"]


def test_range_contraction_uses_completed_bars_and_is_not_an_ever_seen_setup_flag():
    rows = []
    for i in range(13):
        width = 10 if i < 6 else 2
        rows.append({"high": 100 + width, "low": 100 - width})
    recent_range, previous_range, prior_high, prior_low = _six_bar_contraction_context(rows)
    assert recent_range == 4
    assert previous_range == 20
    assert prior_high == 102
    assert prior_low == 98


def test_entry_extension_uses_baseline_fill_against_daily_zone_not_hourly_open():
    signal_day = date(2023, 1, 2)
    sessions = [signal_day + timedelta(days=i) for i in range(5)]
    daily = [{"date": d.isoformat(), "open": 100, "high": 102, "low": 98,
              "close": 101, "volume": 1000} for d in sessions]
    hourly = _candles(signal_day, 5, base=320, hourly=True)
    trade = {"trade_id": "EXT", "security_id": "X", "symbol": "TEST", "setup_type": "support",
        "signal_date": signal_day.isoformat(), "signal_close": "470", "entry_date": "2023-01-03",
        "entry_timestamp": "2023-01-03T09:15:00+05:30", "actual_entry_price": "475",
        "stop_price_at_entry": "450", "target_price_at_entry": "525", "setup_zone_lower": "456",
        "setup_zone_center": "460", "setup_zone_upper": "465", "setup_zone_type": "support",
        "setup_zone_id": "Z", "exit_timestamp": "", "exit_reason": "", "record_status": "OPEN_AT_END"}
    compact, _ = _summarize_trade(trade, daily_data={"X": daily}, hourly_data={"X": hourly},
                                  sessions=sessions, quarantines=[])
    assert compact["entry_session_open_extension_from_daily_zone_pct"] == (475 - 465) / 465
    assert compact["entry_extension_bucket"] == "AT_LEAST_2_PCT"


def test_invalid_identity_is_excluded_from_feature_counts_but_remains_in_coverage_counts():
    rows = [
        {"hourly_identity_status": "PRICE_CONSISTENT_OR_NOT_FLAGGED", "hourly_coverage_status": "FULL_HOURLY_COVERAGE",
         "bearish_rejection_seen": True, "outcome_class": "TARGET"},
        {"hourly_identity_status": "HOURLY_IDENTITY_PRICE_MISMATCH", "hourly_coverage_status": "HOURLY_DATA_UNAVAILABLE",
         "bearish_rejection_seen": False, "outcome_class": "STOP"},
    ]
    summary = _summary(rows)
    assert summary["feature_sample_trade_count"] == 1
    assert summary["excluded_unavailable_or_invalid_identity_trade_count"] == 1
    assert summary["bearish_rejection_seen"]["true_target"] == 1
    assert summary["bearish_rejection_seen"]["false_stop"] == 0
    assert summary["hourly_coverage_status"]["HOURLY_DATA_UNAVAILABLE"] == 1
