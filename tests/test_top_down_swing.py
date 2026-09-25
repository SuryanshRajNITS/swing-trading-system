from datetime import date, datetime, timedelta
from dataclasses import replace

from src.strategy import top_down_swing as strategy
from src.strategy.sr_zone_v2 import Zone, ZoneConfig
from src.strategy.top_down_swing import (TopDownConfig, _hourly_pattern,
                                          confirm_hourly, evaluate_daily_candidate,
                                          structural_location, structural_plan, volume_features)
from src.backtest.top_down_swing_replay import SetupRegistry, setup_instance_id, stable_sample


def _zone(side, lower, upper, seed="z"):
    return Zone(seed, "SEC", side, lower, upper, (lower + upper) / 2,
                "2023-01-01", "2023-01-01", "2024-01-01", ("2023-01-01", "2023-02-01"),
                (), (), False, None, "FRESH", strength="MEANINGFUL", reaction_count=4)


def _bars(closes, start=date(2023, 1, 2)):
    output = []
    for i, close in enumerate(closes):
        day = start + timedelta(days=i)
        output.append({"date": day.isoformat(), "open": close - .2, "high": close + 1,
                       "low": close - 1, "close": close, "volume": 100 + i})
    return output


def test_configuration_defines_two_daily_setups_and_four_hourly_archetypes():
    cfg = TopDownConfig()
    assert cfg.version == "top_down_swing_v1"
    assert cfg.confirmation_expiry_sessions == 10
    assert cfg.minimum_room_r == 1.5
    assert cfg.daily_zone.swing_left == 3
    assert cfg.hourly_zone.swing_right == 3
    assert cfg.daily_zone.departure_atr == 0.5
    assert cfg.hourly_zone.reaction_horizon == 5
    assert cfg.ema_confluence_distance_atr == 0.25


def test_daily_support_reaction_is_a_rejection_not_a_touch(monkeypatch):
    cfg = TopDownConfig(daily_zone=ZoneConfig(swing_left=1, swing_right=1))
    support = _zone("support", 126, 128, "d-support")
    support = replace(support, first_pivot="2023-01-30")
    monkeypatch.setattr(strategy, "_zones", lambda rows, security_id, config:
                        (support,) if config is cfg.daily_zone else ())
    monkeypatch.setattr(strategy, "_lifecycle_snapshot", lambda zones, rows, config, strategy_config=None: zones)
    monkeypatch.setattr(strategy, "_weekly_location", lambda *args: ("APPROACHING_SUPPORT", [], {}))
    rows = _bars([100 + i for i in range(30)])
    rows[-1].update(open=127, high=131, low=126.5, close=130)
    weeks = [{"date": (date(2023, 1, 2) + timedelta(days=7*i)).isoformat(),
              "open": 100+i, "high": 102+i, "low": 99+i, "close": 101+i, "volume": 1000+i}
             for i in range(30)]
    ready = {"component_values": {"completed_weeks": 30}}
    selection = {"features_available": True, "completed_weeks": 30, "relative_strength_4w": .01,
                 "relative_strength_13w": .02, "status": "eligible"}
    result = evaluate_daily_candidate(security_id="SEC", symbol="TEST", daily_rows=rows,
        weekly_rows=weeks, nifty_weekly_regime=ready, weekly_selection=selection, config=cfg)
    assert result["setup_type"] == "SUPPORT_REACTION"
    assert result["state"] == "WAITING_1H_CONFIRMATION"
    rows[-1].update(open=127, high=129, low=126.5, close=127.5)
    weak = evaluate_daily_candidate(security_id="SEC", symbol="TEST", daily_rows=rows,
        weekly_rows=weeks, nifty_weekly_regime=ready, weekly_selection=selection, config=cfg)
    assert weak["setup_type"] is None


def test_daily_breakout_retest_requires_prior_breakout_close(monkeypatch):
    cfg = TopDownConfig()
    resistance = _zone("resistance", 129, 131, "d-resistance")
    monkeypatch.setattr(strategy, "_zones", lambda rows, security_id, config:
                        (resistance,) if config is cfg.daily_zone else ())
    monkeypatch.setattr(strategy, "_weekly_location", lambda *args: ("APPROACHING_SUPPORT", [], {}))
    monkeypatch.setattr(strategy, "_lifecycle_snapshot", lambda zones, rows, config, strategy_config=None: zones)
    resistance = Zone(resistance.zone_id, resistance.security_id, "support", resistance.lower_bound,
        resistance.upper_bound, resistance.center, resistance.creation_date, resistance.first_pivot,
        resistance.last_update, resistance.pivots, resistance.touch_episodes, resistance.reactions,
        True, "resistance_to_support_candidate", resistance.freshness, original_side="resistance",
        role_state="SUPPORT_CONFIRMED", strength="MEANINGFUL", reaction_count=4)
    monkeypatch.setattr(strategy, "_zones", lambda rows, security_id, config:
                        (resistance,) if config is cfg.daily_zone else ())
    rows = _bars([100 + i for i in range(30)])
    rows[-2].update(close=132)
    rows[-1].update(open=130, high=136, low=129.5, close=134)
    weeks = [{"date": (date(2023, 1, 2) + timedelta(days=7*i)).isoformat(),
              "open": 100+i, "high": 102+i, "low": 99+i, "close": 101+i, "volume": 1000+i}
             for i in range(30)]
    result = evaluate_daily_candidate(security_id="SEC", symbol="TEST", daily_rows=rows,
        weekly_rows=weeks, nifty_weekly_regime={"component_values": {"completed_weeks": 30}},
        weekly_selection={"features_available": True, "completed_weeks": 30,
          "relative_strength_4w": .01, "relative_strength_13w": .02, "status": "eligible"}, config=cfg)
    assert result["setup_type"] == "BREAKOUT_RETEST"


def test_weekly_midrange_is_not_actionable_and_nifty_regime_is_soft(monkeypatch):
    cfg = TopDownConfig()
    monkeypatch.setattr(strategy, "_zones", lambda rows, security_id, config: ())
    daily = _bars([100 + (i % 4) for i in range(40)])
    weekly = [{"date": (date(2022, 1, 3) + timedelta(days=7*i)).isoformat(),
               "open": 100, "high": 110, "low": 90, "close": 100, "volume": 1000}
              for i in range(30)]
    result = evaluate_daily_candidate(security_id="SEC", symbol="TEST", daily_rows=daily,
        weekly_rows=weekly, nifty_weekly_regime={"regime": "WEAK"}, config=cfg)
    assert result["state"] == "UNINTERESTING"
    assert result["nifty_regime"] == "WEAK"
    assert "NIFTY_CONTEXT_NOT_READY" in result["reason_codes"]


def test_weekly_breakout_context_uses_broken_resistance_below_price():
    resistance = _zone("resistance", 104, 106, "broken-resistance")
    weeks = [{"date": f"2023-01-{i+1:02}", "open": 100, "high": 110,
              "low": 95, "close": 108, "volume": 100} for i in range(30)]
    location, reasons, components = strategy._weekly_location(
        weeks, [resistance], 108, 2, TopDownConfig())
    assert location == "NO_MEANINGFUL_WEEKLY_STRUCTURE"
    assert "NO_MEANINGFUL_WEEKLY_STRUCTURE" in reasons


def test_all_four_completed_hourly_confirmation_archetypes():
    support = _zone("support", 99, 101, "support")
    resistance = _zone("resistance", 99, 101, "resistance")
    hold = [{"date": "2024-01-02T10:15:00+05:30", "open": 100, "high": 103,
             "low": 99, "close": 102, "volume": 200}]
    assert _hourly_pattern(hold, [support], None, 6)[0] == "SUPPORT_HOLD_CONFIRMATION"
    reclaim = [{"date": "2024-01-02T10:15:00+05:30", "open": 98, "high": 99,
                "low": 97, "close": 98, "volume": 100},
               {"date": "2024-01-02T11:15:00+05:30", "open": 100, "high": 103,
                "low": 99, "close": 102, "volume": 200}]
    assert _hourly_pattern(reclaim, [support], None, 6)[0] == "RECLAIM_CONFIRMATION"
    consolidation = []
    for i in range(12):
        high, low = ((100 + i % 2, 96 - i % 2) if i < 6 else (100, 99))
        consolidation.append({"date": f"2024-01-02T{9+i:02}:15:00+05:30", "open": 99,
                              "high": high, "low": low, "close": 99.5, "volume": 100})
    consolidation.append({"date": "2024-01-03T09:15:00+05:30", "open": 101,
                          "high": 103, "low": 100, "close": 102, "volume": 200})
    assert _hourly_pattern(consolidation, [], None, 6)[0] == "CONSOLIDATION_BREAKOUT_CONFIRMATION"
    retest = [{"date": "2024-01-02T10:15:00+05:30", "open": 102, "high": 104,
               "low": 101, "close": 103, "volume": 100},
              {"date": "2024-01-02T11:15:00+05:30", "open": 102, "high": 104,
               "low": 100, "close": 102, "volume": 110}]
    assert _hourly_pattern(retest, [resistance], None, 6)[0] == "BREAKOUT_RETEST_CONFIRMATION"


def test_hourly_confirmation_uses_completed_bars_and_next_bar_open(monkeypatch):
    support = _zone("support", 99, 101, "support")
    monkeypatch.setattr(strategy, "_zones", lambda rows, security_id, config: (support,))
    bars = [{"date": "2024-01-02T10:15:00+05:30", "open": 100, "high": 103,
             "low": 99, "close": 102, "volume": 200,
             "session_type": "NORMAL_SESSION"},
            {"date": "2024-01-02T11:15:00+05:30", "open": 103, "high": 105,
             "low": 102, "close": 104, "volume": 190,
             "session_type": "NORMAL_SESSION"}]
    after = datetime.fromisoformat("2024-01-01T15:30:00+05:30")
    waiting = confirm_hourly(hourly_rows=bars[:1], after_timestamp=after,
        setup_zone=None, weekly_zones=[], daily_zones=[support], hourly_config=ZoneConfig(),
        config=TopDownConfig())
    assert waiting["state"] == "WAITING_1H_CONFIRMATION"
    result = confirm_hourly(hourly_rows=bars, after_timestamp=after,
        setup_zone=None, weekly_zones=[], daily_zones=[support], hourly_config=ZoneConfig(),
        config=TopDownConfig())
    assert result["confirmation_timestamp"] == bars[0]["date"]
    assert result["entry_timestamp"] == bars[1]["date"]
    assert result["entry_type"] == "NEXT_AVAILABLE_1H_OPEN"
    assert result["plan"]["entry"] == 103


def test_structural_plan_uses_actual_fill_and_ema_is_friction_only():
    support = _zone("support", 97.5, 98.5, "support")
    result = structural_plan(entry=100, confirmation_close=99, weekly_zones=[],
        daily_zones=[_zone("resistance", 104, 106, "resistance")], hourly_zones=[support],
        setup_zone=None, daily_atr=2, hourly_atr=1, cfg=TopDownConfig(minimum_room_r=1),
        ema_obstacles={"DAILY_EMA50": 101.5})
    assert result["entry"] == 100
    assert result["stop"] < 97.5
    assert result["target"] == 104
    assert result["target_reference"] == "DAILY"
    assert result["overhead_friction_present"] is True
    assert result["status"] == "ENTRY_READY"


def test_hourly_volume_baseline_uses_same_provider_start_time_and_normal_sessions():
    rows = []
    for i, volume in enumerate((100, 120, 150)):
        rows.append({"date": f"2024-01-0{i+1}T10:15:00+05:30", "volume": volume,
                     "session_type": "NORMAL_SESSION"})
    rows.append({"date": "2024-01-04T10:15:00+05:30", "volume": 900,
                 "session_type": "SPECIAL_SPLIT_SESSION"})
    result = volume_features(rows)
    assert result["same_time_mean_volume"] == (100 + 120 + 150) / 3
    assert result["same_time_relative_volume"] == 900 / ((100 + 120 + 150) / 3)


def test_daily_candidate_is_deterministic_and_does_not_use_future_week_rows(monkeypatch):
    cfg = TopDownConfig()
    monkeypatch.setattr(strategy, "_zones", lambda rows, security_id, config: ())
    daily = _bars([100 + i for i in range(40)])
    weekly = [{"date": (date(2022, 1, 3) + timedelta(days=7*i)).isoformat(),
               "open": 100+i, "high": 102+i, "low": 99+i, "close": 101+i, "volume": 1000+i}
              for i in range(30)]
    inputs = dict(security_id="SEC", symbol="TEST", daily_rows=daily, weekly_rows=weekly,
                  nifty_weekly_regime={"regime": "NEUTRAL"}, config=cfg)
    first = evaluate_daily_candidate(**inputs)
    later_rows = weekly + [{"date": "2025-01-06", "open": 1, "high": 9000,
                            "low": 1, "close": 9000, "volume": 999999}]
    second = evaluate_daily_candidate(**{**inputs, "weekly_rows": weekly})
    assert first == second


def test_lifecycle_snapshot_requires_two_closes_and_subsequent_retest():
    broken_resistance = _zone("resistance", 104, 106, "lifecycle")
    rows = [
        {"date": "2024-01-01", "open": 102, "high": 103, "low": 101, "close": 102, "volume": 10},
        {"date": "2024-01-02", "open": 110, "high": 111, "low": 109, "close": 110, "volume": 10},
        {"date": "2024-01-03", "open": 110, "high": 111, "low": 109, "close": 110, "volume": 10},
        {"date": "2024-01-04", "open": 106.5, "high": 108, "low": 104, "close": 107, "volume": 10},
    ]
    snap = strategy._lifecycle_snapshot([broken_resistance], rows, ZoneConfig())[0]
    assert snap.broken is True
    assert snap.role_flip == "resistance_to_support_candidate"
    before_retest = strategy._lifecycle_snapshot([broken_resistance], rows[:3], ZoneConfig())[0]
    assert before_retest.broken is True
    assert before_retest.role_flip == "ACCEPTANCE_ABOVE_RESISTANCE"


def test_manual_review_sampling_is_deterministic():
    rows = [{"date": f"2024-01-{day:02}", "security_id": f"SEC-{day}"} for day in range(1, 10)]
    first = stable_sample(rows, group="MID_RANGE", limit=4)
    assert first == stable_sample(list(reversed(rows)), group="MID_RANGE", limit=4)
    assert len(first) == 4


def test_stock_scanning_reopens_after_each_terminal_setup_state():
    for terminal in ("ROOM_INSUFFICIENT", "SETUP_INVALIDATED", "ENTRY_READY_EVENT"):
        registry = SetupRegistry()
        registry.start("SEC", {"setup_instance_id": "first"})
        registry.finish("SEC", terminal)
        assert "SEC" not in registry.active
        registry.start("SEC", {"setup_instance_id": "second"})
        assert registry.active["SEC"]["setup_instance_id"] == "second"
        assert registry.later_setup_counts[terminal] == 1


def test_setup_instance_id_is_deterministic_and_instance_specific():
    first = setup_instance_id("SEC", "2023-04-01", "SUPPORT_REACTION", "z1")
    assert first == setup_instance_id("SEC", "2023-04-01", "SUPPORT_REACTION", "z1")
    assert first != setup_instance_id("SEC", "2023-04-02", "SUPPORT_REACTION", "z1")


def test_wick_below_support_does_not_count_as_acceptance():
    support = replace(_zone("support", 99, 101), first_pivot="2024-01-01")
    rows = [{"date": "2024-01-01", "open": 100, "high": 102, "low": 98, "close": 100, "volume": 1}]
    result = strategy._lifecycle_snapshot([support], rows, ZoneConfig())[0]
    assert not result.broken
    assert result.role_state == "ACTIVE"
    assert result.role_flip == "WICK_BELOW_SUPPORT_THEN_RECLAIMED"


def test_support_break_requires_retest_to_confirm_resistance():
    support = replace(_zone("support", 99, 101), first_pivot="2024-01-01")
    rows = [
        {"date": "2024-01-01", "open": 100, "high": 101, "low": 99, "close": 100, "volume": 1},
        {"date": "2024-01-02", "open": 98, "high": 99, "low": 97, "close": 98, "volume": 1},
        {"date": "2024-01-03", "open": 98, "high": 99, "low": 97, "close": 98, "volume": 1},
        {"date": "2024-01-04", "open": 98, "high": 100, "low": 97, "close": 99, "volume": 1},
    ]
    before = strategy._lifecycle_snapshot([support], rows[:3], ZoneConfig())[0]
    after = strategy._lifecycle_snapshot([support], rows, ZoneConfig())[0]
    assert before.role_state == "RESISTANCE_CANDIDATE"
    assert before.role_flip == "ACCEPTANCE_BELOW_SUPPORT"
    assert after.role_state == "RESISTANCE_CONFIRMED"
    assert after.side == "resistance"


def test_resistance_break_requires_retest_to_confirm_support():
    resistance = replace(_zone("resistance", 99, 101), first_pivot="2024-01-01")
    rows = [
        {"date": "2024-01-01", "open": 100, "high": 101, "low": 99, "close": 100, "volume": 1},
        {"date": "2024-01-02", "open": 102, "high": 103, "low": 102, "close": 102, "volume": 1},
        {"date": "2024-01-03", "open": 102, "high": 103, "low": 102, "close": 102, "volume": 1},
        {"date": "2024-01-04", "open": 101, "high": 103, "low": 100, "close": 102, "volume": 1},
    ]
    result = strategy._lifecycle_snapshot([resistance], rows, ZoneConfig())[0]
    assert result.role_state == "SUPPORT_CONFIRMED"
    assert result.side == "support"
    assert result.original_side == "resistance"


def test_confirmed_role_flipped_support_remains_usable_after_original_break():
    flipped = replace(_zone("support", 99, 101), side="support", original_side="resistance",
        broken=True, role_state="SUPPORT_CONFIRMED", role_flip="resistance_to_support_candidate")
    above = _zone("resistance", 135, 138)
    location, values = structural_location([flipped, above], 105, 5, TopDownConfig())
    assert location == "SUPPORT_LOCATION_ACTIONABLE"
    assert values["nearest_support"]["zone_id"] == flipped.zone_id


def test_location_is_structural_and_26_week_range_cannot_create_midrange():
    cfg = TopDownConfig()
    support, resistance = _zone("support", 98, 100), _zone("resistance", 135, 138)
    location, values = structural_location([support, resistance], 105, 5, cfg)
    assert location == "SUPPORT_LOCATION_ACTIONABLE"
    assert values["structural_rr"] >= cfg.minimum_room_r
    assert structural_location([], 105, 5, cfg)[0] == "NO_MEANINGFUL_ENCLOSING_STRUCTURE"
    weekly_rows = [{"date": f"2023-01-{i+1:02}", "open": 105, "high": 110,
                    "low": 100, "close": 105, "volume": 1} for i in range(30)]
    assert strategy._weekly_location(weekly_rows, [], 105, 5, cfg)[0] == "NO_MEANINGFUL_WEEKLY_STRUCTURE"


def test_near_resistance_poor_rr_and_middle_are_structural():
    cfg = TopDownConfig()
    support, near_resistance = _zone("support", 90, 92), _zone("resistance", 108, 110)
    assert structural_location([support, near_resistance], 106, 5, cfg)[0] == "NEAR_RESISTANCE_POOR_RR"
    far_resistance = _zone("resistance", 128, 130)
    assert structural_location([support, far_resistance], 110, 5, cfg)[0] == "MID_RANGE_NOT_ACTIONABLE"


def test_ema_is_friction_and_no_primary_target_uses_fallback():
    plan = structural_plan(entry=100, confirmation_close=100, weekly_zones=[], daily_zones=[],
        hourly_zones=[replace(_zone("resistance", 101, 102), strength="MINOR", reaction_count=1)],
        setup_zone={"lower": 98, "upper": 99, "zone_id": "setup"}, daily_atr=2, hourly_atr=1,
        cfg=TopDownConfig(), ema_obstacles={"DAILY_EMA50": 101})
    assert plan["status"] == "ENTRY_READY"
    assert plan["target_reference"] == "FIXED_2R_RESEARCH_FALLBACK"
    assert plan["overhead_friction_present"]
    assert not any(x["classification"] == "PRIMARY_STRUCTURAL_OBSTACLE" for x in plan["obstacles"])


def test_strong_hourly_resistance_can_block_room_and_daily_target_is_used():
    strong_hourly = replace(_zone("resistance", 101, 102), strength="MAJOR", reaction_count=5)
    fallback_setup = {"lower": 96, "upper": 98, "zone_id": "setup"}
    blocked = structural_plan(entry=100, confirmation_close=100, weekly_zones=[], daily_zones=[],
        hourly_zones=[strong_hourly], setup_zone=fallback_setup, daily_atr=2, hourly_atr=1,
        cfg=TopDownConfig())
    assert blocked["status"] == "ROOM_INSUFFICIENT"
    assert blocked["target_reference"] == "HOURLY"
    assert len([o for o in blocked["obstacles"] if o["timeframe"] == "HOURLY"]) == 1
    assert blocked["obstacles"][0]["classification"] == "PRIMARY_STRUCTURAL_OBSTACLE"
    daily = _zone("resistance", 106, 108, "daily")
    target = structural_plan(entry=100, confirmation_close=100, weekly_zones=[], daily_zones=[daily],
        hourly_zones=[], setup_zone=fallback_setup, daily_atr=2, hourly_atr=1, cfg=TopDownConfig())
    assert target["target_reference"] == "DAILY"


def test_nifty_readiness_is_required_but_weak_regime_is_not_a_rejection():
    rows = _bars([100+i for i in range(50)])
    selection = {"features_available": True, "completed_weeks": 26, "relative_strength_4w": 0,
                 "relative_strength_13w": 0, "status": "eligible"}
    ready = evaluate_daily_candidate(security_id="SEC", symbol="T", daily_rows=rows,
        weekly_rows=[], nifty_weekly_regime={"regime": "WEAK", "component_values": {"completed_weeks": 26}},
        weekly_selection=selection)
    assert ready["NIFTY_CONTEXT_READY"]
    assert ready["nifty_regime"] == "WEAK"
    not_ready = evaluate_daily_candidate(security_id="SEC", symbol="T", daily_rows=rows,
        weekly_rows=[], nifty_weekly_regime={"regime": "NEUTRAL", "component_values": {"completed_weeks": 0}},
        weekly_selection=selection)
    assert not not_ready["NIFTY_CONTEXT_READY"]
    assert "NIFTY_CONTEXT_NOT_READY" in not_ready["reason_codes"]


def test_confirmation_prefix_does_not_depend_on_future_candles():
    support = _zone("support", 99, 101)
    bars = [{"date": "2024-01-02T10:15:00+05:30", "open": 100, "high": 103,
             "low": 99, "close": 102, "volume": 100}]
    first = _hourly_pattern(bars, [support], None, 6)
    later = bars + [{"date": "2024-01-02T11:15:00+05:30", "open": 103,
                     "high": 106, "low": 102, "close": 105, "volume": 120}]
    assert _hourly_pattern(bars, [support], None, 6) == first
    assert _hourly_pattern(later[:1], [support], None, 6) == first


def test_meaningful_zone_thresholds_are_configurable():
    cfg = TopDownConfig(meaningful_touch_episodes=3, meaningful_pivots=3,
                        major_touch_episodes=5)
    assert cfg.meaningful_touch_episodes == 3
    assert cfg.meaningful_pivots == 3
    assert cfg.major_touch_episodes == 5


def test_broken_support_reclaim_requires_acceptance_and_a_later_hold():
    support = replace(_zone("support", 99, 101), first_pivot="2024-01-01")
    rows = [
        {"date": "2024-01-01", "open": 100, "high": 101, "low": 99, "close": 100, "volume": 1},
        {"date": "2024-01-02", "open": 98, "high": 99, "low": 97, "close": 98, "volume": 1},
        {"date": "2024-01-03", "open": 98, "high": 99, "low": 97, "close": 98, "volume": 1},
        {"date": "2024-01-04", "open": 98, "high": 100, "low": 97, "close": 99, "volume": 1},
        {"date": "2024-01-05", "open": 102, "high": 103, "low": 101, "close": 102, "volume": 1},
        {"date": "2024-01-06", "open": 102, "high": 103, "low": 101, "close": 102, "volume": 1},
        {"date": "2024-01-07", "open": 102, "high": 103, "low": 100, "close": 102, "volume": 1},
    ]
    result = strategy._lifecycle_snapshot([support], rows, ZoneConfig())[0]
    assert result.role_state == "SUPPORT_CONFIRMED"
    assert result.role_flip == "support_reclaimed"


def test_structural_exception_requires_stronger_hourly_confirmation():
    support = _zone("support", 99, 101)
    bars = [{"date": "2024-01-02T10:15:00+05:30", "open": 100, "high": 103,
             "low": 99, "close": 102, "volume": 100}]
    pattern, reasons, evidence = _hourly_pattern(bars, [support], None, 6, require_stronger=True)
    assert pattern is None
    assert "STRUCTURAL_EXCEPTION_REQUIRES_STRONGER_H1_CONFIRMATION" in reasons
    assert evidence["strong_confirmation_required"] is True
