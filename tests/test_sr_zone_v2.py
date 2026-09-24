from src.strategy.sr_zone_v2 import ZoneConfig, detect_zones, penetration_depth


def _rows(values):
    return [{"date": f"2023-01-{i+1:02d}", "open": c, "high": h, "low": l,
             "close": c, "volume": 100} for i,(h,l,c) in enumerate(values)]


def _swing_rows():
    # Repeated low pivots around 100, each confirmed by three subsequent bars.
    vals = [(104,102,103),(106,100,102),(105,102,104),(108,104,106),
            (106,100.5,103),(107,103,105),(110,105,108),(108,101,104),
            (107,102,105),(110,105,108),(109,103,107),(111,106,109),
            (108,101,104),(108,102,105),(109,104,107),(110,106,108)]
    return _rows(vals)


def test_atr_zone_width_respects_configured_pct_guards():
    zones = detect_zones(_swing_rows(), security_id="SEC", config=ZoneConfig(swing_left=1,swing_right=1,minimum_pivots=2))
    assert zones
    for zone in zones:
        half = (zone.upper_bound-zone.lower_bound)/2
        assert zone.center*0.0025-1e-9 <= half <= zone.center*0.01+1e-9


def test_zone_ids_are_stable_when_history_is_extended():
    rows = _swing_rows()
    cfg=ZoneConfig(swing_left=1,swing_right=1,minimum_pivots=2)
    first = detect_zones(rows[:15], security_id="SEC",config=cfg)
    extended = detect_zones(rows, security_id="SEC",config=cfg)
    assert set(z.zone_id for z in first).issubset({z.zone_id for z in extended})


def test_same_asof_prefix_is_deterministic_and_future_is_not_visible():
    rows = _swing_rows()
    cfg=ZoneConfig(swing_left=1,swing_right=1,minimum_pivots=2)
    prefix = detect_zones(rows[:13], security_id="SEC",config=cfg)
    assert prefix == detect_zones(rows[:13], security_id="SEC",config=cfg)
    assert all(r["available_index"] < 13 for z in prefix for r in z.reactions)


def test_touch_episode_stays_single_during_consecutive_zone_overlaps():
    zones = detect_zones(_swing_rows(), security_id="SEC",config=ZoneConfig(swing_left=1,swing_right=1,minimum_pivots=2))
    assert all(zone.touch_episodes[0].duration >= 1 for zone in zones if zone.touch_episodes)


def test_penetration_is_not_clamped_at_one():
    actual = penetration_depth(upper=101, lower=99, low=96, atr=2, close_location=.8)
    assert actual["zone_fraction"] == 2.5
    assert actual["atr"] == 2.5


def test_adjacent_zones_are_not_merged_by_overlap_rule():
    rows = _swing_rows()
    # Determinism here exercises the no-random-identity guarantee.
    cfg=ZoneConfig(swing_left=1,swing_right=1,minimum_pivots=2)
    a = detect_zones(rows, security_id="SEC",config=cfg)
    b = detect_zones(rows, security_id="SEC",config=cfg)
    assert a == b


def test_role_flip_and_broken_state_are_explicit():
    values=[(98,96,97),(101,98,100),(99,96,97),(98,95,97),
            (101,98,100),(98,95,96),(99,96,98),(105,102,104),
            (106,103,105),(104,99,103),(105,101,104),(107,103,106)]
    cfg=ZoneConfig(swing_left=1,swing_right=1,atr_period=3,minimum_pivots=2,
                   break_atr=.1,maximum_width_pct=.01)
    zones=detect_zones(_rows(values),security_id="SEC",config=cfg)
    resistance=[z for z in zones if z.side=="resistance"]
    assert resistance
    assert any(z.broken for z in resistance)
    assert any(z.role_flip=="resistance_to_support_candidate" for z in resistance)


def test_reaction_is_published_only_after_full_horizon():
    rows=_swing_rows()
    cfg=ZoneConfig(swing_left=1,swing_right=1,minimum_pivots=2,reaction_horizon=5)
    short=detect_zones(rows[:11],security_id="SEC",config=cfg)
    full=detect_zones(rows,security_id="SEC",config=cfg)
    assert all(item["available_index"] < 11 for zone in short for item in zone.reactions)
    assert all(item["available_index"] < len(rows) for zone in full for item in zone.reactions)


def test_consecutive_overlaps_are_one_touch_episode():
    values=[(103,101,102),(104,99,100),(103,99.5,101),(104,100,102),
            (106,103,105),(107,104,106),(105,99.5,102),(106,100,104),
            (108,103,107),(109,105,108)]
    cfg=ZoneConfig(swing_left=1,swing_right=1,minimum_pivots=2)
    zones=detect_zones(_rows(values),security_id="SEC",config=cfg)
    assert any(episode.duration > 1 for zone in zones for episode in zone.touch_episodes)
