from datetime import date
from types import SimpleNamespace

import yaml

from src.backtest import top_down_swing_v3_1_replay as replay_module
from src.backtest.top_down_swing_v3_1_portfolio import _ranking_key
from src.backtest.top_down_swing_v3_1_replay import _candidates, _h1_confirmation, _hysteresis


def _bars(closes, lows=None, volumes=None):
    lows = lows or [x - 1 for x in closes]
    volumes = volumes or [100] * len(closes)
    return [{"date": f"2024-01-01T{9+i:02d}:15:00+05:30", "open": c-.2, "high": c+.5,
             "low": lo, "close": c, "volume": v} for i, (c, lo, v) in enumerate(zip(closes, lows, volumes))]


def test_native_candidate_builder_deduplicates_by_family_and_stable_zone_id():
    zone = SimpleNamespace(zone_id="Z1")
    cfg = SimpleNamespace(daily_zone=SimpleNamespace(reaction_horizon=5))
    loc = {"state": "NEAR_RECENT_DAILY_SUPPORT", "support": zone,
           "direct_breakout_candidates": [zone]}
    assert _candidates(loc, cfg) == [("CONSOLIDATION_BREAKOUT", zone), ("SUPPORT_REACTION", zone)]


def test_breakout_retest_requires_acceptance_touch_response_and_breakout_volume():
    zone = SimpleNamespace(lower_bound=99.0, upper_bound=100.0, center=99.5)
    # Two closes accepted above, then a zone touch, then bullish local reclaim.
    rows = _bars([100.4, 101, 101.2, 100.1, 100.8, 102],
                 lows=[100, 100.5, 100.8, 99.8, 100.1, 100.6],
                 volumes=[100, 250, 100, 100, 100, 250])
    ok, _, evidence = _h1_confirmation("BREAKOUT_RETEST", zone, rows, 1.0, 3)
    assert evidence["prior_acceptance_above_zone"]
    assert evidence["retest_touched_from_above"]
    assert evidence["original_breakout_volume_confirmation"]
    assert ok


def test_breakout_retest_rejects_no_role_flip_or_no_touch():
    zone = SimpleNamespace(lower_bound=99.0, upper_bound=100.0, center=99.5)
    no_accept = _bars([99, 100.2, 99.8, 101, 100.5])
    assert not _h1_confirmation("BREAKOUT_RETEST", zone, no_accept, 1.0, 2)[0]
    no_touch = _bars([101, 102, 103, 102, 104], lows=[100.5, 101, 102, 101, 103])
    assert not _h1_confirmation("BREAKOUT_RETEST", zone, no_touch, 1.0, 2)[0]


def test_consolidation_breakout_requires_compression_and_volume_expansion():
    zone = SimpleNamespace(upper_bound=100.0)
    closes = [98, 99, 98.5, 99.2, 98.7, 99.5, 99, 99.4, 99.1, 99.3, 99.2, 99.4, 101]
    rows = _bars(closes, volumes=[100]*12+[300])
    ok, _, evidence = _h1_confirmation("CONSOLIDATION_BREAKOUT", zone, rows, 1.0, 5)
    assert evidence["compression"]
    assert ok


def test_transition_candidate_needs_two_sessions_and_valid_evidence_on_each():
    first = {"regime": "TRANSITION", "candidate_regime": "TREND_UP", "reason_codes": [
        "HIGHER_HIGH_AND_HIGHER_LOW", "ESTABLISHED_RISING_SUPPORT"]}
    one = _hysteresis({"regime": "TRANSITION"}, first, date(2024, 1, 2))
    assert one["regime"] == "TRANSITION"
    invalid_second = {"regime": "TREND_UP", "candidate_regime": "TREND_UP", "reason_codes": []}
    two = _hysteresis(one, invalid_second, date(2024, 1, 3))
    assert two["regime"] == "TRANSITION"
    valid_second = {"regime": "TREND_UP", "candidate_regime": "TREND_UP", "reason_codes": [
        "HIGHER_HIGH_AND_HIGHER_LOW", "ESTABLISHED_RISING_SUPPORT"]}
    three = _hysteresis(one, valid_second, date(2024, 1, 3))
    assert three["regime"] == "TREND_UP"


def test_checkpoint_codec_round_trips_lifecycle_keys_dates_and_zone_dataclasses():
    from src.strategy.top_down_swing_v2 import V2Zone

    zone = V2Zone(zone_id="Z", security_id="S", side="support", lower_bound=9, upper_bound=11,
        center=10, creation_date="2023-01-01", first_pivot="2023-01-01", last_update="2023-01-02",
        pivots=("2023-01-01",), touch_episodes=(), reactions=(), broken=False, role_flip=None,
        freshness="FRESH")
    original = {("S", "SUPPORT_REACTION", "Z"): {"zone": zone, "created": date(2023, 1, 2),
        "history": ["WATCHING"]}}
    restored = replay_module._unpack(replay_module._pack(original))
    key = ("S", "SUPPORT_REACTION", "Z")
    assert restored[key]["created"] == date(2023, 1, 2)
    assert restored[key]["zone"] == zone


def test_v31_ranking_accepts_serialized_h1_evidence_from_replay_csv():
    signal = {"setup_family": "SUPPORT_REACTION", "regime": "RANGE", "state": "ENTRY_READY",
        "plan": {"planned_reward_r": 2.0}, "h1_evidence": '{"buyer_control":true}',
        "horizontal_zone_id": "Z1", "symbol": "TEST", "setup_instance_id": "I1"}
    assert _ranking_key(signal) < _ranking_key({**signal, "h1_evidence": "not-json"})


def test_atomic_checkpoint_and_clean_vs_resumed_outputs_are_identical(tmp_path, monkeypatch):
    config_path = replay_module.CONFIG
    output_dir = tmp_path / "out"
    checkpoint = output_dir / "checkpoints" / "v3_1_replay_state.json"
    replay_module._atomic_checkpoint(checkpoint, {"value": 1})
    assert replay_module._unpack(yaml.safe_load(checkpoint.read_text(encoding="utf-8"))) == {"value": 1}
    assert not list(checkpoint.parent.glob("*.tmp"))

    member = SimpleNamespace(security_id="S1", symbol="TEST", effective_from=date(2023, 1, 1))
    dates = [date(2022, 12, 30)] + [date(2023, 1, d) for d in (2, 3, 4, 5, 6, 9, 10)]
    rows = [{"date": d.isoformat(), "open": 10.0, "high": 11.0, "low": 9.0, "close": 10.5, "volume": 100}
            for d in dates]
    daily = {"S1": rows}
    hourly = {"S1": []}
    monkeypatch.setattr(replay_module, "load_nifty_benchmark", lambda *a, **k: [])
    monkeypatch.setattr(replay_module, "_v2_config", lambda raw: SimpleNamespace(weekly_zone=object(), daily_zone=SimpleNamespace(reaction_horizon=5)))
    monkeypatch.setattr(replay_module, "completed_weekly_rows", lambda *a, **k: [])
    monkeypatch.setattr(replay_module, "enriched_zones", lambda *a, **k: ())
    monkeypatch.setattr(replay_module, "adaptive_recent_structure", lambda data, **k: {
        "zones": (), "rows": [r for r in data if date.fromisoformat(r["date"]) < date.fromisoformat(k["sod_date"])],
        "structure_rows": []})
    monkeypatch.setattr(replay_module, "daily_location", lambda *a, **k: {"state": "NO_CANDIDATE", "daily_atr": 1.0})
    monkeypatch.setattr(replay_module, "classify_regime", lambda *a, **k: {
        "regime": "RANGE", "candidate_regime": "RANGE", "reason_codes": [], "trendlines": {}, "ema": {}})
    monkeypatch.setattr(replay_module.UniverseProvider, "members_over", lambda *a, **k: [member])
    monkeypatch.setattr(replay_module.UniverseProvider, "members_on", lambda self, day, *a, **k: [member])
    monkeypatch.setattr(replay_module.UniverseProvider, "is_security_eligible", lambda *a, **k: True)

    clean = replay_module.replay(config_path=config_path, output_dir=output_dir, daily=daily, hourly=hourly)
    assert clean["completed"] is True
    expected = {p.name: p.read_bytes() for p in output_dir.iterdir() if p.is_file()}

    partial = replay_module.replay(config_path=config_path, output_dir=output_dir, daily=daily, hourly=hourly,
        _stop_after_days=5)
    assert partial["completed"] is False
    assert partial["last_completed_date"] == "2023-01-06"
    resumed = replay_module.replay(config_path=config_path, output_dir=output_dir, daily=daily, hourly=hourly,
        resume=True)
    assert resumed["completed"] is True
    actual = {p.name: p.read_bytes() for p in output_dir.iterdir() if p.is_file()}
    assert actual == expected
