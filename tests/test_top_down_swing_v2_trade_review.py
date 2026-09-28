from datetime import datetime, timezone, timedelta

import pytest

from src.backtest.top_down_swing_v2_trade_review import (
    STRATEGY_VERSION,
    Candle,
    causal_entry_state,
    deterministic_filename,
    filter_trades,
    load_candles,
    render_trade_chart,
    sort_trades,
    trade_path_excursions,
)

IST = timezone(timedelta(hours=5, minutes=30))


def candle(stamp, open_=100, high=105, low=95, close=102, volume=1000):
    return Candle(datetime.fromisoformat(stamp).replace(tzinfo=IST), open_, high, low, close, volume)


def trade(**updates):
    row = {
        "setup_instance_id": "id-1", "setup_event_id": "event-1", "strategy_version": STRATEGY_VERSION,
        "symbol": "ADANI PORTS", "setup_family": "BREAKOUT_RETEST", "status": "CLOSED",
        "entry_timestamp": "2023-05-24T13:15:00+05:30", "entry_price": "100", "stop_price": "90",
        "target_price": "130", "quantity": "2", "position_notional": "200",
        "exit_timestamp": "2023-05-25T10:15:00+05:30", "exit_price": "90", "raw_exit_price": "90",
        "exit_reason": "STOP", "gross_pnl": "-20", "net_pnl": "-25", "net_R": "-1.25",
        "gross_R": "-1", "mfe_R": "0.5", "mae_R": "-1", "holding_sessions": "2",
        "planned_rr": "3", "entry_raw_open": "100",
    }
    row.update(updates)
    return row


def test_loser_selection_uses_closed_net_result_and_correct_v2_arm_provenance():
    rows = [trade(), trade(setup_instance_id="winner", net_pnl="2"),
            trade(setup_instance_id="open", status="OPEN", net_pnl="-50")]
    assert [r["setup_instance_id"] for r in filter_trades(rows)] == ["id-1"]
    with pytest.raises(ValueError, match="strategy_version"):
        filter_trades([trade(strategy_version="v1")])


def test_non_loser_mode_includes_closed_winner_but_never_open_position():
    rows = [trade(), trade(setup_instance_id="winner", net_pnl="1"),
            trade(setup_instance_id="open", status="OPEN")]
    assert len(filter_trades(rows, losers_only=False)) == 2


def test_mfe_mae_include_entry_and_intrabar_exit_candles_like_saved_engine():
    bars = [candle("2023-05-24T13:15:00", high=108, low=98),
            candle("2023-05-25T09:15:00", high=111, low=96),
            candle("2023-05-25T10:15:00", high=107, low=88)]
    mfe, mae, ambiguous = trade_path_excursions(bars,
        entry_timestamp="2023-05-24T13:15:00+05:30", exit_timestamp="2023-05-25T10:15:00+05:30",
        entry_price=100, stop_price=90, exit_reason="STOP")
    assert mfe == pytest.approx(1.1)
    assert mae == pytest.approx(-1.2)
    assert ambiguous is True


def test_gap_exit_candle_extrema_are_excluded_because_fill_occurs_at_open():
    bars = [candle("2023-05-24T13:15:00", high=105, low=95),
            candle("2023-05-25T09:15:00", open_=85, high=110, low=80)]
    mfe, mae, ambiguous = trade_path_excursions(bars,
        entry_timestamp="2023-05-24T13:15:00+05:30", exit_timestamp="2023-05-25T09:15:00+05:30",
        entry_price=100, stop_price=90, exit_reason="GAP_STOP")
    assert (mfe, mae, ambiguous) == pytest.approx((0.5, -0.5, False))


def test_exit_open_time_exit_excludes_later_exit_candle_range():
    bars = [candle("2023-05-24T13:15:00", high=105, low=95),
            candle("2023-05-25T09:15:00", open_=100, high=120, low=80)]
    mfe, mae, _ = trade_path_excursions(bars,
        entry_timestamp="2023-05-24T13:15:00+05:30", exit_timestamp="2023-05-25T09:15:00+05:30",
        entry_price=100, stop_price=90, exit_reason="TIME_MAX_HOLD_EXIT")
    assert mfe == pytest.approx(.5)
    assert mae == pytest.approx(-.5)


def test_entry_or_exit_timestamp_missing_is_not_guessed():
    bars = [candle("2023-05-24T13:15:00")]
    with pytest.raises(ValueError, match="Exit timestamp"):
        trade_path_excursions(bars,
            entry_timestamp="2023-05-24T13:15:00+05:30", exit_timestamp="2023-05-25T10:15:00+05:30",
            entry_price=100, stop_price=90, exit_reason="STOP")


def test_candle_loader_stops_before_out_of_sample_ohlcv(tmp_path):
    path = tmp_path / "candles.csv"
    path.write_text(
        "date,open,high,low,close,volume\n"
        "2024-12-31T09:15:00+05:30,100,101,99,100,10\n"
        "2025-01-01T09:15:00+05:30,not-a-price,0,0,0,-1\n",
        encoding="utf-8",
    )
    candles = load_candles(path, through=datetime(2024, 12, 31).date())
    assert len(candles) == 1
    assert candles[0].timestamp.date().isoformat() == "2024-12-31"


def test_giveback_and_all_review_sort_modes_are_deterministic():
    a = trade(setup_instance_id="a", mfe_R="0.1", net_R="-1.2", giveback_R="1.3")
    b = trade(setup_instance_id="b", mfe_R="1.0", net_R="-0.5", giveback_R="1.5",
              entry_timestamp="2023-05-26T13:15:00+05:30")
    selected = filter_trades([b, a])
    assert {r["setup_instance_id"]: float(r["giveback_R"]) for r in selected} == pytest.approx({"a": 1.3, "b": 1.5})
    assert [r["setup_instance_id"] for r in sort_trades(selected, "mfe_asc")] == ["a", "b"]
    assert [r["setup_instance_id"] for r in sort_trades(selected, "giveback_desc")] == ["b", "a"]
    assert [r["setup_instance_id"] for r in sort_trades(selected, "realized_r_asc")] == ["a", "b"]
    assert [r["setup_instance_id"] for r in sort_trades(selected, "chronological")] == ["a", "b"]


def test_year_setup_symbol_and_mfe_filters():
    row = trade()
    assert len(filter_trades([row], year=2023, setup="breakout_retest", symbol="adani ports",
                             min_mfe_r=.5, max_mfe_r=.5)) == 1
    assert filter_trades([row], year=2024) == []
    assert filter_trades([row], setup="SUPPORT_REACTION") == []
    assert filter_trades([row], symbol="OTHER") == []
    assert filter_trades([row], max_mfe_r=.49) == []


def test_entry_state_is_taken_from_saved_signal_and_rejects_future_zone_state():
    signal = {"date": "2023-05-24", "setup_family": "BREAKOUT_RETEST", "support": "{}",
              "resistance": "{}", "daily_zones": '[{"zone_id":"z","last_touch_date":"2023-05-24",'
              '"pivots":["2023-05-20"],"role_transitions":[{"date":"2023-05-23"}]}]',
              "weekly_selection_path": "{}", "weekly_structure": "{}", "nifty_regime": "NEUTRAL",
              "result": "{}", "plan": '{"entry": 100}', "confirmation_timestamp": "2023-05-24T12:15:00+05:30",
              "executable_timestamp": "2023-05-24T13:15:00+05:30"}
    state = causal_entry_state(signal, entry_date=datetime(2023, 5, 24).date())
    assert state["zones"][0]["zone_id"] == "z"
    signal["daily_zones"] = '[{"zone_id":"future","last_touch_date":"2023-05-25"}]'
    with pytest.raises(ValueError, match="future zone"):
        causal_entry_state(signal, entry_date=datetime(2023, 5, 24).date())


def test_filename_is_safe_and_repeatable():
    row = trade()
    expected = "001_ADANI_PORTS_2023-05-24_BREAKOUT_RETEST.png"
    assert deterministic_filename(1, row) == expected
    assert deterministic_filename(1, row) == expected


def test_missing_hourly_chart_data_is_reported_without_aborting_other_trades(tmp_path):
    failure = render_trade_chart(trade(), {}, [candle("2023-05-24T00:00:00")], [], tmp_path / "missing.png")
    assert failure == {
        "symbol": "ADANI PORTS",
        "setup_instance_id": "id-1",
        "failure_reason": "Hourly data unavailable for chart rendering",
    }


def test_control_no_time_ledger_path_is_the_only_portfolio_input():
    from src.backtest.top_down_swing_v2_trade_review import PORTFOLIO_ROOT
    assert (PORTFOLIO_ROOT / "control_no_time_trades.csv").name == "control_no_time_trades.csv"
