from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Sequence

import pytest

from src.backtest.engine import BacktestConfig, BacktestEngine, BacktestError
from src.backtest.models import CostModel, OperatingCostModel
from src.backtest.smoke import SmokeStrategy
from src.decision.interfaces import EntrySignal, ExitSignal, StrategyContext
from src.risk.position_sizing import FixedFractionRiskPolicy


@dataclass(frozen=True)
class Eligible:
    security_id: str
    symbol: str
    tradable: bool = True


class ToyUniverse:
    def __init__(self, memberships: dict[date, tuple[str, ...]]) -> None:
        self.memberships = memberships

    def members_on(self, day: date, universe_id: str):
        del universe_id
        ids = self.memberships.get(day, self.memberships[max(d for d in self.memberships if d <= day)])
        return [Eligible(security_id, security_id) for security_id in ids]


class SignalStrategy:
    name = "test_signals"
    version = "1.0"

    def __init__(self, signals: dict[date, tuple[str, ...]], *, stop: float = 95,
                 target: float | None = 110, observe=None) -> None:
        self.signals = signals
        self.stop = stop
        self.target = target
        self.observe = observe

    def generate_entries(self, context: StrategyContext) -> Sequence[EntrySignal]:
        if self.observe is not None:
            self.observe(context)
        if context.next_session_open is None:
            return ()
        return [EntrySignal(
            security_id=security_id, symbol=security_id,
            signal_timestamp=context.timestamp,
            execution_timestamp=context.next_session_open,
            stop_price=self.stop, target_price=self.target,
            reason_codes=("TEST_SIGNAL",), strategy_version=self.version,
        ) for security_id in self.signals.get(context.session_date, ())]


def candle(day: date, *, opening=100.0, high=105.0, low=94.0, close=100.0):
    return {"date": datetime.combine(day, datetime.min.time()), "open": opening,
            "high": high, "low": low, "close": close, "volume": 1000}


def setup(*, days=(date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)),
          members=None, capital=10000, risk=0.01, max_positions=10, costs=None,
          operating_costs=None):
    members = members or {day: ("A", "B", "C") for day in days}
    rows = {sid: [candle(day) for day in days if sid in members.get(day, ())]
            for sid in sorted({sid for ids in members.values() for sid in ids})}
    provider = ToyUniverse(members)
    config = BacktestConfig(
        universe_id="TOY_PIT", start_date=days[0], end_date=days[-1],
        initial_capital=capital, max_risk_per_trade=risk,
        max_open_positions=max_positions, costs=costs or CostModel.raw_zero(),
        operating_costs=operating_costs or OperatingCostModel(0, 0, 0),
    )
    return BacktestEngine(config=config, provider=provider), rows, days


def test_eod_signal_executes_next_session_open_and_has_no_future_daily_data() -> None:
    engine, data, days = setup()
    seen = []

    def observe(context):
        seen.append((context.timestamp, {sid: [row["date"].date() for row in rows]
                                         for sid, rows in context.daily_history.items()}))

    data["A"][0] = candle(days[0], opening=98, high=101, low=97, close=100)
    data["A"][1] = candle(days[1], opening=103, high=106, low=99, close=104)
    result = engine.run(strategy=SignalStrategy({days[0]: ("A",)}, stop=95, target=120, observe=observe),
                        daily_data=data)
    entry = next(event for event in result.events if event.event_type == "POSITION_ENTRY")
    signal = next(event for event in result.events if event.event_type == "ENTRY_SIGNAL")
    assert signal.timestamp == datetime(2024, 1, 2, 15, 30, tzinfo=entry.timestamp.tzinfo)
    assert entry.timestamp == datetime(2024, 1, 3, 9, 15, tzinfo=entry.timestamp.tzinfo)
    assert entry.details["fill_price"] == 103
    assert all(day <= stamp.date() for stamp, history in seen for rows in history.values() for day in rows)


def test_stop_exit_uses_gap_open_and_records_trade_ledger_math() -> None:
    engine, data, days = setup()
    data["A"][0] = candle(days[0], low=99)
    data["A"][1] = candle(days[1], opening=100, high=105, low=98, close=102)
    data["A"][2] = candle(days[2], opening=92, high=96, low=90, close=94)
    result = engine.run(strategy=SignalStrategy({days[0]: ("A",)}, stop=95, target=115), daily_data=data)
    trade = result.trades[0]
    assert trade.exit_reason == "STOP"
    assert trade.exit_price == 92
    assert trade.quantity == 20  # 1% of 10,000 / (100 - 95)
    assert trade.initial_risk_rupees == 100
    assert trade.gross_pnl == -160
    assert trade.net_pnl == trade.gross_pnl - trade.costs
    assert trade.R_multiple == pytest.approx(trade.net_pnl / trade.initial_risk_rupees)
    assert trade.holding_period == 2
    assert result.metrics["maximum_drawdown"] > 0
    assert result.realized_pnl == pytest.approx(trade.net_pnl)
    assert result.unrealized_pnl == 0
    assert result.closed_positions == result.trades
    assert result.metrics["ending_equity"] == pytest.approx(
        result.metadata.starting_capital + result.realized_pnl + result.unrealized_pnl
    )


def test_target_exit_and_stop_wins_when_both_intraday_levels_are_touched() -> None:
    engine, data, days = setup()
    data["A"][0] = candle(days[0], low=99)
    data["A"][1] = candle(days[1], opening=100, high=112, low=90, close=105)
    result = engine.run(strategy=SignalStrategy({days[0]: ("A",)}, stop=95, target=110), daily_data=data)
    assert result.trades[0].exit_reason == "STOP"
    assert result.trades[0].exit_price == 95

    data["A"][1] = candle(days[1], opening=100, high=112, low=98, close=111)
    target_result = engine.run(strategy=SignalStrategy({days[0]: ("A",)}, stop=95, target=110), daily_data=data)
    assert target_result.trades[0].exit_reason == "TARGET"
    assert target_result.trades[0].exit_price == 110


def test_position_remains_managed_while_next_candidate_is_entered() -> None:
    engine, data, days = setup(max_positions=2)
    data["A"][0] = candle(days[0], low=99)
    data["A"][1] = candle(days[1], high=107, low=98, close=103)
    data["A"][2] = candle(days[2], high=112, low=98, close=110)
    for day in days:
        data["B"][days.index(day)] = candle(day, low=99)
    result = engine.run(strategy=SignalStrategy({days[0]: ("A",), days[1]: ("B",)},
                                                stop=95, target=110), daily_data=data)
    entries = [event.security_id for event in result.events if event.event_type == "POSITION_ENTRY"]
    exits = [(event.security_id, event.details["reason"]) for event in result.events
             if event.event_type == "POSITION_EXIT"]
    assert entries == ["A", "B"]
    assert exits == [("A", "TARGET")]
    assert [position["security_id"] for position in result.open_positions] == ["B"]


def test_cash_cap_zero_quantity_and_multiple_positions() -> None:
    sizing = FixedFractionRiskPolicy()
    decision = sizing.size(equity=10000, available_cash=10000, entry_price=100,
                           stop_price=90, risk_fraction=0.01)
    assert decision.quantity == 10 and decision.risk_budget == 100
    assert sizing.size(equity=100, available_cash=100, entry_price=500,
                       stop_price=450, risk_fraction=0.01).quantity == 0

    engine, data, days = setup(capital=1000, risk=1.0, max_positions=2)
    for sid in ("A", "B"):
        data[sid][0] = candle(days[0], low=99)
        data[sid][1] = candle(days[1], opening=100, high=105, low=98, close=102)
        data[sid][2] = candle(days[2], opening=100, high=105, low=98, close=102)
    result = engine.run(strategy=SignalStrategy({days[0]: ("A", "B"), days[1]: ("A", "B")},
                                                stop=90, target=120), daily_data=data)
    assert len(result.open_positions) == 1
    assert result.open_positions[0]["quantity"] == 10
    assert sum(item["quantity"] * item["entry_price"] for item in result.open_positions) <= 1000
    assert any(event.event_type == "ENTRY_REJECTED" and "zero quantity" in event.details["reason"]
               for event in result.events)

    engine, data, days = setup(capital=10000, max_positions=2)
    for sid in ("A", "B"):
        data[sid][0] = candle(days[0], low=99)
        data[sid][1] = candle(days[1], opening=100, high=105, low=98, close=102)
        data[sid][2] = candle(days[2], opening=100, high=105, low=98, close=102)
    multiple = engine.run(strategy=SignalStrategy({days[0]: ("A", "B")}, stop=90, target=120), daily_data=data)
    assert {position["security_id"] for position in multiple.open_positions} == {"A", "B"}


def test_cost_model_components_are_configurable_and_applied_to_trade() -> None:
    costs = CostModel(brokerage_rate=0.001, stt_buy_rate=0.001,
                      stt_sell_rate=0.002, nse_transaction_rate=0.001, gst_rate=0.18,
                      sebi_rate=0.0001, stamp_buy_rate=0.0002,
                      ipft_rate=0.000000001, dp_charge_per_sell=15.34,
                      slippage_bps=10, scenario="TEST_FIXTURE", verification_label="TEST_FIXTURE")
    engine, data, days = setup(costs=costs)
    data["A"][0] = candle(days[0], low=99)
    data["A"][1] = candle(days[1], high=112, low=98, close=111)
    result = engine.run(strategy=SignalStrategy({days[0]: ("A",)}, stop=95, target=110), daily_data=data)
    trade = result.trades[0]
    assert trade.entry_price == pytest.approx(100.1)
    assert trade.costs > 0
    assert trade.cost_breakdown["entry"] > 0 and trade.cost_breakdown["exit"] > 0
    assert trade.net_pnl == pytest.approx(trade.gross_pnl - trade.costs)


def test_pending_entry_is_rejected_if_security_left_pit_universe() -> None:
    days = (date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4))
    memberships = {days[0]: ("A",), days[1]: ("B",), days[2]: ("B",)}
    engine, data, _ = setup(days=days, members=memberships)
    data["A"] = [candle(day, low=99) for day in days]
    data["B"] = [candle(day) for day in days]
    result = engine.run(strategy=SignalStrategy({days[0]: ("A",)}), daily_data=data)
    assert not [event for event in result.events if event.event_type == "POSITION_ENTRY"]
    assert any(event.event_type == "ENTRY_REJECTED" and "not PIT eligible" in event.details["reason"]
               for event in result.events)


def test_strategy_exit_signal_executes_next_session_open() -> None:
    engine, data, days = setup()
    data["A"][0] = candle(days[0], low=99)
    data["A"][1] = candle(days[1], opening=100, high=105, low=98, close=102)
    data["A"][2] = candle(days[2], opening=103, high=105, low=102, close=104)

    class ExitAtClose(SignalStrategy):
        def generate_exits(self, context):
            if context.session_date == days[1] and "A" in context.open_security_ids:
                return [ExitSignal("A", context.timestamp, context.next_session_open,
                                   "TEST_RULE", strategy_version=self.version)]
            return ()

    result = engine.run(strategy=ExitAtClose({days[0]: ("A",)}, stop=95, target=120), daily_data=data)
    trade = result.trades[0]
    assert trade.exit_reason == "STRATEGY_EXIT:TEST_RULE"
    assert trade.exit_timestamp == datetime(2024, 1, 4, 9, 15, tzinfo=trade.exit_timestamp.tzinfo)
    assert trade.exit_price == 103
    assert not result.open_positions


def test_weekly_candle_is_hidden_until_the_final_session_close() -> None:
    days = (date(2023, 1, 2), date(2023, 1, 5), date(2023, 1, 6))
    engine, data, _ = setup(days=days, members={day: ("A",) for day in days})
    visible = {}

    def observe(context):
        visible[context.session_date] = context.latest_completed_weekly["A"]

    engine.run(strategy=SignalStrategy({}, observe=observe), daily_data=data)
    assert visible[date(2023, 1, 5)] is None
    assert visible[date(2023, 1, 6)]["last_session"] == "2023-01-06"


def test_repeated_runs_have_identical_ledger_metrics_and_run_identity() -> None:
    engine, data, days = setup()
    data["A"][0] = candle(days[0], low=99)
    data["A"][1] = candle(days[1], high=112, low=98, close=111)
    strategy = SignalStrategy({days[0]: ("A",)}, stop=95, target=110)
    first = engine.run(strategy=strategy, daily_data=data, dataset_version="fixture")
    second = engine.run(strategy=strategy, daily_data=data, dataset_version="fixture")
    assert first.metadata.run_id == second.metadata.run_id
    assert first.trades == second.trades
    assert first.metrics == second.metrics
    assert first.equity_curve == second.equity_curve


def test_strategy_cannot_emit_non_close_or_early_execution_timestamps() -> None:
    engine, data, days = setup()

    class BadStrategy(SignalStrategy):
        def generate_entries(self, context):
            if context.session_date != days[0]:
                return ()
            return [EntrySignal("A", "A", context.timestamp,
                                datetime(2024, 1, 2, 15, 31, tzinfo=context.timestamp.tzinfo),
                                95, 110)]

    with pytest.raises(BacktestError, match="next available session open"):
        engine.run(strategy=BadStrategy({}), daily_data=data)


def test_smoke_strategy_is_explicitly_exercisable_not_a_trading_claim() -> None:
    days = (date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4))
    engine, data, _ = setup(days=days)
    data["A"] = [candle(date(2023, 12, 29), low=98, close=99),
                 candle(days[0], low=99, close=100),
                 candle(days[1], opening=100, low=99.5, high=103, close=100),
                 candle(days[2], opening=100, low=99, high=101, close=100)]
    result = engine.run(strategy=SmokeStrategy(), daily_data=data)
    assert result.metadata.strategy_name == "smoke"
    assert len(result.trades) == 1
    assert "SMOKE_CLOSE_ABOVE_PRIOR_CLOSE" in result.trades[0].metadata["reason_codes"]
    assert result.trades[0].R_multiple == pytest.approx(2.0)
    assert result.metrics["ending_equity"] == pytest.approx(10200)
    assert not result.open_positions


def test_target_r_is_resolved_from_the_actual_next_session_fill() -> None:
    days = (date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4))
    engine, data, _ = setup(days=days)
    data["A"][0] = candle(days[0], low=99)
    data["A"][1] = candle(days[1], opening=103, high=105, low=100, close=104)
    data["A"][2] = candle(days[2], opening=103, high=120, low=100, close=119)

    class TargetR(SignalStrategy):
        def generate_entries(self, context):
            if context.session_date == days[0]:
                return [EntrySignal("A", "A", context.timestamp, context.next_session_open,
                                    95, metadata={"target_r": 2.0})]
            return ()

    result = engine.run(strategy=TargetR({}), daily_data=data)
    assert result.trades[0].entry_price == 103
    assert result.trades[0].exit_reason == "TARGET"
    assert result.trades[0].exit_price == 119


def test_requested_period_must_be_covered_at_both_ends() -> None:
    days = (date(2022, 7, 1), date(2022, 7, 4), date(2022, 7, 5))
    engine, data, _ = setup(days=days)
    engine.config = BacktestConfig(
        universe_id="TOY_PIT", start_date=date(2020, 1, 1), end_date=days[-1],
        initial_capital=10000, max_risk_per_trade=0.01,
    )
    with pytest.raises(BacktestError, match="beginning of the requested evaluation period"):
        engine.run(strategy=SignalStrategy({}), daily_data=data)
