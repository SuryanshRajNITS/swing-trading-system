"""Event-driven, long-only cash-equity backtest engine."""

from __future__ import annotations

import hashlib
import json
from bisect import bisect_right
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta, timezone
from types import MappingProxyType
from typing import Any, Iterator, Mapping, Sequence
from zoneinfo import ZoneInfo

from src.backtest.metrics import calculate_metrics
from src.backtest.models import (
    BacktestConfig, BacktestEvent, Position, RunMetadata, TradeRecord,
)
from src.data.universe.provider import UniverseProvider
from src.data.weekly import resample_daily_to_weekly
from src.decision.interfaces import EntrySignal, ExitSignal, RiskPolicy, Strategy, StrategyContext
from src.risk.position_sizing import FixedFractionRiskPolicy


class BacktestError(ValueError):
    """Invalid input or inconsistent historical simulation state."""


ENGINE_VERSION = "0.1.0"


@dataclass(frozen=True)
class BacktestResult:
    metadata: RunMetadata
    trades: tuple[TradeRecord, ...]
    events: tuple[BacktestEvent, ...]
    equity_curve: tuple[Mapping[str, Any], ...]
    metrics: Mapping[str, Any]
    remaining_cash: float
    reserved_capital: float
    realized_pnl: float
    unrealized_pnl: float
    open_positions: tuple[Mapping[str, Any], ...]

    @property
    def closed_positions(self) -> tuple[TradeRecord, ...]:
        return self.trades


class _AsOfDailyHistory(Mapping[str, Sequence[Mapping[str, Any]]]):
    """Read-only history view that cannot reveal candles after the current date."""

    def __init__(self, days: Mapping[str, tuple[date, ...]],
                 rows: Mapping[str, tuple[Mapping[str, Any], ...]], cutoff: date) -> None:
        self._days = days
        self._rows = rows
        self._cutoff = cutoff

    def __getitem__(self, security_id: str) -> Sequence[Mapping[str, Any]]:
        days = self._days[security_id]
        limit = bisect_right(days, self._cutoff)
        return self._rows[security_id][:limit]

    def __iter__(self) -> Iterator[str]:
        return iter(self._rows)

    def __len__(self) -> int:
        return len(self._rows)


def _canonical_hash(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _to_datetime(value: Any, zone: ZoneInfo) -> datetime:
    if isinstance(value, datetime):
        return value.replace(tzinfo=zone) if value.tzinfo is None else value.astimezone(zone)
    if isinstance(value, date):
        return datetime.combine(value, time.min, tzinfo=zone)
    if isinstance(value, str):
        parsed = datetime.fromisoformat(value)
        return parsed.replace(tzinfo=zone) if parsed.tzinfo is None else parsed.astimezone(zone)
    raise BacktestError("Daily candle contains an invalid date value.")


def _candle_index(data: Mapping[str, Sequence[Mapping[str, Any]]], zone: ZoneInfo) -> tuple[dict[str, dict[date, dict[str, Any]]], list[date]]:
    indexed: dict[str, dict[date, dict[str, Any]]] = {}
    sessions: set[date] = set()
    for security_id, rows in data.items():
        by_day: dict[date, dict[str, Any]] = {}
        for source in rows:
            row = dict(source)
            stamp = _to_datetime(row.get("date"), zone)
            day = stamp.date()
            if day in by_day:
                raise BacktestError(f"Duplicate daily candle for {security_id} on {day}.")
            try:
                for field in ("open", "high", "low", "close"):
                    row[field] = float(row[field])
                row["volume"] = int(row["volume"])
            except (KeyError, TypeError, ValueError):
                raise BacktestError(f"Invalid daily OHLCV candle for {security_id} on {day}.") from None
            row["date"] = stamp
            by_day[day] = row
            sessions.add(day)
        indexed[security_id] = by_day
    return indexed, sorted(sessions)


def _costs_snapshot(costs: Any) -> dict[str, Any]:
    return asdict(costs)


def _position_snapshot(position: Position) -> dict[str, Any]:
    return {
        "security_id": position.security_id, "symbol": position.symbol,
        "entry_timestamp": position.entry_timestamp.isoformat(),
        "entry_price": position.entry_price, "quantity": position.quantity,
        "initial_stop": position.initial_stop, "target_price": position.target_price,
        "last_price": position.last_price,
    }


class BacktestEngine:
    """Historical fills are simulated here; no broker/execution adapter is called."""

    def __init__(
        self, *, config: BacktestConfig, provider: UniverseProvider | None = None,
        risk_policy: RiskPolicy | None = None,
    ) -> None:
        self.config = config
        self.provider = provider or UniverseProvider()
        self.risk_policy = risk_policy or FixedFractionRiskPolicy()

    def run(
        self, *, strategy: Strategy,
        daily_data: Mapping[str, Sequence[Mapping[str, Any]]],
        dataset_version: str = "unspecified",
    ) -> BacktestResult:
        zone = ZoneInfo(self.config.timezone_name)
        indexed, all_sessions = _candle_index(daily_data, zone)
        sessions = [day for day in all_sessions if self.config.start_date <= day <= self.config.end_date]
        if not sessions:
            raise BacktestError("No daily candles exist in the requested evaluation period.")
        if sessions[0] > self.config.start_date + timedelta(days=7):
            raise BacktestError("Daily dataset does not cover the beginning of the requested evaluation period.")
        if sessions[-1] < self.config.end_date - timedelta(days=7):
            raise BacktestError("Daily dataset does not cover the end of the requested evaluation period.")

        weekly = {
            security_id: resample_daily_to_weekly(rows, trading_sessions=all_sessions,
                                                   timezone_name=self.config.timezone_name)
            for security_id, rows in daily_data.items()
        }
        weekly_index: dict[str, tuple[tuple[datetime, ...], tuple[Mapping[str, Any], ...]]] = {}
        for security_id, candles in weekly.items():
            indexed_weekly = sorted(
                ((datetime.fromisoformat(candle["completion_at"]).astimezone(zone), candle)
                 for candle in candles), key=lambda item: item[0],
            )
            weekly_index[security_id] = (
                tuple(item[0] for item in indexed_weekly),
                tuple(MappingProxyType(item[1]) for item in indexed_weekly),
            )
        history_days: dict[str, tuple[date, ...]] = {}
        history_rows: dict[str, tuple[Mapping[str, Any], ...]] = {}
        for security_id, series in indexed.items():
            ordered = sorted(series.items())
            history_days[security_id] = tuple(item[0] for item in ordered)
            history_rows[security_id] = tuple(MappingProxyType(item[1]) for item in ordered)
        cash = self.config.initial_capital
        positions: dict[str, Position] = {}
        pending: dict[date, list[EntrySignal]] = {}
        pending_exits: dict[date, list[ExitSignal]] = {}
        charged_dp_debits: set[tuple[str, date]] = set()
        transaction_costs_total = 0.0
        trades: list[TradeRecord] = []
        events: list[BacktestEvent] = []
        equity_curve: list[dict[str, Any]] = [{
            "date": sessions[0].isoformat(), "equity": cash, "cash": cash,
            "invested": 0.0, "reserved_capital": 0.0,
            "realized_pnl": 0.0, "unrealized_pnl": 0.0, "open_positions": 0,
        }]
        exposure_values: list[float] = []
        session_index = {day: index for index, day in enumerate(sessions)}
        member_cache: dict[date, dict[str, Any]] = {}

        def members(day: date) -> dict[str, Any]:
            if day not in member_cache:
                member_cache[day] = {
                    row.security_id: row
                    for row in self.provider.members_on(day, self.config.universe_id)
                    if row.tradable
                }
            return member_cache[day]

        def mark_to_market(day: date) -> float:
            return cash + sum(position.quantity * position.last_price for position in positions.values())

        def execute_exit(position: Position, *, day: date, raw_price: float, reason: str,
                         exit_timestamp: datetime | None = None) -> None:
            nonlocal cash, transaction_costs_total
            fill = self.config.costs.fill_price(raw_price, side="sell")
            if fill <= 0:
                raise BacktestError("Sell-side slippage produced a non-positive fill price.")
            dp_key = (position.security_id, day)
            include_dp = dp_key not in charged_dp_debits
            exit_costs = self.config.costs.calculate(
                position.quantity * fill, side="sell", include_dp_charge=include_dp,
            )
            if exit_costs.dp_charge:
                charged_dp_debits.add(dp_key)
            cash += position.quantity * fill - exit_costs.total
            exit_slippage_cost = position.quantity * (raw_price - fill)
            transaction_costs_total += exit_costs.total + exit_slippage_cost
            gross = position.quantity * (raw_price - position.raw_entry_price)
            total_cost = (position.entry_costs.total + exit_costs.total
                          + position.entry_slippage_cost + exit_slippage_cost)
            net = gross - total_cost
            index = session_index[day]
            holding = index - position.entry_session_index + 1
            executed_at = exit_timestamp or datetime.combine(day, time(15, 30), tzinfo=zone)
            trades.append(TradeRecord(
                trade_id=f"{strategy.name}:{len(trades) + 1:06d}",
                strategy_version=position.strategy_version, security_id=position.security_id,
                symbol=position.symbol, universe_id=self.config.universe_id,
                signal_timestamp=position.signal_timestamp,
                entry_timestamp=position.entry_timestamp, entry_price=position.entry_price,
                quantity=position.quantity, initial_stop=position.initial_stop,
                initial_risk_rupees=position.initial_risk_rupees,
                initial_r_distance=position.initial_r_distance,
                exit_timestamp=executed_at,
                exit_price=fill, exit_reason=reason, gross_pnl=gross,
                costs=total_cost, net_pnl=net,
                R_multiple=net / position.initial_risk_rupees if position.initial_risk_rupees else 0.0,
                holding_period=holding,
                metadata={**position.metadata, "reason_codes": list(position.reason_codes),
                          "exit_time_resolution": "DAILY_BAR; 15:30 is the bar label, not inferred intraday execution time"},
                cost_breakdown={
                    "entry": position.entry_costs.total,
                    "exit": exit_costs.total,
                    "entry_slippage": position.entry_slippage_cost,
                    "exit_slippage": exit_slippage_cost,
                    "entry_components": position.entry_costs.as_dict(),
                    "exit_components": exit_costs.as_dict(),
                },
            ))
            events.append(BacktestEvent(
                executed_at, "POSITION_EXIT",
                position.security_id,
                {"reason": reason, "fill_price": fill, "quantity": position.quantity,
                 "gross_pnl": gross, "costs": total_cost, "net_pnl": net},
            ))
            positions.pop(position.security_id)

        def process_entry(signal: EntrySignal, day: date) -> None:
            nonlocal cash, transaction_costs_total
            eligible = members(day)
            if signal.security_id not in eligible:
                events.append(BacktestEvent(datetime.combine(day, time(9, 15), tzinfo=zone),
                    "ENTRY_REJECTED", signal.security_id, {"reason": "not PIT eligible on execution date"}))
                return
            if signal.security_id in positions:
                events.append(BacktestEvent(datetime.combine(day, time(9, 15), tzinfo=zone),
                    "ENTRY_REJECTED", signal.security_id, {"reason": "position already open"}))
                return
            if len(positions) >= self.config.max_open_positions:
                events.append(BacktestEvent(datetime.combine(day, time(9, 15), tzinfo=zone),
                    "ENTRY_REJECTED", signal.security_id, {"reason": "maximum open positions reached"}))
                return
            candle = indexed[signal.security_id].get(day)
            if candle is None:
                raise BacktestError(f"Missing execution-day candle for eligible {signal.security_id} on {day}.")
            raw_open = candle["open"]
            fill = self.config.costs.fill_price(raw_open, side="buy")
            if signal.target_price is not None and signal.target_price <= fill:
                events.append(BacktestEvent(datetime.combine(day, time(9, 15), tzinfo=zone),
                    "ENTRY_REJECTED", signal.security_id, {"reason": "target is not above executable entry"}))
                return
            equity = mark_to_market(day)
            sizing = self.risk_policy.size(
                equity=equity, available_cash=cash, entry_price=fill,
                stop_price=signal.stop_price, risk_fraction=self.config.max_risk_per_trade,
            )
            quantity = sizing.quantity
            while quantity > 0:
                entry_cost = self.config.costs.calculate(quantity * fill, side="buy")
                if quantity * fill + entry_cost.total <= cash + 1e-9:
                    break
                quantity -= 1
            if quantity <= 0:
                events.append(BacktestEvent(datetime.combine(day, time(9, 15), tzinfo=zone),
                    "ENTRY_REJECTED", signal.security_id,
                    {"reason": "zero quantity after risk and cash limits", "risk_budget": sizing.risk_budget}))
                return
            entry_cost = self.config.costs.calculate(quantity * fill, side="buy")
            risk_per_share = fill - signal.stop_price
            if risk_per_share <= 0:
                events.append(BacktestEvent(datetime.combine(day, time(9, 15), tzinfo=zone),
                    "ENTRY_REJECTED", signal.security_id, {"reason": "stop is not below executable entry"}))
                return
            security = eligible[signal.security_id]
            entry_timestamp = datetime.combine(day, time(9, 15), tzinfo=zone)
            cash -= quantity * fill + entry_cost.total
            entry_slippage = quantity * (fill - raw_open)
            transaction_costs_total += entry_cost.total + entry_slippage
            positions[signal.security_id] = Position(
                security_id=signal.security_id, symbol=security.symbol,
                entry_timestamp=entry_timestamp, entry_price=fill, raw_entry_price=raw_open,
                quantity=quantity,
                initial_stop=signal.stop_price, target_price=signal.target_price,
                initial_risk_rupees=risk_per_share * quantity,
                initial_r_distance=risk_per_share, entry_costs=entry_cost,
                entry_slippage_cost=entry_slippage,
                signal_timestamp=signal.signal_timestamp,
                strategy_version=signal.strategy_version,
                reason_codes=signal.reason_codes, metadata=dict(signal.metadata),
                entry_session_index=session_index[day], last_price=raw_open,
            )
            events.append(BacktestEvent(entry_timestamp, "POSITION_ENTRY", signal.security_id,
                {"symbol": security.symbol, "fill_price": fill, "quantity": quantity,
                 "risk_budget": sizing.risk_budget, "planned_risk": risk_per_share * quantity,
                 "costs": entry_cost.total}))

        for index, day in enumerate(sessions):
            todays_members = members(day)
            missing = sorted(security_id for security_id in todays_members if day not in indexed.get(security_id, {}))
            if missing:
                raise BacktestError(
                    f"Missing daily candles for {len(missing)} PIT-eligible securities on {day}: {', '.join(missing[:5])}."
                )
            # Prior-close exits fill before entries at today's open.
            for signal in sorted(pending_exits.pop(day, []), key=lambda item: item.security_id):
                position = positions.get(signal.security_id)
                if position is None:
                    events.append(BacktestEvent(datetime.combine(day, time(9, 15), tzinfo=zone),
                        "EXIT_REJECTED", signal.security_id, {"reason": "position no longer open"}))
                    continue
                candle = indexed[signal.security_id].get(day)
                if candle is None:
                    raise BacktestError(f"Missing exit-day candle for {signal.security_id} on {day}.")
                execute_exit(position, day=day, raw_price=candle["open"],
                             reason=f"STRATEGY_EXIT:{signal.reason_code}",
                             exit_timestamp=datetime.combine(day, time(9, 15), tzinfo=zone))

            # Use opening prices to value remaining holdings and resolve overnight
            # stop/target gaps before sizing any new entries against equity.
            for security_id in sorted(tuple(positions)):
                position = positions[security_id]
                candle = indexed[security_id].get(day)
                if candle is None:
                    raise BacktestError(f"Missing opening candle for position {security_id} on {day}.")
                opening = candle["open"]
                if opening <= position.initial_stop:
                    execute_exit(position, day=day, raw_price=opening, reason="STOP",
                                 exit_timestamp=datetime.combine(day, time(9, 15), tzinfo=zone))
                elif position.target_price is not None and opening >= position.target_price:
                    execute_exit(position, day=day, raw_price=opening, reason="TARGET",
                                 exit_timestamp=datetime.combine(day, time(9, 15), tzinfo=zone))
                else:
                    position.last_price = opening

            # Next-session entries are executed separately from historical broker APIs.
            for signal in sorted(pending.pop(day, []), key=lambda item: item.security_id):
                process_entry(signal, day)

            # Existing and just-opened positions are monitored for this session.
            for security_id in sorted(tuple(positions)):
                position = positions[security_id]
                candle = indexed[security_id].get(day)
                if candle is None:
                    raise BacktestError(f"Missing daily candle for open position {security_id} on {day}.")
                position.last_price = candle["close"]
                hit_stop = candle["low"] <= position.initial_stop
                hit_target = position.target_price is not None and candle["high"] >= position.target_price
                if hit_stop:
                    # If stop and target are both inside one daily range, stop-first is conservative.
                    stop_fill = candle["open"] if candle["open"] <= position.initial_stop else position.initial_stop
                    execute_exit(position, day=day, raw_price=stop_fill, reason="STOP")
                elif hit_target:
                    assert position.target_price is not None
                    target_fill = candle["open"] if candle["open"] >= position.target_price else position.target_price
                    execute_exit(position, day=day, raw_price=target_fill, reason="TARGET")

            current_equity = mark_to_market(day)
            invested = sum(position.quantity * position.last_price for position in positions.values())
            reserved = sum(position.quantity * position.entry_price for position in positions.values())
            realized = sum(trade.net_pnl for trade in trades)
            unrealized = sum(
                position.quantity * (position.last_price - position.raw_entry_price)
                - position.entry_costs.total - position.entry_slippage_cost
                for position in positions.values()
            )
            exposure_values.append(invested / current_equity if current_equity else 0.0)
            equity_curve.append({"date": day.isoformat(), "equity": current_equity, "cash": cash,
                                 "invested": invested, "reserved_capital": reserved,
                                 "realized_pnl": realized, "unrealized_pnl": unrealized,
                                 "open_positions": len(positions)})

            # EOD information set: current daily close is available, no future daily rows are exposed.
            eod = datetime.combine(day, time(15, 30), tzinfo=zone)
            eligible = tuple(members(day).values())
            daily_history = _AsOfDailyHistory(history_days, history_rows, day)
            weekly_visible: dict[str, Mapping[str, Any] | None] = {}
            for security_id, (completion_times, weekly_rows) in weekly_index.items():
                visible_count = bisect_right(completion_times, eod)
                weekly_visible[security_id] = weekly_rows[visible_count - 1] if visible_count else None
            context = StrategyContext(
                timestamp=eod, session_date=day, universe_id=self.config.universe_id,
                next_session_open=(datetime.combine(sessions[index + 1], time(9, 15), tzinfo=zone)
                                   if index + 1 < len(sessions) else None),
                eligible=eligible, daily_history=daily_history,
                latest_completed_weekly=weekly_visible,
                open_security_ids=frozenset(positions),
                open_positions={security_id: _position_snapshot(position)
                                for security_id, position in positions.items()},
            )
            exit_signals = list(getattr(strategy, "generate_exits", lambda _: ())(context))
            entry_signals = list(strategy.generate_entries(context))
            exit_ids = {signal.security_id for signal in exit_signals}
            if len(exit_ids) != len(exit_signals):
                raise BacktestError("Strategy emitted duplicate exit signals for one security.")
            entry_ids = {signal.security_id for signal in entry_signals}
            if exit_ids & entry_ids:
                raise BacktestError("Strategy emitted entry and exit signals for the same security at one timestamp.")
            for signal in sorted(exit_signals, key=lambda item: item.security_id):
                if signal.signal_timestamp != eod:
                    raise BacktestError("Exit signal timestamp must equal the current session close.")
                if signal.security_id not in positions:
                    raise BacktestError("Strategy emitted an exit for a security without an open position.")
                if index + 1 >= len(sessions):
                    events.append(BacktestEvent(eod, "EXIT_SIGNAL_EXPIRED", signal.security_id,
                                                {"reason": "no next session in evaluation range"}))
                    continue
                expected_execution = datetime.combine(sessions[index + 1], time(9, 15), tzinfo=zone)
                if signal.execution_timestamp != expected_execution:
                    raise BacktestError("Exit execution timestamp must be the next available session open.")
                pending_exits.setdefault(sessions[index + 1], []).append(signal)
                events.append(BacktestEvent(eod, "EXIT_SIGNAL", signal.security_id,
                    {"execution_timestamp": expected_execution.isoformat(), "reason_code": signal.reason_code}))

            signals = entry_signals
            seen: set[str] = set()
            for signal in sorted(signals, key=lambda item: item.security_id):
                if signal.security_id in seen:
                    raise BacktestError(f"Strategy emitted duplicate entry signals for {signal.security_id}.")
                seen.add(signal.security_id)
                if signal.signal_timestamp != eod:
                    raise BacktestError("Entry signal timestamp must equal the current session close.")
                if index + 1 >= len(sessions):
                    events.append(BacktestEvent(eod, "SIGNAL_EXPIRED", signal.security_id,
                                                {"reason": "no next session in evaluation range"}))
                    continue
                expected_execution = datetime.combine(sessions[index + 1], time(9, 15), tzinfo=zone)
                if signal.execution_timestamp != expected_execution:
                    raise BacktestError("Entry execution timestamp must be the next available session open.")
                eligible_ids = {security.security_id for security in eligible}
                if signal.security_id not in eligible_ids:
                    raise BacktestError("Strategy emitted an entry for a security outside the PIT universe.")
                pending.setdefault(sessions[index + 1], []).append(signal)
                events.append(BacktestEvent(eod, "ENTRY_SIGNAL", signal.security_id,
                    {"execution_timestamp": expected_execution.isoformat(),
                     "reason_codes": list(signal.reason_codes)}))

        ending_equity = mark_to_market(sessions[-1])
        ending_reserved = sum(position.quantity * position.entry_price for position in positions.values())
        ending_realized = sum(trade.net_pnl for trade in trades)
        ending_unrealized = sum(
            position.quantity * (position.last_price - position.raw_entry_price)
            - position.entry_costs.total - position.entry_slippage_cost
            for position in positions.values()
        )
        config_snapshot = {
            "engine_version": ENGINE_VERSION,
            "universe_id": self.config.universe_id,
            "start_date": self.config.start_date.isoformat(), "end_date": self.config.end_date.isoformat(),
            "initial_capital": self.config.initial_capital,
            "max_risk_per_trade": self.config.max_risk_per_trade,
            "max_open_positions": self.config.max_open_positions,
            "timezone_name": self.config.timezone_name,
            "cost_model": _costs_snapshot(self.config.costs),
            "operating_cost_model": asdict(self.config.operating_costs),
            "strategy": {
                "name": strategy.name,
                "version": strategy.version,
                "parameters": getattr(strategy, "parameters", {}),
            },
        }
        config_hash = _canonical_hash(config_snapshot)
        identity = {"config_hash": config_hash, "dataset_version": dataset_version}
        metadata = RunMetadata(
            run_id=_canonical_hash(identity)[:16], strategy_name=strategy.name,
            strategy_version=strategy.version, config_hash=config_hash,
            config_snapshot=config_snapshot, universe_id=self.config.universe_id,
            evaluation_start=self.config.start_date, evaluation_end=self.config.end_date,
            dataset_version=dataset_version, starting_capital=self.config.initial_capital,
            cost_model=_costs_snapshot(self.config.costs), created_at=datetime.now(timezone.utc),
        )
        months, kite_cost, hosting_cost, other_cost, operating_total = self.config.operating_costs.total_for_period(
            self.config.start_date, self.config.end_date,
        )
        metrics = calculate_metrics(
            starting_equity=self.config.initial_capital, ending_equity=ending_equity,
            trades=trades, equity_curve=equity_curve, exposure_values=exposure_values,
            transaction_costs_total=transaction_costs_total,
            operating_costs={
                "kite_api_cost": kite_cost, "hosting_cost": hosting_cost,
                "other_operating_cost": other_cost, "operating_costs_total": operating_total,
                "charged_calendar_months": months,
                "apply_to_economic_results": self.config.operating_costs.apply_to_economic_results,
            },
        )
        return BacktestResult(
            metadata=metadata, trades=tuple(trades), events=tuple(events),
            equity_curve=tuple(equity_curve), metrics=metrics, remaining_cash=cash,
            reserved_capital=ending_reserved, realized_pnl=ending_realized,
            unrealized_pnl=ending_unrealized,
            open_positions=tuple(_position_snapshot(position) for position in positions.values()),
        )
