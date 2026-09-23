"""Small contracts for decisions shared across backtest, paper, and live modes.

These interfaces describe decisions only. Broker orders and historical fills are
implemented by separate adapters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Mapping, Protocol, Sequence


@dataclass(frozen=True)
class EntrySignal:
    security_id: str
    symbol: str
    signal_timestamp: datetime
    execution_timestamp: datetime
    stop_price: float
    target_price: float | None = None
    reason_codes: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    strategy_version: str = "0.0.0"


@dataclass(frozen=True)
class ExitSignal:
    security_id: str
    signal_timestamp: datetime
    execution_timestamp: datetime
    reason_code: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    strategy_version: str = "0.0.0"


@dataclass(frozen=True)
class StrategyContext:
    timestamp: datetime
    session_date: date
    next_session_open: datetime | None
    universe_id: str
    eligible: tuple[Any, ...]
    daily_history: Mapping[str, Sequence[Mapping[str, Any]]]
    latest_completed_weekly: Mapping[str, Mapping[str, Any] | None]
    open_security_ids: frozenset[str]
    open_positions: Mapping[str, Mapping[str, Any]]


class Strategy(Protocol):
    """Create deterministic decisions from data legally available at timestamp."""

    name: str
    version: str
    parameters: Mapping[str, Any]

    def generate_entries(self, context: StrategyContext) -> Sequence[EntrySignal]: ...

    def generate_exits(self, context: StrategyContext) -> Sequence[ExitSignal]: ...


@dataclass(frozen=True)
class SizingDecision:
    quantity: int
    risk_budget: float
    risk_per_share: float
    reason: str


class RiskPolicy(Protocol):
    """Pure position-sizing decision, independent of broker or fill mechanism."""

    def size(
        self, *, equity: float, available_cash: float, entry_price: float,
        stop_price: float, risk_fraction: float,
    ) -> SizingDecision: ...
