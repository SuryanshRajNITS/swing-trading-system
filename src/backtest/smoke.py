"""Tiny deterministic smoke strategy used only to exercise the engine."""

from __future__ import annotations

from typing import Sequence

from src.decision.interfaces import EntrySignal, StrategyContext


class SmokeStrategy:
    name = "smoke"
    version = "0.1.0"

    def __init__(self, *, target_r: float = 2.0) -> None:
        if target_r <= 0:
            raise ValueError("target_r must be positive.")
        self.target_r = target_r
        self.parameters = {"target_r": target_r}

    def generate_entries(self, context: StrategyContext) -> Sequence[EntrySignal]:
        if context.next_session_open is None:
            return ()
        value = context.daily_history
        signals: list[EntrySignal] = []
        for security in context.eligible:
            candles = value.get(security.security_id, ())
            observed = candles
            if len(observed) < 2 or security.security_id in context.open_security_ids:
                continue
            today, previous = observed[-1], observed[-2]
            if today["close"] <= previous["close"] or today["low"] >= today["close"]:
                continue
            risk_distance = today["close"] - today["low"]
            # This is an engine exercise rule, not a candidate real strategy.
            signals.append(EntrySignal(
                security_id=security.security_id, symbol=security.symbol,
                signal_timestamp=context.timestamp, execution_timestamp=context.next_session_open,
                stop_price=today["low"], target_price=today["close"] + self.target_r * risk_distance,
                reason_codes=("SMOKE_CLOSE_ABOVE_PRIOR_CLOSE",),
                metadata={"prior_close": previous["close"], "target_r": self.target_r},
                strategy_version=self.version,
            ))
        return signals
