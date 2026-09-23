"""Deterministic cash-equity position sizing."""

from __future__ import annotations

import math

from src.decision.interfaces import SizingDecision


class FixedFractionRiskPolicy:
    """Risk-budget sizing followed by a strict available-cash cap."""

    def size(
        self, *, equity: float, available_cash: float, entry_price: float,
        stop_price: float, risk_fraction: float,
    ) -> SizingDecision:
        risk_per_share = entry_price - stop_price
        budget = max(0.0, equity) * risk_fraction
        if risk_per_share <= 0 or entry_price <= 0 or risk_fraction <= 0:
            return SizingDecision(0, budget, risk_per_share, "invalid entry or stop")
        quantity = min(math.floor(budget / risk_per_share), math.floor(max(0.0, available_cash) / entry_price))
        reason = "approved" if quantity else "risk or cash budget permits zero shares"
        return SizingDecision(quantity, budget, risk_per_share, reason)

