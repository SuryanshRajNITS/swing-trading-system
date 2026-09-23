"""Deterministic portfolio and trade metrics."""

from __future__ import annotations

from statistics import median
from typing import Any, Iterable, Mapping

from src.backtest.models import TradeRecord


def calculate_metrics(
    *, starting_equity: float, ending_equity: float, trades: Iterable[TradeRecord],
    equity_curve: list[dict[str, Any]], exposure_values: list[float],
    transaction_costs_total: float = 0.0,
    operating_costs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    ledger = list(trades)
    wins = [trade.net_pnl for trade in ledger if trade.net_pnl > 0]
    losses = [trade.net_pnl for trade in ledger if trade.net_pnl < 0]
    net_values = [trade.net_pnl for trade in ledger]
    r_values = [trade.R_multiple for trade in ledger]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    peak = starting_equity
    max_drawdown = 0.0
    max_drawdown_pct = 0.0
    for point in equity_curve:
        value = float(point["equity"])
        peak = max(peak, value)
        drawdown = peak - value
        max_drawdown = max(max_drawdown, drawdown)
        if peak:
            max_drawdown_pct = max(max_drawdown_pct, drawdown / peak)
    operating = operating_costs or {
        "kite_api_cost": 0.0, "hosting_cost": 0.0, "other_operating_cost": 0.0,
        "operating_costs_total": 0.0, "charged_calendar_months": 0,
        "apply_to_economic_results": False,
    }
    trading_profit = ending_equity - starting_equity
    applied_operating_cost = (float(operating["operating_costs_total"])
                              if operating["apply_to_economic_results"] else 0.0)
    economic_profit = trading_profit - applied_operating_cost
    return {
        "starting_equity": starting_equity,
        "ending_equity": ending_equity,
        "net_return": ending_equity / starting_equity - 1.0,
        "trading_ending_equity": ending_equity,
        "trading_net_return": ending_equity / starting_equity - 1.0,
        "transaction_costs_total": transaction_costs_total,
        "kite_api_cost": operating["kite_api_cost"],
        "hosting_cost": operating["hosting_cost"],
        "other_operating_cost": operating["other_operating_cost"],
        "charged_calendar_months": operating["charged_calendar_months"],
        "operating_costs_total": operating["operating_costs_total"],
        "apply_operating_costs_to_economic_results": operating["apply_to_economic_results"],
        "economic_net_profit": economic_profit,
        "economic_net_return": economic_profit / starting_equity,
        "number_of_trades": len(ledger),
        "win_rate": len(wins) / len(ledger) if ledger else 0.0,
        "average_win": sum(wins) / len(wins) if wins else 0.0,
        "average_loss": sum(losses) / len(losses) if losses else 0.0,
        "profit_factor": gross_profit / gross_loss if gross_loss else None,
        "expectancy": sum(net_values) / len(net_values) if ledger else 0.0,
        "average_R": sum(r_values) / len(r_values) if r_values else 0.0,
        "median_R": median(r_values) if r_values else 0.0,
        "maximum_drawdown": max_drawdown,
        "maximum_drawdown_pct": max_drawdown_pct,
        "exposure": sum(exposure_values) / len(exposure_values) if exposure_values else 0.0,
        "average_holding_period": sum(trade.holding_period for trade in ledger) / len(ledger) if ledger else 0.0,
    }
