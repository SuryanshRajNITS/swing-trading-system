from __future__ import annotations

import pytest

from src.backtest.top_down_swing_v2_signal_diagnostic import (
    FROZEN_EXECUTION_FIELDS,
    accounting_views,
)


def _canonical_trade(*, exit_reason: str = "GAP_STOP") -> dict[str, object]:
    gross = -707.85 if exit_reason.endswith("STOP") else 4575.0
    costs = 236.737067629937
    return {
        "status": "CLOSED",
        "setup_event_id": "SZ2-zone:2023-10-25",
        "setup_instance_id": "NSEEQ-0046-SUPPORT_REACTION-2023-10-25",
        "symbol": "INDUSINDBK",
        "entry_timestamp": "2023-10-25T14:15:00+05:30",
        "entry_raw_open": 232.5,
        "entry_price": 232.5,
        "stop_price": 231.03014285714283,
        "target_price": 243.16678571428568,
        # This quantity was sized once by the canonical current-cost run.
        # A separate no-fee sizing pass could afford 430, but must not replace 429.
        "quantity": 429,
        "exit_timestamp": "2023-10-26T09:15:00+05:30",
        "exit_reason": exit_reason,
        "exit_price": 230.85 if exit_reason.endswith("STOP") else 243.16678571428568,
        "raw_exit_price": 230.85 if exit_reason.endswith("STOP") else 243.16678571428568,
        "gap_handling": exit_reason.startswith("GAP_"),
        "holding_sessions": 2,
        "gross_pnl": gross,
        "gross_R": gross / (429 * (232.5 - 231.03014285714283)),
        "net_pnl": gross - costs,
        "transaction_costs": costs,
        "entry_costs": 120.0,
        "exit_costs": costs - 120.0,
        "entry_cost_breakdown": {"total": 120.0},
        "exit_cost_breakdown": {"total": costs - 120.0, "dp_charge": 15.34},
        "entry_slippage_cost": 0.0,
        "exit_slippage_cost": 0.0,
        "transaction_cost_scenario": "CURRENT_ZERODHA_EQUITY_DELIVERY_2026",
    }


def test_raw_view_reuses_current_cost_quantity_despite_fee_affordability_edge() -> None:
    current, raw = accounting_views(_canonical_trade())

    assert current["quantity"] == 429
    assert raw["quantity"] == 429


@pytest.mark.parametrize("exit_reason", ["GAP_STOP", "GAP_TARGET"])
def test_raw_view_reuses_canonical_entry_exit_and_gap_decision(exit_reason: str) -> None:
    current, raw = accounting_views(_canonical_trade(exit_reason=exit_reason))

    assert all(current[field] == raw[field] for field in FROZEN_EXECUTION_FIELDS)
    assert current["entry_timestamp"] == "2023-10-25T14:15:00+05:30"
    assert current["exit_timestamp"] == "2023-10-26T09:15:00+05:30"
    assert current["exit_reason"] == exit_reason
    assert current["gap_handling"] is True


def test_raw_view_changes_only_explicit_cost_accounting_not_slippage() -> None:
    current, raw = accounting_views(_canonical_trade())

    for field in FROZEN_EXECUTION_FIELDS:
        assert current[field] == raw[field]
    assert current["entry_slippage_cost"] == raw["entry_slippage_cost"]
    assert current["exit_slippage_cost"] == raw["exit_slippage_cost"]
    assert current["transaction_costs"] > 0
    assert raw["transaction_costs"] == 0
    assert raw["entry_cost_breakdown"] == {}
    assert raw["exit_cost_breakdown"] == {}


def test_gross_pnl_is_identical_and_net_cost_difference_reconciles() -> None:
    current, raw = accounting_views(_canonical_trade())

    assert current["gross_pnl"] == raw["gross_pnl"]
    assert raw["net_pnl"] == current["gross_pnl"]
    assert current["net_pnl"] == pytest.approx(current["gross_pnl"] - current["transaction_costs"])
    assert current["gross_pnl"] - raw["net_pnl"] == 0
    assert raw["net_pnl"] - current["net_pnl"] == pytest.approx(current["transaction_costs"])


def test_bad_canonical_cost_reconciliation_fails_closed() -> None:
    trade = _canonical_trade()
    trade["net_pnl"] = -1.0

    with pytest.raises(ValueError, match="Canonical cost reconciliation failed"):
        accounting_views(trade)


def test_open_trade_keeps_status_and_unrealized_mark_without_realized_pnl() -> None:
    trade = _canonical_trade()
    trade.update({
        "status": "OPEN",
        "gross_pnl": "",
        "net_pnl": "",
        "exit_timestamp": "",
        "exit_reason": "OPEN_AT_PERIOD_END",
        "exit_price": "",
        "raw_exit_price": None,
        "gap_handling": False,
        "holding_sessions": 18,
        "unrealized_gross_pnl": 325.5,
        "mark_timestamp": "2024-12-31T15:15:00+05:30",
        "transaction_costs": 120.0,
        "exit_costs": 0.0,
        "exit_cost_breakdown": {},
    })

    current, raw = accounting_views(trade)

    assert current["status"] == raw["status"] == "OPEN"
    assert current["unrealized_gross_pnl"] == raw["zero_cost_unrealized_gross_pnl"] == 325.5
    assert current["net_pnl"] is None
    assert raw["net_pnl"] is None
    assert current["quantity"] == raw["quantity"] == 429
