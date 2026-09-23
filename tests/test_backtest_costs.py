from __future__ import annotations

from datetime import date

import pytest
import yaml

from src.backtest.models import CostModel, OperatingCostModel
from src.backtest.run import _cost_model, _operating_cost_model
from src.configuration import DEFAULT_SETTINGS_PATH
from tests.test_backtest_engine import candle, setup, SignalStrategy


def test_current_zerodha_delivery_rates_and_provenance() -> None:
    costs = CostModel.current_zerodha_equity_delivery_2026()
    assert costs.scenario == "CURRENT_ZERODHA_EQUITY_DELIVERY_2026"
    assert costs.verification_label == "CURRENT_ZERODHA_EQUITY_DELIVERY_2026"
    assert costs.verified_on == "2026-09-24"
    assert costs.brokerage_rate == 0
    assert costs.stt_buy_rate == costs.stt_sell_rate == 0.001
    assert costs.nse_transaction_rate == pytest.approx(0.0000307)
    assert costs.gst_rate == 0.18
    assert costs.sebi_rate == pytest.approx(0.000001)
    assert costs.stamp_buy_rate == pytest.approx(0.00015)
    assert costs.ipft_rate == pytest.approx(0.000000001)
    assert costs.dp_charge_per_sell == 15.34
    assert "zerodha.com/charges" in costs.provenance


def test_repository_defaults_select_current_transaction_and_operating_costs() -> None:
    raw = yaml.safe_load(DEFAULT_SETTINGS_PATH.read_text(encoding="utf-8"))["backtest"]
    costs = _cost_model(raw["transaction_costs"])
    operating = _operating_cost_model(raw["operating_costs"])
    assert costs.scenario == "CURRENT_ZERODHA_EQUITY_DELIVERY_2026"
    assert costs.brokerage_rate == 0
    assert operating.kite_connect_monthly == 500
    assert operating.hosting_monthly == operating.other_monthly == 0
    assert operating.apply_to_economic_results is True


def test_buy_sell_stt_stamp_dp_and_gst_components() -> None:
    costs = CostModel.current_zerodha_equity_delivery_2026()
    buy = costs.calculate(100000, side="buy")
    sell = costs.calculate(105000, side="sell", include_dp_charge=True)

    assert buy.brokerage == 0
    assert buy.stt == 100
    assert sell.stt == 105
    assert buy.stamp_duty == pytest.approx(15.0)
    assert sell.stamp_duty == 0
    assert buy.dp_charge == 0
    assert sell.dp_charge == 15.34
    assert buy.exchange_charges == pytest.approx(3.07)
    assert buy.sebi_charges == pytest.approx(0.1)
    assert buy.ipft_charges == pytest.approx(0.0001)
    taxable_buy = buy.brokerage + buy.exchange_charges + buy.sebi_charges + buy.ipft_charges
    assert buy.gst == pytest.approx(taxable_buy * 0.18)
    assert buy.gst != pytest.approx((taxable_buy + buy.stt + buy.stamp_duty) * 0.18)
    taxable_sell = sell.brokerage + sell.exchange_charges + sell.sebi_charges + sell.ipft_charges
    assert sell.gst == pytest.approx(taxable_sell * 0.18)
    assert sell.gst != pytest.approx((taxable_sell + sell.dp_charge) * 0.18)


def test_stt_rounding_and_trade_slippage_are_applied_separately() -> None:
    costs = CostModel.current_zerodha_equity_delivery_2026()
    assert costs.calculate(500, side="buy").stt == 1  # ₹0.50 rounds up as published.
    assert costs.calculate(499, side="sell").stt == 0
    assert costs.fill_price(100, side="buy") == 100
    slipped = CostModel(slippage_bps=10, scenario="SLIPPAGE_TEST", verification_label="TEST")
    assert slipped.fill_price(100, side="buy") == pytest.approx(100.1)
    assert slipped.fill_price(100, side="sell") == pytest.approx(99.9)


def test_current_cost_total_reconciles_with_trade_net_pnl() -> None:
    costs = CostModel.current_zerodha_equity_delivery_2026()
    engine, data, days = setup(costs=costs)
    data["A"][0] = candle(days[0], low=99)
    data["A"][1] = candle(days[1], high=112, low=98, close=111)
    result = engine.run(
        strategy=SignalStrategy({days[0]: ("A",)}, stop=95, target=110),
        daily_data=data,
    )
    repeated = engine.run(
        strategy=SignalStrategy({days[0]: ("A",)}, stop=95, target=110),
        daily_data=data,
    )
    trade = result.trades[0]
    buy_cost = costs.calculate(trade.quantity * trade.entry_price, side="buy").total
    sell_cost = costs.calculate(trade.quantity * trade.exit_price, side="sell", include_dp_charge=True).total
    slippage = trade.cost_breakdown["entry_slippage"] + trade.cost_breakdown["exit_slippage"]
    assert trade.costs == pytest.approx(buy_cost + sell_cost + slippage)
    assert trade.net_pnl == pytest.approx(trade.gross_pnl - trade.costs)
    assert result.metrics["transaction_costs_total"] == pytest.approx(trade.costs)
    assert repeated.metadata.run_id == result.metadata.run_id
    assert repeated.trades == result.trades
    assert repeated.metrics == result.metrics


def test_fixed_operating_costs_do_not_change_cash_fills_quantity_or_trade_r() -> None:
    engine_raw, data_raw, days = setup()
    engine_charged, data_charged, _ = setup(
        operating_costs=OperatingCostModel(kite_connect_monthly=500,
                                           hosting_monthly=120, other_monthly=50),
    )
    for data in (data_raw, data_charged):
        data["A"][0] = candle(days[0], low=99)
        data["A"][1] = candle(days[1], high=112, low=98, close=111)
    strategy = SignalStrategy({days[0]: ("A",)}, stop=95, target=110)
    raw = engine_raw.run(strategy=strategy, daily_data=data_raw)
    charged = engine_charged.run(strategy=strategy, daily_data=data_charged)

    raw_entry = next(event for event in raw.events if event.event_type == "POSITION_ENTRY")
    charged_entry = next(event for event in charged.events if event.event_type == "POSITION_ENTRY")
    assert raw_entry.details["quantity"] == charged_entry.details["quantity"]
    assert raw.remaining_cash == pytest.approx(charged.remaining_cash)
    assert raw.trades[0].net_pnl == pytest.approx(charged.trades[0].net_pnl)
    assert raw.trades[0].R_multiple == pytest.approx(charged.trades[0].R_multiple)
    assert charged.metrics["trading_net_return"] == pytest.approx(raw.metrics["trading_net_return"])
    assert charged.metrics["kite_api_cost"] == 500
    assert charged.metrics["hosting_cost"] == 120
    assert charged.metrics["other_operating_cost"] == 50
    assert charged.metrics["charged_calendar_months"] == 1
    assert charged.metrics["operating_costs_total"] == 670
    assert charged.metrics["economic_net_profit"] == pytest.approx(
        charged.metrics["trading_ending_equity"] - 10000 - 670
    )
    assert charged.metrics["economic_net_return"] == pytest.approx(
        charged.metrics["economic_net_profit"] / 10000
    )


def test_operating_cost_period_charges_each_calendar_month_touched() -> None:
    costs = OperatingCostModel(kite_connect_monthly=500)
    assert costs.total_for_period(date(2025, 1, 31), date(2025, 2, 1)) == (2, 1000, 0, 0, 1000)
    assert costs.total_for_period(date(2025, 1, 1), date(2025, 12, 31)) == (12, 6000, 0, 0, 6000)


def test_raw_zero_cost_model_is_explicit() -> None:
    raw = CostModel.raw_zero()
    assert raw.scenario == "RAW_ZERO_COST"
    assert raw.calculate(100000, side="buy").total == 0
    assert raw.calculate(100000, side="sell", include_dp_charge=True).total == 0
