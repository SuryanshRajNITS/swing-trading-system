"""Backtest configuration, portfolio state, ledger, and cost-model records."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
import math
from typing import Any, Mapping


@dataclass(frozen=True)
class CostBreakdown:
    brokerage: float = 0.0
    stt: float = 0.0
    exchange_charges: float = 0.0
    ipft_charges: float = 0.0
    gst: float = 0.0
    sebi_charges: float = 0.0
    stamp_duty: float = 0.0
    dp_charge: float = 0.0

    @property
    def total(self) -> float:
        return sum((self.brokerage, self.stt, self.exchange_charges, self.ipft_charges,
                    self.gst, self.sebi_charges, self.stamp_duty, self.dp_charge))

    def as_dict(self) -> dict[str, float]:
        return {"brokerage": self.brokerage, "stt": self.stt,
                "exchange_charges": self.exchange_charges, "ipft_charges": self.ipft_charges,
                "gst": self.gst, "sebi_charges": self.sebi_charges,
                "stamp_duty": self.stamp_duty, "dp_charge": self.dp_charge,
                "total": self.total}


@dataclass(frozen=True)
class CostModel:
    """Configurable NSE equity-delivery charges and slippage.

    Rates are fractions of turnover. DP is a rupee amount inclusive of GST and
    is charged once per security per sell date by the engine. GST applies to
    brokerage, NSE transaction, SEBI, and NSE IPFT charges; it does not apply to
    STT or stamp duty. This current schedule is not a historical schedule.
    """

    brokerage_rate: float = 0.0
    stt_buy_rate: float = 0.001
    stt_sell_rate: float = 0.001
    nse_transaction_rate: float = 0.0000307
    gst_rate: float = 0.18
    sebi_rate: float = 0.000001
    stamp_buy_rate: float = 0.00015
    ipft_rate: float = 0.000000001
    dp_charge_per_sell: float = 15.34
    slippage_bps: float = 0.0
    scenario: str = "CURRENT_ZERODHA_EQUITY_DELIVERY_2026"
    verification_label: str = "CURRENT_ZERODHA_EQUITY_DELIVERY_2026"
    verified_on: str = "2026-09-24"
    provenance: str = "https://zerodha.com/charges"

    def __post_init__(self) -> None:
        rates = (self.brokerage_rate, self.stt_buy_rate, self.stt_sell_rate,
                 self.nse_transaction_rate, self.gst_rate, self.sebi_rate,
                 self.stamp_buy_rate, self.ipft_rate, self.dp_charge_per_sell,
                 self.slippage_bps)
        if any(not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 for value in rates):
            raise ValueError("Cost rates and slippage must be non-negative numbers.")
        fractions = (self.brokerage_rate, self.stt_buy_rate, self.stt_sell_rate,
                     self.nse_transaction_rate, self.gst_rate, self.sebi_rate,
                     self.stamp_buy_rate, self.ipft_rate)
        if any(value > 1 for value in fractions) or self.slippage_bps >= 10000:
            raise ValueError("Cost fractions must be at most one and slippage below 10,000 bps.")

    @classmethod
    def raw_zero(cls) -> "CostModel":
        return cls(
            brokerage_rate=0.0, stt_buy_rate=0.0, stt_sell_rate=0.0,
            nse_transaction_rate=0.0, gst_rate=0.0, sebi_rate=0.0,
            stamp_buy_rate=0.0, ipft_rate=0.0, dp_charge_per_sell=0.0,
            slippage_bps=0.0, scenario="RAW_ZERO_COST",
            verification_label="EXPLICIT_RAW_ZERO_COST", verified_on="",
            provenance="Explicitly requested raw zero-cost comparator.",
        )

    @classmethod
    def current_zerodha_equity_delivery_2026(cls) -> "CostModel":
        return cls(provenance=(
            "https://zerodha.com/charges; "
            "https://support.zerodha.com/category/account-opening/resident-individual/ri-charges/articles/what-do-dp-charges-mean"
        ))

    def fill_price(self, price: float, *, side: str) -> float:
        if side not in {"buy", "sell"} or price <= 0:
            raise ValueError("A positive price and buy/sell side are required.")
        impact = self.slippage_bps / 10000.0
        return price * (1.0 + impact if side == "buy" else 1.0 - impact)

    def calculate(self, turnover: float, *, side: str, include_dp_charge: bool = False) -> CostBreakdown:
        if turnover < 0 or side not in {"buy", "sell"}:
            raise ValueError("A non-negative turnover and buy/sell side are required.")
        brokerage = turnover * self.brokerage_rate
        stt_unrounded = turnover * (self.stt_buy_rate if side == "buy" else self.stt_sell_rate)
        # Zerodha's published STT calculator rounds to nearest rupee; .50 rounds up.
        stt = float(int(stt_unrounded + 0.5))
        exchange = turnover * self.nse_transaction_rate
        ipft = turnover * self.ipft_rate
        sebi = turnover * self.sebi_rate
        gst = (brokerage + exchange + ipft + sebi) * self.gst_rate
        return CostBreakdown(
            brokerage=brokerage,
            stt=stt,
            exchange_charges=exchange,
            ipft_charges=ipft,
            gst=gst,
            sebi_charges=sebi,
            stamp_duty=turnover * self.stamp_buy_rate if side == "buy" else 0.0,
            dp_charge=self.dp_charge_per_sell if side == "sell" and include_dp_charge else 0.0,
        )


@dataclass(frozen=True)
class OperatingCostModel:
    """Fixed monthly system costs, independent of broker cash and trade records."""

    kite_connect_monthly: float = 500.0
    hosting_monthly: float = 0.0
    other_monthly: float = 0.0
    apply_to_economic_results: bool = True

    def __post_init__(self) -> None:
        if any(not math.isfinite(value) or value < 0 for value in (self.kite_connect_monthly,
                                                                   self.hosting_monthly,
                                                                   self.other_monthly)):
            raise ValueError("Monthly operating costs must be non-negative.")

    def total_for_period(self, start: date, end: date) -> tuple[int, float, float, float, float]:
        if start > end:
            raise ValueError("Operating-cost period start must not be after end.")
        months = (end.year - start.year) * 12 + end.month - start.month + 1
        kite = months * self.kite_connect_monthly
        hosting = months * self.hosting_monthly
        other = months * self.other_monthly
        return months, kite, hosting, other, kite + hosting + other


@dataclass(frozen=True)
class BacktestConfig:
    universe_id: str
    start_date: date
    end_date: date
    initial_capital: float = 10000.0
    max_risk_per_trade: float = 0.01
    max_open_positions: int = 10
    timezone_name: str = "Asia/Kolkata"
    costs: CostModel = field(default_factory=CostModel.current_zerodha_equity_delivery_2026)
    operating_costs: OperatingCostModel = field(default_factory=OperatingCostModel)

    def __post_init__(self) -> None:
        if self.start_date > self.end_date:
            raise ValueError("Backtest start_date must not be after end_date.")
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive.")
        if not 0 < self.max_risk_per_trade <= 1:
            raise ValueError("max_risk_per_trade must be in (0, 1].")
        if self.max_open_positions < 1:
            raise ValueError("max_open_positions must be at least one.")


@dataclass
class Position:
    security_id: str
    symbol: str
    entry_timestamp: datetime
    entry_price: float
    raw_entry_price: float
    quantity: int
    initial_stop: float
    target_price: float | None
    initial_risk_rupees: float
    initial_r_distance: float
    entry_costs: CostBreakdown
    entry_slippage_cost: float
    signal_timestamp: datetime
    strategy_version: str
    reason_codes: tuple[str, ...]
    metadata: dict[str, Any]
    entry_session_index: int
    last_price: float


@dataclass(frozen=True)
class TradeRecord:
    trade_id: str
    strategy_version: str
    security_id: str
    symbol: str
    universe_id: str
    signal_timestamp: datetime
    entry_timestamp: datetime
    entry_price: float
    quantity: int
    initial_stop: float
    initial_risk_rupees: float
    initial_r_distance: float
    exit_timestamp: datetime
    exit_price: float
    exit_reason: str
    gross_pnl: float
    costs: float
    net_pnl: float
    R_multiple: float
    holding_period: int
    metadata: Mapping[str, Any]
    cost_breakdown: Mapping[str, float]


@dataclass(frozen=True)
class BacktestEvent:
    timestamp: datetime
    event_type: str
    security_id: str | None
    details: Mapping[str, Any]


@dataclass(frozen=True)
class RunMetadata:
    run_id: str
    strategy_name: str
    strategy_version: str
    config_hash: str
    config_snapshot: Mapping[str, Any]
    universe_id: str
    evaluation_start: date
    evaluation_end: date
    dataset_version: str
    starting_capital: float
    cost_model: Mapping[str, Any]
    created_at: datetime
