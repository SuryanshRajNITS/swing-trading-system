"""Deterministic summaries for structural-exit research runs."""

from __future__ import annotations

from collections import Counter
from statistics import median
from typing import Any, Mapping, Sequence


def _summary(trades: Sequence[Any], starting_capital: float) -> dict[str,Any]:
    net=[float(trade.net_pnl) for trade in trades]
    gross=[float(trade.gross_pnl) for trade in trades]
    gains=sum(value for value in net if value>0)
    losses=-sum(value for value in net if value<0)
    exits=Counter(str(trade.exit_reason) for trade in trades)
    time_trades=[trade for trade in trades if trade.exit_reason=="TIME_EXIT"]
    time_net=[float(trade.net_pnl) for trade in time_trades]
    return {
        "trade_count":len(trades),
        "win_rate":sum(value>0 for value in net)/len(net) if net else 0.0,
        "gross_expectancy":sum(gross)/len(gross) if gross else 0.0,
        "net_expectancy":sum(net)/len(net) if net else 0.0,
        "profit_factor":gains/losses if losses else (None if gains else 0.0),
        "net_pnl":sum(net),
        "net_pnl_over_starting_capital":sum(net)/starting_capital if starting_capital else 0.0,
        "average_holding_sessions":sum(int(trade.holding_period) for trade in trades)/len(trades) if trades else 0.0,
        "median_holding_sessions":median(int(trade.holding_period) for trade in trades) if trades else 0.0,
        "exit_counts":dict(sorted(exits.items())),
        "time_exit":{
            "count":len(time_trades),
            "profitable_count":sum(value>0 for value in time_net),
            "losing_count":sum(value<=0 for value in time_net),
            "average_net_pnl":sum(time_net)/len(time_net) if time_net else 0.0,
            "average_R_at_exit":sum(float(trade.R_multiple) for trade in time_trades)/len(time_trades) if time_trades else 0.0,
        },
    }


def build_structural_exit_diagnostics(result: Any, strategy: Any) -> dict[str,Any]:
    trades=list(result.trades)
    initial=float(result.metadata.starting_capital)
    by_year={}
    for year in ("2023","2024"):
        subset=[trade for trade in trades if trade.signal_timestamp.date().isoformat()[:4]==year]
        by_year[year]=_summary(subset,initial)
    by_setup={}
    for setup,label in (("support_test","SUPPORT"),("breakout_retest","BREAKOUT_RETEST")):
        by_setup[label]=_summary([trade for trade in trades if trade.metadata.get("setup_type")==setup],initial)
    cross={}
    for year in ("2023","2024"):
        cross[year]={}
        for setup,label in (("support_test","SUPPORT"),("breakout_retest","BREAKOUT_RETEST")):
            subset=[trade for trade in trades if trade.metadata.get("setup_type")==setup
                    and trade.signal_timestamp.date().isoformat()[:4]==year]
            cross[year][label]=_summary(subset,initial)
    rejection_counts=Counter(strategy.rejection_counts)
    rejection_by_year=Counter(str(row["date"])[:4] for row in strategy.entry_decisions if not row.get("accepted",False))
    rejection_by_setup=Counter(str(row.get("setup_type","unknown")) for row in strategy.entry_decisions
                               if not row.get("accepted",False))
    explicit_fallbacks=sum(row.get("target_status")=="EXPLICIT_FIXED_R_FALLBACK"
                           and row.get("accepted",False) for row in strategy.entry_decisions)
    return {
        "exit_mode":"STRUCTURAL_EXIT",
        "overall":_summary(trades,initial),
        "trading_net_return":float(result.metrics.get("trading_net_return",0.0)),
        "trading_ending_equity":float(result.metrics.get("trading_ending_equity",result.metrics.get("ending_equity",initial))),
        "transaction_costs_total":float(result.metrics.get("transaction_costs_total",0.0)),
        "by_year":by_year,"by_setup":by_setup,"by_year_and_setup":cross,
        "rejected_candidates":{
            "total":sum(rejection_counts.values()),
            "by_reason":dict(sorted(rejection_counts.items())),
            "by_year":dict(sorted(rejection_by_year.items())),
            "by_setup":dict(sorted(rejection_by_setup.items())),
            "NO_STRUCTURAL_TARGET":rejection_counts.get("NO_STRUCTURAL_TARGET",0),
            "INSUFFICIENT_STRUCTURAL_RR":rejection_counts.get("INSUFFICIENT_STRUCTURAL_RR",0),
        },
        "explicit_fixed_r_fallback_count":explicit_fallbacks,
        "event_ordering":[
            "Prior-close strategy exits execute at the next session open.",
            "Opening gaps through stop or target execute before new entries.",
            "Daily-bar stop/target checks occur after entry; stop wins if both are touched.",
            "After daily stop/target resolution at close, the 15th held session may schedule TIME_EXIT for the next session open.",
        ],
        "candidate_decisions":strategy.entry_decisions,
        "time_exit_decisions":strategy.time_exit_decisions,
    }


__all__=["build_structural_exit_diagnostics"]
