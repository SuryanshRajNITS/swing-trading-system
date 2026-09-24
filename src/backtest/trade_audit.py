"""Export a human-readable audit for the saved A_BASELINE backtest."""

from __future__ import annotations

import argparse
import csv
from datetime import date
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.backtest.data import BacktestDataError, load_daily_dataset_through
from src.configuration import PROJECT_ROOT
from src.data.universe.provider import PIT_TRADABLE


DEFAULT_ARTIFACT = PROJECT_ROOT/"data/processed/backtests/weekly_selection/sr_zone_v2_weekly_selection_r1_2023_2024.json"
DEFAULT_OUTPUT = PROJECT_ROOT/"data/processed/backtests/weekly_selection"
AUDIT_FILENAME = "trade_audit_A_baseline_2023_2024"
START = date(2023, 1, 1)
END = date(2024, 12, 31)

CSV_FIELDS = (
    "trade_id", "record_status", "security_id", "symbol", "setup_type",
    "signal_date", "signal_close", "signal_day_open", "signal_day_high",
    "signal_day_low", "signal_day_close", "signal_day_volume",
    "entry_date", "entry_timestamp", "actual_entry_price", "entry_day_open",
    "entry_day_high", "entry_day_low", "entry_day_close", "entry_day_volume",
    "quantity", "setup_zone_id", "setup_zone_type", "setup_zone_lower",
    "setup_zone_center", "setup_zone_upper", "retest_date", "breakout_date",
    "breakout_level", "stop_price_at_entry", "stop_reference_price",
    "stop_type", "stop_reason", "target_price_at_entry", "target_type",
    "target_R", "target_zone_id", "target_zone_lower", "target_zone_upper",
    "planned_risk_per_share", "planned_reward_per_share", "planned_RR",
    "exit_date", "exit_timestamp", "actual_exit_price", "exit_reason",
    "holding_sessions", "gross_pnl", "transaction_costs", "net_pnl",
    "realized_R", "outcome_class", "stopped_trade", "target_trade",
    "weekly_score", "weekly_momentum", "relative_strength_4w",
    "relative_strength_13w", "weekly_trend_efficiency",
    "weekly_selection_status", "weekly_selection_reason",
    "weekly_structure_score", "weekly_relative_strength_score",
    "nearest_weekly_support", "nearest_weekly_resistance", "volume_ratio",
    "breakout_distance_atr", "breakout_close_location", "freshness",
    "touch_count", "retest_age", "breakout_body_range",
    "breakout_consolidation_sessions", "resistance_touch_episodes",
    "nearest_higher_resistance_distance", "zone_age_sessions", "zone_state",
    "zone_broken", "zone_touch_episodes", "days_since_previous_touch",
    "role_flip", "weekly_selection_exception",
    "penetration_into_zone", "penetration_atr", "rejection_strength",
    "entry_execution_model", "distance_entry_from_signal_close_pct",
    "distance_entry_from_setup_zone_pct", "entry_inside_setup_zone",
    "entry_above_setup_zone", "entry_below_setup_zone",
    "entry_day_candle_intersects_setup_zone",
    "would_daily_low_touch_setup_zone_on_entry_day",
    "entry_day_range_intersects_retest_or_broken_zone",
)


class TradeAuditError(ValueError):
    """The saved run or joined daily candle data cannot support a complete audit."""


def _day(value: Any) -> date:
    return date.fromisoformat(str(value)[:10])


def _candle(data: Mapping[str, Sequence[Mapping[str, Any]]], security_id: str,
            day: date) -> Mapping[str, Any]:
    for row in data.get(security_id, ()):
        if _day(row["date"]) == day:
            return row
    raise TradeAuditError(f"Missing normalized daily candle for {security_id} on {day}.")


def _entry_day_metadata(artifact: Mapping[str, Any], position: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    security_id = str(position["security_id"])
    execution_timestamp = str(position["entry_timestamp"])
    diagnostic = next((row for row in artifact.get("diagnostics", ())
                       if row.get("accepted") and row.get("security_id") == security_id
                       and str(row.get("execution_timestamp")) == execution_timestamp), None)
    if diagnostic is None:
        raise TradeAuditError(f"No accepted setup diagnostic matches open position {security_id} at {execution_timestamp}.")
    signal_date = str(diagnostic.get("signal_date", ""))[:10]
    weekly = next((row for row in artifact.get("selection_signal_decisions", ())
                   if row.get("security_id") == security_id
                   and str(row.get("date", ""))[:10] == signal_date), {})
    return diagnostic, weekly


def _stop_reference(setup_type: str, zone: Mapping[str, Any], stop: float,
                    metadata: Mapping[str, Any], artifact: Mapping[str, Any]) -> tuple[float | None, str, str]:
    params = artifact.get("metadata", {}).get("config_snapshot", {}).get("strategy", {}).get("parameters", {})
    daily = params.get("daily_strategy", {})
    if setup_type == "support_test":
        buffer = float(daily.get("support_stop_buffer_pct", 0.002))
        reference = float(zone["lower"])
        return reference, "SUPPORT_ZONE_LOWER_BOUNDARY", "SUPPORT_ZONE_LOWER_MINUS_CENTER_BUFFER"
    buffer = float(daily.get("breakout_stop_buffer_pct", 0.002))
    retest_low = metadata.get("retest_low")
    if retest_low is not None:
        reference = min(float(zone["lower"]), float(retest_low))
    else:
        # The accepted diagnostic omits retest_low, but the saved stop and
        # run-time buffer recover the exact reference used by the strategy.
        reference = stop+float(zone["center"])*buffer
    return reference, "MIN_BROKEN_ZONE_OR_RETEST_LOW", "LOWER_BROKEN_ZONE_OR_RETEST_LOW_MINUS_CENTER_BUFFER"


def _target_fields(*, target_price: float | None, target_r: float | None,
                   metadata: Mapping[str, Any]) -> dict[str, Any]:
    structural = metadata.get("structural_exit", {})
    if target_r is not None:
        return {"target_price_at_entry": target_price, "target_type": "FIXED_R",
                "target_R": target_r, "target_zone_id": None,
                "target_zone_lower": None, "target_zone_upper": None}
    return {"target_price_at_entry": target_price,
            "target_type": ("STRUCTURAL_RESISTANCE" if structural else "UNSPECIFIED"),
            "target_R": target_r,
            "target_zone_id": structural.get("target_resistance_zone_id"),
            "target_zone_lower": structural.get("target_zone_lower_bound"),
            "target_zone_upper": structural.get("target_zone_upper_bound")}


def _build_row(*, trade: Mapping[str, Any] | None, position: Mapping[str, Any] | None,
               artifact: Mapping[str, Any], daily_data: Mapping[str, Sequence[Mapping[str, Any]]],
               open_diagnostic: Mapping[str, Any] | None = None,
               open_weekly: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if trade is not None:
        metadata = trade.get("metadata", {})
        sid = str(trade["security_id"])
        signal_timestamp = str(trade["signal_timestamp"])
        entry_timestamp = str(trade["entry_timestamp"])
        entry_price = float(trade["entry_price"])
        stop = float(trade["initial_stop"])
        target_r = metadata.get("target_r")
        target_r = float(target_r) if target_r is not None else None
        target_price = (entry_price+float(trade["initial_r_distance"])*target_r
                        if target_r is not None else None)
        zone = metadata.get("zone", {})
        setup = str(metadata.get("setup_type", ""))
        weekly = metadata.get("weekly_selection", {})
        status = "CLOSED"
        exit_timestamp = str(trade["exit_timestamp"])
        exit_day = _day(exit_timestamp)
        exit_reason = str(trade["exit_reason"])
        holding = int(trade["holding_period"])
        gross, costs, net, realized_r = (float(trade["gross_pnl"]), float(trade["costs"]),
                                         float(trade["net_pnl"]), float(trade["R_multiple"]))
        exit_price = float(trade["exit_price"])
        trade_id = str(trade["trade_id"])
        quantity = int(trade["quantity"])
        symbol = str(trade["symbol"])
    else:
        assert position is not None and open_diagnostic is not None
        metadata = open_diagnostic
        sid = str(position["security_id"])
        signal_timestamp = str(open_diagnostic["signal_date"])+"T15:30:00"
        entry_timestamp = str(position["entry_timestamp"])
        entry_price = float(position["entry_price"])
        stop = float(position["initial_stop"])
        target_price = float(position["target_price"]) if position.get("target_price") is not None else None
        target_r_raw = open_diagnostic.get("target_r")
        target_r = float(target_r_raw) if target_r_raw is not None else None
        zone = open_diagnostic.get("zone", {})
        setup = str(open_diagnostic.get("setup_type", ""))
        weekly = open_weekly or {}
        status = "OPEN_AT_END"
        exit_timestamp = ""
        exit_day = None
        exit_reason = ""
        holding = 0
        gross = costs = net = realized_r = exit_price = None
        trade_id = f"OPEN:{sid}:{entry_timestamp}"
        quantity = int(position["quantity"])
        symbol = str(position["symbol"])

    signal_day = _day(signal_timestamp)
    entry_day = _day(entry_timestamp)
    signal_bar = _candle(daily_data, sid, signal_day)
    entry_bar = _candle(daily_data, sid, entry_day)
    if not zone or not all(key in zone for key in ("lower", "center", "upper")):
        raise TradeAuditError(f"Setup zone bounds are missing for trade {trade_id}.")
    zone_lower, zone_center, zone_upper = (float(zone[key]) for key in ("lower", "center", "upper"))
    intersects = float(entry_bar["low"]) <= zone_upper and float(entry_bar["high"]) >= zone_lower
    entry_inside = zone_lower <= entry_price <= zone_upper
    entry_above = entry_price > zone_upper
    entry_below = entry_price < zone_lower
    distance_zone = (0.0 if entry_inside else
                     (entry_price-zone_upper)/zone_upper*100 if entry_above else
                     (entry_price-zone_lower)/zone_lower*100)
    signal_close = float(metadata.get("signal_close", signal_bar["close"]))
    risk = entry_price-stop
    reward = target_price-entry_price if target_price is not None else None
    planned_rr = reward/risk if reward is not None and risk > 0 else None
    stop_reference, stop_type, stop_reason = _stop_reference(setup, zone, stop, metadata, artifact)
    target_values = _target_fields(target_price=target_price, target_r=target_r, metadata=metadata)
    net_value = float(net) if net is not None else None
    outcome = ("OPEN" if status == "OPEN_AT_END" else
               "WINNER" if net_value is not None and net_value > 0 else
               "LOSER" if net_value is not None and net_value < 0 else "FLAT")
    retest_low = metadata.get("retest_low")
    breakout_setup = setup == "breakout_retest"
    row = {
        "trade_id": trade_id, "record_status": status, "security_id": sid,
        "symbol": symbol, "setup_type": setup, "signal_date": signal_day.isoformat(),
        "signal_close": signal_close,
        "signal_day_open": float(signal_bar["open"]), "signal_day_high": float(signal_bar["high"]),
        "signal_day_low": float(signal_bar["low"]), "signal_day_close": float(signal_bar["close"]),
        "signal_day_volume": int(signal_bar["volume"]),
        "entry_date": entry_day.isoformat(), "entry_timestamp": entry_timestamp,
        "actual_entry_price": entry_price, "entry_day_open": float(entry_bar["open"]),
        "entry_day_high": float(entry_bar["high"]), "entry_day_low": float(entry_bar["low"]),
        "entry_day_close": float(entry_bar["close"]), "entry_day_volume": int(entry_bar["volume"]),
        "quantity": quantity, "setup_zone_id": metadata.get("zone_id"),
        "setup_zone_type": zone.get("kind"), "setup_zone_lower": zone_lower,
        "setup_zone_center": zone_center, "setup_zone_upper": zone_upper,
        "retest_date": metadata.get("retest_date"), "breakout_date": metadata.get("breakout_date"),
        "breakout_level": metadata.get("breakout_level"), "stop_price_at_entry": stop,
        "stop_reference_price": stop_reference, "stop_type": stop_type, "stop_reason": stop_reason,
        **target_values, "planned_risk_per_share": risk,
        "planned_reward_per_share": reward, "planned_RR": planned_rr,
        "exit_date": exit_day.isoformat() if exit_day else None,
        "exit_timestamp": exit_timestamp or None, "actual_exit_price": exit_price,
        "exit_reason": exit_reason or None, "holding_sessions": holding if status == "CLOSED" else None,
        "gross_pnl": gross, "transaction_costs": costs, "net_pnl": net,
        "realized_R": realized_r, "outcome_class": outcome,
        "stopped_trade": exit_reason == "STOP" if status == "CLOSED" else False,
        "target_trade": exit_reason == "TARGET" if status == "CLOSED" else False,
        "weekly_score": weekly.get("total_score", weekly.get("score")),
        "weekly_momentum": weekly.get("momentum_score"),
        "relative_strength_4w": weekly.get("relative_strength_4w"),
        "relative_strength_13w": weekly.get("relative_strength_13w"),
        "weekly_trend_efficiency": weekly.get("trend_efficiency_13w"),
        "weekly_selection_status": weekly.get("status"),
        "weekly_selection_reason": weekly.get("reason"),
        "weekly_structure_score": weekly.get("structure_score"),
        "weekly_relative_strength_score": weekly.get("relative_strength_score"),
        "nearest_weekly_support": weekly.get("nearest_support"),
        "nearest_weekly_resistance": weekly.get("nearest_resistance"),
        "volume_ratio": metadata.get("breakout_volume_ratio", metadata.get("volume_ratio")),
        "breakout_distance_atr": metadata.get("breakout_distance_atr"),
        "breakout_close_location": metadata.get("breakout_close_location"),
        "freshness": metadata.get("freshness"), "touch_count": metadata.get("touch_count"),
        "retest_age": metadata.get("retest_age"), "breakout_body_range": metadata.get("breakout_body_range"),
        "breakout_consolidation_sessions": metadata.get("breakout_consolidation_sessions"),
        "resistance_touch_episodes": metadata.get("resistance_touch_episodes"),
        "nearest_higher_resistance_distance": metadata.get("nearest_higher_resistance_distance"),
        "zone_age_sessions": metadata.get("zone_age_sessions"),
        "zone_state": metadata.get("zone_state"),
        "zone_broken": metadata.get("zone_broken"),
        "zone_touch_episodes": metadata.get("zone_touch_episodes"),
        "days_since_previous_touch": metadata.get("days_since_previous_touch"),
        "role_flip": metadata.get("role_flip"),
        "weekly_selection_exception": weekly.get("resistance_breakout_exception",
                                                  metadata.get("weekly_selection_exception")),
        "penetration_into_zone": metadata.get("penetration_into_zone"),
        "penetration_atr": metadata.get("penetration_atr"),
        "rejection_strength": metadata.get("rejection_strength"),
        "entry_execution_model": "NEXT_SESSION_OPEN",
        "distance_entry_from_signal_close_pct": (entry_price/signal_close-1)*100 if signal_close else None,
        "distance_entry_from_setup_zone_pct": distance_zone,
        "entry_inside_setup_zone": entry_inside, "entry_above_setup_zone": entry_above,
        "entry_below_setup_zone": entry_below,
        "entry_day_candle_intersects_setup_zone": intersects,
        "would_daily_low_touch_setup_zone_on_entry_day": intersects,
        "entry_day_range_intersects_retest_or_broken_zone": intersects if breakout_setup else None,
    }
    return {field: row.get(field) for field in CSV_FIELDS}


def build_trade_audit(artifact: Mapping[str, Any],
                      daily_data: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    """Join saved A_BASELINE trades and open positions to daily candles."""
    rows = [_build_row(trade=trade, position=None, artifact=artifact, daily_data=daily_data)
            for trade in artifact.get("trades", ())]
    for position in artifact.get("open_positions_at_end", ()):
        diagnostic, weekly = _entry_day_metadata(artifact, position)
        rows.append(_build_row(trade=None, position=position, artifact=artifact,
                               daily_data=daily_data, open_diagnostic=diagnostic,
                               open_weekly=weekly))
    rows.sort(key=lambda row: (str(row.get("entry_timestamp", "")), str(row.get("trade_id", ""))))
    expected = sum(event.get("event_type") == "POSITION_ENTRY" for event in artifact.get("events", ()))
    if len(rows) != expected:
        raise TradeAuditError(f"Audit exported {len(rows)} rows for {expected} executed entries.")
    return rows


def select_deterministic_sample(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Choose up to 30 unique, reproducible outcome examples."""
    closed = [row for row in rows if row.get("record_status") == "CLOSED"]
    categories = (
        ("TARGET_WINNER", [row for row in closed if row.get("exit_reason") == "TARGET"
                           and float(row.get("net_pnl") or 0) > 0], 10, False),
        ("STOP_LOSER", [row for row in closed if row.get("exit_reason") == "STOP"
                        and float(row.get("net_pnl") or 0) < 0], 10, False),
        ("LARGEST_WINNER", sorted((row for row in closed if float(row.get("net_pnl") or 0) > 0),
                                   key=lambda row: (-float(row["net_pnl"]), str(row["trade_id"]))), 5, False),
        ("LARGEST_LOSER", sorted((row for row in closed if float(row.get("net_pnl") or 0) < 0),
                                  key=lambda row: (float(row["net_pnl"]), str(row["trade_id"]))), 5, False),
    )
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for category, candidates, quota, _ in categories:
        ordered = (sorted(candidates, key=lambda row: (str(row.get("signal_date", "")),
                                                       str(row.get("trade_id", ""))))
                   if category in {"TARGET_WINNER", "STOP_LOSER"} else candidates)
        count = 0
        for row in ordered:
            trade_id = str(row["trade_id"])
            if trade_id in seen:
                continue
            selected.append({**dict(row), "sample_category": category})
            seen.add(trade_id)
            count += 1
            if count == quota:
                break
    category_order = {name: index for index, (name, *_rest) in enumerate(categories)}
    selected.sort(key=lambda row: (category_order[row["sample_category"]],
                                   str(row.get("signal_date", "")),
                                   str(row.get("trade_id", ""))))
    return selected


def summarize_audit(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def counts(subset: Sequence[Mapping[str, Any]]) -> dict[str, int]:
        return {
            "entry_day_touched_setup_zone": sum(row.get("entry_day_candle_intersects_setup_zone") is True for row in subset),
            "entry_day_did_not_touch_setup_zone": sum(row.get("entry_day_candle_intersects_setup_zone") is False for row in subset),
            "entry_opened_inside_setup_zone": sum(row.get("entry_inside_setup_zone") is True for row in subset),
            "entry_opened_above_setup_zone": sum(row.get("entry_above_setup_zone") is True for row in subset),
            "entry_opened_below_setup_zone": sum(row.get("entry_below_setup_zone") is True for row in subset),
        }
    closed = [row for row in rows if row.get("record_status") == "CLOSED"]
    return {
        "total_executed_trades": len(rows),
        "closed_trades": len(closed),
        "open_positions_at_end": len(rows)-len(closed),
        "entry_day": counts(rows),
        "stop_trades": counts([row for row in closed if row.get("stopped_trade") is True]),
        "target_trades": counts([row for row in closed if row.get("target_trade") is True]),
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], *, fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_export(*, artifact_path: str | Path = DEFAULT_ARTIFACT,
               manifest_path: str | Path = PROJECT_ROOT/"data/processed/market_data_manifest.csv",
               data_root: str | Path = PROJECT_ROOT/"data/processed/market_data",
               output_directory: str | Path = DEFAULT_OUTPUT) -> dict[str, Path]:
    artifact_file = Path(artifact_path)
    if not artifact_file.is_file():
        raise TradeAuditError(f"Saved A_BASELINE artifact not found: {artifact_file}")
    artifact = json.loads(artifact_file.read_text(encoding="utf-8"))
    if artifact.get("research_controls", {}).get("period") != [START.isoformat(), END.isoformat()]:
        raise TradeAuditError("Saved artifact is not the bounded 2023-2024 A_BASELINE run.")
    if artifact.get("research_controls", {}).get("target_r") != 2.0:
        raise TradeAuditError("Saved artifact does not identify the fixed 2R A_BASELINE configuration.")
    if artifact.get("research_controls", {}).get("strategy_version") != "sr_zone_v2_weekly_selection_v1":
        raise TradeAuditError("Saved artifact is not the expected weekly-selection C-arm run.")
    if float(artifact.get("research_controls", {}).get("slippage_bps", -1)) != 0.0:
        raise TradeAuditError("Saved artifact does not use the required zero-slippage A_BASELINE scenario.")
    daily_data, _ = load_daily_dataset_through(
        universe_id=PIT_TRADABLE, start_date=START, end_date=END,
        data_root=data_root, manifest_path=manifest_path)
    rows = build_trade_audit(artifact, daily_data)
    sample = select_deterministic_sample(rows)
    summary = summarize_audit(rows)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output/f"{AUDIT_FILENAME}.csv"
    json_path = output/f"{AUDIT_FILENAME}.json"
    sample_path = output/"trade_audit_sample_2023_2024.csv"
    _write_csv(csv_path, rows, fields=CSV_FIELDS)
    _write_csv(sample_path, sample, fields=(*CSV_FIELDS, "sample_category"))
    json_path.write_text(json.dumps({"source_artifact": str(artifact_file),
                                     "period": [START.isoformat(), END.isoformat()],
                                     "summary": summary, "trades": rows},
                                    indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    return {"csv": csv_path, "json": json_path, "sample_csv": sample_path,
            "summary": summary, "sample_count": len(sample)}


def _print_summary(result: Mapping[str, Any]) -> None:
    summary = result["summary"]
    print(f"Total executed trades: {summary['total_executed_trades']}")
    print(f"Closed trades: {summary['closed_trades']}")
    print(f"Open positions at end: {summary['open_positions_at_end']}")
    print(f"Entry-day candles: {json.dumps(summary['entry_day'], sort_keys=True)}")
    print(f"STOP trades, entry-day zone interaction: {json.dumps(summary['stop_trades'], sort_keys=True)}")
    print(f"TARGET trades, entry-day zone interaction: {json.dumps(summary['target_trades'], sort_keys=True)}")
    print(f"Full audit CSV: {result['csv']}")
    print(f"Full audit JSON: {result['json']}")
    print(f"Sample CSV ({result['sample_count']} unique rows): {result['sample_csv']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT,
                        help="Saved 2023-2024 A_BASELINE JSON artifact.")
    parser.add_argument("--manifest", type=Path,
                        default=PROJECT_ROOT/"data/processed/market_data_manifest.csv")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT/"data/processed/market_data")
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        result = run_export(artifact_path=args.artifact, manifest_path=args.manifest,
                            data_root=args.data_root, output_directory=args.output_directory)
    except (OSError, ValueError, BacktestDataError) as error:
        parser.error(str(error))
    _print_summary(result)


if __name__ == "__main__":
    main()


__all__ = ["CSV_FIELDS", "TradeAuditError", "build_trade_audit", "run_export",
           "select_deterministic_sample", "summarize_audit"]
