"""Create causal, per-trade chart packets for manual V2 trade review.

The utility reads saved portfolio and ENTRY_READY artifacts only. It never
invokes signal generation or the portfolio simulator.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
BACKTEST_ROOT = ROOT / "data/processed/backtests/top_down_swing_v2_daily_structure"
PORTFOLIO_ROOT = BACKTEST_ROOT / "portfolio_management"
DEFAULT_OUTPUT = BACKTEST_ROOT / "trade_review"
HOURLY_ROOT = ROOT / "data/processed/market_data/NIFTY50_PIT_TRADABLE/60minute"
DAILY_ROOT = ROOT / "data/processed/market_data/NIFTY50_PIT_TRADABLE/day"
INVALID_IDENTITIES = ROOT / "data/processed/hourly/price_integrity/invalid_hourly_identities.csv"
STRATEGY_VERSION = "top_down_swing_v2_daily_structure"
RESEARCH_END = date(2024, 12, 31)
MANUAL_CLASSIFICATIONS = (
    "VALID_SETUP_NORMAL_FAILURE", "BAD_ENTRY_TOO_EARLY", "BAD_ENTRY_WRONG_ZONE",
    "BAD_ENTRY_INTO_FRICTION", "BAD_ENTRY_WEAK_CONFIRMATION", "BAD_ENTRY_WRONG_REGIME",
    "BAD_ENTRY_BREAKOUT_NOT_ACCEPTED", "BAD_ENTRY_RECLAIM_NOT_CONFIRMED",
    "BAD_ENTRY_POOR_VOLUME", "TARGET_SELECTION_ISSUE", "STOP_SELECTION_ISSUE",
    "TRADE_MANAGEMENT_ISSUE", "DATA_OR_EXECUTION_ANOMALY", "OTHER",
)
MANUAL_FIELDS = (
    "manual_classification", "manual_entry_valid", "manual_notes", "manual_zone_issue",
    "manual_confirmation_issue", "manual_friction_issue", "manual_regime_issue",
    "manual_target_issue", "manual_stop_issue",
)
MANIFEST_FIELDS = (
    "review_id", "symbol", "setup_family", "setup_event_id", "setup_instance_id",
    "signal_timestamp", "entry_timestamp", "entry_price", "stop_price", "target_price",
    "planned_R", "quantity", "position_notional", "exit_timestamp", "exit_price",
    "exit_reason", "realized_gross_pnl", "realized_net_pnl", "realized_R", "MFE_R",
    "MAE_R", "giveback_R", "holding_sessions", "chart_file", "strategy_version",
    *MANUAL_FIELDS,
)


@dataclass(frozen=True)
class Candle:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json_field(value: Any, *, default: Any) -> Any:
    if value in (None, ""):
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value))
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Saved JSON field is malformed: {str(value)[:120]}") from exc


def _dt(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"Timestamp is missing timezone: {value}")
    if parsed.utcoffset() != timedelta(hours=5, minutes=30):
        raise ValueError(f"Timestamp is not normalized to Asia/Kolkata: {value}")
    return parsed


def _float(row: Mapping[str, Any], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {key}")
    return value


def load_candles(path: Path, *, through: date | None = None) -> list[Candle]:
    candles = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            # The saved research run is capped at 2024. Stop before parsing any
            # later OHLCV, so OOS candles cannot enter indicators or charts.
            if through is not None and date.fromisoformat(row["date"][:10]) > through:
                break
            candles.append(Candle(_dt(row["date"]),
                *(_float(row, key) for key in ("open", "high", "low", "close", "volume"))))
    stamps = [c.timestamp for c in candles]
    if len(stamps) != len(set(stamps)):
        raise ValueError(f"Duplicate candle timestamps in {path}")
    if stamps != sorted(stamps):
        raise ValueError(f"Candles are not sorted in {path}")
    for candle in candles:
        if candle.high < max(candle.open, candle.close, candle.low) or candle.low > min(candle.open, candle.close):
            raise ValueError(f"Invalid OHLC in {path} at {candle.timestamp.isoformat()}")
        if candle.volume < 0:
            raise ValueError(f"Negative volume in {path} at {candle.timestamp.isoformat()}")
    return candles


def filter_trades(rows: Sequence[Mapping[str, str]], *, losers_only: bool = True,
                  symbol: str | None = None, setup: str | None = None,
                  year: int | None = None, min_mfe_r: float | None = None,
                  max_mfe_r: float | None = None) -> list[dict[str, str]]:
    selected = []
    for raw in rows:
        row = dict(raw)
        if row.get("strategy_version") != STRATEGY_VERSION:
            raise ValueError(f"Unexpected strategy_version in portfolio row {row.get('setup_instance_id')}")
        if row.get("status") != "CLOSED":
            continue
        if losers_only and _float(row, "net_pnl") >= 0:
            continue
        if symbol and row.get("symbol", "").upper() != symbol.upper():
            continue
        if setup and row.get("setup_family", "").upper() != setup.upper():
            continue
        entry = _dt(row["entry_timestamp"])
        if year is not None and entry.year != year:
            continue
        mfe = _float(row, "mfe_R")
        if min_mfe_r is not None and mfe < min_mfe_r:
            continue
        if max_mfe_r is not None and mfe > max_mfe_r:
            continue
        row["giveback_R"] = str(mfe - _float(row, "net_R"))
        selected.append(row)
    return selected


def sort_trades(rows: Sequence[dict[str, str]], mode: str) -> list[dict[str, str]]:
    if mode == "mfe_asc":
        key = lambda r: (_float(r, "mfe_R"), _float(r, "net_R"), _dt(r["entry_timestamp"]))
    elif mode == "giveback_desc":
        key = lambda r: (-float(r["giveback_R"]), _dt(r["entry_timestamp"]))
    elif mode == "realized_r_asc":
        key = lambda r: (_float(r, "net_R"), _dt(r["entry_timestamp"]))
    elif mode == "chronological":
        key = lambda r: _dt(r["entry_timestamp"])
    else:
        raise ValueError(f"Unknown sort mode: {mode}")
    return sorted(rows, key=key)


def trade_path_excursions(candles: Sequence[Candle], *, entry_timestamp: str,
                          exit_timestamp: str, entry_price: float,
                          stop_price: float, exit_reason: str) -> tuple[float, float, bool]:
    """Recompute excursions using the portfolio engine's OHLC event ordering.

    Entry candle contributes after next-open fill. Gap/time exits happen at the
    exit candle open and therefore exclude its later range. Intrabar bracket
    exits include the whole trigger candle, matching the engine's conservative
    stop-first OHLC convention; ordering within that candle remains ambiguous.
    """
    entry, exit_ = _dt(entry_timestamp), _dt(exit_timestamp)
    risk = entry_price - stop_price
    if risk <= 0:
        raise ValueError("Trade has non-positive initial risk")
    path = [c for c in candles if entry <= c.timestamp <= exit_]
    if not path or path[0].timestamp != entry:
        raise ValueError(f"Entry timestamp does not map to an hourly candle: {entry_timestamp}")
    if path[-1].timestamp != exit_:
        raise ValueError(f"Exit timestamp does not map to an hourly candle: {exit_timestamp}")
    gap_or_open_exit = exit_reason.startswith("GAP_") or exit_reason.startswith("TIME_")
    measured = path[:-1] if gap_or_open_exit else path
    if not measured:
        measured = path[:1]
    high = max(entry_price, *(c.high for c in measured))
    low = min(entry_price, *(c.low for c in measured))
    mfe = max(0.0, (high - entry_price) / risk)
    mae = min(0.0, (low - entry_price) / risk)
    ambiguous = not gap_or_open_exit and exit_reason in {
        "STOP", "TARGET", "AMBIGUOUS_INTRABAR_STOP_FIRST"
    }
    return mfe, mae, ambiguous


def causal_entry_state(signal: Mapping[str, str], *, entry_date: date) -> dict[str, Any]:
    signal_date = date.fromisoformat(signal["date"])
    if signal_date > entry_date:
        raise ValueError("Saved entry-state snapshot is later than trade entry")
    state = {
        "signal_date": signal["date"],
        "setup_family": signal.get("setup_family") or signal.get("setup_type"),
        "support": _read_json_field(signal.get("support"), default=None),
        "resistance": _read_json_field(signal.get("resistance"), default=None),
        "zones": _read_json_field(signal.get("daily_zones"), default=[]),
        "weekly_selection": {
            "path": signal.get("weekly_selection_path") or "N/A",
            "components": _read_json_field(signal.get("weekly_selection_components"), default={}),
        },
        "weekly_structure": _read_json_field(signal.get("weekly_structure"), default={}),
        "nifty_regime": signal.get("nifty_regime") or "N/A",
        "result": _read_json_field(signal.get("result"), default={}),
        "plan": _read_json_field(signal.get("plan"), default={}),
        "confirmation_timestamp": signal.get("confirmation_timestamp") or "N/A",
        "executable_timestamp": signal.get("executable_timestamp") or "N/A",
    }
    for zone in state["zones"]:
        last_touch = zone.get("last_touch_date")
        if last_touch and date.fromisoformat(last_touch) > signal_date:
            raise ValueError(f"Entry snapshot contains a future zone touch: {last_touch}")
        for pivot in zone.get("pivots", []):
            if date.fromisoformat(pivot) > signal_date:
                raise ValueError(f"Entry snapshot contains a future pivot: {pivot}")
        for transition in zone.get("role_transitions", []):
            if date.fromisoformat(transition["date"]) > signal_date:
                raise ValueError(f"Entry snapshot contains a future role transition: {transition['date']}")
    return state


def deterministic_filename(index: int, row: Mapping[str, str]) -> str:
    safe = lambda value: re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_") or "NA"
    entry_date = _dt(row["entry_timestamp"]).date().isoformat()
    return f"{index:03d}_{safe(row['symbol'])}_{entry_date}_{safe(row['setup_family'])}.png"


def _ema(values: Sequence[float], period: int) -> list[float | None]:
    alpha = 2 / (period + 1)
    result: list[float | None] = []
    current: float | None = None
    for value in values:
        current = value if current is None else alpha * value + (1 - alpha) * current
        result.append(current)
    return result


def _entry_zone(value: Any, selected_id: str | None = None) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, list):
        candidates = [zone for zone in value if isinstance(zone, Mapping)]
        if selected_id:
            match = next((zone for zone in candidates if zone.get("zone_id") == selected_id), None)
            if match:
                return match
        if len(candidates) == 1:
            return candidates[0]
    return {}


def _panel_candles(ax: Any, candles: Sequence[Candle], *, volume: bool = False) -> None:
    if not candles:
        return
    from matplotlib.patches import Rectangle
    width = 0.62
    for i, c in enumerate(candles):
        color = "#16805d" if c.close >= c.open else "#bf3f45"
        ax.vlines(i, c.low, c.high, color=color, linewidth=0.8)
        bottom = min(c.open, c.close)
        height = max(abs(c.close - c.open), max(abs(c.close), 1.0) * 0.00008)
        ax.add_patch(Rectangle((i - width / 2, bottom), width, height, color=color, alpha=0.88))
    ax.grid(True, alpha=0.18)
    ax.set_xlim(-1, len(candles))
    if volume:
        ax.set_ylabel("Volume")


def _plot_trade(row: Mapping[str, str], state: Mapping[str, Any], daily: Sequence[Candle],
                hourly: Sequence[Candle], output: Path) -> None:
    try:
        os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "swing-trading-mpl-cache"))
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("PNG rendering requires matplotlib; install project requirements") from exc

    entry_dt, exit_dt = _dt(row["entry_timestamp"]), _dt(row["exit_timestamp"])
    entry_day, exit_day = entry_dt.date(), _dt(row["exit_timestamp"]).date()
    daily_before = [c for c in daily if c.timestamp.date() < entry_day]
    left_daily = daily_before[max(0, len(daily_before) - 90):]
    right_daily = [c for c in daily if c.timestamp.date() > exit_day][:5]
    daily_view = [c for c in daily if c.timestamp.date() >= (left_daily[0].timestamp.date() if left_daily else entry_day)
                  and c.timestamp.date() <= exit_day]
    daily_view += [c for c in right_daily if c.timestamp not in {x.timestamp for x in daily_view}]
    if len(daily_view) < 2:
        raise ValueError("Insufficient daily candles for chart")

    hourly_before_dates = sorted({c.timestamp.date() for c in hourly if c.timestamp < entry_dt})
    left_hour_dates = set(hourly_before_dates[-15:])
    after_dates = sorted({c.timestamp.date() for c in hourly if c.timestamp.date() > exit_day})[:2]
    hourly_view = [c for c in hourly if c.timestamp.date() in left_hour_dates
                   or entry_dt <= c.timestamp <= _dt(row["exit_timestamp"])
                   or c.timestamp.date() in after_dates]
    if not hourly_view:
        raise ValueError("Insufficient hourly candles for chart")

    fig, (daily_ax, hour_ax, volume_ax) = plt.subplots(
        3, 1, figsize=(16, 11), gridspec_kw={"height_ratios": [1.2, 1.35, 0.45]})
    fig.subplots_adjust(top=0.75, hspace=0.36, left=0.08, right=0.97, bottom=0.10)
    _panel_candles(daily_ax, daily_view)
    daily_index = {c.timestamp: i for i, c in enumerate(daily_view)}
    full_daily_ema50 = _ema([c.close for c in daily], 50)
    full_daily_ema200 = _ema([c.close for c in daily], 200)
    full_ema_index = {c.timestamp: i for i, c in enumerate(daily)}
    daily_ax.plot(range(len(daily_view)), [full_daily_ema50[full_ema_index[c.timestamp]] for c in daily_view],
                  color="#1769aa", linewidth=1.1, label="EMA50")
    daily_ax.plot(range(len(daily_view)), [full_daily_ema200[full_ema_index[c.timestamp]] for c in daily_view],
                  color="#9a5b13", linewidth=1.0, label="EMA200")

    plan = state.get("plan", {})
    selected_support = plan.get("daily_support_zone_id")
    selected_target = plan.get("target_zone_id")
    zones = state.get("zones", [])
    selected_resistance = plan.get("daily_resistance_zone_id")
    all_lows = [c.low for c in daily_view + hourly_view]
    all_highs = [c.high for c in daily_view + hourly_view]
    price_floor, price_ceiling = min(all_lows), max(all_highs)
    zone_ids = set()
    for zone in zones:
        zid = zone.get("zone_id")
        if not zid or zid in zone_ids:
            continue
        lower, upper = float(zone["lower_bound"]), float(zone["upper_bound"])
        if upper < price_floor or lower > price_ceiling:
            continue
        zone_ids.add(zid)
        color = "#2e9d68" if zone.get("side") == "support" else "#d98a33"
        label = "Saved support zone" if zone.get("side") == "support" else "Saved resistance zone"
        if zone.get("structure_level_kind") == "INNER_BOUNDARY":
            color, label = "#68717a", "Saved inner boundary"
        if zid == selected_support:
            color, label = "#07834f", f"SELECTED SUPPORT {zone.get('strength', '')} {zone.get('role_state', '')}"
        elif zid == selected_target:
            color, label = "#a51c30", f"DOMINANT TARGET {zone.get('strength', '')}"
        elif zid == selected_resistance:
            label = f"SELECTED RESISTANCE {zone.get('strength', '')} {zone.get('role_state', '')}"
        daily_ax.axhspan(lower, upper, color=color, alpha=0.10,
                         label=label if label not in daily_ax.get_legend_handles_labels()[1] else None)

    friction_groups = (
        (daily_ax, "intermediate_daily_friction", "DAILY FRICTION"),
        (daily_ax, "intermediate_structural_friction", "STRUCTURAL FRICTION"),
        (hour_ax, "friction", "HOURLY FRICTION"),
    )
    for ax, key, fallback_label in friction_groups:
        items = plan.get(key) or []
        if isinstance(items, Mapping):
            items = [items]
        for item in items:
            if not isinstance(item, Mapping):
                continue
            lower = item.get("lower_bound")
            upper = item.get("upper_bound")
            price = item.get("price", item.get("friction_price"))
            saved_label = item.get("classification") or item.get("friction_type") or item.get("zone_id") or fallback_label
            label = f"{item.get('timeframe', '')} {saved_label}".strip()
            if lower is not None and upper is not None:
                ax.axhspan(float(lower), float(upper), color="#7c4d9e", alpha=0.12,
                           label=label if label not in ax.get_legend_handles_labels()[1] else None)
            elif price is not None:
                ax.axhline(float(price), color="#7c4d9e", linestyle="-.", linewidth=0.8,
                           label=label if label not in ax.get_legend_handles_labels()[1] else None)

    _draw_trade((daily_ax, hour_ax), row)
    daily_x = daily_index.get(next((c.timestamp for c in daily_view if c.timestamp.date() == entry_day), None))
    exit_x = daily_index.get(next((c.timestamp for c in daily_view if c.timestamp.date() == exit_day), None))
    if daily_x is not None:
        daily_ax.scatter([daily_x], [float(row["entry_price"])], marker="^", s=85, color="#006b45", zorder=5)
    if exit_x is not None:
        daily_ax.scatter([exit_x], [float(row["exit_price"])], marker="X", s=70, color="#911b2b", zorder=5)
    daily_ax.set_title("Daily candles and entry-time structural snapshot", loc="left", fontsize=11)
    daily_ax.set_ylabel("Price")
    daily_ticks = list(range(0, len(daily_view), max(1, len(daily_view) // 12)))
    daily_ax.set_xticks(daily_ticks)
    daily_ax.set_xticklabels([daily_view[i].timestamp.date().isoformat() for i in daily_ticks],
                             rotation=30, ha="right", fontsize=8)
    daily_ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.01), fontsize=7, ncol=4)

    _panel_candles(hour_ax, hourly_view)
    xmap = {c.timestamp: i for i, c in enumerate(hourly_view)}
    h_entry_x, h_exit_x = xmap.get(entry_dt), xmap.get(_dt(row["exit_timestamp"]))
    if h_entry_x is None or h_exit_x is None:
        raise ValueError("Entry/exit candle absent from hourly chart window")
    hour_ax.scatter([h_entry_x], [float(row["entry_price"])], marker="^", s=90, color="#006b45", zorder=6)
    hour_ax.scatter([h_exit_x], [float(row["exit_price"])], marker="X", s=75, color="#911b2b", zorder=6)
    result = state.get("result", {})
    confirmation = result.get("confirmation_timestamp") or state.get("confirmation_timestamp")
    if confirmation and confirmation != "N/A":
        confirm_dt = _dt(confirmation)
        if confirm_dt in xmap:
            bar = hourly_view[xmap[confirm_dt]]
            hour_ax.scatter([xmap[confirm_dt]], [bar.high], marker="*", s=95, color="#493f9a", zorder=6)
    path = [c for c in hourly if entry_dt <= c.timestamp <= _dt(row["exit_timestamp"])]
    if row["exit_reason"].startswith("GAP_") or row["exit_reason"].startswith("TIME_"):
        path = path[:-1]
    if path:
        risk = float(row["entry_price"]) - float(row["stop_price"])
        mfe_candle = max(path, key=lambda c: c.high)
        mae_candle = min(path, key=lambda c: c.low)
        if mfe_candle.timestamp in xmap:
            hour_ax.scatter([xmap[mfe_candle.timestamp]], [mfe_candle.high], marker="P", s=65,
                            color="#16805d", zorder=6, label="MFE bar")
        if mae_candle.timestamp in xmap:
            hour_ax.scatter([xmap[mae_candle.timestamp]], [mae_candle.low], marker="v", s=65,
                            color="#bf3f45", zorder=6, label="MAE bar")
    hour_ax.set_title("Hourly execution path (green triangle entry, red X exit, star confirmation)", loc="left", fontsize=11)
    hour_ax.set_ylabel("Price")
    hour_ticks = list(range(0, len(hourly_view), max(1, len(hourly_view) // 7)))
    hour_ax.set_xticks(hour_ticks)
    hour_ax.set_xticklabels([hourly_view[i].timestamp.strftime("%m-%d %H:%M") for i in hour_ticks],
                            rotation=25, ha="right", fontsize=7)

    volume_ax.grid(True, alpha=0.18)
    volume_ax.set_xlim(-1, len(hourly_view))
    volume_ax.set_ylabel("Volume")
    for i, c in enumerate(hourly_view):
        volume_ax.bar(i, c.volume, color="#16805d" if c.close >= c.open else "#bf3f45", width=0.62, alpha=0.65)
    volume_ax.set_xticks(hour_ticks)
    volume_ax.set_xticklabels([hourly_view[i].timestamp.strftime("%m-%d %H:%M") for i in hour_ticks],
                              rotation=25, ha="right", fontsize=7)
    volume_ax.set_title("Hourly volume", loc="left", fontsize=9)

    mfe, mae, ambiguous = trade_path_excursions(hourly, entry_timestamp=row["entry_timestamp"],
            exit_timestamp=row["exit_timestamp"], entry_price=float(row["entry_price"]),
            stop_price=float(row["stop_price"]), exit_reason=row["exit_reason"])
    planned_r = (float(row["target_price"]) - float(row["entry_price"])) / (
        float(row["entry_price"]) - float(row["stop_price"]))
    support = _entry_zone(state.get("support"), plan.get("daily_support_zone_id"))
    resistance = _entry_zone(state.get("resistance"), plan.get("daily_resistance_zone_id"))
    target_id = plan.get("target_zone_id") or "N/A"
    volume_state = (result.get("confirmation_features") or {}).get("volume") or {}
    weekly_components = state.get("weekly_selection", {}).get("components", {})
    weekly_structure = state.get("weekly_structure", {})
    support_band = (f"[{float(support['lower_bound']):.2f}-{float(support['upper_bound']):.2f}]"
                    if support and support.get("lower_bound") is not None and support.get("upper_bound") is not None
                    else "N/A")
    resistance_band = (f"[{float(resistance['lower_bound']):.2f}-{float(resistance['upper_bound']):.2f}]"
                       if resistance and resistance.get("lower_bound") is not None and resistance.get("upper_bound") is not None
                       else "N/A")
    resistance_label = (str(resistance.get("zone_id")) if resistance.get("zone_id") else
        f"{len(state.get('resistance') or [])} saved candidates" if isinstance(state.get("resistance"), list) else "N/A")
    context = (
        f"Entry-time state ({state['signal_date']}) | Support {support.get('zone_id', 'N/A')} "
        f"{support.get('role_state', '')}/{support.get('strength', '')} {support_band}\n"
        f"Resistance {resistance_label} {resistance.get('role_state', '')}/{resistance.get('strength', '')} {resistance_band} "
        f"| Target {target_id} {plan.get('target_timeframe', 'N/A')} | NIFTY {state.get('nifty_regime', 'N/A')}\n"
        f"Weekly HH={weekly_structure.get('higher_high', 'N/A')} HL={weekly_structure.get('higher_low', 'N/A')} "
        f"score={weekly_structure.get('structure_score', 'N/A')} selector="
        f"{weekly_components.get('status', 'N/A')}/{weekly_components.get('total_score', 'N/A')} "
        f"| Confirmation {result.get('confirmation_type', 'N/A')} "
        f"| Volume required={volume_state.get('required', 'N/A')} passed={volume_state.get('passed', 'N/A')} "
        f"type={volume_state.get('type', 'N/A')} "
        f"| OHLC trigger candle {'ambiguous; saved stop-first convention' if ambiguous else 'not ambiguous'}"
    )
    header = (f"{row['symbol']} | {row['setup_family']}\n"
        f"Entry {float(row['entry_price']):.2f} at {row['entry_timestamp']} | Stop {float(row['stop_price']):.2f} "
        f"| Target {float(row['target_price']):.2f}\n"
        f"Planned {planned_r:.2f}R | Realized {float(row['net_R']):+.2f}R | MFE {mfe:+.2f}R | MAE {mae:+.2f}R "
        f"| Hold {row['holding_sessions']} sessions\n"
        f"Exit {row['exit_reason']} at {row['exit_timestamp']}\n"
        f"Instance {row['setup_instance_id']} | Event {row.get('setup_event_id') or 'N/A'}")
    fig.suptitle(header, fontsize=8.0, y=0.995, linespacing=1.05)
    fig.text(0.08, 0.86, context, fontsize=6.8, va="top", linespacing=1.25)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=115, bbox_inches="tight")
    plt.close(fig)


def _draw_trade(axes: Iterable[Any], row: Mapping[str, str]) -> None:
    lines = ((float(row["entry_price"]), "Entry", "#006b45", "-"),
             (float(row["stop_price"]), "Original stop", "#b22222", "--"),
             (float(row["target_price"]), "Target", "#493f9a", ":"))
    for ax in axes:
        for price, label, color, style in lines:
            ax.axhline(price, color=color, linestyle=style, linewidth=0.9, alpha=0.78, label=label)


def render_trade_chart(row: Mapping[str, str], state: Mapping[str, Any], daily: Sequence[Candle],
                      hourly: Sequence[Candle], output: Path) -> dict[str, str] | None:
    """Render one chart or return a row suitable for chart_failures.csv."""
    if not hourly:
        return {"symbol": str(row.get("symbol", "")),
                "setup_instance_id": str(row.get("setup_instance_id", "")),
                "failure_reason": "Hourly data unavailable for chart rendering"}
    try:
        _plot_trade(row, state, daily, hourly, output)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        return {"symbol": str(row.get("symbol", "")),
                "setup_instance_id": str(row.get("setup_instance_id", "")),
                "failure_reason": f"Chart rendering failed: {exc}"}
    return None


def _mfe_bucket(value: float) -> str:
    bounds = [0, .25, .5, 1, 1.5, 2, 2.5, 3]
    labels = ["<=0R", "0-0.25R", "0.25-0.5R", "0.5-1R", "1-1.5R", "1.5-2R", "2-2.5R", "2.5-3R", ">=3R"]
    if value <= 0:
        return labels[0]
    for i, upper in enumerate(bounds[1:], 1):
        if value <= upper:
            return labels[i]
    return labels[-1]


def build_summary(all_rows: Sequence[Mapping[str, str]], losers: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    bucket_order = ["<=0R", "0-0.25R", "0.25-0.5R", "0.5-1R", "1-1.5R", "1.5-2R", "2-2.5R", "2.5-3R", ">=3R"]
    buckets: dict[str, list[Mapping[str, str]]] = {b: [] for b in bucket_order}
    for row in losers:
        buckets[_mfe_bucket(_float(row, "mfe_R"))].append(row)
    bucket_summary = {}
    for key, trades in buckets.items():
        bucket_summary[key] = {
            "trade_count": len(trades),
            "percentage_of_losers": len(trades) / len(losers) if losers else 0.0,
            "average_realized_R": sum(_float(r, "net_R") for r in trades) / len(trades) if trades else None,
            "average_planned_R": sum((float(r["target_price"]) - float(r["entry_price"])) /
                                      (float(r["entry_price"]) - float(r["stop_price"])) for r in trades) / len(trades) if trades else None,
            "average_holding_sessions": sum(_float(r, "holding_sessions") for r in trades) / len(trades) if trades else None,
        }
    family_counts: dict[str, int] = {}
    for row in losers:
        family_counts[row["setup_family"]] = family_counts.get(row["setup_family"], 0) + 1
    by_giveback = sorted(losers, key=lambda r: (-float(r["giveback_R"]), r["entry_timestamp"]))[:10]
    def stats(key: str) -> dict[str, float | None]:
        values = sorted(_float(r, key) for r in losers)
        if not values:
            return {"average": None, "median": None}
        n = len(values)
        med = values[n // 2] if n % 2 else (values[n // 2 - 1] + values[n // 2]) / 2
        return {"average": sum(values) / n, "median": med}
    return {
        "source_arm": "control_no_time",
        "strategy_version": STRATEGY_VERSION,
        "total_executed_closed": sum(r.get("status") == "CLOSED" for r in all_rows),
        "total_winners": sum(r.get("status") == "CLOSED" and _float(r, "net_pnl") > 0 for r in all_rows),
        "total_losers": len(losers),
        "losers_by_setup_family": dict(sorted(family_counts.items())),
        "loser_mfe_buckets": bucket_summary,
        "loser_mfe": stats("mfe_R"), "loser_mae": stats("mae_R"),
        "top_10_giveback_trades": [{"symbol": r["symbol"], "setup_instance_id": r["setup_instance_id"],
            "entry_timestamp": r["entry_timestamp"], "realized_R": _float(r, "net_R"),
            "MFE_R": _float(r, "mfe_R"), "giveback_R": float(r["giveback_R"])} for r in by_giveback],
        "manual_classification_values": list(MANUAL_CLASSIFICATIONS),
        "MFE_MAE_convention": "Recomputed from entry-open candle through exit. Excludes exit candle for gap/time open exits. Includes full intrabar trigger candle for stop/target exits, matching the saved engine's conservative stop-first OHLC ordering; candle-internal sequence is ambiguous.",
        "entry_time_state_source": "Matching saved v2_entry_ready_events.csv row by setup_instance_id; no later zone-map rows are used.",
    }


def _write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _render_gallery(rows: Sequence[Mapping[str, str]], output: Path) -> None:
    cards = []
    for row in rows:
        image = row["chart_file"].replace("\\", "/")
        cards.append(f"<article class='trade' data-setup='{row['setup_family']}' data-year='{row['entry_timestamp'][:4]}' "
                     f"data-mfe='{float(row['MFE_R']):.6f}' data-giveback='{float(row['giveback_R']):.6f}' "
                     f"data-r='{float(row['realized_R']):.6f}'><h2>{row['review_id']} {row['symbol']} | {row['setup_family']}</h2>"
                     f"<p>Entry {row['entry_timestamp'][:10]} | Realized {float(row['realized_R']):+.2f}R | "
                     f"MFE {float(row['MFE_R']):+.2f}R | Giveback {float(row['giveback_R']):+.2f}R</p>"
                     f"<a href='../{image}'><img loading='lazy' src='../{image}'></a></article>")
    html = """<!doctype html><html><head><meta charset='utf-8'><title>V2 Trade Review</title>
<style>body{font:15px Segoe UI,Arial;margin:20px;background:#f4f6f7;color:#17212b}nav{position:sticky;top:0;background:#fff;padding:12px;z-index:2}article{background:#fff;padding:14px;margin:18px 0;border:1px solid #ccd3d8}img{width:100%;height:auto}select,button{margin-right:8px;padding:6px}</style></head>
<body><nav><b>V2 manual trade review</b> &nbsp; Setup <select id='setup'><option value=''>All</option></select> Year <select id='year'><option value=''>All</option></select>
<button onclick='sortBy("mfe")'>Sort MFE</button><button onclick='sortBy("giveback")'>Sort giveback</button><button onclick='sortBy("r")'>Sort realized R</button>
<button onclick='move(-1)'>Previous</button><button onclick='move(1)'>Next</button><span id='count'></span></nav><main id='cards'>""" + "\n".join(cards) + """</main><script>
const cards=[...document.querySelectorAll('.trade')];let pos=0;const setup=document.querySelector('#setup'),year=document.querySelector('#year');
for(const v of [...new Set(cards.map(x=>x.dataset.setup))].sort())setup.add(new Option(v,v));for(const v of [...new Set(cards.map(x=>x.dataset.year))].sort())year.add(new Option(v,v));
function filter(){for(const c of cards)c.hidden=(setup.value&&c.dataset.setup!==setup.value)||(year.value&&c.dataset.year!==year.value);count.textContent=' Visible '+cards.filter(c=>!c.hidden).length+' / '+cards.length;}
function sortBy(k){cards.sort((a,b)=>Number(a.dataset[k])-Number(b.dataset[k]));if(k==='giveback')cards.reverse();const m=document.querySelector('#cards');cards.forEach(c=>m.append(c));}
function move(d){const visible=cards.filter(c=>!c.hidden);if(!visible.length)return;pos=Math.max(0,Math.min(visible.length-1,pos+d));visible[pos].scrollIntoView({behavior:'smooth'});}
setup.onchange=year.onchange=filter;filter();</script></body></html>"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")


def _selection(args: argparse.Namespace, rows: Sequence[dict[str, str]]) -> tuple[list[dict[str, str]], str]:
    losers_only = not args.all_closed
    sort_mode = args.sort
    if args.entry_audit:
        losers_only, sort_mode = True, "mfe_asc"
    if args.profit_giveback_audit:
        losers_only, sort_mode = True, "giveback_desc"
    selected = filter_trades(rows, losers_only=losers_only, symbol=args.symbol, setup=args.setup,
                             year=args.year, min_mfe_r=args.min_mfe_r, max_mfe_r=args.max_mfe_r)
    if args.worst_givebacks:
        sort_mode = "giveback_desc"
    return sort_trades(selected, sort_mode), sort_mode


def generate(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.output) if args.output else DEFAULT_OUTPUT
    for name in ("losers", "winners", "html"):
        (out / name).mkdir(parents=True, exist_ok=True)
    ledger_path = PORTFOLIO_ROOT / "control_no_time_trades.csv"
    signal_path = BACKTEST_ROOT / "v2_entry_ready_events.csv"
    if not ledger_path.is_file() or not signal_path.is_file():
        raise FileNotFoundError("Saved control_no_time ledger or V2 ENTRY_READY artifact is missing")
    ledger = _read_csv(ledger_path)
    if len(ledger) != 74 or sum(r.get("status") == "CLOSED" for r in ledger) != 73:
        raise ValueError("control_no_time ledger failed expected saved-artifact reconciliation (74 rows, 73 closed)")
    if any(r.get("strategy_version") != STRATEGY_VERSION for r in ledger):
        raise ValueError("Portfolio ledger contains a non-V2 strategy_version")
    if any(date.fromisoformat(r["entry_timestamp"][:10]) > RESEARCH_END
           or (r.get("exit_timestamp") and date.fromisoformat(r["exit_timestamp"][:10]) > RESEARCH_END)
           for r in ledger):
        raise ValueError("Saved control_no_time ledger contains a date beyond the 2024 research cutoff")
    events = [r for r in _read_csv(signal_path) if r.get("state") == "ENTRY_READY"]
    by_setup: dict[str, dict[str, str]] = {}
    for signal in events:
        key = signal.get("setup_instance_id", "")
        if not key or key in by_setup:
            raise ValueError(f"Missing or duplicate saved ENTRY_READY setup id: {key}")
        by_setup[key] = signal
    # Every closed trade must map to exactly one saved causal V2 signal.
    for trade in ledger:
        if trade["status"] != "CLOSED":
            continue
        if trade["setup_instance_id"] not in by_setup:
            raise ValueError(f"No saved ENTRY_READY snapshot for {trade['setup_instance_id']}")

    hourly_manifest_path = ROOT / "data/processed/hourly_market_data_manifest.csv"
    if not hourly_manifest_path.is_file():
        raise ValueError("Hourly market-data manifest is missing; identity provenance cannot be checked")
    hourly_manifest = _read_csv(hourly_manifest_path)
    manifest_by_identity: dict[tuple[str, str], list[dict[str, str]]] = {}
    for item in hourly_manifest:
        manifest_by_identity.setdefault((item.get("security_id", ""), item.get("symbol_used", "")), []).append(item)

    if INVALID_IDENTITIES.is_file():
        bad = {r["security_id"]: r["identity_status"] for r in _read_csv(INVALID_IDENTITIES)}
        for trade in ledger:
            if trade["status"] == "CLOSED" and trade["security_id"] in bad:
                raise ValueError(f"Ambiguous hourly identity for executed trade {trade['setup_instance_id']}: {bad[trade['security_id']]}")

    selected, sort_mode = _selection(args, ledger)
    all_losers = filter_trades(ledger, losers_only=True)
    if args.default_mode and (len(selected) != 46 or len(selected) != len(all_losers)):
        raise ValueError("Default loser selection does not reconcile to all 46 saved closed net losers")
    failures = []
    output_rows = []
    trade_candles: dict[str, tuple[list[Candle], list[Candle]]] = {}
    states: dict[str, dict[str, Any]] = {}
    for row in selected:
        sid, symbol, setup_id = row["security_id"], row["symbol"], row["setup_instance_id"]
        identity_rows = manifest_by_identity.get((sid, symbol), [])
        if len(identity_rows) != 1 or identity_rows[0].get("validation_status") != "VALIDATED":
            raise ValueError(f"Hourly manifest identity is missing/ambiguous/unvalidated for {setup_id}")
        signal = by_setup[setup_id]
        state = causal_entry_state(signal, entry_date=_dt(row["entry_timestamp"]).date())
        if state["nifty_regime"] == "N/A" or not state["plan"]:
            raise ValueError(f"Saved entry-time state incomplete for {setup_id}")
        if not math.isclose(float(state["plan"].get("entry")), _float(row, "entry_raw_open"), abs_tol=0.01, rel_tol=0):
            raise ValueError(f"Saved signal entry does not reconcile for {setup_id}")
        dpath = DAILY_ROOT / sid / f"{symbol}.csv"
        hfiles = list((HOURLY_ROOT / sid).glob(f"{symbol}_*.csv"))
        if len(hfiles) != 1:
            failures.append({"symbol": symbol, "setup_instance_id": setup_id,
                             "failure_reason": f"Expected one identity-matched hourly file, found {len(hfiles)}"})
            continue
        try:
            if not dpath.is_file():
                raise ValueError("Daily data file is missing")
            daily, hourly = load_candles(dpath, through=RESEARCH_END), load_candles(hfiles[0], through=RESEARCH_END)
            entry, exit_ = _dt(row["entry_timestamp"]), _dt(row["exit_timestamp"])
            entry_bar = next((c for c in hourly if c.timestamp == entry), None)
            exit_bar = next((c for c in hourly if c.timestamp == exit_), None)
            if entry_bar is None or exit_bar is None:
                raise ValueError("Entry or exit timestamp does not map to actual hourly data")
            if not math.isclose(entry_bar.open, _float(row, "entry_raw_open"), abs_tol=0.01, rel_tol=0):
                raise ValueError("Hourly entry open differs from portfolio raw execution price")
            raw_exit = _float(row, "raw_exit_price")
            reason = row["exit_reason"]
            if reason.startswith("GAP_"):
                if not math.isclose(exit_bar.open, raw_exit, abs_tol=0.05, rel_tol=0):
                    raise ValueError("Gap exit raw fill does not match exit candle open")
            elif not exit_bar.low - 0.05 <= raw_exit <= exit_bar.high + 0.05:
                raise ValueError("Intrabar raw exit fill lies outside exit candle range")
            if not math.isclose(_float(row, "exit_price"), raw_exit, abs_tol=0.05, rel_tol=0):
                raise ValueError("Execution fill differs from raw exit; slippage/cost basis is not reconciled")
            mfe, mae, ambiguous = trade_path_excursions(hourly, entry_timestamp=row["entry_timestamp"],
                exit_timestamp=row["exit_timestamp"], entry_price=_float(row, "entry_price"),
                stop_price=_float(row, "stop_price"), exit_reason=reason)
            if not math.isclose(mfe, _float(row, "mfe_R"), abs_tol=1e-7, rel_tol=0):
                raise ValueError(f"Recomputed MFE_R {mfe} differs from ledger {_float(row, 'mfe_R')}")
            if not math.isclose(mae, _float(row, "mae_R"), abs_tol=1e-7, rel_tol=0):
                raise ValueError(f"Recomputed MAE_R {mae} differs from ledger {_float(row, 'mae_R')}")
            planned_r = (float(row["target_price"]) - float(row["entry_price"])) / (
                float(row["entry_price"]) - float(row["stop_price"]))
            if not math.isclose(planned_r, _float(row, "planned_rr"), abs_tol=1e-7, rel_tol=0):
                raise ValueError("Planned R does not reconcile with saved entry, stop, and target")
            trade_candles[setup_id], states[setup_id] = (daily, hourly), state
            row["_computed_mfe"], row["_computed_mae"], row["_ambiguous"] = str(mfe), str(mae), str(ambiguous)
        except (OSError, ValueError, KeyError) as exc:
            failures.append({"symbol": symbol, "setup_instance_id": setup_id, "failure_reason": str(exc)})
    if failures:
        _write_csv(out / "chart_failures.csv", ("symbol", "setup_instance_id", "failure_reason"), failures)
        raise ValueError(f"{len(failures)} trade(s) failed identity/data/execution/MFE validation; see {out / 'chart_failures.csv'}")
    if len(trade_candles) != len(selected):
        raise ValueError("Validated chart input count differs from selected ledger rows")

    manifest = []
    render_failures = []
    for index, row in enumerate(selected, 1):
        filename = deterministic_filename(index, row)
        chart_subdir = "winners" if _float(row, "net_pnl") > 0 else "losers"
        image_path = out / chart_subdir / filename
        daily, hourly = trade_candles[row["setup_instance_id"]]
        failure = render_trade_chart(row, states[row["setup_instance_id"]], daily, hourly, image_path)
        if failure:
            render_failures.append(failure)
            continue
        manifest.append({
            "review_id": f"{index:03d}", "symbol": row["symbol"], "setup_family": row["setup_family"],
            "setup_event_id": row.get("setup_event_id", ""), "setup_instance_id": row["setup_instance_id"],
            "signal_timestamp": states[row["setup_instance_id"]]["confirmation_timestamp"],
            "entry_timestamp": row["entry_timestamp"], "entry_price": row["entry_price"],
            "stop_price": row["stop_price"], "target_price": row["target_price"],
            "planned_R": (float(row["target_price"]) - float(row["entry_price"])) /
                          (float(row["entry_price"]) - float(row["stop_price"])),
            "quantity": row["quantity"], "position_notional": row["position_notional"],
            "exit_timestamp": row["exit_timestamp"], "exit_price": row["exit_price"],
            "exit_reason": row["exit_reason"], "realized_gross_pnl": row["gross_pnl"],
            "realized_net_pnl": row["net_pnl"], "realized_R": row["net_R"],
            "MFE_R": row["_computed_mfe"], "MAE_R": row["_computed_mae"],
            "giveback_R": float(row["_computed_mfe"]) - float(row["net_R"]),
            "holding_sessions": row["holding_sessions"], "chart_file": f"{chart_subdir}/{filename}",
            "strategy_version": row["strategy_version"], **{field: "" for field in MANUAL_FIELDS},
        })
    _write_csv(out / "review_manifest.csv", MANIFEST_FIELDS, manifest)
    _write_csv(out / "chart_failures.csv", ("symbol", "setup_instance_id", "failure_reason"), render_failures)
    summary = build_summary(ledger, all_losers)
    summary.update({"generated_review_count": len(manifest), "selected_trade_count": len(selected),
                    "selection": "LOSERS_ONLY" if not args.all_closed else "ALL_CLOSED",
                    "sort_mode": sort_mode, "chart_failures": len(failures) + len(render_failures),
                    "market_data_cutoff": RESEARCH_END.isoformat(), "out_of_sample_data_loaded": False,
                    "saved_portfolio_ledger": str(ledger_path.relative_to(ROOT)).replace("\\", "/"),
                    "saved_signal_artifact": str(signal_path.relative_to(ROOT)).replace("\\", "/"),
                    "chart_subdirectories": sorted({Path(r["chart_file"]).parts[0] for r in manifest})})
    (out / "review_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    _render_gallery(manifest, out / "html/index.html")
    return {"output": out, "manifest": manifest, "summary": summary,
            "failures": failures + render_failures, "sort_mode": sort_mode}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate manual-review charts from saved V2 portfolio trades")
    parser.add_argument("--losers-only", action="store_true", default=True, help="review CLOSED net losers (default)")
    parser.add_argument("--all-closed", action="store_true", help="include winners and losers")
    parser.add_argument("--symbol")
    parser.add_argument("--setup")
    parser.add_argument("--year", type=int)
    parser.add_argument("--min-mfe-r", type=float)
    parser.add_argument("--max-mfe-r", type=float)
    parser.add_argument("--sort", choices=("mfe_asc", "giveback_desc", "realized_r_asc", "chronological"), default="mfe_asc")
    parser.add_argument("--worst-givebacks", action="store_true")
    parser.add_argument("--entry-audit", action="store_true")
    parser.add_argument("--profit-giveback-audit", action="store_true")
    parser.add_argument("--output", help="optional output directory")
    parser.add_argument("--default-mode", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.default_mode = not any((args.all_closed, args.symbol, args.setup, args.year,
        args.min_mfe_r is not None, args.max_mfe_r is not None, args.worst_givebacks,
        args.entry_audit, args.profit_giveback_audit)) and args.sort == "mfe_asc"
    try:
        result = generate(args)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"Trade review stopped: {exc}\n")
    print(f"Reviewed {len(result['manifest'])} trades; failures={len(result['failures'])}; sort={result['sort_mode']}")
    print(f"Manifest: {result['output'] / 'review_manifest.csv'}")
    print(f"Gallery:  {result['output'] / 'html/index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
