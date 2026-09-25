"""Causal replay for top_down_swing_v1 on frozen 2023-2024 research inputs."""

from __future__ import annotations

import argparse
import csv
from datetime import date, datetime, time
import hashlib
import json
import csv
from pathlib import Path
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

import yaml

from src.backtest.data import (BacktestDataError, load_daily_dataset_through,
                               load_hourly_dataset_through)
from src.backtest.weekly_selection_run import load_nifty_benchmark
from src.configuration import PROJECT_ROOT
from src.data.market_data import DATA_ROOT, MANIFEST_PATH
from src.data.universe.provider import PIT_TRADABLE, UniverseProvider
from src.strategy.top_down_swing import (TopDownConfig, _lifecycle_snapshot, _zones, confirm_hourly,
                                         evaluate_daily_candidate)
from src.strategy.weekly_regime import WeeklyRegimeConfig, weekly_regime
from src.strategy.weekly_selection import WeeklySelectionConfig, completed_weekly_rows, weekly_features


START, END = date(2023, 1, 1), date(2024, 12, 31)
DAILY_WARMUP, HOURLY_WARMUP = date(2022, 7, 1), date(2022, 10, 1)
TZ = ZoneInfo("Asia/Kolkata")
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "top_down_swing_v1.yaml"
DEFAULT_BENCHMARK = PROJECT_ROOT / "data" / "raw" / "reference" / "nifty50_daily.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "backtests" / "top_down_swing_v1"


def _csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        if columns:
            writer.writeheader()
            for row in rows:
                writer.writerow({key: json.dumps(value, sort_keys=True, ensure_ascii=True)
                                 if isinstance(value, (dict, list, tuple)) else value
                                 for key, value in row.items()})


def _dt(value: Any) -> datetime:
    result = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
    return result.replace(tzinfo=TZ) if result.tzinfo is None else result.astimezone(TZ)


def stable_sample(rows: Sequence[Mapping[str, Any]], *, group: str, limit: int = 25) -> list[dict[str, Any]]:
    """Select the same review rows independent of process hash/random state."""
    ordered = sorted(rows, key=lambda row: hashlib.sha256(
        f"{group}|{row.get('security_id')}|{row.get('date')}".encode("utf-8")).hexdigest())
    return [{"sample_group": group, **dict(row)} for row in ordered[:limit]]


def setup_instance_id(security_id: str, signal_date: str, setup_type: str,
                      zone_id: str | None) -> str:
    seed = f"{security_id}|{signal_date}|{setup_type}|{zone_id or 'NO_ZONE'}"
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:20]


class SetupRegistry:
    """Keep stock scanning persistent while individual setup instances terminate."""

    def __init__(self) -> None:
        self.active: dict[str, dict[str, Any]] = {}
        self.last_terminal: dict[str, str] = {}
        self.later_setup_counts: dict[str, int] = {}

    def start(self, security_id: str, setup: dict[str, Any]) -> None:
        previous = self.last_terminal.get(security_id)
        if previous:
            self.later_setup_counts[previous] = self.later_setup_counts.get(previous, 0) + 1
        self.active[security_id] = setup

    def finish(self, security_id: str, terminal_state: str) -> dict[str, Any]:
        setup = self.active.pop(security_id)
        self.last_terminal[security_id] = terminal_state
        return setup


def _completed_week(rows: Sequence[Mapping[str, Any]], day: date) -> list[dict[str, Any]]:
    return completed_weekly_rows(rows, day)


def replay(*, start: date = START, end: date = END, output_dir: str | Path = DEFAULT_OUTPUT,
           config_path: str | Path = DEFAULT_CONFIG, data_root: str | Path = DATA_ROOT,
           daily_manifest: str | Path = MANIFEST_PATH,
           hourly_manifest: str | Path | None = None,
           benchmark_path: str | Path = DEFAULT_BENCHMARK,
           max_days: int | None = None) -> dict[str, Any]:
    if start < START or end > END or start > end:
        raise ValueError("top_down_swing_v1 replay is limited to 2023-01-01 through 2024-12-31.")
    raw_cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    if raw_cfg.get("version") != "top_down_swing_v1":
        raise ValueError("Configuration version must be top_down_swing_v1.")
    risk = raw_cfg.get("risk", {})
    daily_cfg, weekly_cfg, hourly_cfg = raw_cfg.get("daily", {}), raw_cfg.get("weekly", {}), raw_cfg.get("hourly", {})
    ema_cfg = raw_cfg.get("ema_context", {})
    from src.strategy.sr_zone_v2 import ZoneConfig
    cfg = TopDownConfig(
        weekly_approach_distance_atr=float(weekly_cfg.get("approach_distance_atr", 1.5)),
        mid_range_lower_fraction=float(weekly_cfg.get("mid_range_lower_fraction", .35)),
        mid_range_upper_fraction=float(weekly_cfg.get("mid_range_upper_fraction", .65)),
        daily_approach_distance_atr=float(daily_cfg.get("approach_distance_atr", 1.0)),
        support_reaction_close_location=float(daily_cfg.get("support_reaction_close_location", .60)),
        breakout_retest_max_sessions=int(daily_cfg.get("breakout_retest_max_sessions", 20)),
        confirmation_expiry_sessions=int(hourly_cfg.get("confirmation_expiry_sessions", 10)),
        consolidation_bars=int(hourly_cfg.get("consolidation_bars", 6)),
        stop_buffer_atr=float(hourly_cfg.get("stop_buffer_atr", .15)),
        minimum_room_r=float(risk.get("minimum_room_r", 1.5)),
        initial_capital=float(risk.get("initial_capital", 10_000)),
        risk_fraction=float(risk.get("risk_fraction", .01)),
        max_open_positions=int(risk.get("max_open_positions", 10)),
        zone_acceptance_closes=int(hourly_cfg.get("zone_acceptance_closes", 2)),
        zone_break_buffer_atr=float(hourly_cfg.get("zone_break_buffer_atr", .1)),
        meaningful_touch_episodes=int(weekly_cfg.get("meaningful_touch_episodes", 2)),
        major_touch_episodes=int(weekly_cfg.get("major_touch_episodes", 4)),
        meaningful_reaction_atr=float(weekly_cfg.get("meaningful_reaction_atr", .5)),
        major_reaction_atr=float(weekly_cfg.get("major_reaction_atr", 1.0)),
        meaningful_recency_bars=int(weekly_cfg.get("meaningful_recency_bars", 130)),
        major_recency_bars=int(weekly_cfg.get("major_recency_bars", 65)),
        meaningful_pivots=int(weekly_cfg.get("meaningful_pivots", 2)),
        actionable_support_distance_atr=float(weekly_cfg.get("actionable_support_distance_atr", 1.5)),
        actionable_resistance_distance_atr=float(weekly_cfg.get("actionable_resistance_distance_atr", 1.0)),
        context_min_completed_weeks=int(weekly_cfg.get("context_min_completed_weeks", 26)),
        context_min_rs_weeks=int(weekly_cfg.get("context_min_rs_weeks", 13)),
        no_resistance_fallback_r=float(risk.get("no_resistance_fallback_r", 2.0)),
        ema_confluence_distance_atr=float(ema_cfg.get("confluence_distance_atr", .25)),
        weekly_zone=ZoneConfig(swing_left=int(weekly_cfg.get("swing_left", 2)),
            swing_right=int(weekly_cfg.get("swing_right", 2)), minimum_pivots=2, maximum_age=104,
            minimum_width_pct=.003, maximum_width_pct=.02,
            departure_atr=float(weekly_cfg.get("departure_atr", .5)),
            reaction_horizon=int(weekly_cfg.get("reaction_horizon", 5))),
        daily_zone=ZoneConfig(swing_left=int(daily_cfg.get("swing_left", 3)),
            swing_right=int(daily_cfg.get("swing_right", 3)), minimum_pivots=int(daily_cfg.get("minimum_zone_pivots", 2)),
            maximum_age=int(daily_cfg.get("maximum_age", 252)),
            departure_atr=float(daily_cfg.get("departure_atr", .5)),
            reaction_horizon=int(daily_cfg.get("reaction_horizon", 5))),
        hourly_zone=ZoneConfig(swing_left=int(hourly_cfg.get("swing_left", 3)),
            swing_right=int(hourly_cfg.get("swing_right", 3)), minimum_pivots=int(hourly_cfg.get("minimum_zone_pivots", 2)),
            maximum_age=int(hourly_cfg.get("maximum_age", 130)),
            departure_atr=float(hourly_cfg.get("departure_atr", .5)),
            reaction_horizon=int(hourly_cfg.get("reaction_horizon", 5))),
    )
    daily, daily_hash = load_daily_dataset_through(universe_id=PIT_TRADABLE,
        start_date=DAILY_WARMUP, end_date=end, data_root=data_root, manifest_path=daily_manifest)
    hourly, hourly_hash = load_hourly_dataset_through(universe_id=PIT_TRADABLE,
        start_date=HOURLY_WARMUP, end_date=end, data_root=data_root, manifest_path=hourly_manifest,
        daily_manifest_path=daily_manifest)
    benchmark = load_nifty_benchmark(benchmark_path)
    # Preserve the prior version's summary and room-rejection target references
    # before deterministic output paths are replaced by this replay.
    output = Path(output_dir)
    old_report: dict[str, Any] = {}
    old_room_target_counts: dict[str, int] = {}
    old_room_instances: dict[tuple[str, str], str] = {}
    old_report_path = output / "top_down_swing_v1_replay_2023_2024.json"
    old_trace_path = output / "strategy_trace_2023_2024.csv"
    if old_report_path.exists():
        old_report = json.loads(old_report_path.read_text(encoding="utf-8"))
    if old_trace_path.exists():
        old_room_seen: set[tuple[str, str]] = set()
        with old_trace_path.open(encoding="utf-8", newline="") as old_file:
            for old_row in csv.DictReader(old_file):
                if old_row.get("state") != "ROOM_INSUFFICIENT":
                    continue
                instance_key = (old_row.get("security_id", ""), old_row.get("setup_signal_date", ""))
                if instance_key in old_room_seen:
                    continue
                old_room_seen.add(instance_key)
                try:
                    result = json.loads(old_row.get("hourly_result") or "{}")
                    reference = str(result.get("plan", {}).get("target_reference") or "NO_TARGET")
                except (ValueError, TypeError):
                    reference = "UNKNOWN"
                bucket = ("EMA_HARD_TARGET" if "EMA" in reference else
                          "HOURLY_RESISTANCE_HARD_TARGET" if reference == "HOURLY" else
                          "NO_STRUCTURAL_TARGET" if reference == "NO_TARGET" else "DAILY_OR_WEEKLY_STRUCTURAL")
                old_room_target_counts[bucket] = old_room_target_counts.get(bucket, 0) + 1
                old_room_instances[instance_key] = bucket
    provider = UniverseProvider()
    reg_cfg = WeeklyRegimeConfig()
    sel_cfg = WeeklySelectionConfig()
    # NIFTY and stock selector outputs are cached per signal date; all functions
    # receive data through that date and weekly_features excludes its calendar week.
    dates = sorted({date.fromisoformat(str(row["date"])[:10]) for rows in daily.values() for row in rows
                    if start <= date.fromisoformat(str(row["date"])[:10]) <= end})
    if max_days is not None:
        dates = dates[:max_days]
    nifty_by_day = {date.fromisoformat(str(row["date"])[:10]): row for row in benchmark
                    if start <= date.fromisoformat(str(row["date"])[:10]) <= end}
    trace: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    registry = SetupRegistry()
    active = registry.active
    previous_state: dict[str, str] = {}
    last_terminal_by_sid = registry.last_terminal
    later_setup_counts = registry.later_setup_counts
    terminal_states = {"ROOM_INSUFFICIENT", "SETUP_INVALIDATED", "HOURLY_DATA_UNAVAILABLE",
                       "SETUP_EXPIRED", "ENTRY_READY_EVENT"}
    by_sid_date = {sid: {date.fromisoformat(str(r["date"])[:10]): i for i, r in enumerate(rows)}
                   for sid, rows in daily.items()}
    for day in dates:
        day_members = {member.security_id: member for member in provider.members_on(day, PIT_TRADABLE)}
        regime = weekly_regime(nifty_daily=benchmark, as_of=day, config=reg_cfg)
        for sid in sorted(day_members):
            if sid not in daily or day not in by_sid_date.get(sid, {}):
                continue
            was_active = sid in active
            index = by_sid_date[sid][day]
            prefix = daily[sid][:index+1]
            weeks = _completed_week(prefix, day)
            symbol = day_members[sid].symbol
            state_row: dict[str, Any]
            if sid in active:
                pending = active[sid]
                instance_id = pending["setup_instance_id"]
                if pending["state"] == "WAITING_1H_CONFIRMATION" and previous_state.get(sid) == "SETUP_ACTIVE":
                    events.append({"date": day.isoformat(), "security_id": sid, "symbol": symbol,
                                   "from_state": "SETUP_ACTIVE", "to_state": "WAITING_1H_CONFIRMATION",
                                   "reason_codes": ["NEXT_SESSION_HOURLY_MONITORING_STARTED"]})
                    previous_state[sid] = "WAITING_1H_CONFIRMATION"
                elapsed = (day - date.fromisoformat(pending["signal_date"])).days
                session_count = sum(date.fromisoformat(str(r["date"])[:10]) > date.fromisoformat(pending["signal_date"])
                                    and date.fromisoformat(str(r["date"])[:10]) <= day
                                    for r in daily[sid])
                if pending["state"] == "WAITING_1H_CONFIRMATION":
                    if not hourly.get(sid):
                        pending["state"] = "HOURLY_DATA_UNAVAILABLE"
                        pending["result"] = {"reason_codes": ["HOURLY_DATA_UNAVAILABLE"]}
                        pending["updated_on"] = day.isoformat()
                    elif float(prefix[-1]["close"]) < float(pending["setup_zone"]["lower"]):
                        pending["state"] = "SETUP_INVALIDATED"
                        pending["result"] = {"reason_codes": ["DAILY_SETUP_ZONE_LOST"]}
                        pending["updated_on"] = day.isoformat()
                    else:
                        h1 = [r for r in hourly.get(sid, ()) if _dt(r["date"]).date() <= day]
                        h_at_signal = datetime.combine(date.fromisoformat(pending["signal_date"]), time(15, 30), tzinfo=TZ)
                        result = confirm_hourly(hourly_rows=h1, after_timestamp=h_at_signal,
                            setup_zone=pending["setup_zone"], weekly_zones=pending["weekly_zones"],
                            daily_zones=pending["daily_zones"], hourly_config=cfg.hourly_zone,
                            ema_obstacles={key: pending["candidate"].get(key) for key in
                                           ("daily_ema50", "daily_ema200", "weekly_ema50", "weekly_ema200")},
                            require_stronger_confirmation=(pending["candidate"].get("weekly_selection_path") ==
                                                           "STRUCTURAL_LOCATION_EXCEPTION"),
                            start_index=max(0, int(pending.get("last_hourly_scan_index", -1)) -
                                            (1 if pending.get("awaiting_next_bar") else 0)),
                            security_id=sid,
                            config=cfg)
                        pending["result"] = result
                        pending["updated_on"] = day.isoformat()
                        pending["awaiting_next_bar"] = "AWAITING_NEXT_EXECUTABLE_HOURLY_BAR" in result.get("reason_codes", [])
                        pending["last_hourly_scan_index"] = (result.get("confirmation_index", len(h1)-1)
                            if pending["awaiting_next_bar"] else len(h1)-1)
                        if result["state"] == "ENTRY_READY":
                            pending["state"] = "ENTRY_READY_EVENT"
                            events.append({"date": day.isoformat(), "security_id": sid, "symbol": symbol,
                                           "setup_instance_id": instance_id,
                                           "from_state": "WAITING_1H_CONFIRMATION", "to_state": "ENTRY_READY_EVENT",
                                           "reason_codes": result.get("reason_codes"), "details": result})
                        elif result["state"] in {"ROOM_INSUFFICIENT", "SETUP_INVALIDATED"}:
                            pending["state"] = result["state"]
                        elif session_count > cfg.confirmation_expiry_sessions:
                            pending["state"] = "WATCHLIST_EXPIRED"
                            pending["result"] = {"reason_codes": ["HOURLY_CONFIRMATION_WINDOW_EXPIRED"]}
                            pending["updated_on"] = day.isoformat()
                    if pending["state"] in terminal_states - {"ENTRY_READY_EVENT"}:
                        events.append({"date": day.isoformat(), "security_id": sid, "symbol": symbol,
                                       "setup_instance_id": instance_id,
                                       "from_state": "WAITING_1H_CONFIRMATION", "to_state": pending["state"],
                                       "reason_codes": pending["result"].get("reason_codes", [])})
                state_row = {**pending["candidate"], "state": pending["state"],
                             "reason_codes": pending["result"].get("reason_codes", ["WAITING_FOR_HOURLY_CONFIRMATION"]),
                             "hourly_result": pending["result"] if pending.get("updated_on") == day.isoformat() else None,
                             "setup_signal_date": pending["signal_date"],
                             "confirmation_sessions_elapsed": session_count,
                             "setup_instance_id": instance_id,
                             "setup_state": pending["state"], "stock_state": "WAITING_1H_CONFIRMATION"}
                if pending["state"] in terminal_states:
                    registry.finish(sid, pending["state"])
                    state_row["stock_state"] = "APPROACHING_ZONE"
                    if pending["state"] == "ENTRY_READY_EVENT":
                        state_row["stock_state"] = "ENTRY_READY_EVENT"
            else:
                selector = weekly_features(stock_daily=prefix, benchmark_daily=benchmark,
                    as_of=day, security_id=sid, config=sel_cfg, zone_config=cfg.weekly_zone)
                candidate = evaluate_daily_candidate(security_id=sid, symbol=symbol,
                    daily_rows=prefix, weekly_rows=weeks, nifty_weekly_regime=regime,
                    weekly_selection=selector, config=cfg)
                state_row = candidate
                if candidate.get("state") == "WAITING_1H_CONFIRMATION":
                    setup_id = setup_instance_id(sid, day.isoformat(), candidate.get("setup_type", "UNKNOWN"),
                        (candidate.get("setup_zone") or {}).get("zone_id"))
                    state_row = {**candidate, "state": "SETUP_ACTIVE"}
                    registry.start(sid, {"signal_date": day.isoformat(), "state": "WAITING_1H_CONFIRMATION",
                                   "setup_instance_id": setup_id,
                                   "candidate": {**candidate, "setup_instance_id": setup_id},
                                   "result": {"reason_codes": ["DAILY_SETUP_CONFIRMED"]},
                                   "setup_zone": candidate["setup_zone"],
                                   "weekly_zones": _lifecycle_snapshot(_zones(weeks, security_id=sid, config=cfg.weekly_zone), weeks, cfg.weekly_zone, cfg),
                                   "daily_zones": _lifecycle_snapshot(_zones(prefix, security_id=sid, config=cfg.daily_zone), prefix, cfg.daily_zone, cfg)})
                    state_row.update({"setup_instance_id": setup_id, "setup_state": "SETUP_ACTIVE",
                                      "stock_state": "SETUP_ACTIVE"})
            state = str(state_row.get("state", "UNINTERESTING"))
            old = previous_state.get(sid)
            if not was_active and old != state:
                events.append({"date": day.isoformat(), "security_id": sid, "symbol": symbol,
                               "setup_instance_id": state_row.get("setup_instance_id"),
                               "from_state": old, "to_state": state,
                               "reason_codes": state_row.get("reason_codes", [])})
            if was_active:
                previous_state[sid] = state
            elif old != state:
                previous_state[sid] = state
            trace.append({"date": day.isoformat(), "security_id": sid, "symbol": symbol,
                          "state": state, "stock_state": state_row.get("stock_state", state),
                          "setup_state": state_row.get("setup_state"),
                          "setup_instance_id": state_row.get("setup_instance_id"),
                          "setup_type": state_row.get("setup_type"),
                          "setup_signal_date": state_row.get("setup_signal_date", state_row.get("date")),
                          "reason_codes": state_row.get("reason_codes", []),
                          "nifty_regime": state_row.get("nifty_regime", regime.get("regime")),
                          "nifty_regime_components": state_row.get("nifty_regime_component_values",
                                                                      regime.get("component_values", {})),
                          "weekly_location": state_row.get("weekly_location"),
                          "weekly_location_components": state_row.get("weekly_location_components"),
                          "structural_range_position": state_row.get("structural_range_position"),
                          "weekly_structure": state_row.get("weekly_structure"),
                          "weekly_selector_status": state_row.get("weekly_selector_status"),
                          "weekly_selector_score": state_row.get("weekly_selector_score"),
                          "weekly_selection_components": state_row.get("weekly_selection_components"),
                          "weekly_selection_path": state_row.get("weekly_selection_path"),
                          "NIFTY_CONTEXT_READY": state_row.get("NIFTY_CONTEXT_READY"),
                          "WEEKLY_STOCK_CONTEXT_READY": state_row.get("WEEKLY_STOCK_CONTEXT_READY"),
                          "daily_zones": state_row.get("daily_zones"),
                          "weekly_zones": state_row.get("weekly_zones"),
                          "daily_atr": state_row.get("daily_atr"), "weekly_atr": state_row.get("weekly_atr"),
                          "daily_ema50": state_row.get("daily_ema50"), "daily_ema200": state_row.get("daily_ema200"),
                          "daily_volume": state_row.get("daily_volume"),
                          "weekly_volume": state_row.get("weekly_volume"),
                          "daily_bar": state_row.get("daily_bar"), "setup_zone": state_row.get("setup_zone"),
                          "hourly_result": state_row.get("hourly_result"),
                          "confirmation_sessions_elapsed": state_row.get("confirmation_sessions_elapsed")})

    # Deterministic, event-level audit samples; stock-state observations are not
    # mistaken for setup-instance outcomes.
    trace_by_instance = {r.get("setup_instance_id"): r for r in trace if r.get("setup_instance_id")}
    trace_by_day_sid = {(r["date"], r["security_id"]): r for r in trace}
    def enrich(row: Mapping[str, Any], group: str) -> dict[str, Any]:
        value = dict(row)
        result = value.get("hourly_result") or {}
        plan = result.get("plan") or {}
        daily_context = value.get("daily_zones") or {}
        weekly_context = value.get("weekly_zones") or {}
        value.update({"sample_group": group,
            "nifty_context": {"regime": value.get("nifty_regime"), "components": value.get("nifty_regime_components")},
            "weekly_selection_path": value.get("weekly_selection_path"),
            "weekly_support": weekly_context.get("support"), "weekly_resistance": weekly_context.get("resistance"),
            "daily_support": daily_context.get("support"), "daily_resistance": daily_context.get("resistance"),
            "current_zone_state": (value.get("setup_zone") or {}).get("role_state"),
            "role_flip_state": (value.get("setup_zone") or {}).get("role_flip"),
            "structural_range_position": value.get("structural_range_position"),
            "entry": plan.get("entry"), "stop": plan.get("stop"), "risk": plan.get("risk_per_share"),
            "primary_target": plan.get("target"), "target_timeframe": plan.get("target_timeframe"),
            "reward": plan.get("planned_reward_r"), "planned_rr": plan.get("planned_reward_r"),
            "minor_friction": [o for o in plan.get("obstacles", []) if o.get("classification") == "SECONDARY_FRICTION"],
            "ema_context": plan.get("ema_context"),
            "confirmation_type": result.get("confirmation_type"),
            "confirmation_evidence": result.get("confirmation_features"),
            "volume_evidence": {"daily": value.get("daily_volume"),
                                "weekly": value.get("weekly_volume"),
                                "hourly": (result.get("confirmation_features") or {})},
        })
        return value
    ready_events = [e for e in events if e.get("to_state") == "ENTRY_READY_EVENT"]
    reject_events = [e for e in events if e.get("to_state") in {"ROOM_INSUFFICIENT", "SETUP_INVALIDATED"}]
    mid_rows = [r for r in trace if r.get("state") == "MID_RANGE_NOT_ACTIONABLE"]
    earlier_terminal: dict[str, list[Mapping[str, Any]]] = {}
    for e in reject_events:
        earlier_terminal.setdefault(str(e["security_id"]), []).append(e)
    later_ready = []
    for event in ready_events:
        previous = [e for e in earlier_terminal.get(str(event["security_id"]), [])
                    if str(e["date"]) < str(event["date"])]
        if previous:
            row = trace_by_instance.get(event.get("setup_instance_id")) or trace_by_day_sid.get(
                (event["date"], event["security_id"]))
            if row:
                later_ready.append(row)
    def sample_events(events_to_sample: Sequence[Mapping[str, Any]], group: str, limit: int) -> list[dict[str, Any]]:
        rows = []
        for event in events_to_sample:
            row = trace_by_instance.get(event.get("setup_instance_id")) or trace_by_day_sid.get(
                (event.get("date"), event.get("security_id")))
            if row:
                rows.append(enrich(row, group))
        return stable_sample(rows, group=group, limit=limit)
    samples = (sample_events(ready_events, "ENTRY_READY", 30) +
               sample_events(reject_events, "ROOM_OR_INVALIDATED", 30) +
               [enrich(row, "MID_RANGE") for row in stable_sample(mid_rows, group="MID_RANGE", limit=30)] +
               [enrich(row, "LATER_ACTIONABLE_AFTER_TERMINAL") for row in
                stable_sample(later_ready, group="LATER_ACTIONABLE_AFTER_TERMINAL", limit=20)])
    changed_old_room: dict[str, int] = {"EMA_NO_LONGER_HARD_TARGET": 0,
        "HOURLY_RESISTANCE_RECLASSIFIED_BY_STRENGTH": 0, "NO_MAJOR_TARGET_FIXED_2R_FALLBACK": 0}
    for event in ready_events:
        row = trace_by_instance.get(event.get("setup_instance_id"))
        if not row:
            continue
        old_bucket = old_room_instances.get((str(row["security_id"]), str(row.get("setup_signal_date"))))
        if old_bucket == "EMA_HARD_TARGET":
            changed_old_room["EMA_NO_LONGER_HARD_TARGET"] += 1
        elif old_bucket == "HOURLY_RESISTANCE_HARD_TARGET":
            changed_old_room["HOURLY_RESISTANCE_RECLASSIFIED_BY_STRENGTH"] += 1
        elif old_bucket == "NO_STRUCTURAL_TARGET":
            changed_old_room["NO_MAJOR_TARGET_FIXED_2R_FALLBACK"] += 1
    def breakdown(events_to_count: Sequence[Mapping[str, Any]], key_fn) -> dict[str, int]:
        result: dict[str, int] = {}
        for event in events_to_count:
            trace_row = trace_by_instance.get(event.get("setup_instance_id"), {})
            label = str(key_fn(event, trace_row))
            result[label] = result.get(label, 0) + 1
        return dict(sorted(result.items()))
    entry_ready_breakdowns = {
        "year": breakdown(ready_events, lambda e, r: str(e.get("date", ""))[:4]),
        "setup_family": breakdown(ready_events, lambda e, r: r.get("setup_type") or "UNKNOWN"),
        "confirmation_archetype": breakdown(ready_events, lambda e, r: (r.get("hourly_result") or {}).get("confirmation_type") or "UNKNOWN"),
        "weekly_selection_path": breakdown(ready_events, lambda e, r: r.get("weekly_selection_path") or "UNKNOWN"),
        "primary_target_timeframe": breakdown(ready_events, lambda e, r: ((r.get("hourly_result") or {}).get("plan") or {}).get("target_timeframe") or "UNKNOWN"),
        "primary_target_strength": breakdown(ready_events, lambda e, r: ((r.get("hourly_result") or {}).get("plan") or {}).get("target_zone_strength") or "NO_STRUCTURAL_TARGET_FALLBACK"),
    }
    output = Path(output_dir)
    trace_path = output / "strategy_trace_2023_2024.csv"
    events_path = output / "state_transitions_2023_2024.csv"
    sample_path = output / "manual_review_sample_2023_2024.csv"
    report_path = output / "top_down_swing_v1_replay_2023_2024.json"
    _csv(trace_path, trace)
    _csv(events_path, events)
    _csv(sample_path, samples)
    setup_instances = {r.get("setup_instance_id") for r in trace if r.get("setup_instance_id")}
    terminal_counts = {state: sum(e.get("to_state") == state for e in events)
        for state in sorted(terminal_states)}
    full_context_dates = [r["date"] for r in trace if r.get("NIFTY_CONTEXT_READY") and
                          r.get("WEEKLY_STOCK_CONTEXT_READY")]
    report = {"strategy_version": "top_down_swing_v1", "period": {"start": start.isoformat(), "end": end.isoformat()},
              "warmup": {"daily": DAILY_WARMUP.isoformat(), "hourly": HOURLY_WARMUP.isoformat()},
              "universe": PIT_TRADABLE, "frozen_data_inputs_read_only": True,
              "2025_2026_loaded": False, "nifty_regime_is_soft_context": True,
              "daily_signal_hourly_data_excluded": True,
              "not_a_performance_backtest": True, "exit_simulation_implemented": False,
              "max_open_positions_enforced": False,
              "position_sizing_scope": "per-entry 1% risk and initial-capital cash ceiling; no portfolio exit/release simulation",
              "entry_execution": "next available Kite 60minute provider bar open after completed confirmation bar",
              "config": raw_cfg, "data_hashes": {"daily_prefix_sha256": daily_hash,
                "hourly_prefix_sha256": hourly_hash},
              "coverage": {"trace_rows": len(trace), "state_transitions": len(events),
                "securities": len({r["security_id"] for r in trace}),
                "hourly_data_securities": len(hourly),
                "excluded_tatamotors_by_identity_manifest": True},
              "state_observation_counts": {state: sum(row["state"] == state for row in trace)
                               for state in sorted({row["state"] for row in trace})},
              "transition_counts": {state: sum(event.get("to_state") == state for event in events)
                                    for state in sorted({str(event.get("to_state")) for event in events})},
              "setup_lifecycle": {"total_setup_instances": len(setup_instances),
                "terminal_event_counts": terminal_counts,
                "entry_ready_breakdowns": entry_ready_breakdowns,
                "later_setup_instances_after_terminal": later_setup_counts,
                "unique_securities_with_multiple_setups": sum(1 for sid in daily
                    if len({r.get("setup_instance_id") for r in trace if r.get("security_id") == sid and r.get("setup_instance_id")}) > 1),
                "mid_range_not_actionable_observations": sum(r.get("state") == "MID_RANGE_NOT_ACTIONABLE" for r in trace),
                "first_full_context_date": min(full_context_dates) if full_context_dates else None},
              "old_vs_new": {"old_transition_counts": old_report.get("old_vs_new", {}).get(
                    "old_transition_counts", old_report.get("transition_counts", {})),
                "old_room_instances_by_previous_target_reference": old_report.get("old_vs_new", {}).get(
                    "old_room_instances_by_previous_target_reference", old_room_target_counts),
                "old_room_instances_affected_by_revised_target_policy": {
                    "EMA_NOW_FRICTION_ONLY": old_report.get("old_vs_new", {}).get(
                        "old_room_instances_by_previous_target_reference", old_room_target_counts).get("EMA_HARD_TARGET", 0),
                    "HOURLY_TARGET_REQUIRES_MEANINGFUL_STRENGTH": old_report.get("old_vs_new", {}).get(
                        "old_room_instances_by_previous_target_reference", old_room_target_counts).get("HOURLY_RESISTANCE_HARD_TARGET", 0),
                    "NO_TARGET_NOW_USES_FIXED_2R_FALLBACK": old_report.get("old_vs_new", {}).get(
                        "old_room_instances_by_previous_target_reference", old_room_target_counts).get("NO_STRUCTURAL_TARGET", 0)},
                "same_signal_date_ready_crosswalk_count": sum(changed_old_room.values()) if not old_report.get("old_vs_new") else
                    old_report.get("old_vs_new", {}).get("same_signal_date_ready_crosswalk_count", 0),
                "same_signal_date_crosswalk_note": "Signals were re-screened and many setup dates changed; the prior target-reference counts identify old rejection decisions exposed to revised rules, not a claim each became an entry.",
                "new_transition_counts": {state: sum(event.get("to_state") == state for event in events)
                    for state in sorted(terminal_states)}},
              "known_hourly_exceptions": {"HDFC": "NO_CURRENT_TOKEN", "LTIM": "FAIL_CLOSED_IDENTITY_BOUNDARY",
                  "TATAMOTORS": "HOURLY_IDENTITY_PRICE_MISMATCH"},
              "known_quarantined_hourly_candles": 3,
              "audit_samples": {group: sum(row["sample_group"] == group for row in samples)
                for group in ("ENTRY_READY", "ROOM_OR_INVALIDATED", "MID_RANGE", "LATER_ACTIONABLE_AFTER_TERMINAL")},
              "files": {"trace": str(trace_path), "transitions": str(events_path),
                        "sample": str(sample_path)}}
    output.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"state_observation_counts": report["state_observation_counts"],
                      "transition_counts": report["transition_counts"], "audit_samples": report["audit_samples"],
                      "trace": str(trace_path), "transitions": str(events_path),
                      "sample": str(sample_path), "report": str(report_path)}, indent=2))
    return {"report": report, "trace": trace, "events": events,
            "trace_path": trace_path, "events_path": events_path,
            "sample_path": sample_path, "report_path": report_path}


def _date(value: str) -> date:
    return date.fromisoformat(str(value)[:10])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, default=START)
    parser.add_argument("--end", type=date.fromisoformat, default=END)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--daily-manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--hourly-manifest", type=Path, default=None)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--max-days", type=int, default=None, help="small smoke replay only")
    args = parser.parse_args()
    try:
        replay(start=args.start, end=args.end, config_path=args.config,
               output_dir=args.output_directory, data_root=args.data_root,
               daily_manifest=args.daily_manifest, hourly_manifest=args.hourly_manifest,
               benchmark_path=args.benchmark, max_days=args.max_days)
    except (ValueError, OSError, BacktestDataError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
