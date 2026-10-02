"""Native, PIT-gated V3.1 daily SOD replay for the frozen 2023-2024 period.

V2 is used only for stateless causal structure measurements (zone construction,
daily location and volume features). V3.1 owns candidate qualification,
watch/event lifecycle, H1 confirmation and ENTRY_READY decisions. No V2 signal
artifact is read.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import fields, is_dataclass
from datetime import date, datetime
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

import yaml

from src.backtest.data import load_daily_dataset_through, load_hourly_dataset_through
from src.backtest.top_down_swing_v2_replay import _cfg as _v2_config
from src.backtest.weekly_selection_run import load_nifty_benchmark
from src.configuration import PROJECT_ROOT
from src.data.hourly import HOURLY_MANIFEST
from src.data.market_data import DATA_ROOT, MANIFEST_PATH
from src.data.universe.provider import PIT_TRADABLE, UniverseProvider
from src.strategy.structural_regime_v3 import classify_regime
from src.strategy.top_down_swing import volume_features
from src.strategy.top_down_swing_v2 import (adaptive_recent_structure, daily_location,
    enriched_zones, structural_target, zone_row)
from src.strategy.sr_zone_v2 import Zone, TouchEpisode
from src.strategy.top_down_swing_v2 import V2Zone
from src.strategy.top_down_swing_v3 import V3Config, family_allowed, trendline_pullback_candidate
from src.strategy.weekly_selection import completed_weekly_rows

START = date(2023, 1, 1)
END = date(2024, 12, 31)
CONFIG = PROJECT_ROOT / "config" / "top_down_swing_v3_1.yaml"
OUTPUT = PROJECT_ROOT / "data" / "processed" / "backtests" / "top_down_swing_v3_1"
BENCHMARK = PROJECT_ROOT / "data" / "raw" / "reference" / "nifty50_daily.csv"
CHECKPOINT_INTERVAL_DAYS = 5
CHECKPOINT_VERSION = 1


def _write(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r)) or ["empty"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, sort_keys=True, ensure_ascii=True, default=str)
                             if isinstance(v, (dict, list, tuple)) else v for k, v in row.items()})


def _iso_day(row: Mapping[str, Any]) -> date:
    return date.fromisoformat(str(row["date"])[:10])


def _date_value(value: Any) -> date:
    return value if isinstance(value, date) else date.fromisoformat(str(value))


_CHECKPOINT_CLASSES = {cls.__name__: cls for cls in (Zone, V2Zone, TouchEpisode)}


def _pack(value: Any) -> Any:
    """Encode replay state as safe, typed JSON (no executable pickle payload)."""
    if isinstance(value, datetime):
        return {"__kind__": "datetime", "value": value.isoformat()}
    if isinstance(value, date):
        return {"__kind__": "date", "value": value.isoformat()}
    if is_dataclass(value):
        return {"__kind__": "dataclass", "type": type(value).__name__,
                "fields": {f.name: _pack(getattr(value, f.name)) for f in fields(value)}}
    if isinstance(value, dict):
        return {"__kind__": "dict", "items": [[_pack(k), _pack(v)] for k, v in value.items()]}
    if isinstance(value, tuple):
        return {"__kind__": "tuple", "items": [_pack(v) for v in value]}
    if isinstance(value, set):
        return {"__kind__": "set", "items": [_pack(v) for v in sorted(value, key=str)]}
    if isinstance(value, list):
        return [_pack(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"Unsupported V3.1 checkpoint value: {type(value).__name__}")


def _unpack(value: Any) -> Any:
    if isinstance(value, list):
        return [_unpack(v) for v in value]
    if not isinstance(value, dict) or "__kind__" not in value:
        return {k: _unpack(v) for k, v in value.items()} if isinstance(value, dict) else value
    kind = value["__kind__"]
    if kind == "date": return date.fromisoformat(value["value"])
    if kind == "datetime": return datetime.fromisoformat(value["value"])
    if kind == "tuple": return tuple(_unpack(v) for v in value["items"])
    if kind == "set": return set(_unpack(v) for v in value["items"])
    if kind == "dict": return {_unpack(k): _unpack(v) for k, v in value["items"]}
    if kind == "dataclass":
        cls = _CHECKPOINT_CLASSES.get(value["type"])
        if cls is None:
            raise ValueError(f"Unknown checkpoint dataclass: {value['type']}")
        return cls(**{k: _unpack(v) for k, v in value["fields"].items()})
    raise ValueError(f"Unknown checkpoint value kind: {kind}")


def _atomic_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(_pack(dict(payload)), handle, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def _input_fingerprint(config_path: Path, daily: Mapping[str, Sequence[Mapping[str, Any]]],
                       hourly: Mapping[str, Sequence[Mapping[str, Any]]],
                       manifests: Sequence[Path]) -> str:
    digest = hashlib.sha256(config_path.read_bytes())
    for path in manifests:
        digest.update(str(path.resolve()).encode("utf-8"))
        digest.update(path.read_bytes() if path.is_file() else b"MISSING")
    for label, dataset in ((b"daily", daily), (b"hourly", hourly)):
        digest.update(label)
        for sid in sorted(dataset):
            rows = dataset[sid]
            digest.update(f"{sid}|{len(rows)}|{_iso_day(rows[0]) if rows else ''}|{_iso_day(rows[-1]) if rows else ''}\n".encode())
    return digest.hexdigest()


def _atr(rows: Sequence[Mapping[str, Any]], n: int = 14) -> float:
    tail = rows[-n:]
    return max(sum(float(x["high"]) - float(x["low"]) for x in tail) / len(tail), 1e-9) if tail else 1e-9


def _candidates(location: Mapping[str, Any], cfg: Any) -> list[tuple[str, Any]]:
    """Map current causal structural context to V3 candidate families."""
    out: list[tuple[str, Any]] = []
    state = str(location.get("state") or "")
    support = location.get("support") or location.get("broken_zone")
    resistance = location.get("resistance") or location.get("nearest_resistance")
    if state in {"NEAR_RECENT_DAILY_SUPPORT", "INSIDE_RECENT_DAILY_SUPPORT"} and support:
        out.append(("SUPPORT_REACTION", support))
    if state in {"RETESTING_BROKEN_RESISTANCE", "NEAR_DAILY_RESISTANCE_BREAKOUT_WATCH",
                 "BREAKOUT_ABOVE_RECENT_DAILY_RESISTANCE"} and resistance:
        family = ("CONSOLIDATION_BREAKOUT" if int(location.get("resistance_hovering_bars", 0)) >=
                  cfg.daily_zone.reaction_horizon else "BREAKOUT_RETEST")
        out.append((family, resistance))
    if state in {"BROKEN_SUPPORT_RECLAIM_WATCH", "DAILY_SUPPORT_BROKEN", "WAIT_FOR_RECLAIM"} and support:
        out.append(("RECLAIM", support))
    for zone in location.get("direct_breakout_candidates", ()):
        out.append(("CONSOLIDATION_BREAKOUT", zone))
    unique: dict[tuple[str, str], tuple[str, Any]] = {}
    for family, zone in out:
        unique[(family, zone.zone_id)] = (family, zone)
    return [unique[k] for k in sorted(unique)]


def _volume_pass(features: Mapping[str, Any], family: str) -> bool:
    expanded = features.get("same_time_expansion") is True or features.get("expansion_vs_prior_mean") is True
    contracted = features.get("same_time_contraction") is True
    return expanded if family in {"BREAKOUT_RETEST", "CONSOLIDATION_BREAKOUT"} else (expanded or contracted)


def _h1_confirmation(family: str, zone: Any, prefix: Sequence[Mapping[str, Any]],
                     daily_atr: float, volume_lookback: int) -> tuple[bool, str, dict[str, Any]]:
    """V3-owned, causal H1 setup checks; the returned prefix ends on the signal bar."""
    if len(prefix) < 3:
        return False, "INSUFFICIENT_H1_CONTEXT", {}
    bar = prefix[-1]
    prior = prefix[-4:-1]
    o, h, lo, c = (float(bar[k]) for k in ("open", "high", "low", "close"))
    bullish = c > o
    volume = volume_features(prefix, lookback=volume_lookback)
    expansion = volume.get("same_time_expansion") is True or volume.get("expansion_vs_prior_mean") is True
    if family == "SUPPORT_REACTION":
        touched = any(float(x["low"]) <= zone.upper_bound + daily_atr for x in prefix[-4:])
        rejected = lo <= zone.upper_bound + daily_atr and c >= zone.center and c > max(float(x["high"]) for x in prior)
        passed = touched and rejected and bullish and _volume_pass(volume, family)
        return passed, "SUPPORT_H1_BUYER_CONTROL" if passed else "SUPPORT_H1_CONFIRMATION_MISSING", {
            "zone_touched": touched, "bullish_rejection": rejected, "breaks_local_high": c > max(float(x["high"]) for x in prior),
            "volume_features": volume, "volume_passed": _volume_pass(volume, family)}
    if family == "BREAKOUT_RETEST":
        threshold = zone.upper_bound + 0.1 * daily_atr
        accepted_index = None
        for j in range(1, len(prefix)):
            first, second = prefix[j-1], prefix[j]
            first_ok = float(first["close"]) > threshold
            second_ok = float(second["close"]) > threshold
            first_vol = volume_features(prefix[:j], lookback=volume_lookback)
            first_expansion = first_vol.get("same_time_expansion") is True or first_vol.get("expansion_vs_prior_mean") is True
            if first_ok and second_ok and first_expansion:
                accepted_index = j
        close_above = c > threshold
        role_flip = accepted_index is not None
        touched = bool(role_flip and any(float(x["low"]) <= zone.upper_bound
            for x in prefix[accepted_index+1:-1]))
        response = bullish and c > max(float(x["high"]) for x in prior) and c >= zone.center
        passed = close_above and role_flip and touched and response and expansion
        return passed, "BREAKOUT_ROLE_FLIP_CONFIRMED" if passed else "BREAKOUT_ROLE_FLIP_INCOMPLETE", {
            "breakout_close_above_zone": close_above, "prior_acceptance_above_zone": role_flip,
            "original_breakout_index": accepted_index,
            "retest_touched_from_above": touched, "buyer_control_reclaim": response,
            "volume_features": volume, "original_breakout_volume_confirmation": expansion}
    if family == "CONSOLIDATION_BREAKOUT":
        n = 6
        base = prefix[-n-1:-1]
        before = prefix[-2*n-1:-n-1]
        compression = len(before) == n and (max(float(x["high"]) for x in base)-min(float(x["low"]) for x in base) <
                      max(float(x["high"]) for x in before)-min(float(x["low"]) for x in before))
        boundary = max(float(x["high"]) for x in base) if base else zone.upper_bound
        passed = compression and bullish and c > max(boundary, zone.upper_bound) and expansion
        return passed, "CAUSAL_CONSOLIDATION_BREAKOUT" if passed else "CONSOLIDATION_BREAKOUT_INCOMPLETE", {
            "compression": compression, "boundary": boundary, "volume_features": volume, "volume_passed": expansion}
    if family in {"RECLAIM", "FAILED_BREAKDOWN_RECLAIM"}:
        failed = any(float(x["low"]) < zone.lower_bound for x in prefix[-4:-1])
        reclaimed = bullish and float(prefix[-2]["close"]) < zone.lower_bound and c > zone.upper_bound
        passed = failed and reclaimed and (expansion or volume.get("same_time_contraction") is True)
        return passed, "FAILED_BREAKDOWN_RECLAIM" if passed else "RECLAIM_INCOMPLETE", {
            "failed_downside_attempt": failed, "reclaimed_zone": reclaimed,
            "volume_features": volume, "volume_passed": _volume_pass(volume, family)}
    return False, "UNSUPPORTED_FAMILY", {}


def _hysteresis(state: dict[str, Any], raw: Mapping[str, Any], day: date,
                required: int = 2) -> dict[str, Any]:
    candidate = str(raw.get("candidate_regime") or raw["regime"])
    previous = str(state.get("regime") or "")
    next_state = dict(raw)
    reasons = set(raw.get("reason_codes") or ())
    evidence_valid = ((candidate == "TREND_UP" and "HIGHER_HIGH_AND_HIGHER_LOW" in reasons and
                       "ESTABLISHED_RISING_SUPPORT" in reasons) or
                      (candidate == "TREND_DOWN" and "LOWER_HIGH_AND_LOWER_LOW" in reasons and
                       "ESTABLISHED_FALLING_RESISTANCE" in reasons) or
                      (candidate == "RANGE" and "REPEATED_HORIZONTAL_SWINGS_AND_RANGE_WIDTH" in reasons))
    if candidate not in {"TREND_UP", "TREND_DOWN", "RANGE"} or not evidence_valid:
        count = 0
    else:
        count = int(state.get("candidate_sessions", 0)) + 1 if state.get("candidate_regime") == candidate else 1
    if previous in {"TREND_UP", "TREND_DOWN", "RANGE"} and candidate != previous:
        next_state["regime"] = "TRANSITION" if count < required else candidate
    elif previous == "TRANSITION" and candidate in {"TREND_UP", "TREND_DOWN", "RANGE"}:
        next_state["regime"] = candidate if count >= required else "TRANSITION"
    else:
        next_state["regime"] = previous if previous and candidate == "AMBIGUOUS" else candidate
    if next_state["regime"] != previous:
        next_state["regime_start_date"] = day.isoformat()
    else:
        next_state["regime_start_date"] = state.get("regime_start_date", day.isoformat())
    next_state.update({"previous_regime": previous or None, "candidate_regime": candidate,
                       "candidate_sessions": count if candidate == next_state["regime"] else count,
                       "candidate_evidence_valid": evidence_valid,
                       "decision_date": day.isoformat()})
    return next_state


def replay(*, config_path: Path = CONFIG, output_dir: Path = OUTPUT,
           data_root: Path = DATA_ROOT, daily_manifest: Path = MANIFEST_PATH,
           hourly_manifest: Path = HOURLY_MANIFEST, benchmark_path: Path = BENCHMARK,
           daily: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
           hourly: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
           resume: bool = False, _stop_after_days: int | None = None) -> dict[str, Any]:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if raw.get("version") != "top_down_swing_v3_1":
        raise ValueError("V3.1 configuration version mismatch")
    period = raw["research_period"]
    if _date_value(period["start"]) != START or _date_value(period["end"]) != END:
        raise ValueError("V3.1 is restricted to the frozen 2023-2024 period")
    if daily is None:
        daily, _ = load_daily_dataset_through(universe_id=PIT_TRADABLE, start_date=_date_value(period["daily_warmup_start"]),
            end_date=END, data_root=data_root, manifest_path=daily_manifest)
    if hourly is None:
        hourly, _ = load_hourly_dataset_through(universe_id=PIT_TRADABLE, start_date=_date_value(period["hourly_warmup_start"]),
            end_date=END, data_root=data_root, manifest_path=hourly_manifest, daily_manifest_path=daily_manifest)
    benchmark = load_nifty_benchmark(benchmark_path, start=_date_value(period["daily_warmup_start"]), end=END)
    fingerprint = _input_fingerprint(config_path, daily, hourly, (daily_manifest, hourly_manifest))
    provider = UniverseProvider()
    v2raw = yaml.safe_load((PROJECT_ROOT / "config" / "top_down_swing_v2.yaml").read_text(encoding="utf-8")) or {}
    cfg = _v2_config(v2raw)
    vc = V3Config(**{k: float(v) if k != "volume_lookback" else int(v) for k, v in raw["setups"].items()
                     if k in V3Config.__dataclass_fields__})
    days = sorted({d for rows in daily.values() for r in rows if START <= (d := _iso_day(r)) <= END})
    d_index = {sid: {_iso_day(r): i for i, r in enumerate(rows)} for sid, rows in daily.items()}
    h_by_day: dict[str, dict[date, list[tuple[int, Mapping[str, Any]]]]] = {}
    for sid, rows in hourly.items():
        grouped: dict[date, list[tuple[int, Mapping[str, Any]]]] = defaultdict(list)
        for i, row in enumerate(rows):
            grouped[_iso_day(row)].append((i, row))
        h_by_day[sid] = grouped
    regime_rows: list[dict[str, Any]] = []
    signal_rows: list[dict[str, Any]] = []
    watch_rows: list[dict[str, Any]] = []
    reject_rows: list[dict[str, Any]] = []
    funnel: Counter[str] = Counter()
    prior_regime: dict[str, dict[str, Any]] = {}
    active: dict[tuple[str, str, str], dict[str, Any]] = {}
    emitted_ids: set[str] = set()
    events_by_day: dict[date, list[dict[str, Any]]] = defaultdict(list)
    processed: set[tuple[str, date]] = set()
    membership_starts = {m.security_id: m.effective_from for m in provider.members_over(START, END, PIT_TRADABLE)}
    trendline_consumed: set[tuple[str, str]] = set()
    checkpoint_path = output_dir / "checkpoints" / "v3_1_replay_state.json"
    last_completed: date | None = None
    elapsed_before = 0.0
    if resume:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"No V3.1 replay checkpoint exists at {checkpoint_path}")
        saved = _unpack(json.loads(checkpoint_path.read_text(encoding="utf-8")))
        if saved.get("checkpoint_version") != CHECKPOINT_VERSION or saved.get("input_fingerprint") != fingerprint:
            raise ValueError("V3.1 checkpoint is incompatible with current config/data inputs")
        last_completed = date.fromisoformat(saved["last_completed_date"])
        regime_rows = saved["regime_rows"]; signal_rows = saved["signal_rows"]
        watch_rows = saved["watch_rows"]; reject_rows = saved["reject_rows"]
        funnel = Counter(saved["funnel"]); prior_regime = saved["regime_state_by_security"]
        active = saved["active_watch_lifecycle_state"]; emitted_ids = saved["emitted_event_ids"]
        processed = saved["processed_security_dates"]
        trendline_consumed = saved["trendline_consumed"]
        elapsed_before = float(saved.get("elapsed_seconds", 0.0))
        if last_completed not in days:
            raise ValueError("Checkpoint date is not present in the current replay calendar")
    run_started = time.monotonic()
    remaining_days = [day for day in days if last_completed is None or day > last_completed]

    def save_checkpoint(day: date) -> None:
        elapsed = elapsed_before + time.monotonic() - run_started
        portfolio_cfg = raw.get("portfolio", {})
        initial_capital = float(portfolio_cfg.get("initial_capital", 0.0))
        _atomic_checkpoint(checkpoint_path, {
            "checkpoint_version": CHECKPOINT_VERSION,
            "input_fingerprint": fingerprint,
            "last_completed_date": day.isoformat(),
            "regime_state_by_security": prior_regime,
            "active_watch_lifecycle_state": active,
            "open_portfolio_positions": {},
            "portfolio_cash": initial_capital,
            "portfolio_equity": initial_capital,
            "portfolio_state_status": "PORTFOLIO_STAGE_NOT_RUN_DURING_SIGNAL_REPLAY",
            "emitted_event_ids": emitted_ids,
            "regime_rows": regime_rows, "signal_rows": signal_rows,
            "watch_rows": watch_rows, "reject_rows": reject_rows,
            "funnel": dict(funnel), "processed_security_dates": processed,
            "trendline_consumed": trendline_consumed,
            "elapsed_seconds": elapsed,
        })

    for day_number, day in enumerate(remaining_days, start=1):
        members = {m.security_id: m for m in provider.members_on(day, PIT_TRADABLE)}
        for sid, member in sorted(members.items()):
            if sid not in d_index or day not in d_index[sid]:
                continue
            processed.add((sid, day)); funnel["PIT_ELIGIBLE_SECURITIES"] += 1
            idx = d_index[sid][day]
            prefix = list(daily[sid][:idx])
            if not prefix:
                continue
            # Only this eligible SOD date is classified. Earlier candles are
            # warmup inputs, not output regime rows or decisions.
            regime = classify_regime(prefix, as_of=day.isoformat(), previous_state=prior_regime.get(sid))
            regime = _hysteresis(prior_regime.get(sid, {}), regime, day,
                                  int(raw["regime"].get("candidate_confirmation_sessions", 2)))
            prior_regime[sid] = regime
            regrow = {"date": day.isoformat(), "security_id": sid, "symbol": member.symbol, **regime}
            regime_rows.append(regrow); funnel["REGIME_VALID"] += int(regime["regime"] != "AMBIGUOUS")
            em = regime.get("ema", {})
            regrow.update({"ema50": em.get("ema50", {}).get("value"), "ema200": em.get("ema200", {}).get("value"),
                "ema50_slope": em.get("ema50", {}).get("normalized_slope"), "ema200_slope": em.get("ema200", {}).get("normalized_slope"),
                "ema50_price_state": em.get("ema50", {}).get("price_vs_ema"), "ema200_price_state": em.get("ema200", {}).get("price_vs_ema"),
                "ema_structural_state": em.get("structural_state")})
            adaptive = adaptive_recent_structure(daily[sid], sod_date=day.isoformat(), security_id=sid, cfg=cfg)
            zones, context = adaptive["zones"], adaptive["rows"]
            if not context:
                continue
            loc = daily_location(context, zones, cfg, structure_rows=adaptive["structure_rows"])
            current_close = float(context[-1]["close"])
            current_atr = float(loc.get("daily_atr") or _atr(context))
            for old_key, old_watch in list(active.items()):
                if old_key[0] != sid or old_watch.get("status") != "CONSUMED":
                    continue
                old_zone = old_watch["zone"]
                if old_watch["family"] in {"SUPPORT_REACTION", "RECLAIM", "FAILED_BREAKDOWN_RECLAIM"}:
                    moved_away = current_close > old_zone.upper_bound + current_atr
                else:
                    moved_away = current_close < old_zone.lower_bound - current_atr
                old_watch["revisit_armed"] = bool(old_watch.get("revisit_armed") or moved_away)
            weeks = []
            if len(prefix) >= 5:
                from src.strategy.weekly_selection import completed_weekly_rows
                weeks = completed_weekly_rows(prefix, day)
            weekly_zones = enriched_zones(weeks, sid, cfg.weekly_zone, cfg) if weeks else ()
            # V3-native inclined-geometry watch. Pre-membership H1 candles are
            # excluded; daily bars remain available as explicitly permitted
            # indicator/structure warmup.
            line = regime.get("trendlines", {}).get("rising_support")
            if regime.get("regime") == "TREND_UP" and line and line.get("status") in {"ESTABLISHED", "STRONG"}:
                tkey = (sid, str(line.get("trendline_id")))
                hrows = [r for r in hourly.get(sid, ()) if _iso_day(r) >= membership_starts.get(sid, START)]
                eligible_h_indices = [i for i, r in enumerate(hrows) if _iso_day(r) == day]
                target_candidates = []
                for z in zones:
                    if z.side == "resistance" and z.lower_bound > float(prefix[-1]["close"]) and z.strength in {"MAJOR", "MEANINGFUL"}:
                        target_candidates.append({"price": z.lower_bound, "timeframe": "DAILY",
                            "classification": "DOMINANT" if z.strength == "MAJOR" else "SECONDARY",
                            "target_id": z.zone_id})
                if not any(t.get("classification") == "DOMINANT" for t in target_candidates) and target_candidates:
                    strongest = max(target_candidates, key=lambda t: next((z.dominance_score for z in zones if z.zone_id == t["target_id"]), 0))
                    strongest["classification"] = "DOMINANT"
                frictions = [{"zone_id": z.zone_id, "lower_bound": z.lower_bound, "upper_bound": z.upper_bound,
                              "dominance_score": z.dominance_score} for z in zones if z.side == "resistance"]
                if tkey not in trendline_consumed:
                    for i in eligible_h_indices:
                        if i + 1 >= len(hrows):
                            continue
                        candidate = trendline_pullback_candidate(daily_prefix=prefix, hourly_prefix=hrows[:i+1],
                            entry_bar=hrows[i+1], target_candidates=target_candidates, config=vc,
                            decision_date=day.isoformat(), regime_snapshot=regime, friction_candidates=frictions)
                        if candidate.get("status") != "ENTRY_READY":
                            continue
                        event_id = f"{sid}|TRENDLINE_PULLBACK|{line['trendline_id']}|{day.isoformat()}"
                        sig = {"date": day.isoformat(), "setup_signal_date": day.isoformat(), "security_id": sid,
                            "symbol": member.symbol, "state": "ENTRY_READY", "strategy_version": "top_down_swing_v3_1",
                            "setup_family": "TRENDLINE_PULLBACK", "setup_type": "TRENDLINE_PULLBACK",
                            "setup_event_id": event_id, "setup_instance_id": event_id + "|1",
                            "confirmation_timestamp": candidate["confirmation_timestamp"],
                            "executable_timestamp": candidate["entry_timestamp"], "regime": regime["regime"],
                            "regime_start_date": regime.get("regime_start_date"), "regime_evidence": regime.get("reason_codes"),
                            "primary_geometry": "INCLINED", "secondary_geometry": "HORIZONTAL",
                            "trendline": candidate["trendline"], "trendline_id": line["trendline_id"],
                            "ema50_state": candidate["ema50_state"], "ema200_state": candidate["ema200_state"],
                            "h1_confirmation_type": candidate["h1_confirmation_type"],
                            "h1_buyer_control": candidate.get("h1_buyer_control"),
                            "volume_state": candidate["volume_state"],
                            "h1_evidence": {"volume_features": candidate["volume_features"],
                                            "buyer_control": candidate.get("h1_buyer_control")},
                            "setup_lifecycle": ["WATCHING", "CONFIRMED", "ENTRY_READY", "CONSUMED"],
                            "plan": {"entry": candidate["entry"], "stop": candidate["stop"],
                                "target": candidate["target"], "target_timeframe": candidate["target_type"],
                                "planned_reward_r": candidate["planned_rr"], "volume_confirmation_passed": True}}
                        if sig["setup_instance_id"] in emitted_ids:
                            raise ValueError("Duplicate V3.1 trendline event id")
                        emitted_ids.add(sig["setup_instance_id"]); signal_rows.append(sig); events_by_day[day].append(sig)
                        watch_rows.append({"date": day.isoformat(), "security_id": sid,
                            "setup_family": "TRENDLINE_PULLBACK", "setup_event_id": event_id,
                            "state": "CONSUMED", "trendline_id": line["trendline_id"]})
                        trendline_consumed.add(tkey); funnel["H1_CONFIRMATION"] += 1; funnel["ENTRY_READY"] += 1
                        break
            # Carry only explicit unconsumed watches, expiring by config.
            for family, zone in _candidates(loc, cfg):
                key = (sid, family, zone.zone_id)
                old_watch = active.get(key)
                if old_watch and old_watch.get("status") == "CONSUMED":
                    if not old_watch.get("revisit_armed"):
                        continue
                    old_watch["status"] = "RESET"
                    old_watch["history"].append("RESET_AFTER_SEPARATED_REVISIT")
                    active.pop(key, None)
                if key not in active:
                    event_id = f"{sid}|{family}|{zone.zone_id}|{day.isoformat()}"
                    active[key] = {"security_id": sid, "symbol": member.symbol, "family": family, "zone": zone,
                        "event_id": event_id, "instance_id": event_id + "|1", "created": day,
                        "last_h1": -1, "status": "WATCHING", "history": ["WATCHING"],
                        "activation_atr": float(loc.get("daily_atr") or _atr(context)), "activation_location": loc}
                    watch_rows.append({"date": day.isoformat(), "security_id": sid, "setup_family": family,
                                       "setup_event_id": event_id, "state": "WATCHING", "zone_id": zone.zone_id})
                    funnel["WATCH_CREATED"] += 1
            for key, watch in list(active.items()):
                if key[0] != sid or watch["status"] != "WATCHING":
                    continue
                age = len([x for x in d_index[sid] if watch["created"] <= x <= day]) - 1
                if age > int(raw["setups"].get("confirmation_expiry_sessions", 10)):
                    watch["status"] = "INVALIDATED"; watch["history"].append("INVALIDATED")
                    reject_rows.append({"date": day.isoformat(), "security_id": sid, "setup_event_id": watch["event_id"],
                                        "setup_family": watch["family"], "reason_code": "WATCH_EXPIRED"})
                    continue
                bars = hourly.get(sid, ())
                current = h_by_day.get(sid, {}).get(day, ())
                if not current:
                    continue
                last = current[-1][0]
                begin = max(watch["last_h1"] + 1, current[0][0])
                for i in range(begin, last):  # reserve next H1 open for execution
                    hprefix = list(bars[:i+1]); hbar = hprefix[-1]
                    passed, reason, evidence = _h1_confirmation(watch["family"], watch["zone"], hprefix,
                        watch["activation_atr"], int(raw["setups"].get("volume_lookback", 20)))
                    if not passed:
                        continue
                    allowed, regime_reason = family_allowed(regime["regime"], watch["family"],
                        reclaim_confirmed=watch["family"] in {"RECLAIM", "FAILED_BREAKDOWN_RECLAIM"},
                        unusually_strong=bool(evidence.get("volume_passed") or evidence.get("original_breakout_volume_confirmation")))
                    if not allowed:
                        reject_rows.append({"date": day.isoformat(), "security_id": sid, "setup_event_id": watch["event_id"],
                            "setup_family": watch["family"], "reason_code": regime_reason})
                        continue
                    entry_bar = bars[i+1]; entry = float(entry_bar["open"])
                    stop = float(watch["zone"].lower_bound) - watch["activation_atr"] * float(raw["setups"].get("stop_buffer_atr", .25))
                    if stop >= entry:
                        continue
                    target = structural_target(entry, stop, zones, weekly_zones, watch["activation_atr"], cfg)
                    if target.get("status") != "READY" or float(target.get("planned_structural_rr", 0)) < 1.5:
                        reject_rows.append({"date": day.isoformat(), "security_id": sid, "setup_event_id": watch["event_id"],
                            "setup_family": watch["family"], "reason_code": target.get("reason_code", "STRUCTURAL_RR_BELOW_MINIMUM")})
                        continue
                    sig = {"date": day.isoformat(), "setup_signal_date": day.isoformat(), "security_id": sid,
                        "symbol": member.symbol, "state": "ENTRY_READY", "strategy_version": "top_down_swing_v3_1",
                        "setup_family": watch["family"], "setup_type": watch["family"],
                        "setup_event_id": watch["event_id"], "setup_instance_id": watch["instance_id"],
                        "confirmation_timestamp": str(hbar["date"]), "executable_timestamp": str(entry_bar["date"]),
                        "regime": regime["regime"], "regime_start_date": regime.get("regime_start_date"),
                        "regime_evidence": regime.get("reason_codes"), "primary_geometry": regime.get("primary_geometry"),
                        "secondary_geometry": regime.get("secondary_geometry"), "horizontal_zone_id": watch["zone"].zone_id,
                        "ema50_state": em.get("ema50", {}).get("state"), "ema200_state": em.get("ema200", {}).get("state"),
                        "h1_confirmation_type": reason, "h1_evidence": evidence,
                        "setup_lifecycle": ["WATCHING", "CONFIRMED", "ENTRY_READY", "CONSUMED"],
                        "plan": {"entry": entry, "stop": stop, "target": target["target"],
                                 "target_timeframe": target["target_timeframe"],
                                 "planned_reward_r": target["planned_structural_rr"],
                                 "target_zone_id": target["target_zone_id"],
                                 "volume_confirmation_passed": True}}
                    if sig["setup_instance_id"] in emitted_ids:
                        raise ValueError("Duplicate V3.1 setup instance id")
                    emitted_ids.add(sig["setup_instance_id"]); signal_rows.append(sig); events_by_day[day].append(sig)
                    watch["status"] = "CONSUMED"; watch["history"].append("CONSUMED")
                    funnel["H1_CONFIRMATION"] += 1; funnel["ENTRY_READY"] += 1
                    break
                watch["last_h1"] = max(watch["last_h1"], last - 1)
        # Only checkpoint at completed daily boundaries. The current-day H1
        # scan is complete before the boundary is persisted.
        if day_number % CHECKPOINT_INTERVAL_DAYS == 0 or day == remaining_days[-1]:
            save_checkpoint(day)
            elapsed = elapsed_before + time.monotonic() - run_started
            open_position_count = 0  # This module emits signals; portfolio simulation is a later stage.
            print(f"V3.1 progress date={day.isoformat()} elapsed_seconds={elapsed:.1f} "
                  f"securities_processed={len(processed)} signals_generated={len(signal_rows)} "
                  f"open_positions={open_position_count} (portfolio stage pending)", flush=True)
        if _stop_after_days is not None and day_number >= _stop_after_days:
            return {"completed": False, "checkpoint": str(checkpoint_path),
                    "last_completed_date": day.isoformat(), "signals": signal_rows,
                    "regimes": regime_rows}
    exceptions = []
    for row in regime_rows:
        if not provider.is_security_eligible(row["security_id"], date.fromisoformat(row["date"]), PIT_TRADABLE):
            exceptions.append({"security_id": row["security_id"], "date": row["date"], "kind": "OUT_OF_PIT_REGIME"})
    for row in signal_rows:
        if not provider.is_security_eligible(row["security_id"], date.fromisoformat(row["setup_signal_date"]), PIT_TRADABLE):
            exceptions.append({"security_id": row["security_id"], "date": row["setup_signal_date"], "kind": "OUT_OF_PIT_SIGNAL"})
    _write(output_dir / "pit_exceptions.csv", exceptions)
    pit_summary = {"regime_rows": len(regime_rows), "signal_rows": len(signal_rows), "out_of_pit_regime_rows": sum(x["kind"] == "OUT_OF_PIT_REGIME" for x in exceptions),
                   "out_of_pit_signal_rows": sum(x["kind"] == "OUT_OF_PIT_SIGNAL" for x in exceptions),
                   "eligible_security_dates": len(processed), "warmup_only_before_start": True}
    (output_dir / "pit_audit_summary.json").write_text(json.dumps(pit_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if exceptions:
        raise RuntimeError("PIT audit found exceptions; V3.1 replay stopped")
    _write(output_dir / "v3_1_regime_daily.csv", regime_rows)
    _write(output_dir / "v3_1_entry_ready_events.csv", signal_rows)
    _write(output_dir / "v3_1_watches.csv", watch_rows)
    _write(output_dir / "v3_1_rejections.csv", reject_rows)
    regimes_by_sid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in regime_rows:
        regimes_by_sid[str(row["security_id"])].append(row)
    episode_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    regime_counts = Counter(row["regime"] for row in regime_rows)
    flip_counts: Counter[tuple[str, int]] = Counter()
    for sid, rows in regimes_by_sid.items():
        ordered = sorted(rows, key=lambda r: r["date"])
        start_i = 0
        for i in range(1, len(ordered)+1):
            if i == len(ordered) or ordered[i]["regime"] != ordered[start_i]["regime"]:
                first, last = ordered[start_i], ordered[i-1]
                length = i-start_i
                episode_rows.append({"security_id": sid, "regime": first["regime"],
                    "start_date": first["date"], "end_date": last["date"], "sessions": length,
                    "transition_episode_over_20": first["regime"] == "TRANSITION" and length > 20})
                if i < len(ordered):
                    nxt = ordered[i]
                    transition_rows.append({"security_id": sid, "date": nxt["date"],
                        "previous_regime": first["regime"], "new_regime": nxt["regime"],
                        "reason": nxt.get("regime_transition_reason") or nxt.get("reason_codes")})
                    flip_counts[(sid, int(str(nxt["date"])[:4]))] += 1
                start_i = i
    distribution_rows = [{"regime": name, "stock_days": regime_counts[name],
        "percent_stock_days": regime_counts[name]/len(regime_rows) if regime_rows else 0.0,
        "median_episode_sessions": median([x["sessions"] for x in episode_rows if x["regime"] == name])
            if any(x["regime"] == name for x in episode_rows) else 0,
        "one_day_episodes": sum(x["regime"] == name and x["sessions"] == 1 for x in episode_rows),
        "episodes_over_20_sessions": sum(x["regime"] == name and x["sessions"] > 20 for x in episode_rows)}
        for name in ("TREND_UP", "TREND_DOWN", "RANGE", "TRANSITION", "AMBIGUOUS")]
    _write(output_dir / "regime_distribution.csv", distribution_rows)
    _write(output_dir / "regime_episode_summary.csv", episode_rows)
    _write(output_dir / "regime_transition_log.csv", transition_rows)
    ema_rows = []
    for period in (50, 200):
        key = f"ema{period}"
        values = [float(r[key]) for r in regime_rows if r.get(key) is not None]
        slopes = [float(r[f"ema{period}_slope"]) for r in regime_rows if r.get(f"ema{period}_slope") is not None]
        ema_rows.append({"ema": key.upper(), "valid_count": len(values),
            "valid_percent": len(values)/len(regime_rows) if regime_rows else 0,
            "mean_value": sum(values)/len(values) if values else None,
            "mean_normalized_slope": sum(slopes)/len(slopes) if slopes else None,
            "median_normalized_slope": median(slopes) if slopes else None,
            "nonzero_slope_count": sum(abs(x) > 1e-12 for x in slopes),
            "unavailable_count": len(regime_rows)-len(values)})
    _write(output_dir / "ema_state_distribution.csv", ema_rows)
    summary = {"strategy_version": "top_down_swing_v3_1", "native_replay": True,
        "v2_signal_artifact_used": False, "research_period": {"start": START.isoformat(), "end": END.isoformat()},
        "2025_2026_loaded": False, "pit_audit": pit_summary, "signal_count": len(signal_rows),
        "signals_by_family": dict(Counter(x["setup_family"] for x in signal_rows)),
        "regime_distribution": distribution_rows, "regime_transition_count": len(transition_rows),
        "average_flips_per_security_year": sum(flip_counts.values())/len(flip_counts) if flip_counts else 0,
        "ema_validity": ema_rows,
        "funnel": dict(funnel), "output_directory": str(output_dir),
        "limitations": ["V3.1 setup qualification is native; stateless V2 structure and volume measurements are reused.",
                        "Hourly identity exclusions remain governed by the frozen data quality manifest."]}
    (output_dir / "v3_1_replay_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"completed": True, "checkpoint": str(checkpoint_path), "summary": summary,
            "signals": signal_rows, "regimes": regime_rows, "watches": watch_rows,
            "rejections": reject_rows, "benchmark": benchmark}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output-directory", type=Path, default=OUTPUT)
    parser.add_argument("--resume", action="store_true", help="Resume from the latest atomic daily checkpoint")
    args = parser.parse_args()
    result = replay(config_path=args.config, output_dir=args.output_directory, resume=args.resume)
    print(json.dumps(result["summary"], indent=2, default=str))


if __name__ == "__main__":
    main()
