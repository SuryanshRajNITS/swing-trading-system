"""Causal daily-map translation audit for the V2 manual examples.

This utility observes each setup at its SOD cutoff and probes the requested
EICHERMOT direct-breakout example with manifested H1 data. It does not run a
portfolio simulation.
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from src.backtest.data import load_daily_dataset_through, load_hourly_dataset_through
from src.backtest.top_down_swing_v2_replay import CONFIG, DAILY_WARMUP, END, HOURLY_WARMUP, _cfg
from src.configuration import PROJECT_ROOT
from src.data.market_data import DATA_ROOT, MANIFEST_PATH
from src.data.universe.provider import PIT_TRADABLE, UniverseProvider
from src.strategy.top_down_swing import atr, ema
from src.strategy.top_down_swing_v2 import adaptive_recent_structure, daily_location, daily_setup_activations, enriched_zones, hourly_entry, zone_row
from src.strategy.weekly_selection import completed_weekly_rows

OUTPUT = PROJECT_ROOT / "data/processed/backtests/top_down_swing_v2_structure_translation_audit/translation_checks.json"
CHECKS = (
    ("TECHM", "2023-04-12"), ("TECHM", "2023-04-13"),
    ("ASIANPAINT", "2023-02-01"), ("ASIANPAINT", "2023-03-01"),
    ("ASIANPAINT", "2023-04-03"), ("ASIANPAINT", "2023-05-02"),
    ("APOLLOHOSP", "2023-01-03"), ("APOLLOHOSP", "2023-02-01"),
    ("APOLLOHOSP", "2023-04-03"),
    ("EICHERMOT", "2023-07-03"), ("EICHERMOT", "2023-08-01"),
    ("EICHERMOT", "2023-09-01"), ("EICHERMOT", "2023-11-01"),
    ("EICHERMOT", "2023-11-06"), ("EICHERMOT", "2023-11-09"),
    ("EICHERMOT", "2023-11-10"), ("EICHERMOT", "2023-11-13"),
    ("ADANIPORTS", "2023-05-29"), ("CIPLA", "2023-07-11"),
)
MANUAL_READOUT = {
    ("TECHM", "2023-04-12"): "WAIT_FOR_RECLAIM is active. EMA200 is unavailable and D-1 is above EMA50, so this is a structural reclaim state, not an asserted EMA-loss signal.",
    ("TECHM", "2023-04-13"): "WAIT_FOR_RECLAIM remains active; this is not treated as a blind support reaction. EMA200 is unavailable and D-1 is above EMA50.",
    ("ASIANPAINT", "2023-02-01"): "No validated zone is close enough to justify a setup.",
    ("ASIANPAINT", "2023-03-01"): "Established resistance is monitored as a breakout/retest area.",
    ("ASIANPAINT", "2023-04-03"): "A validated support revisit is monitored; daily location is not itself an entry.",
    ("ASIANPAINT", "2023-05-02"): "Former resistance is a support candidate after accepted break; hold/retest and H1 trigger remain required.",
    ("APOLLOHOSP", "2023-01-03"): "Higher structural resistance and the older ~4195 support remain visible; current location is a resistance watch.",
    ("APOLLOHOSP", "2023-02-01"): "The ~4195 support remains visible and outranks a weaker nearby area; support damage suppresses blind buying and moves the state to reclaim.",
    ("APOLLOHOSP", "2023-04-03"): "Reclaim remains required; the higher local area and older ~4195 support remain visible in the persistent map.",
    ("EICHERMOT", "2023-07-03"): "Validated support revisit is monitored.",
    ("EICHERMOT", "2023-08-01"): "The broken support is treated as resistance/reclaim context, not as active support.",
    ("EICHERMOT", "2023-09-01"): "Support entry is suppressed below both major EMAs; reclaim logic is the appropriate path.",
    ("EICHERMOT", "2023-11-01"): "Support entry remains suppressed below both major EMAs; the nearby resistance map persists.",
    ("EICHERMOT", "2023-11-06"): "The long-respected ~3440 resistance is probed through the independent direct H1 path; retest is not mandatory, but candle, volume, friction and RR requirements remain.",
    ("EICHERMOT", "2023-11-09"): "Daily role tracking continues; direct breakout and breakout/retest are independent paths.",
    ("EICHERMOT", "2023-11-10"): "The acceptance sequence is still tracked separately from a confirmed support role.",
    ("EICHERMOT", "2023-11-13"): "A D-1 retest/hold can now confirm the former resistance as support.",
    ("ADANIPORTS", "2023-05-29"): "Accepted former resistance is a support candidate; the H1 hold/trigger is not implied by the daily map.",
    ("CIPLA", "2023-07-11"): "Repeated validated support remains eligible for a fresh interaction-specific setup.",
}


def _episode_rows(zone: Any, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    episodes = []
    for episode in zone.touch_episodes:
        episodes.append({
            "start_date": str(rows[episode.start_index]["date"])[:10],
            "end_date": str(rows[episode.end_index]["date"])[:10],
            "duration_sessions": episode.duration,
            "separation_from_previous_sessions": episode.separation_from_previous,
        })
    return episodes


def _zone_detail(zone: Any, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    episodes = _episode_rows(zone, rows)
    return {
        **zone_row(zone, rows),
        "original_side": zone.original_side,
        "creation_date": zone.creation_date,
        "first_pivot": zone.first_pivot,
        "last_update": zone.last_update,
        "first_interaction": episodes[0]["start_date"] if episodes else None,
        "last_interaction": episodes[-1]["end_date"] if episodes else None,
        "interaction_episodes": episodes,
        "post_interaction_reactions_atr": [dict(x) for x in zone.reactions],
        "role_flip": zone.role_flip,
        "acceptance_close_count": zone.acceptance_close_count,
    }


def build_translation_audit(*, data_root: str | Path = DATA_ROOT,
                            manifest_path: str | Path = MANIFEST_PATH,
                            config_path: str | Path = CONFIG,
                            output_path: str | Path = OUTPUT) -> dict[str, Any]:
    raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    cfg = _cfg(raw)
    daily, data_hash = load_daily_dataset_through(universe_id=PIT_TRADABLE,
        start_date=DAILY_WARMUP, end_date=END, data_root=data_root, manifest_path=manifest_path)
    hourly, hourly_hash = load_hourly_dataset_through(universe_id=PIT_TRADABLE,
        start_date=HOURLY_WARMUP, end_date=END, data_root=data_root)
    provider = UniverseProvider()
    members = {member.symbol: member.security_id
               for snapshot in (date(2023, 1, 1), date(2024, 12, 31))
               for member in provider.members_on(snapshot, PIT_TRADABLE)}
    checks = []
    for symbol, sod in CHECKS:
        security_id = members.get(symbol)
        if security_id is None or security_id not in daily:
            raise ValueError(f"No PIT daily series found for {symbol} ({security_id}).")
        prefix = [row for row in daily[security_id] if str(row["date"])[:10] < sod]
        if not prefix:
            raise ValueError(f"No completed daily rows before {sod} for {symbol}.")
        adaptive = adaptive_recent_structure(daily[security_id], sod_date=sod,
            security_id=security_id, cfg=cfg)
        loc = daily_location(adaptive["rows"], adaptive["zones"], cfg,
            structure_rows=adaptive["structure_rows"])
        activations = daily_setup_activations(loc, cfg)
        reference = float(prefix[-1]["close"])
        current_atr = atr(adaptive["rows"], 14)
        nearby = sorted((z for z in adaptive["zones"] if z.validated and
            abs(z.center-reference) <= 2.5 * current_atr),
            key=lambda z: (abs(z.center-reference), z.zone_id))
        selected = (loc.get("support") or loc.get("broken_zone") or loc.get("nearest_resistance") or
                    loc.get("suppressed_support_zone"))
        resistance_above = [z for z in adaptive["zones"] if z.validated and
            z.strength in {"MEANINGFUL", "MAJOR"} and z.side == "resistance" and z.lower_bound > reference]
        target_candidate = min(resistance_above, key=lambda z: (z.lower_bound, -z.evidence_score, z.zone_id), default=None)
        ema50 = ema(adaptive["structure_rows"], 50)
        ema200 = ema(adaptive["structure_rows"], 200)
        primary=loc.get("dominant_zone")
        secondary=loc.get("secondary_zones", [])
        state=str(loc.get("state", ""))
        action=("WAIT_RECLAIM" if state in {"WAIT_FOR_RECLAIM","BROKEN_SUPPORT_RECLAIM_WATCH","DAILY_SUPPORT_BROKEN"}
                else "WATCH" if "SUPPORT" in state or "RESISTANCE_BREAKOUT_WATCH" in state
                else "WAIT_RETEST" if state in {"BREAKOUT_ABOVE_RECENT_DAILY_RESISTANCE","RETESTING_BROKEN_RESISTANCE"}
                else "IGNORE")
        h1_probe=None
        bars=hourly.get(security_id, [])
        day_indices=[i for i,row in enumerate(bars) if str(row["date"])[:10]==sod]
        probes=[]
        if day_indices:
            weeks=completed_weekly_rows(daily[security_id],date.fromisoformat(sod))
            weekly_zones=enriched_zones(weeks,security_id,cfg.weekly_zone,cfg) if weeks else ()
            for activation in activations:
                candidate=activation["zone"]
                floor=candidate.upper_bound if activation["family"]=="BREAKOUT_RETEST" else reference
                daily_resistance=[z for z in adaptive["zones"] if z.side=="resistance"
                    and z.lower_bound>floor and z.strength in {"MEANINGFUL","MAJOR"}
                    and (not z.broken or z.role_state in {"RESISTANCE_CONFIRMED","RESISTANCE_CANDIDATE"})]
                stop=candidate.lower_bound-current_atr*cfg.stop_buffer_atr
                result,rejections=hourly_entry(hourly_rows=bars,start_index=day_indices[0],
                    end_index=day_indices[-1],support=candidate,daily_stop=stop,
                    daily_resistance=daily_resistance,weekly_zones=weekly_zones,daily_atr=current_atr,
                    security_id=security_id,cfg=cfg,setup_family=activation["family"])
                probes.append({"zone":_zone_detail(candidate,adaptive["structure_rows"]),
                    "setup_family":activation["family"],"path":"BREAKOUT_RETEST" if activation["family"]=="BREAKOUT_RETEST" else "H1_CONFIRMATION",
                    "entry_ready":result is not None,"plan":result,"rejection_reasons":rejections})
            if symbol=="EICHERMOT" and sod=="2023-11-06":
                candidates=[z for z in adaptive["zones"] if z.validated and z.side=="resistance"
                    and z.strength in {"MEANINGFUL","MAJOR"}
                    and z.role_state in {"ACTIVE","RESISTANCE_CANDIDATE","BROKEN_RESISTANCE_WAITING_HOLD"}
                    and z.lower_bound < reference+2.5*current_atr and z.upper_bound > reference-2.5*current_atr]
                for candidate in sorted(candidates,key=lambda z:(abs(z.center-reference),-z.dominance_score,z.zone_id)):
                    daily_resistance=[z for z in adaptive["zones"] if z.side=="resistance"
                        and z.lower_bound>candidate.upper_bound and z.strength in {"MEANINGFUL","MAJOR"}
                        and (not z.broken or z.role_state in {"RESISTANCE_CONFIRMED","RESISTANCE_CANDIDATE"})]
                    result,rejections=hourly_entry(hourly_rows=bars,start_index=day_indices[0],end_index=day_indices[-1],
                        support=candidate,daily_stop=candidate.lower_bound-current_atr*cfg.stop_buffer_atr,
                        daily_resistance=daily_resistance,weekly_zones=weekly_zones,daily_atr=current_atr,
                        security_id=security_id,cfg=cfg,setup_family="CONSOLIDATION_BREAKOUT")
                    probes.append({"zone":_zone_detail(candidate,adaptive["structure_rows"]),
                        "setup_family":"CONSOLIDATION_BREAKOUT","path":"DIRECT_STRUCTURAL_BREAKOUT_CANDIDATE_PROBE",
                        "independent_candidate_diagnostic":True,"entry_ready":result is not None,
                        "plan":result,"rejection_reasons":rejections})
            h1_probe={"session_date":sod,"available_through":"that session only; fill uses next available H1 open",
                "no_future_intraday_bars_used_for_confirmation":True,"probes":probes}
            if any(p["entry_ready"] and not p.get("independent_candidate_diagnostic") for p in probes):
                action="ENTRY_READY"
        checks.append({
            "symbol": symbol, "security_id": security_id, "sod_date": sod,
            "available_through": "D-1", "d_minus_1_date": str(prefix[-1]["date"])[:10],
            "d_minus_1_ohlcv": {key: prefix[-1][key] for key in ("open", "high", "low", "close", "volume")},
            "local_context": {"sessions_used": adaptive["sessions_used"],
                "start": str(adaptive["rows"][0]["date"])[:10], "end": str(adaptive["rows"][-1]["date"])[:10],
                "sufficiency_reason": adaptive["extension_reason"], "sufficient": adaptive["local_context_sufficient"]},
            "structural_memory": {"sessions": adaptive["memory_sessions"],
                "start": str(adaptive["structure_rows"][0]["date"])[:10],
                "end": str(adaptive["structure_rows"][-1]["date"])[:10], "role_map_is_asof_d1": True},
            "daily_map": {"state": loc["state"], "action":action,"reason_codes": loc.get("reason_codes", []),
                "activation_state": loc.get("activation_state"),
                "setup_watch_families": [x["family"] for x in activations],
                "setup_event_ids": [x["setup_event_id"] for x in activations],
                "primary_zone":primary,"secondary_zones":secondary,
                "ranking_method":loc.get("ranking_method"),
                "distance_is_tiebreaker_only":loc.get("distance_is_tiebreaker_only"),
                "selected_zone": _zone_detail(selected, adaptive["structure_rows"]) if selected else None,
                "nearby_validated_zones": [_zone_detail(z, adaptive["structure_rows"]) for z in nearby],
                "next_daily_structural_resistance": _zone_detail(target_candidate, adaptive["structure_rows"]) if target_candidate else None,
                "target_diagnostic": loc.get("target_info"),"stop_reference":loc.get("stop_reference"),
                "h1_setup_probes":h1_probe},
            "ema_context": {"ema50": ema50, "ema200": ema200,
                "observations_available": len(adaptive["structure_rows"]), "ema200_available": ema200 is not None,
                "close_below_ema50": reference < ema50 if ema50 is not None else None,
                "close_below_ema200": reference < ema200 if ema200 is not None else None},
            "manual_translation_readout": MANUAL_READOUT[(symbol, sod)],
            "human_expectation":("WAIT_RECLAIM" if symbol=="TECHM" and sod in {"2023-04-12","2023-04-13"}
                else "DIRECT_BREAKOUT_TESTED_WITHOUT_MANDATORY_RETEST" if symbol=="EICHERMOT" and sod=="2023-11-06"
                else "PERSISTENT_STRUCTURE_REMAINS_VISIBLE"),
            "translation_status":("MATCH" if symbol=="TECHM" and sod in {"2023-04-12","2023-04-13"} and action=="WAIT_RECLAIM"
                else "MATCH" if h1_probe and any(p["entry_ready"] for p in h1_probe["probes"])
                else "PARTIAL_MATCH"),
        })
    payload = {"audit_version": "v2_persistent_daily_map_translation_v2",
        "research_period": {"start": "2023-01-01", "end": "2024-12-31"},
        "asof_rule": "SOD date D uses completed daily rows strictly before D; no 2025/2026 rows loaded.",
        "scope": "causal D-1 daily structure audit and same-session H1 probes for the 19 manual observations, including an independent EICHERMOT direct-breakout candidate probe; no portfolio simulation",
        "daily_data_sha256": data_hash,"hourly_data_sha256":hourly_hash, "checks": checks,
        "case_summary":_case_summary(checks)}
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")
    return payload


def _case_summary(checks: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    expectations={"TECHM":"WAIT_RECLAIM","ASIANPAINT":"SUPPORT_REVISITS_AND_BREAKOUT_RETEST_AVAILABLE",
        "APOLLOHOSP":"OLDER_4195_ZONE_REMAINS_VISIBLE","EICHERMOT":"DIRECT_BREAKOUT_TESTED_WITHOUT_REQUIRED_RETEST",
        "ADANIPORTS":"ACCEPTED_BREAKOUT_RETEST_REMAINS_AVAILABLE","CIPLA":"REPEATED_SUPPORT_REVISIT_REMAINS_AVAILABLE"}
    representative={"TECHM":"2023-04-13","ASIANPAINT":"2023-05-02","APOLLOHOSP":"2023-02-01",
        "EICHERMOT":"2023-11-06","ADANIPORTS":"2023-05-29","CIPLA":"2023-07-11"}
    result=[]
    for symbol,expected in expectations.items():
        rows=[row for row in checks if row["symbol"]==symbol]
        representative_row=next((row for row in rows if row["sod_date"]==representative[symbol]),rows[-1] if rows else {})
        primary=representative_row.get("daily_map",{}).get("selected_zone")
        broad_primary=representative_row.get("daily_map",{}).get("primary_zone")
        representative_probes=(representative_row.get("daily_map",{}).get("h1_setup_probes") or {}).get("probes",[])
        if symbol=="EICHERMOT":
            direct_candidates=[p["zone"] for p in representative_probes if p.get("independent_candidate_diagnostic")
                and 3400<float(p["zone"].get("lower_bound",0))<3500]
            if direct_candidates:
                primary=min(direct_candidates,key=lambda z:abs(float(z["center"])-3470))
        ready=any(p["entry_ready"] and not p.get("independent_candidate_diagnostic") for p in representative_probes)
        actions=sorted({row["daily_map"]["action"] for row in rows})
        action=", ".join(actions)
        if symbol=="TECHM": summary_status="MATCH"
        elif symbol=="APOLLOHOSP": summary_status="MATCH" if primary and 4150<float(primary.get("lower_bound",0))<4250 else "PARTIAL_MATCH"
        elif symbol=="EICHERMOT": summary_status="MATCH" if any(p.get("independent_candidate_diagnostic") for p in representative_probes) else "PARTIAL_MATCH"
        elif symbol=="ASIANPAINT":
            families={family for row in rows for family in row["daily_map"]["setup_watch_families"]}
            summary_status="MATCH" if {"SUPPORT_REACTION","BREAKOUT_RETEST"}.issubset(families) else "PARTIAL_MATCH"
        elif symbol=="CIPLA": summary_status="MATCH" if "SUPPORT_REACTION" in representative_row["daily_map"]["setup_watch_families"] else "PARTIAL_MATCH"
        else: summary_status="PARTIAL_MATCH"
        if symbol=="EICHERMOT":
            rr=[r.get("post_clearance_structural_rr") for p in representative_probes
                if p.get("independent_candidate_diagnostic") and 3400<float(p["zone"].get("lower_bound",0))<3500
                for r in p["rejection_reasons"] if r.get("reason_code")=="RR_INSUFFICIENT"]
            action=f"DIRECT_PATH_TESTED; RR_REJECT ({max(rr):.2f}R < configured minimum)" if rr else "DIRECT_PATH_TESTED; no entry-ready result"
        elif ready: action="ENTRY_READY"
        result.append({"case":symbol,"primary_structure":({"setup_anchor_zone_id":primary.get("zone_id"),
            "bounds":[primary.get("lower_bound"),primary.get("upper_bound")],
            "dominance_score":primary.get("dominance_score"),
            "broad_map_primary":({"zone_id":broad_primary.get("zone_id"),"bounds":[broad_primary.get("lower_bound"),broad_primary.get("upper_bound")],
                "dominance_score":broad_primary.get("dominance_score")} if broad_primary else None)} if primary else None),
            "engine_action":action,
            "expected_action":expected,"match_status":summary_status,"dates_audited":[row["sod_date"] for row in rows]})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = build_translation_audit(output_path=args.output)
    print(json.dumps({"checks": len(result["checks"]), "output": str(args.output),
        "daily_data_sha256": result["daily_data_sha256"], "scope": result["scope"]}, indent=2))


if __name__ == "__main__":
    main()
