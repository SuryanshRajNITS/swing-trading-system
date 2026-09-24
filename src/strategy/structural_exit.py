"""Experimental causal structural stops, targets, and maximum-hold exits."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, replace
from datetime import date
from statistics import fmean
from typing import Any, Mapping, Sequence

from src.decision.interfaces import EntrySignal, ExitSignal
from src.strategy.sr_zone_v2 import Zone, ZoneConfig, detect_zones


@dataclass(frozen=True)
class StructuralExitConfig:
    stop_buffer_atr: float = 0.25
    minimum_stop_buffer_pct: float = 0.001
    target_buffer_atr: float = 0.10
    minimum_target_buffer_pct: float = 0.001
    minimum_reward_risk: float = 1.5
    maximum_holding_sessions: int = 15
    no_target_policy: str = "REJECT"
    fallback_reward_risk: float = 2.0

    def __post_init__(self) -> None:
        if min(self.stop_buffer_atr,self.minimum_stop_buffer_pct,self.target_buffer_atr,
               self.minimum_target_buffer_pct) < 0:
            raise ValueError("Structural exit buffers must not be negative.")
        if self.minimum_reward_risk <= 0:
            raise ValueError("minimum_reward_risk must be positive.")
        if self.maximum_holding_sessions < 1:
            raise ValueError("maximum_holding_sessions must be positive.")
        if self.no_target_policy not in {"REJECT", "FIXED_R_FALLBACK"}:
            raise ValueError("no_target_policy must be REJECT or FIXED_R_FALLBACK.")
        if self.fallback_reward_risk <= 0:
            raise ValueError("fallback_reward_risk must be positive.")


def _day(value: Any) -> date:
    return date.fromisoformat(str(value)[:10])


def daily_atr(rows: Sequence[Mapping[str,Any]], period: int=14) -> float | None:
    if len(rows)<period:
        return None
    true_ranges=[]
    first=len(rows)-period
    for index in range(first,len(rows)):
        row=rows[index]
        high,low=float(row["high"]),float(row["low"])
        previous=float(rows[index-1]["close"]) if index else float(row["close"])
        true_ranges.append(max(high-low,abs(high-previous),abs(low-previous)))
    return fmean(true_ranges)


def active_resistance_zones_asof(*, daily_rows: Sequence[Mapping[str,Any]], security_id: str,
                                 as_of: date, zone_config: ZoneConfig) -> tuple[Zone,...]:
    """Build active zones from only candles dated on or before the signal close."""
    visible=[row for row in daily_rows if _day(row["date"])<=as_of]
    if not visible:
        return ()
    zones=detect_zones(visible,security_id=security_id,config=zone_config,include_lifecycle=True)
    return tuple(zone for zone in zones if zone.side=="resistance" and not zone.broken
                 and len(zone.pivots)>=zone_config.minimum_pivots)


def build_structural_exit_plan(*, setup_type: str, signal_metadata: Mapping[str,Any],
                               active_resistances: Sequence[Any], atr: float | None,
                               config: StructuralExitConfig|None=None,
                               enforce_minimum_reward_risk: bool=True) -> dict[str,Any]:
    """Create a structural plan or an explicit rejection; no synthetic target fallback."""
    cfg=config or StructuralExitConfig()
    expected_entry=float(signal_metadata.get("proposed_entry") or signal_metadata.get("signal_close") or 0)
    zone=signal_metadata.get("zone") or {}
    zone_id=signal_metadata.get("zone_id")
    if expected_entry<=0 or atr is None or atr<=0:
        return {"accepted":False,"reason_code":"INVALID_STRUCTURAL_INPUT",
                "expected_entry":expected_entry,"atr":atr}

    if setup_type=="support_test":
        reference=zone.get("lower")
        reference_type="SUPPORT_ZONE_LOWER_BOUNDARY"
        alternatives={"support_zone_lower_boundary":reference}
    elif setup_type=="breakout_retest":
        broken_lower=zone.get("lower")
        retest_low=signal_metadata.get("retest_low")
        candidates={key:float(value) for key,value in
                    (("retest_low",retest_low),("broken_zone_lower_boundary",broken_lower))
                    if value is not None}
        if not candidates:
            return {"accepted":False,"reason_code":"INVALID_STRUCTURAL_INPUT",
                    "expected_entry":expected_entry,"atr":atr}
        # Taking the lower of the as-of retest low and broken boundary preserves
        # the more conservative structural invalidation level.
        reference_type,reference_key=min(candidates.items(),key=lambda item:item[1])
        reference=reference_key
        reference_type=("RETEST_LOW" if reference_type=="retest_low" else
                        "BROKEN_ZONE_LOWER_BOUNDARY")
        alternatives=candidates
    else:
        return {"accepted":False,"reason_code":"UNSUPPORTED_SETUP_TYPE",
                "expected_entry":expected_entry,"atr":atr}

    if reference is None:
        return {"accepted":False,"reason_code":"INVALID_STRUCTURAL_INPUT",
                "expected_entry":expected_entry,"atr":atr}
    reference=float(reference)
    stop_buffer=max(atr*cfg.stop_buffer_atr,abs(reference)*cfg.minimum_stop_buffer_pct)
    stop=reference-stop_buffer
    risk=expected_entry-stop
    stop_fields={
        "structural_reference_price":reference,"structural_reference_type":reference_type,
        "structural_reference_candidates":alternatives,"structural_zone_id":zone_id,
        "stop_buffer":stop_buffer,"stop_price":stop,
        "stop_distance_percent":risk/expected_entry*100,
        "stop_distance_atr":risk/atr,
        "stop_reason_code":f"{reference_type}_BUFFERED",
    }
    if risk<=0:
        return {"accepted":False,"reason_code":"INVALID_STRUCTURAL_STOP",
                "expected_entry":expected_entry,"atr":atr,**stop_fields}

    higher=[item for item in active_resistances
            if not getattr(item,"broken",False)
            and float(getattr(item,"lower_bound"))>expected_entry]
    target_zone=min(higher,key=lambda item:float(item.lower_bound),default=None)
    if target_zone is None:
        if cfg.no_target_policy=="FIXED_R_FALLBACK":
            target=expected_entry+cfg.fallback_reward_risk*risk
            target_fields={
                "target_resistance_zone_id":None,"target_zone_lower_bound":None,
                "target_zone_upper_bound":None,"target_buffer":None,"target_price":target,
                "target_distance_percent":(target-expected_entry)/expected_entry*100,
                "resulting_reward_risk_ratio":cfg.fallback_reward_risk,
                "target_type":"EXPLICIT_FIXED_R_FALLBACK",
            }
            if cfg.fallback_reward_risk<cfg.minimum_reward_risk:
                return {"accepted":False,"reason_code":"INSUFFICIENT_STRUCTURAL_RR",
                        "expected_entry":expected_entry,"atr":atr,**stop_fields,**target_fields,
                        "target_status":"EXPLICIT_FIXED_R_FALLBACK"}
            return {"accepted":True,"reason_code":"EXPLICIT_FIXED_R_FALLBACK_NO_STRUCTURAL_TARGET",
                    "expected_entry":expected_entry,"atr":atr,**stop_fields,**target_fields,
                    "target_status":"EXPLICIT_FIXED_R_FALLBACK"}
        return {"accepted":False,"reason_code":"NO_STRUCTURAL_TARGET",
                "expected_entry":expected_entry,"atr":atr,**stop_fields,
                "target_status":"NO_STRUCTURAL_TARGET"}
    lower=float(target_zone.lower_bound)
    upper=float(target_zone.upper_bound)
    target_buffer=max(atr*cfg.target_buffer_atr,abs(lower)*cfg.minimum_target_buffer_pct)
    target=lower-target_buffer
    reward=target-expected_entry
    rr=reward/risk
    target_fields={
        "target_resistance_zone_id":target_zone.zone_id,
        "target_zone_lower_bound":lower,"target_zone_upper_bound":upper,
        "target_buffer":target_buffer,"target_price":target,
        "target_type":"STRUCTURAL_RESISTANCE",
        "target_distance_percent":reward/expected_entry*100,
        "resulting_reward_risk_ratio":rr,
    }
    if reward<=0:
        return {"accepted":False,"reason_code":"NO_STRUCTURAL_TARGET",
                "expected_entry":expected_entry,"atr":atr,**stop_fields,**target_fields,
                "target_status":"NO_STRUCTURAL_TARGET"}
    if enforce_minimum_reward_risk and rr<cfg.minimum_reward_risk:
        return {"accepted":False,"reason_code":"INSUFFICIENT_STRUCTURAL_RR",
                "expected_entry":expected_entry,"atr":atr,**stop_fields,**target_fields,
                "target_status":"VALID_STRUCTURAL_TARGET"}
    return {"accepted":True,"reason_code":"STRUCTURAL_PLAN_ACCEPTED",
            "expected_entry":expected_entry,"atr":atr,**stop_fields,**target_fields,
            "target_status":"VALID_STRUCTURAL_TARGET"}


class StructuralExitStrategy:
    """Decorate the saved C-arm signal strategy without changing its setup rules."""

    name="sr_zone_v2_weekly_selection_structural_exit"
    version="sr_zone_v2_weekly_selection_structural_exit_v1"

    def __init__(self, *, signal_strategy: Any, config: StructuralExitConfig|None=None) -> None:
        self.signal_strategy=signal_strategy
        self.config=config or StructuralExitConfig()
        self.parameters={"exit_mode":"STRUCTURAL_EXIT","config":asdict(self.config),
                         "signal_strategy":dict(signal_strategy.parameters)}
        self.entry_decisions: list[dict[str,Any]]=[]
        self.time_exit_decisions: list[dict[str,Any]]=[]
        self.rejection_counts: Counter[str]=Counter()
        self.zone_config=self._find_zone_config()

    def _find_zone_config(self) -> ZoneConfig:
        current=self.signal_strategy
        while current is not None:
            config=getattr(current,"zone_config",None)
            if config is not None:
                return config
            current=getattr(current,"daily_strategy",None)
        raise ValueError("Structural exits require the wrapped sr_zone_v2 zone configuration.")

    @property
    def diagnostics(self):
        return self.signal_strategy.diagnostics

    def _record_diagnostic(self, signal: EntrySignal, context: Any, reason: str) -> None:
        setup=str(signal.metadata.get("setup_type","unknown"))
        for row in reversed(self.diagnostics):
            if (row.get("security_id")==signal.security_id
                    and row.get("signal_date")==context.session_date.isoformat()
                    and row.get("setup_type")==setup and row.get("accepted")):
                row.update({"accepted":False,"rejection_reason":reason,"reason_code":reason,
                            "structural_exit_mode":"STRUCTURAL_EXIT"})
                return

    def generate_entries(self, context: Any):
        signals=self.signal_strategy.generate_entries(context)
        accepted=[]
        for signal in signals:
            rows=context.daily_history.get(signal.security_id,())
            atr=daily_atr(rows,self.zone_config.atr_period)
            resistances=active_resistance_zones_asof(
                daily_rows=rows,security_id=signal.security_id,as_of=context.session_date,
                zone_config=self.zone_config)
            plan=build_structural_exit_plan(setup_type=str(signal.metadata.get("setup_type","")),
                                            signal_metadata=signal.metadata,
                                            active_resistances=resistances,atr=atr,config=self.config)
            decision={"date":context.session_date.isoformat(),"security_id":signal.security_id,
                      "symbol":signal.symbol,"setup_type":signal.metadata.get("setup_type"),**plan}
            self.entry_decisions.append(decision)
            if not plan["accepted"]:
                reason=str(plan["reason_code"])
                self.rejection_counts[reason]+=1
                self._record_diagnostic(signal,context,reason)
                continue
            metadata={**dict(signal.metadata),"target_r":None,"exit_mode":"STRUCTURAL_EXIT",
                      "maximum_holding_sessions":self.config.maximum_holding_sessions,
                      "structural_exit":{key:value for key,value in plan.items() if key not in {"accepted","reason_code"}}}
            accepted.append(replace(signal,stop_price=float(plan["stop_price"]),
                                    target_price=float(plan["target_price"]),metadata=metadata,
                                    strategy_version=self.version,
                                    reason_codes=tuple(signal.reason_codes)+("STRUCTURAL_EXIT",)))
        return tuple(accepted)

    def generate_exits(self, context: Any):
        existing=list(getattr(self.signal_strategy,"generate_exits",lambda _:())(context))
        scheduled={signal.security_id for signal in existing}
        for security_id,position in context.open_positions.items():
            if security_id in scheduled or context.next_session_open is None:
                continue
            entry_day=_day(position["entry_timestamp"])
            rows=context.daily_history.get(security_id,())
            held=sum(entry_day<=_day(row["date"])<=context.session_date for row in rows)
            if held<self.config.maximum_holding_sessions:
                continue
            details={"holding_sessions":held,"maximum_holding_sessions":self.config.maximum_holding_sessions,
                     "profitable_at_signal_close":float(position.get("last_price",0))>float(position["entry_price"]),
                     "exit_price_basis":"NEXT_SESSION_OPEN"}
            self.time_exit_decisions.append({"date":context.session_date.isoformat(),
                                             "security_id":security_id,**details})
            existing.append(ExitSignal(security_id=security_id,signal_timestamp=context.timestamp,
                                       execution_timestamp=context.next_session_open,reason_code="TIME_EXIT",
                                       metadata=details,strategy_version=self.version))
        return tuple(existing)


__all__=["StructuralExitConfig","StructuralExitStrategy","active_resistance_zones_asof",
         "build_structural_exit_plan","daily_atr"]
