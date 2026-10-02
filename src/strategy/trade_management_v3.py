"""One-way V3 structural profit protection, independent of portfolio accounting."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class ProtectionState:
    original_stop: float
    current_stop: float
    initial_risk: float
    mfe_r: float = 0.0
    milestone_r: float = 0.0
    stop_move_timestamp: str | None = None
    stop_move_reason: str | None = None
    structural_reference: float | None = None
    r_floor: float | None = None
    r_floor_price: float | None = None


MILESTONE_FLOORS = ((3.0, 1.75), (2.5, 1.25), (2.0, 0.75), (1.5, 0.0), (1.0, None))


def confirmed_hourly_higher_lows(rows: Sequence[Mapping[str, Any]], *, left: int = 2,
                                 right: int = 2) -> list[dict[str, Any]]:
    """Confirmed lows only; a candidate is available after `right` later bars."""
    output=[]
    for i in range(left, len(rows)-right):
        low=float(rows[i]["low"])
        if low > min(float(rows[j]["low"]) for j in range(i-left,i+right+1) if j!=i):
            continue
        if output and low <= float(output[-1]["price"]):
            continue
        output.append({"index":i,"confirmed_index":i+right,"timestamp":str(rows[i]["date"]),"price":low})
    return output


def initial_protection(entry: float, original_stop: float) -> ProtectionState:
    if entry <= original_stop:
        raise ValueError("Long trade requires entry above original structural stop")
    return ProtectionState(original_stop, original_stop, entry-original_stop)


def update_protection(state: ProtectionState, *, entry: float, high_since_entry: float,
                      timestamp: str, confirmed_h1_rows: Sequence[Mapping[str, Any]],
                      h1_atr: float, trailing_buffer_atr: float = 0.25,
                      estimated_round_trip_cost_per_share: float = 0.0) -> ProtectionState:
    """Advance stop from information available at `timestamp`; never lower it.

    `confirmed_h1_rows` must end at the latest completed H1 bar available at the
    update timestamp. The cost allowance affects only the breakeven floor.
    """
    risk=state.initial_risk
    mfe=max(state.mfe_r,(high_since_entry-entry)/risk)
    floor_r=next((floor for milestone,floor in MILESTONE_FLOORS if floor is not None and mfe>=milestone),None)
    milestone=max(state.milestone_r, max((m for m,_ in MILESTONE_FLOORS if mfe>=m),default=0.0))
    if floor_r is None:
        return ProtectionState(state.original_stop,state.current_stop,risk,mfe,milestone,
            state.stop_move_timestamp,state.stop_move_reason,state.structural_reference,state.r_floor,state.r_floor_price)
    floor_price=entry+floor_r*risk
    if floor_r==0:
        floor_price=max(floor_price,entry+max(0.0,estimated_round_trip_cost_per_share))
    lows=confirmed_hourly_higher_lows(confirmed_h1_rows)
    valid=[x for x in lows if int(x["confirmed_index"])<len(confirmed_h1_rows)]
    reference=float(valid[-1]["price"]) if valid else None
    structural=(reference-max(0.0,trailing_buffer_atr*h1_atr)) if reference is not None else None
    candidate=max(floor_price,structural if structural is not None else floor_price)
    next_stop=max(state.current_stop,candidate)
    moved=next_stop>state.current_stop+1e-12
    return ProtectionState(state.original_stop,next_stop,risk,mfe,milestone,
        timestamp if moved else state.stop_move_timestamp,
        ("MILESTONE_FLOOR_AND_CONFIRMED_H1_HIGHER_LOW" if moved else state.stop_move_reason),
        reference,state.r_floor if not moved else floor_r,
        state.r_floor_price if not moved else floor_price)


def stop_execution(open_price: float, low_price: float, stop: float) -> tuple[bool,float|None,str|None]:
    """Gap-through protective/original stop fills at open; otherwise at stop."""
    if open_price <= stop:
        return True,open_price,"GAP_STOP"
    if low_price <= stop:
        return True,stop,"PROTECTIVE_STOP"
    return False,None,None


def protection_record(state: ProtectionState) -> dict[str,Any]:
    return asdict(state)
