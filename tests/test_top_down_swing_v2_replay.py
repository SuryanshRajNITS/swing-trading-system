from dataclasses import replace

from src.backtest.top_down_swing_v2_replay import (
    _consume_zone_thesis,
    _lifecycle_transition,
    _refresh_consumed_theses,
    _zone_thesis_consumed,
)
from src.strategy.sr_zone_v2 import Zone


def _zone(zone_id="zone-a", *, side="support", role_state="SUPPORT_CONFIRMED"):
    return Zone(zone_id,"SEC",side,100,102,101,"2023-01-01","2023-01-01","2023-01-02",
        ("2023-01-01","2023-01-02"),(),(),False,None,"FRESH",original_side="resistance",
        role_state=role_state,strength="MEANINGFUL",last_touch_index=1)


def test_consumed_interaction_suppresses_duplicate_but_allows_later_revisit():
    consumed={}
    zone=_zone()
    lifecycle=[{"status":"WATCHING"},{"status":"ENTRY_READY"},{"status":"CONSUMED"}]
    _consume_zone_thesis(consumed,"SEC",zone,lifecycle,"zone-a:2023-01-10")

    # The same zone blocks a fresh family/watch on a later SOD.
    assert _zone_thesis_consumed(consumed,"SEC",zone,"zone-a:2023-01-10")
    assert not _zone_thesis_consumed(consumed,"SEC",zone,"zone-a:2023-02-10")
    assert not _zone_thesis_consumed(consumed,"OTHER",zone,"zone-a:2023-01-10")
    assert not _refresh_consumed_theses(consumed,"SEC",[zone],101)
    reidentified=replace(zone,zone_id="zone-a-expanded",lower_bound=99.5,upper_bound=102.5)
    assert not _refresh_consumed_theses(consumed,"SEC",[reidentified],101)
    assert not _zone_thesis_consumed(consumed,"SEC",reidentified,"zone-a:2023-01-10")

    # A daily close below the consumed support is an explicit invalidation.
    released=_refresh_consumed_theses(consumed,"SEC",[zone],99)
    assert released[0]["reason_code"]=="LONG_SUPPORT_THESIS_INVALIDATED_BY_CLOSE_BELOW_ZONE"
    assert ("SEC",zone.zone_id) not in consumed


def test_changed_role_or_removed_zone_resets_consumed_thesis():
    consumed={}
    zone=_zone()
    _consume_zone_thesis(consumed,"SEC",zone,[{"status":"CONSUMED"}],"event-1")
    flipped=replace(zone,side="resistance",role_state="RESISTANCE_CONFIRMED")
    assert _refresh_consumed_theses(consumed,"SEC",[flipped],101)[0]["reason_code"]=="ZONE_ROLE_CHANGED"
    _consume_zone_thesis(consumed,"SEC",flipped,[{"status":"CONSUMED"}],"event-2")
    assert _refresh_consumed_theses(consumed,"SEC",[],101)[0]["reason_code"]=="ZONE_REMOVED_FROM_CURRENT_STRUCTURE_MAP"


def test_lifecycle_transition_is_deterministic_and_does_not_duplicate_state():
    watch={"lifecycle_status":"WATCHING","lifecycle":[{"status":"WATCHING","timestamp":"2023-01-01"}]}
    _lifecycle_transition(watch,"BREAKOUT_ACCEPTED","2023-01-02",breakout_volume_passed=True)
    _lifecycle_transition(watch,"BREAKOUT_ACCEPTED","2023-01-02",breakout_volume_passed=True)
    _lifecycle_transition(watch,"RETEST_WAIT","2023-01-02")
    assert [item["status"] for item in watch["lifecycle"]]==["WATCHING","BREAKOUT_ACCEPTED","RETEST_WAIT"]
