from datetime import date

from src.data.universe.provider import PIT_TRADABLE, UniverseProvider


def test_jiofin_is_eligible_only_during_its_dated_pit_membership():
    universe = UniverseProvider()
    security_id = "NSEEQ-0036"

    assert universe.is_security_eligible(security_id, date(2023, 8, 21), PIT_TRADABLE)
    assert not universe.is_security_eligible(security_id, date(2023, 12, 20), PIT_TRADABLE)
    assert not universe.is_security_eligible(security_id, date(2024, 12, 9), PIT_TRADABLE)
