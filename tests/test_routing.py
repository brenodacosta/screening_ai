"""Smoke tests for the deterministic parts of routing (no LLM)."""

from api.models import CandidateRecord, ConversationState, Stage
from api.nodes import _all_required_filled, _next_missing_stage


def test_next_missing_stage_walks_in_order():
    c = CandidateRecord()
    assert _next_missing_stage(c) == Stage.ASK_NAME
    c.full_name = "Juan Pérez"
    assert _next_missing_stage(c) == Stage.ASK_LICENSE
    c.drivers_license = True
    assert _next_missing_stage(c) == Stage.ASK_CITY
    c.city_zone = "Madrid"
    assert _next_missing_stage(c) == Stage.ASK_AVAILABILITY
    c.availability = ["full_time"]
    assert _next_missing_stage(c) == Stage.ASK_SCHEDULE
    c.preferred_schedule = ["morning"]
    assert _next_missing_stage(c) == Stage.ASK_EXPERIENCE
    c.experience_years = 2
    assert _next_missing_stage(c) == Stage.ASK_START_DATE
    c.start_date = "2026-05-20"
    assert _next_missing_stage(c) == Stage.QUALIFIED


def test_all_required_filled_requires_license_true():
    c = CandidateRecord(
        full_name="x",
        drivers_license=False,
        city_zone="Madrid",
        availability=["full_time"],
        preferred_schedule=["morning"],
        experience_years=0,
        start_date="immediately",
    )
    # Hard disqualifier — but _all_required_filled doesn't gate on that, it only
    # checks presence. The route_node applies the disqualifier rule itself.
    assert _all_required_filled(c) is False  # license must be True


def test_fresh_state_has_no_transcript():
    s = ConversationState()
    assert s.transcript == []
    assert s.reask_counts == {}
