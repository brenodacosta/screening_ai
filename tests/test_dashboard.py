"""Unit tests for dashboard aggregation."""

from datetime import datetime, timedelta

import pytest

from api.dashboard import compute_metrics, load_candidates
from api.models import (
    CandidateRecord,
    ConversationState,
    Stage,
    Turn,
)


def _state(
    *,
    status: str,
    stage: Stage,
    started: datetime,
    ended: datetime | None,
    reason: str | None = None,
    job_id: str = "delivery_driver",
    sources: list[str] | None = None,
) -> ConversationState:
    transcript = []
    for src in sources or []:
        transcript.append(Turn(role="agent", text="x", ts=started, source=src))
    return ConversationState(
        started_at=started,
        ended_at=ended,
        current_stage=stage,
        candidate=CandidateRecord(),
        transcript=transcript,
        qualification_status=status,  # type: ignore[arg-type]
        disqualification_reason=reason,  # type: ignore[arg-type]
        job_id=job_id,
    )


def test_empty_returns_zeroed_schema():
    m = compute_metrics([])
    assert m.totals == {"qualified": 0, "disqualified": 0, "abandoned": 0, "total": 0}
    assert m.completion_rate == 0.0
    assert len(m.funnel) == 7
    assert all(step.reached == 0 for step in m.funnel)
    assert m.duration_seconds.overall_seconds is None
    assert m.message_source_ratio == {"llm": 0, "template": 0, "unknown": 0}
    assert m.throughput == []


def test_completion_rate_and_totals():
    t0 = datetime(2026, 5, 11, 8, 0, 0)
    states = [
        _state(status="qualified", stage=Stage.QUALIFIED, started=t0, ended=t0 + timedelta(minutes=2)),
        _state(status="qualified", stage=Stage.QUALIFIED, started=t0, ended=t0 + timedelta(minutes=4)),
        _state(status="disqualified", stage=Stage.ASK_CITY, started=t0, ended=t0 + timedelta(minutes=1), reason="outside_service_area"),
        _state(status="abandoned", stage=Stage.ASK_AVAILABILITY, started=t0, ended=t0 + timedelta(minutes=3)),
    ]
    m = compute_metrics(states)
    assert m.totals == {"qualified": 2, "disqualified": 1, "abandoned": 1, "total": 4}
    assert m.completion_rate == 0.5
    assert m.disqualification_reasons == {"outside_service_area": 1}


def test_drop_off_only_counts_non_qualified():
    t0 = datetime(2026, 5, 11, 8, 0, 0)
    states = [
        _state(status="qualified", stage=Stage.QUALIFIED, started=t0, ended=t0 + timedelta(seconds=60)),
        _state(status="disqualified", stage=Stage.ASK_CITY, started=t0, ended=t0 + timedelta(seconds=30), reason="outside_service_area"),
        _state(status="abandoned", stage=Stage.ASK_NAME, started=t0, ended=t0 + timedelta(seconds=10)),
    ]
    m = compute_metrics(states)
    assert m.drop_off_by_stage["ask_city"] == 1
    assert m.drop_off_by_stage["ask_name"] == 1
    assert m.drop_off_by_stage["ask_start_date"] == 0


def test_funnel_includes_qualified_at_all_stages():
    t0 = datetime(2026, 5, 11, 8, 0, 0)
    states = [
        _state(status="qualified", stage=Stage.QUALIFIED, started=t0, ended=t0 + timedelta(seconds=60)),
        _state(status="abandoned", stage=Stage.ASK_CITY, started=t0, ended=t0 + timedelta(seconds=30)),
    ]
    m = compute_metrics(states)
    by_stage = {step.stage: step.reached for step in m.funnel}
    assert by_stage["ask_name"] == 2  # both reached the first stage
    assert by_stage["ask_city"] == 2  # qualified + abandoned-at-city
    assert by_stage["ask_schedule"] == 1  # only the qualified one
    assert by_stage["ask_start_date"] == 1


def test_duration_handles_missing_ended_at():
    t0 = datetime(2026, 5, 11, 8, 0, 0)
    states = [
        _state(status="qualified", stage=Stage.QUALIFIED, started=t0, ended=t0 + timedelta(seconds=120)),
        _state(status="abandoned", stage=Stage.ASK_NAME, started=t0, ended=None),
    ]
    m = compute_metrics(states)
    assert m.duration_seconds.overall_seconds == 120.0
    assert m.duration_seconds.by_status["qualified"] == 120.0
    assert m.duration_seconds.by_status["abandoned"] is None


def test_message_source_ratio():
    t0 = datetime(2026, 5, 11, 8, 0, 0)
    states = [
        _state(
            status="qualified", stage=Stage.QUALIFIED, started=t0,
            ended=t0 + timedelta(seconds=60),
            sources=["template", "llm", "llm"],
        ),
        _state(
            status="abandoned", stage=Stage.ASK_NAME, started=t0,
            ended=t0 + timedelta(seconds=30),
            sources=["template"],
        ),
    ]
    m = compute_metrics(states)
    assert m.message_source_ratio == {"llm": 2, "template": 2, "unknown": 0}


def test_throughput_buckets_by_utc_date():
    states = [
        _state(status="qualified", stage=Stage.QUALIFIED,
               started=datetime(2026, 5, 10, 23, 0), ended=datetime(2026, 5, 10, 23, 5)),
        _state(status="qualified", stage=Stage.QUALIFIED,
               started=datetime(2026, 5, 11, 1, 0), ended=datetime(2026, 5, 11, 1, 5)),
        _state(status="abandoned", stage=Stage.ASK_NAME,
               started=datetime(2026, 5, 11, 2, 0), ended=datetime(2026, 5, 11, 2, 5)),
    ]
    m = compute_metrics(states)
    assert [b.date for b in m.throughput] == ["2026-05-10", "2026-05-11"]
    assert [b.count for b in m.throughput] == [1, 2]


def test_job_id_filter_via_load_candidates(tmp_path, monkeypatch):
    from api import dashboard as dash
    from api import config as cfg

    # Redirect data_dir to a temp directory.
    monkeypatch.setattr(cfg.settings, "data_dir", tmp_path)
    monkeypatch.setattr(dash.settings, "data_dir", tmp_path)

    qual_dir = tmp_path / "candidates" / "qualified"
    qual_dir.mkdir(parents=True)
    t0 = datetime(2026, 5, 11, 8, 0, 0)
    s1 = _state(status="qualified", stage=Stage.QUALIFIED, started=t0,
                ended=t0 + timedelta(seconds=60), job_id="delivery_driver")
    s2 = _state(status="qualified", stage=Stage.QUALIFIED, started=t0,
                ended=t0 + timedelta(seconds=60), job_id="other_role")
    (qual_dir / "a.json").write_text(s1.model_dump_json(), encoding="utf-8")
    (qual_dir / "b.json").write_text(s2.model_dump_json(), encoding="utf-8")
    (qual_dir / "broken.json").write_text("{not valid json", encoding="utf-8")

    assert len(load_candidates()) == 2  # broken file silently skipped
    assert len(load_candidates("delivery_driver")) == 1
    assert len(load_candidates("does_not_exist")) == 0
