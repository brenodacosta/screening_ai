from api.models import CandidateRecord, ConversationState, Stage, Turn


def test_default_state():
    s = ConversationState()
    assert s.current_stage == Stage.GREETING
    assert s.qualification_status == "in_progress"
    assert s.candidate.full_name is None
    assert s.transcript == []


def test_candidate_record_lists_default_empty():
    c = CandidateRecord()
    assert c.availability == []
    assert c.preferred_schedule == []
    assert c.experience_platforms == []


def test_state_serializes_round_trip():
    s = ConversationState()
    s.transcript.append(Turn(role="user", text="Hola"))
    blob = s.model_dump_json()
    s2 = ConversationState.model_validate_json(blob)
    assert s2.transcript[0].text == "Hola"
    assert s2.transcript[0].role == "user"
