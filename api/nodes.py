"""LangGraph node functions.

The graph is intentionally tiny: extract → route → render → END (or terminate → END).
Per-turn flow:
  1. extract:   classify the latest user message; merge extracted fields into state.candidate
  2. route:     advance current_stage based on what's now filled, apply hard disqualifiers
  3. render:    produce the next agent message (or a terminal message in terminate_node)

NOTE: This is scaffold-quality. Prompt edge cases, sentiment aggregation policy, and the
exact "no_availability" detection rule are deliberately simple — refine as we iterate.
"""

import json
from datetime import datetime
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage

from .config import settings
from .jd_loader import JD, load_jd
from .llm import get_llm
from .models import (
    CandidateRecord,
    ConversationState,
    SentimentSummary,
    Stage,
    Turn,
)
from .prompts import (
    EXTRACTION_SYSTEM,
    INACTIVITY_NUDGE_PREFIX,
    OTHER_LANGUAGE_NUDGE,
    QA_FALLBACK,
    QA_SYSTEM,
    RENDER_SYSTEM,
    STAGE_QUESTIONS,
    TERMINAL_MESSAGES,
    looks_like_refusal,
)
from .storage import append_log, append_notification, write_candidate


_jd: Optional[JD] = None
_jd_context_cached: Optional[str] = None


def get_jd() -> JD:
    global _jd
    if _jd is None:
        _jd = load_jd(settings.jd_path)
    return _jd


def _build_jd_context() -> str:
    """Build the candidate-facing JD text used for Q&A.

    Includes user-safe fields only: role title, employment types, shifts,
    service areas, compensation, perks, requirements, and the body. Excludes
    operational metadata (contact, agent_version_compat, job_id).
    """
    global _jd_context_cached
    if _jd_context_cached is not None:
        return _jd_context_cached
    jd = get_jd()
    parts: list[str] = [
        f"Role: {jd.title} at {jd.company}",
        f"Employment types: {', '.join(jd.employment_types)}",
        f"Shifts: {', '.join(jd.shifts)}",
    ]
    if jd.compensation:
        for region, comp in jd.compensation.items():
            lo, hi = (comp.get("hourly_range") or [None, None])[:2]
            cur = comp.get("currency", "")
            note = comp.get("notes", "")
            parts.append(f"Compensation ({region}): {lo}-{hi} {cur}/hour. {note}".strip())
    if jd.service_areas:
        parts.append("Service areas:")
        for region, cities in jd.service_areas.items():
            parts.append(f"  {region}: {', '.join(cities)}")
    if jd.perks:
        parts.append("Perks: " + "; ".join(jd.perks))
    if jd.requirements:
        parts.append("Requirements: " + "; ".join(jd.requirements))
    if jd.body:
        parts.append("")
        parts.append("--- DESCRIPTION ---")
        parts.append(jd.body)
    _jd_context_cached = "\n".join(parts)
    return _jd_context_cached


def build_nudge_text(state: ConversationState) -> str:
    """Compose the 'are you still there?' inactivity nudge for the current stage.

    Uses the deterministic STAGE_QUESTIONS template prefixed with the
    language-appropriate "are you still there?" line. If the stage doesn't have
    a template (e.g. brand-new conversation still on GREETING) we default to
    asking the name — that's the natural first ask.
    """
    lang = state.language if state.language in ("es", "en") else "es"
    prefix = INACTIVITY_NUDGE_PREFIX[lang]
    stage = state.current_stage.value
    question = STAGE_QUESTIONS.get((stage, lang)) or STAGE_QUESTIONS[("ask_name", lang)]
    return f"{prefix}{question}"


def _answer_from_jd(question: str, language: str, conversation_id: str) -> str:
    """Answer a candidate's question strictly from the JD. Injection-resistant."""
    lang = language if language in ("es", "en") else "es"
    prompt = QA_SYSTEM.format(
        language=lang,
        jd_text=_build_jd_context(),
        question=question,
    )
    text = ""
    try:
        resp = get_llm(temperature=0).invoke([SystemMessage(content=prompt)])
        text = (resp.content if isinstance(resp.content, str) else str(resp.content)).strip().strip('"')
    except Exception as e:  # noqa: BLE001
        append_log(conversation_id, {"event": "qa_error", "error": str(e)})

    if looks_like_refusal(text):
        text = QA_FALLBACK.get(lang, QA_FALLBACK["en"])

    append_log(conversation_id, {
        "event": "qa_answered",
        "language": lang,
        "question": question[:200],
        "answer": text[:500],
    })
    return text


# --------------------------------------------------------------------------------------
# extract
# --------------------------------------------------------------------------------------

def extract_node(state: ConversationState) -> dict:
    """Run the extraction LLM call against the latest user message."""
    user_text = _latest_user_text(state)
    if not user_text:
        # First turn — no user message yet. Nothing to extract.
        return {}

    # A real user message arrived → reset the inactivity-nudge counter regardless
    # of what extraction returns. The candidate is clearly still engaged.
    base_meta = {**state.metadata, "inactivity_nudges": 0}

    llm = get_llm(temperature=0)
    raw = ""
    try:
        resp = llm.invoke([
            SystemMessage(content=EXTRACTION_SYSTEM),
            HumanMessage(content=user_text),
        ])
        raw = resp.content if isinstance(resp.content, str) else str(resp.content)
        data = _safe_json(raw)
    except Exception as e:  # noqa: BLE001 — log and continue with empty extraction
        append_log(state.conversation_id, {"event": "extract_error", "error": str(e)})
        return {"metadata": base_meta}

    if not data:
        append_log(state.conversation_id, {"event": "extract_parse_error", "raw": raw[:500]})
        return {"metadata": base_meta}

    # Merge new field values into the existing CandidateRecord.
    candidate = state.candidate.model_dump()
    for key, value in (data.get("fields") or {}).items():
        if value in (None, "", []):
            continue
        candidate[key] = value

    # Language: only honor es/en. "other" preserves previous language.
    new_language = data.get("language")
    language = new_language if new_language in ("es", "en") else state.language

    # Tag the most recent user turn with this sentiment.
    transcript = [t.model_copy() for t in state.transcript]
    sentiment = data.get("sentiment")
    if transcript and transcript[-1].role == "user" and sentiment in {"positive", "neutral", "frustrated", "confused"}:
        transcript[-1].sentiment = sentiment

    # Capture a candidate-asked question (if any) for the QA path in render_node.
    raw_question = data.get("user_question")
    user_question = raw_question.strip() if isinstance(raw_question, str) and raw_question.strip() else None

    append_log(state.conversation_id, {
        "event": "extract",
        "language": language,
        "sentiment": sentiment,
        "fields": data.get("fields"),
        "raw_language": new_language,
        "user_question": user_question,
    })

    return {
        "candidate": CandidateRecord(**candidate),
        "language": language,
        "transcript": transcript,
        "metadata": {
            **base_meta,
            "last_extract_language": new_language or state.language,
            "user_question": user_question,
        },
    }


# --------------------------------------------------------------------------------------
# route
# --------------------------------------------------------------------------------------

def route_node(state: ConversationState) -> dict:
    """Update current_stage and apply hard disqualifiers."""
    # Brand-new conversation: no turns yet. Keep at GREETING.
    if not state.transcript:
        return {}

    c = state.candidate
    jd = get_jd()

    # Hard disqualifiers — first match wins.
    if c.drivers_license is False:
        return _terminal(Stage.DISQUALIFIED, "no_license")
    if c.city_zone and not jd.is_in_service_area(c.city_zone):
        return _terminal(Stage.DISQUALIFIED, "outside_service_area")
    # "No availability at all" is hard to infer from absence alone. We treat it as a hard
    # disqualifier only if the extraction returns availability = [] AFTER we asked for it
    # twice (re-ask cap). For scaffold: only flag when reask_counts for ask_availability >= 2.
    if (
        state.current_stage == Stage.ASK_AVAILABILITY
        and not c.availability
        and state.reask_counts.get(Stage.ASK_AVAILABILITY.value, 0) >= 2
    ):
        return _terminal(Stage.DISQUALIFIED, "no_availability")

    # If everything's filled, qualify.
    if _all_required_filled(c):
        return _terminal(Stage.QUALIFIED)

    # Otherwise advance to the next missing field.
    next_stage = _next_missing_stage(c)
    reask_counts = dict(state.reask_counts)
    if next_stage == state.current_stage:
        reask_counts[next_stage.value] = reask_counts.get(next_stage.value, 0) + 1
    return {"current_stage": next_stage, "reask_counts": reask_counts}


def _next_missing_stage(c: CandidateRecord) -> Stage:
    if not c.full_name:
        return Stage.ASK_NAME
    if c.drivers_license is None:
        return Stage.ASK_LICENSE
    if not c.city_zone:
        return Stage.ASK_CITY
    if not c.availability:
        return Stage.ASK_AVAILABILITY
    if not c.preferred_schedule:
        return Stage.ASK_SCHEDULE
    if c.experience_years is None:
        return Stage.ASK_EXPERIENCE
    if not c.start_date:
        return Stage.ASK_START_DATE
    return Stage.QUALIFIED


def _all_required_filled(c: CandidateRecord) -> bool:
    return all([
        c.full_name,
        c.drivers_license is True,
        c.city_zone,
        c.availability,
        c.preferred_schedule,
        c.experience_years is not None,
        c.start_date,
    ])


def _terminal(stage: Stage, reason: Optional[str] = None) -> dict:
    return {
        "current_stage": stage,
        "qualification_status": stage.value,
        "disqualification_reason": reason,
        "ended_at": datetime.utcnow(),
    }


# --------------------------------------------------------------------------------------
# render
# --------------------------------------------------------------------------------------

def render_node(state: ConversationState) -> dict:
    """Generate the next agent message via the render LLM call."""
    # Unsupported-language nudge takes precedence (route doesn't handle this).
    last_lang = state.metadata.get("last_extract_language")
    if last_lang == "other":
        return _append_agent(state, OTHER_LANGUAGE_NUDGE, source="template")

    # First turn: send the bilingual greeting deterministically (no LLM cost).
    if state.current_stage == Stage.GREETING and not state.transcript:
        text = (
            "¡Hola! Gracias por tu interés en la vacante de repartidor de Grupos Sazón. "
            "Hi! Thanks for your interest in the Grupos Sazón delivery driver role. "
            "¿Empezamos con unas preguntas rápidas? / Shall we start with a few quick questions?"
        )
        return _append_agent(state, text, source="template")

    # If the candidate's last message contained a question about the role,
    # answer it (strictly from the JD) before asking the next stage question.
    qa_answer = ""
    user_question = state.metadata.get("user_question")
    if user_question:
        qa_answer = _answer_from_jd(user_question, state.language, state.conversation_id)

    # Standard render: ask the question for the current stage in the current language.
    llm = get_llm(temperature=0.4)
    reask_count = state.reask_counts.get(state.current_stage.value, 0)
    prompt = RENDER_SYSTEM.format(
        language=state.language,
        stage=state.current_stage.value,
        candidate=json.dumps(state.candidate.model_dump(), ensure_ascii=False),
        user_text=_latest_user_text(state) or "",
        reask_count=reask_count,
        is_reask=reask_count > 0,
    )
    llm_error: Optional[str] = None
    try:
        resp = llm.invoke([SystemMessage(content=prompt)])
        text = (resp.content if isinstance(resp.content, str) else str(resp.content)).strip().strip('"')
    except Exception as e:  # noqa: BLE001
        llm_error = f"{type(e).__name__}: {e}"
        text = ""

    # Fallback to deterministic template if LLM returned empty / refused.
    source = "llm"
    if looks_like_refusal(text):
        template = STAGE_QUESTIONS.get((state.current_stage.value, state.language))
        if template:
            append_log(state.conversation_id, {
                "event": "render_fallback",
                "stage": state.current_stage.value,
                "language": state.language,
                "raw_text": text[:200],
                "error": llm_error,
            })
            text = template
            source = "template"
        elif llm_error:
            append_log(state.conversation_id, {"event": "render_error", "error": llm_error})
            text = "Disculpa, ¿podrías repetir? / Sorry, could you repeat?"
            source = "template"

    # Combine: Q&A answer first, then the next stage question.
    if qa_answer:
        text = f"{qa_answer}\n\n{text}"
        # source stays whatever the next-question path produced; the QA call
        # itself is an LLM call, so source="llm" is also accurate here.
        if source == "template":
            source = "llm"

    # Clear the question from metadata so the next turn doesn't re-answer it.
    return _append_agent(state, text, source=source, metadata={"user_question": None})


# --------------------------------------------------------------------------------------
# terminate
# --------------------------------------------------------------------------------------

def terminate_node(state: ConversationState) -> dict:
    """Render the closing message, summarize sentiment, persist outputs."""
    # Compose terminal text.
    key_status = state.qualification_status
    if key_status == "disqualified" and state.disqualification_reason:
        key = f"disqualified_{state.disqualification_reason}"
    else:
        key = key_status
    name = state.candidate.full_name or ("" if state.language == "en" else "")
    template = TERMINAL_MESSAGES.get((key, state.language), "")
    closing_text = template.format(name=name).strip() if template else ""

    transcript = [t.model_copy() for t in state.transcript]
    if closing_text:
        transcript.append(Turn(role="agent", text=closing_text, source="template"))

    # Sentiment summary: simple rule — "frustrated" if any frustrated turn, else majority.
    sentiments = [t.sentiment for t in transcript if t.role == "user" and t.sentiment]
    overall = "neutral"
    if sentiments:
        if "frustrated" in sentiments:
            overall = "frustrated"
        elif sentiments.count("confused") >= max(2, len(sentiments) // 3):
            overall = "confused"
        elif sentiments.count("positive") > sentiments.count("neutral"):
            overall = "positive"
    sentiment = SentimentSummary(
        overall=overall,
        per_turn=[
            {"turn": i, "label": t.sentiment}
            for i, t in enumerate(transcript)
            if t.role == "user" and t.sentiment
        ],
    )

    update = {
        "transcript": transcript,
        "sentiment": sentiment,
        "ended_at": state.ended_at or datetime.utcnow(),
    }
    final_state = state.model_copy(update=update)
    path = write_candidate(final_state)
    append_notification(final_state)
    append_log(state.conversation_id, {
        "event": "terminate",
        "status": state.qualification_status,
        "reason": state.disqualification_reason,
        "file": str(path),
    })
    return update


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------

def _latest_user_text(state: ConversationState) -> Optional[str]:
    for t in reversed(state.transcript):
        if t.role == "user":
            return t.text
    return None


def _append_agent(
    state: ConversationState,
    text: str,
    source: str = "llm",
    metadata: Optional[dict] = None,
) -> dict:
    transcript = [t.model_copy() for t in state.transcript]
    transcript.append(Turn(role="agent", text=text, source=source))
    update: dict = {"transcript": transcript}
    if metadata is not None:
        update["metadata"] = {**state.metadata, **metadata}
    return update


def _safe_json(raw: str) -> dict:
    """Tolerant JSON parser: strips markdown fences and trailing prose."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json\n"):
            raw = raw[len("json\n"):]
    # Find the outermost {...}
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1:
        return {}
    try:
        return json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        return {}
