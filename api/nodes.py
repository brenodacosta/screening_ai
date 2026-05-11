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
    OTHER_LANGUAGE_NUDGE,
    RENDER_SYSTEM,
    STAGE_QUESTIONS,
    TERMINAL_MESSAGES,
    looks_like_refusal,
)
from .storage import append_log, append_notification, write_candidate


_jd: Optional[JD] = None


def get_jd() -> JD:
    global _jd
    if _jd is None:
        _jd = load_jd(settings.jd_path)
    return _jd


# --------------------------------------------------------------------------------------
# extract
# --------------------------------------------------------------------------------------

def extract_node(state: ConversationState) -> dict:
    """Run the extraction LLM call against the latest user message."""
    user_text = _latest_user_text(state)
    if not user_text:
        # First turn — no user message yet. Nothing to extract.
        return {}

    llm = get_llm(temperature=0)
    try:
        resp = llm.invoke([
            SystemMessage(content=EXTRACTION_SYSTEM),
            HumanMessage(content=user_text),
        ])
        raw = resp.content if isinstance(resp.content, str) else str(resp.content)
        data = _safe_json(raw)
    except Exception as e:  # noqa: BLE001 — log and continue with empty extraction
        append_log(state.conversation_id, {"event": "extract_error", "error": str(e)})
        return {}

    if not data:
        append_log(state.conversation_id, {"event": "extract_parse_error", "raw": raw[:500]})
        return {}

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

    append_log(state.conversation_id, {
        "event": "extract",
        "language": language,
        "sentiment": sentiment,
        "fields": data.get("fields"),
        "raw_language": new_language,
    })

    return {
        "candidate": CandidateRecord(**candidate),
        "language": language,
        "transcript": transcript,
        # Stash a flag so route_node knows whether the user spoke an unsupported language.
        "metadata": {**state.metadata, "last_extract_language": new_language or state.language},
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

    return _append_agent(state, text, source=source)


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


def _append_agent(state: ConversationState, text: str, source: str = "llm") -> dict:
    transcript = [t.model_copy() for t in state.transcript]
    transcript.append(Turn(role="agent", text=text, source=source))
    return {"transcript": transcript}


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
