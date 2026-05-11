"""FastAPI entrypoint — wires Chainlit ↔ LangGraph ↔ storage."""

import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import BaseModel

from .config import settings
from .graph import build_graph
from .inactivity import forget, start_sweeper, touch
from .models import ConversationState, Stage, Turn
from .nodes import build_nudge_text, terminate_node
from .storage import append_log


# Module-level state set during lifespan startup.
_runtime: dict = {}


# --------------------------------------------------------------------------------------
# Request / response models
# --------------------------------------------------------------------------------------

class CreateConvOut(BaseModel):
    conversation_id: str
    agent_text: str
    status: str
    source: Optional[Literal["llm", "template"]] = "template"


class MessageIn(BaseModel):
    text: str


class MessageOut(BaseModel):
    agent_text: str
    status: str
    disqualification_reason: Optional[str] = None
    source: Optional[Literal["llm", "template"]] = "llm"


# --------------------------------------------------------------------------------------
# Lifespan: open SQLite checkpointer, compile graph, start sweeper
# --------------------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.checkpoint_db_path.parent.mkdir(parents=True, exist_ok=True)

    saver_cm = SqliteSaver.from_conn_string(str(settings.checkpoint_db_path))
    saver = saver_cm.__enter__()
    _runtime["saver_cm"] = saver_cm
    _runtime["saver"] = saver
    _runtime["graph"] = build_graph(saver)

    _runtime["sweeper"] = start_sweeper(_finalize_abandoned)

    try:
        yield
    finally:
        sweeper = _runtime.get("sweeper")
        if sweeper:
            sweeper.cancel()
        saver_cm.__exit__(None, None, None)


app = FastAPI(title="screening_ai", version=settings.agent_version, lifespan=lifespan)


# --------------------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------------------

@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True, "version": settings.agent_version}


@app.post("/conversations", response_model=CreateConvOut)
def create_conversation() -> CreateConvOut:
    conv_id = str(uuid.uuid4())
    state = _invoke(conv_id, user_text=None)
    touch(conv_id)
    text, source = _last_agent_turn(state)
    return CreateConvOut(
        conversation_id=conv_id,
        agent_text=text,
        status=state.qualification_status,
        source=source or "template",
    )


@app.post("/conversations/{conversation_id}/messages", response_model=MessageOut)
def post_message(conversation_id: str, msg: MessageIn) -> MessageOut:
    state = _invoke(conversation_id, user_text=msg.text)
    if state.qualification_status == "in_progress":
        touch(conversation_id)
    else:
        forget(conversation_id)
    text, source = _last_agent_turn(state)
    return MessageOut(
        agent_text=text,
        status=state.qualification_status,
        disqualification_reason=state.disqualification_reason,
        source=source or "llm",
    )


@app.get("/conversations/{conversation_id}")
def get_conversation(conversation_id: str) -> dict:
    state = _load_state(conversation_id)
    if state is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    return state.model_dump(mode="json")


@app.post("/conversations/{conversation_id}/inactivity", response_model=MessageOut)
def post_inactivity(conversation_id: str) -> MessageOut:
    """Simulate a 10-minute candidate silence.

    First call → friendly nudge that re-asks the current stage question.
    Second call (no real reply in between) → terminate as abandoned.

    The background sweeper hits the same code path on real timeouts, so the
    simulation matches production behaviour.
    """
    return handle_inactivity_event(conversation_id)


# --------------------------------------------------------------------------------------
# Internals
# --------------------------------------------------------------------------------------

def _config_for(conversation_id: str) -> dict:
    return {"configurable": {"thread_id": conversation_id}}


def _invoke(conversation_id: str, user_text: Optional[str]) -> ConversationState:
    graph = _runtime["graph"]
    config = _config_for(conversation_id)

    # Build update: append user turn if present.
    update: dict = {"conversation_id": conversation_id}
    if user_text is not None:
        existing = _load_state(conversation_id)
        transcript = list(existing.transcript) if existing else []
        transcript.append(Turn(role="user", text=user_text, ts=datetime.utcnow()))
        update.update({"transcript": transcript, "last_user_ts": datetime.utcnow()})

    try:
        result = graph.invoke(update, config=config)
    except Exception as e:  # noqa: BLE001
        append_log(conversation_id, {"event": "graph_invoke_error", "error": str(e)})
        raise HTTPException(status_code=500, detail=f"graph error: {e}") from e

    return _coerce_state(result)


def _load_state(conversation_id: str) -> Optional[ConversationState]:
    graph = _runtime["graph"]
    snap = graph.get_state(_config_for(conversation_id))
    values = snap.values if snap else None
    if not values:
        return None
    return _coerce_state(values)


def _coerce_state(values) -> ConversationState:
    if isinstance(values, ConversationState):
        return values
    if isinstance(values, dict):
        return ConversationState(**values)
    raise TypeError(f"unexpected graph state type: {type(values)}")


def _last_agent_turn(state: ConversationState) -> tuple[str, Optional[str]]:
    """Return (text, source) for the most recent agent turn, or ('', None)."""
    for turn in reversed(state.transcript):
        if turn.role == "agent":
            return turn.text, turn.source
    return "", None


def handle_inactivity_event(conversation_id: str) -> MessageOut:
    """One inactivity 'strike'.

    First strike on an in-progress conversation: append a deterministic nudge
    that re-asks the current stage question, set `metadata.inactivity_nudges=1`,
    and reset the sweeper's idle timer (so a real candidate gets another full
    `INACTIVITY_TIMEOUT_SECONDS` after the nudge before being abandoned).

    Second strike (counter already >=1): transition to ABANDONED and run
    terminate_node directly so the closing message is written and the candidate
    JSON file lands in `data/candidates/abandoned/`.
    """
    state = _load_state(conversation_id)
    if state is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    if state.qualification_status != "in_progress":
        # Already terminal — no-op; surface the last agent turn so callers can
        # render it idempotently.
        text, source = _last_agent_turn(state)
        return MessageOut(
            agent_text=text,
            status=state.qualification_status,
            disqualification_reason=state.disqualification_reason,
            source=source or "template",
        )

    graph = _runtime["graph"]
    config = _config_for(conversation_id)
    nudges = state.metadata.get("inactivity_nudges", 0)

    if nudges == 0:
        # First strike — nudge and increment.
        nudge_text = build_nudge_text(state)
        transcript = [*state.transcript, Turn(role="agent", text=nudge_text, source="template")]
        graph.update_state(config, {
            "transcript": transcript,
            "metadata": {**state.metadata, "inactivity_nudges": 1},
        })
        # Reset the background sweeper's timer so we don't re-fire immediately.
        touch(conversation_id)
        append_log(conversation_id, {
            "event": "inactivity_nudge",
            "stage": state.current_stage.value,
            "language": state.language,
        })
        return MessageOut(
            agent_text=nudge_text,
            status="in_progress",
            source="template",
        )

    # Second strike — terminate as abandoned.
    _terminate_as_abandoned(conversation_id)
    final = _load_state(conversation_id)
    text, source = _last_agent_turn(final) if final else ("", None)
    append_log(conversation_id, {"event": "inactivity_abandoned"})
    forget(conversation_id)
    return MessageOut(
        agent_text=text,
        status="abandoned",
        source=source or "template",
    )


def _terminate_as_abandoned(conversation_id: str) -> None:
    """Mark the conversation abandoned and run terminate_node directly.

    Bypasses graph.invoke because re-entering at the entry point would let
    route_node overwrite our `ABANDONED` stage. terminate_node is a pure
    function over state, so we call it ourselves and persist its update.
    """
    graph = _runtime["graph"]
    config = _config_for(conversation_id)
    graph.update_state(config, {
        "current_stage": Stage.ABANDONED,
        "qualification_status": "abandoned",
        "ended_at": datetime.utcnow(),
    })
    state = _load_state(conversation_id)
    if state is None:
        return
    update = terminate_node(state)
    graph.update_state(config, update)


def _finalize_abandoned(conversation_id: str) -> None:
    """Sweeper entry point. One sweeper tick == one inactivity strike."""
    handle_inactivity_event(conversation_id)
