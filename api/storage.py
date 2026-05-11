import json
import re
from datetime import datetime
from pathlib import Path

from .config import settings
from .models import ConversationState


def _slug(name: str | None) -> str:
    if not name:
        return "anon"
    s = re.sub(r"[^a-zA-Z0-9]+", "-", name.lower()).strip("-")
    return s[:40] or "anon"


def _subdir_for(status: str) -> str:
    if status in ("qualified", "disqualified", "abandoned"):
        return status
    return "abandoned"


def write_candidate(state: ConversationState) -> Path:
    sub = _subdir_for(state.qualification_status)
    out_dir = settings.data_dir / "candidates" / sub
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = (state.ended_at or datetime.utcnow()).strftime("%Y-%m-%dT%H-%M-%S")
    path = out_dir / f"{ts}_{_slug(state.candidate.full_name)}.json"
    path.write_text(state.model_dump_json(indent=2), encoding="utf-8")
    return path


def append_log(conversation_id: str, event: dict) -> None:
    log_dir = settings.data_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    event = {"ts": datetime.utcnow().isoformat(), **event}
    with (log_dir / f"{conversation_id}.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, default=str) + "\n")


def append_notification(state: ConversationState) -> None:
    """Recruiter feed — only qualified candidates land here."""
    if state.qualification_status != "qualified":
        return
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": datetime.utcnow().isoformat(),
        "conversation_id": state.conversation_id,
        "language": state.language,
        "candidate": state.candidate.model_dump(),
        "job_id": state.job_id,
    }
    with (settings.data_dir / "notifications.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")
