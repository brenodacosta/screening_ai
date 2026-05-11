"""Background sweeper that finalizes conversations stuck `in_progress`.

We track last-activity timestamps in memory (lost on restart — acceptable for prototype).
On each tick the sweeper looks for stale conversations, forces them through the
`terminate_node` via a synthetic invocation, and writes the abandoned JSON.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Callable

from .config import settings
from .storage import append_log


_last_activity: dict[str, datetime] = {}


def touch(conversation_id: str) -> None:
    _last_activity[conversation_id] = datetime.utcnow()


def forget(conversation_id: str) -> None:
    _last_activity.pop(conversation_id, None)


def start_sweeper(finalize_abandoned: Callable[[str], None]) -> asyncio.Task:
    """Start the periodic sweeper.

    `finalize_abandoned(conversation_id)` is invoked for each stale conversation —
    the caller (main.py) owns graph access and writes the abandoned record.
    """
    async def _loop() -> None:
        await asyncio.sleep(settings.inactivity_sweep_interval_seconds)
        while True:
            try:
                threshold = datetime.utcnow() - timedelta(seconds=settings.inactivity_timeout_seconds)
                stale = [cid for cid, ts in list(_last_activity.items()) if ts < threshold]
                for cid in stale:
                    try:
                        finalize_abandoned(cid)
                        append_log(cid, {"event": "sweep_abandoned"})
                    except Exception as e:  # noqa: BLE001
                        append_log(cid, {"event": "sweep_finalize_error", "error": str(e)})
                    finally:
                        forget(cid)
            except Exception as e:  # noqa: BLE001
                append_log("_sweeper", {"event": "sweep_loop_error", "error": str(e)})
            await asyncio.sleep(settings.inactivity_sweep_interval_seconds)

    return asyncio.create_task(_loop(), name="inactivity-sweeper")
