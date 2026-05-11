"""Smoke test: verify the configured LLM client can reach the configured endpoint.

Run with the venv's Python:
    .venv\\Scripts\\python.exe smoke_test.py

Prints config (without the PAT), sends a one-shot prompt, prints the reply or
the error type/message. Exit code 0 on success, 1 on failure.
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path


def main() -> int:
    # Run from screening_ai/ so pydantic-settings finds .env.
    os.chdir(Path(__file__).resolve().parent)
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    from api.config import settings  # noqa: PLC0415 (intentional after chdir)
    from api.llm import get_llm  # noqa: PLC0415

    pat = settings.github_pat or ""
    print("=== screening_ai smoke test ===")
    print(f"Base URL : {settings.llm_base_url}")
    print(f"Model    : {settings.model_name}")
    print(f"PAT set  : {bool(pat) and pat != 'replace-me'}")
    print(f"PAT len  : {len(pat)}")
    print(f"PAT prefix: {pat[:10] + '...' if len(pat) > 10 else '(too short)'}")
    print()

    if not pat or pat == "replace-me":
        print("FAIL: GITHUB_PAT is not set. Edit .env and retry.")
        return 1

    from api.llm import shutdown_llm  # noqa: PLC0415
    from langchain_core.messages import HumanMessage, SystemMessage  # noqa: PLC0415

    llm = get_llm(temperature=0)

    print("Sending: ping -> expecting 'pong'...")
    try:
        resp = llm.invoke([
            SystemMessage(content="Reply with exactly one word: pong. No punctuation, no extra text."),
            HumanMessage(content="ping"),
        ])
        content = resp.content if isinstance(resp.content, str) else str(resp.content)
        print(f"OK  : received {content!r}")
        return 0
    except Exception as e:  # noqa: BLE001
        print(f"FAIL: {type(e).__name__}: {e}")
        print()
        print("--- traceback ---")
        traceback.print_exc()
        return 1
    finally:
        shutdown_llm()


if __name__ == "__main__":
    sys.exit(main())
