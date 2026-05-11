"""Chainlit frontend — thin client that calls the FastAPI screening API.

The agent / LLM logic lives entirely in `api/`. This module only:
  1. Creates a conversation on chat start (and shows the greeting).
  2. Forwards user messages to /conversations/{id}/messages.
  3. Displays the agent's reply and a terminal hint when the screening ends.
"""

import os

import chainlit as cl
import httpx


API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
HTTP_TIMEOUT = 30.0


@cl.on_chat_start
async def on_chat_start() -> None:
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            resp = await client.post(f"{API_BASE_URL}/conversations")
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPError as e:
        await cl.Message(content=f"⚠️ Could not reach screening API: {e}").send()
        return

    cl.user_session.set("conversation_id", data["conversation_id"])
    cl.user_session.set("status", data.get("status", "in_progress"))
    author = "Template" if data.get("source") == "template" else "Assistant"
    await cl.Message(content=data["agent_text"], author=author).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    conv_id = cl.user_session.get("conversation_id")
    if not conv_id:
        await cl.Message(content="No active conversation. Refresh to start over.").send()
        return

    if cl.user_session.get("status") != "in_progress":
        await cl.Message(content="_(this screening has ended — refresh to start a new one)_").send()
        return

    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            resp = await client.post(
                f"{API_BASE_URL}/conversations/{conv_id}/messages",
                json={"text": message.content},
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPError as e:
        await cl.Message(content=f"⚠️ API error: {e}").send()
        return

    author = "Template" if data.get("source") == "template" else "Assistant"
    await cl.Message(content=data["agent_text"], author=author).send()
    cl.user_session.set("status", data.get("status", "in_progress"))

    if data.get("status") != "in_progress":
        status = data["status"]
        reason = data.get("disqualification_reason")
        suffix = f" ({reason})" if reason else ""
        await cl.Message(content=f"_(screening ended — status: **{status}**{suffix})_").send()
