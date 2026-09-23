"""The Jenosize content agent: one chat turn, streamed as console events.

Event stream (what the console renders):

    thread       {id, title}                      first event of every turn
    token        {text}                           the agent's reply, token by token
    tool_start   {id, name, args, started_at}     a tool call began
    tool_progress{tool, message}                  a tool's own progress note
    artifact_start / artifact_delta {artifact_id, text} / artifact
                                                  an article streaming in from the
                                                  fine-tuned writer, then saved
    tool_end     {id, name, result, duration_ms}  a tool call finished
    message      {id, role, content, model}       the saved assistant message
    error        {message}
    done         {}
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ModelFallbackMiddleware,
    ToolCallLimitMiddleware,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool

from app.core.logging import get_logger
from studio import html as page
from studio.store import StudioStore

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are the Jenosize Ideas content agent. You help the team plan, write,
design and refine business trend articles.

How you work:
- When a brief needs current data, statistics, named examples or facts you
  are not already confident about, call `web_search` first — you may call it
  several times with different angles in the same turn to research in
  parallel. Ground the article in what you find (pass the best source's URL
  as write_article's source_url, or cite sources in your reply); never invent
  statistics or sources.
- Article text is written by Jenosize's fine-tuned writer model. ALWAYS call
  `write_article` to draft or rewrite an article; never write the article body
  yourself. Pass an `artifact_id` to revise an existing artifact.
- To turn an article into a branded web page, call `design_page`. Call
  `list_images` first when the user mentions images, and pass their ids.
- `write_article` also creates the article's hero image (GPT Image) and puts
  it at the top of the article, and `design_page` keeps it automatically. Call
  `generate_image` only when the user asks for an additional or different
  image, then pass its asset id to `design_page`.
- Ask one short clarifying question only when the topic is genuinely unclear;
  otherwise pick sensible defaults (medium length, business-leader audience).
- After a tool finishes, reply in 1-3 sentences: what you made and what the
  user can do next. Don't paste the article into the chat — it is in the panel.
- Publishing is the user's decision: tell them to use the Publish button.

Brand context:
"""

MAX_MODEL_CALLS = 14
# A research pass (several parallel web_search calls) plus write/design still fits.
MAX_TOOL_CALLS = 12
TITLE_CHARS = 60


def build_agent(
    primary: BaseChatModel, fallbacks: Sequence[BaseChatModel], tools: list[BaseTool]
) -> Any:
    middleware: list[Any] = [
        # A 4B orchestrator can loop; cap each turn so a bad run costs seconds.
        ModelCallLimitMiddleware(run_limit=MAX_MODEL_CALLS, exit_behavior="end"),
        ToolCallLimitMiddleware(run_limit=MAX_TOOL_CALLS, exit_behavior="end"),
    ]
    if fallbacks:
        middleware.insert(0, ModelFallbackMiddleware(*fallbacks))
    return create_agent(
        primary,
        tools=tools,
        system_prompt=SYSTEM_PROMPT + page.brand_context(),
        middleware=middleware,
    )


def history_to_messages(
    rows: list[dict[str, Any]], assets: dict[UUID, dict[str, Any]]
) -> list[BaseMessage]:
    """Rebuild the conversation for the model from saved messages.

    Tool traces are folded into the assistant text as a compact note: the model
    needs to know an artifact exists (and its id) to revise it, not the raw
    tool payloads.
    """
    messages: list[BaseMessage] = []
    for row in rows:
        if row["role"] == "user":
            text = row["content"]
            attached = [assets[a] for a in row.get("asset_ids") or [] if a in assets]
            if attached:
                text += (
                    "\n\n[attached images: "
                    + ", ".join(f"{a['id']} ({a['filename']})" for a in attached)
                    + "]"
                )
            messages.append(HumanMessage(text))
        else:
            notes = [
                f"[{c['name']} → {c.get('result', '')[:300]}]"
                for c in row.get("tool_calls") or []
                if c.get("name")
            ]
            messages.append(AIMessage("\n".join([*notes, row["content"]]).strip()))
    return messages


async def run_turn(
    *,
    agent: Any,
    store: StudioStore,
    thread_id: UUID,
    user_text: str,
    asset_ids: list[UUID],
    model_names: list[str],
    save_user_message: bool = True,
) -> AsyncIterator[dict[str, Any]]:
    """Stream the agent's work on one turn and persist the reply.

    `save_user_message=False` when the user message was already stored (the
    background-run path stores it when the turn is queued, so a reload shows it).
    """
    thread = await store.get_thread(thread_id)
    if thread is None:
        yield {"type": "error", "message": f"no chat {thread_id}"}
        return
    if save_user_message:
        await store.add_message(thread_id, "user", user_text, asset_ids=asset_ids)
    if _is_placeholder_title(thread["title"]):
        await store.touch_thread(
            thread_id, title=user_text.strip().splitlines()[0][:TITLE_CHARS] or "New chat"
        )
    yield {
        "type": "thread",
        "id": str(thread_id),
        "title": (await store.get_thread(thread_id) or thread)["title"],
    }

    assets = {a["id"]: a for a in await store.list_assets(thread_id)}
    history = history_to_messages(await store.list_messages(thread_id), assets)

    reply_parts: list[str] = []
    calls: dict[str, dict[str, Any]] = {}
    clocks: dict[str, float] = {}  # call id -> monotonic start, for durations
    model_used: str | None = None
    final_text = ""
    try:
        async for mode, chunk in agent.astream(
            {"messages": history}, stream_mode=["messages", "updates", "custom"]
        ):
            if mode == "messages":
                message, meta = chunk
                # Only the agent's own model node is the reply; tokens from LLM
                # calls *inside* tools (the page designer) are not chat text.
                # Streaming models send AIMessageChunks; non-streaming ones (or a
                # fallback without streaming) send one whole AIMessage.
                if meta.get("langgraph_node") == "model" and isinstance(
                    message, AIMessage | AIMessageChunk
                ):
                    text = message.content if isinstance(message.content, str) else ""
                    calling = getattr(message, "tool_call_chunks", None) or message.tool_calls
                    if text and not calling:
                        reply_parts.append(text)
                        yield {"type": "token", "text": text}
            elif mode == "custom":
                if isinstance(chunk, dict) and chunk.get("type") == "tool_progress":
                    # Keep the notes with the (latest) running call of that tool,
                    # so a saved turn shows the same step details as a live one.
                    running = [
                        c
                        for c in calls.values()
                        if c.get("name") == chunk.get("tool") and "result" not in c
                    ]
                    if running and chunk.get("message"):
                        notes = running[-1].setdefault("notes", [])
                        if not notes or notes[-1] != chunk["message"]:
                            notes.append(str(chunk["message"]))
                yield chunk
            elif mode == "updates":
                for _node, update in chunk.items():
                    for msg in (
                        (update or {}).get("messages", []) if isinstance(update, dict) else []
                    ):
                        if isinstance(msg, AIMessage):
                            model_used = msg.response_metadata.get("model_name") or model_used
                            if msg.tool_calls:
                                # Text before a tool call was a preamble, not the answer.
                                reply_parts.clear()
                            elif isinstance(msg.content, str) and msg.content.strip():
                                final_text = msg.content
                            for call in msg.tool_calls:
                                call_id = call["id"] or f"call-{len(calls)}"
                                started = datetime.now(UTC)
                                calls[call_id] = {
                                    "name": call["name"],
                                    "args": call["args"],
                                    "started_at": started.isoformat(),
                                }
                                clocks[call_id] = time.monotonic()
                                yield {
                                    "type": "tool_start",
                                    "id": call_id,
                                    "name": call["name"],
                                    "args": call["args"],
                                    "started_at": started.isoformat(),
                                }
                        elif isinstance(msg, ToolMessage):
                            result = str(msg.content)
                            entry = calls.setdefault(msg.tool_call_id, {"name": msg.name})
                            entry["result"] = result
                            if msg.tool_call_id in clocks:
                                entry["duration_ms"] = round(
                                    (time.monotonic() - clocks.pop(msg.tool_call_id)) * 1000
                                )
                            yield {
                                "type": "tool_end",
                                "id": msg.tool_call_id,
                                "name": entry.get("name"),
                                "result": _parse(result),
                                "duration_ms": entry.get("duration_ms"),
                            }
    except Exception as exc:
        logger.exception("agent_turn_failed", extra={"thread_id": str(thread_id)})
        yield {"type": "error", "message": f"{type(exc).__name__}: {exc}"[:500]}

    reply = (
        "".join(reply_parts).strip()
        or final_text.strip()
        or ("Done — see the panel." if calls else "Sorry, I couldn't produce a reply this time.")
    )
    saved = await store.add_message(
        thread_id,
        "assistant",
        reply,
        tool_calls=[{"id": k, **v} for k, v in calls.items()],
        model=model_used or (model_names[0] if model_names else None),
    )
    await store.touch_thread(thread_id)
    yield {
        "type": "message",
        "id": str(saved["id"]),
        "role": "assistant",
        "content": reply,
        "model": saved["model"],
    }
    yield {"type": "done"}


def _is_placeholder_title(title: str) -> bool:
    """Titles the console gives a thread before its first message; the first
    user brief replaces them. Includes older console defaults still in the DB."""
    return title in ("New chat", "Jenosize demo workspace") or title.startswith("New article — ")


def _parse(result: str) -> Any:
    try:
        return json.loads(result)
    except ValueError:
        return result
