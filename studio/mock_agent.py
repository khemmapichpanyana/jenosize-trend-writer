"""A deterministic stand-in for the agent's orchestrating model (AGENT_PROVIDER=mock).

Lets the whole studio run with no GPU and no LLM account — the same idea as
MODEL_PROVIDER=mock for the writer. It makes the tool calls a real model would
for the common request ("write about X", optionally with images), so the real
tools, streaming, artifacts and publishing are all exercised; it only replaces
the model's *decisions*, and says so in its replies.
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

_IMAGES = re.compile(r"\[attached images: (.+?)\]")
_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


class MockAgentModel(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "mock-agent"

    def bind_tools(self, tools: Any, **kwargs: Any) -> MockAgentModel:  # type: ignore[override]
        return self

    def _generate(
        self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kwargs: Any
    ) -> ChatResult:
        reply = self._decide(messages)
        reply.response_metadata = {"model_name": "mock-agent"}
        return ChatResult(generations=[ChatGeneration(message=reply)])

    def _decide(self, messages: list[BaseMessage]) -> AIMessage:
        # The page designer calls this model too (system prompt = layout brief):
        # answer with no markup so the tool falls back to the standard layout.
        if isinstance(messages[0], SystemMessage) and "HTML body fragment" in str(
            messages[0].content
        ):
            return AIMessage("")

        last_user = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
        turn = messages[messages.index(last_user) + 1 :] if last_user else []
        results = [json.loads(str(m.content)) for m in turn if isinstance(m, ToolMessage)]
        text = str(last_user.content) if last_user else ""
        images = _UUID.findall(m.group(1)) if (m := _IMAGES.search(text)) else []
        topic = (
            _IMAGES.sub("", text).strip().splitlines()[0][:180]
            if text.strip()
            else "Business trends"
        )

        if not results:
            return AIMessage(
                "",
                tool_calls=[
                    {
                        "name": "write_article",
                        "args": {"topic": topic, "length": "short"},
                        "id": "mock-write",
                    }
                ],
            )
        written = results[0]
        if "error" in written:
            return AIMessage(f"(mock agent) The writer failed: {written['error']}")
        if len(results) == 1:
            return AIMessage(
                "",
                tool_calls=[
                    {
                        "name": "design_page",
                        "args": {
                            "artifact_id": str(written["artifact_id"]),
                            "image_asset_ids": images,
                        },
                        "id": "mock-design",
                    }
                ],
            )
        return AIMessage(
            f"(mock agent) Drafted “{written.get('title')}” with the writer model and laid it out as a "
            "Jenosize page — preview it on the right and press Publish when it's ready."
        )
