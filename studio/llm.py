"""Which chat models the agent uses, and in what order.

Primary: the orchestrating model on the Modal vLLM server (AGENT_MODEL, the base
Qwen weights, same server as the fine-tuned writer). Fallback: AGENT_FALLBACK_*
or, when unset, the labelling LLM, so that a cold or down GPU degrades to a
working agent instead of an error. Article *writing* never falls back here:
write_article always uses the fine-tuned model through GenerationService.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from app.core.config import Settings


@dataclass
class AgentModels:
    primary: BaseChatModel
    fallbacks: list[BaseChatModel] = field(default_factory=list)
    names: list[str] = field(default_factory=list)


class NoAgentModelError(RuntimeError):
    pass


def _chat(base_url: str, api_key: str | None, model: str, *, timeout: float) -> ChatOpenAI:
    return ChatOpenAI(
        base_url=base_url.rstrip("/"),
        api_key=api_key or "not-needed",  # type: ignore[arg-type]
        model=model,
        temperature=0.3,
        timeout=timeout,
        max_retries=1,
        streaming=True,
    )


def build_agent_models(settings: Settings) -> AgentModels:
    if settings.agent_provider == "mock":
        from studio.mock_agent import MockAgentModel

        return AgentModels(primary=MockAgentModel(), names=["mock-agent"])
    candidates: list[tuple[str, ChatOpenAI]] = []
    if settings.model_provider == "openai_compatible" and settings.model_base_url:
        candidates.append(
            (
                f"modal:{settings.agent_model}",
                # A cold L4 needs 1-3 min to answer; the fallback covers the gap.
                _chat(
                    settings.model_base_url,
                    settings.model_api_key,
                    settings.agent_model,
                    timeout=settings.model_timeout_s,
                ),
            )
        )
    fb_url = settings.agent_fallback_base_url or settings.labeler_base_url
    fb_model = settings.agent_fallback_model or settings.labeler_model
    fb_key = settings.agent_fallback_api_key or settings.labeler_api_key
    if fb_url and fb_model:
        candidates.append((f"fallback:{fb_model}", _chat(fb_url, fb_key, fb_model, timeout=120)))
    if not candidates:
        raise NoAgentModelError(
            "No agent model configured: set MODEL_PROVIDER=openai_compatible + MODEL_BASE_URL "
            "(Modal vLLM), and/or AGENT_FALLBACK_* / LABELER_*"
        )
    return AgentModels(
        primary=candidates[0][1],
        fallbacks=[m for _, m in candidates[1:]],
        names=[n for n, _ in candidates],
    )
