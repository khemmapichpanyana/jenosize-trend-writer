"""Model providers.

`LLMProvider` is the seam between the API and whatever is generating text. The
API layer never imports an SDK directly, so:
  * graders run the whole service with `MODEL_PROVIDER=mock` and no accounts;
  * production points `MODEL_PROVIDER=openai_compatible` at the Modal vLLM server;
  * swapping vLLM for anything else OpenAI-shaped is an env-var change.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import AsyncIterator
from typing import Protocol, cast, runtime_checkable

from openai import APIError, APITimeoutError, AsyncOpenAI, AsyncStream
from openai.types.chat import ChatCompletionChunk

from app.core.config import Settings
from app.core.errors import UpstreamError
from app.core.logging import get_logger

logger = get_logger(__name__)

Message = dict[str, str]


class GenerationResult:
    """Text plus whatever usage the provider reported (mock reports nothing)."""

    __slots__ = ("completion_tokens", "model", "prompt_tokens", "text")

    def __init__(
        self,
        text: str,
        *,
        model: str,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
    ) -> None:
        self.text = text
        self.model = model
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


@runtime_checkable
class LLMProvider(Protocol):
    name: str

    async def complete(
        self, messages: list[Message], *, max_tokens: int = 2048
    ) -> GenerationResult:
        """Blocking generation."""
        ...

    def stream(self, messages: list[Message], *, max_tokens: int = 2048) -> AsyncIterator[str]:
        """Token-by-token generation."""
        ...

    async def warmup(self) -> None:
        """Cheap request whose only job is to wake a scaled-to-zero GPU."""
        ...


# --------------------------------------------------------------------------- #
# Mock
# --------------------------------------------------------------------------- #

_KEYWORDS_RE = re.compile(r"^SEO keywords: (.+)$", re.MULTILINE)
_TOPIC_RE = re.compile(r"^Topic: (.+)$", re.MULTILINE)
_WORDS_RE = re.compile(r"^Target length: about (\d+) words$", re.MULTILINE)
_AUDIENCE_RE = re.compile(r"^Target audience: (.+)$", re.MULTILINE)

_FILLER = (
    "The shift is already visible in operating metrics rather than in press releases.",
    "Budget follows attention, and attention has moved to measurable outcomes.",
    "Teams that instrument the change early compound their advantage quarter over quarter.",
    "The constraint is rarely technology; it is the decision rights around it.",
    "Pilots are cheap, but the integration work is where the value is unlocked.",
    "Regional buyers expect the same fluency they get from global platforms.",
)


class MockProvider:
    """Deterministic fake writer.

    It re-reads the structured brief out of the user prompt so the generated
    article actually contains the requested keywords and roughly the requested
    length — otherwise the quality gate and its retry path would never be
    exercised locally, and the mock would test nothing.
    """

    name = "mock"

    async def complete(
        self, messages: list[Message], *, max_tokens: int = 2048
    ) -> GenerationResult:
        return GenerationResult(self._render(messages), model="mock")

    async def stream(
        self, messages: list[Message], *, max_tokens: int = 2048
    ) -> AsyncIterator[str]:
        text = self._render(messages)
        # Chunk on whitespace to imitate real token streaming for the SSE path.
        for piece in re.findall(r"\S+\s*", text):
            yield piece

    async def warmup(self) -> None:
        return None

    def _render(self, messages: list[Message]) -> str:
        user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        topic = t.group(1).strip() if (t := _TOPIC_RE.search(user)) else "Business trends"
        audience = a.group(1).strip() if (a := _AUDIENCE_RE.search(user)) else "business leaders"
        keywords = (
            [k.strip() for k in m.group(1).split(",")] if (m := _KEYWORDS_RE.search(user)) else []
        )
        target = int(w.group(1)) if (w := _WORDS_RE.search(user)) else 1000

        # Seeded by the brief so the same request always yields the same article.
        seed = int(hashlib.sha256(user.encode()).hexdigest()[:8], 16)

        title = f"{topic[:48].strip()}: What Comes Next"
        meta = (
            f"A Jenosize view on {topic.lower()} — the signals that matter, "
            "what they change, and where to act first."
        )[:160]

        sections = [
            ("Why this matters now", keywords[:2]),
            (f"What is changing for {audience}", keywords[2:4]),
            ("Where the value shows up first", keywords[4:6]),
            ("How to act in the next two quarters", keywords[6:]),
        ]

        body = [
            f"{topic} has moved from a strategy-deck talking point to a line item "
            f"with an owner. This is a mock article produced by the deterministic "
            f"local provider, and it exists so the full pipeline can be exercised "
            f"without a GPU."
        ]
        words_per_section = max(60, target // len(sections))
        for i, (heading, section_keywords) in enumerate(sections):
            body.append(f"\n## {heading}\n")
            para: list[str] = []
            if section_keywords:
                para.append(
                    f"The practical question is how {', '.join(section_keywords)} "
                    f"changes the way work is funded and measured."
                )
            while len(" ".join(para).split()) < words_per_section:
                para.append(_FILLER[(seed + i + len(para)) % len(_FILLER)])
            body.append(" ".join(para))

        # Any keyword the section loop could not place is stated plainly, so
        # coverage is deterministic rather than luck-of-the-draw.
        leftover = [k for k in keywords if k.lower() not in " ".join(body).lower()]
        if leftover:
            body.append(f"\nRelated themes to track: {', '.join(leftover)}.")

        return f"TITLE: {title}\nMETA: {meta}\n---\n" + "\n".join(body).strip() + "\n"


# --------------------------------------------------------------------------- #
# OpenAI-compatible (Modal vLLM in production)
# --------------------------------------------------------------------------- #


class OpenAICompatibleProvider:
    """Talks to any OpenAI-compatible `/v1` endpoint."""

    name = "openai_compatible"

    def __init__(self, settings: Settings) -> None:
        if not settings.model_base_url:
            raise ValueError("MODEL_BASE_URL is required when MODEL_PROVIDER=openai_compatible")
        self._model = settings.model_name
        self._client = AsyncOpenAI(
            base_url=settings.model_base_url.rstrip("/"),
            # vLLM requires *some* key; a placeholder keeps the SDK from erroring
            # when the server was started without --api-key.
            api_key=settings.model_api_key or "not-needed",
            # A cold L4 container can take a couple of minutes to load weights,
            # hence the generous default timeout and no client-side retries
            # (retrying a cold start just queues another cold start).
            timeout=settings.model_timeout_s,
            max_retries=1,
        )

    async def complete(
        self, messages: list[Message], *, max_tokens: int = 2048
    ) -> GenerationResult:
        try:
            resp = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,  # type: ignore[arg-type]
                max_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9,
            )
        except (APIError, APITimeoutError) as exc:
            raise UpstreamError(f"Model endpoint failed: {exc}") from exc

        usage = resp.usage
        return GenerationResult(
            resp.choices[0].message.content or "",
            model=self._model,
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
        )

    async def stream(
        self, messages: list[Message], *, max_tokens: int = 2048
    ) -> AsyncIterator[str]:
        try:
            # `messages` needs a cast to the SDK's TypedDict, which in turn
            # defeats overload resolution on `stream=True` — hence the cast back.
            stream = cast(
                AsyncStream[ChatCompletionChunk],
                await self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,  # type: ignore[arg-type]
                    max_tokens=max_tokens,
                    temperature=0.7,
                    top_p=0.9,
                    stream=True,
                ),
            )
            async for event in stream:
                if not event.choices:
                    continue
                if delta := event.choices[0].delta.content:
                    yield delta
        except (APIError, APITimeoutError) as exc:
            raise UpstreamError(f"Model stream failed: {exc}") from exc

    async def warmup(self) -> None:
        """One-token completion: enough to trigger a container start, cheap if warm."""
        try:
            await self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
            )
        except (APIError, APITimeoutError) as exc:
            raise UpstreamError(f"Model warmup failed: {exc}") from exc


def build_provider(settings: Settings) -> LLMProvider:
    if settings.model_provider == "openai_compatible":
        return OpenAICompatibleProvider(settings)
    return MockProvider()


# --------------------------------------------------------------------------- #
# Output parsing
# --------------------------------------------------------------------------- #

_TITLE_LINE = re.compile(r"^\s*TITLE:\s*(.+)$", re.MULTILINE)
_META_LINE = re.compile(r"^\s*META:\s*(.+)$", re.MULTILINE)


def parse_article(raw: str) -> tuple[str, str, str]:
    """Split the `TITLE:/META:/---/body` contract into its three parts.

    Falls back gracefully: a small fine-tuned model will occasionally drop the
    header, and a missing header should degrade the response, not 500 it.
    """
    title_match = _TITLE_LINE.search(raw)
    meta_match = _META_LINE.search(raw)

    body = raw
    if "\n---" in raw:
        body = raw.split("\n---", 1)[1]
    elif title_match or meta_match:
        # No separator: drop the header lines we recognised.
        body = "\n".join(
            line for line in raw.splitlines() if not line.startswith(("TITLE:", "META:"))
        )
    body = body.lstrip("-\n ").strip()

    title = title_match.group(1).strip() if title_match else _first_heading(body)
    meta = meta_match.group(1).strip() if meta_match else _first_sentence(body)
    return title, meta, body


def _first_heading(body: str) -> str:
    for line in body.splitlines():
        if line.startswith("#"):
            return line.lstrip("# ").strip()
    return "Untitled"


def _first_sentence(body: str, limit: int = 160) -> str:
    text = re.sub(r"[#*`]", "", body).strip().replace("\n", " ")
    return text[:limit].rsplit(" ", 1)[0] if len(text) > limit else text
