"""The agent's tools. Each is built per turn with the thread's context bound in.

Tools emit live events through LangGraph's stream writer (`studio.events`), so
the console can show the article streaming into the artifact panel while the
fine-tuned model is still writing it.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import re
from dataclasses import dataclass
from typing import Any, cast
from uuid import UUID, uuid4

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool, tool

from app.core.config import Settings
from app.core.logging import get_logger
from app.schemas.articles import ArticleRequest
from app.services.generation import GenerationService
from app.storage.base import Storage
from studio import html as page
from studio.events import emit
from studio.store import StudioStore

logger = get_logger(__name__)

DESIGN_SYSTEM = """You lay out a finished article as an HTML page body for Jenosize Ideas.

Output ONLY the HTML body fragment — no <html>, <head>, <body>, <style> or
<script>, no markdown fences. Do not change the article's wording except to
split it into the elements below.

Available classes (defined by the brand theme; use no others):
- <h1> for the headline, then <p class="jz-standfirst"> for the opening paragraph
- <figure class="jz-hero"><img src="asset://ID" alt="..."><figcaption>..</figcaption></figure> for a lead image
- <h2>/<h3> for sections, <p>, <ul>/<ol>, <strong>, <em>
- <figure class="jz-figure"> for an inline image, <blockquote class="jz-quote"> for ONE pull quote
- <div class="jz-callout"> for a key takeaway, <section class="jz-actions"> for the closing next steps

Images: use ONLY the asset:// URLs listed in the request, each at most once,
with descriptive alt text. Never invent URLs.

Brand context:
"""


@dataclass
class ToolContext:
    thread_id: UUID
    store: StudioStore
    generation: GenerationService
    designer: BaseChatModel
    settings: Settings
    storage: Storage


def _summary(**fields: Any) -> str:
    return json.dumps(fields, default=str, ensure_ascii=False)


def build_tools(ctx: ToolContext) -> list[BaseTool]:
    @tool
    async def write_article(
        topic: str,
        industry: str | None = None,
        audience: str | None = None,
        keywords: list[str] | None = None,
        length: str = "medium",
        category: str | None = None,
        source_url: str | None = None,
        artifact_id: str | None = None,
    ) -> str:
        """Write (or rewrite) a Jenosize Ideas article with the fine-tuned writer model.

        ALWAYS use this to produce article text — never write articles yourself.
        length: short (~600 words) | medium (~1000) | long (~1500).
        source_url: optional page to ground facts in.
        artifact_id: pass the existing artifact's id to save a new version of it
        instead of creating a new artifact.
        """
        request = ArticleRequest(
            topic=topic,
            industry=industry,
            audience=audience,
            keywords=keywords or [],
            length=length if length in ("short", "medium", "long") else "medium",  # type: ignore[arg-type]
            category=category,
            source_url=source_url,  # type: ignore[arg-type]
        )
        if artifact_id:
            artifact = await ctx.store.get_artifact(UUID(artifact_id))
            if artifact is None or artifact["thread_id"] != ctx.thread_id:
                return _summary(error=f"no artifact {artifact_id} in this chat")
        else:
            artifact = await ctx.store.create_artifact(ctx.thread_id, title=topic)
        emit("artifact_start", artifact_id=str(artifact["id"]), title=topic, writer="fine-tuned")

        result: dict[str, Any] | None = None
        async for event, payload in ctx.generation.stream(request):
            if event == "token":
                emit("artifact_delta", artifact_id=str(artifact["id"]), text=payload["text"])
            elif event == "status":
                emit(
                    "tool_progress",
                    tool="write_article",
                    message=payload.get("message") or payload.get("stage"),
                )
            elif event == "result":
                result = payload
            elif event == "error":
                emit(
                    "artifact_error",
                    artifact_id=str(artifact["id"]),
                    message=payload.get("message"),
                )
                return _summary(error=payload.get("message", "generation failed"))

        if not result:
            return _summary(error="the writer returned no article")
        markdown = f"# {result['title']}\n\n{result['article_markdown']}"
        version = await ctx.store.add_version(
            artifact["id"],
            markdown=markdown,
            html=page.markdown_to_html(markdown),
            meta={
                "title": result["title"],
                "meta_description": result.get("meta_description"),
                "quality_report": result.get("quality_report"),
                "generation_id": result.get("id"),
                "model_version": result.get("model_version"),
            },
            note="written by the fine-tuned model",
            title=result["title"],
        )
        emit(
            "artifact",
            artifact_id=str(artifact["id"]),
            version=version["version"],
            title=result["title"],
        )
        quality = result.get("quality_report") or {}
        return _summary(
            artifact_id=artifact["id"],
            version=version["version"],
            title=result["title"],
            word_count=quality.get("word_count"),
            quality_passed=quality.get("passed"),
            quality_warnings=quality.get("warnings", [])[:4],
        )

    @tool
    async def design_page(
        artifact_id: str, instructions: str = "", image_asset_ids: list[str] | None = None
    ) -> str:
        """Lay out the artifact's current article as a branded Jenosize HTML page.

        image_asset_ids: ids of images available in this chat (see list_images)
        to place on the page. instructions: optional layout wishes.
        Saves a new artifact version that the user can preview and publish.
        """
        artifact = await ctx.store.get_artifact(UUID(artifact_id))
        if artifact is None or artifact["thread_id"] != ctx.thread_id:
            return _summary(error=f"no artifact {artifact_id} in this chat")
        current = await ctx.store.get_version(artifact["id"])
        if current is None or not current.get("markdown"):
            return _summary(error="the artifact has no article yet; call write_article first")

        wanted = {i.lower() for i in (image_asset_ids or [])}
        images = [a for a in await ctx.store.list_assets(ctx.thread_id) if str(a["id"]) in wanted]
        image_lines = (
            "\n".join(
                f"- asset://{a['id']} — {a['filename']} ({a.get('width')}x{a.get('height')})"
                for a in images
            )
            or "- (no images: do not add any <img>)"
        )

        emit("tool_progress", tool="design_page", message="Designing the page…")
        body = ""
        try:
            reply = await ctx.designer.ainvoke(
                [
                    SystemMessage(DESIGN_SYSTEM + page.brand_context()),
                    HumanMessage(
                        f"Layout wishes: {instructions or 'none'}\n\nImages you may use:\n{image_lines}\n\n"
                        f"Article (markdown):\n\n{current['markdown']}"
                    ),
                ]
            )
            body = page.sanitize(page.strip_code_fence(str(reply.content)))
        except Exception as exc:  # designer failure must not lose the article
            emit(
                "tool_progress",
                tool="design_page",
                message=f"Designer failed ({type(exc).__name__}); using the standard layout",
            )

        # Reject a design that dropped most of the article or invented images.
        allowed_refs = {a["id"] for a in images}
        design_ok = (
            len(body) > 0.5 * len(page.markdown_to_html(current["markdown"]))
            and set(page.asset_refs(body)) <= allowed_refs
        )
        if not design_ok:
            hero = ""
            if images:
                first = images[0]
                hero = f'<figure class="jz-hero"><img src="asset://{first["id"]}" alt="{first["filename"]}"></figure>'
            body = page.sanitize(hero + page.markdown_to_html(current["markdown"]))

        version = await ctx.store.add_version(
            artifact["id"],
            markdown=current["markdown"],
            html=body,
            meta={
                **current["meta"],
                "designed": design_ok,
                "asset_ids": [str(a) for a in page.asset_refs(body)],
            },
            note="designed page" if design_ok else "standard layout (designer output rejected)",
        )
        emit(
            "artifact",
            artifact_id=str(artifact["id"]),
            version=version["version"],
            title=artifact["title"],
        )
        return _summary(
            artifact_id=artifact["id"],
            version=version["version"],
            layout="designed" if design_ok else "standard",
            images_used=len(page.asset_refs(body)),
        )

    @tool
    async def list_images() -> str:
        """List images available in this chat (uploaded or generated)."""
        assets = await ctx.store.list_assets(ctx.thread_id)
        return _summary(
            images=[
                {
                    "id": a["id"],
                    "filename": a["filename"],
                    "width": a["width"],
                    "height": a["height"],
                }
                for a in assets
            ]
        )

    @tool
    async def generate_image(
        prompt: str,
        filename: str = "jenosize-generated.png",
        alt_text: str | None = None,
        size: str | None = None,
    ) -> str:
        """Generate an editorial image and save it as a thread asset.

        Use this when the user asks for a hero or supporting image and has not
        supplied a suitable upload. Pass the returned asset_id to
        `design_page` in `image_asset_ids`.
        """
        if not ctx.settings.image_api_key:
            return _summary(error="image generation is not configured (set OPENAI_API_KEY)")

        emit("tool_progress", tool="generate_image", message="Generating an editorial image…")
        try:
            from openai import AsyncOpenAI

            async with AsyncOpenAI(
                api_key=ctx.settings.image_api_key,
                base_url=ctx.settings.image_base_url.rstrip("/"),
                timeout=ctx.settings.image_timeout_s,
                max_retries=2,
            ) as client:
                result = await client.images.generate(
                    model=cast(Any, ctx.settings.image_model),
                    prompt=prompt[:32_000],
                    size=cast(Any, size or ctx.settings.image_size),
                    quality=cast(Any, ctx.settings.image_quality),
                )
            items = result.data or []
            if not items:
                return _summary(error="image provider returned no image bytes")
            item = items[0]
            encoded = getattr(item, "b64_json", None)
            if not encoded:
                return _summary(error="image provider returned no image bytes")
            data = base64.b64decode(encoded)
        except Exception as exc:
            logger.exception("image_generation_failed", extra={"thread_id": str(ctx.thread_id)})
            return _summary(error=f"image generation failed: {type(exc).__name__}")

        from PIL import Image

        try:
            with Image.open(io.BytesIO(data)) as image:
                image.verify()
            with Image.open(io.BytesIO(data)) as image:
                width, height = image.size
                content_type = Image.MIME.get(image.format or "PNG", "image/png")
        except Exception as exc:
            return _summary(error=f"generated image failed validation: {type(exc).__name__}")

        asset_id = uuid4()
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "-", filename).strip(".-") or "generated.png"
        if "." not in safe_name:
            safe_name += ".png"
        key = f"assets/{asset_id}/{safe_name}"
        await ctx.storage.put_bytes(key, data, content_type=content_type)
        asset = await ctx.store.add_asset(
            id=asset_id,
            thread_id=ctx.thread_id,
            filename=safe_name,
            content_type=content_type,
            bytes=len(data),
            width=width,
            height=height,
            sha256=hashlib.sha256(data).hexdigest(),
            r2_key=key,
        )
        emit(
            "asset",
            asset_id=str(asset_id),
            filename=safe_name,
            width=width,
            height=height,
            generated=True,
        )
        return _summary(
            asset_id=asset["id"],
            filename=safe_name,
            width=width,
            height=height,
            generated=True,
            alt_text=alt_text or prompt[:180],
        )

    @tool
    async def get_artifact(artifact_id: str) -> str:
        """Read an artifact's current version (title, version number, article excerpt)."""
        artifact = await ctx.store.get_artifact(UUID(artifact_id))
        if artifact is None or artifact["thread_id"] != ctx.thread_id:
            return _summary(error=f"no artifact {artifact_id} in this chat")
        current = await ctx.store.get_version(artifact["id"])
        return _summary(
            artifact_id=artifact["id"],
            title=artifact["title"],
            version=artifact["current_version"],
            excerpt=(current or {}).get("markdown", "")[:1500],
        )

    @tool
    async def web_search(query: str, max_results: int = 5) -> str:
        """Search the web for current facts, stats or named sources to research a topic.

        Call this (several times in parallel with different angles, if useful)
        BEFORE write_article whenever the brief needs current data, statistics,
        named examples or facts you are not already confident about. Cite what
        you find by passing the most useful result's url as write_article's
        source_url, or by mentioning sources in your reply.
        max_results: how many results to return (1-8).
        """
        if not ctx.settings.tavily_api_key:
            return _summary(error="web search is not configured (set TAVILY_API_KEY)")

        from tavily import AsyncTavilyClient

        emit("tool_progress", tool="web_search", message=f"Searching: {query}")
        client = AsyncTavilyClient(api_key=cast(str, ctx.settings.tavily_api_key))
        try:
            response = await client.search(
                query,
                max_results=max(1, min(max_results, 8)),
                include_answer=True,
            )
        except Exception as exc:
            return _summary(error=f"web search failed: {type(exc).__name__}: {exc}")
        results = [
            {
                "title": r.get("title"),
                "url": r.get("url"),
                "snippet": (r.get("content") or "")[:600],
            }
            for r in response.get("results", [])
        ]
        return _summary(query=query, answer=response.get("answer"), results=results)

    return [write_article, design_page, list_images, generate_image, get_artifact, web_search]
