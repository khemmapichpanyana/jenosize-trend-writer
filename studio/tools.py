"""The agent's tools. Each is built per turn with the thread's context bound in.

Tools emit live events through LangGraph's stream writer (`studio.events`), so
the console can show the article streaming into the artifact panel while the
fine-tuned model is still writing it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool, tool

from app.schemas.articles import ArticleRequest
from app.services.generation import GenerationService
from studio import html as page
from studio.events import emit
from studio.store import StudioStore

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

        image_asset_ids: ids of images the user uploaded (see list_images) to
        place on the page. instructions: optional layout wishes.
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
        """List the images the user has uploaded in this chat (ids, names, sizes)."""
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

    return [write_article, design_page, list_images, get_artifact]
