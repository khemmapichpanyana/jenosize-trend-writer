"""HTML for agent-authored pages: sanitising, image references, the page shell.

Pages are written by an LLM and then served publicly, so the model's markup is
treated as untrusted input: it is reduced to an allowlist of tags, attributes
and URL schemes before it is stored or served, and published pages also get a
Content-Security-Policy that forbids scripts outright.

Images are referenced as `asset://<uuid>` inside stored HTML and resolved to a
concrete URL only at render time. The same body can then point at the
console's authenticated preview route or at the public route of a published
page, without a rewrite pass over stored content.
"""

from __future__ import annotations

import html as html_lib
import re
from collections.abc import Callable
from pathlib import Path
from uuid import UUID

import markdown as md_lib
import nh3

BRAND_DIR = Path(__file__).resolve().parent / "brand"

ALLOWED_TAGS = {
    "section", "article", "header", "footer", "div", "span", "p", "br", "hr",
    "h1", "h2", "h3", "h4", "strong", "em", "b", "i", "u", "small", "sup", "sub",
    "blockquote", "q", "cite", "ul", "ol", "li", "a", "img", "figure", "figcaption",
    "table", "thead", "tbody", "tr", "th", "td", "code", "pre",
}  # fmt: skip
ALLOWED_ATTRIBUTES = {
    "*": {"class"},
    "a": {"href", "title"},
    "img": {"src", "alt", "width", "height", "loading"},
    "th": {"colspan", "rowspan"},
    "td": {"colspan", "rowspan"},
}
# `asset` is our own scheme, resolved at render time; nothing else gets through.
URL_SCHEMES = {"http", "https", "mailto", "asset"}

ASSET_REF = re.compile(r"asset://([0-9a-fA-F-]{36})")
SAFE_ASSET_BASE = re.compile(r"^/[A-Za-z0-9/_\-.]*$")

PAGE_CSP = (
    "default-src 'none'; img-src 'self' https: data:; style-src 'unsafe-inline'; "
    "font-src https: data:; base-uri 'none'; form-action 'none'; frame-ancestors *"
)


def brand_context() -> str:
    return (BRAND_DIR / "jenosize.md").read_text(encoding="utf-8")


def theme_css() -> str:
    return (BRAND_DIR / "theme.css").read_text(encoding="utf-8")


def sanitize(body: str) -> str:
    return nh3.clean(
        body,
        tags=ALLOWED_TAGS,
        attributes=ALLOWED_ATTRIBUTES,
        url_schemes=URL_SCHEMES,
        link_rel="noopener noreferrer",
        strip_comments=True,
    )


def strip_code_fence(text: str) -> str:
    """Models often wrap HTML in ```html fences despite being told not to."""
    fenced = re.search(r"```(?:html)?\s*(.*?)```", text, re.S | re.I)
    return (fenced.group(1) if fenced else text).strip()


def markdown_to_html(markdown: str) -> str:
    """The deterministic page body used when no designed HTML exists (or design fails)."""
    body = md_lib.markdown(markdown or "", extensions=["extra", "sane_lists"])
    return sanitize(body)


def asset_refs(body: str) -> list[UUID]:
    seen: dict[str, None] = {}
    for match in ASSET_REF.finditer(body or ""):
        seen.setdefault(match.group(1).lower(), None)
    return [UUID(a) for a in seen]


def resolve_assets(body: str, url_for: Callable[[UUID], str]) -> str:
    return ASSET_REF.sub(lambda m: url_for(UUID(m.group(1))), body)


def safe_asset_base(value: str | None, default: str) -> str:
    """Accept only a same-origin absolute path (e.g. the console's proxy route)."""
    if value and SAFE_ASSET_BASE.match(value) and "//" not in value:
        return value.rstrip("/")
    return default


def render_page(
    *,
    title: str,
    meta_description: str | None,
    body: str,
    asset_url: Callable[[UUID], str],
    canonical_url: str | None = None,
) -> str:
    """Wrap a sanitised body in the branded page shell, with images resolved."""
    esc = html_lib.escape
    description = esc(meta_description or "")
    og = ""
    if canonical_url:
        og = (
            f'<link rel="canonical" href="{esc(canonical_url)}">'
            f'<meta property="og:url" content="{esc(canonical_url)}">'
        )
    first_image = next(iter(asset_refs(body)), None)
    if first_image:
        og += f'<meta property="og:image" content="{esc(asset_url(first_image))}">'
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)} — Jenosize Ideas</title>
<meta name="description" content="{description}">
<meta property="og:title" content="{esc(title)}">
<meta property="og:description" content="{description}">
<meta property="og:type" content="article">
{og}
<style>{theme_css()}</style>
</head>
<body>
<header class="jz-bar"><div class="jz-bar__inner">
  <span class="jz-wordmark">Jeno<span>size</span></span>
  <span class="jz-tag">Ideas</span>
</div></header>
<main class="jz-article">
{resolve_assets(body, asset_url)}
</main>
<footer class="jz-foot"><div class="jz-foot__inner">Jenosize Ideas</div></footer>
</body>
</html>"""
