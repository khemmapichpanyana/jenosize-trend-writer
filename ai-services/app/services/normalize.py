"""Request normalization — the only fully-implemented business rule on Day 0.

Why this exists at all: the model is fine-tuned on a *consistent* prompt shape.
If "fintech", "FinTech" and "Fin-Tech" reach the prompt as three different
strings, the adapter sees three different conditioning signals and style drifts.
Normalizing here means the training-time and inference-time prompts match.
"""

from __future__ import annotations

import re

from app.core.errors import ValidationError
from app.schemas.articles import TARGET_WORDS, ArticleRequest, NormalizedParams

MAX_KEYWORDS = 10
DEFAULT_AUDIENCE = "Business leaders"

_WHITESPACE = re.compile(r"\s+")

# Canonical industry vocabulary. Kept small and explicit rather than fuzzy-matched:
# a wrong silent mapping is worse than passing the user's own wording through.
CANONICAL_INDUSTRIES: tuple[str, ...] = (
    "Financial Services",
    "Retail & E-commerce",
    "Healthcare",
    "Manufacturing",
    "Real Estate",
    "Technology",
    "Media & Entertainment",
    "Travel & Hospitality",
    "Education",
    "Energy & Utilities",
    "Logistics & Supply Chain",
    "Telecommunications",
    "Automotive",
    "Food & Beverage",
)

# Synonym -> canonical. Keys are compared lowercased with punctuation stripped.
INDUSTRY_SYNONYMS: dict[str, str] = {
    # Financial Services
    "fintech": "Financial Services",
    "finance": "Financial Services",
    "financial": "Financial Services",
    "financial services": "Financial Services",
    "banking": "Financial Services",
    "bank": "Financial Services",
    "insurance": "Financial Services",
    "insurtech": "Financial Services",
    "wealth management": "Financial Services",
    # Retail & E-commerce
    "ecommerce": "Retail & E-commerce",
    "e commerce": "Retail & E-commerce",
    "retail": "Retail & E-commerce",
    "retail e commerce": "Retail & E-commerce",
    "commerce": "Retail & E-commerce",
    "d2c": "Retail & E-commerce",
    "dtc": "Retail & E-commerce",
    "marketplace": "Retail & E-commerce",
    # Healthcare
    "healthcare": "Healthcare",
    "health care": "Healthcare",
    "health": "Healthcare",
    "healthtech": "Healthcare",
    "medtech": "Healthcare",
    "pharma": "Healthcare",
    "pharmaceutical": "Healthcare",
    "hospital": "Healthcare",
    # Manufacturing
    "manufacturing": "Manufacturing",
    "industrial": "Manufacturing",
    "factory": "Manufacturing",
    "industry 4 0": "Manufacturing",
    # Real Estate
    "real estate": "Real Estate",
    "property": "Real Estate",
    "proptech": "Real Estate",
    "construction": "Real Estate",
    # Technology
    "tech": "Technology",
    "technology": "Technology",
    "it": "Technology",
    "software": "Technology",
    "saas": "Technology",
    "ai": "Technology",
    "cloud": "Technology",
    # Media & Entertainment
    "media": "Media & Entertainment",
    "entertainment": "Media & Entertainment",
    "advertising": "Media & Entertainment",
    "marketing": "Media & Entertainment",
    "gaming": "Media & Entertainment",
    "publishing": "Media & Entertainment",
    # Travel & Hospitality
    "travel": "Travel & Hospitality",
    "tourism": "Travel & Hospitality",
    "hospitality": "Travel & Hospitality",
    "hotel": "Travel & Hospitality",
    "airline": "Travel & Hospitality",
    # Education
    "education": "Education",
    "edtech": "Education",
    "university": "Education",
    "learning": "Education",
    # Energy & Utilities
    "energy": "Energy & Utilities",
    "utilities": "Energy & Utilities",
    "oil and gas": "Energy & Utilities",
    "renewables": "Energy & Utilities",
    "power": "Energy & Utilities",
    # Logistics & Supply Chain
    "logistics": "Logistics & Supply Chain",
    "supply chain": "Logistics & Supply Chain",
    "shipping": "Logistics & Supply Chain",
    "freight": "Logistics & Supply Chain",
    "transportation": "Logistics & Supply Chain",
    # Telecommunications
    "telecom": "Telecommunications",
    "telco": "Telecommunications",
    "telecommunications": "Telecommunications",
    "mobile operator": "Telecommunications",
    # Automotive
    "automotive": "Automotive",
    "auto": "Automotive",
    "mobility": "Automotive",
    "ev": "Automotive",
    # Food & Beverage
    "f b": "Food & Beverage",
    "fnb": "Food & Beverage",
    "food": "Food & Beverage",
    "beverage": "Food & Beverage",
    "food and beverage": "Food & Beverage",
    "restaurant": "Food & Beverage",
    "qsr": "Food & Beverage",
}
# Canonical names map to themselves so round-tripping a normalized value is stable.
INDUSTRY_SYNONYMS.update({name.lower(): name for name in CANONICAL_INDUSTRIES})


def collapse_whitespace(value: str) -> str:
    """Trim and collapse every run of whitespace (incl. newlines/tabs) to one space."""
    return _WHITESPACE.sub(" ", value).strip()


def _synonym_key(value: str) -> str:
    """Lowercase and reduce punctuation to single spaces so "E-Commerce" == "ecommerce"."""
    lowered = collapse_whitespace(value).lower()
    return collapse_whitespace(re.sub(r"[^a-z0-9฀-๿]+", " ", lowered))


def normalize_keywords(keywords: list[str] | None) -> list[str]:
    """Lowercase, dedupe (first occurrence wins) and cap at MAX_KEYWORDS.

    Order is preserved because the caller's first keywords are their most
    important ones, and the quality check weights coverage over that same list.
    """
    seen: set[str] = set()
    out: list[str] = []
    for raw in keywords or []:
        cleaned = collapse_whitespace(str(raw)).lower()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        out.append(cleaned)
        if len(out) == MAX_KEYWORDS:
            break
    return out


def normalize_industry(industry: str | None) -> str | None:
    """Map a synonym to the canonical list; otherwise title-case the user's own wording."""
    if industry is None:
        return None
    cleaned = collapse_whitespace(industry)
    if not cleaned:
        return None
    if canonical := INDUSTRY_SYNONYMS.get(_synonym_key(cleaned)):
        return canonical
    # Unknown industry: keep it, title-cased, so the prompt still gets a signal.
    return cleaned.title()


def normalize_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    return collapse_whitespace(value) or None


def normalize_request(request: ArticleRequest) -> NormalizedParams:
    """Clean an inbound request into the canonical params the pipeline uses."""
    topic = collapse_whitespace(request.topic)
    if not topic:
        raise ValidationError("topic must not be empty")

    audience = normalize_optional_text(request.audience) or DEFAULT_AUDIENCE

    return NormalizedParams(
        topic=topic,
        category=normalize_optional_text(request.category),
        industry=normalize_industry(request.industry),
        audience=audience,
        keywords=normalize_keywords(request.keywords),
        length=request.length,
        language=request.language,
        target_words=TARGET_WORDS[request.length],
    )
