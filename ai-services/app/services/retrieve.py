"""Chunking + BM25 retrieval.

Design principle for the whole product: *fine-tuning teaches style, retrieval
supplies facts*. The adapter should never be asked to remember a statistic, so
the retrieval side only has to be good enough to surface the right paragraphs
from a handful of user-supplied documents.

BM25 over pure Python is deliberate: the corpus per request is a few documents,
not a warehouse, and an embedding model would drag torch into the Vercel bundle
(hard 500 MB limit). Swap in a vector store behind `retrieve_chunks` if the
corpus ever grows.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from uuid import UUID

from app.schemas.sources import SourceDocument

_TOKEN = re.compile(r"[a-z0-9]+|[฀-๿]+")

# Common words carry no discriminative signal and would otherwise dominate short
# queries like "the future of retail".
STOPWORDS: frozenset[str] = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "to",
        "was",
        "were",
        "will",
        "with",
        "what",
        "how",
        "why",
        "when",
        "which",
        "this",
        "these",
        "those",
        "there",
        "their",
        "we",
        "you",
        "they",
        "our",
        "your",
    ]
)

# BM25 constants; k1 controls term-frequency saturation, b the length penalty.
K1 = 1.5
B = 0.75


@dataclass(slots=True)
class Chunk:
    text: str
    source_id: UUID | None = None
    source_title: str | None = None
    source_url: str | None = None
    chunk_index: int = 0
    score: float = 0.0
    tokens: list[str] = field(default_factory=list)


def tokenize(text: str) -> list[str]:
    """Lowercase word/Thai-run tokens with stopwords dropped."""
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


def chunk_text(
    text: str,
    *,
    max_chars: int = 1200,
    overlap_chars: int = 150,
) -> list[str]:
    """Split on paragraph boundaries, packing paragraphs up to `max_chars`.

    Paragraph-aligned chunks keep sentences intact, which matters because the
    chunk text is pasted verbatim into the prompt as evidence. `overlap_chars`
    carries the tail of the previous chunk forward so a fact that straddles a
    boundary is still retrievable.
    """
    if not text or not text.strip():
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        paragraphs = [text.strip()]

    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        # A single oversized paragraph is hard-split; nothing else can be done.
        if len(para) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            for i in range(0, len(para), max_chars):
                chunks.append(para[i : i + max_chars])
            continue
        if not current:
            current = para
        elif len(current) + 2 + len(para) <= max_chars:
            current = f"{current}\n\n{para}"
        else:
            chunks.append(current)
            tail = current[-overlap_chars:] if overlap_chars else ""
            current = f"{tail}\n\n{para}".strip() if tail else para
    if current:
        chunks.append(current)
    return chunks


def build_chunks(documents: list[SourceDocument], **kwargs: int) -> list[Chunk]:
    """Flatten documents into scored-ready chunks, keeping provenance on each."""
    chunks: list[Chunk] = []
    for doc in documents:
        for idx, piece in enumerate(chunk_text(doc.text, **kwargs)):
            chunks.append(
                Chunk(
                    text=piece,
                    source_id=doc.id,
                    source_title=doc.title,
                    source_url=doc.url,
                    chunk_index=idx,
                    tokens=tokenize(piece),
                )
            )
    return chunks


def bm25_scores(query_tokens: list[str], chunks: list[Chunk]) -> list[float]:
    """Okapi BM25 of the query against every chunk (chunk == document)."""
    if not chunks or not query_tokens:
        return [0.0] * len(chunks)

    n = len(chunks)
    lengths = [len(c.tokens) for c in chunks]
    avgdl = sum(lengths) / n or 1.0

    doc_freq: Counter[str] = Counter()
    for chunk in chunks:
        doc_freq.update(set(chunk.tokens))

    scores = [0.0] * n
    for term in set(query_tokens):
        df = doc_freq.get(term, 0)
        if df == 0:
            continue
        # Standard BM25+ style IDF; the +1 inside log keeps it non-negative even
        # when a term appears in every chunk.
        idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
        for i, chunk in enumerate(chunks):
            tf = chunk.tokens.count(term)
            if not tf:
                continue
            denom = tf + K1 * (1 - B + B * lengths[i] / avgdl)
            scores[i] += idf * (tf * (K1 + 1)) / denom
    return scores


def retrieve_chunks(
    query: str,
    documents: list[SourceDocument],
    *,
    top_k: int = 6,
) -> list[Chunk]:
    """Return the `top_k` highest-scoring chunks for `query`, best first."""
    chunks = build_chunks(documents)
    if not chunks:
        return []

    query_tokens = tokenize(query)
    for chunk, score in zip(chunks, bm25_scores(query_tokens, chunks), strict=True):
        chunk.score = score

    ranked = sorted(chunks, key=lambda c: (-c.score, c.chunk_index))
    # Drop zero-score chunks: padding the prompt with irrelevant text costs
    # tokens and invites the model to cite something the user never asked about.
    hits = [c for c in ranked if c.score > 0][:top_k]
    return hits
