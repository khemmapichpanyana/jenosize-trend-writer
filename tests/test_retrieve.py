"""Chunking and BM25 behaviour."""

from __future__ import annotations

from uuid import uuid4

import pytest

from app.schemas.sources import SourceDocument
from app.services.retrieve import bm25_scores, build_chunks, chunk_text, retrieve_chunks, tokenize


def _doc(text: str, title: str = "doc") -> SourceDocument:
    return SourceDocument(id=uuid4(), kind="upload", title=title, text=text, text_chars=len(text))


def test_tokenize_drops_stopwords_and_lowercases() -> None:
    assert tokenize("The Future of Retail") == ["future", "retail"]


def test_chunk_text_returns_empty_for_blank() -> None:
    assert chunk_text("   \n\n  ") == []


def test_chunk_text_keeps_short_documents_whole() -> None:
    assert chunk_text("one paragraph only") == ["one paragraph only"]


def test_chunk_text_respects_max_chars() -> None:
    text = "\n\n".join("word " * 60 for _ in range(10))
    chunks = chunk_text(text, max_chars=400, overlap_chars=0)
    assert len(chunks) > 1
    assert all(len(c) <= 400 for c in chunks)


def test_chunk_text_hard_splits_oversized_paragraph() -> None:
    chunks = chunk_text("x" * 1000, max_chars=300, overlap_chars=0)
    assert [len(c) for c in chunks] == [300, 300, 300, 100]


def test_bm25_ranks_the_relevant_chunk_first() -> None:
    docs = [
        _doc("Solar panel efficiency improved across the region.", "solar"),
        _doc("Retail media networks are reshaping advertising budgets.", "retail"),
    ]
    chunks = build_chunks(docs)
    scores = bm25_scores(tokenize("retail media advertising"), chunks)
    assert scores[1] > scores[0]


def test_bm25_is_zero_without_query_terms() -> None:
    chunks = build_chunks([_doc("anything at all")])
    assert bm25_scores([], chunks) == [0.0]


def test_retrieve_drops_irrelevant_chunks() -> None:
    docs = [
        _doc("Agentic AI agents book appointments for retail customers.", "relevant"),
        _doc("Completely unrelated text about volcanoes and geology.", "irrelevant"),
    ]
    hits = retrieve_chunks("agentic ai retail", docs, top_k=5)
    assert len(hits) == 1
    assert hits[0].source_title == "relevant"
    assert hits[0].score > 0


def test_retrieve_respects_top_k() -> None:
    docs = [_doc(f"retail trend number {i} about retail commerce", f"d{i}") for i in range(8)]
    assert len(retrieve_chunks("retail", docs, top_k=3)) == 3


@pytest.mark.parametrize("documents", [[], [_doc("")]])
def test_retrieve_handles_empty_corpus(documents: list[SourceDocument]) -> None:
    assert retrieve_chunks("anything", documents) == []


def test_chunks_carry_provenance() -> None:
    doc = _doc("retail media is growing fast in the region", "Provenance Doc")
    hit = retrieve_chunks("retail media", [doc])[0]
    assert hit.source_id == doc.id
    assert hit.source_title == "Provenance Doc"
    assert hit.chunk_index == 0
