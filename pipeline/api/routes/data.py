"""The data side: corpus, scraping, labelling, datasets."""

from __future__ import annotations

import json
import random
from typing import Any, Literal

from fastapi import APIRouter, Path, Query
from fastapi.responses import JSONResponse, PlainTextResponse

from app.core.errors import AppError, NotFoundError
from app.storage.keys import dataset_key
from pipeline import build_dataset, label
from pipeline._cli import recorded_run
from pipeline.api.deps import GUARDED, DispatcherDep, SettingsDep, StorageDep, Stores
from pipeline.api.runs import start_job
from pipeline.api.schemas import (
    ArticleDetail,
    ArticlePage,
    ArticleSummary,
    CorpusStatus,
    DatasetRequest,
    DatasetValidation,
    DatasetVersion,
    JobRun,
    LabelPreview,
    LabelPreviewRequest,
    StageRun,
)
from pipeline.clean import CLEAN_VERSION, corpus_stats
from pipeline.jobs import VERSION_PATTERN, LabelParams, ScrapeParams
from pipeline.label import LABEL_VERSION
from pipeline.schemas import Split, TrainingArticle

router = APIRouter(prefix="/v1", dependencies=GUARDED)

ArticleState = Literal["all", "pending", "fetched", "cleaned", "rejected", "duplicates", "labelled"]
Version = Path(pattern=VERSION_PATTERN)


def _summary(a: TrainingArticle) -> ArticleSummary:
    return ArticleSummary(
        url=a.url,
        category_slug=a.category_slug,
        title=a.title,
        word_count=a.word_count,
        is_duplicate=a.is_duplicate,
        labelled=bool(a.labels and a.labelled_hash and a.labelled_hash == a.cleaned_hash),
        error=a.error,
        last_checked_at=a.last_checked_at,
    )


def _detail(a: TrainingArticle, raw_url: str | None) -> ArticleDetail:
    return ArticleDetail(
        **_summary(a).model_dump(),
        meta_description=a.meta_description,
        r2_raw_key=a.r2_raw_key,
        raw_url=raw_url,
        content_hash=a.content_hash,
        clean_markdown=a.clean_markdown,
        labels=a.labels,
        split=a.split,
        content_changed_at=a.content_changed_at,
    )


# ------------------------------------------------------------------ corpus


@router.get("/corpus", response_model=CorpusStatus, tags=["corpus"])
async def corpus_status(pair: Stores) -> CorpusStatus:
    """Progress per pipeline stage, and the most recent stage runs."""
    corpus, _ = pair
    counts = await corpus.counts(clean_version=CLEAN_VERSION, label_version=LABEL_VERSION)
    recent = [StageRun.model_validate(r) for r in await corpus.recent_runs(10)]
    return CorpusStatus(counts=counts, recent_stages=recent)


@router.get("/corpus/stats", tags=["corpus"])
async def corpus_statistics(pair: Stores) -> dict[str, Any]:
    """Word-count distribution, categories, and why articles were rejected."""
    return corpus_stats(await pair[0].all())


@router.get("/corpus/articles", response_model=ArticlePage, tags=["corpus"])
async def list_articles(
    pair: Stores,
    state: ArticleState = "all",
    category: str | None = None,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> ArticlePage:
    total, items = await pair[0].list_articles(
        state=state, category=category, limit=limit, offset=offset
    )
    return ArticlePage(total=total, limit=limit, offset=offset, items=[_summary(a) for a in items])


@router.get("/corpus/article", response_model=ArticleDetail, tags=["corpus"])
async def get_article(
    pair: Stores, storage: StorageDep, url: str = Query(..., description="Article URL")
) -> ArticleDetail:
    """One article in full: cleaned markdown, labels, and a signed link to its raw HTML."""
    article = await pair[0].get(url)
    if article is None:
        raise NotFoundError(f"no article {url}")
    raw_url = storage.signed_url(article.r2_raw_key) if article.r2_raw_key else None
    return _detail(article, raw_url)


@router.get("/corpus/sample", response_model=list[ArticleDetail], tags=["corpus"])
async def sample_articles(
    pair: Stores,
    storage: StorageDep,
    n: int = Query(5, ge=1, le=50),
    labelled: bool = Query(False, description="Only articles with current labels (label audit)."),
    seed: float = Query(0.42, ge=-1, le=1, description="Same seed, same sample."),
) -> list[ArticleDetail]:
    """Random articles for human review of cleaning (or, with labelled=true, of labels)."""
    articles = await pair[0].sample(n=n, labelled=labelled, seed=seed)
    return [
        _detail(a, storage.signed_url(a.r2_raw_key) if a.r2_raw_key else None) for a in articles
    ]


@router.post("/scrape", status_code=202, response_model=JobRun, tags=["jobs"])
async def start_scrape(
    pair: Stores, dispatcher: DispatcherDep, params: ScrapeParams | None = None
) -> JSONResponse:
    """Pull new articles: discover -> crawl -> clean. Only new/changed content is written."""
    return await start_job(dispatcher, pair[1], "scrape", params or ScrapeParams())


# ------------------------------------------------------------------ labels


def _require_labeler(settings: Any, model: str | None = None) -> str:
    chosen = model or settings.labeler_model
    if not settings.labeler_base_url or not chosen:
        raise AppError(
            "LABELER_BASE_URL / LABELER_MODEL are not configured",
            code="not_configured",
            status_code=503,
        )
    return str(chosen)


@router.post("/label", status_code=202, response_model=JobRun, tags=["jobs"])
async def start_label(
    pair: Stores,
    dispatcher: DispatcherDep,
    settings: SettingsDep,
    params: LabelParams | None = None,
) -> JSONResponse:
    """Reverse-label new or changed articles with the labelling model."""
    params = params or LabelParams()
    _require_labeler(settings, params.model)
    return await start_job(dispatcher, pair[1], "label", params)


@router.post("/label/preview", response_model=LabelPreview, tags=["labels"])
async def preview_label(
    pair: Stores, settings: SettingsDep, request: LabelPreviewRequest | None = None
) -> LabelPreview:
    """Label one article and return it WITHOUT saving — check the prompt before a full run."""
    model = _require_labeler(settings)
    corpus = pair[0]
    if request and request.url:
        article = await corpus.get(request.url)
        if article is None or not article.clean_markdown:
            raise NotFoundError(f"no cleaned article {request.url}")
    else:
        pending = await corpus.due_for_label(label_version=LABEL_VERSION)
        if not pending:
            raise NotFoundError("nothing left to label")
        article = pending[0]
    client = label.labeler_client(settings.labeler_base_url or "", settings.labeler_api_key)
    labels = await label.label_one(client, model, article)
    return LabelPreview(
        url=article.url,
        title=article.title,
        labels=labels,
        excerpt=(article.clean_markdown or "")[:600],
    )


# ------------------------------------------------------------------ datasets


@router.get("/datasets", response_model=list[DatasetVersion], tags=["datasets"])
async def list_datasets(pair: Stores) -> list[DatasetVersion]:
    return [DatasetVersion.model_validate(r) for r in await pair[0].dataset_versions()]


@router.post("/datasets", tags=["datasets"])
async def publish_dataset(
    pair: Stores, storage: StorageDep, request: DatasetRequest
) -> JSONResponse:
    """Validate and publish an immutable dataset version to R2 (synchronous, seconds).

    `201` published · `200` unchanged · `409` version exists with different
    contents · `422` invalid dataset (nothing uploaded).
    """
    corpus = pair[0]
    try:
        async with recorded_run(corpus, "build") as stats:
            stats.update(
                await build_dataset.run_build(
                    corpus,
                    storage,
                    version=request.version,
                    eval_frac=request.eval_frac,
                    seed=request.seed,
                    overwrite=request.overwrite,
                )
            )
    except build_dataset.DatasetError as exc:
        message = str(exc)
        immutable = "immutable" in message
        return JSONResponse(
            status_code=409 if immutable else 422,
            content={
                "error": {
                    "code": "dataset_exists" if immutable else "dataset_invalid",
                    "message": message,
                }
            },
        )
    return JSONResponse(status_code=200 if stats["status"] == "unchanged" else 201, content=stats)


async def _published(pair: Any, version: str) -> dict[str, Any]:
    row = await pair[0].dataset_version(version)
    if row is None:
        raise NotFoundError(f"dataset {version} is not published")
    return dict(row)


@router.get("/datasets/{version}/validate", response_model=DatasetValidation, tags=["datasets"])
async def validate_dataset(
    pair: Stores, storage: StorageDep, version: str = Version
) -> DatasetValidation:
    """Re-validate a published version straight from R2."""
    await _published(pair, version)
    splits: dict[Split, list[dict[str, Any]]] = {"train": [], "eval": []}
    for split in splits:
        raw = await storage.get_bytes(dataset_key(version, f"{split}.jsonl"))
        splits[split] = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    problems = build_dataset.validate_rows(splits)
    return DatasetValidation(
        version=version,
        train=len(splits["train"]),
        eval=len(splits["eval"]),
        ok=not problems,
        problems=problems,
    )


@router.get("/datasets/{version}/card", response_class=PlainTextResponse, tags=["datasets"])
async def dataset_card(
    pair: Stores, storage: StorageDep, version: str = Version
) -> PlainTextResponse:
    """The data card (markdown) published with the version."""
    row = await _published(pair, version)
    body = await storage.get_bytes(row["card_key"])
    return PlainTextResponse(body.decode("utf-8"), media_type="text/markdown; charset=utf-8")


@router.get("/datasets/{version}/examples", tags=["datasets"])
async def dataset_examples(
    pair: Stores,
    storage: StorageDep,
    version: str = Version,
    split: Literal["train", "eval"] = "train",
    n: int = Query(2, ge=1, le=20),
) -> list[dict[str, Any]]:
    """A few rows exactly as the trainer will see them."""
    await _published(pair, version)
    raw = await storage.get_bytes(dataset_key(version, f"{split}.jsonl"))
    rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    return random.Random(0).sample(rows, min(n, len(rows)))
