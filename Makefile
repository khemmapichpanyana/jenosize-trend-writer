.DEFAULT_GOAL := help
UV ?= uv

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

install:  ## Create the venv: runtime + dev + pipeline + Modal CLI
	$(UV) sync --group dev --group pipeline --group modal --group agent

dev:  ## Run the API locally with autoreload (mock provider, no cloud needed)
	$(UV) run python scripts/dev_server.py

dev-assignment:  ## Run article + local Jobs APIs in one process
	$(UV) run python scripts/dev_server.py --with-jobs

test:  ## Run the test suite
	$(UV) run pytest

lint:  ## Lint + typecheck
	$(UV) run ruff check app pipeline studio modal scripts tests
	$(UV) run ruff format --check app pipeline studio modal scripts tests
	$(UV) run mypy

fmt:  ## Auto-format and auto-fix
	$(UV) run ruff format app pipeline studio modal scripts tests
	$(UV) run ruff check --fix app pipeline studio modal scripts tests

reqs:  ## Regenerate requirements.txt (what Vercel installs) from the runtime deps
	./scripts/gen_requirements.sh

smoke:  ## Smoke-test a running deployment: make smoke URL=https://...
	./scripts/smoke_test.sh $(URL)

.PHONY: help install dev dev-assignment test lint fmt reqs smoke

# --- data pipeline: writes to Postgres (DATABASE_URL) + Cloudflare R2 ----------
# Every target is incremental: re-running it only processes what changed.

migrate:  ## Apply supabase/migrations/*.sql to DATABASE_URL
	$(UV) run python -m pipeline.migrate up

scrape:  ## Discover new URLs + crawl pages we don't have yet
	$(UV) run python -m pipeline.scrape discover
	$(UV) run python -m pipeline.scrape crawl

recheck:  ## Also re-fetch pages older than DAYS (default 30) to catch edits
	$(UV) run python -m pipeline.scrape crawl --recheck-days $(or $(DAYS),30)

clean:  ## Clean new/changed articles; flag near-duplicates
	$(UV) run python -m pipeline.clean run

label:  ## Reverse-label new/changed articles (needs LABELER_* in .env)
	$(UV) run python -m pipeline.label run

corpus: scrape clean  ## scrape + clean, then show stats
	$(UV) run python -m pipeline.clean stats

dataset:  ## Publish an immutable dataset version: make dataset VERSION=v1
	$(UV) run python -m pipeline.build_dataset run --version $(or $(VERSION),v1)

pipeline: scrape clean label  ## Everything up to (not including) publishing a dataset

status:  ## Corpus progress per stage + recent pipeline runs
	$(UV) run python -m pipeline.scrape status

# --- Modal: jobs API + GPU jobs ---------------------------------------------------
modal-setup:  ## Authenticate the Modal CLI (opens a browser)
	$(UV) run modal setup

modal-secrets:  ## Create/update the Modal secrets from .env (least privilege, values never printed)
	$(UV) run python scripts/modal_secrets.py

modal-doctor:  ## Check secrets, Postgres, R2 and imports from inside Modal (~1 cent)
	$(UV) run modal run modal/doctor.py

deploy-modal:  ## Deploy jobs API + vLLM server + train/eval (one-time bootstrap)
	$(UV) run modal deploy modal/deploy.py

jobs-dev:  ## Run the jobs API locally (scrape/label in-process; train/eval need Modal)
	$(UV) run uvicorn --factory pipeline.api:create_local_app --reload --port 8001

train:  ## Fine-tune from the CLI (the jobs API does the same): make train VERSION=v1
	$(UV) run modal run modal/train.py --version $(or $(VERSION),v1)

eval:  ## Base vs fine-tuned from the CLI: make eval ENDPOINT=https://...modal.run/v1 VERSION=v1
	$(UV) run modal run modal/eval.py --endpoint $(ENDPOINT) --briefs-uri r2://datasets/$(or $(VERSION),v1)/eval.jsonl

.PHONY: migrate scrape recheck clean label corpus dataset pipeline status modal-setup modal-secrets modal-doctor deploy-modal jobs-dev train eval
