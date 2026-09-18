.DEFAULT_GOAL := help
UV ?= uv

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

install:  ## Create the venv and install runtime + dev deps
	$(UV) sync --group dev

dev:  ## Run the API locally with autoreload (mock provider, no cloud needed)
	$(UV) run uvicorn app.main:app --reload --port 8000

test:  ## Run the test suite
	$(UV) run pytest

lint:  ## Lint + typecheck
	$(UV) run ruff check .
	$(UV) run ruff format --check .
	$(UV) run mypy

fmt:  ## Auto-format and auto-fix
	$(UV) run ruff format .
	$(UV) run ruff check --fix .

reqs:  ## Regenerate requirements.txt (what Vercel installs) from the runtime deps
	./scripts/gen_requirements.sh

smoke:  ## Smoke-test a running deployment: make smoke URL=https://...
	./scripts/smoke_test.sh $(URL)

.PHONY: help install dev test lint fmt reqs smoke

# --- data pipeline (Day 1) ---------------------------------------------------
corpus:  ## Full corpus build: discover -> crawl -> clean -> stats
	$(UV) run python -m pipeline.scrape discover
	$(UV) run python -m pipeline.scrape crawl
	$(UV) run python -m pipeline.clean run
	$(UV) run python -m pipeline.clean stats

label:  ## Reverse-label the cleaned corpus (needs LABELER_* in .env)
	$(UV) run python -m pipeline.label run

dataset:  ## Build and validate train/eval JSONL: make dataset VERSION=v1
	$(UV) run python -m pipeline.build_dataset run --version $(or $(VERSION),v1)
	$(UV) run python -m pipeline.build_dataset validate --version $(or $(VERSION),v1)

corpus-status:  ## How far the corpus has progressed
	$(UV) run python -m pipeline.scrape status

# --- GPU jobs (Day 2) --------------------------------------------------------
modal-setup:  ## Authenticate the Modal CLI (opens a browser)
	$(UV) run modal setup

train:  ## Fine-tune on Modal: make train DATASET=r2://datasets/v1/train.jsonl
	$(UV) run modal run modal/train.py --dataset-uri $(or $(DATASET),r2://datasets/v1/train.jsonl)

serve:  ## Deploy the vLLM server on Modal and print its URL
	$(UV) run modal deploy modal/serve.py

eval:  ## Base vs fine-tuned: make eval ENDPOINT=https://...modal.run/v1
	$(UV) run modal run modal/eval.py --endpoint $(ENDPOINT)

.PHONY: corpus label dataset corpus-status modal-setup train serve eval
