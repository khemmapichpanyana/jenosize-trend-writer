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
