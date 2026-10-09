VENV ?= .venv
PY := $(VENV)/bin/python

.PHONY: install run test lint typecheck eval check docker-up docker-milvus

install:
	python3 -m venv $(VENV)
	$(PY) -m pip install -U pip
	$(PY) -m pip install -e ".[dev]"

run:
	$(VENV)/bin/uvicorn docsearch.main:app --reload --port 8000

test:
	$(PY) -m pytest

lint:
	$(VENV)/bin/ruff check . && $(VENV)/bin/ruff format --check .

typecheck:
	$(VENV)/bin/mypy

eval:
	$(PY) -m docsearch.evaluation
	$(PY) -m docsearch.evaluation --sweep

check: lint typecheck test

docker-up:
	docker compose up --build

docker-milvus:
	VECTOR_STORE=milvus docker compose --profile milvus up --build
