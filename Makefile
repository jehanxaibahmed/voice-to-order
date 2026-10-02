.PHONY: install lint typecheck test check run

install:
	python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"

lint:
	.venv/bin/ruff check src tests
	.venv/bin/ruff format --check src tests

typecheck:
	.venv/bin/mypy

test:
	.venv/bin/pytest -q

check: lint typecheck test

run:
	.venv/bin/uvicorn voice_to_order.api.app:create_app --factory --reload
