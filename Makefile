check:
	uv run ruff check .
	uv run ruff format --check .
	uv run pytest

.PHONY: check
