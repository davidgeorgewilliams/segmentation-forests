.PHONY: install test lint format clean build hooks fix

install:
	uv pip install -e ".[dev]"
	make hooks

hooks:
	uv run pre-commit install
	uv run pre-commit autoupdate

test:
	uv run pytest tests/

lint:
	ruff check .
	uv run mypy -p segmentation_forests
	uv run pre-commit run --all-files

format:
	uv tool run isort .
	uv tool run black .
	ruff check --fix --unsafe-fixes .

fix:
	ruff check --fix --unsafe-fixes .

clean:
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info
	find . -type d -name '__pycache__' -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	find . -type f -name '*.pyo' -delete
	find . -type f -name '.coverage' -delete

build:
	uv build

all: clean install format lint test build
