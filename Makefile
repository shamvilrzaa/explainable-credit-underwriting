.PHONY: install lint format test

install:
	pip install -e ".[api,dashboard,dev]"

lint:
	ruff check . && ruff format --check .

format:
	ruff format . && ruff check --fix .

test:
	pytest
