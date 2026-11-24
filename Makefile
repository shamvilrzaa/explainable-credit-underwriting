.PHONY: install lint format test train report

install:
	pip install -e ".[api,dashboard,dev]"

lint:
	ruff check . && ruff format --check .

format:
	ruff format . && ruff check --fix .

test:
	pytest

train:
	python -m credit_underwriting.train --model scorecard

report:
	python -m credit_underwriting.compare
