.PHONY: install lint format test train train-gbm report serve docker-build docker-run

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
	python -m credit_underwriting.explain_report
	python -m credit_underwriting.fairness_report

train-gbm:
	python -m credit_underwriting.train --model gbm

serve: train-gbm
	uvicorn credit_underwriting.api:app --reload

docker-build: train-gbm
	docker build -t credit-underwriting .

docker-run:
	docker run --rm -p 8000:8000 credit-underwriting
