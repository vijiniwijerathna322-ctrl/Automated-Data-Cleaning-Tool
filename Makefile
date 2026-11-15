.PHONY: install lint format type test check run-api run-ui docker-up docker-down clean

PYTHON ?= python

install:
	$(PYTHON) -m pip install -e ".[all]"
	pre-commit install

lint:
	ruff check .
	ruff format --check .

format:
	ruff check --fix .
	ruff format .

type:
	mypy src

test:
	pytest

check: lint type test

run-api:
	uvicorn datacleaner.api.main:app --reload --port 8000

run-ui:
	streamlit run src/datacleaner/ui/app.py

docker-up:
	docker compose up --build

docker-down:
	docker compose down

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov dist build *.egg-info
