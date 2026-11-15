# Contributing

## Setup

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[all]"
pre-commit install
```

## Before you push

```bash
make check      # ruff check + format check, mypy (strict on core/ and io/), pytest
```

CI runs the same checks on Python 3.11, 3.12 and 3.13, plus a Docker build and health check.
The coverage gate is 85 %.

## Conventions

* **Layering:** `core` must not import FastAPI, Streamlit or touch the filesystem. `ui` talks to
  the API only through `ui/api_client.py`.
* **Types:** full annotations in `core/` and `io/` (mypy strict).
* **Immutability:** steps return new DataFrames and never mutate their input.
* **Errors:** raise `InvalidInputError` / `InvalidConfigError`, never bare `ValueError`, for
  problems the user can fix; the API maps them to 4xx.
* **Adding a step:** subclass `Step` in `core/steps/`, add it to `STEP_REGISTRY` and
  `DEFAULT_STEP_ORDER`, add a toggle to `PipelineConfig`, and write unit tests for it.
* **Commits:** [Conventional Commits](https://www.conventionalcommits.org/) —
  `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`, `ci:`.
* **Versioning:** SemVer. Update `CHANGELOG.md` and tag `vX.Y.Z` to trigger the release workflow.
