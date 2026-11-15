# Changelog

All notable changes are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning follows
[SemVer](https://semver.org/).

## [Unreleased]

### Fixed
- Live demo on Streamlit Community Cloud broke after the restructure (no root `app.py` or
  `requirements.txt`, and no API to call). Restored a root `app.py` launcher and
  `requirements.txt`, and the UI now runs the API in-process when `DATACLEANER_API_URL` is unset.

### Changed
- `datacleaner-ui` / `make run-ui` no longer need a separately started API.
- The UI reuses one HTTP client per API URL instead of creating one per call.

## [0.1.0] - 2026-11-14

### Added
- `src/` layout package `datacleaner` with a pure `core` (config, pipeline, report, profile and
  one class per cleaning step), an `io` layer (CSV, XLSX, Parquet), a FastAPI service, a
  Streamlit UI that uses the API, and a Typer CLI.
- Pydantic-validated, YAML/JSON-loadable `PipelineConfig` and a serialisable `CleaningReport`
  with per-step timings and warnings.
- RFC 7807 error responses, request IDs, structured JSON logging, upload size/row limits.
- Dockerfile (multi-stage, non-root), docker-compose, GitHub Actions CI and release workflows.
- ruff, mypy, pytest + coverage, pre-commit, Makefile; 139 tests at ~99 % coverage.
- Architecture notes and four ADRs.

### Changed
- ML dataset: ID columns are dropped, the target is never encoded/scaled, target-derived
  features are excluded, and only numeric originals are scaled.
- Pipeline order is now standardize → de-duplicate → outliers → impute → features.

### Fixed
- The "Standardize Categories" and "Feature Engineering" toggles were shadowed by functions of
  the same name and had no effect.
