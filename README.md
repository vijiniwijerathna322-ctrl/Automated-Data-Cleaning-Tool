# 🧹 Automated Data Cleaning Tool

[![CI](https://github.com/vijiniwijerathna322-ctrl/Automated-Data-Cleaning-Tool/actions/workflows/ci.yml/badge.svg)](https://github.com/vijiniwijerathna322-ctrl/Automated-Data-Cleaning-Tool/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%20|%203.12%20|%203.13-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A config-driven pipeline that inspects, cleans and prepares tabular data for machine learning,
exposed three ways: a **REST API** (FastAPI), a **web UI** (Streamlit) and a **CLI** (Typer).

## 🌐 Live Demo

🚀 **Try the application here:**  
[Open Live Application](https://automated-data-cleaning-tool-uit2pvmyetqptrzfyrn5rk.streamlit.app/)

## Why it exists

Real datasets arrive with missing values, duplicates, inconsistent categories and outliers. This
tool applies a deliberate, tested sequence of cleaning steps, produces a ready-to-model dataset
that **avoids target leakage**, and records exactly what it did in a machine-readable report.

## Features

- **Formats:** CSV, Excel (`.xlsx`) and Parquet in; the same out.
- **Pipeline steps:** numeric coercion · category standardization · de-duplication · IQR outlier
  detection · imputation · feature engineering · one-hot encoding · scaling.
- **ML-safe output:** ID columns are dropped, the target is never encoded or scaled, and features
  derived from the target are excluded (no leakage). Only real numeric columns are scaled, never
  the dummies.
- **Config as data:** every behaviour is a validated Pydantic setting (YAML/JSON/API body).
- **Observable:** per-step timings and warnings in the report, JSON structured logs, request IDs.
- **Production habits:** typed errors → RFC 7807 problem responses, upload size/row limits,
  12-factor settings, non-root multi-stage Docker image, CI on every push.

## Architecture

```text
            ┌────────────┐   HTTP    ┌────────────────┐
 browser ──▶│ Streamlit  │──────────▶│  FastAPI       │
            │  ui/       │ api_client│  api/          │
            └────────────┘           └───────┬────────┘
                                             │            ┌──────────┐
            ┌────────────┐                   ├───────────▶│  io/     │ csv · xlsx · parquet
   shell ──▶│ Typer CLI  │───────────────────┤            └──────────┘
            │  cli.py    │                   ▼
            └────────────┘             ┌──────────┐
                                       │  core/   │  Pipeline → Steps → Report
                                       └──────────┘  (pure: no FastAPI / Streamlit imports)
```

`core` depends on nothing outward; the API, UI and CLI are thin adapters. See
[docs/architecture.md](docs/architecture.md) and the decision records in [docs/adr](docs/adr).

```text
src/datacleaner/
├── core/      config · pipeline · report · profile · steps/ (one class per step)
├── io/        readers / writers / format detection
├── api/       FastAPI app factory, routes, settings, error handlers
├── ui/        Streamlit app + api_client + components
├── cli.py     datacleaner clean | profile | init-config | version
└── logging.py JSON logging
tests/         unit/ · integration/ (API, CLI, client) · ui/ (Streamlit smoke)
configs/       default.yaml        data/sample/   sample_data.csv
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[all]"

make run-api      # http://localhost:8000/docs
make run-ui       # http://localhost:8501   (needs the API running)
```

Or with Docker:

```bash
docker compose up --build        # API :8000, UI :8501
```

### CLI

```bash
datacleaner clean data/sample/sample_data.csv -o out/ --target Exam_Score --feature-source Exam_Score
datacleaner profile data/sample/sample_data.csv
datacleaner init-config config.yaml          # write every default to edit
```

`clean` writes `cleaned.<fmt>`, `ml_ready.<fmt>` and `report.json` (formats: csv, xlsx, parquet).

### API

```bash
curl -F file=@data/sample/sample_data.csv \
     -F 'config={"target":"Exam_Score","feature_source":"Exam_Score"}' \
     http://localhost:8000/v1/clean
```

| Method | Path | Purpose |
|---|---|---|
| GET | `/healthz` | liveness probe |
| GET | `/v1/config/default` | default `PipelineConfig` |
| POST | `/v1/profile` | profile a raw file (types, missing, histograms, preview) |
| POST | `/v1/clean` | cleaned + ML preview and the full `CleaningReport` |
| POST | `/v1/clean/download?dataset=cleaned\|ml&format=csv\|xlsx\|parquet` | result as a file |

Interactive docs at `/docs`. Errors are `application/problem+json`
(`400` bad file, `413` too large, `422` bad config) and carry the `X-Request-ID`.

### Python

```python
from datacleaner import Pipeline, PipelineConfig
from datacleaner.io import read_path

result = Pipeline(PipelineConfig(target="Exam_Score")).run(read_path("data.csv"))
result.cleaned, result.ml, result.report
```

## Configuration

Pipeline options live in [configs/default.yaml](configs/default.yaml) (all keys optional, unknown
keys rejected). Service options are environment variables (see [.env.example](.env.example)):

| Variable | Default | Meaning |
|---|---|---|
| `DATACLEANER_MAX_UPLOAD_MB` | `25` | reject larger uploads (413) |
| `DATACLEANER_MAX_ROWS` | `500000` | reject datasets with more rows |
| `DATACLEANER_LOG_LEVEL` | `INFO` | log verbosity |
| `DATACLEANER_CORS_ORIGINS` | `["http://localhost:8501"]` | allowed browser origins |
| `DATACLEANER_API_URL` | `http://localhost:8000` | where the UI finds the API |

## Development

```bash
make install     # editable install + pre-commit hooks
make check       # ruff, mypy, pytest (≥85% coverage gate)
```

Tests: `tests/unit` (steps, pipeline, config, IO), `tests/integration` (API through
`TestClient`, CLI through `CliRunner`, the UI's HTTP client), `tests/ui` (Streamlit `AppTest`
against an in-process API). Conventions are in [CONTRIBUTING.md](CONTRIBUTING.md).

## Design notes & limitations

- **Step order is deliberate:** standardize → de-duplicate → outliers → impute → features.
  Standardizing first catches `Male`/`male` duplicates; de-duplicating before imputing avoids
  false duplicates created by filled-in medians.
- **Outliers are reported, never removed** — removing rows silently is rarely what you want.
- Scaling is fit on the whole upload. For a real model, fit the scaler on the training split only.
- The whole dataset is processed in memory with pandas (bounded by `MAX_ROWS`/`MAX_UPLOAD_MB`);
  it is not designed for out-of-core data.
- Date columns are passed through untouched.
- Free hosting tiers cold-start, so the first request to a deployed demo can take ~30–60 s.

## License

[MIT](LICENSE)
