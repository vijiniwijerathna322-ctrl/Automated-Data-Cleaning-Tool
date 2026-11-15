# Architecture

## Layers and dependency direction

```text
ui ──▶ (HTTP) ──▶ api ──▶ io ──▶ core
                   cli ───▶ io ──▶ core
```

* **core** – pure domain logic. Imports pandas, scikit-learn and Pydantic only. No HTTP, no UI,
  no file I/O. This is what the unit tests exercise most.
* **io** – reads/writes CSV, XLSX, Parquet and the JSON report; converts parser failures to
  `InvalidInputError`.
* **api** – FastAPI: validates uploads (size, rows, type), parses config, calls `core.Pipeline`,
  maps domain errors to RFC 7807 problem responses.
* **ui** – Streamlit. Talks to the API *only* through `ui/api_client.py`, which returns plain
  dicts, so the UI has no dependency on server or domain models.
  On single-process hosts (Streamlit Community Cloud) `ui/embedded.py` starts the same FastAPI
  app on a loopback port in a daemon thread when `DATACLEANER_API_URL` is unset, so the UI still
  goes over real HTTP and the layering holds.
* **cli** – a second adapter over the same `core` + `io`, proving the logic is reusable.

## The pipeline

```text
prepare_dataframe ─▶ validate config ─▶ detect ID columns
   ─▶ [coerce_numeric] ─▶ [standardize] ─▶ [deduplicate] ─▶ [outliers] ─▶ [impute] ─▶ [features]
   ─▶ build ML dataset (drop IDs/leaky features, scale numerics, one-hot text)
   ─▶ CleaningReport
```

* Each bracketed stage is a `Step` subclass (`core/steps/`) with a `name`, a config `toggle`, and
  `apply(df, ctx) -> df`. Steps never mutate their input.
* `PipelineContext` is the typed scratchpad steps write into; `Pipeline` turns it into a
  Pydantic `CleaningReport` (counts, columns touched, outlier summaries, per-step timings,
  warnings).
* New step = new class + one line in `STEP_REGISTRY`. `Pipeline(steps=[...])` accepts any custom
  sequence, so the order is data, not control flow.
* The ML dataset is *derived* from the cleaned one rather than being a step, because it produces
  a second DataFrame.

## Error model

| Exception (core) | HTTP | When |
|---|---|---|
| `InvalidInputError` | 400 | unreadable/empty file, unsupported type, too many rows, duplicate headers |
| `PayloadTooLargeError` (api) | 413 | upload bigger than `DATACLEANER_MAX_UPLOAD_MB` |
| `InvalidConfigError` | 422 | bad/unknown config keys, target/feature column not in the data |
| anything else | 500 | logged with traceback; the client gets a generic message |

All errors are `application/problem+json` with the request ID.

## Decision records

* [ADR 0001 – FastAPI backend with a thin Streamlit front end](adr/0001-fastapi-plus-streamlit.md)
* [ADR 0002 – Step pattern for the pipeline](adr/0002-step-pattern.md)
* [ADR 0003 – Target-leakage handling in the ML dataset](adr/0003-leakage-handling.md)
* [ADR 0004 – pandas over Polars](adr/0004-pandas-over-polars.md)
