# ADR 0001 – FastAPI backend with a thin Streamlit front end

**Status:** accepted

## Context
The first version was a single Streamlit script that mixed UI and data logic. That cannot be
reused from a script or another service, and it is hard to test or deploy independently.

## Decision
Put all behaviour behind a FastAPI service built on a pure `core` package. Keep Streamlit as a
presentation layer that calls the API through one client module. Add a CLI as another adapter.

## Consequences
* (+) The same logic is reachable from HTTP, the browser and the shell, and is tested once.
* (+) The API gets OpenAPI docs, validation and standard error semantics for free.
* (+) UI and API can be scaled and deployed separately.
* (−) Two processes to run locally (mitigated by `docker compose` and `make` targets).
* (−) Datasets cross HTTP on every call; the UI caches calls and the API enforces size limits.

## Alternatives considered
* *React + FastAPI* – more impressive UI, but a second language and build chain for little
  extra value in a data-tooling project.
* *Streamlit only, packaged* – smallest change, but leaves no API and weaker separation.
