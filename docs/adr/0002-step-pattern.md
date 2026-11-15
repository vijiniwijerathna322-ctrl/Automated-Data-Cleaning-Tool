# ADR 0002 – Step pattern for the pipeline

**Status:** accepted

## Context
Cleaning logic started as one function with a chain of `if option:` blocks. Adding or reordering a
stage meant editing that function, and each stage was hard to test in isolation.

## Decision
Each stage is a `Step` subclass with a stable `name`, a config `toggle`, and
`apply(df, ctx) -> df`. A registry maps names to classes; `Pipeline` runs the enabled steps in
order and builds a `CleaningReport` from a shared typed `PipelineContext`.

## Consequences
* (+) Open/closed: a new step is a new class and one registry line.
* (+) Every step has focused unit tests; the pipeline test only checks orchestration and order.
* (+) Per-step timing and logging are uniform.
* (−) Steps communicate through the context object, which is a small amount of shared mutable
  state (kept typed and scoped to one run).

## Order rationale
standardize → de-duplicate → outliers → impute → features. Standardizing first catches
case/whitespace duplicates; de-duplicating before imputing avoids false duplicates from
filled-in medians; outliers are measured on observed values only.
