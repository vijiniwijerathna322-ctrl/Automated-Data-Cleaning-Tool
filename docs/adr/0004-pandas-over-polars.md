# ADR 0004 – pandas over Polars

**Status:** accepted

## Context
Polars is faster and more memory-efficient for large data. The tool targets interactive,
upload-sized datasets, and downstream ML code (scikit-learn) speaks pandas.

## Decision
Keep pandas. Bound memory with an upload size limit and a row limit enforced in the API, and
document the in-memory design as a limitation.

## Consequences
* (+) Direct interoperability with scikit-learn and the wider ecosystem; far less migration risk.
* (+) Smaller, easier-to-review change set.
* (−) Not suitable for datasets that do not fit in memory.

Revisit if out-of-core or multi-GB inputs become a real requirement; the `Step` interface
(`DataFrame -> DataFrame`) is the seam where a Polars implementation would plug in.
