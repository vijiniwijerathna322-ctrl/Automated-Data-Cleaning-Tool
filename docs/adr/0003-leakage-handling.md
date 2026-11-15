# ADR 0003 – Target-leakage handling in the ML dataset

**Status:** accepted

## Context
The original tool binned `Exam_Score` into `Performance_Category`, one-hot encoded it and shipped
it next to `Exam_Score`. Anyone predicting `Exam_Score` would get the answer as a feature.

## Decision
`PipelineConfig.target` names the target column. When building the ML dataset:
* the target is kept raw — never scaled or encoded;
* engineered features recorded as *derived from the target* are dropped;
* ID columns (`id`, `*_id`, `*identifier*`) are dropped;
* only original numeric columns are scaled, never one-hot dummies.

The cleaned dataset still contains the engineered feature; only the ML export excludes it.

## Consequences
* (+) The default ML export is safe to train on.
* (−) Detecting derivation relies on the pipeline's own record, so a feature the user adds
  outside the pipeline is not covered.
* (−) Scaling is still fit on the whole upload; the README notes that a real model should fit on
  the training split only.
