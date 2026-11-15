"""Cleaning steps and their registry.

Add a step by subclassing :class:`Step` and listing it in ``STEP_REGISTRY``;
the pipeline itself never needs to change (open/closed principle).
"""

from __future__ import annotations

from datacleaner.core.steps.base import PipelineContext, Step
from datacleaner.core.steps.coercion import NumericCoercionStep
from datacleaner.core.steps.deduplicate import DeduplicateStep
from datacleaner.core.steps.features import FeatureStep
from datacleaner.core.steps.impute import ImputeStep
from datacleaner.core.steps.outliers import OutlierStep
from datacleaner.core.steps.standardize import StandardizeStep

#: Canonical, deliberate order. Standardizing before de-duplicating catches rows
#: that differ only by case/spacing; de-duplicating before imputing avoids false
#: duplicates created by filled-in medians; outliers are measured on real values.
DEFAULT_STEP_ORDER: tuple[str, ...] = (
    "coerce_numeric",
    "standardize",
    "deduplicate",
    "outliers",
    "impute",
    "features",
)

STEP_REGISTRY: dict[str, type[Step]] = {
    cls.name: cls
    for cls in (
        NumericCoercionStep,
        StandardizeStep,
        DeduplicateStep,
        OutlierStep,
        ImputeStep,
        FeatureStep,
    )
}


def build_steps(order: tuple[str, ...] | list[str] = DEFAULT_STEP_ORDER) -> list[Step]:
    """Instantiate steps by registry name, in the given order."""
    unknown = [n for n in order if n not in STEP_REGISTRY]

    if unknown:
        raise KeyError(f"Unknown step(s): {', '.join(unknown)}")

    return [STEP_REGISTRY[name]() for name in order]


__all__ = ["DEFAULT_STEP_ORDER", "STEP_REGISTRY", "PipelineContext", "Step", "build_steps"]
