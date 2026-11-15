"""datacleaner: a config-driven data cleaning and ML-preparation pipeline."""

from datacleaner.core.config import PipelineConfig
from datacleaner.core.errors import DataCleanerError, InvalidConfigError, InvalidInputError
from datacleaner.core.pipeline import Pipeline, PipelineResult
from datacleaner.core.report import CleaningReport

__version__ = "0.1.0"

__all__ = [
    "CleaningReport",
    "DataCleanerError",
    "InvalidConfigError",
    "InvalidInputError",
    "Pipeline",
    "PipelineConfig",
    "PipelineResult",
    "__version__",
]
