from pathlib import Path

import pandas as pd
import pytest

from datacleaner.core.config import PipelineConfig

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_CSV = ROOT / "data" / "sample" / "sample_data.csv"


@pytest.fixture
def sample_path() -> Path:
    return SAMPLE_CSV


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_CSV)


@pytest.fixture
def sample_config() -> PipelineConfig:
    """Config matching the sample student dataset."""
    return PipelineConfig(target="Exam_Score", feature_source="Exam_Score")
