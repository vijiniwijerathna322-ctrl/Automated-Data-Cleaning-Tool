import json

import pandas as pd
import pytest

from datacleaner import Pipeline, PipelineConfig
from datacleaner.core.errors import InvalidConfigError, InvalidInputError
from datacleaner.core.steps import build_steps


class TestSamplePipeline:
    def test_default_run_matches_known_results(
        self, sample_df: pd.DataFrame, sample_config: PipelineConfig
    ) -> None:
        result = Pipeline(sample_config).run(sample_df)
        report = result.report

        assert report.duplicates_removed == 1
        assert report.missing_fixed == 4
        assert report.after.missing_values == 0
        assert report.before.rows == 19
        assert report.after.rows == 18
        assert sorted(result.cleaned["Gender"].unique()) == ["Female", "Male"]
        assert "Study_Hours" in report.outliers
        assert report.created_features == ["Exam_Score_Category"]
        assert [s.name for s in report.steps] == [
            "coerce_numeric",
            "standardize",
            "deduplicate",
            "outliers",
            "impute",
            "features",
        ]

    def test_no_leakage_and_no_ids_in_ml_dataset(
        self, sample_df: pd.DataFrame, sample_config: PipelineConfig
    ) -> None:
        result = Pipeline(sample_config).run(sample_df)
        cols = result.ml.columns.tolist()

        assert not any(c.startswith("Exam_Score_Category") for c in cols)
        assert "Student_ID" not in cols
        assert sorted(result.report.dropped_from_ml) == ["Exam_Score_Category", "Student_ID"]

    def test_target_is_raw_in_ml_dataset(
        self, sample_df: pd.DataFrame, sample_config: PipelineConfig
    ) -> None:
        result = Pipeline(sample_config).run(sample_df)
        assert "Exam_Score" not in result.report.scaled_columns
        pd.testing.assert_series_equal(
            result.ml["Exam_Score"].reset_index(drop=True),
            result.cleaned["Exam_Score"].reset_index(drop=True),
        )

    def test_input_is_not_mutated(self, sample_df: pd.DataFrame) -> None:
        snapshot = sample_df.copy()
        Pipeline().run(sample_df)
        pd.testing.assert_frame_equal(sample_df, snapshot)

    def test_report_is_json_serialisable(
        self, sample_df: pd.DataFrame, sample_config: PipelineConfig
    ) -> None:
        payload = Pipeline(sample_config).run(sample_df).report.model_dump_json()
        assert json.loads(payload)["duplicates_removed"] == 1


class TestOrderingAndToggles:
    def test_standardize_before_dedupe_catches_case_duplicates(self) -> None:
        df = pd.DataFrame({"g": ["Male", "male", " MALE"], "n": [1, 1, 1]})
        result = Pipeline().run(df)
        assert result.report.duplicates_removed == 2
        assert len(result.cleaned) == 1

    def test_all_toggles_off(self, sample_df: pd.DataFrame) -> None:
        off = dict.fromkeys(
            (
                "coerce_numeric",
                "standardize_categories",
                "remove_duplicates",
                "detect_outliers",
                "handle_missing",
                "create_features",
                "apply_encoding",
                "apply_scaling",
            ),
            False,
        )
        result = Pipeline(PipelineConfig(**off)).run(sample_df)
        report = result.report

        assert report.steps == []
        assert report.duplicates_removed == 0
        assert report.missing_fixed == 0
        assert report.outliers == {}
        assert report.encoded_columns == []
        assert report.scaled_columns == []
        assert "MALE" in result.cleaned["Gender"].tolist()

    def test_non_string_column_names_do_not_crash(self) -> None:
        df = pd.DataFrame({1: [1, 2, 3, 4], "b": ["x", "y", "x", "y"]})
        assert len(Pipeline().run(df).cleaned) == 4

    def test_custom_step_list(self, sample_df: pd.DataFrame) -> None:
        pipeline = Pipeline(steps=build_steps(["deduplicate"]))
        result = pipeline.run(sample_df)
        assert [s.name for s in result.report.steps] == ["deduplicate"]
        assert result.report.duplicates_removed == 1

    def test_warnings_surface_in_report(self) -> None:
        df = pd.DataFrame({"t": ["a", "b", "c", "d"]})
        result = Pipeline(PipelineConfig(feature_source="t")).run(df)
        assert any("not numeric" in w for w in result.report.warnings)


class TestValidation:
    def test_empty_dataset(self) -> None:
        with pytest.raises(InvalidInputError):
            Pipeline().run(pd.DataFrame())

    def test_duplicate_column_names(self) -> None:
        df = pd.DataFrame([[1, 2]], columns=["a", "a"])
        with pytest.raises(InvalidInputError, match="Duplicate column"):
            Pipeline().run(df)

    def test_unknown_target(self, sample_df: pd.DataFrame) -> None:
        with pytest.raises(InvalidConfigError, match="target"):
            Pipeline(PipelineConfig(target="nope")).run(sample_df)

    def test_unknown_feature_source(self, sample_df: pd.DataFrame) -> None:
        with pytest.raises(InvalidConfigError, match="feature_source"):
            Pipeline(PipelineConfig(feature_source="nope")).run(sample_df)

    def test_feature_source_ignored_when_feature_step_off(self, sample_df: pd.DataFrame) -> None:
        config = PipelineConfig(feature_source="nope", create_features=False)
        assert Pipeline(config).run(sample_df).report.created_features == []
