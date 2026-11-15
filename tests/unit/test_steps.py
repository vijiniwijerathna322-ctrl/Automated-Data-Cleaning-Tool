import numpy as np
import pandas as pd
import pytest

from datacleaner.core.columns import detect_id_columns, numeric_columns, text_columns
from datacleaner.core.config import PipelineConfig
from datacleaner.core.errors import InvalidConfigError
from datacleaner.core.steps import DEFAULT_STEP_ORDER, STEP_REGISTRY, build_steps
from datacleaner.core.steps.base import PipelineContext
from datacleaner.core.steps.coercion import NumericCoercionStep, coerce_numeric_columns
from datacleaner.core.steps.deduplicate import DeduplicateStep
from datacleaner.core.steps.features import FeatureStep, create_category_feature
from datacleaner.core.steps.impute import ImputeStep, impute_missing
from datacleaner.core.steps.ml_dataset import build_ml_dataset
from datacleaner.core.steps.outliers import OutlierStep, find_outliers
from datacleaner.core.steps.standardize import StandardizeStep, standardize_categories


def ctx(**config: object) -> PipelineContext:
    return PipelineContext(config=PipelineConfig(**config))  # type: ignore[arg-type]


class TestColumns:
    def test_detect_id_columns(self) -> None:
        df = pd.DataFrame(columns=["Student_ID", "id", "identifier_x", "Age", 5])
        assert detect_id_columns(df) == ["Student_ID", "id", "identifier_x"]

    def test_text_and_numeric_columns(self) -> None:
        df = pd.DataFrame({"a": [1], "b": ["x"], "c": [1.5]})
        assert numeric_columns(df) == ["a", "c"]
        assert text_columns(df) == ["b"]


class TestCoercion:
    def test_is_lossless(self) -> None:
        df = pd.DataFrame({"a": ["1", " 2", None], "b": ["1", "x", "3"]})
        out, converted = coerce_numeric_columns(df)
        assert converted == ["a"]
        assert out["a"].dropna().tolist() == [1, 2]
        assert out["b"].tolist() == ["1", "x", "3"]

    def test_skips_requested_and_empty_columns(self) -> None:
        df = pd.DataFrame({"id": ["1", "2"], "e": [None, None]})
        _, converted = coerce_numeric_columns(df, skip=["id"])
        assert converted == []

    def test_step_records_context(self) -> None:
        context = ctx()
        NumericCoercionStep().apply(pd.DataFrame({"a": ["1", "2"]}), context)
        assert context.converted_columns == ["a"]


class TestStandardize:
    def test_skips_ids_and_keeps_nan(self) -> None:
        df = pd.DataFrame({"code": ["abc1", "abc2", "abc3"], "g": [" MALE", "female ", None]})
        out, done = standardize_categories(df, skip=["code"])
        assert out["code"].tolist() == ["abc1", "abc2", "abc3"]
        assert out["g"].tolist()[:2] == ["Male", "Female"]
        assert pd.isna(out["g"].iloc[2])
        assert done == ["g"]

    def test_skips_free_text(self) -> None:
        df = pd.DataFrame({"note": [f"unique text {i}" for i in range(80)]})
        out, done = standardize_categories(df)
        assert done == []
        assert out["note"].iloc[0] == "unique text 0"

    def test_thresholds_come_from_config(self) -> None:
        df = pd.DataFrame({"c": [f"v{i}" for i in range(10)]})
        _, done = standardize_categories(df, max_unique=5, max_unique_ratio=0.5)
        assert done == []

    def test_step_uses_context(self) -> None:
        context = ctx()
        context.id_columns = ["code"]
        df = pd.DataFrame({"code": ["aB"], "g": ["xY"]})
        out = StandardizeStep().apply(df, context)
        assert out["g"].iloc[0] == "Xy"
        assert out["code"].iloc[0] == "aB"
        assert context.standardized_columns == ["g"]


class TestDeduplicate:
    def test_counts_removed(self) -> None:
        context = ctx()
        out = DeduplicateStep().apply(pd.DataFrame({"a": [1, 1, 2]}), context)
        assert len(out) == 2
        assert context.duplicates_removed == 1


class TestOutliers:
    def test_find_outliers(self) -> None:
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 100]})
        summaries, rows = find_outliers(df)
        assert summaries["x"].count == 1
        assert rows["x"]["x"].tolist() == [100]

    def test_ignores_small_and_constant_columns(self) -> None:
        assert find_outliers(pd.DataFrame({"few": [1, 2, 3]}))[0] == {}
        assert find_outliers(pd.DataFrame({"const": [5, 5, 5, 5, 5]}))[0] == {}

    def test_multiplier_is_configurable(self) -> None:
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 12]})
        assert "x" in find_outliers(df, multiplier=1.5)[0]
        assert find_outliers(df, multiplier=10)[0] == {}

    def test_step_leaves_data_untouched(self) -> None:
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 100]})
        context = ctx()
        out = OutlierStep().apply(df, context)
        pd.testing.assert_frame_equal(out, df)
        assert "x" in context.outliers


class TestImpute:
    def test_median_and_unknown(self) -> None:
        df = pd.DataFrame({"n": [1.0, None, 3.0], "t": ["a", None, "b"]})
        out, filled = impute_missing(df)
        assert filled == 2
        assert out["n"].tolist() == [1.0, 2.0, 3.0]
        assert out["t"].tolist() == ["a", "Unknown", "b"]

    def test_leaves_all_empty_column(self) -> None:
        out, filled = impute_missing(pd.DataFrame({"n": [np.nan, np.nan, np.nan]}))
        assert filled == 0
        assert out["n"].isna().all()

    def test_step_records_count(self) -> None:
        context = ctx()
        ImputeStep().apply(pd.DataFrame({"n": [1.0, None, 3.0]}), context)
        assert context.missing_fixed == 1


class TestFeatures:
    def test_exam_score_bands(self) -> None:
        df = pd.DataFrame({"Exam_Score": [90, 75, 65, 10, None]})
        out, name, warning = create_category_feature(df, "Exam_Score")
        assert name == "Exam_Score_Category"
        assert warning is None
        assert out[name].tolist() == [
            "Excellent",
            "Good",
            "Average",
            "Needs Improvement",
            "Unknown",
        ]

    def test_generic_quartiles(self) -> None:
        out, name, _ = create_category_feature(pd.DataFrame({"v": list(range(20))}), "v")
        assert name == "v_Category"
        assert out["v_Category"].nunique() == 4

    def test_non_numeric_source_warns(self) -> None:
        _, name, warning = create_category_feature(pd.DataFrame({"t": ["a", "b"]}), "t")
        assert name is None
        assert warning is not None
        assert "not numeric" in warning

    def test_too_few_distinct_values_warns(self) -> None:
        _, name, warning = create_category_feature(pd.DataFrame({"v": [1, 1, 1, 1]}), "v")
        assert name is None
        assert warning is not None

    def test_existing_column_not_overwritten(self) -> None:
        df = pd.DataFrame({"v": range(8), "v_Category": "keep"})
        out, name, warning = create_category_feature(df, "v")
        assert name is None
        assert warning is not None
        assert (out["v_Category"] == "keep").all()

    def test_unknown_source_raises(self) -> None:
        with pytest.raises(InvalidConfigError):
            create_category_feature(pd.DataFrame({"v": [1]}), "nope")

    def test_step_without_source_is_noop(self) -> None:
        df = pd.DataFrame({"v": [1, 2, 3]})
        context = ctx()
        out = FeatureStep().apply(df, context)
        assert list(out.columns) == ["v"]
        assert context.created_features == []

    def test_step_registers_derivation(self) -> None:
        context = ctx(feature_source="v")
        FeatureStep().apply(pd.DataFrame({"v": range(20)}), context)
        assert context.derived_from == {"v_Category": "v"}


class TestMLDataset:
    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Student_ID": ["a", "b", "c", "d"],
                "age": [20.0, 21.0, 22.0, 23.0],
                "score": [50.0, 60.0, 70.0, 80.0],
                "city": ["x", "y", "x", "y"],
                "score_Category": ["lo", "lo", "hi", "hi"],
            }
        )

    def test_drops_ids_and_target_derived_features(self) -> None:
        ml, _, _, dropped = build_ml_dataset(
            self.frame(),
            id_columns=["Student_ID"],
            target="score",
            derived_from={"score_Category": "score"},
        )
        assert sorted(dropped) == ["Student_ID", "score_Category"]
        assert not any(c.startswith("score_Category") for c in ml.columns)

    def test_target_not_scaled_or_encoded(self) -> None:
        df = self.frame()
        ml, encoded, scaled, _ = build_ml_dataset(df, target="score")
        assert "score" not in scaled
        assert "score" not in encoded
        assert ml["score"].tolist() == df["score"].tolist()

    def test_dummies_are_binary_and_not_scaled(self) -> None:
        ml, _, _, _ = build_ml_dataset(self.frame())
        for col in (c for c in ml.columns if c.startswith("city_")):
            assert set(ml[col].unique()) <= {0, 1}

    def test_toggles_off(self) -> None:
        ml, encoded, scaled, _ = build_ml_dataset(self.frame(), encode=False, scale=False)
        assert encoded == []
        assert scaled == []
        assert ml["city"].tolist() == ["x", "y", "x", "y"]


class TestRegistry:
    def test_default_order_is_registered(self) -> None:
        assert set(DEFAULT_STEP_ORDER) == set(STEP_REGISTRY)
        assert [s.name for s in build_steps()] == list(DEFAULT_STEP_ORDER)

    def test_unknown_step_name_rejected(self) -> None:
        with pytest.raises(KeyError):
            build_steps(["nope"])
