import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cleaning  # noqa: E402
from cleaning import Options, run_pipeline  # noqa: E402


def sample():
    return pd.read_csv(ROOT / "sample_data.csv")


class HelperTests(unittest.TestCase):

    def test_detect_id_columns(self):
        df = pd.DataFrame(columns=["Student_ID", "id", "identifier_x", "Age", 5])
        self.assertEqual(
            cleaning.detect_id_columns(df), ["Student_ID", "id", "identifier_x"]
        )

    def test_standardize_skips_ids_and_keeps_nan(self):
        df = pd.DataFrame({
            "code": ["abc1", "abc2", "abc3"],
            "g": [" MALE", "female ", None],
        })
        out, done = cleaning.standardize_categories(df, skip=["code"])
        self.assertEqual(out["code"].tolist(), ["abc1", "abc2", "abc3"])
        self.assertEqual(out["g"].tolist()[:2], ["Male", "Female"])
        self.assertTrue(pd.isna(out["g"].iloc[2]))
        self.assertEqual(done, ["g"])

    def test_standardize_skips_free_text(self):
        df = pd.DataFrame({"note": [f"unique text {i}" for i in range(80)]})
        out, done = cleaning.standardize_categories(df)
        self.assertEqual(done, [])
        self.assertEqual(out["note"].iloc[0], "unique text 0")

    def test_coerce_numeric_is_lossless(self):
        df = pd.DataFrame({"a": ["1", " 2", None], "b": ["1", "x", "3"]})
        out, converted = cleaning.coerce_numeric_columns(df)
        self.assertEqual(converted, ["a"])
        self.assertEqual(out["a"].dropna().tolist(), [1, 2])
        self.assertEqual(out["b"].tolist(), ["1", "x", "3"])

    def test_impute_missing(self):
        df = pd.DataFrame({"n": [1.0, None, 3.0], "t": ["a", None, "b"]})
        out, filled = cleaning.impute_missing(df)
        self.assertEqual(filled, 2)
        self.assertEqual(out["n"].tolist(), [1.0, 2.0, 3.0])
        self.assertEqual(out["t"].tolist(), ["a", "Unknown", "b"])

    def test_impute_leaves_all_empty_column(self):
        df = pd.DataFrame({"n": [np.nan, np.nan, np.nan]})
        out, filled = cleaning.impute_missing(df)
        self.assertEqual(filled, 0)
        self.assertTrue(out["n"].isna().all())

    def test_find_outliers(self):
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 100]})
        res = cleaning.find_outliers(df)
        self.assertEqual(res["x"]["count"], 1)

    def test_create_features_exam_score_and_generic(self):
        df = pd.DataFrame({"Exam_Score": [90, 75, 65, 10, None]})
        out, created, derived = cleaning.create_features(df, "Exam_Score")
        self.assertEqual(created, ["Exam_Score_Category"])
        self.assertEqual(
            out["Exam_Score_Category"].tolist(),
            ["Excellent", "Good", "Average", "Needs Improvement", "Unknown"],
        )
        self.assertEqual(derived, {"Exam_Score_Category": "Exam_Score"})

        df = pd.DataFrame({"v": list(range(20))})
        out, created, _ = cleaning.create_features(df, "v")
        self.assertEqual(created, ["v_Category"])
        self.assertEqual(out["v_Category"].nunique(), 4)

    def test_create_features_without_source_is_noop(self):
        df = pd.DataFrame({"v": [1, 2, 3]})
        out, created, derived = cleaning.create_features(df, None)
        self.assertEqual((created, derived), ([], {}))
        self.assertEqual(list(out.columns), ["v"])


class PipelineTests(unittest.TestCase):

    def run_sample(self, **kwargs):
        return run_pipeline(
            sample(),
            Options(target="Exam_Score", feature_source="Exam_Score", **kwargs),
        )

    def test_default_pipeline_on_sample(self):
        res = self.run_sample()
        self.assertEqual(res.duplicates_removed, 1)
        self.assertEqual(res.missing_fixed, 4)
        self.assertEqual(int(res.cleaned.isnull().sum().sum()), 0)
        self.assertEqual(
            sorted(res.cleaned["Gender"].unique()), ["Female", "Male"]
        )
        self.assertIn("Study_Hours", res.outliers)
        self.assertEqual(res.created_features, ["Exam_Score_Category"])

    def test_no_target_leakage_or_ids_in_ml_dataset(self):
        res = self.run_sample()
        cols = res.ml.columns.tolist()
        self.assertFalse(any(c.startswith("Exam_Score_Category") for c in cols))
        self.assertNotIn("Student_ID", cols)
        self.assertEqual(
            sorted(res.dropped_from_ml), ["Exam_Score_Category", "Student_ID"]
        )

    def test_target_is_not_scaled(self):
        res = self.run_sample()
        self.assertNotIn("Exam_Score", res.scaled_columns)
        pd.testing.assert_series_equal(
            res.ml["Exam_Score"].reset_index(drop=True),
            res.cleaned["Exam_Score"].reset_index(drop=True),
        )

    def test_dummies_are_not_scaled(self):
        res = self.run_sample()
        dummies = [c for c in res.ml.columns if c.startswith("Gender_")]
        self.assertTrue(dummies)
        for c in dummies:
            self.assertTrue(set(res.ml[c].unique()) <= {0, 1})

    def test_standardize_before_dedupe_catches_case_duplicates(self):
        df = pd.DataFrame({"g": ["Male", "male", " MALE"], "n": [1, 1, 1]})
        res = run_pipeline(df, Options())
        self.assertEqual(res.duplicates_removed, 2)
        self.assertEqual(len(res.cleaned), 1)

    def test_toggles_are_respected(self):
        off = dict(
            handle_missing=False,
            remove_duplicates=False,
            standardize_categories=False,
            detect_outliers=False,
            create_features=False,
            apply_encoding=False,
            apply_scaling=False,
        )
        res = run_pipeline(sample(), Options(**off))
        self.assertEqual(res.duplicates_removed, 0)
        self.assertEqual(res.missing_fixed, 0)
        self.assertEqual(res.outliers, {})
        self.assertEqual(res.created_features, [])
        self.assertEqual(res.encoded_columns, [])
        self.assertEqual(res.scaled_columns, [])
        # Standardization really is off: raw casing survives.
        self.assertIn("MALE", res.cleaned["Gender"].tolist())

    def test_non_string_column_names_do_not_crash(self):
        df = pd.DataFrame({1: [1, 2, 3, 4], "b": ["x", "y", "x", "y"]})
        res = run_pipeline(df, Options())
        self.assertEqual(len(res.cleaned), 4)


if __name__ == "__main__":
    unittest.main()
