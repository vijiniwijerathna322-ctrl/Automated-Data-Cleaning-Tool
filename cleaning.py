"""Pure data-cleaning functions used by the Streamlit app.

Nothing in this module imports Streamlit, so every step can be unit-tested.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# Text columns with more distinct values than this (and mostly unique values)
# are treated as free text / identifiers and are left untouched.
MAX_CATEGORY_UNIQUE = 50
MAX_CATEGORY_UNIQUE_RATIO = 0.5


# =========================================================
# COLUMN HELPERS
# =========================================================

def text_columns(df):
    """Names of text (object / string) columns."""
    return df.select_dtypes(include=["object", "string"]).columns.tolist()


def detect_id_columns(df):
    """Columns whose name looks like an identifier (id, *_id, *identifier*)."""
    id_columns = []

    for column in df.columns:
        name = str(column).lower()

        if name == "id" or name.endswith("_id") or "identifier" in name:
            id_columns.append(column)

    return id_columns


def coerce_numeric_columns(df, skip=()):
    """Convert text columns to numbers when *every* non-null value parses.

    Nothing is converted if even one value would be lost, so this never
    silently turns bad values into NaN.
    """
    df = df.copy()
    converted = []

    for column in text_columns(df):
        if column in skip:
            continue

        values = df[column]
        non_null = values.dropna()

        if non_null.empty:
            continue

        parsed = pd.to_numeric(non_null.astype(str).str.strip(), errors="coerce")

        if parsed.notna().all():
            df[column] = pd.to_numeric(
                values.astype("object").map(
                    lambda v: v.strip() if isinstance(v, str) else v
                ),
                errors="coerce",
            )
            converted.append(column)

    return df, converted


# =========================================================
# CLEANING STEPS
# =========================================================

def _tidy_text(value):
    if isinstance(value, str):
        return value.strip().lower().title()
    return value


def standardize_categories(df, skip=()):
    """Trim whitespace and normalise case in categorical text columns.

    Skips ID columns (via ``skip``) and high-cardinality columns, which are
    almost certainly free text or identifiers. Missing values stay missing.
    Returns ``(dataframe, standardized_column_names)``.
    """
    df = df.copy()
    standardized = []

    for column in text_columns(df):
        if column in skip:
            continue

        unique = df[column].nunique(dropna=True)
        non_null = int(df[column].notna().sum())

        if (
            unique > MAX_CATEGORY_UNIQUE
            and non_null
            and unique / non_null > MAX_CATEGORY_UNIQUE_RATIO
        ):
            continue

        df[column] = df[column].map(_tidy_text)
        standardized.append(column)

    return df, standardized


def impute_missing(df):
    """Median for numeric columns, "Unknown" for text columns.

    Returns ``(dataframe, number_of_values_filled)``.
    """
    df = df.copy()
    filled = 0

    for column in df.select_dtypes(include=np.number).columns:
        count = int(df[column].isnull().sum())

        # An all-empty column has no median; leave it alone.
        if count > 0 and count < len(df):
            df[column] = df[column].fillna(df[column].median())
            filled += count

    for column in text_columns(df):
        count = int(df[column].isnull().sum())

        if count > 0:
            df[column] = df[column].fillna("Unknown")
            filled += count

    return df, filled


def find_outliers(df, columns=None):
    """IQR (1.5x) outlier report for numeric columns."""
    results = {}

    if columns is None:
        columns = df.select_dtypes(include=np.number).columns

    for column in columns:
        values = df[column].dropna()

        if len(values) < 4:
            continue

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            continue

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        mask = (df[column] < lower) | (df[column] > upper)
        count = int(mask.sum())

        if count > 0:
            results[column] = {
                "count": count,
                "lower": lower,
                "upper": upper,
                "rows": df[mask],
            }

    return results


# =========================================================
# FEATURE ENGINEERING
# =========================================================

def _score_category(score):
    if pd.isna(score):
        return "Unknown"
    if score >= 80:
        return "Excellent"
    if score >= 70:
        return "Good"
    if score >= 60:
        return "Average"
    return "Needs Improvement"


def create_features(df, source_column=None):
    """Bin one numeric column into a categorical feature.

    ``Exam_Score`` uses fixed grade bands; any other column is split into
    quartile bands (Low / Medium-Low / Medium-High / High).

    Returns ``(dataframe, created_feature_names, {feature: source_column})``.
    """
    df = df.copy()
    created = []
    derived_from = {}

    if source_column is None or source_column not in df.columns:
        return df, created, derived_from

    if not pd.api.types.is_numeric_dtype(df[source_column]):
        return df, created, derived_from

    name = f"{source_column}_Category"

    if name in df.columns:
        return df, created, derived_from

    if source_column == "Exam_Score":
        df[name] = df[source_column].apply(_score_category)
    else:
        labels = ["Low", "Medium-Low", "Medium-High", "High"]

        try:
            binned = pd.qcut(df[source_column], q=4, labels=labels)
        except ValueError:
            # Too few distinct values to form four bins.
            return df, created, derived_from

        df[name] = binned.astype("object").where(binned.notna(), "Unknown")

    created.append(name)
    derived_from[name] = source_column

    return df, created, derived_from


# =========================================================
# ML-READY DATASET
# =========================================================

def build_ml_dataset(
    df,
    id_columns=(),
    target=None,
    derived_from=None,
    encode=True,
    scale=True,
):
    """One-hot encode and scale a cleaned dataframe for modelling.

    * ID columns are dropped (they carry no signal and leak row identity).
    * The target column is never encoded or scaled.
    * Engineered features derived from the target are dropped (leakage).
    * Only the original numeric columns are scaled, not the 0/1 dummies.

    Returns ``(ml_df, encoded_columns, scaled_columns, dropped_columns)``.
    """
    derived_from = derived_from or {}
    ml_df = df.copy()

    dropped = [c for c in id_columns if c in ml_df.columns]

    if target is not None:
        dropped += [
            feature
            for feature, source in derived_from.items()
            if source == target and feature in ml_df.columns
        ]

    ml_df = ml_df.drop(columns=dropped)

    protected = {target} if target is not None else set()

    encoded_columns = []
    scaled_columns = []

    numeric_columns = [
        c
        for c in ml_df.select_dtypes(include=np.number).columns
        if c not in protected
    ]

    if scale and numeric_columns:
        ml_df[numeric_columns] = StandardScaler().fit_transform(
            ml_df[numeric_columns]
        )
        scaled_columns = numeric_columns

    if encode:
        encoded_columns = [c for c in text_columns(ml_df) if c not in protected]

        if encoded_columns:
            ml_df = pd.get_dummies(ml_df, columns=encoded_columns, dtype=int)

    return ml_df, encoded_columns, scaled_columns, dropped


# =========================================================
# FULL PIPELINE
# =========================================================

@dataclass
class Options:
    handle_missing: bool = True
    remove_duplicates: bool = True
    standardize_categories: bool = True
    detect_outliers: bool = True
    create_features: bool = True
    apply_encoding: bool = True
    apply_scaling: bool = True
    target: str | None = None
    feature_source: str | None = None


@dataclass
class Result:
    cleaned: pd.DataFrame
    ml: pd.DataFrame
    missing_fixed: int = 0
    duplicates_removed: int = 0
    created_features: list = field(default_factory=list)
    outliers: dict = field(default_factory=dict)
    converted_columns: list = field(default_factory=list)
    standardized_columns: list = field(default_factory=list)
    id_columns: list = field(default_factory=list)
    encoded_columns: list = field(default_factory=list)
    scaled_columns: list = field(default_factory=list)
    dropped_from_ml: list = field(default_factory=list)


def run_pipeline(original, options):
    """Run every enabled step, in a deliberate order.

    standardize -> deduplicate -> outliers -> impute -> features -> ML dataset

    Standardizing first lets "Male" / "male" duplicates be caught, and
    deduplicating before imputing avoids creating false duplicates out of
    rows that merely share a filled-in median.
    """
    df = original.copy()
    result = Result(cleaned=df, ml=df)

    result.id_columns = detect_id_columns(df)

    df, result.converted_columns = coerce_numeric_columns(
        df, skip=result.id_columns
    )

    if options.standardize_categories:
        df, result.standardized_columns = standardize_categories(
            df, skip=result.id_columns
        )

    if options.remove_duplicates:
        before = len(df)
        df = df.drop_duplicates()
        result.duplicates_removed = before - len(df)

    if options.detect_outliers:
        result.outliers = find_outliers(df)

    if options.handle_missing:
        df, result.missing_fixed = impute_missing(df)

    derived_from = {}

    if options.create_features:
        df, result.created_features, derived_from = create_features(
            df, options.feature_source
        )

    target = options.target if options.target in df.columns else None

    (
        result.ml,
        result.encoded_columns,
        result.scaled_columns,
        result.dropped_from_ml,
    ) = build_ml_dataset(
        df,
        id_columns=result.id_columns,
        target=target,
        derived_from=derived_from,
        encode=options.apply_encoding,
        scale=options.apply_scaling,
    )

    result.cleaned = df

    return result
