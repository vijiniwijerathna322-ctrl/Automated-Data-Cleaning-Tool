"""Streamlit UI. All data work happens in the REST API; this file is presentation only."""

from __future__ import annotations

import json
from typing import Any

import streamlit as st

from datacleaner.core.config import PipelineConfig
from datacleaner.ui import api_client
from datacleaner.ui.components.charts import histogram_chart, missing_values_chart
from datacleaner.ui.components.tables import preview_caption, preview_to_frame, report_table

NONE_LABEL = "(none)"
FORMATS = {"CSV": "csv", "Excel (.xlsx)": "xlsx", "Parquet": "parquet"}

st.set_page_config(page_title="Data Cleaning & Feature Engineering", page_icon="🧹", layout="wide")

st.markdown(
    """
<style>
.main-title { font-size: 42px; font-weight: 700; margin-bottom: 5px; }
.subtitle { font-size: 18px; opacity: 0.75; margin-bottom: 25px; }
.section-title { font-size: 26px; font-weight: 650; margin-top: 20px; }
</style>
""",
    unsafe_allow_html=True,
)


def section(title: str) -> None:
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)


# ------------------------------------------------------------------ API access
# Cached so that widget clicks (which rerun the whole script) don't re-call the API.
@st.cache_data(show_spinner="Profiling dataset…")
def fetch_profile(content: bytes, filename: str) -> dict[str, Any]:
    return api_client.default_client().profile(content, filename)


@st.cache_data(show_spinner="Cleaning dataset…")
def fetch_clean(content: bytes, filename: str, config: dict[str, Any]) -> dict[str, Any]:
    return api_client.default_client().clean(content, filename, config)


@st.cache_data(show_spinner=False)
def fetch_download(
    content: bytes, filename: str, config: dict[str, Any], dataset: str, fmt: str
) -> bytes:
    return api_client.default_client().download(content, filename, config, dataset, fmt)


# ----------------------------------------------------------------------- header
st.markdown(
    '<div class="main-title">🧹 Automated Data Cleaning & Feature Engineering</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">Clean, transform and prepare your dataset for Machine Learning.</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------- sidebar
st.sidebar.title("⚙️ Cleaning Options")
st.sidebar.caption("Choose which preprocessing steps should be applied.")

opt_coerce = st.sidebar.checkbox("🔢 Convert Numeric Text", value=True)
opt_standardize = st.sidebar.checkbox("🔤 Standardize Categories", value=True)
opt_duplicates = st.sidebar.checkbox("♻️ Remove Duplicate Rows", value=True)
opt_outliers = st.sidebar.checkbox("📊 Detect Outliers", value=True)
opt_missing = st.sidebar.checkbox("🕳️ Handle Missing Values", value=True)
opt_features = st.sidebar.checkbox("🛠️ Feature Engineering", value=True)
opt_encoding = st.sidebar.checkbox("🔢 Categorical Encoding", value=True)
opt_scaling = st.sidebar.checkbox("📏 Numerical Scaling", value=True)

# ----------------------------------------------------------------------- upload
uploaded = st.file_uploader("📂 Upload your dataset", type=["csv", "xlsx", "parquet"])

if uploaded is None:
    st.info("👆 Upload a CSV, Excel or Parquet file above to start.")
    st.markdown(
        """
        ### 🚀 What this tool does

        | Step | Process |
        |---|---|
        | 1 | 🔍 Dataset profiling |
        | 2 | 🔢 Numeric conversion of text columns |
        | 3 | 🔤 Category standardization |
        | 4 | ♻️ Duplicate removal |
        | 5 | 📊 Outlier detection |
        | 6 | 🕳️ Missing value handling |
        | 7 | 🛠️ Feature engineering |
        | 8 | 🔢 Categorical encoding & 📏 scaling (ML-ready) |
        | 9 | 📋 Cleaning report |
        | 10 | 📥 Export cleaned / ML-ready data |

        **Built with:** FastAPI • Pandas • Scikit-learn • Pydantic • Streamlit • Plotly
        """
    )
    st.stop()

content, filename = uploaded.getvalue(), uploaded.name

try:
    profile_response = fetch_profile(content, filename)
except api_client.ApiError as error:
    st.error(f"❌ {error.title}: {error.detail}")
    st.stop()

profile = profile_response["profile"]
column_profiles = profile["column_profiles"]
all_columns = [c["name"] for c in column_profiles]
numeric_columns = profile["numeric_columns"]

# --------------------------------------------------------- column choices (sidebar)
st.sidebar.divider()
st.sidebar.caption(
    "The target column is kept as-is (never encoded or scaled), and features derived "
    "from it are left out of the ML-ready dataset to avoid target leakage."
)

target_choices = [NONE_LABEL, *all_columns]
target_choice = st.sidebar.selectbox(
    "🎯 Target column (optional)",
    target_choices,
    index=target_choices.index("Exam_Score") if "Exam_Score" in target_choices else 0,
)

source_choices = [NONE_LABEL, *numeric_columns]
source_choice = st.sidebar.selectbox(
    "🛠️ Numeric column to bin into categories",
    source_choices,
    index=source_choices.index("Exam_Score") if "Exam_Score" in source_choices else 0,
    disabled=not opt_features,
)

output_label = st.sidebar.selectbox("💾 Download format", list(FORMATS))
output_format = FORMATS[output_label]

config = PipelineConfig(
    coerce_numeric=opt_coerce,
    standardize_categories=opt_standardize,
    remove_duplicates=opt_duplicates,
    detect_outliers=opt_outliers,
    handle_missing=opt_missing,
    create_features=opt_features,
    apply_encoding=opt_encoding,
    apply_scaling=opt_scaling,
    target=None if target_choice == NONE_LABEL else target_choice,
    feature_source=None if source_choice == NONE_LABEL else source_choice,
).model_dump()

# -------------------------------------------------------------------- run clean
try:
    result = fetch_clean(content, filename, config)
except api_client.ApiError as error:
    st.error(f"❌ {error.title}: {error.detail}")
    st.stop()

report = result["report"]
before, after = report["before"], report["after"]

section("📊 Data Quality Dashboard")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", after["rows"], delta=after["rows"] - before["rows"])
c2.metric("Columns", after["columns"], delta=after["columns"] - before["columns"])
c3.metric(
    "Missing Values",
    after["missing_values"],
    delta=after["missing_values"] - before["missing_values"],
)
c4.metric(
    "Duplicates",
    after["duplicate_rows"],
    delta=after["duplicate_rows"] - before["duplicate_rows"],
)

section("✨ Processing Summary")
s1, s2, s3, s4 = st.columns(4)
s1.metric("Missing Fixed", report["missing_fixed"])
s2.metric("Duplicates Removed", report["duplicates_removed"])
s3.metric("Features Created", len(report["created_features"]))
s4.metric("Potential Outlier Columns", len(report["outliers"]))

if report["converted_columns"]:
    st.info("Converted text columns to numbers: " + ", ".join(report["converted_columns"]))

for warning in report["warnings"]:
    st.warning(warning)

# ----------------------------------------------------------------------- charts
section("📈 Data Quality Analysis")
chart1, chart2 = st.columns(2)

with chart1:
    figure = missing_values_chart(column_profiles)

    if figure is not None:
        st.plotly_chart(figure, width="stretch")
    else:
        st.success("No missing values found in the original dataset.")

with chart2:
    if numeric_columns:
        selected = st.selectbox("Select numerical column", numeric_columns)
        profile_for_column = next(c for c in column_profiles if c["name"] == selected)
        figure = histogram_chart(profile_for_column)

        if figure is not None:
            st.plotly_chart(figure, width="stretch")
        else:
            st.info("This column has no values to plot.")
    else:
        st.info("No numerical columns available.")

# ---------------------------------------------------------------------- preview
section("📋 Dataset Preview")
tab_raw, tab_clean, tab_ml = st.tabs(
    ["📄 Raw Dataset", "✨ Cleaned Dataset", "🤖 ML-Ready Dataset"]
)

with tab_raw:
    st.dataframe(preview_to_frame(profile_response["preview"]), width="stretch", height=400)
    caption = preview_caption(profile_response["preview"])
    if caption:
        st.caption(caption)

with tab_clean:
    st.dataframe(preview_to_frame(result["cleaned"]), width="stretch", height=400)
    caption = preview_caption(result["cleaned"])
    if caption:
        st.caption(caption)

with tab_ml:
    if report["dropped_from_ml"]:
        st.caption(
            "Left out of this dataset: "
            + ", ".join(f"`{c}`" for c in report["dropped_from_ml"])
            + " (ID columns and features derived from the target)."
        )

    st.dataframe(preview_to_frame(result["ml"]), width="stretch", height=400)
    caption = preview_caption(result["ml"])
    if caption:
        st.caption(caption)

# --------------------------------------------------------------------- outliers
section("⚠️ Outlier Analysis")

if report["outliers"]:
    st.warning("Potential outliers were detected. They are reported but not automatically removed.")

    for column, summary in report["outliers"].items():
        with st.expander(f"{column} — {summary['count']} potential outliers"):
            st.write(f"Lower Bound: `{summary['lower']:.2f}`")
            st.write(f"Upper Bound: `{summary['upper']:.2f}`")

            if column in result["outlier_rows"]:
                st.dataframe(preview_to_frame(result["outlier_rows"][column]), width="stretch")
elif opt_outliers:
    st.success("No potential outliers detected.")
else:
    st.info("Outlier detection is turned off.")

if report["created_features"]:
    section("🛠️ Feature Engineering")
    st.success("Created features: " + ", ".join(report["created_features"]))

# ----------------------------------------------------------------------- report
section("📋 Cleaning Report")
st.dataframe(report_table(report), width="stretch")

with st.expander("Step timings"):
    st.dataframe(report["steps"], width="stretch")

# -------------------------------------------------------------------- downloads
section("⬇️ Download Results")

try:
    cleaned_bytes = fetch_download(content, filename, config, "cleaned", output_format)
    ml_bytes = fetch_download(content, filename, config, "ml", output_format)
except api_client.ApiError as error:
    st.error(f"❌ {error.title}: {error.detail}")
    st.stop()

mime = {
    "csv": "text/csv",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "parquet": "application/vnd.apache.parquet",
}[output_format]

d1, d2, d3 = st.columns(3)
d1.download_button(
    "📥 Download Cleaned Data",
    cleaned_bytes,
    f"cleaned_dataset.{output_format}",
    mime,
    width="stretch",
)
d2.download_button(
    "🤖 Download ML-Ready Data",
    ml_bytes,
    f"ml_ready_dataset.{output_format}",
    mime,
    width="stretch",
)

d3.download_button(
    "📋 Download Report (JSON)",
    json.dumps(report, indent=2),
    "cleaning_report.json",
    "application/json",
    width="stretch",
)

st.success("🎉 Dataset processing completed successfully!")
