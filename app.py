import io
from dataclasses import asdict

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from cleaning import Options, run_pipeline


NONE_LABEL = "(none)"


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Data Cleaning & Feature Engineering",
    page_icon="🧹",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 18px;
    opacity: 0.75;
    margin-bottom: 25px;
}

.section-title {
    font-size: 26px;
    font-weight: 650;
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)


def section(title):
    st.markdown(
        f'<div class="section-title">{title}</div>',
        unsafe_allow_html=True
    )


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🧹 Automated Data Cleaning & Feature Engineering</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Clean, transform and prepare your dataset for Machine Learning.'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# CACHED HELPERS
# (the whole script reruns on every widget click, so cache the heavy work)
# =========================================================

@st.cache_data(show_spinner=False)
def load_csv(raw_bytes):
    df = pd.read_csv(io.BytesIO(raw_bytes))
    df.columns = df.columns.astype(str)
    return df


@st.cache_data(show_spinner=False)
def process(df, option_values):
    return run_pipeline(df, Options(**option_values))


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("⚙️ Cleaning Options")

st.sidebar.caption(
    "Choose which preprocessing steps should be applied."
)

# NOTE: these names must not collide with the helper functions in cleaning.py.
opt_missing = st.sidebar.checkbox("🕳️ Handle Missing Values", value=True)
opt_duplicates = st.sidebar.checkbox("♻️ Remove Duplicate Rows", value=True)
opt_standardize = st.sidebar.checkbox("🔤 Standardize Categories", value=True)
opt_outliers = st.sidebar.checkbox("📊 Detect Outliers", value=True)
opt_features = st.sidebar.checkbox("🛠️ Feature Engineering", value=True)
opt_encoding = st.sidebar.checkbox("🔢 Categorical Encoding", value=True)
opt_scaling = st.sidebar.checkbox("📏 Numerical Scaling", value=True)


# =========================================================
# FILE UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "📂 Upload your CSV dataset",
    type=["csv"]
)


# =========================================================
# MAIN APP
# =========================================================

if uploaded_file is not None:

    try:

        original_df = load_csv(uploaded_file.getvalue())

        if original_df.empty:
            st.error("The uploaded dataset is empty.")
            st.stop()

        # -------------------------------------------------
        # COLUMN CHOICES (sidebar)
        # -------------------------------------------------

        all_columns = original_df.columns.tolist()

        numeric_columns_original = (
            original_df.select_dtypes(include=np.number).columns.tolist()
        )

        st.sidebar.divider()

        st.sidebar.caption(
            "The target column is kept as-is (never encoded or scaled), and "
            "features derived from it are left out of the ML-ready dataset "
            "to avoid target leakage."
        )

        target_choices = [NONE_LABEL] + all_columns

        target_choice = st.sidebar.selectbox(
            "🎯 Target column (optional)",
            target_choices,
            index=(
                target_choices.index("Exam_Score")
                if "Exam_Score" in target_choices
                else 0
            )
        )

        source_choices = [NONE_LABEL] + numeric_columns_original

        source_choice = st.sidebar.selectbox(
            "🛠️ Numeric column to bin into categories",
            source_choices,
            index=(
                source_choices.index("Exam_Score")
                if "Exam_Score" in source_choices
                else 0
            ),
            disabled=not opt_features
        )

        # -------------------------------------------------
        # RUN PIPELINE
        # -------------------------------------------------

        options = Options(
            handle_missing=opt_missing,
            remove_duplicates=opt_duplicates,
            standardize_categories=opt_standardize,
            detect_outliers=opt_outliers,
            create_features=opt_features,
            apply_encoding=opt_encoding,
            apply_scaling=opt_scaling,
            target=None if target_choice == NONE_LABEL else target_choice,
            feature_source=None if source_choice == NONE_LABEL else source_choice
        )

        result = process(original_df, asdict(options))

        df = result.cleaned
        ml_df = result.ml

        # -------------------------------------------------
        # ORIGINAL / FINAL METRICS
        # -------------------------------------------------

        original_rows = len(original_df)
        original_columns = len(original_df.columns)
        original_missing = int(original_df.isnull().sum().sum())
        original_duplicates = int(original_df.duplicated().sum())

        final_rows = len(df)
        final_columns = len(df.columns)
        final_missing = int(df.isnull().sum().sum())
        final_duplicates = int(df.duplicated().sum())

        section("📊 Data Quality Dashboard")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Rows", final_rows, delta=final_rows - original_rows)

        col2.metric(
            "Columns",
            final_columns,
            delta=final_columns - original_columns
        )

        col3.metric(
            "Missing Values",
            final_missing,
            delta=final_missing - original_missing
        )

        col4.metric(
            "Duplicates",
            final_duplicates,
            delta=final_duplicates - original_duplicates
        )

        # -------------------------------------------------
        # PROCESSING SUMMARY
        # -------------------------------------------------

        section("✨ Processing Summary")

        s1, s2, s3, s4 = st.columns(4)

        s1.metric("Missing Fixed", result.missing_fixed)
        s2.metric("Duplicates Removed", result.duplicates_removed)
        s3.metric("Features Created", len(result.created_features))
        s4.metric("Potential Outlier Columns", len(result.outliers))

        if result.converted_columns:
            st.info(
                "Converted text columns to numbers: "
                + ", ".join(result.converted_columns)
            )

        # -------------------------------------------------
        # DATA QUALITY CHARTS
        # -------------------------------------------------

        section("📈 Data Quality Analysis")

        chart1, chart2 = st.columns(2)

        with chart1:

            missing_data = (
                original_df.isnull().sum().reset_index()
            )

            missing_data.columns = ["Column", "Missing Values"]

            missing_data = missing_data[missing_data["Missing Values"] > 0]

            if not missing_data.empty:

                fig_missing = px.bar(
                    missing_data,
                    x="Column",
                    y="Missing Values",
                    title="Missing Values by Column"
                )

                fig_missing.update_layout(height=350)

                st.plotly_chart(fig_missing, width="stretch")

            else:

                st.success(
                    "No missing values found in the original dataset."
                )

        with chart2:

            if numeric_columns_original:

                selected_column = st.selectbox(
                    "Select numerical column",
                    numeric_columns_original
                )

                fig_distribution = px.histogram(
                    original_df,
                    x=selected_column,
                    title=f"{selected_column} Distribution"
                )

                fig_distribution.update_layout(height=350)

                st.plotly_chart(fig_distribution, width="stretch")

            else:

                st.info("No numerical columns available.")

        # -------------------------------------------------
        # DATA PREVIEW TABS
        # -------------------------------------------------

        section("📋 Dataset Preview")

        tab1, tab2, tab3 = st.tabs([
            "📄 Raw Dataset",
            "✨ Cleaned Dataset",
            "🤖 ML-Ready Dataset"
        ])

        with tab1:
            st.dataframe(original_df, width="stretch", height=400)

        with tab2:
            st.dataframe(df, width="stretch", height=400)

        with tab3:

            if result.dropped_from_ml:
                st.caption(
                    "Left out of this dataset: "
                    + ", ".join(f"`{c}`" for c in result.dropped_from_ml)
                    + " (ID columns and features derived from the target)."
                )

            st.dataframe(ml_df, width="stretch", height=400)

        # -------------------------------------------------
        # OUTLIERS
        # -------------------------------------------------

        section("⚠️ Outlier Analysis")

        if result.outliers:

            st.warning(
                "Potential outliers were detected. "
                "They are reported but not automatically removed."
            )

            for column, outlier in result.outliers.items():

                with st.expander(
                    f"{column} — {outlier['count']} potential outliers"
                ):

                    st.write(f"Lower Bound: `{outlier['lower']:.2f}`")
                    st.write(f"Upper Bound: `{outlier['upper']:.2f}`")

                    st.dataframe(outlier["rows"], width="stretch")

        elif opt_outliers:

            st.success("No potential outliers detected.")

        else:

            st.info("Outlier detection is turned off.")

        # -------------------------------------------------
        # FEATURE ENGINEERING
        # -------------------------------------------------

        if result.created_features:

            section("🛠️ Feature Engineering")

            st.success(
                "Created features: " + ", ".join(result.created_features)
            )

        # -------------------------------------------------
        # CLEANING REPORT
        # -------------------------------------------------

        section("📋 Cleaning Report")

        report = pd.DataFrame({
            "Metric": [
                "Original Rows",
                "Final Rows",
                "Original Columns",
                "Final Columns",
                "Missing Values Before",
                "Missing Values After",
                "Missing Values Fixed",
                "Duplicates Before",
                "Duplicates Removed",
                "Potential Outlier Columns",
                "Features Created",
                "Encoded Columns",
                "Scaled Columns",
                "Columns Left Out of ML Dataset"
            ],
            "Value": [
                original_rows,
                final_rows,
                original_columns,
                final_columns,
                original_missing,
                final_missing,
                result.missing_fixed,
                original_duplicates,
                result.duplicates_removed,
                len(result.outliers),
                len(result.created_features),
                len(result.encoded_columns),
                len(result.scaled_columns),
                len(result.dropped_from_ml)
            ]
        })

        st.dataframe(report, width="stretch")

        # -------------------------------------------------
        # DOWNLOAD
        # -------------------------------------------------

        section("⬇️ Download Results")

        download1, download2 = st.columns(2)

        with download1:

            st.download_button(
                "📥 Download Cleaned CSV",
                df.to_csv(index=False),
                "cleaned_dataset.csv",
                "text/csv",
                width="stretch"
            )

        with download2:

            st.download_button(
                "🤖 Download ML-Ready CSV",
                ml_df.to_csv(index=False),
                "ml_ready_dataset.csv",
                "text/csv",
                width="stretch"
            )

        st.success("🎉 Dataset processing completed successfully!")

    except Exception as error:

        st.error("❌ Something went wrong while processing the dataset.")

        st.exception(error)


# =========================================================
# NO FILE
# =========================================================

else:

    st.info("👆 Upload a CSV dataset above to start.")

    st.markdown(
        """
        ### 🚀 What this tool does

        | Step | Process |
        |---|---|
        | 1 | 🔍 Dataset inspection |
        | 2 | 🔤 Category standardization |
        | 3 | ♻️ Duplicate removal |
        | 4 | 📊 Outlier detection |
        | 5 | 🕳️ Missing value handling |
        | 6 | 🛠️ Feature engineering |
        | 7 | 🔢 Categorical encoding |
        | 8 | 📏 Numerical scaling |
        | 9 | 📋 Cleaning report |
        | 10 | 📥 Export cleaned / ML-ready data |

        **Built with:** Python • Pandas • NumPy • Scikit-learn • Streamlit • Plotly
        """
    )
