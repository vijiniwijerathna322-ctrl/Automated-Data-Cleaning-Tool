import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler


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

.metric-card {
    padding: 18px;
    border-radius: 12px;
    border: 1px solid rgba(128,128,128,0.2);
}

</style>
""", unsafe_allow_html=True)


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
# SIDEBAR
# =========================================================

st.sidebar.title("⚙️ Cleaning Options")

st.sidebar.caption(
    "Choose which preprocessing steps should be applied."
)

handle_missing = st.sidebar.checkbox(
    "🕳️ Handle Missing Values",
    value=True
)

remove_duplicates = st.sidebar.checkbox(
    "♻️ Remove Duplicate Rows",
    value=True
)

standardize_categories = st.sidebar.checkbox(
    "🔤 Standardize Categories",
    value=True
)

detect_outliers = st.sidebar.checkbox(
    "📊 Detect Outliers",
    value=True
)

create_features = st.sidebar.checkbox(
    "🛠️ Feature Engineering",
    value=True
)

apply_encoding = st.sidebar.checkbox(
    "🔢 Categorical Encoding",
    value=True
)

apply_scaling = st.sidebar.checkbox(
    "📏 Numerical Scaling",
    value=True
)


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def detect_id_columns(dataframe):

    id_columns = []

    for column in dataframe.columns:

        name = column.lower()

        if (
            name == "id"
            or name.endswith("_id")
            or "identifier" in name
        ):
            id_columns.append(column)

    return id_columns


def standardize_categories(dataframe):

    df = dataframe.copy()

    categorical_columns = df.select_dtypes(
        include=["object", "string"]
    ).columns

    for column in categorical_columns:

        df[column] = (
            df[column]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
            .str.lower()
            .str.capitalize()
        )

    return df


def create_features(dataframe):

    df = dataframe.copy()

    created = []

    # Student performance feature
    if "Exam_Score" in df.columns:

        def category(score):

            if pd.isna(score):
                return "Unknown"

            if score >= 80:
                return "Excellent"

            elif score >= 70:
                return "Good"

            elif score >= 60:
                return "Average"

            else:
                return "Needs Improvement"

        df["Performance_Category"] = (
            df["Exam_Score"].apply(category)
        )

        created.append("Performance_Category")

    return df, created


def find_outliers(dataframe):

    results = {}

    numeric_columns = dataframe.select_dtypes(
        include=np.number
    ).columns

    for column in numeric_columns:

        values = dataframe[column].dropna()

        if len(values) < 4:
            continue

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)

        iqr = q3 - q1

        if iqr == 0:
            continue

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        mask = (
            (dataframe[column] < lower)
            |
            (dataframe[column] > upper)
        )

        count = int(mask.sum())

        if count > 0:

            results[column] = {
                "count": count,
                "lower": lower,
                "upper": upper,
                "rows": dataframe[mask]
            }

    return results


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

        # -------------------------------------------------
        # READ DATA
        # -------------------------------------------------

        original_df = pd.read_csv(uploaded_file)

        if original_df.empty:

            st.error("The uploaded dataset is empty.")

            st.stop()

        df = original_df.copy()

        # -------------------------------------------------
        # ORIGINAL METRICS
        # -------------------------------------------------

        original_rows = len(original_df)
        original_columns = len(original_df.columns)

        original_missing = int(
            original_df.isnull().sum().sum()
        )

        original_duplicates = int(
            original_df.duplicated().sum()
        )

        # =================================================
        # PROCESSING
        # =================================================

        missing_fixed = 0
        duplicates_removed = 0
        created_features = []
        outlier_results = {}

        # -------------------------------------------------
        # MISSING VALUES
        # -------------------------------------------------

        if handle_missing:

            numeric_columns = df.select_dtypes(
                include=np.number
            ).columns

            for column in numeric_columns:

                count = int(
                    df[column].isnull().sum()
                )

                if count > 0:

                    median = df[column].median()

                    df[column] = df[column].fillna(
                        median
                    )

                    missing_fixed += count

            categorical_columns = df.select_dtypes(
                include=["object", "string"]
            ).columns

            for column in categorical_columns:

                count = int(
                    df[column].isnull().sum()
                )

                if count > 0:

                    df[column] = df[column].fillna(
                        "Unknown"
                    )

                    missing_fixed += count

        # -------------------------------------------------
        # DUPLICATES
        # -------------------------------------------------

        if remove_duplicates:

            before = len(df)

            df = df.drop_duplicates()

            duplicates_removed = (
                before - len(df)
            )

        # -------------------------------------------------
        # STANDARDIZATION
        # -------------------------------------------------

        if standardize_categories:

            df = standardize_categories(df)

        # -------------------------------------------------
        # OUTLIERS
        # -------------------------------------------------

        if detect_outliers:

            outlier_results = find_outliers(df)

        # -------------------------------------------------
        # FEATURE ENGINEERING
        # -------------------------------------------------

        if create_features:

            df, created_features = create_features(df)

        # =================================================
        # ML DATASET
        # =================================================

        ml_df = df.copy()

        id_columns = detect_id_columns(
            ml_df
        )

        encoded_columns = []
        scaled_columns = []

        # -------------------------------------------------
        # ENCODING
        # -------------------------------------------------

        if apply_encoding:

            categorical_columns = (
                ml_df
                .select_dtypes(
                    include=["object", "string"]
                )
                .columns
                .tolist()
            )

            categorical_columns = [
                col
                for col in categorical_columns
                if col not in id_columns
            ]

            if categorical_columns:

                encoded_columns = (
                    categorical_columns.copy()
                )

                ml_df = pd.get_dummies(
                    ml_df,
                    columns=categorical_columns,
                    dtype=int
                )

        # -------------------------------------------------
        # SCALING
        # -------------------------------------------------

        if apply_scaling:

            numeric_columns = (
                ml_df
                .select_dtypes(
                    include=np.number
                )
                .columns
                .tolist()
            )

            numeric_columns = [
                col
                for col in numeric_columns
                if col not in id_columns
            ]

            if numeric_columns:

                scaler = StandardScaler()

                ml_df[numeric_columns] = (
                    scaler.fit_transform(
                        ml_df[numeric_columns]
                    )
                )

                scaled_columns = numeric_columns

        # =================================================
        # DASHBOARD METRICS
        # =================================================

        st.markdown(
            '<div class="section-title">'
            '📊 Data Quality Dashboard'
            '</div>',
            unsafe_allow_html=True
        )

        final_rows = len(df)
        final_columns = len(df.columns)

        final_missing = int(
            df.isnull().sum().sum()
        )

        final_duplicates = int(
            df.duplicated().sum()
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Rows",
                final_rows,
                delta=final_rows - original_rows
            )

        with col2:

            st.metric(
                "Columns",
                final_columns,
                delta=final_columns - original_columns
            )

        with col3:

            st.metric(
                "Missing Values",
                final_missing,
                delta=final_missing - original_missing
            )

        with col4:

            st.metric(
                "Duplicates",
                final_duplicates,
                delta=final_duplicates - original_duplicates
            )

        # =================================================
        # PROCESSING SUMMARY
        # =================================================

        st.markdown(
            '<div class="section-title">'
            '✨ Processing Summary'
            '</div>',
            unsafe_allow_html=True
        )

        s1, s2, s3, s4 = st.columns(4)

        with s1:

            st.metric(
                "Missing Fixed",
                missing_fixed
            )

        with s2:

            st.metric(
                "Duplicates Removed",
                duplicates_removed
            )

        with s3:

            st.metric(
                "Features Created",
                len(created_features)
            )

        with s4:

            st.metric(
                "Potential Outlier Columns",
                len(outlier_results)
            )

        # =================================================
        # DATA QUALITY CHARTS
        # =================================================

        st.markdown(
            '<div class="section-title">'
            '📈 Data Quality Analysis'
            '</div>',
            unsafe_allow_html=True
        )

        chart1, chart2 = st.columns(2)

        # -------------------------------------------------
        # MISSING VALUE CHART
        # -------------------------------------------------

        with chart1:

            missing_data = (
                original_df
                .isnull()
                .sum()
                .reset_index()
            )

            missing_data.columns = [
                "Column",
                "Missing Values"
            ]

            missing_data = missing_data[
                missing_data["Missing Values"] > 0
            ]

            if not missing_data.empty:

                fig_missing = px.bar(
                    missing_data,
                    x="Column",
                    y="Missing Values",
                    title="Missing Values by Column"
                )

                fig_missing.update_layout(
                    height=350
                )

                st.plotly_chart(
                    fig_missing,
                    use_container_width=True
                )

            else:

                st.success(
                    "No missing values found in the original dataset."
                )

        # -------------------------------------------------
        # NUMERIC COLUMN CHART
        # -------------------------------------------------

        with chart2:

            numeric_columns_original = (
                original_df
                .select_dtypes(
                    include=np.number
                )
                .columns
            )

            if len(numeric_columns_original) > 0:

                selected_column = st.selectbox(
                    "Select numerical column",
                    numeric_columns_original
                )

                fig_distribution = px.histogram(
                    original_df,
                    x=selected_column,
                    title=f"{selected_column} Distribution"
                )

                fig_distribution.update_layout(
                    height=350
                )

                st.plotly_chart(
                    fig_distribution,
                    use_container_width=True
                )

            else:

                st.info(
                    "No numerical columns available."
                )

        # =================================================
        # DATA PREVIEW TABS
        # =================================================

        st.markdown(
            '<div class="section-title">'
            '📋 Dataset Preview'
            '</div>',
            unsafe_allow_html=True
        )

        tab1, tab2, tab3 = st.tabs(
            [
                "📄 Raw Dataset",
                "✨ Cleaned Dataset",
                "🤖 ML-Ready Dataset"
            ]
        )

        with tab1:

            st.dataframe(
                original_df,
                use_container_width=True,
                height=400
            )

        with tab2:

            st.dataframe(
                df,
                use_container_width=True,
                height=400
            )

        with tab3:

            st.dataframe(
                ml_df,
                use_container_width=True,
                height=400
            )

        # =================================================
        # OUTLIERS
        # =================================================

        st.markdown(
            '<div class="section-title">'
            '⚠️ Outlier Analysis'
            '</div>',
            unsafe_allow_html=True
        )

        if outlier_results:

            st.warning(
                "Potential outliers were detected. "
                "They are reported but not automatically removed."
            )

            for column, result in outlier_results.items():

                with st.expander(
                    f"{column} — {result['count']} potential outliers"
                ):

                    st.write(
                        f"Lower Bound: `{result['lower']:.2f}`"
                    )

                    st.write(
                        f"Upper Bound: `{result['upper']:.2f}`"
                    )

                    st.dataframe(
                        result["rows"],
                        use_container_width=True
                    )

        else:

            st.success(
                "No potential outliers detected."
            )

        # =================================================
        # FEATURE ENGINEERING
        # =================================================

        if created_features:

            st.markdown(
                '<div class="section-title">'
                '🛠️ Feature Engineering'
                '</div>',
                unsafe_allow_html=True
            )

            st.success(
                "Created features: "
                + ", ".join(created_features)
            )

        # =================================================
        # CLEANING REPORT
        # =================================================

        st.markdown(
            '<div class="section-title">'
            '📋 Cleaning Report'
            '</div>',
            unsafe_allow_html=True
        )

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
                "Scaled Columns"
            ],
            "Value": [
                original_rows,
                final_rows,
                original_columns,
                final_columns,
                original_missing,
                final_missing,
                missing_fixed,
                original_duplicates,
                duplicates_removed,
                len(outlier_results),
                len(created_features),
                len(encoded_columns),
                len(scaled_columns)
            ]
        })

        st.dataframe(
            report,
            use_container_width=True
        )

        # =================================================
        # DOWNLOAD
        # =================================================

        st.markdown(
            '<div class="section-title">'
            '⬇️ Download Results'
            '</div>',
            unsafe_allow_html=True
        )

        download1, download2 = st.columns(2)

        with download1:

            cleaned_csv = df.to_csv(
                index=False
            )

            st.download_button(
                "📥 Download Cleaned CSV",
                cleaned_csv,
                "cleaned_dataset.csv",
                "text/csv",
                use_container_width=True
            )

        with download2:

            ml_csv = ml_df.to_csv(
                index=False
            )

            st.download_button(
                "🤖 Download ML-Ready CSV",
                ml_csv,
                "ml_ready_dataset.csv",
                "text/csv",
                use_container_width=True
            )

        # =================================================
        # SUCCESS
        # =================================================

        st.success(
            "🎉 Dataset processing completed successfully!"
        )

    except Exception as error:

        st.error(
            "❌ Something went wrong while processing the dataset."
        )

        st.exception(error)


# =========================================================
# NO FILE
# =========================================================

else:

    st.info(
        "👆 Upload a CSV dataset above to start."
    )

    st.markdown(
        """
        ### 🚀 What this tool does

        | Step | Process |
        |---|---|
        | 1 | 🔍 Dataset inspection |
        | 2 | 🕳️ Missing value handling |
        | 3 | ♻️ Duplicate removal |
        | 4 | 🔤 Category standardization |
        | 5 | 📊 Outlier detection |
        | 6 | 🛠️ Feature engineering |
        | 7 | 🔢 Categorical encoding |
        | 8 | 📏 Numerical scaling |
        | 9 | 📋 Cleaning report |
        | 10 | 📥 Export cleaned / ML-ready data |

        **Built with:** Python • Pandas • NumPy • Scikit-learn • Streamlit • Plotly
        """
    )