# 🧹 Automated Data Cleaning & Feature Engineering Tool

An interactive Streamlit application that automatically inspects, cleans,
transforms, and prepares CSV datasets for Machine Learning.

## 🚀 Project Overview

Real-world datasets often contain missing values, duplicate records,
inconsistent categorical values, and potential outliers.

This project provides an automated preprocessing pipeline that helps users
clean and prepare datasets before applying Machine Learning models.

## ✨ Features

- 📂 CSV file upload
- 🔍 Automatic dataset inspection
- 🔢 Text columns that are really numbers are converted automatically
- 🔤 Categorical data standardization (skips ID and free-text columns)
- ♻️ Duplicate row removal
- 📊 IQR-based outlier detection (reported, never removed)
- 🕳️ Missing value handling (median for numbers, "Unknown" for text)
- 🛠️ Feature engineering: bin any numeric column into categories
- 🎯 Optional target column, kept raw and protected from leakage
- 🔢 One-Hot Encoding
- 📏 Numerical feature scaling (original numeric columns only)
- 📈 Interactive data-quality visualizations
- 📋 Before/after cleaning report
- 📥 Cleaned dataset export
- 🤖 ML-ready dataset export

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Plotly

## ▶️ Running the app

```bash
python -m venv venv
venv\Scripts\activate        # Windows  (use `source venv/bin/activate` on macOS/Linux)
pip install -r requirements.txt
streamlit run app.py
```

Then upload [sample_data.csv](sample_data.csv) to try it out.

## 🧪 Running the tests

```bash
python -m unittest discover -s tests -v
```

## 🔄 Data Processing Pipeline

The order is deliberate: standardizing before de-duplicating catches rows
that only differ by case or spacing (`Male` / `male`), and de-duplicating
before imputing avoids creating false duplicates out of rows that merely
share a filled-in median.

```text
CSV Upload
    ↓
Numeric Conversion
    ↓
Categorical Standardization
    ↓
Duplicate Removal
    ↓
Outlier Detection
    ↓
Missing Value Handling
    ↓
Feature Engineering
    ↓
ML-ready dataset:
    drop ID columns and target-derived features
    → scale numeric columns (not the target)
    → one-hot encode text columns (not the target)
    ↓
Cleaning Report
    ↓
Export Cleaned / ML-Ready Dataset
```

## 📝 Notes

- **Target leakage:** if a target column is selected (defaults to
  `Exam_Score` when present), it is never encoded or scaled, and any feature
  engineered from it (such as `Exam_Score_Category`) is left out of the
  ML-ready file.
- **Scaling** is fit on the whole uploaded dataset. For a real model, fit the
  scaler on the training split only.

## 📁 Project layout

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI |
| `cleaning.py` | Pure, testable cleaning pipeline (no Streamlit imports) |
| `tests/test_cleaning.py` | Unit tests |
| `sample_data.csv` | Small student dataset with deliberate data problems |
