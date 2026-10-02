# 🧹 Automated Data Cleaning & Feature Engineering Tool

An interactive Streamlit application that automatically inspects, cleans,
transforms, and prepares CSV datasets for Machine Learning.
## 🌐 Live Demo

🚀 **Try the application here:**  
[Open Live Application]( https://automated-data-cleaning-tool-uit2pvmyetqptrzfyrn5rk.streamlit.app/)

## 🚀 Project Overview

Real-world datasets often contain missing values, duplicate records,
inconsistent categorical values, and potential outliers.

This project provides an automated preprocessing pipeline that helps users
clean and prepare datasets before applying Machine Learning models.

## ✨ Features

- 📂 CSV file upload
- 🔍 Automatic dataset inspection
- 🕳️ Missing value detection and handling
- ♻️ Duplicate row removal
- 🔤 Categorical data standardization
- 📊 IQR-based outlier detection
- 🛠️ Feature engineering
- 🔢 One-Hot Encoding
- 📏 Numerical feature scaling
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

## 🔄 Data Processing Pipeline

```text
CSV Upload
    ↓
Dataset Inspection
    ↓
Missing Value Handling
    ↓
Duplicate Removal
    ↓
Categorical Standardization
    ↓
Outlier Detection
    ↓
Feature Engineering
    ↓
Categorical Encoding
    ↓
Numerical Scaling
    ↓
Cleaning Report
    ↓
Export Cleaned / ML-Ready Dataset
