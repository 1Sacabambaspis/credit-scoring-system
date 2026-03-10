
# 🏦 AI-Driven Credit Scoring & Loan Approval System

**TEB 2043 Data Science Project**

## Project Overview

This project implements an end-to-end data science pipeline to predict loan default risk using the German Credit dataset. The business objective is to minimize financial loss by correctly identifying high-risk applicants, strictly adhering to the dataset's cost matrix (misclassifying a bad loan is 5 times more costly than rejecting a good one).

## System Architecture

This pipeline utilizes a polyglot architecture, splitting tasks between R and Python:

* **R (Data ETL):** Handles data extraction, statistical profiling, outlier capping (IQR), and one-hot encoding.
* **Python (Machine Learning):** Handles feature engineering, synthetic data generation (SMOTE), hyperparameter tuning, and model inference.
* **Streamlit (Frontend):** Serves an interactive web dashboard to visualize risk metrics and simulate live applicant scoring.

---

## 💻 How to Run Locally

### 1. Prerequisites

You must have both **R** and **Python (3.9 or newer)** installed on your local machine. If you are using a Mac with Anaconda/Miniconda, ensure your base environment is activated.

### 2. Install Python Dependencies

Open your terminal, navigate to the project folder, and install the required machine learning and dashboard libraries:

```bash
python -m pip install pandas scikit-learn imbalanced-learn xgboost streamlit plotly
```


### 3. Verify Data Placement

Ensure the raw dataset is located at the exact following path in your project folder:
`data/raw/german_credit.csv`

### 4. Execute the Pipeline

Run the workstreams in sequential order from your terminal.

**Phase 1: Data Preparation (R)**

**Bash**

```
Rscript src/ws1_profiling.R
Rscript src/ws2_cleaning.R
```

**Phase 2: Machine Learning & Prediction (Python)**

**Bash**

```
python src/ws3_feature_design.py
python src/ws4_model_training.py
python src/ws5_evaluation.py
python src/ws6_prediction.py
```

### 5. Launch the Dashboard

Once WS6 is complete and `predictions.csv` is generated, launch the interactive user interface:

**Bash**

```
python -m streamlit run src/ws7_dashboard.py
```

This will automatically open the dashboard in your default web browser (usually at `http://localhost:8501`).
