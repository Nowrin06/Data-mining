# Prediction of Kidney Disease Severity Using Data Mining Techniques

##  Overview
This repository contains code and documentation for predicting **kidney disease severity** (Mild, Moderate, Severe) in hemodialysis patients using classical machine learning. It summarizes background, dataset, methodology, and results from the accompanying project report and provides a clean structure to run and extend the work.

##  Objectives
- Analyze demographic, clinical, and dialysis-related variables associated with disease severity.
- Train and evaluate baseline classifiers (Logistic Regression, Decision Tree, Random Forest).
- Provide a reproducible workflow with clear evaluation and visualization.

##  Project Structure
```
.
├── data/
│   ├── raw/                      # Original CSVs (e.g., Kaggle download)
│   └── processed/                # Cleaned / transformed data
├── notebooks/
│   ├── 01_exploration.ipynb      # EDA: distributions, counts, boxplots
│   ├── 02_preprocess_pca.ipynb   # Cleaning, encoding, scaling, PCA
│   └── 03_models_eval.ipynb      # Training & evaluation of models
├── src/
│   ├── __init__.py
│   ├── data_prep.py              # Load, clean, encode, split
│   ├── features.py               # Feature selection & PCA helpers
│   ├── train.py                  # CLI/funcs to train models
│   ├── evaluate.py               # Metrics, confusion matrices, reports
│   └── viz.py                    # Plotting utilities
├── models/
│   ├── artifacts/                # Saved model files (e.g., .pkl)
│   └── reports/                  # Metrics JSON/CSV, figures
├── figures/                      # Exported plots (corr matrix, scree, etc.)
├── README.md
└── requirements.txt
```

##  Dataset
**Source:** Hemodialysis Realtime Hospital Dataset (Kaggle).  
- **Rows:** ~5,000  
- **Features:** 27 independent variables (8 categorical, 19 continuous) + 1 target (severity).  
- **Target:** `severity` with classes: `mild`, `moderate`, `severe`.

> Tip: Place the downloaded CSV under `data/raw/` and update the file path in the notebooks or `src/data_prep.py`.

##  Methodology (Reproducible Flow)
1. **Preprocess**
   - Handle missing values/outliers; encode categoricals; scale numeric features.
2. **Exploratory Analysis**
   - Count plots by class & demographics; boxplots; correlation matrix.
3. **Dimensionality Reduction**
   - Apply **PCA** to reduce noise and address multicollinearity.
4. **Modeling**
   - Train **Logistic Regression**, **Decision Tree**, and **Random Forest**.
5. **Evaluation**
   - Report **Accuracy**, **Precision**, **Recall**, **F1-score** (per-class + macro/weighted).
   - Confusion matrices; learning curves (optional).
6. **Validation**
   - 80/20 train-test split; optional K-fold CV.

## Key Findings (from the report)
- **Logistic Regression** achieved ~**82% accuracy** and the best overall balance.
- **Decision Tree** and **Random Forest** baselines reached ~**67% accuracy** in this setup.
- Factors associated with severity include **age**, **gender**, **diabetes**, **hypertension**, and lab/treatment measures like **urea**, **albumin**, and **Kt/V**.

> Note: If your class distribution is imbalanced (e.g., many “severe”), consider **class weighting**, **resampling** (SMOTE), or **threshold tuning** to improve minority-class performance.

##  Quickstart
```bash
# 0) (Optional) Create a virtual environment
python -m venv .venv && source .venv/bin/activate  # on Windows: .venv\Scripts\activate

# 1) Install dependencies
pip install -r requirements.txt

# 2) Put the Kaggle CSV under data/raw/
# e.g., data/raw/Hemodialysis_Data_2.csv

# 3) Open notebooks for an end-to-end run
jupyter notebook notebooks/01_exploration.ipynb
```

##  Example CLI (optional)
If you use the `src/` helpers, you might run:
```bash
python -m src.train --data data/raw/Hemodialysis_Data_2.csv --model logistic --out models/artifacts/logreg.pkl
python -m src.evaluate --model models/artifacts/logreg.pkl --data data/raw/Hemodialysis_Data_2.csv --report models/reports/logreg_metrics.json
```

##  Repro Notes
- Fix random seeds for reproducibility.
- Log model configs and metrics (JSON/CSV in `models/reports/`).
- Save trained artifacts (`models/artifacts/`) with versioned filenames (e.g., include date/hash).

##  Recommended Enhancements
- **Imbalance handling:** class weights or resampling (SMOTE).
- **Modeling:** try XGBoost/LightGBM, calibrated probabilities, and ROC-AUC/PR-AUC.
- **Explainability:** SHAP for global & local explanations.
- **Validation:** stratified K-fold CV and nested CV for model selection.
- **MLOps:** add a `Makefile`, `pre-commit`, and GitHub Actions CI for tests/flake8.

##  Acknowledgements
This work is based on a hemodialysis severity analysis project by the authors credited in the accompanying report. Data sourced from Kaggle.
