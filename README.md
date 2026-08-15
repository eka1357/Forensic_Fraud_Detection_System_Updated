# Forensic Fraud Detection System

> A multi-signal financial crime and fraud detection pipeline combining **Supervised XGBoost**, **Unsupervised Isolation Forest**, and **Benford's Law Forensic Accounting**, mapped directly to **UK Regulatory Compliance Rules** (MLR 2017, FCA SYSC 6.1, POCA 2002, PSR 2017) with an interactive Streamlit investigation dashboard.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost%20%7C%20IsolationForest-green.svg)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red.svg)](https://streamlit.io/)
[![Tests](https://img.shields.io/badge/Tests-21%20Passing-brightgreen.svg)](tests/)

---

## Table of Contents

1. [Overview](#1-overview)
2. [Key Features](#2-key-features)
3. [Architecture & Multi-Signal Design](#3-architecture--multi-signal-design)
4. [Dataset & Forensic Signals](#4-dataset--forensic-signals)
5. [Tech Stack](#5-tech-stack)
6. [Project Structure](#6-project-structure)
7. [Prerequisites & Installation](#7-prerequisites--installation)
8. [How to Run the Project](#8-how-to-run-the-project)
9. [Model Performance & Evaluation Metrics](#9-model-performance--evaluation-metrics)
10. [Interactive Dashboard](#10-interactive-dashboard)
11. [Compliance & Regulatory Mapping Layer](#11-compliance--regulatory-mapping-layer)
12. [Testing & Quality Assurance](#12-testing--quality-assurance)
13. [Troubleshooting](#13-troubleshooting)
14. [Design Decisions & Trade-Offs](#14-design-decisions--trade-offs)
15. [Limitations & Future Improvements](#15-limitations--future-improvements)
16. [Interview Preparation (Q&A)](#16-interview-preparation-qa)
17. [Resume & Project Pitch](#17-resume--project-pitch)
18. [License & Disclaimers](#18-license--disclaimers)

---

## 1. Overview

### The Business Problem
Financial institutions process millions of transactions daily, but fraud occurs in less than **1%** of cases. Typical rule-based systems generate excessive false alarms, overwhelming compliance teams. Conversely, pure black-box machine learning models output arbitrary anomaly probabilities that fail to explain *why* an alert matters from an audit or legal perspective.

A compliance officer or fraud analyst at a bank needs:
1. **High Precision:** Avoiding alert fatigue by keeping false positives to a minimum.
2. **Unsupervised Outlier Detection:** Flagging novel fraud vectors that historical supervised models have never seen.
3. **Forensic Accounting Checks:** Detecting number fabrication using established statistical laws (Benford's Law).
4. **Regulatory Explainability:** Translating statistical flags into statutory triggers (e.g., Suspicious Activity Report obligations under UK Money Laundering Regulations).

### The Solution
This project implements an end-to-end forensic detection pipeline that fuses three independent signals into a calibrated risk score $[0, 1]$ and categorizes transactions into **Low**, **Medium**, and **High** risk tiers, immediately linking flagged items to UK statutory reporting triggers.

---

## 2. Key Features

- **Multi-Signal Ensemble (50/30/20):**
  - **50% Supervised XGBoost:** High-precision classification trained on SMOTE-balanced historical fraud patterns.
  - **30% Unsupervised Isolation Forest:** Isolates statistical anomalies without relying on ground-truth labels.
  - **20% Benford's Law Chi-Square Scoring:** Quantifies leading-digit distribution anomalies per merchant category.
- **Data-Driven Threshold Calibration:**
  - Optimizes decision thresholds on a held-out validation split (`threshold = 0.9790`), achieving **82.58% test precision** and reducing false alarms by **98.8%** compared to the standard 0.5 cutoff.
- **Deterministic Feature Engineering:**
  - True great-circle Haversine geographic distance calculation between cardholder and merchant coordinates.
  - Deterministic category label encoding with safe fallback for unseen categories.
  - Logarithmic transformations to mitigate positive skew in transaction amounts.
- **UK Regulatory Compliance Engine:**
  - Transparent rule engine mapping transaction metadata and risk tiers to **MLR 2017**, **FCA SYSC 6.1**, **POCA 2002**, and **PSR 2017**.
- **Interactive Streamlit Investigation Dashboard:**
  - Real-time filtering by Risk Tier, Transaction Amount range, and Merchant Category.
  - Visual Benford distribution comparison (Observed vs. Logarithmic expected curve).
  - Model explainability view featuring XGBoost feature importances and calibration comparisons.
- **Production-Ready Quality:**
  - Centralized configuration (`src/config.py`), structured logging, and 21 automated unit tests.

---

## 3. Architecture & Multi-Signal Design

```text
┌─────────────────────────────────────────────────────────────────────────┐
│              Raw Transaction Ingestion (Sparkov Simulation)             │
│                      train: ~1.29M rows | test: ~555K rows              │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                              ┌──────▼──────┐
                              │  Phase 1    │
                              │  Clean &    │
                              │  EDA        │
                              └──────┬──────┘
                                     │
                              ┌──────▼──────┐
                              │  Phase 2-3  │
                              │  Feature    │
                              │  Pipeline   │
                              └──────┬──────┘
                                     │
           ┌─────────────────────────┼─────────────────────────┐
           │                         │                         │
    ┌──────▼──────┐           ┌──────▼──────┐           ┌──────▼──────┐
    │  Phase 4    │           │  Phase 5    │           │  Phase 2    │
    │  Isolation  │           │  XGBoost    │           │  Benford's  │
    │  Forest     │           │  Classifier │           │  Law Chi²   │
    │ (Weight: 30%)│          │(Weight: 50%)│           │(Weight: 20%)│
    └──────┬──────┘           └──────┬──────┘           └──────┬──────┘
           │                         │                         │
           └─────────────────────────┼─────────────────────────┘
                                     │
                              ┌──────▼──────┐
                              │  Phase 6    │
                              │  Ensemble   │
                              │  Risk Score │
                              └──────┬──────┘
                                     │
                              ┌──────▼──────┐
                              │  Phase 7    │
                              │  Compliance │
                              │  Engine     │
                              └──────┬──────┘
                                     │
                              ┌──────▼──────┐
                              │  Phase 8    │
                              │  Streamlit  │
                              │  Dashboard  │
                              └─────────────┘
```

### Risk Score Formulation
The final transaction risk score is computed as a weighted linear combination of min-max normalized signals:

$$\text{Risk Score} = 0.50 \cdot \text{XGB}_{\text{prob}} + 0.30 \cdot \text{ISO}_{\text{score}} + 0.20 \cdot \text{BEN}_{\text{score}}$$

Transactions are then classified into percentiles:
- **Low Risk:** Bottom 90% ($\text{Score} < P_{90}$)
- **Medium Risk:** 90th to 97th percentile ($P_{90} \le \text{Score} < P_{97}$)
- **High Risk:** Top 3% ($\text{Score} \ge P_{97}$)

---

## 4. Dataset & Forensic Signals

This project utilizes the **Sparkov Fraud Simulation Dataset** — an anonymized synthetic simulation of legitimate and fraudulent credit card transactions generated with realistic merchant, cardholder, and geographic dynamics.

| Dataset Split | Total Transactions | Fraud Count | Fraud Proportion |
|---|---|---|---|
| **Training Set (`fraudTrain.csv`)** | 1,296,675 | 7,506 | **0.5789%** |
| **Held-Out Test Set (`fraudTest.csv`)** | 555,719 | 2,145 | **0.3860%** |

> **Data Provenance Notice:** This dataset is entirely synthetic simulation data. No real customer PII or banking transaction records were used.

### Forensic Accounting: Benford's Law
Natural financial numbers follow Benford's Law, which states that the probability $P(d)$ of a number beginning with digit $d \in \{1, \dots, 9\}$ is:

$$P(d) = \log_{10}\left(1 + \frac{1}{d}\right)$$

| Leading Digit | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| **Expected Frequency** | **30.1%** | **17.6%** | **12.5%** | **9.7%** | **7.9%** | **6.7%** | **5.8%** | **5.1%** | **4.6%** |

When fraudsters invent amounts or test stolen cards with arbitrary values, the leading-digit distribution shifts. We compute a per-category $\chi^2$ goodness-of-fit statistic:

$$\chi^2 = \sum_{d=1}^9 \frac{(O_d - E_d)^2}{E_d}$$

High $\chi^2$ scores indicate unnatural numeric distributions and contribute directly to the risk score.

---

## 5. Tech Stack

- **Language:** Python 3.10+
- **Machine Learning:** `scikit-learn` (Isolation Forest, train-test splitting, metrics), `xgboost` (Gradient Boosted Decision Trees), `imbalanced-learn` (SMOTE oversampling)
- **Scientific Computing & Statistics:** `numpy`, `pandas`, `scipy` (Chi-square goodness-of-fit)
- **Visualization & Dashboard:** `streamlit` (UI application), `matplotlib`, `seaborn`
- **Serialization & Config:** `joblib` (Model persistence), `json`, `pathlib`
- **Testing:** `unittest` (Built-in standard library), `pytest`

---

## 6. Project Structure

```text
Forensic_Fraud_Detection_system/
├── app/
│   └── streamlit_app.py          # Interactive Streamlit investigation dashboard
├── Datasets/                     # Raw transaction CSVs (git-ignored)
│   ├── fraudTrain.csv
│   └── fraudTest.csv
├── data/
│   └── processed/                # Preprocessed clean datasets (git-ignored)
│       ├── train_processed.csv
│       └── test_processed.csv
├── models/                       # Persisted model binaries & metadata (git-ignored)
│   ├── isolation_forest.pkl
│   ├── xgboost_fraud.pkl
│   └── pipeline_metadata.json
├── reports/                      # Generated evaluation outputs & figures
│   ├── amount_distribution.png
│   ├── eda_report.md
│   ├── fraud_by_category.png
│   ├── fraud_by_hour.png
│   └── model_metrics.md
├── src/
│   ├── config.py                 # Centralized project configuration & paths
│   ├── compliance/
│   │   └── compliance_engine.py  # UK regulatory rule mapping (MLR, FCA, POCA, PSR)
│   ├── features/
│   │   ├── benford.py            # Benford's Law leading digit extraction & chi2
│   │   └── feature_engineering.py# Haversine distance, time, and category encoding
│   ├── models/
│   │   ├── ensemble.py           # Multi-signal weighted risk scoring & tiering
│   │   ├── isolation_forest_model.py # Unsupervised anomaly detection
│   │   └── xgboost_model.py      # Supervised classifier with threshold tuning
│   ├── pipeline/
│   │   ├── run_phase1.py         # Standalone EDA runner
│   │   └── run_pipeline.py       # Master end-to-end training & evaluation pipeline
│   └── utils/
│       ├── clean_data.py         # Standalone data preparation utility
│       ├── data_loader.py        # Ingestion, validation, and storage routines
│       └── eda.py                # EDA chart generators
├── tests/
│   ├── test_benford.py           # Tests for Benford's Law calculations
│   ├── test_compliance.py        # Tests for UK regulatory triggers
│   ├── test_ensemble.py          # Tests for normalization and tier logic
│   └── test_feature_engineering.py # Tests for Haversine & category consistency
├── improvements.md               # Detailed audit, validation status & decisions
├── instructions.md               # Project architecture rules & operating guardrails
├── pyproject.toml                # Project packaging & pytest configuration
├── requirements.txt              # Pinned dependencies
├── setup_env.ps1                 # Windows PowerShell virtual environment setup
└── README.md
```

---

## 7. Prerequisites & Installation

### Prerequisites
- **Python:** Version 3.10, 3.11, 3.12, or 3.13
- **Disk Space:** ~1.5 GB for datasets and virtual environment
- **RAM:** Minimum 8 GB recommended (16 GB optimal for 1.3M row processing)

### Step 1 — Clone the Repository
```bash
git clone https://github.com/your-username/Forensic_Fraud_Detection_system.git
cd Forensic_Fraud_Detection_system
```

### Step 2 — Create and Activate Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3 — Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 8. How to Run the Project

### 1. Execute the Master Pipeline
To clean the data, generate EDA reports, extract features, train models, tune thresholds, and score the test set:

```bash
python -m src.pipeline.run_pipeline
```
*Expected Execution Time: ~2 minutes on modern hardware.*

### 2. Launch the Streamlit Dashboard
```bash
streamlit run app/streamlit_app.py
```
Open your browser and navigate to:
```text
http://localhost:8501
```

### 3. Run Automated Unit Tests
```bash
python -m unittest discover -s tests -v
```
*Expected Output: `Ran 21 tests in 0.107s — OK`*

---

## 9. Model Performance & Evaluation Metrics

In severe class imbalance (<1% fraud), traditional accuracy is deceptive (a dummy model predicting 0 gets 99.6% accuracy). The primary objective is **optimizing Precision and Recall while maintaining high PR-AUC**.

### Held-Out Test Evaluation (555,719 Transactions)

| Metric | Baseline (Default 0.50 Cutoff) | Calibrated Operating Point (`threshold = 0.9790`) | Business Impact |
|---|---|---|---|
| **Precision** | **9.27%** | **82.58%** | **+73.31%** (Alerts are 8.9x more reliable) |
| **Recall** | 88.39% | **57.25%** | Catches majority of fraud with minimal noise |
| **False Positive Alerts** | 22,287 alarms | **259 alarms** | **-98.8%** alert reduction (prevents analyst fatigue) |
| **True Positives Caught** | 1,896 | **1,228** | Verified real fraud caught |
| **F1-Score** | 0.1679 | **0.6762** | **+0.5083** harmonic balance |
| **PR-AUC** | 0.7019 | **0.7019** | Robust across all decision thresholds |
| **ROC-AUC** | 0.9796 | **0.9796** | Near-perfect separation |

### Confusion Matrix (Test Split)
```text
                       Predicted Legit (0)    Predicted Fraud (1)
Actual Legit (0)             553,315                  259   (False Positives)
Actual Fraud (1)                 917                1,228   (True Positives)
```

### Feature Importance (Top Predictive Signals)
1. `amt` (41.0%) — Raw transaction amount
2. `amt_log` (20.4%) — Log1p transformed transaction amount
3. `benford_score` (12.0%) — Category-level leading-digit forensic anomaly
4. `hour` (9.8%) — Time of day (elevated night-time activity)
5. `category_code` (6.7%) — Merchant category index
6. `dayofweek` (4.7%) — Day of week patterns
7. `city_pop` (1.5%) — Target city population density
8. `distance` (1.2%) — Great-circle distance between cardholder & merchant

---

## 10. Interactive Dashboard

The Streamlit application provides a compliance investigation console:

1. **Investigation Filters (Sidebar):**
   - Filter by Risk Tier (`High`, `Medium`, `Low`).
   - Transaction Amount Range slider (`£0 – £10,000+`).
   - Merchant Category selector (e.g. `grocery_pos`, `shopping_net`, `gas_transport`).
2. **High-Risk Transaction Inspector:**
   - Interactive table detailing transaction amount, timestamp, location, distance, risk score, and real-time UK regulatory trigger descriptions.
3. **Forensic Accounting View:**
   - Bar chart visualizing the observed leading-digit distribution against Benford's Law theoretical curve.
4. **Model Performance & Calibration View:**
   - Threshold calibration comparison table and feature importance ranking chart.

---

## 11. Compliance & Regulatory Mapping Layer

The compliance engine maps flagged transactions to UK statutory frameworks using an auditable, transparent rule structure:

| Regulation | Code | Trigger Condition | Practical Regulatory Action |
|---|---|---|---|
| **Money Laundering Regulations 2017** | `MLR2017` | High Risk Tier | Mandatory consideration for filing a Suspicious Activity Report (SAR) |
| **Proceeds of Crime Act 2002** | `POCA2002` | High Risk Tier | Screening for potential proceeds-of-crime reporting to the National Crime Agency (NCA) |
| **FCA SYSC 6.1 (Financial Crime)** | `FCA_SYSC` | Elevated Risk + Amount $\ge £5,000$ | Requirement for Enhanced Due Diligence (EDD) and senior management oversight |
| **Payment Services Regulations 2017** | `PSR2017` | Elevated Risk + Off-Peak Hours (00:00–05:00) | Review of Strong Customer Authentication (SCA) bypass or account takeover indicators |

---

## 12. Testing & Quality Assurance

The test suite contains **21 automated unit tests** built with Python's standard `unittest` framework (also compatible with `pytest`):

```bash
python -m unittest discover -s tests -v
```

### Coverage Areas:
- **`test_benford.py` (6 tests):** Validates leading-digit extraction, low $\chi^2$ on natural Benford distributions, high $\chi^2$ on artificial data, and zero-handling edge cases.
- **`test_compliance.py` (6 tests):** Validates regulatory flag combinations (`MLR2017`, `POCA2002`, `FCA_SYSC`, `PSR2017`), low-risk clean states, and text formatting.
- **`test_feature_engineering.py` (5 tests):** Validates Haversine distance against known London-to-Paris coordinates (~343 km), log transforms, time decomposition, and consistent category encoding.
- **`test_ensemble.py` (4 tests):** Validates min-max scaling, constant-series safety, mock model combination, vectorized tier assignment, and single transaction scoring.

---

## 13. Troubleshooting

### 1. `ModuleNotFoundError: No module named 'src'`
**Cause:** Running scripts directly without setting the Python package path.
**Fix:** Run using module syntax from the root directory:
```bash
python -m src.pipeline.run_pipeline
```
Or ensure `pyproject.toml` is present and run `pip install -e .`.

### 2. `FileNotFoundError: Raw training file not found at Datasets/...`
**Cause:** Raw dataset files (`fraudTrain.csv`, `fraudTest.csv`) are missing.
**Fix:** Ensure `fraudTrain.csv` and `fraudTest.csv` are placed in the `Datasets/` directory.

### 3. `FileNotFoundError: Model files not found...`
**Cause:** Attempting to launch Streamlit before training models.
**Fix:** Execute `python -m src.pipeline.run_pipeline` first to fit and save the model artifacts.

---

## 14. Design Decisions & Trade-Offs

1. **Why 50/30/20 Ensemble Weights?**
   - XGBoost (50%) is the primary signal because it learns historical fraud relationships directly. Isolation Forest (30%) acts as an unsupervised safety net to catch novel anomalies without labels. Benford's Law (20%) adds an independent forensic accounting signal specific to numeric tampering.
2. **Why SMOTE on Training Only?**
   - Severe class imbalance (0.58% fraud) causes standard classifiers to bias heavily toward the majority class. Oversampling training data gives XGBoost balanced class visibility, but SMOTE is strictly excluded from validation and test splits to prevent data leakage.
3. **Why Threshold Calibration at 0.9790?**
   - With SMOTE, XGBoost's raw predicted probabilities shift upward. At 0.5, the model generates 22K false alarms (9.3% precision). Calibrating the threshold on a validation split to maximize F1 yields 82.6% precision with only 259 false alarms.
4. **Why Great-Circle Haversine over Euclidean?**
   - Latitude/longitude degrees do not represent uniform physical distances on Earth's surface. True Haversine calculates spherical distances in kilometers, accurately capturing geographical proximity between cardholder residence and merchant location.

---

## 15. Limitations & Future Improvements

### Current Limitations
- **Synthetic Simulation Data:** The Sparkov dataset models credit card behavior but lacks corporate multi-currency flows or wire transfer structuring.
- **Batch Pipeline:** The system scores transactions in batch mode; real-world card fraud detection requires sub-100ms API inference latency.
- **Rule-Based Regulatory Mappings:** Statutory mappings are illustrative and require institutional compliance tuning for production deployment.

### Future Improvements
- **Account Velocity & Graph Features:** Rolling count of transactions per card in 1-hour / 24-hour windows and graph network analysis of merchant-cardholder relationships.
- **Real-Time Streaming:** Ingestion via Apache Kafka with inference served via FastAPI / Triton.
- **Automated Drift Monitoring:** Tracking Population Stability Index (PSI) and concept drift across continuous transaction streams.

---

## 16. Interview Preparation (Q&A)

### Project Understanding
**Q: What is this project and what business problem does it solve?**
> **A:** It is a forensic fraud detection system designed for financial crime analysts. It solves alert fatigue by combining supervised machine learning (XGBoost), unsupervised anomaly detection (Isolation Forest), and forensic accounting (Benford's Law) into a calibrated risk score mapped directly to UK regulatory compliance reporting rules.

**Q: Walk me through the end-to-end architecture.**
> **A:** Data is ingested and cleaned in Phase 1; Phase 2 & 3 extract temporal, Haversine distance, log-amount, and Benford $\chi^2$ features. In Phase 4 & 5, Isolation Forest and XGBoost (with SMOTE and validation threshold tuning) are trained. Phase 6 combines normalized scores into risk tiers (Low, Medium, High). Phase 7 maps high-risk flags to UK regulations (MLR 2017, FCA SYSC 6.1, POCA 2002, PSR 2017), and Phase 8 serves an interactive Streamlit investigation dashboard.

### Technical & Code-Level Questions
**Q: Why does XGBoost need threshold calibration, and how did you implement it?**
> **A:** On <1% fraud data with SMOTE oversampling, the model's posterior probabilities are shifted. Using a standard 0.5 threshold produces 22K false alarms (9.3% precision). We split the 1.3M training set into an 80/20 train/validation split. We fit XGBoost on the train split, evaluate precision-recall curves on the unmanipulated validation split, and pick the threshold that maximizes F1 (`threshold = 0.9790`). On the held-out test set, this increased precision to **82.58%** and reduced false alarms to **259**.

**Q: How does the Benford's Law module work?**
> **A:** Benford's Law states that in natural datasets, digit 1 appears as the leading digit ~30.1% of the time, whereas 9 appears ~4.6%. We extract the first non-zero digit of transaction amounts and calculate a $\chi^2$ goodness-of-fit statistic against Benford's distribution per merchant category using a vectorized `groupby().apply()` and `map()` pattern. High deviation indicates unnatural numeric patterns.

**Q: How did you prevent data leakage?**
> **A:** Categorical encoders are fit strictly on training data with fixed mappings. Train and test datasets are kept in separate CSVs. Furthermore, the training dataset is partitioned into train/validation before applying SMOTE, ensuring validation and test sets remain unaltered by synthetic sampling.

---

## 17. Resume & Project Pitch

### 30-Second Pitch
> *"I built a multi-signal forensic fraud detection system that combines XGBoost, Isolation Forest, and Benford's Law leading-digit forensic accounting. It solves compliance alert fatigue by using validation-calibrated thresholding to achieve 82.6% precision on 555K transactions, and automatically maps flagged transactions to UK regulatory compliance obligations like the Money Laundering Regulations 2017 and FCA SYSC 6.1."*

### 1-Minute Deep Dive
> *"Traditional bank fraud systems either produce overwhelming false positives or output unexplainable black-box probabilities. I designed an end-to-end Python pipeline using the Sparkov simulation dataset (1.8M transactions). The pipeline derives Haversine distances, temporal features, and category-level Benford's Law deviation scores. I trained an unsupervised Isolation Forest alongside an XGBoost classifier with SMOTE. By implementing data-driven threshold calibration on a validation split, I boosted test precision from 9.3% to 82.6% while cutting false alarms by 98.8%. Finally, I built a rule-based compliance engine that links risk tiers to UK statutory reporting triggers (MLR 2017, POCA 2002, FCA SYSC 6.1) and exposed the entire system via an interactive Streamlit investigation dashboard backed by 21 automated unit tests."*

---

## 18. License & Disclaimers

### Disclaimer
This project is an academic and portfolio demonstration using synthetic simulation data. The regulatory mappings (MLR 2017, FCA SYSC 6.1, POCA 2002, PSR 2017) are illustrative and do **NOT** constitute legal, financial, or regulatory compliance advice.

### License
This project is open-source and available under the [MIT License](LICENSE).
