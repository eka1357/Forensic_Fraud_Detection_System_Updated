# Forensic Fraud Detection System

## Business Problem

Financial institutions must detect fraudulent transactions quickly **and** understand the regulatory relevance of each flag. A compliance analyst at a mid-size bank needs a system that:

1. Flags statistical outliers that humans would miss (volume is too high for manual review).
2. Learns from historical fraud patterns to predict new ones.
3. Catches number-fabrication using forensic accounting techniques.
4. Maps every flag to a relevant UK regulation so the analyst knows *why* it matters.

This project demonstrates how **unsupervised anomaly detection**, **supervised classification**, and **forensic accounting** techniques can be combined into a single, explainable risk score.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Raw Transaction Data                        │
│              (Sparkov Fraud Simulation Dataset)                 │
└──────────────────────────┬──────────────────────────────────────┘
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
                    │  Engineering│
                    │  + Benford  │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
       ┌──────▼─────┐ ┌───▼────┐ ┌─────▼──────┐
       │ Isolation   │ │XGBoost │ │ Benford    │
       │ Forest      │ │(Supv.) │ │ Score      │
       │ (Unsupv.)   │ │        │ │            │
       └──────┬──────┘ └───┬────┘ └─────┬──────┘
              │            │            │
              └────────────┼────────────┘
                           │
                    ┌──────▼──────┐
                    │  Phase 6    │
                    │  Ensemble   │
                    │  50/30/20   │
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

## Dataset

This project uses the **Sparkov Fraud Simulation** dataset — synthetic credit-card transactions generated to mimic real-world fraud patterns.

| Split | Rows      | Fraud Rate |
|-------|-----------|------------|
| Train | ~1.3M     | ~0.58%     |
| Test  | ~550K     | ~0.39%     |

**NOTE:** This is entirely synthetic data. No real transaction data was used.

### Key columns
`trans_date_trans_time`, `cc_num`, `merchant`, `category`, `amt`, `lat/long`, `merch_lat/merch_long`, `city_pop`, `is_fraud`

---

## Repository Structure

```
Forensic_Fraud_Detection_system/
├── app/
│   └── streamlit_app.py          # Phase 8: Interactive dashboard
├── data/
│   └── processed/                # Cleaned CSVs (git-ignored)
├── Datasets/                     # Raw CSVs (git-ignored / LFS)
├── models/                       # Saved .pkl model files
├── reports/
│   ├── eda_report.md             # Phase 1: EDA summary
│   └── model_metrics.md          # Phase 5: Evaluation metrics
├── src/
│   ├── compliance/
│   │   └── compliance_engine.py  # Phase 7: UK regulatory mapping
│   ├── features/
│   │   ├── benford.py            # Phase 2: Benford's Law scoring
│   │   └── feature_engineering.py# Phase 3: Feature pipeline
│   ├── models/
│   │   ├── isolation_forest_model.py  # Phase 4: Unsupervised model
│   │   ├── xgboost_model.py           # Phase 5: Supervised model
│   │   └── ensemble.py                # Phase 6: Combined risk score
│   ├── pipeline/
│   │   ├── run_pipeline.py       # Full pipeline runner
│   │   └── run_phase1.py         # Phase 1 standalone
│   └── utils/
│       ├── data_loader.py        # Data I/O utilities
│       └── eda.py                # EDA helpers
├── tests/
│   └── test_benford.py           # Unit tests for Benford module
├── requirements.txt
├── setup_env.ps1
└── README.md
```

---

## Getting Started

### 1. Set up the environment

```bash
# Create virtual environment and install dependencies
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Or use the provided PowerShell script:
```powershell
.\setup_env.ps1
```

### 2. Download the dataset

Place `fraudTrain.csv` and `fraudTest.csv` in the `Datasets/` folder.
(Available from the Sparkov dataset on Kaggle.)

### 3. Run the full pipeline

```bash
python src/pipeline/run_pipeline.py
```

This will:
- Clean the data and generate EDA reports
- Engineer features (time, log-amount, distance, category, Benford score)
- Train Isolation Forest and XGBoost
- Compute ensemble risk scores
- Display compliance flag samples

### 4. Launch the dashboard

```bash
streamlit run app/streamlit_app.py
```

### 5. Run tests

```bash
pytest tests/ -v
```

---

## How Each Component Works

### Benford's Law (Phase 2)
People fabricating transaction amounts tend to choose "round" or arbitrary numbers. Natural financial data follows Benford's Law — the leading digit is "1" about 30% of the time, "9" only ~4.6%. We compute a chi-square statistic per merchant category comparing observed vs expected leading-digit distributions. High deviation = suspicious.

### Isolation Forest (Phase 4)
An unsupervised model that isolates anomalies by randomly partitioning data. Points that are easy to isolate (few splits needed) are anomalies. We set `contamination=0.006` to match the ~0.6% fraud rate — this means the model expects about that fraction to be outliers.

### XGBoost (Phase 5)
A supervised gradient-boosted tree classifier trained on labeled fraud data. We handle the severe class imbalance with SMOTE oversampling + `scale_pos_weight`. Evaluated on precision, recall, F1, and PR-AUC (not accuracy — accuracy is meaningless at <1% fraud rate).

### Ensemble (Phase 6)
Weighted combination: XGBoost (50%) + Isolation Forest (30%) + Benford (20%). XGBoost gets the highest weight because it's seen labels and has the best PR-AUC. IF adds value for novel patterns. Benford catches numeric fabrication. Risk tiers are percentile-based: Low (bottom 90%), Medium (90-97th), High (top 3%).

### Compliance Engine (Phase 7)
A transparent rule-based lookup that maps risk tiers to UK regulations:
- **Money Laundering Regulations 2017** — SAR consideration for high-risk flags
- **Proceeds of Crime Act 2002** — potential NCA reporting obligation
- **FCA SYSC 6.1** — financial crime controls for high-amount + elevated risk
- **Payment Services Regulations 2017** — SCA review for unusual-hour transactions

**⚠️ Disclaimer:** These mappings are illustrative for portfolio purposes. They are NOT legal or compliance advice.

---

## How I'd Explain This in an Interview

> "I built a fraud detection system that combines three different signals. First, Benford's Law — a forensic accounting technique that catches people fabricating numbers, because fake amounts don't follow the natural leading-digit distribution. Second, Isolation Forest — an unsupervised model that flags statistical outliers without needing labels, which is valuable because most real fraud is unlabeled. Third, XGBoost — a supervised classifier trained on historical fraud data with SMOTE to handle the extreme class imbalance (fraud is less than 1% of transactions). I combine all three into a weighted risk score and map high-risk flags to relevant UK regulations like the Money Laundering Regulations 2017. The whole thing runs as a pipeline with a Streamlit dashboard so a compliance analyst can see results without touching code."

---

## Limitations & Next Steps

### Current Limitations
- **Synthetic data**: The Sparkov dataset is simulated, not real bank transactions. Model performance on real data would differ.
- **Compliance mappings are illustrative**: Real compliance requires legal expertise and institution-specific policies.
- **No real-time scoring**: This is a batch pipeline. Production fraud detection typically needs sub-second latency.
- **No account-level velocity features**: The pipeline scores transactions individually; real systems track per-account patterns over time.
- **Single contamination threshold**: Ideally this would be tuned per-segment.

### Future Ideas (out of scope for this prototype)
- Real-time streaming with Kafka + Flink
- Account-level velocity features (transactions per hour per card)
- Graph-based fraud detection (network analysis of merchant-cardholder relationships)
- Model monitoring and drift detection
- Integration with actual SAR filing systems
- A/B testing of threshold sensitivity

---

*Built as a portfolio project demonstrating forensic fraud detection techniques.*
