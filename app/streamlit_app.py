"""streamlit_app.py — Forensic Fraud Detection Dashboard

An interactive dashboard demonstrating the multi-signal forensic fraud detection pipeline:
  1. Data ingestion & feature transformation
  2. Multi-signal ensemble scoring (XGBoost + Isolation Forest + Benford's Law)
  3. Interactive filtering by risk tier, amount, and merchant category
  4. Leading-digit Benford's Law distribution analysis
  5. Feature importance and model performance views
  6. UK regulatory compliance mapping (MLR 2017, FCA SYSC 6.1, POCA 2002, PSR 2017)
"""

import json
import logging
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import (
    METADATA_PATH,
    COMPLIANCE_HIGH_AMOUNT_THRESHOLD,
)
from src.utils.data_loader import load_processed_data
from src.features.feature_engineering import engineer_features, get_feature_matrix
from src.features.benford import leading_digit, BENFORD_EXPECTED
from src.models.ensemble import compute_ensemble, load_models
from src.compliance.compliance_engine import get_compliance_flags, format_flags_text

# Page config
st.set_page_config(
    page_title="Forensic Fraud Detection Dashboard",
    page_icon="🔍",
    layout="wide",
)

# Custom Styling
st.markdown("""
<style>
    .block-container { padding-top: 1.2rem; }
    .metric-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 1.1rem;
        border-radius: 0.7rem;
        color: white;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .metric-card h3 { margin: 0; font-size: 1.8rem; font-weight: 700; }
    .metric-card p  { margin: 0; opacity: 0.85; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)

st.title("🔍 Forensic Fraud Detection Dashboard")
st.caption(
    "Multi-signal forensic accounting and machine learning pipeline (XGBoost + Isolation Forest + Benford's Law) "
    "with automated UK regulatory compliance mapping."
)


@st.cache_data(show_spinner="Loading models and scoring test transactions...")
def load_and_score_data():
    """Load test dataset, apply feature engineering, and compute multi-signal ensemble scores."""
    _, test_df = load_processed_data()
    # Sample if large for snappy dashboard interactions
    df = test_df.copy()
    df_fe = engineer_features(df)
    X, y = get_feature_matrix(df_fe)

    iso_model, xgb_model = load_models()
    scored = compute_ensemble(X, iso_model=iso_model, xgb_model=xgb_model)

    # Attach human-readable columns
    for col in ["amt", "category", "trans_date_trans_time", "city", "state", "lat", "long", "distance"]:
        if col in df_fe.columns:
            scored[col] = df_fe[col].values
    if y is not None:
        scored["actual_fraud"] = y.values

    return scored, df_fe


try:
    scored_df, raw_fe = load_and_score_data()
except Exception as e:
    st.error(
        f"**Model or data files not ready.** Please run the training pipeline first:\n\n"
        f"`python src/pipeline/run_pipeline.py`\n\n*Error details: {e}*"
    )
    st.stop()


# ── Sidebar Filters ───────────────────────────────────────────────────
st.sidebar.header("🎯 Investigation Filters")

# Risk Tier Multiselect
available_tiers = ["High", "Medium", "Low"]
selected_tiers = st.sidebar.multiselect(
    "Filter by Risk Tier",
    options=available_tiers,
    default=["High", "Medium", "Low"],
)

# Amount Range Filter
min_amt = float(scored_df["amt"].min())
max_amt = float(scored_df["amt"].max())
amt_range = st.sidebar.slider(
    "Transaction Amount Range (£)",
    min_value=0.0,
    max_value=max(max_amt, 1000.0),
    value=(0.0, max_amt),
    step=10.0,
)

# Category Filter
categories = sorted(scored_df["category"].dropna().unique().tolist())
selected_categories = st.sidebar.multiselect(
    "Merchant Category",
    options=categories,
    default=[],
    help="Leave empty to include all categories",
)

# Apply Filters
filtered_df = scored_df[
    (scored_df["risk_tier"].isin(selected_tiers))
    & (scored_df["amt"] >= amt_range[0])
    & (scored_df["amt"] <= amt_range[1])
]
if selected_categories:
    filtered_df = filtered_df[filtered_df["category"].isin(selected_categories)]


# ── KPI Summary Cards ─────────────────────────────────────────────────
st.subheader("📊 Transaction Risk Overview")
col1, col2, col3, col4 = st.columns(4)

total_count = len(filtered_df)
high_count = (filtered_df["risk_tier"] == "High").sum()
med_count = (filtered_df["risk_tier"] == "Medium").sum()
low_count = (filtered_df["risk_tier"] == "Low").sum()

with col1:
    st.markdown(
        f'<div class="metric-card"><h3>{total_count:,}</h3><p>Selected Transactions</p></div>',
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        f'<div class="metric-card" style="background:linear-gradient(135deg,#e53935,#e35d5b)">'
        f'<h3>{high_count:,}</h3><p>High Risk (Top 3%)</p></div>',
        unsafe_allow_html=True,
    )
with col3:
    st.markdown(
        f'<div class="metric-card" style="background:linear-gradient(135deg,#fb8c00,#ffa726)">'
        f'<h3>{med_count:,}</h3><p>Medium Risk (90th-97th %)</p></div>',
        unsafe_allow_html=True,
    )
with col4:
    st.markdown(
        f'<div class="metric-card" style="background:linear-gradient(135deg,#1e88e5,#42a5f5)">'
        f'<h3>{low_count:,}</h3><p>Low Risk (Bottom 90%)</p></div>',
        unsafe_allow_html=True,
    )


# ── Analytics Tabs ────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🚨 High-Risk Transaction Inspector",
    "📈 Benford's Law & Distributions",
    "⚙️ Model Performance & Features",
])


# ── Tab 1: High-Risk Inspector ────────────────────────────────────────
with tab1:
    st.markdown("### Flagged Transactions & UK Regulatory Triggers")
    top_n = st.slider("Number of transactions to display", min_value=10, max_value=100, value=25)

    inspected = filtered_df.sort_values("risk_score", ascending=False).head(top_n).copy()

    if not inspected.empty:
        # Attach compliance triggers
        inspected["compliance_triggers"] = inspected.apply(
            lambda row: format_flags_text(
                get_compliance_flags(row["risk_tier"], row.to_dict())
            ),
            axis=1,
        )

        display_cols = [
            "amt", "category", "risk_score", "risk_tier",
            "compliance_triggers", "distance", "trans_date_trans_time", "city", "state"
        ]
        available_display_cols = [c for c in display_cols if c in inspected.columns]

        st.dataframe(
            inspected[available_display_cols].reset_index(drop=True),
            use_container_width=True,
            height=450,
        )
    else:
        st.info("No transactions match the selected filter criteria.")


# ── Tab 2: Benford's Law & Forensic Accounting ────────────────────────
with tab2:
    st.markdown("### Forensic Accounting: Leading-Digit Benford Distribution")
    st.write(
        "Natural financial transactions conform to Benford's Law (leading digit 1 ~30.1%, digit 9 ~4.6%). "
        "Fabricated or manipulated numbers deviate significantly from this curve."
    )

    col_b1, col_b2 = st.columns(2)

    with col_b1:
        # Compute observed leading digit distribution on filtered transactions
        digits = leading_digit(filtered_df["amt"]).dropna().astype(int)
        if len(digits) > 10:
            obs_counts = np.bincount(digits, minlength=10)[1:]  # digits 1-9
            obs_pct = (obs_counts / len(digits)) * 100
            expected_pct = BENFORD_EXPECTED * 100

            digit_df = pd.DataFrame({
                "Digit": list(range(1, 10)),
                "Observed (%)": obs_pct,
                "Benford Expected (%)": expected_pct,
            }).melt(id_vars="Digit", var_name="Distribution", value_name="Percentage")

            fig_ben, ax_ben = plt.subplots(figsize=(6, 3.5))
            sns.barplot(data=digit_df, x="Digit", y="Percentage", hue="Distribution", ax=ax_ben, palette=["#2a5298", "#ff7043"])
            ax_ben.set_title("Observed vs. Benford Expected Leading Digits")
            ax_ben.set_ylabel("Frequency (%)")
            ax_ben.set_xlabel("Leading Digit (1-9)")
            plt.tight_layout()
            st.pyplot(fig_ben)
        else:
            st.info("Insufficient data to plot leading digit distribution.")

    with col_b2:
        # Score distribution by tier
        fig_dist, ax_dist = plt.subplots(figsize=(6, 3.5))
        sns.histplot(
            data=filtered_df,
            x="risk_score",
            hue="risk_tier",
            bins=50,
            palette={"Low": "#42a5f5", "Medium": "#ffa726", "High": "#e53935"},
            ax=ax_dist,
            kde=True,
        )
        ax_dist.set_title("Multi-Signal Risk Score Distribution")
        ax_dist.set_xlabel("Ensemble Risk Score (0-1)")
        plt.tight_layout()
        st.pyplot(fig_dist)


# ── Tab 3: Model Performance & Explainability ─────────────────────────
with tab3:
    st.markdown("### Supervised & Anomaly Detection Signals")

    col_m1, col_m2 = st.columns(2)

    with col_m1:
        st.markdown("#### Top Predictive Features (XGBoost)")
        if METADATA_PATH.exists():
            try:
                with open(METADATA_PATH, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                features_df = pd.DataFrame(meta.get("top_features", []))
                if not features_df.empty:
                    fig_imp, ax_imp = plt.subplots(figsize=(6, 4))
                    sns.barplot(
                        data=features_df.head(8),
                        x="importance",
                        y="feature",
                        ax=ax_imp,
                        palette="Blues_r",
                    )
                    ax_imp.set_title("Feature Importance Ranking")
                    ax_imp.set_xlabel("Relative Importance")
                    plt.tight_layout()
                    st.pyplot(fig_imp)
            except Exception as e:
                st.warning(f"Could not load metadata: {e}")
        else:
            st.info("Run the training pipeline to export feature importance rankings.")

    with col_m2:
        st.markdown("#### Calibrated Decision Thresholds")
        if METADATA_PATH.exists():
            try:
                with open(METADATA_PATH, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                thresh = meta.get("optimal_threshold", 0.5)
                m_opt = meta.get("test_metrics_calibrated", {})
                m_def = meta.get("test_metrics_default", {})

                st.write(f"**Calibrated Operating Threshold:** `{thresh:.4f}`")

                comparison_df = pd.DataFrame({
                    "Metric": ["Precision", "Recall", "F1-Score", "PR-AUC", "ROC-AUC"],
                    "Default (0.50 Threshold)": [
                        f"{m_def.get('precision', 0):.4f}",
                        f"{m_def.get('recall', 0):.4f}",
                        f"{m_def.get('f1', 0):.4f}",
                        f"{m_def.get('pr_auc', 0):.4f}",
                        f"{m_def.get('roc_auc', 0):.4f}",
                    ],
                    f"Calibrated ({thresh:.4f} Threshold)": [
                        f"{m_opt.get('precision', 0):.4f}",
                        f"{m_opt.get('recall', 0):.4f}",
                        f"{m_opt.get('f1', 0):.4f}",
                        f"{m_opt.get('pr_auc', 0):.4f}",
                        f"{m_opt.get('roc_auc', 0):.4f}",
                    ],
                })
                st.table(comparison_df)
            except Exception as e:
                st.warning(f"Could not parse metrics metadata: {e}")


# ── Disclaimer ────────────────────────────────────────────────────────
st.divider()
st.caption(
    "⚠️ **Compliance & Data Disclaimer:** This application demonstrates multi-signal forensic accounting techniques "
    "on synthetic simulation data (Sparkov dataset). UK regulatory references (MLR 2017, FCA SYSC 6.1, POCA 2002, PSR 2017) "
    "are illustrative mappings and do NOT constitute legal or regulatory compliance advice."
)
