"""streamlit_app.py — Forensic Fraud Detection Dashboard

A lightweight Streamlit app that demonstrates the full pipeline:
  1. Load processed data
  2. Engineer features (including Benford score)
  3. Score with the ensemble (IF + XGBoost + Benford)
  4. Map high-risk tiers to UK regulatory triggers
  5. Display results in an interactive table + charts

Run with:
    streamlit run app/streamlit_app.py
"""

import sys, os
# Ensure project root is on the path so internal imports work
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.data_loader import load_processed_data
from src.features.feature_engineering import engineer_features, get_feature_matrix, FEATURE_COLS
from src.models.ensemble import compute_ensemble
from src.compliance.compliance_engine import get_compliance_flags, format_flags_text

# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Forensic Fraud Detection",
    page_icon="🔍",
    layout="wide",
)

# ── Custom CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1rem; }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 0.8rem;
        color: white;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .metric-card h3 { margin: 0; font-size: 1.8rem; }
    .metric-card p  { margin: 0; opacity: 0.85; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

st.title("🔍 Forensic Fraud Detection Dashboard")
st.caption("Combines Isolation Forest, XGBoost, and Benford's Law analysis "
           "with UK regulatory compliance mapping.")


# ── Load & score ─────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading and scoring data …")
def load_and_score():
    train_df, test_df = load_processed_data()
    # Use test set for the dashboard demo (smaller, unseen data)
    df = test_df.copy()
    df_fe = engineer_features(df)
    X, y = get_feature_matrix(df_fe)
    scored = compute_ensemble(X)
    # Bring back human-readable columns for display
    for col in ["amt", "category", "trans_date_trans_time", "city", "state"]:
        if col in df_fe.columns:
            scored[col] = df_fe[col].values
    if y is not None:
        scored["actual_fraud"] = y.values
    return scored


try:
    scored_df = load_and_score()
except FileNotFoundError as e:
    st.error(
        f"**Model files not found.** Please train the models first by running "
        f"the pipeline scripts.\n\n`{e}`"
    )
    st.stop()


# ── Summary metrics ──────────────────────────────────────────────────
st.subheader("📊 Summary")
col1, col2, col3, col4 = st.columns(4)

total = len(scored_df)
n_high = (scored_df["risk_tier"] == "High").sum()
n_med = (scored_df["risk_tier"] == "Medium").sum()
n_low = (scored_df["risk_tier"] == "Low").sum()

with col1:
    st.markdown(f'<div class="metric-card"><h3>{total:,}</h3><p>Total Transactions</p></div>',
                unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="metric-card" style="background:linear-gradient(135deg,#f5576c,#ff6a00)">'
                f'<h3>{n_high:,}</h3><p>High Risk</p></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="metric-card" style="background:linear-gradient(135deg,#f093fb,#f5576c)">'
                f'<h3>{n_med:,}</h3><p>Medium Risk</p></div>', unsafe_allow_html=True)
with col4:
    st.markdown(f'<div class="metric-card" style="background:linear-gradient(135deg,#4facfe,#00f2fe)">'
                f'<h3>{n_low:,}</h3><p>Low Risk</p></div>', unsafe_allow_html=True)


# ── Risk tier distribution ───────────────────────────────────────────
st.subheader("📈 Risk Tier Distribution")
fig1, ax1 = plt.subplots(figsize=(6, 3))
tier_counts = scored_df["risk_tier"].value_counts().reindex(["Low", "Medium", "High"])
colors = ["#4facfe", "#f093fb", "#f5576c"]
tier_counts.plot.bar(ax=ax1, color=colors, edgecolor="white")
ax1.set_ylabel("Count")
ax1.set_title("Transactions by Risk Tier")
plt.xticks(rotation=0)
plt.tight_layout()
st.pyplot(fig1)


# ── Risk score histogram ─────────────────────────────────────────────
st.subheader("📉 Risk Score Distribution")
fig2, ax2 = plt.subplots(figsize=(8, 3))
sns.histplot(scored_df["risk_score"], bins=80, kde=True, ax=ax2, color="#667eea")
ax2.set_xlabel("Risk Score")
ax2.set_title("Distribution of Ensemble Risk Scores")
plt.tight_layout()
st.pyplot(fig2)


# ── Top high-risk transactions ───────────────────────────────────────
st.subheader("🚨 Top 25 High-Risk Transactions")

top = scored_df.sort_values("risk_score", ascending=False).head(25).copy()

# Add compliance flags
top["compliance_flags"] = top.apply(
    lambda row: format_flags_text(
        get_compliance_flags(row["risk_tier"], row.to_dict())
    ),
    axis=1,
)

display_cols = [c for c in [
    "amt", "category", "risk_score", "risk_tier",
    "compliance_flags", "trans_date_trans_time", "city", "state",
] if c in top.columns]

st.dataframe(
    top[display_cols].reset_index(drop=True),
    use_container_width=True,
    height=500,
)


# ── Disclaimer ───────────────────────────────────────────────────────
st.divider()
st.caption(
    "⚠️ **Disclaimer:** This is a portfolio demonstration using synthetic data "
    "(Sparkov fraud simulation). The compliance flags are illustrative and "
    "do NOT constitute legal or regulatory advice."
)
