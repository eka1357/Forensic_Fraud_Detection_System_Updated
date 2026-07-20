"""benford.py
Benford's Law deviation scoring for transaction amounts.

Why it matters for fraud detection
----------------------------------
People fabricating transaction amounts tend to choose "round" or arbitrary
numbers whose leading-digit distribution deviates from the natural log
distribution predicted by Benford's Law. A high chi-square score signals
that a group of transactions has an unnatural leading-digit pattern.
"""

import numpy as np
import pandas as pd
from scipy.stats import chisquare

# Benford expected proportions for digits 1-9
BENFORD_EXPECTED = np.log10(1 + 1 / np.arange(1, 10))


def leading_digit(series: pd.Series) -> pd.Series:
    """Extract the leading digit (1-9) from each positive value.
    Non-positive values become NaN.
    """
    positive = series[series > 0].astype(str).str.replace(".", "", n=1)
    # strip leading zeros that appear after decimal removal
    digits = positive.str.lstrip("0").str[0].astype(float)
    return digits


def benford_chi2(series: pd.Series) -> float:
    """Chi-square statistic comparing observed leading-digit distribution
    to Benford's expected distribution. Higher = more suspicious.
    """
    digits = leading_digit(series).dropna()
    if len(digits) < 10:
        return 0.0
    obs_counts = np.bincount(digits.astype(int), minlength=10)[1:]  # digits 1-9
    expected_counts = BENFORD_EXPECTED * len(digits)
    chi2, _ = chisquare(f_obs=obs_counts, f_exp=expected_counts)
    return float(chi2)


def benford_deviation_per_row(
    df: pd.DataFrame,
    amount_col: str = "amt",
    group_col: str = "category",
) -> pd.Series:
    """Per-transaction Benford deviation score.

    We compute the chi-square score *per group* (e.g. merchant category)
    and assign it to every transaction in that group. This way each
    transaction inherits the anomaly level of its group.

    If ``group_col`` is missing, a single global score is used.
    """
    if group_col in df.columns:
        scores = df.groupby(group_col)[amount_col].transform(benford_chi2)
    else:
        global_score = benford_chi2(df[amount_col])
        scores = pd.Series(global_score, index=df.index)
    return scores


def add_benford_score(
    df: pd.DataFrame,
    amount_col: str = "amt",
    group_col: str = "category",
) -> pd.DataFrame:
    """Return a copy of ``df`` with a new ``benford_score`` column."""
    df = df.copy()
    df["benford_score"] = benford_deviation_per_row(df, amount_col, group_col)
    return df
