"""test_benford.py
Unit tests for the Benford's Law module.
"""

import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import pytest
from src.features.benford import leading_digit, benford_chi2, add_benford_score


def test_leading_digit_basic():
    """Leading digit of 123.45 is 1, of 9.99 is 9."""
    s = pd.Series([123.45, 9.99, 5000, 0.078])
    digits = leading_digit(s).dropna().astype(int).tolist()
    assert digits == [1, 9, 5, 7]


def test_leading_digit_ignores_nonpositive():
    """Negative and zero values should become NaN."""
    s = pd.Series([-10, 0, 5])
    result = leading_digit(s).dropna()
    assert len(result) == 1
    assert int(result.iloc[0]) == 5


def test_benford_chi2_natural():
    """A series that follows Benford's Law should have a LOW chi-square."""
    rng = np.random.default_rng(42)
    # Generate Benford-distributed leading digits
    probs = np.log10(1 + 1 / np.arange(1, 10))
    digits = rng.choice(np.arange(1, 10), size=5000, p=probs)
    amounts = digits * 10.0 ** rng.uniform(0, 3, size=5000)
    score = benford_chi2(pd.Series(amounts))
    # Should be relatively small (< 30 with 8 dof is non-significant)
    assert score < 50, f"Expected low chi2 for Benford data, got {score}"


def test_benford_chi2_artificial():
    """A series of all-9-leading amounts should have a HIGH chi-square."""
    artificial = pd.Series([9.0] * 500 + [99.0] * 500)
    score = benford_chi2(artificial)
    # 9 is the rarest Benford digit (~4.6%), so forcing it should spike chi2
    assert score > 100, f"Expected high chi2 for artificial data, got {score}"


def test_add_benford_score_adds_column():
    """add_benford_score should return a DataFrame with 'benford_score'."""
    df = pd.DataFrame({
        "amt": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "category": ["A"] * 10,
    })
    result = add_benford_score(df)
    assert "benford_score" in result.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
