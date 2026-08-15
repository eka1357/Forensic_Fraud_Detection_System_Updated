"""test_benford.py — Unit tests for the Benford's Law module."""

import unittest
import numpy as np
import pandas as pd
from src.features.benford import (
    leading_digit,
    benford_chi2,
    benford_deviation_per_row,
    add_benford_score,
)


class TestBenfordModule(unittest.TestCase):

    def test_leading_digit_basic(self):
        """Leading digit of 123.45 is 1, of 9.99 is 9."""
        s = pd.Series([123.45, 9.99, 5000, 0.078])
        digits = leading_digit(s).dropna().astype(int).tolist()
        self.assertEqual(digits, [1, 9, 5, 7])

    def test_leading_digit_ignores_nonpositive(self):
        """Negative and zero values should become NaN."""
        s = pd.Series([-10, 0, 5])
        result = leading_digit(s).dropna()
        self.assertEqual(len(result), 1)
        self.assertEqual(int(result.iloc[0]), 5)

    def test_benford_chi2_natural(self):
        """A series that follows Benford's Law should have a LOW chi-square."""
        rng = np.random.default_rng(42)
        probs = np.log10(1 + 1 / np.arange(1, 10))
        digits = rng.choice(np.arange(1, 10), size=5000, p=probs)
        amounts = digits * (10.0 ** rng.uniform(0, 3, size=5000))
        score = benford_chi2(pd.Series(amounts))
        self.assertLess(score, 50, f"Expected low chi2 for Benford data, got {score}")

    def test_benford_chi2_artificial(self):
        """A series of all-9-leading amounts should have a HIGH chi-square."""
        artificial = pd.Series([9.0] * 500 + [99.0] * 500)
        score = benford_chi2(artificial)
        self.assertGreater(score, 100, f"Expected high chi2 for artificial data, got {score}")

    def test_add_benford_score_adds_column(self):
        """add_benford_score should return a DataFrame with 'benford_score'."""
        df = pd.DataFrame({
            "amt": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            "category": ["grocery_pos"] * 10,
        })
        result = add_benford_score(df)
        self.assertIn("benford_score", result.columns)
        self.assertEqual(len(result), 10)
        self.assertFalse(result["benford_score"].isna().any())

    def test_benford_empty_and_small_series(self):
        """Small or empty series should return 0.0 without errors."""
        empty_s = pd.Series(dtype=float)
        self.assertEqual(benford_chi2(empty_s), 0.0)
        small_s = pd.Series([12.0, 34.0])
        self.assertEqual(benford_chi2(small_s), 0.0)


if __name__ == "__main__":
    unittest.main()
