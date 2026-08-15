"""test_ensemble.py — Unit tests for multi-signal ensemble risk scoring."""

import unittest
from unittest.mock import MagicMock
import numpy as np
import pandas as pd
from src.models.ensemble import _minmax, compute_ensemble, score_single_transaction


class TestEnsemble(unittest.TestCase):

    def test_minmax_standard_series(self):
        """Min-max maps values linearly between 0 and 1."""
        s = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
        norm = _minmax(s)
        self.assertTrue(np.isclose(norm.min(), 0.0))
        self.assertTrue(np.isclose(norm.max(), 1.0))
        self.assertTrue(np.isclose(norm.iloc[2], 0.5))

    def test_minmax_constant_series(self):
        """Constant series returns all zeros without division by zero errors."""
        s = pd.Series([5.0, 5.0, 5.0])
        norm = _minmax(s)
        self.assertTrue((norm == 0.0).all())

    def test_compute_ensemble_with_mock_models(self):
        """Verify weighted combination logic and tier assignment with mock models."""
        n_samples = 100
        X = pd.DataFrame({
            "amt": np.linspace(10, 1000, n_samples),
            "benford_score": np.linspace(0, 100, n_samples),
        })

        # Mock Isolation Forest
        mock_iso = MagicMock()
        mock_iso.decision_function.return_value = np.linspace(1.0, -1.0, n_samples)

        # Mock XGBoost
        mock_xgb = MagicMock()
        probs = np.zeros((n_samples, 2))
        probs[:, 1] = np.linspace(0.01, 0.99, n_samples)
        probs[:, 0] = 1 - probs[:, 1]
        mock_xgb.predict_proba.return_value = probs

        scored = compute_ensemble(X, iso_model=mock_iso, xgb_model=mock_xgb)

        self.assertIn("risk_score", scored.columns)
        self.assertIn("risk_tier", scored.columns)
        self.assertTrue(set(scored["risk_tier"].unique()).issubset({"Low", "Medium", "High"}))

        # Check that highest score receives High tier
        highest_idx = scored["risk_score"].idxmax()
        self.assertEqual(scored.loc[highest_idx, "risk_tier"], "High")

        # Check that lowest score receives Low tier
        lowest_idx = scored["risk_score"].idxmin()
        self.assertEqual(scored.loc[lowest_idx, "risk_tier"], "Low")

    def test_score_single_transaction(self):
        """score_single_transaction returns a dict with risk_score and risk_tier."""
        row = {"amt": 500.0, "benford_score": 15.0}

        mock_iso = MagicMock()
        mock_iso.decision_function.return_value = np.array([0.2])

        mock_xgb = MagicMock()
        mock_xgb.predict_proba.return_value = np.array([[0.8, 0.2]])

        result = score_single_transaction(row, iso_model=mock_iso, xgb_model=mock_xgb)
        self.assertIn("risk_score", result)
        self.assertIn("risk_tier", result)
        self.assertIsInstance(result["risk_score"], float)
        self.assertIsInstance(result["risk_tier"], str)


if __name__ == "__main__":
    unittest.main()
