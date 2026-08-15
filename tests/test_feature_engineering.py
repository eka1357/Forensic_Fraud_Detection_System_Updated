"""test_feature_engineering.py — Unit tests for the feature engineering pipeline."""

import unittest
import numpy as np
import pandas as pd
from src.features.feature_engineering import (
    _add_time_features,
    _add_amount_features,
    _add_distance,
    _encode_category,
    _haversine_distance,
    engineer_features,
    get_feature_matrix,
    DEFAULT_CATEGORY_MAP,
)
from src.config import FEATURE_COLS


class TestFeatureEngineering(unittest.TestCase):

    def test_haversine_distance_known_coordinates(self):
        """Haversine distance between London (51.5074, -0.1278) and Paris (48.8566, 2.3522) is ~343 km."""
        dist = _haversine_distance(
            np.array([51.5074]),
            np.array([-0.1278]),
            np.array([48.8566]),
            np.array([2.3522]),
        )[0]
        self.assertTrue(340 < dist < 346, f"Expected ~343 km, got {dist}")

    def test_time_features_extraction(self):
        """Extract hour and dayofweek from trans_date_trans_time."""
        df = pd.DataFrame({
            "trans_date_trans_time": ["2023-01-01 14:30:00", "2023-01-02 03:15:00"]
        })
        result = _add_time_features(df)
        self.assertEqual(result["hour"].tolist(), [14, 3])
        self.assertEqual(result["dayofweek"].tolist(), [6, 0])

    def test_amount_log_transformation(self):
        """Verify log1p amount transformation."""
        df = pd.DataFrame({"amt": [0.0, 99.0, 1000.0]})
        result = _add_amount_features(df)
        self.assertTrue(np.isclose(result["amt_log"].iloc[0], 0.0))
        self.assertTrue(np.isclose(result["amt_log"].iloc[1], np.log1p(99.0)))

    def test_consistent_category_encoding(self):
        """Category encoding is deterministic across train and test sets."""
        df_train = pd.DataFrame({"category": ["grocery_pos", "shopping_net", "entertainment"]})
        df_test = pd.DataFrame({"category": ["entertainment", "grocery_pos", "unseen_category"]})

        df_train_enc, cat_map = _encode_category(df_train)
        df_test_enc, _ = _encode_category(df_test, category_map=cat_map)

        ent_code_train = df_train_enc.loc[df_train_enc["category"] == "entertainment", "category_code"].iloc[0]
        ent_code_test = df_test_enc.loc[df_test_enc["category"] == "entertainment", "category_code"].iloc[0]
        self.assertEqual(ent_code_train, ent_code_test)

        unknown_code = df_test_enc.loc[df_test_enc["category"] == "unseen_category", "category_code"].iloc[0]
        self.assertEqual(unknown_code, len(cat_map))

    def test_full_engineer_features_and_matrix(self):
        """engineer_features outputs expected feature columns in correct order."""
        df = pd.DataFrame({
            "trans_date_trans_time": ["2023-01-01 12:00:00"] * 10,
            "amt": [10.0, 50.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0],
            "category": ["grocery_pos"] * 10,
            "lat": [51.5] * 10,
            "long": [-0.1] * 10,
            "merch_lat": [51.52] * 10,
            "merch_long": [-0.08] * 10,
            "city_pop": [100000] * 10,
            "is_fraud": [0] * 9 + [1],
        })
        enriched = engineer_features(df)
        X, y = get_feature_matrix(enriched)

        self.assertEqual(list(X.columns), FEATURE_COLS)
        self.assertEqual(len(X), 10)
        self.assertIsNotNone(y)
        self.assertEqual(len(y), 10)
        self.assertFalse(X.isna().any().any())


if __name__ == "__main__":
    unittest.main()
