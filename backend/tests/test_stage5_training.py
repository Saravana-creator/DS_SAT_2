"""
Unit and Integration Tests for Stage 5: Model Training Pipeline
Tests:
1. Metric calculations (MAE, RMSE, R2, WAPE)
2. Chronological split partitioning
3. Model training and evaluation
4. Artifact saving and loading
"""

import os
import json
import tempfile
import unittest
import numpy as np
import pandas as pd
import joblib

from backend.app.services.training_service import (
    compute_metrics,
    train_and_evaluate_pipeline
)


class TestStage5Training(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.tmp_path = self.test_dir.name

        # Create synthetic feature matrix with chronological dates
        dates = pd.date_range("2022-01-01", periods=100, freq="D").strftime("%Y-%m-%d").tolist()
        records = []
        for i, d in enumerate(dates):
            demand_val = 50.0 + (i % 7) * 5.0 + np.sin(i / 10.0) * 10.0
            records.append({
                "Date": d,
                "Store ID": "S001",
                "Product ID": "P0001",
                "Demand": demand_val,
                "Units Sold": demand_val - 2.0,
                "Category": "Groceries",
                "demand_lag_1": demand_val - 1.0,
                "rolling_mean_7": demand_val - 0.5,
                "price_ratio": 1.0,
                "Promotion": 1 if i % 5 == 0 else 0,
                "Inventory Level": 200.0,
                "Units Ordered": 50.0,
                "Price": 25.0,
                "Discount": 5.0,
                "Competitor Pricing": 25.0,
                "Epidemic": 0
            })
        self.features_df = pd.DataFrame(records)

    def tearDown(self):
        self.test_dir.cleanup()

    def test_compute_metrics(self):
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([110.0, 190.0, 310.0])
        m = compute_metrics(y_true, y_pred)
        self.assertEqual(m["MAE"], 10.0)
        self.assertAlmostEqual(m["RMSE"], 10.0, places=2)
        self.assertAlmostEqual(m["WAPE_pct"], 30.0 / 600.0 * 100, places=2)
        self.assertGreater(m["R2"], 0.9)

    def test_train_pipeline_execution(self):
        feat_csv = os.path.join(self.tmp_path, "feat.csv")
        self.features_df.to_csv(feat_csv, index=False)
        mod_dir = os.path.join(self.tmp_path, "models")
        rep_dir = os.path.join(self.tmp_path, "reports")
        viz_dir = os.path.join(self.tmp_path, "viz")

        report = train_and_evaluate_pipeline(
            features_csv_path=feat_csv,
            models_dir=mod_dir,
            reports_dir=rep_dir,
            viz_dir=viz_dir
        )

        best_artifact = report["champion_model"]["artifact_path"]
        self.assertTrue(os.path.exists(best_artifact))
        loaded = joblib.load(best_artifact)
        self.assertIn("model", loaded)
        self.assertIn("feature_columns", loaded)


if __name__ == "__main__":
    unittest.main()
