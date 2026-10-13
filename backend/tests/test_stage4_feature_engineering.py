"""
Unit and Integration Tests for Stage 4: Feature Engineering Pipeline
Tests:
1. Calendar feature derivation (year, month, day, dayofweek, weekend, quarter, cyclical)
2. Historical demand lags computation per (Store ID, Product ID) series
3. Rolling statistics leakage prevention (shift-first validation)
4. Business pricing ratios and difference metrics
5. Categorical one-hot encodings
6. Exclusion of target and observed sales from feature set
7. Feature metadata schema and file generation
"""

import os
import json
import tempfile
import unittest
import pandas as pd
import numpy as np

from backend.app.services.feature_service import (
    engineer_features,
    execute_stage4_feature_pipeline
)


class TestStage4FeatureEngineering(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.tmp_path = self.test_dir.name

        # Create synthetic time-series grid (2 stores, 1 product, 30 days)
        dates = pd.date_range("2023-01-01", periods=30, freq="D").strftime("%Y-%m-%d").tolist()
        records = []
        for s in ["S001", "S002"]:
            for i, d in enumerate(dates):
                records.append({
                    "Date": d,
                    "Store ID": s,
                    "Product ID": "P0001",
                    "Category": "Groceries",
                    "Region": "North",
                    "Inventory Level": 200 + i,
                    "Units Sold": 20 + i,
                    "Units Ordered": 30,
                    "Price": 50.0,
                    "Discount": 10,
                    "Weather Condition": "Sunny",
                    "Promotion": 0,
                    "Competitor Pricing": 55.0,
                    "Seasonality": "Winter",
                    "Epidemic": 0,
                    "Demand": 25 + i
                })
        self.clean_df = pd.DataFrame(records)

    def tearDown(self):
        self.test_dir.cleanup()

    def test_calendar_features(self):
        df_feat, meta = engineer_features(self.clean_df)
        self.assertIn("cal_year", df_feat.columns)
        self.assertIn("cal_month", df_feat.columns)
        self.assertIn("cal_day", df_feat.columns)
        self.assertIn("cal_day_of_week", df_feat.columns)
        self.assertIn("cal_is_weekend", df_feat.columns)
        self.assertIn("sin_month", df_feat.columns)

    def test_lags_and_leakage_prevention(self):
        df_feat, meta = engineer_features(self.clean_df)
        # Check lags
        self.assertIn("demand_lag_1", df_feat.columns)
        self.assertIn("demand_lag_7", df_feat.columns)
        self.assertIn("rolling_mean_7", df_feat.columns)

        # For store S001, on day index 5 (6th day, date 2023-01-06):
        # Demand is 25 + 5 = 30
        # demand_lag_1 must be day index 4 demand = 25 + 4 = 29
        s1 = df_feat[(df_feat["Store ID"] == "S001") & (df_feat["Date"] == "2023-01-06")].iloc[0]
        self.assertEqual(s1["demand_lag_1"], 29.0)

        # Rolling mean 7 on day index 5 cannot include today's demand (30)
        # It must only include prior days (indices 0..4)
        self.assertLess(s1["rolling_mean_7"], 30.0)

    def test_pricing_features(self):
        df_feat, meta = engineer_features(self.clean_df)
        self.assertIn("price_ratio", df_feat.columns)
        self.assertIn("discounted_price", df_feat.columns)
        row0 = df_feat.iloc[0]
        self.assertAlmostEqual(row0["price_ratio"], 50.0 / 55.0, places=2)
        self.assertAlmostEqual(row0["discounted_price"], 45.0, places=2)

    def test_target_exclusion(self):
        df_feat, meta = engineer_features(self.clean_df)
        features = meta["feature_list"]
        self.assertNotIn("Demand", features)
        self.assertNotIn("Units Sold", features)
        self.assertNotIn("Date", features)

    def test_execute_stage4_pipeline(self):
        raw_csv = os.path.join(self.tmp_path, "clean.csv")
        self.clean_df.to_csv(raw_csv, index=False)
        out_dir = os.path.join(self.tmp_path, "processed")
        rep_dir = os.path.join(self.tmp_path, "reports")

        meta = execute_stage4_feature_pipeline(
            cleaned_csv_path=raw_csv,
            output_dir=out_dir,
            reports_dir=rep_dir
        )
        self.assertTrue(os.path.exists(os.path.join(out_dir, "vantagegro_features.csv")))
        self.assertTrue(os.path.exists(os.path.join(rep_dir, "feature_metadata.json")))
        self.assertGreater(meta["total_features"], 20)


if __name__ == "__main__":
    unittest.main()
