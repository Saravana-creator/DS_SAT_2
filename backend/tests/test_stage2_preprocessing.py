"""
Unit and Integration Tests for Stage 2: Preprocessing and Data Quality Assessment
Tests:
1. Data quality assessment on baseline clean data
2. Detection and correction of negative values (inventory, sales, demand)
3. Handling of invalid discount bounds (>100 or <0)
4. Replacement of non-positive prices
5. Deduplication behavior
6. Chronological sorting verification
7. Stockout censoring identification
8. Outlier bounds calculation via IQR
9. Pipeline execution and report schema validation
"""

import os
import json
import tempfile
import unittest
import pandas as pd
import numpy as np

from backend.app.services.preprocessing_service import (
    assess_data_quality,
    clean_and_preprocess_dataset,
    execute_stage2_pipeline
)


class TestStage2Preprocessing(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.tmp_path = self.test_dir.name

        # Create clean synthetic dataframe
        self.clean_df = pd.DataFrame({
            "Date": ["2023-01-02", "2023-01-01", "2023-01-03"],
            "Store ID": ["S001", "S001", "S001"],
            "Product ID": ["P0001", "P0001", "P0001"],
            "Category": ["Electronics", "Electronics", "Electronics"],
            "Region": ["North", "North", "North"],
            "Inventory Level": [150, 200, 50],
            "Units Sold": [30, 40, 50],
            "Units Ordered": [100, 0, 200],
            "Price": [99.99, 99.99, 99.99],
            "Discount": [5, 10, 0],
            "Weather Condition": ["Sunny", "Rainy", "Cloudy"],
            "Promotion": [1, 0, 0],
            "Competitor Pricing": [105.00, 102.00, 98.00],
            "Seasonality": ["Winter", "Winter", "Winter"],
            "Epidemic": [0, 0, 0],
            "Demand": [35, 40, 80]
        })

    def tearDown(self):
        self.test_dir.cleanup()

    def test_assess_data_quality_clean(self):
        qa = assess_data_quality(self.clean_df)
        self.assertEqual(qa["total_records"], 3)
        self.assertEqual(qa["full_duplicates"], 0)
        self.assertEqual(qa["anomalies"]["negative_inventory"], 0)
        self.assertEqual(qa["anomalies"]["invalid_price"], 0)
        self.assertEqual(qa["anomalies"]["invalid_discount"], 0)
        self.assertEqual(qa["anomalies"]["units_sold_exceeds_inventory"], 0)

    def test_negative_values_handling(self):
        dirty_df = self.clean_df.copy()
        dirty_df.loc[0, "Inventory Level"] = -50
        dirty_df.loc[1, "Units Sold"] = -10
        dirty_df.loc[2, "Demand"] = -5

        qa_dirty = assess_data_quality(dirty_df)
        self.assertEqual(qa_dirty["anomalies"]["negative_inventory"], 1)
        self.assertEqual(qa_dirty["anomalies"]["negative_units_sold"], 1)
        self.assertEqual(qa_dirty["anomalies"]["negative_demand"], 1)

        cleaned_df, summary = clean_and_preprocess_dataset(dirty_df)
        self.assertTrue((cleaned_df["Inventory Level"] >= 0).all())
        self.assertTrue((cleaned_df["Units Sold"] >= 0).all())
        self.assertTrue((cleaned_df["Demand"] >= 0).all())

    def test_invalid_discount_and_price_correction(self):
        dirty_df = self.clean_df.copy()
        dirty_df.loc[0, "Discount"] = 150  # Out of bounds (>100)
        dirty_df.loc[1, "Discount"] = -20  # Out of bounds (<0)
        dirty_df.loc[2, "Price"] = 0.0     # Non-positive price

        cleaned_df, summary = clean_and_preprocess_dataset(dirty_df)
        self.assertLessEqual(cleaned_df.loc[0, "Discount"], 100)
        self.assertGreaterEqual(cleaned_df.loc[1, "Discount"], 0)
        self.assertGreater(cleaned_df.loc[2, "Price"], 0)

    def test_deduplication(self):
        dupe_df = pd.concat([self.clean_df, self.clean_df.iloc[[0]]], ignore_index=True)
        self.assertEqual(len(dupe_df), 4)

        cleaned_df, summary = clean_and_preprocess_dataset(dupe_df)
        self.assertEqual(len(cleaned_df), 3)

    def test_chronological_sorting(self):
        # Dates are intentionally out of order: 2023-01-02, 2023-01-01, 2023-01-03
        cleaned_df, summary = clean_and_preprocess_dataset(self.clean_df)
        dates = cleaned_df["Date"].tolist()
        self.assertEqual(dates, ["2023-01-01", "2023-01-02", "2023-01-03"])

    def test_stockout_censoring_detection(self):
        # Row 2: Inventory Level = 50, Units Sold = 50, Demand = 80 -> stockout censored
        qa = assess_data_quality(self.clean_df)
        stockout = qa["stockout_analysis"]
        self.assertEqual(stockout["stockout_censored_records"], 1)
        self.assertEqual(stockout["demand_exceeds_sales_count"], 2)

    def test_outlier_summary_iqr(self):
        large_series = [10, 11, 12, 13, 14, 15, 1000] # 1000 is outlier
        df_out = self.clean_df.copy()
        df_out = pd.concat([df_out] * 5, ignore_index=True)
        df_out.loc[0, "Inventory Level"] = 5000 # extreme outlier
        qa = assess_data_quality(df_out)
        self.assertIn("Inventory Level", qa["outlier_summary"])
        self.assertGreater(qa["outlier_summary"]["Inventory Level"]["outlier_count"], 0)

    def test_full_stage2_pipeline_execution(self):
        test_raw_csv = os.path.join(self.tmp_path, "test_raw.csv")
        self.clean_df.to_csv(test_raw_csv, index=False)
        out_dir = os.path.join(self.tmp_path, "processed")
        rep_dir = os.path.join(self.tmp_path, "reports")

        report = execute_stage2_pipeline(
            raw_input_path=test_raw_csv,
            output_dir=out_dir,
            reports_dir=rep_dir
        )

        self.assertTrue(os.path.exists(report["cleaned_output"]))
        self.assertIn("quality_before", report)
        self.assertIn("quality_after", report)
        self.assertIn("cleaning_summary", report)
        self.assertEqual(report["quality_after"]["full_duplicates"], 0)


if __name__ == "__main__":
    unittest.main()
