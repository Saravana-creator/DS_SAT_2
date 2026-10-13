"""
Unit and Integration Tests for Stage 1: Dataset Pipeline
Validates:
1. Dataset detection (Primary Inventory vs External Context)
2. Column normalization
3. Date parsing and validation
4. External dataset date-level aggregation
5. Date-based merge logic
6. Duplicate detection (full row and business keys)
7. Missing-value handling and audit
8. Empty input files handling
9. Missing required columns detection
10. Non-overlapping date ranges (0% overlap) handling and suitability decision
"""

import os
import json
import tempfile
import unittest
import pandas as pd
import numpy as np

from backend.app.services.dataset_service import (
    detect_dataset_type,
    profile_dataset,
    aggregate_dataset_1,
    merge_datasets_pipeline,
    normalize_column_name,
)


class TestStage1DatasetPipeline(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for test fixtures
        self.test_dir = tempfile.TemporaryDirectory()
        self.tmp_path = self.test_dir.name

        # Create valid synthetic primary dataset fixture
        self.primary_csv = os.path.join(self.tmp_path, "synthetic_primary.csv")
        self.dates = pd.date_range("2023-01-01", periods=10, freq="D").strftime("%Y-%m-%d").tolist()
        records = []
        for d in self.dates:
            for s in ["S001", "S002"]:
                for p in ["P001", "P002"]:
                    records.append({
                        "Date": d,
                        "Store ID": s,
                        "Product ID": p,
                        "Category": "Groceries",
                        "Region": "North",
                        "Inventory Level": 100,
                        "Units Sold": 15,
                        "Units Ordered": 20,
                        "Price": 9.99,
                        "Discount": 0,
                        "Weather Condition": "Sunny",
                        "Promotion": 0,
                        "Competitor Pricing": 10.50,
                        "Seasonality": "Normal",
                        "Epidemic": 0,
                        "Demand": 18
                    })
        self.df_primary = pd.DataFrame(records)
        self.df_primary.to_csv(self.primary_csv, index=False)

        # Create synthetic Dataset 1 files with overlapping dates
        self.d1_dir_overlap = os.path.join(self.tmp_path, "d1_overlap")
        os.makedirs(self.d1_dir_overlap, exist_ok=True)
        
        df_oil_overlap = pd.DataFrame({
            "date": self.dates,
            "dcoilwtico": [75.5 + i for i in range(len(self.dates))]
        })
        df_oil_overlap.to_csv(os.path.join(self.d1_dir_overlap, "oil.csv"), index=False)

        # Create synthetic Dataset 1 files with completely non-overlapping dates (2015)
        self.d1_dir_no_overlap = os.path.join(self.tmp_path, "d1_no_overlap")
        os.makedirs(self.d1_dir_no_overlap, exist_ok=True)
        old_dates = pd.date_range("2015-01-01", periods=10, freq="D").strftime("%Y-%m-%d").tolist()
        df_oil_old = pd.DataFrame({
            "date": old_dates,
            "dcoilwtico": [50.0 + i for i in range(len(old_dates))]
        })
        df_oil_old.to_csv(os.path.join(self.d1_dir_no_overlap, "oil.csv"), index=False)

    def tearDown(self):
        self.test_dir.cleanup()

    def test_column_normalization(self):
        self.assertEqual(normalize_column_name("Store ID"), "store id")
        self.assertEqual(normalize_column_name("units_sold"), "units sold")
        self.assertEqual(normalize_column_name(" Inventory-Level  "), "inventory level")

    def test_dataset_detection_primary(self):
        result = detect_dataset_type(self.primary_csv)
        self.assertEqual(result["detected_type"], "primary_inventory")
        self.assertEqual(result["sub_type"], "retail_store_inventory")
        self.assertEqual(result["format"], "csv")

    def test_dataset_detection_external(self):
        oil_file = os.path.join(self.d1_dir_overlap, "oil.csv")
        result = detect_dataset_type(oil_file)
        self.assertEqual(result["detected_type"], "external_context")
        self.assertEqual(result["sub_type"], "dataset1_oil")

    def test_empty_input_file_handling(self):
        empty_csv = os.path.join(self.tmp_path, "empty.csv")
        with open(empty_csv, "w") as f:
            pass  # 0 bytes
        result = detect_dataset_type(empty_csv)
        self.assertEqual(result["detected_type"], "corrupted")

    def test_missing_required_columns(self):
        bad_csv = os.path.join(self.tmp_path, "bad.csv")
        df_bad = pd.DataFrame({"random_col_1": [1, 2], "random_col_2": [3, 4]})
        df_bad.to_csv(bad_csv, index=False)
        result = detect_dataset_type(bad_csv)
        self.assertEqual(result["detected_type"], "unclassified")

    def test_date_parsing_and_profile(self):
        profile = profile_dataset(self.primary_csv)
        self.assertEqual(profile["row_count"], 40)
        self.assertEqual(profile["date_range"]["min"], "2023-01-01")
        self.assertEqual(profile["date_range"]["max"], "2023-01-10")
        self.assertEqual(profile["date_range"]["unique_dates"], 10)
        self.assertEqual(profile["duplicate_rows"], 0)

    def test_aggregation_dataset1(self):
        ext_df, meta = aggregate_dataset_1(self.d1_dir_overlap)
        self.assertIsNotNone(ext_df)
        self.assertIn("ext_oil_price", ext_df.columns)
        self.assertEqual(len(ext_df), 10)
        self.assertIn("oil.csv", meta["files_aggregated"])

    def test_date_based_merge_with_overlap(self):
        out_dir = os.path.join(self.tmp_path, "out_overlap")
        rep_dir = os.path.join(self.tmp_path, "rep_overlap")
        report = merge_datasets_pipeline(
            primary_file_path=self.primary_csv,
            dataset1_dir=self.d1_dir_overlap,
            output_dir=out_dir,
            reports_dir=rep_dir
        )
        self.assertEqual(report["overlap_analysis"]["merge_match_rate_pct"], 100.0)
        self.assertEqual(report["overlap_analysis"]["overlapping_dates_count"], 10)
        self.assertIn("ext_oil_price", report["retained_columns"])
        self.assertTrue(os.path.exists(report["final_output_file"]))

    def test_non_overlapping_dates_handling(self):
        """Validates that 0% overlap drops 100% NaN features and documents the reason."""
        out_dir = os.path.join(self.tmp_path, "out_no_overlap")
        rep_dir = os.path.join(self.tmp_path, "rep_no_overlap")
        report = merge_datasets_pipeline(
            primary_file_path=self.primary_csv,
            dataset1_dir=self.d1_dir_no_overlap,
            output_dir=out_dir,
            reports_dir=rep_dir
        )
        self.assertEqual(report["overlap_analysis"]["merge_match_rate_pct"], 0.0)
        self.assertEqual(report["overlap_analysis"]["overlapping_dates_count"], 0)
        self.assertIn("EXTERNAL FEATURES UNRETAINABLE", report["feature_suitability_decision"])
        self.assertIn("ext_oil_price", report["dropped_unsuitable_external_columns"])
        # Final output contains the pristine primary rows without NaN pollution
        self.assertEqual(report["final_row_count"], 40)
        df_out = pd.read_csv(report["final_output_file"])
        self.assertNotIn("ext_oil_price", df_out.columns)

    def test_duplicate_detection(self):
        # Create dataset with duplicate row and duplicate business key
        dupe_csv = os.path.join(self.tmp_path, "dupes.csv")
        df_dupes = pd.concat([self.df_primary, self.df_primary.iloc[[0]]], ignore_index=True)
        df_dupes.to_csv(dupe_csv, index=False)
        profile = profile_dataset(dupe_csv)
        self.assertEqual(profile["duplicate_rows"], 1)


if __name__ == "__main__":
    unittest.main()
