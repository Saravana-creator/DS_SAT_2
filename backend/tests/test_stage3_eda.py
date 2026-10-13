"""
Unit and Integration Tests for Stage 3: Exploratory Data Analysis (EDA)
Tests:
1. Category summary calculations
2. Store and regional breakdown
3. Risk metrics (stockout and overstock counts)
4. Promotional uplift metrics
5. Product volatility (Coefficient of Variation)
6. Filters metadata structure
7. EDA summary persistence and schema
"""

import os
import json
import tempfile
import unittest
import pandas as pd
import numpy as np

from backend.app.services.eda_service import (
    generate_eda_summary,
    execute_stage3_eda
)


class TestStage3EDA(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.tmp_path = self.test_dir.name

        self.df = pd.DataFrame({
            "Date": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"],
            "Store ID": ["S001", "S001", "S002", "S002"],
            "Product ID": ["P0001", "P0002", "P0001", "P0002"],
            "Category": ["Electronics", "Clothing", "Electronics", "Clothing"],
            "Region": ["North", "North", "South", "South"],
            "Inventory Level": [50, 200, 0, 500],
            "Units Sold": [50, 40, 0, 80],
            "Units Ordered": [100, 50, 200, 0],
            "Price": [100.0, 50.0, 100.0, 50.0],
            "Discount": [0, 5, 10, 15],
            "Weather Condition": ["Sunny", "Rainy", "Sunny", "Cloudy"],
            "Promotion": [0, 1, 0, 1],
            "Competitor Pricing": [105.0, 52.0, 98.0, 48.0],
            "Seasonality": ["Winter", "Winter", "Winter", "Winter"],
            "Epidemic": [0, 0, 0, 0],
            "Demand": [80, 45, 60, 90]
        })

    def tearDown(self):
        self.test_dir.cleanup()

    def test_category_summary(self):
        eda = generate_eda_summary(self.df)
        cats = eda["category_summary"]
        self.assertEqual(len(cats), 2)
        cat_names = [c["Category"] for c in cats]
        self.assertIn("Electronics", cat_names)
        self.assertIn("Clothing", cat_names)

    def test_store_and_region_summary(self):
        eda = generate_eda_summary(self.df)
        stores = eda["store_summary"]
        self.assertEqual(len(stores), 2)
        regions = eda["region_summary"]
        self.assertEqual(len(regions), 2)

    def test_risk_metrics(self):
        eda = generate_eda_summary(self.df)
        risks = eda["risk_metrics"]
        # In our fixture:
        # Row 1: Inv=200, Demand=45 (200 > 3*45=135 -> Overstock risk)
        # Row 3: Inv=500, Demand=90 (500 > 3*90=270 -> Overstock risk)
        self.assertEqual(risks["severe_zero_inventory_records"], 1)
        self.assertEqual(risks["stockout_risk_records"], 2)
        self.assertEqual(risks["overstock_risk_records"], 2)

    def test_promotional_impact(self):
        eda = generate_eda_summary(self.df)
        promos = eda["promotion_analysis"]
        self.assertEqual(len(promos), 2)
        # Promo 0: Demands 80, 60 -> mean 70
        # Promo 1: Demands 45, 90 -> mean 67.5
        mean_p0 = next(p["mean_demand"] for p in promos if p["Promotion"] == 0)
        self.assertEqual(mean_p0, 70.0)

    def test_product_volatility_cv(self):
        eda = generate_eda_summary(self.df)
        ranks = eda["product_rankings"]
        self.assertEqual(len(ranks), 2)
        for r in ranks:
            self.assertIn("cv_demand", r)
            self.assertIn("volatility_class", r)

    def test_filter_metadata(self):
        eda = generate_eda_summary(self.df)
        meta = eda["filter_metadata"]
        self.assertEqual(meta["stores"], ["S001", "S002"])
        self.assertEqual(meta["categories"], ["Clothing", "Electronics"])
        self.assertEqual(meta["regions"], ["North", "South"])

    def test_execute_stage3_eda_pipeline(self):
        csv_path = os.path.join(self.tmp_path, "clean.csv")
        self.df.to_csv(csv_path, index=False)
        rep_dir = os.path.join(self.tmp_path, "reports")

        summary = execute_stage3_eda(cleaned_csv_path=csv_path, reports_dir=rep_dir)
        summary_file = os.path.join(rep_dir, "eda_summary.json")
        self.assertTrue(os.path.exists(summary_file))
        self.assertIn("business_insights", summary)


if __name__ == "__main__":
    unittest.main()
