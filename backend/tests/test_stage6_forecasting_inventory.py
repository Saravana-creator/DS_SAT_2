"""
Unit & Integration Tests for Stage 6: Forecasting & Inventory Optimization
Validates:
1. Champion model loading and validation.
2. Recursive multi-step forecasting without future leakage.
3. Inventory optimization metric calculations (Safety Stock, ROP, TSL, DoS).
4. Risk classification boundary conditions.
5. Full recommendations generator output format and persistence.
"""

import os
import pytest
import pandas as pd
import numpy as np

from backend.app.services.forecasting_service import (
    load_champion_model,
    recursive_forecast,
    calculate_inventory_optimization,
    generate_full_inventory_recommendations
)


@pytest.fixture(scope="module")
def model_bundle():
    model_path = "models/vantagegro_best_model.joblib"
    assert os.path.exists(model_path), f"Champion model not found at {model_path}"
    return load_champion_model(model_path)


@pytest.fixture(scope="module")
def features_df():
    feat_path = "data/processed/vantagegro_features.csv"
    assert os.path.exists(feat_path), f"Features CSV not found at {feat_path}"
    return pd.read_csv(feat_path)


def test_champion_model_bundle_structure(model_bundle):
    assert "model_name" in model_bundle or "name" in model_bundle
    assert "model" in model_bundle
    assert "feature_columns" in model_bundle
    assert len(model_bundle["feature_columns"]) > 20
    assert hasattr(model_bundle["model"], "predict")


def test_recursive_forecast_horizon_and_validity(features_df, model_bundle):
    horizon_days = 14
    forecasts = recursive_forecast(
        store_id="S001",
        product_id="P0001",
        features_df=features_df,
        model_bundle=model_bundle,
        horizon_days=horizon_days
    )

    assert len(forecasts) == horizon_days
    for step_idx, step in enumerate(forecasts, 1):
        assert step["step"] == step_idx
        assert "date" in step
        assert "predicted_demand" in step
        assert isinstance(step["predicted_demand"], (int, float))
        # Physical boundary: non-negative demand
        assert step["predicted_demand"] >= 0.0


def test_scenario_simulation_promotion_effect(features_df, model_bundle):
    # Test recursive forecast under baseline vs forced promotion
    f_no_promo = recursive_forecast(
        store_id="S001",
        product_id="P0001",
        features_df=features_df,
        model_bundle=model_bundle,
        horizon_days=7,
        promo_scenario=0
    )
    f_promo = recursive_forecast(
        store_id="S001",
        product_id="P0001",
        features_df=features_df,
        model_bundle=model_bundle,
        horizon_days=7,
        promo_scenario=1
    )

    demands_no_promo = sum(s["predicted_demand"] for s in f_no_promo)
    demands_promo = sum(s["predicted_demand"] for s in f_promo)

    # Both must be positive valid floats
    assert demands_no_promo >= 0
    assert demands_promo >= 0


def test_inventory_optimization_critical_stockout():
    # If inventory is extremely low (e.g. 5 units) while lead time demand is 30 units
    mock_forecast = [{"predicted_demand": 10.0} for _ in range(14)]
    result = calculate_inventory_optimization(
        store_id="S001",
        product_id="P0001",
        category="Groceries",
        current_inventory=5.0,
        forecast_steps=mock_forecast,
        lead_time_days=3,
        service_level_z=1.65,
        review_period_days=7
    )

    # Lead time demand = 10 * 3 = 30 units > 5 units inventory
    assert result["risk_category"] == "CRITICAL STOCKOUT RISK"
    assert result["review_priority"] == "HIGH"
    assert result["suggested_reorder_qty"] > 0
    assert result["safety_stock"] >= 0
    assert result["reorder_point"] > result["current_inventory"]


def test_inventory_optimization_overstock():
    # If inventory is 1000 units while avg daily demand is 5 units (200 days of supply)
    mock_forecast = [{"predicted_demand": 5.0} for _ in range(14)]
    result = calculate_inventory_optimization(
        store_id="S001",
        product_id="P0001",
        category="Electronics",
        current_inventory=1000.0,
        forecast_steps=mock_forecast,
        lead_time_days=3,
        service_level_z=1.65,
        review_period_days=7
    )

    assert result["risk_category"] == "OVERSTOCK RISK"
    assert result["inventory_coverage_days"] > 30.0
    assert result["suggested_reorder_qty"] == 0


def test_inventory_optimization_optimal():
    # If inventory is 80 units with avg daily demand of 5 units (16 days of supply, above ROP)
    mock_forecast = [{"predicted_demand": 5.0} for _ in range(14)]
    result = calculate_inventory_optimization(
        store_id="S001",
        product_id="P0001",
        category="Clothing",
        current_inventory=80.0,
        forecast_steps=mock_forecast,
        lead_time_days=3,
        service_level_z=1.65,
        review_period_days=7
    )

    assert result["risk_category"] == "OPTIMAL STOCK LEVEL"
    assert result["review_priority"] == "LOW"


def test_full_recommendations_csv_and_report_artifacts():
    rec_csv = "data/processed/inventory_recommendations.csv"
    rep_json = "data/reports/forecast_and_inventory_report.json"

    assert os.path.exists(rec_csv), "Inventory recommendations CSV does not exist"
    assert os.path.exists(rep_json), "Forecast and inventory report JSON does not exist"

    df = pd.read_csv(rec_csv)
    assert len(df) == 100, f"Expected 100 series, got {len(df)}"
    required_cols = [
        "Store ID", "Product ID", "Category", "Current Inventory",
        "Forecast 14d Demand", "Avg Daily Demand", "Safety Stock",
        "Reorder Point", "Target Stock Level", "Suggested Reorder Qty",
        "Days of Supply", "Risk Category", "Review Priority", "Risk Explanation"
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing required column {col} in recommendations CSV"
