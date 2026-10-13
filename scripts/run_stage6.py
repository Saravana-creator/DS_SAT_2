"""
VantageGro Retail Stock Insight - Stage 6 Forecasting & Inventory Optimization CLI Runner
Executes:
1. Multi-step recursive demand forecasting across all 100 Store-Product SKUs using the champion ML model.
2. Inventory optimization engine (Safety Stock, Reorder Point, Target Stock Level, Days of Supply).
3. Risk classification (Critical Stockout, High Stockout, Overstock, Optimal).
4. Generation of all 8 Stage 6 visualizations (HTML Plotly + PNG + JSON).
5. Persistence of recommendations CSV and comprehensive JSON report.
"""

import sys
import os
import logging
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.services.forecasting_service import generate_full_inventory_recommendations
from backend.app.services.visualization_service import generate_stage6_recommendation_visualizations

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_stage6_cli")


def main():
    features_csv = "data/processed/vantagegro_features.csv"
    model_path = "models/vantagegro_best_model.joblib"
    output_dir = "data/processed"
    reports_dir = "data/reports"
    viz_dir = "data/reports/recommendation_visualizations"

    print("\n" + "=" * 75)
    print("VANTAGEGRO RETAIL STOCK INSIGHT - STAGE 6 FORECASTING & INVENTORY OPTIMIZATION")
    print("=" * 75)

    if not os.path.exists(features_csv):
        print(f"[ERROR] Features file not found at {features_csv}. Run Stage 4 first!")
        sys.exit(1)

    if not os.path.exists(model_path):
        print(f"[ERROR] Champion model not found at {model_path}. Run Stage 5 first!")
        sys.exit(1)

    # 1. Run full inventory recommendations
    results = generate_full_inventory_recommendations(
        features_csv_path=features_csv,
        model_path=model_path,
        output_dir=output_dir,
        reports_dir=reports_dir,
        horizon_days=14,
        lead_time_days=3,
        service_level_z=1.65
    )

    df_recs = results["df_recommendations"]
    report = results["report"]
    recs_list = results["full_recommendations_list"]

    # 2. Generate all 8 Stage 6 Visualizations
    features_df = pd.read_csv(features_csv)
    # Find sample forecast for S001 and P0001
    sample_item = next((r for r in recs_list if r["store_id"] == "S001" and r["product_id"] == "P0001"), recs_list[0])
    sample_forecast = sample_item["forecast_trajectory"]

    print("\nGenerating 8 Stage 6 Plotly & Matplotlib Diagnostic Visualizations...")
    viz_files = generate_stage6_recommendation_visualizations(
        df_recs=df_recs,
        features_df=features_df,
        sample_forecast=sample_forecast,
        output_dir=viz_dir
    )

    # 3. Print Results Summary
    print("\n" + "=" * 75)
    print("INVENTORY HEALTH & RISK BREAKDOWN (100 STORE-SKU COMBINATIONS)")
    print("=" * 75)
    for risk_cat, count in report["risk_breakdown"].items():
        pct = (count / len(df_recs)) * 100
        print(f"  * {risk_cat:<26}: {count:>3} SKUs ({pct:>5.1f}%)")

    print("\n" + "-" * 75)
    print("REVIEW PRIORITY BREAKDOWN:")
    print("-" * 75)
    for prio, count in report["priority_breakdown"].items():
        pct = (count / len(df_recs)) * 100
        print(f"  * Priority {prio:<6}: {count:>3} SKUs ({pct:>5.1f}%)")

    print("\n" + "-" * 75)
    print(f"TOTAL IMMEDIATE REORDER QUANTITY REQUIRED: {report['total_suggested_reorder_units']:,} units")
    print("-" * 75)

    print("\nTOP 5 HIGH-PRIORITY REORDER ACTIONS:")
    high_priority = df_recs[df_recs["Review Priority"] == "HIGH"].sort_values("Suggested Reorder Qty", ascending=False).head(5)
    cols_to_print = ["Store ID", "Product ID", "Category", "Current Inventory", "Reorder Point", "Suggested Reorder Qty", "Days of Supply", "Risk Category"]
    print(high_priority[cols_to_print].to_string(index=False))

    print("\n" + "=" * 75)
    print("STAGE 6 ARTIFACTS GENERATED:")
    print("=" * 75)
    print(f"  1. Recommendations CSV : data/processed/inventory_recommendations.csv (100 SKUs)")
    print(f"  2. Inventory Report    : data/reports/forecast_and_inventory_report.json")
    print(f"  3. Visualizations ({len(viz_files)} files): {viz_dir}/")
    print("     - 01_high_stockout_risk_products (HTML, PNG, JSON)")
    print("     - 02_excess_stock_overstock_products (HTML, PNG, JSON)")
    print("     - 03_inventory_coverage_distribution (HTML, PNG, JSON)")
    print("     - 04_recommended_reorder_quantities (HTML, PNG, JSON)")
    print("     - 05_forecast_vs_current_inventory (HTML, PNG, JSON)")
    print("     - 06_risk_by_category_and_store (HTML, PNG, JSON)")
    print("     - 07_forecast_with_historical_demand (HTML, PNG, JSON)")
    print("     - 08_priority_reorder_list (HTML, PNG, JSON)")
    print("=" * 75)
    print("Stage 6 Forecasting & Inventory Optimization Completed Successfully!\n")


if __name__ == "__main__":
    main()
