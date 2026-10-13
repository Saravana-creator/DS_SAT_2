"""
VantageGro Retail Stock Insight - Exploratory Data Analysis Service (Stage 3)
Computes rigorous descriptive analytics, answers core retail business questions,
evaluates demand variability, stockout/overstock risk indicators, and generates chart payloads.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def generate_eda_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes comprehensive EDA summary answering all business questions:
    1. Category Performance (Sales, Demand, Inventory)
    2. Store & Regional Demand Breakdown
    3. Temporal Demand Dynamics (Daily, Weekly, Monthly)
    4. Stockout Risk vs Overstock Risk Counts
    5. Promotion & Discount Relationships
    6. Weather & Seasonal Patterns
    7. Product Demand Volatility & CV Analysis
    8. Segmented Inventory Policy Recommendations
    """
    data = df.copy()
    data["Date"] = pd.to_datetime(data["Date"])
    total_records = len(data)

    # 1. Category Performance
    cat_summary = data.groupby("Category").agg(
        total_units_sold=("Units Sold", "sum"),
        total_demand=("Demand", "sum"),
        avg_daily_demand=("Demand", "mean"),
        avg_inventory_level=("Inventory Level", "mean"),
        avg_price=("Price", "mean"),
        avg_discount=("Discount", "mean"),
        product_count=("Product ID", "nunique")
    ).reset_index()
    cat_summary["demand_share_pct"] = (cat_summary["total_demand"] / cat_summary["total_demand"].sum() * 100).round(2)
    cat_summary["sales_to_demand_ratio"] = (cat_summary["total_units_sold"] / cat_summary["total_demand"]).round(3)
    cat_summary = cat_summary.sort_values("total_demand", ascending=False).to_dict(orient="records")

    # 2. Store & Regional Performance
    store_summary = data.groupby(["Store ID", "Region"]).agg(
        total_units_sold=("Units Sold", "sum"),
        total_demand=("Demand", "sum"),
        avg_inventory_level=("Inventory Level", "mean"),
        unique_products=("Product ID", "nunique"),
        stockout_days=("Inventory Level", lambda x: int((x == 0).sum()))
    ).reset_index()
    store_summary["demand_share_pct"] = (store_summary["total_demand"] / store_summary["total_demand"].sum() * 100).round(2)
    store_summary = store_summary.sort_values("total_demand", ascending=False).to_dict(orient="records")

    region_summary = data.groupby("Region").agg(
        total_units_sold=("Units Sold", "sum"),
        total_demand=("Demand", "sum"),
        avg_inventory_level=("Inventory Level", "mean")
    ).reset_index().sort_values("total_demand", ascending=False).to_dict(orient="records")

    # 3. Temporal Demand Aggregations
    daily_trend = data.groupby(data["Date"].dt.strftime("%Y-%m-%d")).agg(
        daily_demand=("Demand", "sum"),
        daily_units_sold=("Units Sold", "sum"),
        daily_inventory=("Inventory Level", "sum")
    ).reset_index().rename(columns={"Date": "date"})

    weekly_trend = data.groupby(data["Date"].dt.to_period("W").dt.start_time.dt.strftime("%Y-%m-%d")).agg(
        weekly_demand=("Demand", "sum"),
        weekly_units_sold=("Units Sold", "sum")
    ).reset_index().rename(columns={"Date": "week_start"})

    monthly_trend = data.groupby(data["Date"].dt.strftime("%Y-%m")).agg(
        monthly_demand=("Demand", "sum"),
        monthly_units_sold=("Units Sold", "sum"),
        avg_inventory=("Inventory Level", "mean")
    ).reset_index().rename(columns={"Date": "year_month"})

    # 4. Stockout & Overstock Risk Analysis
    # Stockout risk: Inventory Level == 0 or Inventory Level < Demand
    # Overstock risk: Inventory Level > 3 * Demand (holding excess unneeded stock)
    stockout_count = int((data["Inventory Level"] < data["Demand"]).sum())
    severe_stockout_count = int((data["Inventory Level"] == 0).sum())
    overstock_count = int((data["Inventory Level"] > (3 * data["Demand"])).sum())
    balanced_stock_count = total_records - stockout_count - overstock_count

    risk_metrics = {
        "stockout_risk_records": stockout_count,
        "stockout_risk_pct": round(stockout_count / total_records * 100, 2),
        "severe_zero_inventory_records": severe_stockout_count,
        "severe_zero_inventory_pct": round(severe_stockout_count / total_records * 100, 2),
        "overstock_risk_records": overstock_count,
        "overstock_risk_pct": round(overstock_count / total_records * 100, 2),
        "balanced_inventory_records": balanced_stock_count,
        "balanced_inventory_pct": round(balanced_stock_count / total_records * 100, 2)
    }

    # 5. Promotion & Discount Analysis
    promo_stats = data.groupby("Promotion").agg(
        record_count=("Demand", "count"),
        mean_demand=("Demand", "mean"),
        median_demand=("Demand", "median"),
        std_demand=("Demand", "std"),
        mean_units_sold=("Units Sold", "mean")
    ).reset_index().to_dict(orient="records")

    discount_binned = pd.cut(data["Discount"], bins=[-1, 0, 5, 10, 15, 20, 100], labels=["0%", "1-5%", "6-10%", "11-15%", "16-20%", ">20%"])
    discount_analysis = data.groupby(discount_binned, observed=False).agg(
        record_count=("Demand", "count"),
        mean_demand=("Demand", "mean"),
        mean_units_sold=("Units Sold", "mean"),
        avg_price=("Price", "mean")
    ).reset_index().rename(columns={"Discount": "discount_tier"}).to_dict(orient="records")

    # 6. Seasonality & Weather Analysis
    seasonality_summary = data.groupby("Seasonality").agg(
        mean_demand=("Demand", "mean"),
        total_demand=("Demand", "sum"),
        record_count=("Demand", "count")
    ).reset_index().sort_values("mean_demand", ascending=False).to_dict(orient="records")

    weather_summary = data.groupby("Weather Condition").agg(
        mean_demand=("Demand", "mean"),
        total_demand=("Demand", "sum"),
        record_count=("Demand", "count")
    ).reset_index().sort_values("mean_demand", ascending=False).to_dict(orient="records")

    # 7. Product Volatility (Coefficient of Variation)
    product_stats = data.groupby(["Product ID", "Category"]).agg(
        mean_demand=("Demand", "mean"),
        std_demand=("Demand", "std"),
        min_demand=("Demand", "min"),
        max_demand=("Demand", "max"),
        total_demand=("Demand", "sum"),
        avg_inventory=("Inventory Level", "mean")
    ).reset_index()
    product_stats["cv_demand"] = (product_stats["std_demand"] / product_stats["mean_demand"]).round(3)
    product_stats["volatility_class"] = np.where(
        product_stats["cv_demand"] > 0.45, "High Volatility",
        np.where(product_stats["cv_demand"] > 0.35, "Moderate Volatility", "Stable Demand")
    )
    product_rankings = product_stats.sort_values("total_demand", ascending=False).to_dict(orient="records")

    # 8. Answers to Business Questions
    business_insights = {
        "top_selling_categories": [cat_summary[0]["Category"], cat_summary[1]["Category"]],
        "lowest_selling_category": cat_summary[-1]["Category"],
        "top_performing_store": store_summary[0]["Store ID"],
        "stockout_prevalence": f"{risk_metrics['stockout_risk_pct']}% of observation days experience stockout risk where demand exceeds on-hand stock.",
        "overstock_prevalence": f"{risk_metrics['overstock_risk_pct']}% of records exhibit excess inventory exceeding 3x daily demand.",
        "promotion_impact": (
            f"Promotions increase average daily demand from {round(promo_stats[0]['mean_demand'], 1)} to "
            f"{round(promo_stats[1]['mean_demand'], 1)} units (an uplift of "
            f"{round((promo_stats[1]['mean_demand'] - promo_stats[0]['mean_demand'])/promo_stats[0]['mean_demand']*100, 1)}%)."
        ),
        "policy_recommendation": (
            "Products classified as High Volatility (e.g. fashion/seasonal clothing) require dynamic safety stock buffers, "
            "whereas Stable Demand products (groceries) can operate with fixed periodic review replenishment cycles."
        )
    }

    return {
        "category_summary": cat_summary,
        "store_summary": store_summary,
        "region_summary": region_summary,
        "temporal_trends": {
            "daily_sample": daily_trend.tail(90).to_dict(orient="records"),
            "weekly": weekly_trend.to_dict(orient="records"),
            "monthly": monthly_trend.to_dict(orient="records")
        },
        "risk_metrics": risk_metrics,
        "promotion_analysis": promo_stats,
        "discount_analysis": discount_analysis,
        "seasonality_summary": seasonality_summary,
        "weather_summary": weather_summary,
        "product_rankings": product_rankings,
        "business_insights": business_insights,
        "filter_metadata": {
            "date_range": [data["Date"].min().strftime("%Y-%m-%d"), data["Date"].max().strftime("%Y-%m-%d")],
            "stores": sorted(data["Store ID"].unique().tolist()),
            "categories": sorted(data["Category"].unique().tolist()),
            "regions": sorted(data["Region"].unique().tolist()),
            "products": sorted(data["Product ID"].unique().tolist())
        }
    }


def execute_stage3_eda(
    cleaned_csv_path: str = "data/processed/vantagegro_cleaned.csv",
    reports_dir: str = "data/reports"
) -> Dict[str, Any]:
    """
    Executes Stage 3 EDA computation and saves data/reports/eda_summary.json.
    """
    os.makedirs(reports_dir, exist_ok=True)
    if not os.path.exists(cleaned_csv_path):
        raise FileNotFoundError(f"Cleaned dataset not found: {cleaned_csv_path}")

    df = pd.read_csv(cleaned_csv_path)
    logger.info(f"Loaded {len(df)} cleaned records for Exploratory Data Analysis...")

    eda_summary = generate_eda_summary(df)

    summary_path = os.path.join(reports_dir, "eda_summary.json")
    with open(summary_path, "w") as f:
        json.dump(eda_summary, f, indent=2)

    logger.info(f"EDA Summary successfully generated and saved to {summary_path}")
    return eda_summary
