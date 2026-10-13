"""
VantageGro Retail Stock Insight - Forecasting & Inventory Optimization Service (Stage 6)
Implements:
1. Recursive multi-step demand forecasting per (Store, Product) series without target leakage.
2. Scenario simulation (known future promotions or price adjustments).
3. Transparent, mathematically documented Inventory Optimization Engine:
   - Safety Stock (SS) = z * sigma_demand * sqrt(lead_time)
   - Reorder Point (ROP) = Demand_During_Lead_Time + SS
   - Target Stock Level (TSL) = Expected_Demand * (lead_time + review_period) + SS
   - Suggested Reorder Quantity = max(0, TSL - Current_Inventory)
   - Inventory Coverage (Days of Supply) = Current_Inventory / Avg_Daily_Demand
4. Categorical Risk Classification (Critical Stockout, High Stockout Risk, Overstock Risk, Optimal).
5. Comprehensive recommendations generator across all 100 store-product series.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import joblib

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = "models/vantagegro_best_model.joblib"


def load_champion_model(model_path: str = DEFAULT_MODEL_PATH) -> Dict[str, Any]:
    """Loads trained model bundle containing estimator and feature schema."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Champion model not found at {model_path}. Train model first (Stage 5)!")
    return joblib.load(model_path)


def recursive_forecast(
    store_id: str,
    product_id: str,
    features_df: pd.DataFrame,
    model_bundle: Dict[str, Any],
    horizon_days: int = 14,
    start_date: Optional[str] = None,
    promo_scenario: Optional[int] = None,
    price_scenario: Optional[float] = None
) -> List[Dict[str, Any]]:
    """
    Performs multi-step recursive demand forecasting for a single (Store, Product) series.
    Iteratively forecasts t+1, updating autoregressive lags and rolling averages
    using predicted values rather than leaking actual future demand.
    """
    model = model_bundle["model"]
    feature_cols = model_bundle["feature_columns"]

    # Filter historical series
    series = features_df[(features_df["Store ID"] == store_id) & (features_df["Product ID"] == product_id)].copy()
    if len(series) == 0:
        raise ValueError(f"No series data found for Store {store_id}, Product {product_id}")

    series["Date_dt"] = pd.to_datetime(series["Date"])
    series = series.sort_values("Date_dt").reset_index(drop=True)

    latest_record = series.iloc[-1]
    base_date = pd.to_datetime(start_date) if start_date else latest_record["Date_dt"]

    # Keep a buffer of past demand for lag/rolling calculations
    past_demands = series["Demand"].tail(35).tolist()

    forecast_results = []
    current_date = base_date

    # Static category & region context
    category_val = latest_record.get("Category", "Groceries")
    region_val = latest_record.get("Region", "North")
    base_price = price_scenario if price_scenario is not None else float(latest_record.get("Price", 50.0))
    base_comp_price = float(latest_record.get("Competitor Pricing", base_price))
    base_discount = float(latest_record.get("Discount", 5.0))

    for h in range(1, horizon_days + 1):
        step_date = current_date + timedelta(days=h)
        promo_val = promo_scenario if promo_scenario is not None else int(latest_record.get("Promotion", 0))

        # 1. Calendar Features
        cal_year = step_date.year
        cal_month = step_date.month
        cal_day = step_date.day
        cal_day_of_week = step_date.dayofweek
        cal_is_weekend = int(cal_day_of_week in [5, 6])
        cal_week_of_year = int(step_date.isocalendar().week)
        cal_is_month_start = int(step_date.is_month_start)
        cal_is_month_end = int(step_date.is_month_end)
        cal_quarter = step_date.quarter
        sin_month = round(np.sin(2 * np.pi * cal_month / 12), 4)
        cos_month = round(np.cos(2 * np.pi * cal_month / 12), 4)
        sin_dow = round(np.sin(2 * np.pi * cal_day_of_week / 7), 4)
        cos_dow = round(np.cos(2 * np.pi * cal_day_of_week / 7), 4)

        # 2. Historical Lags from past demands buffer (recursive)
        lag_1 = past_demands[-1]
        lag_7 = past_demands[-7] if len(past_demands) >= 7 else past_demands[-1]
        lag_14 = past_demands[-14] if len(past_demands) >= 14 else past_demands[-1]
        lag_28 = past_demands[-28] if len(past_demands) >= 28 else past_demands[-1]

        # 3. Rolling Statistics
        roll_7 = round(float(np.mean(past_demands[-7:])), 2)
        roll_14 = round(float(np.mean(past_demands[-14:])), 2)
        roll_28 = round(float(np.mean(past_demands[-28:])), 2)
        roll_std_7 = round(float(np.std(past_demands[-7:])), 2)
        roll_std_14 = round(float(np.std(past_demands[-14:])), 2)
        recent_trend = round(roll_7 - roll_28, 2)

        # 4. Business & Pricing Features
        price_val = base_price
        disc_val = base_discount
        comp_price = base_comp_price
        price_ratio = round(price_val / comp_price, 3) if comp_price > 0 else 1.0
        price_diff = round(price_val - comp_price, 2)
        disc_price = round(price_val * (1 - disc_val / 100), 2)

        # 5. Assemble single row feature dict
        feat_dict = {
            "cal_year": cal_year,
            "cal_month": cal_month,
            "cal_day": cal_day,
            "cal_day_of_week": cal_day_of_week,
            "cal_is_weekend": cal_is_weekend,
            "cal_week_of_year": cal_week_of_year,
            "cal_is_month_start": cal_is_month_start,
            "cal_is_month_end": cal_is_month_end,
            "cal_quarter": cal_quarter,
            "sin_month": sin_month,
            "cos_month": cos_month,
            "sin_day_of_week": sin_dow,
            "cos_day_of_week": cos_dow,
            "demand_lag_1": lag_1,
            "demand_lag_7": lag_7,
            "demand_lag_14": lag_14,
            "demand_lag_28": lag_28,
            "rolling_mean_7": roll_7,
            "rolling_mean_14": roll_14,
            "rolling_mean_28": roll_28,
            "rolling_std_7": roll_std_7,
            "rolling_std_14": roll_std_14,
            "recent_demand_trend": recent_trend,
            "Price": price_val,
            "Discount": disc_val,
            "Promotion": promo_val,
            "Competitor Pricing": comp_price,
            "price_ratio": price_ratio,
            "price_diff": price_diff,
            "discounted_price": disc_price,
            "Inventory Level": float(latest_record.get("Inventory Level", 100.0)),
            "Units Ordered": float(latest_record.get("Units Ordered", 0.0)),
            "Epidemic": int(latest_record.get("Epidemic", 0))
        }

        # Populate one-hot categories from the training schema
        for c in feature_cols:
            if c not in feat_dict:
                # check if one-hot match
                if c.startswith("Category_"):
                    feat_dict[c] = int(c == f"Category_{category_val}")
                elif c.startswith("Region_"):
                    feat_dict[c] = int(c == f"Region_{region_val}")
                elif c.startswith("Weather Condition_"):
                    feat_dict[c] = int(c == "Weather Condition_Sunny")
                elif c.startswith("Seasonality_"):
                    # Seasonality inference
                    feat_dict[c] = int(c == "Seasonality_Winter" if cal_month in [12, 1, 2] else False)
                else:
                    feat_dict[c] = 0

        # Construct single-row vector aligned with model feature order
        input_vector = [feat_dict.get(c, 0.0) for c in feature_cols]
        predicted_demand = float(model.predict([input_vector])[0])
        # Physical boundary: non-negative demand
        predicted_demand = max(0.0, round(predicted_demand, 2))

        # Append to recursive history for subsequent steps
        past_demands.append(predicted_demand)

        forecast_results.append({
            "step": h,
            "date": step_date.strftime("%Y-%m-%d"),
            "predicted_demand": predicted_demand,
            "promotion_active": promo_val,
            "projected_price": price_val
        })

    return forecast_results


def calculate_inventory_optimization(
    store_id: str,
    product_id: str,
    category: str,
    current_inventory: float,
    forecast_steps: List[Dict[str, Any]],
    lead_time_days: int = 3,
    service_level_z: float = 1.65,
    review_period_days: int = 7
) -> Dict[str, Any]:
    """
    Computes rigorous inventory optimization metrics and risk classifications.
    Formulas:
    - Expected Daily Demand (d_bar) = Mean(Forecasted Demand)
    - Demand During Lead Time (DDLT) = d_bar * lead_time_days
    - Demand Std Dev (sigma_d) = StdDev(Forecasted Demand)
    - Safety Stock (SS) = z * sigma_d * sqrt(lead_time_days)
    - Reorder Point (ROP) = DDLT + SS
    - Target Stock Level (TSL) = d_bar * (lead_time_days + review_period_days) + SS
    - Suggested Reorder Quantity = max(0, TSL - current_inventory)
    - Inventory Coverage (Days of Supply) = current_inventory / d_bar
    """
    predicted_demands = [f["predicted_demand"] for f in forecast_steps]
    avg_daily_demand = float(np.mean(predicted_demands)) if len(predicted_demands) > 0 else 1.0
    std_daily_demand = float(np.std(predicted_demands)) if len(predicted_demands) > 0 else 5.0
    # Avoid zero std
    std_daily_demand = max(1.0, std_daily_demand)

    # 1. Lead Time Demand & Safety Stock
    ddlt = avg_daily_demand * lead_time_days
    safety_stock = service_level_z * std_daily_demand * np.sqrt(lead_time_days)
    reorder_point = ddlt + safety_stock
    target_stock_level = avg_daily_demand * (lead_time_days + review_period_days) + safety_stock

    # 2. Suggested Reorder Quantity
    suggested_order_qty = max(0.0, target_stock_level - current_inventory)

    # 3. Inventory Coverage (Days of Supply)
    days_of_supply = (current_inventory / avg_daily_demand) if avg_daily_demand > 0 else 999.0

    # 4. Risk Classification & Review Priority
    if current_inventory < ddlt:
        risk_category = "CRITICAL STOCKOUT RISK"
        risk_explanation = f"Current inventory ({int(current_inventory)}) is below demand during lead time ({int(ddlt)} units). Stockout will occur before delivery arrives."
        review_priority = "HIGH"
    elif current_inventory < reorder_point:
        risk_category = "HIGH STOCKOUT RISK"
        risk_explanation = f"Current inventory ({int(current_inventory)}) is below reorder point ({int(reorder_point)} units). Safety stock threshold breached."
        review_priority = "HIGH"
    elif days_of_supply > 30.0:
        risk_category = "OVERSTOCK RISK"
        risk_explanation = f"Inventory coverage is {round(days_of_supply, 1)} days (>30 days threshold). Capital is locked in excess inventory."
        review_priority = "MEDIUM"
    else:
        risk_category = "OPTIMAL STOCK LEVEL"
        risk_explanation = f"Inventory coverage ({round(days_of_supply, 1)} days) is balanced and within target buffer."
        review_priority = "LOW"

    return {
        "store_id": store_id,
        "product_id": product_id,
        "category": category,
        "current_inventory": int(current_inventory),
        "forecast_horizon_days": len(forecast_steps),
        "avg_daily_predicted_demand": round(avg_daily_demand, 2),
        "total_forecasted_demand": round(float(np.sum(predicted_demands)), 1),
        "demand_std_dev": round(std_daily_demand, 2),
        "lead_time_days": lead_time_days,
        "service_level_z": service_level_z,
        "safety_stock": int(np.ceil(safety_stock)),
        "reorder_point": int(np.ceil(reorder_point)),
        "target_stock_level": int(np.ceil(target_stock_level)),
        "suggested_reorder_qty": int(np.ceil(suggested_order_qty)),
        "inventory_coverage_days": round(days_of_supply, 1),
        "risk_category": risk_category,
        "risk_explanation": risk_explanation,
        "review_priority": review_priority,
        "forecast_trajectory": forecast_steps
    }


def generate_full_inventory_recommendations(
    features_csv_path: str = "data/processed/vantagegro_features.csv",
    model_path: str = DEFAULT_MODEL_PATH,
    output_dir: str = "data/processed",
    reports_dir: str = "data/reports",
    horizon_days: int = 14,
    lead_time_days: int = 3,
    service_level_z: float = 1.65
) -> Dict[str, Any]:
    """
    Generates demand forecasts and inventory optimization recommendations
    across all 100 Store-Product combinations in the dataset.
    Saves:
    - data/processed/inventory_recommendations.csv
    - data/reports/forecast_and_inventory_report.json
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    model_bundle = load_champion_model(model_path)
    df_feat = pd.read_csv(features_csv_path)

    # Get latest date snapshot for all 100 series
    latest_date = df_feat["Date"].max()
    snapshot = df_feat[df_feat["Date"] == latest_date].copy()

    logger.info(f"Generating forecasts and inventory recommendations for {len(snapshot)} SKUs as of {latest_date}...")

    recommendations = []
    for _, row in snapshot.iterrows():
        s_id = str(row["Store ID"])
        p_id = str(row["Product ID"])
        cat = str(row.get("Category", "Groceries"))
        curr_inv = float(row.get("Inventory Level", 100.0))

        # 1. Recursive Forecast
        f_steps = recursive_forecast(
            store_id=s_id,
            product_id=p_id,
            features_df=df_feat,
            model_bundle=model_bundle,
            horizon_days=horizon_days,
            start_date=latest_date
        )

        # 2. Inventory Optimization
        rec = calculate_inventory_optimization(
            store_id=s_id,
            product_id=p_id,
            category=cat,
            current_inventory=curr_inv,
            forecast_steps=f_steps,
            lead_time_days=lead_time_days,
            service_level_z=service_level_z
        )
        recommendations.append(rec)

    # Convert to tabular DataFrame
    rec_rows = []
    for r in recommendations:
        rec_rows.append({
            "Store ID": r["store_id"],
            "Product ID": r["product_id"],
            "Category": r["category"],
            "Current Inventory": r["current_inventory"],
            "Forecast 14d Demand": r["total_forecasted_demand"],
            "Avg Daily Demand": r["avg_daily_predicted_demand"],
            "Safety Stock": r["safety_stock"],
            "Reorder Point": r["reorder_point"],
            "Target Stock Level": r["target_stock_level"],
            "Suggested Reorder Qty": r["suggested_reorder_qty"],
            "Days of Supply": r["inventory_coverage_days"],
            "Risk Category": r["risk_category"],
            "Review Priority": r["review_priority"],
            "Risk Explanation": r["risk_explanation"]
        })
    df_recs = pd.DataFrame(rec_rows)

    # Save CSV
    out_csv = os.path.join(output_dir, "inventory_recommendations.csv")
    df_recs.to_csv(out_csv, index=False)
    logger.info(f"Inventory recommendations saved to {out_csv}")

    # Summary Statistics
    risk_counts = df_recs["Risk Category"].value_counts().to_dict()
    priority_counts = df_recs["Review Priority"].value_counts().to_dict()
    total_reorder_units = int(df_recs["Suggested Reorder Qty"].sum())

    report_payload = {
        "timestamp": datetime.now().isoformat(),
        "as_of_date": latest_date,
        "forecast_horizon_days": horizon_days,
        "total_sku_series": len(df_recs),
        "risk_breakdown": risk_counts,
        "priority_breakdown": priority_counts,
        "total_suggested_reorder_units": total_reorder_units,
        "parameters": {
            "lead_time_days": lead_time_days,
            "service_level_z": service_level_z,
            "service_level_pct": "95.0%",
            "review_period_days": 7
        },
        "output_file": out_csv,
        "recommendations_sample": df_recs.head(10).to_dict(orient="records")
    }

    report_json = os.path.join(reports_dir, "forecast_and_inventory_report.json")
    with open(report_json, "w") as f:
        json.dump(report_payload, f, indent=2)
    logger.info(f"Forecast and inventory report saved to {report_json}")

    return {
        "report": report_payload,
        "df_recommendations": df_recs,
        "full_recommendations_list": recommendations
    }
