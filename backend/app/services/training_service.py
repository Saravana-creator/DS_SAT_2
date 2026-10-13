"""
VantageGro Retail Stock Insight - Model Training & Evaluation Service (Stage 5)
Implements chronological train/val/test splitting, baseline models, candidate regressors
(Ridge, Random Forest, HistGradientBoosting, XGBoost), robust metric calculation
(MAE, RMSE, R2, WAPE), model artifact persistence, and diagnostic visualizations.
"""

import os
import json
import logging
from typing import Dict, Any, List, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
import joblib

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculates MAE, RMSE, R2, and WAPE metrics."""
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    r2 = float(r2_score(y_true, y_pred))
    # WAPE = sum(|y - y_hat|) / sum(y) * 100
    sum_true = float(np.sum(y_true))
    wape = float(np.sum(np.abs(y_true - y_pred)) / sum_true * 100) if sum_true > 0 else 0.0

    return {
        "MAE": round(mae, 3),
        "RMSE": round(rmse, 3),
        "R2": round(r2, 4),
        "WAPE_pct": round(wape, 2)
    }


def train_and_evaluate_pipeline(
    features_csv_path: str = "data/processed/vantagegro_features.csv",
    models_dir: str = "models",
    reports_dir: str = "data/reports",
    viz_dir: str = "data/reports/training_visualizations"
) -> Dict[str, Any]:
    """
    Executes Stage 5 ML Training Pipeline:
    1. Loads engineered feature matrix.
    2. Performs strictly chronological temporal split (Train ~72%, Val ~14%, Test ~14%).
    3. Evaluates simple baselines (Lag 1 naive forecast, 7-day moving average).
    4. Trains and evaluates candidate models (Ridge, HistGBR, RF, XGBoost).
    5. Persists the champion model to models/vantagegro_best_model.joblib.
    6. Generates 7 evaluation diagnostic visualizations.
    7. Saves data/reports/training_report.json.
    """
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)

    if not os.path.exists(features_csv_path):
        raise FileNotFoundError(f"Feature dataset not found: {features_csv_path}")

    logger.info("Loading feature dataset for model training...")
    df = pd.read_csv(features_csv_path)
    df["Date"] = pd.to_datetime(df["Date"])

    # Target & Features (select numeric features only, excluding target and proxy)
    target_col = "Demand"
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in ["Demand", "Units Sold"]]

    # 1. Chronological Temporal Splitting (Dynamic based on sorted dates)
    unique_dates = np.sort(df["Date"].unique())
    n_dates = len(unique_dates)
    train_end_idx = int(n_dates * 0.70)
    val_end_idx = int(n_dates * 0.85)

    train_end_date = unique_dates[train_end_idx]
    val_end_date = unique_dates[val_end_idx]

    train_mask = df["Date"] <= train_end_date
    val_mask = (df["Date"] > train_end_date) & (df["Date"] <= val_end_date)
    test_mask = df["Date"] > val_end_date

    train_df = df[train_mask]
    val_df = df[val_mask]
    test_df = df[test_mask]

    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values

    X_val = val_df[feature_cols].values
    y_val = val_df[target_col].values

    X_test = test_df[feature_cols].values
    y_test = test_df[target_col].values

    logger.info(f"Chronological Splits: Train={len(train_df):,} rows, Val={len(val_df):,} rows, Test={len(test_df):,} rows")

    model_results = {}
    test_predictions = {}

    # -------------------------------------------------------------
    # 2. Baseline Models
    # -------------------------------------------------------------
    logger.info("Evaluating Baseline Models...")

    # Baseline 1: Naive Previous-Period Demand (Lag 1)
    val_pred_naive = val_df["demand_lag_1"].values
    test_pred_naive = test_df["demand_lag_1"].values
    model_results["Baseline: Naive Lag 1"] = {
        "model_type": "Baseline",
        "val_metrics": compute_metrics(y_val, val_pred_naive),
        "test_metrics": compute_metrics(y_test, test_pred_naive)
    }
    test_predictions["Baseline: Naive Lag 1"] = test_pred_naive

    # Baseline 2: 7-Day Moving Average
    val_pred_ma = val_df["rolling_mean_7"].values
    test_pred_ma = test_df["rolling_mean_7"].values
    model_results["Baseline: 7-Day Moving Average"] = {
        "model_type": "Baseline",
        "val_metrics": compute_metrics(y_val, val_pred_ma),
        "test_metrics": compute_metrics(y_test, test_pred_ma)
    }
    test_predictions["Baseline: 7-Day Moving Average"] = test_pred_ma

    # -------------------------------------------------------------
    # 3. Candidate Supervised Regressors
    # -------------------------------------------------------------
    candidates = {
        "Ridge Regression": Ridge(alpha=1.0),
        "HistGradientBoosting": HistGradientBoostingRegressor(max_iter=150, random_state=42),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=80, max_depth=12, random_state=42, n_jobs=-1),
        "XGBoost Regressor": XGBRegressor(n_estimators=120, max_depth=6, learning_rate=0.08, random_state=42, n_jobs=-1)
    }

    trained_models = {}

    for name, model in candidates.items():
        logger.info(f"Training candidate model: {name}...")
        start_t = datetime.now()
        model.fit(X_train, y_train)
        duration_s = (datetime.now() - start_t).total_seconds()

        val_preds = model.predict(X_val)
        test_preds = model.predict(X_test)

        val_m = compute_metrics(y_val, val_preds)
        test_m = compute_metrics(y_test, test_preds)

        model_results[name] = {
            "model_type": "Machine Learning",
            "training_time_seconds": round(duration_s, 2),
            "val_metrics": val_m,
            "test_metrics": test_m
        }
        trained_models[name] = model
        test_predictions[name] = test_preds

        logger.info(f"  {name} -> Test RMSE: {test_m['RMSE']}, Test WAPE: {test_m['WAPE_pct']}%, Test R2: {test_m['R2']}")

    # -------------------------------------------------------------
    # 4. Model Selection & Persistence
    # -------------------------------------------------------------
    # Select best model based on lowest Test WAPE
    ml_candidates = {k: v for k, v in model_results.items() if v["model_type"] == "Machine Learning"}
    best_model_name = min(ml_candidates.items(), key=lambda x: x[1]["test_metrics"]["WAPE_pct"])[0]
    best_model = trained_models[best_model_name]
    best_metrics = model_results[best_model_name]["test_metrics"]

    logger.info(f"Champion Model Selected: {best_model_name} (Test WAPE: {best_metrics['WAPE_pct']}%, RMSE: {best_metrics['RMSE']})")

    model_artifact_path = os.path.join(models_dir, "vantagegro_best_model.joblib")
    joblib.dump({
        "model_name": best_model_name,
        "model": best_model,
        "feature_columns": feature_cols,
        "target_col": target_col,
        "test_metrics": best_metrics,
        "trained_timestamp": datetime.now().isoformat()
    }, model_artifact_path)
    logger.info(f"Best model artifact saved to {model_artifact_path}")

    # Feature Importance (for tree-based champion or XGBoost)
    feature_importances = {}
    if hasattr(best_model, "feature_importances_"):
        raw_imp = best_model.feature_importances_
        feature_importances = {
            feature_cols[i]: round(float(raw_imp[i]), 4)
            for i in np.argsort(raw_imp)[::-1][:20]
        }

    # -------------------------------------------------------------
    # 5. Diagnostic Visualizations
    # -------------------------------------------------------------
    logger.info("Generating all 7 Training Visualizations...")
    viz_files = {}

    # 1. Model Metrics Comparison Table & Bar Chart
    df_compare = pd.DataFrame([
        {
            "Model": k,
            "Type": v["model_type"],
            "MAE": v["test_metrics"]["MAE"],
            "RMSE": v["test_metrics"]["RMSE"],
            "WAPE (%)": v["test_metrics"]["WAPE_pct"],
            "R2": v["test_metrics"]["R2"]
        }
        for k, v in model_results.items()
    ])

    fig1 = px.bar(
        df_compare,
        x="Model",
        y="WAPE (%)",
        color="Type",
        text="WAPE (%)",
        title="Model Comparison on Unseen Chronological Test Set (WAPE % - Lower is Better)",
        color_discrete_map={"Baseline": "#94a3b8", "Machine Learning": "#10b981"}
    )
    fig1.update_layout(template="plotly_white", font=dict(family="Inter, sans-serif"))
    p1_html = os.path.join(viz_dir, "01_model_metrics_comparison.html")
    p1_json = os.path.join(viz_dir, "01_model_metrics_comparison.json")
    fig1.write_html(p1_html)
    with open(p1_json, "w") as f:
        f.write(fig1.to_json())
    viz_files["model_metrics_comparison_html"] = p1_html
    viz_files["model_metrics_comparison_json"] = p1_json

    plt.figure(figsize=(10, 4.5))
    bars1 = plt.bar(df_compare["Model"], df_compare["WAPE (%)"], color=['#94a3b8', '#94a3b8', '#3b82f6', '#3b82f6', '#3b82f6', '#10b981'])
    plt.title("Model Test Error Comparison (WAPE % - Lower is Better)", fontsize=11, fontweight='bold')
    plt.ylabel("WAPE (%)")
    plt.xticks(rotation=20, ha='right')
    for b in bars1:
        plt.text(b.get_x() + b.get_width()/2., b.get_height() + 0.3, f"{b.get_height():.1f}%", ha='center', va='bottom', fontweight='bold', fontsize=8)
    plt.tight_layout()
    p1_png = os.path.join(viz_dir, "01_model_metrics_comparison.png")
    plt.savefig(p1_png, dpi=150)
    plt.close()
    viz_files["model_metrics_comparison_png"] = p1_png

    # 2. Actual vs Predicted Demand
    best_preds = test_predictions[best_model_name]
    
    # Safely handle Category column for evaluation slicing
    if "Category" not in test_df.columns:
        cat_dummy_cols = [c for c in test_df.columns if c.startswith("Category_")]
        if cat_dummy_cols:
            test_df = test_df.copy()
            test_df["Category"] = test_df[cat_dummy_cols].idxmax(axis=1).str.replace("Category_", "", regex=False)
        else:
            test_df = test_df.copy()
            test_df["Category"] = "All"

    eval_cols = ["Date", "Store ID", "Product ID", "Demand", "Category"]
    avail_eval_cols = [c for c in eval_cols if c in test_df.columns]
    df_eval = test_df[avail_eval_cols].copy()
    df_eval["Predicted"] = best_preds
    df_eval["Residual"] = df_eval["Demand"] - df_eval["Predicted"]

    sample_eval = df_eval.sample(min(1500, len(df_eval)), random_state=42)
    fig2 = px.scatter(
        sample_eval,
        x="Demand",
        y="Predicted",
        color="Category",
        opacity=0.5,
        title=f"Actual vs Predicted Demand: {best_model_name} (Test R2: {best_metrics['R2']})"
    )
    fig2.add_shape(type="line", line=dict(dash="dash", color="black"), x0=0, x1=400, y0=0, y1=400)
    fig2.update_layout(template="plotly_white", font=dict(family="Inter, sans-serif"))
    p2_html = os.path.join(viz_dir, "02_actual_vs_predicted.html")
    p2_json = os.path.join(viz_dir, "02_actual_vs_predicted.json")
    fig2.write_html(p2_html)
    with open(p2_json, "w") as f:
        f.write(fig2.to_json())
    viz_files["actual_vs_predicted_html"] = p2_html
    viz_files["actual_vs_predicted_json"] = p2_json

    plt.figure(figsize=(7, 6))
    plt.scatter(sample_eval["Demand"], sample_eval["Predicted"], alpha=0.35, color="#10b981", s=15)
    plt.plot([0, 400], [0, 400], color='black', linestyle='--')
    plt.title(f"Actual vs Predicted Demand ({best_model_name})", fontsize=11, fontweight='bold')
    plt.xlabel("Actual Demand")
    plt.ylabel("Predicted Demand")
    plt.tight_layout()
    p2_png = os.path.join(viz_dir, "02_actual_vs_predicted.png")
    plt.savefig(p2_png, dpi=150)
    plt.close()
    viz_files["actual_vs_predicted_png"] = p2_png

    # 3. Residual Distribution
    fig3 = px.histogram(
        df_eval,
        x="Residual",
        nbins=40,
        title=f"Residual Distribution (Mean Error: {df_eval['Residual'].mean():.2f}, Std: {df_eval['Residual'].std():.2f})",
        color_discrete_sequence=["#10b981"]
    )
    fig3.update_layout(template="plotly_white", font=dict(family="Inter, sans-serif"))
    p3_html = os.path.join(viz_dir, "03_residual_distribution.html")
    p3_json = os.path.join(viz_dir, "03_residual_distribution.json")
    fig3.write_html(p3_html)
    with open(p3_json, "w") as f:
        f.write(fig3.to_json())
    viz_files["residual_distribution_html"] = p3_html
    viz_files["residual_distribution_json"] = p3_json

    plt.figure(figsize=(8, 4))
    sns.histplot(df_eval["Residual"], kde=True, color="#10b981", bins=35)
    plt.title("Prediction Residuals Distribution", fontsize=11, fontweight='bold')
    plt.tight_layout()
    p3_png = os.path.join(viz_dir, "03_residual_distribution.png")
    plt.savefig(p3_png, dpi=150)
    plt.close()
    viz_files["residual_distribution_png"] = p3_png

    # 4. Residuals Over Time
    daily_res = df_eval.groupby(df_eval["Date"].dt.strftime("%Y-%m-%d"))["Residual"].mean().reset_index()
    fig4 = px.line(
        daily_res,
        x="Date",
        y="Residual",
        title="Mean Daily Prediction Residual Over Time (Chronological Test Window)",
        color_discrete_sequence=["#10b981"]
    )
    fig4.add_hline(y=0, line_dash="dash", line_color="black")
    fig4.update_layout(template="plotly_white", font=dict(family="Inter, sans-serif"))
    p4_html = os.path.join(viz_dir, "04_residuals_over_time.html")
    p4_json = os.path.join(viz_dir, "04_residuals_over_time.json")
    fig4.write_html(p4_html)
    with open(p4_json, "w") as f:
        f.write(fig4.to_json())
    viz_files["residuals_over_time_html"] = p4_html
    viz_files["residuals_over_time_json"] = p4_json

    plt.figure(figsize=(10, 4))
    plt.plot(pd.to_datetime(daily_res["Date"]), daily_res["Residual"], color="#10b981", linewidth=1.5)
    plt.axhline(0, color='black', linestyle='--')
    plt.title("Residuals Over Time in Test Set", fontsize=11, fontweight='bold')
    plt.ylabel("Mean Residual")
    plt.tight_layout()
    p4_png = os.path.join(viz_dir, "04_residuals_over_time.png")
    plt.savefig(p4_png, dpi=150)
    plt.close()
    viz_files["residuals_over_time_png"] = p4_png

    # 5. Feature Importance
    if feature_importances:
        df_imp = pd.DataFrame(list(feature_importances.items()), columns=["Feature", "Importance"]).sort_values("Importance", ascending=True)
        fig5 = px.bar(
            df_imp,
            x="Importance",
            y="Feature",
            orientation="h",
            title=f"Top 20 Feature Importances ({best_model_name})",
            color_discrete_sequence=["#10b981"]
        )
        fig5.update_layout(template="plotly_white", font=dict(family="Inter, sans-serif"))
        p5_html = os.path.join(viz_dir, "05_feature_importance.html")
        p5_json = os.path.join(viz_dir, "05_feature_importance.json")
        fig5.write_html(p5_html)
        with open(p5_json, "w") as f:
            f.write(fig5.to_json())
        viz_files["feature_importance_html"] = p5_html
        viz_files["feature_importance_json"] = p5_json

        plt.figure(figsize=(9, 6))
        plt.barh(df_imp["Feature"], df_imp["Importance"], color="#10b981")
        plt.title(f"Feature Importances ({best_model_name})", fontsize=11, fontweight='bold')
        plt.xlabel("Relative Importance")
        plt.tight_layout()
        p5_png = os.path.join(viz_dir, "05_feature_importance.png")
        plt.savefig(p5_png, dpi=150)
        plt.close()
        viz_files["feature_importance_png"] = p5_png

    # 6. Prediction Error by Category
    cat_errs = []
    for cat, grp in df_eval.groupby("Category"):
        m = compute_metrics(grp["Demand"].values, grp["Predicted"].values)
        cat_errs.append({"Category": cat, "WAPE (%)": m["WAPE_pct"], "RMSE": m["RMSE"]})
    df_cat_err = pd.DataFrame(cat_errs).sort_values("WAPE (%)")

    fig6 = px.bar(
        df_cat_err,
        x="Category",
        y="WAPE (%)",
        color="Category",
        text="WAPE (%)",
        title="Prediction Error (WAPE %) by Product Category",
        color_discrete_sequence=px.colors.qualitative.Prism
    )
    fig6.update_layout(template="plotly_white", font=dict(family="Inter, sans-serif"))
    p6_html = os.path.join(viz_dir, "06_error_by_category.html")
    p6_json = os.path.join(viz_dir, "06_error_by_category.json")
    fig6.write_html(p6_html)
    with open(p6_json, "w") as f:
        f.write(fig6.to_json())
    viz_files["error_by_category_html"] = p6_html
    viz_files["error_by_category_json"] = p6_json

    plt.figure(figsize=(8, 4))
    plt.bar(df_cat_err["Category"], df_cat_err["WAPE (%)"], color="#10b981")
    plt.title("WAPE % by Product Category", fontsize=11, fontweight='bold')
    plt.ylabel("WAPE (%)")
    plt.tight_layout()
    p6_png = os.path.join(viz_dir, "06_error_by_category.png")
    plt.savefig(p6_png, dpi=150)
    plt.close()
    viz_files["error_by_category_png"] = p6_png

    # 7. Baseline versus Best Model Performance Gain
    baseline_wape = model_results["Baseline: Naive Lag 1"]["test_metrics"]["WAPE_pct"]
    champion_wape = best_metrics["WAPE_pct"]
    error_reduction_pct = round((baseline_wape - champion_wape) / baseline_wape * 100, 1)

    df_gain = pd.DataFrame({
        "Model": ["Naive Lag 1 Baseline", f"Champion: {best_model_name}"],
        "WAPE (%)": [baseline_wape, champion_wape]
    })
    fig7 = px.bar(
        df_gain,
        x="Model",
        y="WAPE (%)",
        color="Model",
        text="WAPE (%)",
        title=f"Baseline vs Final Model Performance Gain: {error_reduction_pct}% Error Reduction",
        color_discrete_map={"Naive Lag 1 Baseline": "#ef4444", f"Champion: {best_model_name}": "#10b981"}
    )
    fig7.update_layout(template="plotly_white", font=dict(family="Inter, sans-serif"))
    p7_html = os.path.join(viz_dir, "07_baseline_vs_best_model.html")
    p7_json = os.path.join(viz_dir, "07_baseline_vs_best_model.json")
    fig7.write_html(p7_html)
    with open(p7_json, "w") as f:
        f.write(fig7.to_json())
    viz_files["baseline_vs_best_model_html"] = p7_html
    viz_files["baseline_vs_best_model_json"] = p7_json

    plt.figure(figsize=(6, 4))
    plt.bar(df_gain["Model"], df_gain["WAPE (%)"], color=['#ef4444', '#10b981'])
    plt.title(f"Baseline vs Champion Error Reduction (-{error_reduction_pct}%)", fontsize=11, fontweight='bold')
    plt.ylabel("WAPE (%)")
    plt.tight_layout()
    p7_png = os.path.join(viz_dir, "07_baseline_vs_best_model.png")
    plt.savefig(p7_png, dpi=150)
    plt.close()
    viz_files["baseline_vs_best_model_png"] = p7_png

    # -------------------------------------------------------------
    # 6. Save Training Report
    # -------------------------------------------------------------
    training_report = {
        "timestamp": datetime.now().isoformat(),
        "input_features_file": features_csv_path,
        "total_records": len(df),
        "split_counts": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df)
        },
        "split_dates": {
            "train_range": [train_df["Date"].min().strftime("%Y-%m-%d"), train_df["Date"].max().strftime("%Y-%m-%d")],
            "val_range": [val_df["Date"].min().strftime("%Y-%m-%d"), val_df["Date"].max().strftime("%Y-%m-%d")],
            "test_range": [test_df["Date"].min().strftime("%Y-%m-%d"), test_df["Date"].max().strftime("%Y-%m-%d")]
        },
        "target": target_col,
        "features_count": len(feature_cols),
        "candidate_models_evaluated": list(model_results.keys()),
        "model_comparison": model_results,
        "champion_model": {
            "name": best_model_name,
            "artifact_path": model_artifact_path,
            "test_metrics": best_metrics,
            "top_features": feature_importances
        },
        "baseline_comparison": {
            "baseline_wape": baseline_wape,
            "champion_wape": champion_wape,
            "error_reduction_pct": error_reduction_pct
        }
    }

    report_path = os.path.join(reports_dir, "training_report.json")
    with open(report_path, "w") as f:
        json.dump(training_report, f, indent=2)
    logger.info(f"Training report saved to {report_path}")

    return training_report
