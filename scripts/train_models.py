"""
VantageGro Retail Stock Insight - Stage 5 Model Training CLI Runner
Run this script directly in the terminal to execute Chronological Train/Val/Test Splitting,
evaluate Baselines vs Candidate Regressors (Ridge, RF, HistGBR, XGBoost),
persist the Champion Model to models/vantagegro_best_model.joblib,
and generate all 7 diagnostic visualizations.
"""

import sys
import os
import logging
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.services.training_service import train_and_evaluate_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_models_cli")


def main():
    features_csv = "data/processed/vantagegro_features.csv"
    models_dir = "models"
    reports_dir = "data/reports"
    viz_dir = "data/reports/training_visualizations"

    print("\n" + "="*70)
    print("VANTAGEGRO RETAIL STOCK INSIGHT - STAGE 5 MODEL TRAINING")
    print("="*70)

    if not os.path.exists(features_csv):
        print(f"[ERROR] Features file not found at {features_csv}. Run Stage 4 first!")
        sys.exit(1)

    report = train_and_evaluate_pipeline(
        features_csv_path=features_csv,
        models_dir=models_dir,
        reports_dir=reports_dir,
        viz_dir=viz_dir
    )

    print("\n" + "="*70)
    print("MODEL EVALUATION COMPARISON RESULTS (UNSEEN CHRONOLOGICAL TEST SET)")
    print("="*70)
    print(f"{'Model Name':<32} | {'Test MAE':<10} | {'Test RMSE':<10} | {'Test WAPE':<10} | {'Test R^2':<8}")
    print("-" * 78)

    for m_name, res in report["model_comparison"].items():
        tm = res["test_metrics"]
        print(f"{m_name:<32} | {tm['MAE']:<10.2f} | {tm['RMSE']:<10.2f} | {tm['WAPE_pct']:<9.2f}% | {tm['R2']:<8.4f}")

    champ = report["champion_model"]
    print("="*78)
    print(f"\n>>> CHAMPION MODEL SELECTED: {champ['name']}")
    print(f"    - Test MAE:   {champ['test_metrics']['MAE']:.2f} units")
    print(f"    - Test RMSE:  {champ['test_metrics']['RMSE']:.2f} units")
    print(f"    - Test WAPE:  {champ['test_metrics']['WAPE_pct']:.2f}%")
    print(f"    - Test R^2:   {champ['test_metrics']['R2']:.4f}")
    print(f"    - Error Reduction vs Naive Baseline: -{report['baseline_comparison']['error_reduction_pct']}%\n")
    print(f"Model artifact saved to: {champ['artifact_path']}")
    print(f"Training report saved to: data/reports/training_report.json")
    print(f"Visualizations saved to:  data/reports/training_visualizations/\n")
    print("Stage 5 Model Training Completed Successfully!\n")


if __name__ == "__main__":
    main()
