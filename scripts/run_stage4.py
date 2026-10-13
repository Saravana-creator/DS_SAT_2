"""
VantageGro Retail Stock Insight - Stage 4 Execution Script
Executes Feature Engineering, constructs calendar, lag, rolling, and business features,
prevents target leakage, and generates all 5 Feature Visualizations.
"""

import sys
import os
import json
import logging
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.services.feature_service import execute_stage4_feature_pipeline
from backend.app.services.visualization_service import generate_stage4_feature_visualizations

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("stage4_runner")


def main():
    cleaned_csv_path = "data/processed/vantagegro_cleaned.csv"
    output_dir = "data/processed"
    reports_dir = "data/reports"
    viz_dir = "data/reports/feature_visualizations"

    logger.info("==================================================================")
    logger.info("Starting Stage 4: Feature Engineering Pipeline")
    logger.info("==================================================================")

    # 1. Execute Feature Engineering
    metadata = execute_stage4_feature_pipeline(
        cleaned_csv_path=cleaned_csv_path,
        output_dir=output_dir,
        reports_dir=reports_dir
    )

    features_csv = os.path.join(output_dir, "vantagegro_features.csv")
    df_features = pd.read_csv(features_csv)
    logger.info(f"Loaded engineered feature dataset: {df_features.shape[0]} rows x {df_features.shape[1]} columns")
    logger.info(f"Total features created: {metadata['total_features']}")

    # 2. Generate 5 Visualizations
    logger.info("Generating all 5 Stage 4 Feature Visualizations (Plotly HTML/JSON & PNG)...")
    viz_files = generate_stage4_feature_visualizations(
        feature_metadata=metadata,
        df_features=df_features,
        output_dir=viz_dir
    )

    for k, v in viz_files.items():
        logger.info(f"  Generated: {k} -> {v}")

    logger.info("Stage 4 Feature Engineering Pipeline Completed Successfully!")


if __name__ == "__main__":
    main()
