"""
VantageGro Retail Stock Insight - Stage 2 Execution Script
Executes Data Quality Assessment, Domain Sanity Verification,
Reproducible Preprocessing & Cleaning, Report Generation, and Visualizations.
"""

import sys
import os
import json
import logging
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.services.preprocessing_service import execute_stage2_pipeline
from backend.app.services.visualization_service import generate_stage2_preprocessing_visualizations

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("stage2_runner")


def main():
    raw_input_path = "data/processed/vantagegro_merged_raw.csv"
    output_dir = "data/processed"
    reports_dir = "data/reports"
    viz_dir = "data/reports/preprocessing_visualizations"

    logger.info("==================================================================")
    logger.info("Starting Stage 2: Data Quality Assessment & Preprocessing Pipeline")
    logger.info("==================================================================")

    df_before = pd.read_csv(raw_input_path)
    logger.info(f"Loaded raw merged dataset: {df_before.shape[0]} rows, {df_before.shape[1]} columns")

    # 1. Execute Pipeline
    report = execute_stage2_pipeline(
        raw_input_path=raw_input_path,
        output_dir=output_dir,
        reports_dir=reports_dir
    )

    df_after = pd.read_csv(report["cleaned_output"])
    logger.info(f"Cleaned dataset saved: {df_after.shape[0]} rows, {df_after.shape[1]} columns")
    logger.info(f"Zero inventory days: {report['quality_after']['stockout_analysis'].get('zero_inventory_days')}")
    logger.info(f"Stockout censored records: {report['quality_after']['stockout_analysis'].get('stockout_censored_records')}")

    # 2. Generate 9 Stage 2 Visualizations
    logger.info("Generating all 9 required Stage 2 Preprocessing visualizations (Plotly HTML/JSON & PNG)...")
    viz_files = generate_stage2_preprocessing_visualizations(
        preprocessing_report=report,
        df_before=df_before,
        df_after=df_after,
        output_dir=viz_dir
    )

    for k, v in viz_files.items():
        logger.info(f"  Generated: {k} -> {v}")

    logger.info("Stage 2 Pipeline Execution Completed Successfully!")


if __name__ == "__main__":
    main()
