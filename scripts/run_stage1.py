"""
VantageGro Retail Stock Insight - Stage 1 Execution Script
Executes Dataset Ingestion, Inspection, Aggregation, Chronological Merge, Audit Report,
and Visualization Generation.
"""

import sys
import os
import json
import logging

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.services.dataset_service import merge_datasets_pipeline, profile_dataset, aggregate_dataset_1
from backend.app.services.visualization_service import generate_stage1_merge_visualizations
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("stage1_runner")


def main():
    primary_path = "data/raw/dataset2_retail_inventory/sales_data.csv"
    dataset1_dir = "data/raw/dataset1_store_sales"
    output_dir = "data/processed"
    reports_dir = "data/reports"
    viz_dir = "data/reports/merge_visualizations"

    logger.info("==================================================================")
    logger.info("Starting Stage 1: Dataset Inspection, Merging & Validation Pipeline")
    logger.info("==================================================================")

    # 1. Profile primary dataset
    logger.info(f"Profiling primary inventory dataset at {primary_path}...")
    p_profile = profile_dataset(primary_path)
    logger.info(f"Primary dataset rows: {p_profile['row_count']}, columns: {p_profile['column_count']}")
    logger.info(f"Primary date range: {p_profile['date_range']['min']} to {p_profile['date_range']['max']} ({p_profile['date_range']['unique_dates']} unique dates)")
    logger.info(f"Full duplicate rows: {p_profile['duplicate_rows']}")

    # 2. Aggregate Dataset 1
    logger.info(f"Aggregating Dataset 1 from {dataset1_dir}...")
    ext_df, ext_meta = aggregate_dataset_1(dataset1_dir)
    logger.info(f"Files aggregated: {ext_meta['files_aggregated']}")
    logger.info(f"Aggregated date range: {ext_meta.get('date_min')} to {ext_meta.get('date_max')} ({ext_meta.get('aggregated_date_count')} unique dates)")

    # 3. Execute Merge Pipeline
    logger.info("Executing chronological date merge and suitability evaluation...")
    merge_report = merge_datasets_pipeline(
        primary_file_path=primary_path,
        dataset1_dir=dataset1_dir,
        output_dir=output_dir,
        reports_dir=reports_dir
    )

    logger.info(f"Merge Match Rate: {merge_report['overlap_analysis']['merge_match_rate_pct']}%")
    logger.info(f"Overlapping Dates Count: {merge_report['overlap_analysis']['overlapping_dates_count']}")
    logger.info(f"Feature Suitability Decision:\n{merge_report['feature_suitability_decision']}")
    logger.info(f"Output saved to: {merge_report['final_output_file']} (Rows: {merge_report['final_row_count']}, Cols: {merge_report['final_column_count']})")

    # 4. Generate Visualizations
    logger.info("Generating all 6 required Stage 1 visualizations (Plotly HTML/JSON & PNG)...")
    primary_df = pd.read_csv(primary_path)
    viz_files = generate_stage1_merge_visualizations(
        merge_report=merge_report,
        primary_df=primary_df,
        ext_df=ext_df,
        output_dir=viz_dir
    )
    for k, v in viz_files.items():
        logger.info(f"  Generated: {k} -> {v}")

    logger.info("Stage 1 Pipeline Execution Completed Successfully!")


if __name__ == "__main__":
    main()
