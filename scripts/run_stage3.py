"""
VantageGro Retail Stock Insight - Stage 3 Execution Script
Executes Exploratory Data Analysis (EDA) on cleaned dataset,
calculates descriptive analytics, answers retail business questions,
and generates all 12 interactive & static EDA charts.
"""

import sys
import os
import json
import logging
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.services.eda_service import execute_stage3_eda
from backend.app.services.visualization_service import generate_stage3_eda_visualizations

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("stage3_runner")


def main():
    cleaned_csv_path = "data/processed/vantagegro_cleaned.csv"
    reports_dir = "data/reports"
    viz_dir = "data/reports/eda_visualizations"

    logger.info("==================================================================")
    logger.info("Starting Stage 3: Exploratory Data Analysis (EDA) Pipeline")
    logger.info("==================================================================")

    df = pd.read_csv(cleaned_csv_path)
    logger.info(f"Loaded cleaned dataset: {len(df)} rows, {len(df.columns)} columns")

    # 1. Execute EDA Summaries
    eda_summary = execute_stage3_eda(
        cleaned_csv_path=cleaned_csv_path,
        reports_dir=reports_dir
    )

    logger.info(f"Top selling category: {eda_summary['business_insights']['top_selling_categories'][0]}")
    logger.info(f"Top performing store: {eda_summary['business_insights']['top_performing_store']}")
    logger.info(f"Stockout risk rate: {eda_summary['risk_metrics']['stockout_risk_pct']}%")
    logger.info(f"Overstock risk rate: {eda_summary['risk_metrics']['overstock_risk_pct']}%")

    # 2. Generate all 12 Visualizations
    logger.info("Generating all 12 Stage 3 EDA visualizations (Plotly HTML/JSON & PNG)...")
    viz_files = generate_stage3_eda_visualizations(
        eda_summary=eda_summary,
        df=df,
        output_dir=viz_dir
    )

    for k, v in viz_files.items():
        logger.info(f"  Generated: {k} -> {v}")

    logger.info("Stage 3 EDA Pipeline Execution Completed Successfully!")


if __name__ == "__main__":
    main()
