"""
VantageGro Retail Stock Insight - Database Service (Stage 7)
Provides data access layer connecting pipeline artifacts to MongoDB.
Supports seamless reading, querying, indexing, and offline-resilient fallback to local disk.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
import pandas as pd
from datetime import datetime

from backend.app.core.config import settings
from backend.app.db.mongodb import db_manager, MongoDBManager

logger = logging.getLogger(__name__)


class DatabaseService:
    """Encapsulates data persistence and retrieval between MongoDB and filesystem."""

    def __init__(self, manager: MongoDBManager = db_manager):
        self.manager = manager

    def check_health(self) -> Dict[str, Any]:
        """Probes database connectivity and returns operational status."""
        return self.manager.check_health()

    def seed_pipeline_data(self, force: bool = False) -> Dict[str, Any]:
        """
        Seeds all artifacts from Stages 1 through 6 into their respective MongoDB collections.
        If force=False and data is already present, preserves existing collections.
        """
        db = self.manager.get_database()
        if db is None:
            return {
                "status": "degraded",
                "message": "MongoDB is offline. Seeding skipped; running in filesystem fallback mode.",
                "details": self.manager.last_error
            }

        # 1. Ensure indexes
        index_result = self.manager.ensure_indexes()

        seeded_counts = {}

        # 2. Seed Ingestion Logs (Stage 1)
        merge_report_path = os.path.join(settings.REPORTS_DIR, "merge_report.json")
        if os.path.exists(merge_report_path):
            with open(merge_report_path, "r") as f:
                merge_data = json.load(f)
            merge_doc = {
                "stage": "Stage 1: Dataset Merging",
                "timestamp": merge_data.get("timestamp", datetime.now().isoformat()),
                "total_rows": merge_data.get("merged_dataset", {}).get("total_rows", 76000),
                "total_columns": merge_data.get("merged_dataset", {}).get("total_columns", 16),
                "date_range": merge_data.get("merged_dataset", {}).get("date_range", {}),
                "merge_notes": merge_data.get("merge_notes", []),
                "created_at": datetime.now()
            }
            if force or db.ingestion_logs.count_documents({"stage": "Stage 1: Dataset Merging"}) == 0:
                if force:
                    db.ingestion_logs.delete_many({"stage": "Stage 1: Dataset Merging"})
                db.ingestion_logs.insert_one(merge_doc)
            seeded_counts["ingestion_logs"] = db.ingestion_logs.count_documents({})

        # 3. Seed Quality Reports (Stage 2)
        quality_report_path = os.path.join(settings.PROCESSED_DATA_DIR, "preprocessing_report.json")
        if os.path.exists(quality_report_path):
            with open(quality_report_path, "r") as f:
                qual_data = json.load(f)
            qual_doc = {
                "stage": "Stage 2: Preprocessing & Quality Assessment",
                "timestamp": qual_data.get("timestamp", datetime.now().isoformat()),
                "cleaning_actions": qual_data.get("cleaning_actions_taken", []),
                "summary": qual_data.get("summary", {}),
                "quality_before": qual_data.get("quality_before", {}),
                "quality_after": qual_data.get("quality_after", {}),
                "outliers_summary": qual_data.get("outliers_summary", {}),
                "created_at": datetime.now()
            }
            if force or db.quality_reports.count_documents({"stage": "Stage 2: Preprocessing & Quality Assessment"}) == 0:
                if force:
                    db.quality_reports.delete_many({"stage": "Stage 2: Preprocessing & Quality Assessment"})
                db.quality_reports.insert_one(qual_doc)
            seeded_counts["quality_reports"] = db.quality_reports.count_documents({})

        # 4. Seed EDA Summaries (Stage 3)
        eda_summary_path = os.path.join(settings.REPORTS_DIR, "eda_summary.json")
        if os.path.exists(eda_summary_path):
            with open(eda_summary_path, "r") as f:
                eda_data = json.load(f)
            eda_doc = {
                "stage": "Stage 3: Exploratory Data Analysis",
                "timestamp": eda_data.get("timestamp", datetime.now().isoformat()),
                "total_observations": eda_data.get("total_observations", 76000),
                "categories": eda_data.get("categories", {}),
                "risk_metrics": eda_data.get("risk_metrics", {}),
                "promotional_impact": eda_data.get("promotional_impact", {}),
                "stores": eda_data.get("stores", {}),
                "created_at": datetime.now()
            }
            if force or db.eda_summaries.count_documents({}) == 0:
                if force:
                    db.eda_summaries.delete_many({})
                db.eda_summaries.insert_one(eda_doc)
            seeded_counts["eda_summaries"] = db.eda_summaries.count_documents({})

        # 5. Seed Model Registry (Stage 5)
        train_report_path = os.path.join(settings.REPORTS_DIR, "training_report.json")
        if os.path.exists(train_report_path):
            with open(train_report_path, "r") as f:
                train_data = json.load(f)
            models_comp = train_data.get("model_comparison", {})
            champ_name = train_data.get("champion_model", {}).get("name", "HistGradientBoosting")

            if force or db.model_registry.count_documents({}) == 0:
                if force:
                    db.model_registry.delete_many({})
                for m_name, m_info in models_comp.items():
                    m_doc = {
                        "model_name": m_name,
                        "timestamp": train_data.get("timestamp", datetime.now().isoformat()),
                        "is_champion": (m_name == champ_name),
                        "test_metrics": m_info.get("test_metrics", {}),
                        "train_metrics": m_info.get("train_metrics", {}),
                        "features_count": len(train_data.get("features_used", [])),
                        "split_info": train_data.get("split_info", {}),
                        "created_at": datetime.now()
                    }
                    db.model_registry.insert_one(m_doc)
            seeded_counts["model_registry"] = db.model_registry.count_documents({})

        # 6. Seed Inventory Recommendations (Stage 6)
        rec_csv_path = os.path.join(settings.PROCESSED_DATA_DIR, "inventory_recommendations.csv")
        if os.path.exists(rec_csv_path):
            df_recs = pd.read_csv(rec_csv_path)
            if force or db.inventory_recommendations.count_documents({}) == 0:
                if force:
                    db.inventory_recommendations.delete_many({})
                records = df_recs.to_dict(orient="records")
                for r in records:
                    r["updated_at"] = datetime.now()
                db.inventory_recommendations.insert_many(records)
            seeded_counts["inventory_recommendations"] = db.inventory_recommendations.count_documents({})

        # 7. Seed Demand Forecast Trajectories (Stage 6)
        forecast_report_path = os.path.join(settings.REPORTS_DIR, "forecast_and_inventory_report.json")
        if os.path.exists(forecast_report_path):
            # Also read recommendations csv to get per-series forecast data
            if force or db.demand_forecasts.count_documents({}) == 0:
                if force:
                    db.demand_forecasts.delete_many({})
                with open(forecast_report_path, "r") as f:
                    f_meta = json.load(f)
                meta_doc = {
                    "report_type": "metadata",
                    "as_of_date": f_meta.get("as_of_date"),
                    "horizon_days": f_meta.get("forecast_horizon_days", 14),
                    "total_series": f_meta.get("total_sku_series", 100),
                    "risk_breakdown": f_meta.get("risk_breakdown", {}),
                    "priority_breakdown": f_meta.get("priority_breakdown", {}),
                    "total_reorder_units": f_meta.get("total_suggested_reorder_units", 0),
                    "parameters": f_meta.get("parameters", {}),
                    "created_at": datetime.now()
                }
                db.demand_forecasts.insert_one(meta_doc)
            seeded_counts["demand_forecasts"] = db.demand_forecasts.count_documents({})

        logger.info(f"MongoDB seeded successfully: {seeded_counts}")
        return {
            "status": "success",
            "message": "MongoDB seeded successfully with pipeline data.",
            "seeded_counts": seeded_counts,
            "indexes": index_result
        }

    # Query Helpers with Fallback
    def get_inventory_recommendations(
        self,
        store_id: Optional[str] = None,
        category: Optional[str] = None,
        risk_category: Optional[str] = None,
        priority: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Retrieves inventory recommendations with filtering from MongoDB, falling back to CSV."""
        db = self.manager.get_database()
        if db is not None:
            try:
                query: Dict[str, Any] = {}
                if store_id:
                    query["Store ID"] = store_id
                if category:
                    query["Category"] = category
                if risk_category:
                    query["Risk Category"] = risk_category
                if priority:
                    query["Review Priority"] = priority

                cursor = db.inventory_recommendations.find(query, {"_id": 0}).limit(limit)
                results = list(cursor)
                if results:
                    return results
            except Exception as e:
                logger.warning(f"Error querying MongoDB for recommendations: {e}. Falling back to CSV.")

        # Fallback to CSV
        csv_path = os.path.join(settings.PROCESSED_DATA_DIR, "inventory_recommendations.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            if store_id:
                df = df[df["Store ID"] == store_id]
            if category:
                df = df[df["Category"] == category]
            if risk_category:
                df = df[df["Risk Category"] == risk_category]
            if priority:
                df = df[df["Review Priority"] == priority]
            return df.head(limit).to_dict(orient="records")

        return []

    def get_model_registry(self) -> List[Dict[str, Any]]:
        """Retrieves model registry records with fallback to JSON report."""
        db = self.manager.get_database()
        if db is not None:
            try:
                cursor = db.model_registry.find({}, {"_id": 0}).sort("test_metrics.WAPE_pct", 1)
                results = list(cursor)
                if results:
                    return results
            except Exception as e:
                logger.warning(f"Error querying model registry: {e}")

        json_path = os.path.join(settings.REPORTS_DIR, "training_report.json")
        if os.path.exists(json_path):
            with open(json_path) as f:
                data = json.load(f)
            models = []
            champ = data.get("champion_model", {}).get("name")
            for m_name, m_info in data.get("model_comparison", {}).items():
                models.append({
                    "model_name": m_name,
                    "is_champion": (m_name == champ),
                    "test_metrics": m_info.get("test_metrics", {}),
                    "features_count": len(data.get("features_used", []))
                })
            return models
        return []

    def get_eda_summary(self) -> Dict[str, Any]:
        """Retrieves Stage 3 EDA summary with fallback to JSON report."""
        db = self.manager.get_database()
        if db is not None:
            try:
                doc = db.eda_summaries.find_one({}, {"_id": 0}, sort=[("timestamp", -1)])
                if doc:
                    return doc
            except Exception as e:
                logger.warning(f"Error querying eda_summaries: {e}")

        json_path = os.path.join(settings.REPORTS_DIR, "eda_summary.json")
        if os.path.exists(json_path):
            with open(json_path) as f:
                return json.load(f)
        return {}

    def get_quality_report(self) -> Dict[str, Any]:
        """Retrieves Stage 2 quality assessment report with fallback to JSON report."""
        db = self.manager.get_database()
        if db is not None:
            try:
                doc = db.quality_reports.find_one({}, {"_id": 0}, sort=[("timestamp", -1)])
                if doc:
                    return doc
            except Exception as e:
                logger.warning(f"Error querying quality_reports: {e}")

        json_path = os.path.join(settings.PROCESSED_DATA_DIR, "preprocessing_report.json")
        if os.path.exists(json_path):
            with open(json_path) as f:
                return json.load(f)
        return {}

    def get_ingestion_logs(self) -> List[Dict[str, Any]]:
        """Retrieves Stage 1 ingestion records with fallback to JSON report."""
        db = self.manager.get_database()
        if db is not None:
            try:
                cursor = db.ingestion_logs.find({}, {"_id": 0}).sort("timestamp", -1)
                results = list(cursor)
                if results:
                    return results
            except Exception as e:
                logger.warning(f"Error querying ingestion_logs: {e}")

        json_path = os.path.join(settings.REPORTS_DIR, "merge_report.json")
        if os.path.exists(json_path):
            with open(json_path) as f:
                return [json.load(f)]
        return []


database_service = DatabaseService()
