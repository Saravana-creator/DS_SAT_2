"""
VantageGro Retail Stock Insight - MongoDB Client & Lifecycle Management (Stage 7)
Provides:
1. Resilient connection management to local MongoDB community database.
2. Connection pooling, health probing, and server detection.
3. Automated index creation for optimized querying (Date, Store ID, Product ID, Category).
4. Graceful offline fallback mode with structured error diagnostics.
"""

import logging
from typing import Dict, Any, Optional
import pymongo
from pymongo import MongoClient, IndexModel, ASCENDING, DESCENDING
from pymongo.database import Database
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

from backend.app.core.config import settings

logger = logging.getLogger(__name__)


class MongoDBManager:
    """
    Manages MongoDB client lifecycle, database references, and indexing.
    Gracefully detects when local MongoDB is unavailable and maintains status flags.
    """

    def __init__(
        self,
        uri: str = settings.MONGODB_URI,
        db_name: str = settings.MONGODB_DB_NAME,
        timeout_ms: int = settings.MONGODB_TIMEOUT_MS
    ):
        self.uri = uri
        self.db_name = db_name
        self.timeout_ms = timeout_ms
        self._client: Optional[MongoClient] = None
        self._db: Optional[Database] = None
        self._is_connected: bool = False
        self._last_error: Optional[str] = None

    def connect(self) -> bool:
        """Attempts connection to MongoDB and validates with ping."""
        try:
            self._client = MongoClient(
                self.uri,
                serverSelectionTimeoutMS=self.timeout_ms,
                connectTimeoutMS=self.timeout_ms,
                socketTimeoutMS=self.timeout_ms
            )
            # Trigger server selection to verify connectivity
            self._client.admin.command("ping")
            self._db = self._client[self.db_name]
            self._is_connected = True
            self._last_error = None
            logger.info(f"Connected successfully to MongoDB at {self.uri} (database: '{self.db_name}')")
            return True
        except (ConnectionFailure, ServerSelectionTimeoutError, Exception) as e:
            self._is_connected = False
            self._client = None
            self._db = None
            self._last_error = str(e)
            logger.warning(
                f"MongoDB connection failed to '{self.uri}': {e}. System running in offline/filesystem-fallback mode."
            )
            return False

    def close(self):
        """Closes active MongoDB client connection."""
        if self._client:
            self._client.close()
            self._client = None
            self._db = None
            self._is_connected = False
            logger.info("MongoDB connection closed.")

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    def get_database(self) -> Optional[Database]:
        """Returns the database instance if connected, else None."""
        if not self._is_connected or self._db is None:
            self.connect()
        return self._db

    def check_health(self) -> Dict[str, Any]:
        """Returns comprehensive diagnostic health dictionary."""
        if not self._is_connected:
            # Attempt a reconnect check
            self.connect()

        if self._is_connected and self._client:
            try:
                server_info = self._client.server_info()
                db = self.get_database()
                colls = db.list_collection_names() if db is not None else []
                return {
                    "status": "healthy",
                    "connected": True,
                    "uri": self.uri,
                    "database": self.db_name,
                    "mongodb_version": server_info.get("version", "unknown"),
                    "collections": colls,
                    "collection_count": len(colls),
                    "error": None
                }
            except Exception as e:
                self._is_connected = False
                self._last_error = str(e)

        return {
            "status": "degraded",
            "connected": False,
            "uri": self.uri,
            "database": self.db_name,
            "mongodb_version": None,
            "collections": [],
            "collection_count": 0,
            "error": self._last_error or "Could not establish connection to local MongoDB."
        }

    def ensure_indexes(self) -> Dict[str, Any]:
        """
        Creates mandatory indexes across all VantageGro collections:
        1. ingestion_logs: (timestamp, run_id)
        2. quality_reports: (stage, timestamp)
        3. eda_summaries: (timestamp)
        4. model_registry: (model_name, is_champion, timestamp)
        5. demand_forecasts: (store_id, product_id), (date), (category)
        6. inventory_recommendations: (store_id, product_id), (risk_category), (review_priority), (category)
        7. sales_records: (Date, Store ID, Product ID), (Date), (Category), (Store ID)
        """
        db = self.get_database()
        if db is None:
            return {"status": "skipped", "message": "MongoDB is offline; cannot create indexes."}

        results = {}
        try:
            # 1. Ingestion Logs
            db.ingestion_logs.create_index([("timestamp", DESCENDING)])
            db.ingestion_logs.create_index([("stage", ASCENDING)])
            results["ingestion_logs"] = ["timestamp_desc", "stage_asc"]

            # 2. Quality Reports
            db.quality_reports.create_index([("stage", ASCENDING), ("timestamp", DESCENDING)])
            results["quality_reports"] = ["stage_timestamp"]

            # 3. EDA Summaries
            db.eda_summaries.create_index([("timestamp", DESCENDING)])
            results["eda_summaries"] = ["timestamp_desc"]

            # 4. Model Registry
            db.model_registry.create_index([("model_name", ASCENDING), ("timestamp", DESCENDING)])
            db.model_registry.create_index([("is_champion", ASCENDING)])
            results["model_registry"] = ["model_name_timestamp", "is_champion_asc"]

            # 5. Demand Forecasts
            db.demand_forecasts.create_index([("store_id", ASCENDING), ("product_id", ASCENDING)])
            db.demand_forecasts.create_index([("date", ASCENDING)])
            db.demand_forecasts.create_index([("category", ASCENDING)])
            results["demand_forecasts"] = ["store_product", "date_asc", "category_asc"]

            # 6. Inventory Recommendations
            db.inventory_recommendations.create_index([("Store ID", ASCENDING), ("Product ID", ASCENDING)], unique=True)
            db.inventory_recommendations.create_index([("Risk Category", ASCENDING)])
            db.inventory_recommendations.create_index([("Review Priority", ASCENDING)])
            db.inventory_recommendations.create_index([("Category", ASCENDING)])
            results["inventory_recommendations"] = ["store_product_unique", "risk_category", "review_priority", "category"]

            # 7. Sales Records
            db.sales_records.create_index([("Date", ASCENDING), ("Store ID", ASCENDING), ("Product ID", ASCENDING)])
            db.sales_records.create_index([("Date", ASCENDING)])
            db.sales_records.create_index([("Category", ASCENDING)])
            results["sales_records"] = ["date_store_product", "date_asc", "category_asc"]

            logger.info("Successfully established indexes across MongoDB collections.")
            return {"status": "success", "indexes_created": results}
        except Exception as e:
            logger.error(f"Error creating indexes: {e}")
            return {"status": "error", "error": str(e)}


# Global instance
db_manager = MongoDBManager()
