"""
VantageGro Retail Stock Insight - Core Configuration Settings
"""

import os
from typing import Optional

class Settings:
    PROJECT_NAME: str = "VantageGro Retail Stock Insight"
    PROJECT_VERSION: str = "1.0.0"
    API_V1_PREFIX: str = "/api/v1"
    
    # MongoDB Configuration
    MONGODB_URI: str = os.getenv("MONGODB_URI", "mongodb://127.0.0.1:27017/")
    MONGODB_DB_NAME: str = os.getenv("MONGODB_DB_NAME", "vantagegro_stock_insight")
    MONGODB_TIMEOUT_MS: int = int(os.getenv("MONGODB_TIMEOUT_MS", "3000"))
    
    # Paths
    BASE_DIR: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    DATA_DIR: str = os.path.join(BASE_DIR, "data")
    PROCESSED_DATA_DIR: str = os.path.join(DATA_DIR, "processed")
    REPORTS_DIR: str = os.path.join(DATA_DIR, "reports")
    MODELS_DIR: str = os.path.join(BASE_DIR, "models")

settings = Settings()
