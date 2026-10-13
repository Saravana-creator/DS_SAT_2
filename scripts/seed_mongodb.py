"""
VantageGro Retail Stock Insight - Stage 7 MongoDB Seeding CLI Runner
Connects to local MongoDB, builds indexed collections, and seeds all pipeline artifacts:
- Ingestion Logs (Stage 1)
- Data Quality & Cleaning Reports (Stage 2)
- Exploratory Data Analysis Summaries (Stage 3)
- Machine Learning Model Registry & Metrics (Stage 5)
- Inventory Recommendations & Risk Classifications (Stage 6)
- Demand Forecast Trajectories (Stage 6)
"""

import sys
import os
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.db.mongodb import db_manager
from backend.app.services.database_service import database_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("seed_mongodb_cli")


def main():
    print("\n" + "=" * 75)
    print("VANTAGEGRO RETAIL STOCK INSIGHT - STAGE 7 MONGODB INTEGRATION & SEEDING")
    print("=" * 75)

    # 1. Health check
    health = db_manager.check_health()
    print(f"\n[1] MongoDB Connectivity Check:")
    print(f"    * Status:             {health['status'].upper()}")
    print(f"    * URI:                {health['uri']}")
    print(f"    * Database Name:      {health['database']}")
    print(f"    * MongoDB Version:    {health.get('mongodb_version', 'N/A')}")
    print(f"    * Connected:          {health['connected']}")

    if not health["connected"]:
        print(f"\n[WARNING] Local MongoDB is currently unreachable or offline.")
        print(f"Error detail: {health.get('error')}")
        print("Backend is configured with graceful fallback to disk caches.")
        sys.exit(1)

    # 2. Seed pipeline data
    print(f"\n[2] Seeding Collections & Building Indexes...")
    seed_result = database_service.seed_pipeline_data(force=True)

    if seed_result["status"] == "success":
        print("\n" + "-" * 75)
        print("SEEDED COLLECTIONS SUMMARY:")
        print("-" * 75)
        for coll_name, count in seed_result["seeded_counts"].items():
            print(f"  * Collection '{coll_name:<26}': {count:>5} documents")

        print("\n" + "-" * 75)
        print("INDEXES CREATED:")
        print("-" * 75)
        for coll, idx_list in seed_result.get("indexes", {}).get("indexes_created", {}).items():
            print(f"  * {coll:<28}: {', '.join(idx_list)}")

        # 3. Sample query verification
        print("\n" + "-" * 75)
        print("SAMPLE QUERY VERIFICATION (Top 5 High-Priority Reorders from MongoDB):")
        print("-" * 75)
        sample_recs = database_service.get_inventory_recommendations(priority="HIGH", limit=5)
        for r in sample_recs:
            print(f"  - Store: {r['Store ID']:<5} | SKU: {r['Product ID']:<6} | Cat: {r['Category']:<12} | "
                  f"Curr: {r['Current Inventory']:>4} | Order: {r['Suggested Reorder Qty']:>4} units | Risk: {r['Risk Category']}")

        print("\n" + "=" * 75)
        print("Stage 7 MongoDB Integration & Seeding Completed Successfully!")
        print("=" * 75 + "\n")
    else:
        print(f"\n[ERROR] Seeding failed: {seed_result.get('message')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
