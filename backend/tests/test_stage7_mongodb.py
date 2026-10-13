"""
Unit & Integration Tests for Stage 7: MongoDB Local Database Integration
Validates:
1. Local MongoDB connectivity, ping check, and server version detection.
2. Graceful offline fallback handling when MongoDB is unreachable.
3. Automated index creation across frequently queried business keys.
4. Seeding and persistence of pipeline artifacts across collections.
5. Filtered querying (Store ID, Category, Risk Category, Review Priority).
6. Transparent fallback to disk files when database is disconnected.
"""

import pytest
from backend.app.db.mongodb import MongoDBManager, db_manager
from backend.app.services.database_service import DatabaseService, database_service


@pytest.fixture(scope="module")
def active_db_manager():
    """Returns the default configured MongoDBManager."""
    db_manager.connect()
    return db_manager


def test_mongodb_connection_and_health(active_db_manager):
    health = active_db_manager.check_health()
    assert "status" in health
    assert "uri" in health
    assert "database" in health
    assert health["database"] == "vantagegro_stock_insight"
    # When local MongoDB is running, health is 'healthy' and connected is True
    if health["connected"]:
        assert health["status"] == "healthy"
        assert health["mongodb_version"] is not None
        assert isinstance(health["collections"], list)


def test_mongodb_graceful_offline_fallback():
    """Verify that an unreachable host/port returns degraded status without crashing."""
    offline_manager = MongoDBManager(
        uri="mongodb://127.0.0.1:59999/",
        db_name="vantagegro_test_offline",
        timeout_ms=500
    )
    health = offline_manager.check_health()
    assert health["status"] == "degraded"
    assert health["connected"] is False
    assert health["error"] is not None
    assert offline_manager.get_database() is None


def test_mongodb_ensure_indexes(active_db_manager):
    if not active_db_manager.is_connected:
        pytest.skip("Local MongoDB is not running; skipping live index test.")

    res = active_db_manager.ensure_indexes()
    assert res["status"] == "success"
    assert "indexes_created" in res
    assert "inventory_recommendations" in res["indexes_created"]
    assert "model_registry" in res["indexes_created"]


def test_seed_pipeline_data(active_db_manager):
    if not active_db_manager.is_connected:
        pytest.skip("Local MongoDB is not running; skipping live seeding test.")

    seed_res = database_service.seed_pipeline_data(force=True)
    assert seed_res["status"] == "success"
    assert "seeded_counts" in seed_res
    counts = seed_res["seeded_counts"]

    assert counts.get("ingestion_logs", 0) >= 1
    assert counts.get("quality_reports", 0) >= 1
    assert counts.get("eda_summaries", 0) >= 1
    assert counts.get("model_registry", 0) >= 4
    assert counts.get("inventory_recommendations", 0) == 100
    assert counts.get("demand_forecasts", 0) >= 1


def test_query_inventory_recommendations_from_mongo(active_db_manager):
    if not active_db_manager.is_connected:
        pytest.skip("Local MongoDB is not running; skipping live query test.")

    # 1. Query all recommendations
    recs = database_service.get_inventory_recommendations(limit=100)
    assert len(recs) == 100

    # 2. Filter by Store ID
    s001_recs = database_service.get_inventory_recommendations(store_id="S001", limit=50)
    assert len(s001_recs) == 20
    for r in s001_recs:
        assert r["Store ID"] == "S001"

    # 3. Filter by Category
    groc_recs = database_service.get_inventory_recommendations(category="Groceries", limit=50)
    assert len(groc_recs) > 0
    for r in groc_recs:
        assert r["Category"] == "Groceries"

    # 4. Filter by Risk Category
    critical_recs = database_service.get_inventory_recommendations(risk_category="CRITICAL STOCKOUT RISK")
    assert len(critical_recs) == 63
    for r in critical_recs:
        assert r["Risk Category"] == "CRITICAL STOCKOUT RISK"


def test_query_model_registry(active_db_manager):
    models = database_service.get_model_registry()
    assert len(models) >= 4
    champion_found = any(m.get("is_champion") is True for m in models)
    assert champion_found, "At least one model must be marked as champion"


def test_query_reports_endpoints(active_db_manager):
    eda = database_service.get_eda_summary()
    assert "total_observations" in eda or "risk_metrics" in eda

    quality = database_service.get_quality_report()
    assert "stage" in quality or "summary" in quality

    ingestion = database_service.get_ingestion_logs()
    assert len(ingestion) >= 1


def test_filesystem_fallback_service():
    """Verify service falls back cleanly to disk when manager is intentionally offline."""
    offline_manager = MongoDBManager(
        uri="mongodb://127.0.0.1:59999/",
        db_name="vantagegro_offline",
        timeout_ms=500
    )
    fallback_svc = DatabaseService(manager=offline_manager)

    # Recommendations query should fallback to CSV without raising error
    recs = fallback_svc.get_inventory_recommendations(limit=10)
    assert len(recs) == 10
    assert "Store ID" in recs[0]

    # Models query should fallback to JSON without error
    models = fallback_svc.get_model_registry()
    assert len(models) >= 4
