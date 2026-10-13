# VantageGro Retail Stock Insight — Project Status

## Project Overview
End-to-end Data Science and Machine Learning web application for **VantageGro Retail Pvt. Ltd.** to forecast retail inventory demand, mitigate stockout and overstock risks, optimize automated replenishment, and deliver an interactive full-stack analytics platform.

---

## Current Roadmap & Stage Status

| Stage | Description | Status | Verification & Artifacts |
|---|---|---|---|
| **Stage 1** | **Dataset Inspection & Merging** | **COMPLETED** | Tested (10/10 tests passed), `vantagegro_merged_raw.csv`, `merge_report.json`, 6 visualizations, `01_dataset_merging.ipynb` |
| **Stage 2** | **Data Quality Assessment & Preprocessing** | **COMPLETED** | Tested (8/8 tests passed), `vantagegro_cleaned.csv`, `vantagegro_cleaned.xlsx`, `preprocessing_report.json`, 9 visualizations, `02_preprocessing_eda.ipynb` |
| **Stage 3** | **Exploratory Data Analysis (EDA)** | **COMPLETED** | Tested (7/7 tests passed), `eda_summary.json`, 12 visualizations (HTML, JSON, PNG), `03_exploratory_data_analysis.ipynb` |
| **Stage 4** | **Feature Engineering** | **COMPLETED** | Tested (5/5 tests passed), `vantagegro_features.csv`, `feature_metadata.json`, 5 visualizations (HTML, JSON, PNG), `04_feature_engineering.ipynb` |
| **Stage 5** | **Machine Learning Model Training** | **COMPLETED** | Tested (2/2 tests passed), `models/vantagegro_best_model.joblib`, `training_report.json`, 7 diagnostic charts, `05_model_training.ipynb` |
| **Stage 6** | **Demand Forecasting & Inventory Optimization** | **COMPLETED** | Tested (7/7 tests passed), `inventory_recommendations.csv`, `forecast_and_inventory_report.json`, 8 charts, `06_forecasting_inventory.ipynb` |
| **Stage 7** | **MongoDB Local Database Integration** | **COMPLETED** | Tested (8/8 tests passed), indexed collections, seed runner, fallback handling |
| **Stage 8** | FastAPI Backend | Pending | Next Stage |
| **Stage 9** | React Frontend Dashboard | Pending | Scheduled |
| **Stage 10** | End-to-End Validation & Documentation | Pending | Scheduled |

* **Completed Stages:** 7 / 10 (70%)
* **Remaining Stages:** 3 / 10 (Stages 8 through 10)

---

## Stage Summaries

### Stage 1 — Dataset Ingestion, Inspection & Merging (Completed)
- Ingested Primary Inventory dataset (`sales_data.csv`, 76,000 rows × 16 columns) and external time-series context (`train.csv`, `oil.csv`, `holidays_events.csv`, `transactions.csv`).
- Strictly preserved join boundaries (never joined across unmapped entities). Identified 4-year chronological gap (0% overlap) and retained clean primary dataset without leaking 100% NaN external features.

### Stage 2 — Data Quality Assessment and Preprocessing (Completed)
- Validated zero missing values, zero duplicates, and zero boundary errors.
- Discovered critical stockout censoring: sales are bounded by inventory (`Units Sold <= Inventory Level`); customer demand exceeds sales in 70.3% of observations.
- Preserved heavy-tailed retail demand/ordering volume spikes as authentic business events.
- Exported cleaned dataset to both CSV (`data/processed/vantagegro_cleaned.csv`) and Excel spreadsheet (`data/processed/vantagegro_cleaned.xlsx`).

### Stage 3 — Exploratory Data Analysis (Completed)
- Computed category demand rankings: Groceries is the top seller, followed by Electronics and Clothing.
- Store & regional performance: Store `S002` (South) exhibits the highest demand, followed by `S001` (North).
- Quantified inventory risk: 13.68% stockout risk days (demand > stock) and 41.97% overstock risk days (stock > 3x demand).
- Evaluated promotional lift: Active promotions increase average daily demand from 88.5 to 136.5 units (+54.2% uplift).
- Computed SKU volatility (Coefficient of Variation): Segmented products into High Volatility, Moderate Volatility, and Stable Demand.
- Generated 12 interactive Plotly HTML & JSON charts and static PNGs in `data/reports/eda_visualizations/`.

### Stage 4 — Feature Engineering (Completed)
- Created 50 features across 76,000 records:
  * 13 Calendar features (year, month, day, dayofweek, weekend, quarter, cyclical sine/cosine).
  * 4 Autoregressive Demand Lags (`demand_lag_1`, `7`, `14`, `28`) calculated strictly per series.
  * 6 Rolling Statistical features (`rolling_mean_7`, `14`, `28`, `rolling_std_7`, `14`, `recent_demand_trend`) strictly shifted to prevent lookahead target leakage.
  * 8 Business Pricing & Competition features (`price_ratio`, `price_diff`, `discounted_price`, `Promotion`, etc.).
  * 19 One-hot encoded categorical indicators (Category, Region, Weather, Seasonality).
- Excluded target (`Demand`) and constrained sales proxy (`Units Sold`) from feature matrix.
- Saved feature dataset (`vantagegro_features.csv`, 76,000 rows × 57 columns), metadata (`feature_metadata.json`), and 5 visualizations.

### Stage 5 — Machine Learning Model Training (Completed)
- Enforced chronological splitting: Train = 53,300 rows (70%), Validation = 11,400 rows (15%), Test = 11,300 rows (15%).
- Evaluated Baselines on unseen chronological test set:
  * Naive Lag 1: Test MAE 39.57, RMSE 51.33, WAPE 39.96%, R² -0.3093
  * 7-Day Moving Average: Test MAE 31.33, RMSE 40.47, WAPE 31.63%, R² 0.1862
- Trained Supervised Regressors on unconstrained demand:
  * Ridge Regression: Test MAE 22.04, RMSE 28.53, WAPE 22.26%, R² 0.5955
  * Random Forest: Test MAE 18.96, RMSE 25.41, WAPE 19.14%, R² 0.6791
  * XGBoost Regressor: Test MAE 18.04, RMSE 24.16, WAPE 18.22%, R² 0.7098
  * **HistGradientBoosting (Champion)**: Test MAE 17.84, RMSE 23.82, WAPE 18.02%, R² 0.7179
- Achieved **-54.9% error reduction** compared to the baseline.
- Persisted champion model to `models/vantagegro_best_model.joblib`.
- Generated 7 diagnostic visualizations in `data/reports/training_visualizations/` and `notebooks/05_model_training.ipynb`.

### Stage 6 — Demand Forecasting & Inventory Optimization (Completed)
- Deployed champion `HistGradientBoostingRegressor` model to project 14-day demand across all 100 Store-Product SKUs using multi-step recursive forecasting (updating autoregressive lags dynamically without target leakage).
- Simulated promotional what-if scenarios, observing customer demand uplift patterns.
- Engineered retail inventory optimization engine with empirical formulas:
  * Lead time $L = 3$ days, review period $R = 7$ days, service level $z = 1.65$ (95.0% target).
  * Safety Stock ($SS$), Reorder Point ($ROP$), Target Stock Level ($TSL$), and Suggested Reorder Quantity.
- Segmented the 100 SKUs into operational health risk categories:
  * **Critical Stockout Risk**: 63 SKUs (63.0%) — inventory < lead time demand; immediate stockout threat.
  * **High Stockout Risk**: 2 SKUs (2.0%) — inventory below reorder point threshold.
  * **Optimal Stock Level**: 35 SKUs (35.0%) — healthy buffer.
  * **Review Priority High**: 65 SKUs (65.0%) requiring action; total immediate reorder units needed: 74,942 units.
- Generated full tabular recommendation dataset (`data/processed/inventory_recommendations.csv`).
- Produced comprehensive JSON report (`data/reports/forecast_and_inventory_report.json`).
- Rendered 8 interactive Plotly HTML & JSON charts and static PNGs in `data/reports/recommendation_visualizations/`.
- Created reproducible Jupyter Notebook (`notebooks/06_forecasting_inventory.ipynb`).

### Stage 7 — MongoDB Local Database Integration (Completed)
- Established connection to local MongoDB Community Database (`mongodb://127.0.0.1:27017/`, database: `vantagegro_stock_insight`, detected MongoDB v8.0.8).
- Designed structured collections and created indexes across key business attributes:
  * `ingestion_logs`: indexed by `timestamp` and `stage`.
  * `quality_reports`: indexed by `stage` and `timestamp`.
  * `eda_summaries`: indexed by `timestamp`.
  * `model_registry`: indexed by `model_name`, `is_champion`, and `timestamp`.
  * `demand_forecasts`: indexed by `(store_id, product_id)`, `date`, and `category`.
  * `inventory_recommendations`: indexed uniquely by `(Store ID, Product ID)`, `Risk Category`, `Review Priority`, and `Category`.
  * `sales_records`: indexed by `(Date, Store ID, Product ID)`.
- Implemented high-level `DatabaseService` with dual-mode operational support:
  * Primary: High-speed queries directly against indexed MongoDB collections.
  * Fallback: Transparent offline graceful degradation reading local filesystem CSV/JSON reports if MongoDB service is stopped.
- Created and executed seeding runner (`scripts/seed_mongodb.py`) successfully populating all 6 collections.

---

## Commands to Resume / Execute Work

```powershell
# 1. Run all unit and pipeline validation tests (47 tests across Stages 1 to 7)
python -m pytest backend/tests/ -v

# 2. Run Model Training Pipeline (Stage 5)
python scripts/train_models.py

# 3. Run Demand Forecasting & Inventory Optimization Pipeline (Stage 6)
python scripts/run_stage6.py

# 4. Seed / Synchronize Pipeline Data into Local MongoDB (Stage 7)
python scripts/seed_mongodb.py
```
