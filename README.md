# 🧠 Dynamic Pricing System with Demand Forecasting

A full-stack machine learning project designed to **simulate, forecast, and optimize product prices in near real-time**, based on demand, competitor pricing, inventory, and behavioral data. Built using a **hybrid Pandas + PySpark stack**, and designed to be modular, transparent, and production-ready.

---

## 🧩 What This Project Does

This dynamic pricing engine:
- Simulates **daily sales data** across 1000 SKUs for one full year
- Builds a **demand forecasting model** using engineered features and LightGBM
- Optimizes SKU-level pricing using a **rule-based margin/demand/competition strategy**
- Provides **real-time pricing recommendations** via a FastAPI server
- Includes **post-deployment monitoring** with NannyML (CBPE + feature drift)
- Supports reproducibility via DVC and MLflow tracking
- Is fully testable and containerized (CI/CD + Docker ready)

---

## Model Selection Rationale
- LightGBM was chosen for demand forecasting due to:
 --Robust performance on tabular data
 --Support for nonlinear interactions (lag, calendar, elasticity)
 --Fast training and inference
 --Interpretability and scalability

- Rule-based pricing logic was used in this version to reflect real-world pricing guardrails:
 --Cost floor
 --Competitor cap
 --Demand-based adjustments (+5%, -5%)

In the original project, pricing involved constrained optimization (planned for future upgrade).

---

## 🔍 Project Context

This project is a **replication of an actual client engagement**, where we built a demand-driven pricing system involving:
- Merchandising and category teams
- Data scientists and ML engineers
- Ops, Pricing, and Engineering stakeholders

To preserve **client confidentiality and NDAs**, this version uses **synthetically generated data** very similar in structure and behavior to the original datasets.

Where possible, I rebuilt the core system **entirely myself** to demonstrate ownership and transparency. That said, I’ve also **collaborated with former teammates** (with their permission) to include shared utilities like:
- Dataset generation logic
- FastAPI server scaffolding
- Testing structure (`test_api.py`)
- Utility module (`helpers.py`)

---

🤝 Credits
Parts of this codebase (such as generate_data.py, helpers.py, test_api.py, and the FastAPI scaffolding) were collaboratively developed with colleagues I worked with during the original engagement. 
Their review and permission were obtained for reuse here. The rest of the system was independently built from scratch to reflect what I contributed and learned on the project.

---

## 📁 File Architecture

```bash
dynamic-pricing-system/
│
├── data/
│   ├── raw/                      # Generated product & sales data
│   ├── processed/                # Cleaned, featured, forecasted, optimized
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_eng.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_forecasting_eval.ipynb
│
├── src/
│   ├── data/
│   │   ├── generate_data.py      # Simulated daily SKU data
│   │   ├── load_data.py          # Hybrid Pandas + Spark loaders
│   │   └── preprocess.py
│   ├── features/
│   │   └── build_features.py     # Lag features, elasticity, temporal
│   ├── forecasting/
│   │   └── forecaster.py         # LightGBM demand model
│   ├── pricing/
│   │   └── pricing_engine.py     # Rule-based optimizer
│   ├── monitoring/
│   │   ├── post_deploy_monitor.py
│   │   └── drift_detection.py
│   ├── models/
│   │   └── train_model.py        # Generic LGBM training module
│   ├── pipelines/
│   │   ├── forecasting_pipeline.py
│   │   └── pricing_pipeline.py
│   ├── api/
│   │   └── fastapi_server.py     # Real-time price recommendation API
│   └── utils/
│       └── helpers.py            # Logger, config loader, summarizer
│
├── tests/                        # Full test coverage
│   ├── test_data.py
│   ├── test_preprocess.py
│   ├── test_features.py
│   ├── test_forecasting.py
│   ├── test_pricing.py
│   └── test_api.py
│
├── experiments/
│   └── tracking_with_mlflow/     # MLflow runs + model tracking
│
├── logs/
│   └── pipeline_logs.log         # Runtime logging
│
├── .github/
│   └── workflows/
│       └── ci_cd.yml             # GitHub Actions for CI/CD
│
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── README.md
├── requirements.txt
├── environment.yml
├── setup.py
├── .gitignore
└── dvc.yaml