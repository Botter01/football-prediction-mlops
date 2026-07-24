# Premier League Match Outcome Prediction End-to-End Azure MLOps Project

An end-to-end machine learning pipeline built entirely on Microsoft Azure, predicting Premier League match outcomes (home win / draw / away win) from team form, goal difference trends, and head-to-head history. Built as a hands-on portfolio project to apply MLOps practices on Azure Machine Learning.

## Architecture

```
Data collection (football-data.co.uk)
    │
    ▼
Azure Blob Storage  ─────►  Azure ML Data Assets (versioned)
    │
    ▼
Feature Engineering (time-aware, leakage-free)
    │
    ▼
Azure ML AutoML  ─────►  MLflow experiment tracking
    │
    ▼
Model Registry  ─────►  Managed Online Endpoint (REST API)
    │
    ▼
Streamlit frontend (calls the endpoint)
    │
    ▼
Docker container  ─────►  Azure Container Registry  ─────►  Azure App Service
```

## What this project covers

- **Data ingestion**: automated download of 5+ seasons of Premier League match data, uploaded to Azure Blob Storage
- **Data versioning**: registered as Azure ML Data Assets with version history
- **Feature engineering**: rolling team form, goal difference trend, and head-to-head statistics, computed strictly from information available *before* each match (no data leakage)
- **A data leakage bug I caught and fixed**: an early version of the training data accidentally included the match result column itself, producing a suspicious 100% accuracy. I traced this back to the feature selection step, removed the leaking column, and retrained landing on a realistic ~53% accuracy (vs. a ~44.5% baseline of always predicting a home win)
- **Model training**: Azure ML AutoML classification job, comparing LightGBM, XGBoost, Logistic Regression, SVM, and ensemble methods, tracked via MLflow
- **Deployment**: best model registered in the Azure ML Model Registry and deployed as a Managed Online Endpoint (real-time REST API), tested via Postman
- **Frontend**: a Streamlit app that lets a user pick two teams, computes live features from the historical data in Blob Storage, calls the Azure endpoint, and displays the predicted outcome
- **Containerization**: packaged the app in Docker, pushed to Azure Container Registry, and deployed to Azure App Service (Web App for Containers)

## Results

| Model | Accuracy |
|---|---|
| Voting Ensemble | 53.2% |
| LightGBM | 52.0% |
| Stack Ensemble | 52.0% |
| Logistic Regression | 51.9% |
| SVM | 51.7% |
| Baseline (always predict home win) | 44.5% |

A ~53% accuracy is in line with what's realistic for football outcome prediction using only form/statistics-based features professional betting models with far richer data (injuries, lineups, odds movement) typically land in the 55-60% range.

## Tech stack

Python, pandas, scikit-learn, Azure Machine Learning (AutoML, Managed Online Endpoints, Data Assets, MLflow tracking), Azure Blob Storage, Docker, Azure Container Registry, Azure App Service, Streamlit

## Repository structure

```
notebooks/  Jupyter notebooks used in Azure ML Studio (data collection → feature engineering → training → deployment)
app/ Streamlit frontend, Dockerfile, and dependencies
screenshots/ Screenshots from Azure ML Studio (AutoML leaderboard, endpoint, etc.)
```

## Running the app locally

```bash
cd app
uv venv
uv pip install -r requirements.txt
# create a .env file with SCORING_URI, ENDPOINT_KEY, STORAGE_ACCOUNT_NAME, STORAGE_ACCOUNT_KEY, CONTAINER_NAME
uv run streamlit run football_pred_app.py
```

## Notes / future improvements

- Deployment to Azure App Service is functional end-to-end (image builds and runs correctly, verified via logs)
- Richer features (injuries, odds, squad changes) would likely push accuracy meaningfully higher

## Why I built this

Built as a hands-on project to apply Azure Machine Learning end-to-end from raw data to a deployed, callable model while job-hunting for AI/ML engineering roles in Denmark. The most valuable part wasn't the model itself, but working through the full MLOps lifecycle: data versioning, catching and fixing a real data leakage bug, experiment tracking, deployment, and containerization.
