# Premier League Match Outcome Prediction (End-to-End Azure MLOps Project)

An end-to-end machine learning pipeline built entirely on Microsoft Azure, predicting Premier League match outcomes (home win / draw / away win) from team form, goal difference trends, and head-to-head history. Beyond training and deploying a model, this project covers what happens *after* deployment: a validation-gated CI/CD pipeline that guards against shipping a worse model, and a drift-monitoring layer that watches whether live input data still resembles what the model was trained on.

## Architecture

```
Data collection (football-data.co.uk)
        │
        ▼
Azure Blob Storage  ──────────►  Azure ML Data Assets (versioned)
        │
        ▼
Feature Engineering (time-aware, leakage-free)
        │
        ▼
Azure ML AutoML  ──────────►  MLflow experiment tracking
        │
        ▼
Model Registry  ──────────►  Managed Online Endpoint (REST API)
        │
        ▼
Streamlit frontend (calls the endpoint)
        │
        ▼
Docker container  ──────────►  Azure Container Registry  ──────────►  Azure App Service


        ┌─────────────────────────────────────────────────────┐
        │              After deployment                       │
        └─────────────────────────────────────────────────────┘

GitHub push (code or data change)
        │
        ▼
GitHub Actions ──► Azure ML retraining job ──► validation gate vs. baseline accuracy
        │                                               │
        │                                    fails ─────┴───── passes
        │                                      │                │
        │                                 job fails,       registered as new
        │                                 nothing deploys   model version
        ▼
(separate, manually-run) Drift monitoring
        │
        ▼
Simulated traffic → Evidently drift analysis → PostgreSQL → Grafana dashboard
```

## What this project covers

### Core ML pipeline
- **Data ingestion**: automated download of 5+ seasons of Premier League match data, uploaded to Azure Blob Storage
- **Data versioning**: registered as Azure ML Data Assets with version history
- **Feature engineering**: rolling team form, goal difference trend, and head-to-head statistics, computed strictly from information available *before* each match (no data leakage)
- **A data leakage bug I caught and fixed**: an early version of the training data accidentally included the match result column itself, producing a suspicious 100% accuracy. I traced this back to the feature selection step, removed the leaking column, and retrained  landing on a realistic ~53% accuracy (vs. a ~44.5% baseline of always predicting a home win)
- **Model training**: Azure ML AutoML classification job, comparing LightGBM, XGBoost, Logistic Regression, SVM, and ensemble methods, tracked via MLflow
- **Deployment**: best model registered in the Azure ML Model Registry and deployed as a Managed Online Endpoint (real-time REST API), tested via Postman
- **Frontend**: a Streamlit app that lets a user pick two teams, computes live features from the historical data in Blob Storage, calls the Azure endpoint, and displays the predicted outcome
- **Containerization**: packaged the app in Docker, pushed to Azure Container Registry, and deployed to Azure App Service (Web App for Containers)

### CI/CD: gated automatic retraining
- A standalone `train.py` script (parameterized via `argparse`, not hardcoded) runs as an Azure ML Job, triggered automatically by a **GitHub Actions** workflow whenever `scripts/train.py` or the training data changes
- The script trains a new model and compares its accuracy against a fixed baseline. If the new model is worse, the script raises an error, the Azure ML Job fails, and GitHub Actions reports the pipeline as failed  **the model is never registered or deployed**
- If the new model matches or beats the baseline, it's saved and ready to be promoted to the Model Registry
- Authentication from GitHub Actions to Azure runs through a scoped Service Principal (least-privilege, limited to the project's resource group) rather than a personal login, and compute runs on an auto-scaling Azure ML **Compute Cluster** (not a personal Compute Instance)  this distinction matters because Compute Instances are tied to an individual user and can't be triggered by an automated identity
- Verified both directions of the gate in practice: a model trained with default hyperparameters that scored below baseline was correctly rejected (job failed as designed), and after adjusting the baseline threshold to reflect a realistic target, an equivalent model passed and completed successfully

### Monitoring: input drift detection
- Since there's no real production traffic hitting this model, the monitoring setup uses two simulated traffic sets: one resampled directly from the training data (should show *no* drift), and one with specific features (`home_form`, `away_form`, `home_goal_diff_trend`) deliberately shifted (should show drift *only* in those features)
- **Evidently** computes per-feature drift scores (Wasserstein distance) between each simulated batch and the original training distribution
- Results are written to **PostgreSQL** (running in Docker) with a timestamp per check, and visualized in a **Grafana** dashboard (also Docker) showing drift score per feature over time, with a threshold line marking the drift cutoff
- Confirmed the mechanism works both ways: the "normal" batch showed no dataset-level drift, and the "shifted" batch correctly flagged exactly the three deliberately-shifted features, leaving the untouched ones below threshold

**Important scoping note:** the monitoring pipeline currently runs on manual invocation (each point on the dashboard corresponds to a test run I triggered by hand), not a scheduled job  in a real deployment this script would run on a schedule (e.g. daily via cron or Task Scheduler). It's also intentionally **not** wired into the CI/CD pipeline: a drift alert here is a signal for human review, not an automatic retraining trigger, since a shift in input distribution doesn't necessarily mean the model's predictions have gotten worse  that can only be confirmed against actual match outcomes once they're known.

## Results

| Model | Accuracy |
|---|---|
| Voting Ensemble | 53.2% |
| LightGBM | 52.0% |
| Stack Ensemble | 52.0% |
| Logistic Regression | 51.9% |
| SVM | 51.7% |
| Baseline (always predict home win) | 44.5% |

A ~53% accuracy is in line with what's realistic for football outcome prediction using only form/statistics-based features  professional betting models with far richer data (injuries, lineups, odds movement) typically land in the 55-60% range.

## Tech stack

Python, pandas, scikit-learn, Azure Machine Learning (AutoML, Managed Online Endpoints, Data Assets, Compute Clusters, MLflow tracking), Azure Blob Storage, Docker, Azure Container Registry, Azure App Service, Streamlit, GitHub Actions, Evidently, PostgreSQL, Grafana

## Repository structure

```
notebooks/       Jupyter notebooks used in Azure ML Studio (data collection → feature engineering → training → deployment)
app/              Streamlit frontend, Dockerfile, and dependencies
scripts/          Standalone training script (train.py) and Azure ML job definition (job.yml) used by the CI/CD pipeline
monitoring/       Drift detection script, Docker Compose (PostgreSQL + Grafana)
.github/workflows/ GitHub Actions workflow that triggers retraining
docs/             Screenshots from Azure ML Studio (AutoML leaderboard, endpoint, CI/CD runs, Grafana dashboard, etc.)
```

## Running the app locally

```bash
cd app
uv venv
uv pip install -r requirements.txt
# create a .env file with SCORING_URI, ENDPOINT_KEY, STORAGE_ACCOUNT_NAME, STORAGE_ACCOUNT_KEY, CONTAINER_NAME
uv run streamlit run football_pred_app.py
```

## Running the drift monitor locally

```bash
cd monitoring
docker-compose up -d       # starts PostgreSQL + Grafana
uv venv
uv add pandas evidently psycopg2-binary
uv run python drift_monitor.py
```

Grafana is then available at `http://localhost:3000` (default login `admin`/`admin`); add PostgreSQL (`localhost:5432`, db `drift_monitoring`) as a data source to view the dashboard.

## Notes / future improvements

- Deployment to Azure App Service is functional end-to-end (image builds and runs correctly, verified via logs), but the app is currently hard to reach through the public gateway  likely a health-check/routing configuration issue on the App Service side that I'm still tracking down
- The feature-computation logic currently lives inside the Streamlit app; a cleaner architecture would move it into a separate Azure Function, decoupling the feature pipeline from the frontend
- The drift monitor currently runs on-demand; scheduling it (cron/Task Scheduler, or an Azure ML pipeline) would make it a genuine continuous-monitoring system rather than a demonstration of the mechanism
- Once enough real match outcomes accumulate post-deployment, the monitoring layer could be extended to track actual prediction accuracy over time (concept/label drift), not just input feature drift  this would let a drift alert be cross-checked against real performance before deciding whether retraining is actually warranted
- Richer features (injuries, odds, squad changes) would likely push accuracy meaningfully higher

## Why I built this

Built as a hands-on project to apply Azure Machine Learning end-to-end  from raw data to a deployed, callable model  while job-hunting for AI/ML engineering roles in Denmark. The initial version covered training and deployment; I extended it afterward specifically to address what a lot of portfolio ML projects skip: what happens to a model after it ships. The CI/CD gate came from wanting a concrete answer to "how do you stop a worse model from being deployed automatically," and the drift monitor came from wanting to demonstrate  and be able to explain clearly  the difference between "the input data changed" and "the model got worse," which are not the same thing and shouldn't be treated as if they were.
