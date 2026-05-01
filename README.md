# Used Car Price Predictor

This project is now packaged as a single deployable FastAPI service with a minimal web UI for prediction only.

It is built for one job:

- select a valid car from the trained catalog
- enter mileage, registration, color, and optional inspection
- get a predicted listing price and range
- store every prediction server-side
- optionally mirror every prediction to GitHub

## Runtime Shape

The deployed app does **not** need the raw `usedCars.db` file at runtime.

Instead it runs from:

- trained pricing artifacts in `artifacts/`
- a compact vehicle catalog in `artifacts/catalog.json`
- a minimal UI in `web/`
- a FastAPI backend in `src/api.py`

That makes the repo small enough to publish cleanly and realistic to deploy on a normal web host.

## What Is Stored

Every prediction is stored in a local SQLite file:

- `artifacts/predictions.db`

Each row records:

- timestamp
- input payload
- prediction output
- prediction mode
- predicted price and range
- GitHub mirror status

If GitHub logging env vars are configured, each prediction is also mirrored to GitHub as JSONL under:

- `prediction-logs/YYYY/MM/YYYY-MM-DD.jsonl`

Important:

- local SQLite is the primary store
- GitHub mirroring is a secondary audit trail
- GitHub is not a good primary live database for high traffic

## Files That Matter

- `src/api.py`: API, catalog endpoints, prediction endpoint, storage integration
- `src/train.py`: training pipeline and catalog generation
- `src/serving.py`: prediction logic
- `src/predict.py`: single-record predictor
- `src/catalog.py`: deploy-time catalog builder and lookup helpers
- `src/prediction_store.py`: prediction storage
- `src/github_mirror.py`: optional GitHub mirroring
- `web/index.html`: minimal predictor UI
- `web/app.js`: chained dropdowns and prediction calls
- `web/styles.css`: minimal styling
- `run_dashboard.py`: production entrypoint
- `Dockerfile`: container deployment
- `render.yaml`: Render blueprint

## Train Artifacts

Train from the raw SQLite database:

```powershell
& "C:\Users\mutee\Downloads\30 Dec 2025\CarMandi_Release_v3\backend\venv\Scripts\python.exe" src\train.py --db-path usedCars.db --table car_listings_old --model-dir artifacts
```

This writes:

- `anchor_point_model.cbm`
- `anchor_lower_model.cbm`
- `anchor_upper_model.cbm`
- `inspection_delta_model.cbm`
- `inspection_isotonic.pkl`
- `proxy_delta_model.cbm`
- `comparables.pkl`
- `catalog.json`
- `metrics.json`
- `metadata.json`
- `calibration.json`

## Run Locally

Use a Python environment that can load CatBoost. On this machine the working one was:

```powershell
& "C:\Users\mutee\Downloads\30 Dec 2025\CarMandi_Release_v3\backend\venv\Scripts\python.exe" run_dashboard.py
```

Then open:

```text
http://127.0.0.1:8000
```

## Core Endpoints

- `GET /`
- `GET /api/health`
- `GET /api/catalog/options`
- `GET /api/catalog/spec`
- `POST /api/predict`
- `GET /api/predictions/recent`

## GitHub Mirror Env Vars

Set these on the deployed service if you want every prediction mirrored to GitHub:

```text
GITHUB_LOG_TOKEN
GITHUB_LOG_REPO
GITHUB_LOG_BRANCH
GITHUB_LOG_DIR
```

Example:

```text
GITHUB_LOG_REPO=muteekhan06/used-car-price-predictor
GITHUB_LOG_BRANCH=main
GITHUB_LOG_DIR=prediction-logs
```

## Deploy

This repo is prepared for container deployment.

### Docker

```powershell
docker build -t used-car-price-predictor .
docker run -p 8000:8000 used-car-price-predictor
```

### Render

The repo includes `render.yaml` and `Dockerfile`.

Connect the GitHub repo to Render and deploy as a web service. Then set:

- `GITHUB_LOG_TOKEN`
- `GITHUB_LOG_REPO`
- `GITHUB_LOG_BRANCH`
- `GITHUB_LOG_DIR`

if you want prediction mirroring enabled.

## Current Model Notes

This model predicts **listing price**, not sold price.

It is strongest for:

- market asking-price estimation
- comparable-grounded pricing
- inspection-aware adjustment when score or sections are present

It is not yet:

- a sold-price valuation engine
- a perfect fair-market-value oracle

