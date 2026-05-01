from __future__ import annotations

import json
import os
import threading
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from catalog import load_catalog, option_payload, spec_payload
from db_dataset import load_raw_dataset, load_and_prepare_dataset
from github_mirror import github_mirror_enabled, mirror_prediction
from predict import predict_record
from prediction_store import init_prediction_store, log_prediction, recent_predictions, update_github_status
from train import TrainingConfig, train_pipeline


ROOT_DIR = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT_DIR / "web"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
DB_PATH = ROOT_DIR / "usedCars.db"
TRAINING_STATUS_PATH = ARTIFACTS_DIR / "training_status.json"
CATALOG_PATH = ARTIFACTS_DIR / "catalog.json"
PREDICTION_STORE_PATH = Path(os.getenv("PREDICTION_STORE_PATH", str(ARTIFACTS_DIR / "predictions.db")))


class PredictRequest(BaseModel):
    make: str
    model: str
    variant: str
    year: int
    mileage: float
    transmission: str | None = None
    fuel_type: str | None = None
    registered_in: str
    color: str
    assembly: str | None = None
    body_type: str | None = "Unknown"
    engine_capacity_cc: float | None = None
    inspection_score: float | None = None
    section_interior_pct: float | None = None
    section_engine_transmission_clutch_pct: float | None = None
    section_electrical_electronics_pct: float | None = None
    section_body_frame_accident_pct: float | None = None
    section_exterior_body_pct: float | None = None
    section_ac_heater_pct: float | None = None
    section_brakes_pct: float | None = None
    section_suspension_steering_pct: float | None = None
    section_tyres_pct: float | None = None


class TrainingRequest(BaseModel):
    db_path: str = Field(default=str(DB_PATH))
    table: str = Field(default="car_listings_old")
    model_dir: str = Field(default=str(ARTIFACTS_DIR))
    validation_date_fraction: float = Field(default=0.15, ge=0.05, le=0.4)
    random_seed: int = 42
    validation_comp_sample: int = Field(default=1200, ge=200, le=5000)


app = FastAPI(title="Used Car Price Predictor")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")
init_prediction_store(PREDICTION_STORE_PATH)


_training_lock = threading.Lock()
_training_thread: threading.Thread | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_training_status(payload: dict) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(TRAINING_STATUS_PATH, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def read_training_status() -> dict:
    if not TRAINING_STATUS_PATH.exists():
        return {
            "status": "idle",
            "updated_at": utc_now(),
        }
    with open(TRAINING_STATUS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def load_json_file(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


@lru_cache(maxsize=1)
def load_catalog_artifact() -> dict | None:
    if not CATALOG_PATH.exists():
        return None
    return load_catalog(CATALOG_PATH)


@lru_cache(maxsize=2)
def load_catalog_frame(db_path: str, table: str) -> pd.DataFrame:
    frame = load_and_prepare_dataset(db_path, table=table).copy()
    frame["year"] = frame["year"].astype(int)
    frame["engine_capacity_cc"] = pd.to_numeric(frame["engine_capacity_cc"], errors="coerce")
    return frame


def values_for(frame: pd.DataFrame, column: str, sort_numeric: bool = False) -> list:
    series = frame[column].dropna()
    if sort_numeric:
        return sorted({int(value) for value in series.tolist()})
    return sorted({str(value) for value in series.astype(str).tolist() if str(value).strip()})


def mode_or_unknown(frame: pd.DataFrame, column: str) -> str | None:
    series = frame[column].dropna().astype(str)
    if series.empty:
        return None
    return series.mode().iloc[0]


def median_or_none(frame: pd.DataFrame, column: str) -> int | None:
    series = pd.to_numeric(frame[column], errors="coerce").dropna()
    if series.empty:
        return None
    return int(round(float(series.median())))


def enrich_request_from_catalog(request: PredictRequest) -> dict:
    payload = request.model_dump(exclude_none=True)
    catalog = load_catalog_artifact()
    if catalog is None:
        return payload

    spec = spec_payload(
        catalog,
        make=request.make,
        model=request.model,
        year=request.year,
        variant=request.variant,
    )
    if spec is None:
        return payload

    spec_fields = spec.get("spec", {})
    payload["catalog_source_rows"] = int(spec.get("source_rows", 0))
    payload["spec_locked_from_catalog"] = True
    for field in ["transmission", "fuel_type", "assembly", "body_type", "engine_capacity_cc"]:
        value = spec_fields.get(field)
        if value not in [None, ""]:
            payload[field] = value
    return payload


@lru_cache(maxsize=2)
def compute_data_overview(db_path: str, table: str) -> dict:
    raw = load_raw_dataset(db_path, table=table)
    cleaned = load_and_prepare_dataset(db_path, table=table)
    raw_rows = int(len(raw))
    cleaned_rows = int(len(cleaned))

    missing_counts = {}
    for column in [
        "make",
        "model",
        "variant",
        "year",
        "price",
        "mileage",
        "vehicle_transmission",
        "fuel_type",
        "registered_in",
        "color",
        "assembly",
        "rating",
    ]:
        series = raw[column]
        if series.dtype == object:
            missing_counts[column] = int(series.isna().sum() + series.fillna("").astype(str).str.strip().eq("").sum())
        else:
            missing_counts[column] = int(series.isna().sum())

    top_makes = cleaned["make"].value_counts().head(12).to_dict()
    top_models = cleaned["model"].value_counts().head(12).to_dict()
    top_assemblies = cleaned["assembly"].value_counts().to_dict()
    top_transmissions = cleaned["transmission"].value_counts().to_dict()

    return {
        "raw_rows": raw_rows,
        "cleaned_rows": cleaned_rows,
        "inspection_rows": int(cleaned["has_inspection_score"].sum()),
        "proxy_rows": int(cleaned["has_proxy_signal"].sum()),
        "missing_counts": missing_counts,
        "top_makes": top_makes,
        "top_models": top_models,
        "top_assemblies": top_assemblies,
        "top_transmissions": top_transmissions,
        "median_price": float(cleaned["price"].median()),
        "median_year": float(cleaned["year"].median()),
        "median_mileage": float(cleaned["mileage"].median()),
    }


def run_training_job(config: TrainingConfig) -> None:
    try:
        write_training_status(
            {
                "status": "running",
                "started_at": utc_now(),
                "updated_at": utc_now(),
                "config": asdict(config),
            }
        )
        metrics = train_pipeline(config)
        compute_data_overview.cache_clear()
        load_catalog_frame.cache_clear()
        load_catalog_artifact.cache_clear()
        write_training_status(
            {
                "status": "completed",
                "started_at": read_training_status().get("started_at"),
                "completed_at": utc_now(),
                "updated_at": utc_now(),
                "config": asdict(config),
                "metrics": metrics,
            }
        )
    except Exception as exc:
        write_training_status(
            {
                "status": "failed",
                "updated_at": utc_now(),
                "config": asdict(config),
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )


@app.get("/")
def root() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/favicon.ico")
def favicon() -> Response:
    return Response(status_code=204)


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "db_exists": DB_PATH.exists(),
        "artifacts_exists": ARTIFACTS_DIR.exists(),
        "catalog_exists": CATALOG_PATH.exists(),
        "prediction_store_exists": PREDICTION_STORE_PATH.exists(),
        "github_mirror_enabled": github_mirror_enabled(),
        "timestamp": utc_now(),
    }


@app.get("/api/summary")
def summary() -> dict:
    return {
        "metrics": load_json_file(ARTIFACTS_DIR / "metrics.json"),
        "metadata": load_json_file(ARTIFACTS_DIR / "metadata.json"),
        "calibration": load_json_file(ARTIFACTS_DIR / "calibration.json"),
        "data_profile": load_json_file(ARTIFACTS_DIR / "data_profile.json"),
        "training_status": read_training_status(),
    }


@app.get("/api/feature-importance")
def feature_importance() -> list[dict]:
    path = ARTIFACTS_DIR / "feature_importance.csv"
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    return frame.to_dict(orient="records")


@app.get("/api/data-overview")
def data_overview(db_path: str = str(DB_PATH), table: str = "car_listings_old") -> dict:
    return compute_data_overview(db_path, table)


@app.get("/api/catalog/options")
def catalog_options(
    db_path: str = str(DB_PATH),
    table: str = "car_listings_old",
    make: str | None = None,
    model: str | None = None,
    year: int | None = None,
    variant: str | None = None,
) -> dict:
    catalog = load_catalog_artifact()
    if catalog is not None:
        return option_payload(catalog, make=make, model=model, year=year, variant=variant)

    frame = load_catalog_frame(db_path, table)

    model_frame = frame[frame["make"] == make] if make else frame.iloc[0:0]
    year_frame = model_frame[model_frame["model"] == model] if make and model else frame.iloc[0:0]
    variant_frame = year_frame[year_frame["year"] == year] if make and model and year is not None else frame.iloc[0:0]
    spec_frame = (
        variant_frame[variant_frame["variant"] == variant]
        if make and model and year is not None and variant
        else frame.iloc[0:0]
    )

    location_frame = spec_frame if not spec_frame.empty else (
        variant_frame if not variant_frame.empty else (
            year_frame if not year_frame.empty else (
                model_frame if not model_frame.empty else frame
            )
        )
    )

    return {
        "makes": values_for(frame, "make"),
        "models": values_for(model_frame, "model"),
        "years": values_for(year_frame, "year", sort_numeric=True),
        "variants": values_for(variant_frame, "variant"),
        "registered_in": values_for(location_frame, "registered_in"),
        "color": values_for(location_frame, "color"),
        "transmission": values_for(frame, "transmission"),
        "fuel_type": values_for(frame, "fuel_type"),
        "assembly": values_for(frame, "assembly"),
        "body_type": values_for(frame, "body_type"),
    }


@app.get("/api/catalog/spec")
def catalog_spec(
    make: str,
    model: str,
    year: int,
    variant: str,
    db_path: str = str(DB_PATH),
    table: str = "car_listings_old",
) -> dict:
    catalog = load_catalog_artifact()
    if catalog is not None:
        payload = spec_payload(catalog, make=make, model=model, year=year, variant=variant)
        if payload is None:
            raise HTTPException(status_code=404, detail="No exact catalog match found for that vehicle selection.")
        return payload

    frame = load_catalog_frame(db_path, table)
    matched = frame[
        (frame["make"] == make)
        & (frame["model"] == model)
        & (frame["year"] == year)
        & (frame["variant"] == variant)
    ].copy()

    if matched.empty:
        raise HTTPException(status_code=404, detail="No exact catalog match found for that vehicle selection.")

    return {
        "spec": {
            "transmission": mode_or_unknown(matched, "transmission"),
            "fuel_type": mode_or_unknown(matched, "fuel_type"),
            "assembly": mode_or_unknown(matched, "assembly"),
            "body_type": mode_or_unknown(matched, "body_type"),
            "engine_capacity_cc": median_or_none(matched, "engine_capacity_cc"),
        },
        "available_registered_in": values_for(matched, "registered_in"),
        "available_colors": values_for(matched, "color"),
        "source_rows": int(len(matched)),
    }


@app.get("/api/train/status")
def training_status() -> dict:
    return read_training_status()


@app.post("/api/train/start")
def training_start(request: TrainingRequest) -> dict:
    global _training_thread
    with _training_lock:
        if _training_thread is not None and _training_thread.is_alive():
            raise HTTPException(status_code=409, detail="Training job is already running.")

        config = TrainingConfig(
            db_path=request.db_path,
            table=request.table,
            model_dir=request.model_dir,
            validation_date_fraction=request.validation_date_fraction,
            random_seed=request.random_seed,
            validation_comp_sample=request.validation_comp_sample,
        )
        _training_thread = threading.Thread(target=run_training_job, args=(config,), daemon=True)
        _training_thread.start()
        return {
            "started": True,
            "status": "running",
            "config": asdict(config),
        }


@app.post("/api/predict")
def api_predict(request: PredictRequest) -> dict:
    try:
        input_payload = enrich_request_from_catalog(request)
        result = predict_record(input_payload, model_dir=ARTIFACTS_DIR)
        result["logged_to_github"] = False

        prediction_id = log_prediction(
            PREDICTION_STORE_PATH,
            source="api",
            input_payload=input_payload,
            output_payload=result,
        )
        result["prediction_id"] = prediction_id

        github_detail = None
        github_status = "disabled"
        if github_mirror_enabled():
            github_payload = {
                "prediction_id": prediction_id,
                "created_at": utc_now(),
                "input": input_payload,
                "output": result,
            }
            try:
                mirror_result = mirror_prediction(github_payload)
                github_status = "mirrored" if mirror_result.get("mirrored") else "disabled"
                github_detail = mirror_result.get("path") or mirror_result.get("detail")
                result["logged_to_github"] = bool(mirror_result.get("mirrored"))
            except Exception as mirror_exc:
                github_status = "failed"
                github_detail = str(mirror_exc)
        if github_status != "disabled":
            update_github_status(PREDICTION_STORE_PATH, prediction_id, github_status, github_detail)

        return result
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/predictions/recent")
def predictions_recent(limit: int = 20) -> list[dict]:
    return recent_predictions(PREDICTION_STORE_PATH, limit=max(1, min(limit, 200)))
