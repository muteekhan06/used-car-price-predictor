from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split

from comparables import ComparableRetriever
from catalog import build_catalog, save_catalog
from db_dataset import load_and_prepare_dataset
from features import (
    ANCHOR_CATEGORICAL_COLUMNS,
    ANCHOR_FEATURE_COLUMNS,
    DELTA_CATEGORICAL_COLUMNS,
    DELTA_FEATURE_COLUMNS,
    PROXY_DELTA_CATEGORICAL_COLUMNS,
    PROXY_DELTA_FEATURE_COLUMNS,
    engineer_feature_frame,
)
from serving import predict_frame


@dataclass
class TrainingConfig:
    db_path: str = "usedCars.db"
    table: str = "car_listings_old"
    model_dir: str = "artifacts"
    validation_date_fraction: float = 0.15
    random_seed: int = 42
    validation_comp_sample: int = 1200


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the production used-car pricing pipeline.")
    parser.add_argument("--db-path", default="usedCars.db", help="Path to the SQLite database.")
    parser.add_argument("--table", default="car_listings_old", help="Source table name.")
    parser.add_argument("--model-dir", default="artifacts", help="Directory for saved artifacts.")
    parser.add_argument(
        "--validation-date-fraction",
        type=float,
        default=0.15,
        help="Fraction of unique scrape dates reserved for validation.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for training.",
    )
    parser.add_argument(
        "--validation-comp-sample",
        type=int,
        default=1200,
        help="Maximum validation rows used for end-to-end comparable-based evaluation.",
    )
    return parser.parse_args()


def build_model(loss_function: str, random_seed: int) -> CatBoostRegressor:
    return CatBoostRegressor(
        iterations=260,
        depth=6,
        learning_rate=0.08,
        loss_function=loss_function,
        eval_metric="MAE",
        random_seed=random_seed,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    )


def _fit_model(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
    target_column: str,
    loss_function: str,
    random_seed: int,
) -> CatBoostRegressor:
    model = build_model(loss_function, random_seed)
    model.fit(
        train_df[feature_columns],
        train_df[target_column],
        cat_features=[feature_columns.index(col) for col in categorical_columns],
        eval_set=(valid_df[feature_columns], valid_df[target_column]),
        use_best_model=True,
    )
    return model


def time_split(df: pd.DataFrame, validation_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    dated = df[df["ad_last_updated_date"].notna()].copy()
    undated = df[df["ad_last_updated_date"].isna()].copy()
    unique_dates = sorted(dated["ad_last_updated_date"].dt.normalize().unique())
    if len(unique_dates) < 10:
        shuffled = df.sample(frac=1.0, random_state=42)
        cutoff = int(len(shuffled) * (1.0 - validation_fraction))
        return shuffled.iloc[:cutoff].copy(), shuffled.iloc[cutoff:].copy()

    validation_dates = set(unique_dates[max(1, int(len(unique_dates) * (1.0 - validation_fraction))):])
    valid = dated[dated["ad_last_updated_date"].dt.normalize().isin(validation_dates)].copy()
    train = dated[~dated["ad_last_updated_date"].dt.normalize().isin(validation_dates)].copy()
    train = pd.concat([train, undated], ignore_index=True)
    return train.reset_index(drop=True), valid.reset_index(drop=True)


def metrics_from_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    denominator = np.maximum(np.abs(y_true), 1.0)
    ape = np.abs(y_true - y_pred) / denominator
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "median_ae": float(median_absolute_error(y_true, y_pred)),
        "mape": float(np.mean(ape)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def split_delta_subset(
    subset: pd.DataFrame,
    *,
    min_train_rows: int,
    min_valid_rows: int,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(subset) < (min_train_rows + min_valid_rows):
        return subset.iloc[0:0].copy(), subset.iloc[0:0].copy()

    dated = subset[subset["ad_last_updated_date"].notna()].copy()
    if len(dated) >= (min_train_rows + min_valid_rows):
        train_part, valid_part = time_split(dated, 0.2)
        if len(train_part) >= min_train_rows and len(valid_part) >= min_valid_rows:
            return train_part.reset_index(drop=True), valid_part.reset_index(drop=True)

    train_part, valid_part = train_test_split(
        subset,
        test_size=max(min_valid_rows / len(subset), 0.2),
        random_state=random_seed,
        shuffle=True,
    )
    return train_part.reset_index(drop=True), valid_part.reset_index(drop=True)


def _signed_error_calibration(validation_df: pd.DataFrame, predictions: pd.DataFrame) -> dict:
    joined = validation_df[["price", "prediction_mode"]].join(predictions[["predicted_price", "prediction_mode"]], rsuffix="_pred")
    joined["signed_relative_error"] = (joined["price"] - joined["predicted_price"]) / np.maximum(joined["predicted_price"], 1.0)

    calibration = {}
    for mode, group in joined.groupby("prediction_mode_pred"):
        if len(group) < 25:
            continue
        calibration[mode] = {
            "q10": float(group["signed_relative_error"].quantile(0.10)),
            "q90": float(group["signed_relative_error"].quantile(0.90)),
            "count": int(len(group)),
        }

    calibration["default"] = {
        "q10": float(joined["signed_relative_error"].quantile(0.10)),
        "q90": float(joined["signed_relative_error"].quantile(0.90)),
        "count": int(len(joined)),
    }
    return calibration


def main() -> None:
    args = parse_args()
    metrics = train_pipeline(
        TrainingConfig(
            db_path=args.db_path,
            table=args.table,
            model_dir=args.model_dir,
            validation_date_fraction=args.validation_date_fraction,
            random_seed=args.random_seed,
            validation_comp_sample=args.validation_comp_sample,
        )
    )
    print("Training complete.")
    print(json.dumps(metrics, indent=2))


def train_pipeline(config: TrainingConfig) -> dict:
    model_dir = Path(config.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    raw_df = load_and_prepare_dataset(config.db_path, table=config.table)
    prepared = engineer_feature_frame(raw_df)

    train_df, valid_df = time_split(prepared, config.validation_date_fraction)
    train_df["log_price"] = np.log1p(train_df["price"])
    valid_df["log_price"] = np.log1p(valid_df["price"])

    anchor_point_model = _fit_model(
        train_df,
        valid_df,
        ANCHOR_FEATURE_COLUMNS,
        ANCHOR_CATEGORICAL_COLUMNS,
        "log_price",
        "RMSE",
        config.random_seed,
    )
    anchor_lower_model = _fit_model(
        train_df,
        valid_df,
        ANCHOR_FEATURE_COLUMNS,
        ANCHOR_CATEGORICAL_COLUMNS,
        "log_price",
        "Quantile:alpha=0.10",
        config.random_seed,
    )
    anchor_upper_model = _fit_model(
        train_df,
        valid_df,
        ANCHOR_FEATURE_COLUMNS,
        ANCHOR_CATEGORICAL_COLUMNS,
        "log_price",
        "Quantile:alpha=0.90",
        config.random_seed,
    )

    train_df["anchor_log_pred"] = anchor_point_model.predict(train_df[ANCHOR_FEATURE_COLUMNS])
    valid_df["anchor_log_pred"] = anchor_point_model.predict(valid_df[ANCHOR_FEATURE_COLUMNS])

    inspection_delta_model = None
    inspection_isotonic_model = None
    inspected_subset = prepared[prepared["has_inspection_score"] == 1].copy()
    inspected_subset["log_price"] = np.log1p(inspected_subset["price"])
    inspected_subset["anchor_log_pred"] = anchor_point_model.predict(inspected_subset[ANCHOR_FEATURE_COLUMNS])
    if len(inspected_subset) >= 150:
        inspection_isotonic_model = IsotonicRegression(increasing=True, out_of_bounds="clip")
        inspection_isotonic_model.fit(
            inspected_subset["inspection_score"].to_numpy(dtype=float),
            (inspected_subset["log_price"] - inspected_subset["anchor_log_pred"]).to_numpy(dtype=float),
        )
    inspection_train, inspection_valid = split_delta_subset(
        inspected_subset,
        min_train_rows=250,
        min_valid_rows=50,
        random_seed=config.random_seed,
    )
    if len(inspection_train) >= 250 and len(inspection_valid) >= 50:
        inspection_train["delta_target"] = inspection_train["log_price"] - inspection_train["anchor_log_pred"]
        inspection_valid["delta_target"] = inspection_valid["log_price"] - inspection_valid["anchor_log_pred"]
        inspection_delta_model = _fit_model(
            inspection_train,
            inspection_valid,
            DELTA_FEATURE_COLUMNS,
            DELTA_CATEGORICAL_COLUMNS,
            "delta_target",
            "RMSE",
            config.random_seed,
        )

    proxy_delta_model = None
    proxy_subset = prepared[(prepared["has_inspection_score"] == 0) & (prepared["has_proxy_signal"] == 1)].copy()
    proxy_subset["log_price"] = np.log1p(proxy_subset["price"])
    proxy_subset["anchor_log_pred"] = anchor_point_model.predict(proxy_subset[ANCHOR_FEATURE_COLUMNS])
    proxy_train, proxy_valid = split_delta_subset(
        proxy_subset,
        min_train_rows=500,
        min_valid_rows=100,
        random_seed=config.random_seed,
    )
    if len(proxy_train) >= 500 and len(proxy_valid) >= 100:
        proxy_train["delta_target"] = proxy_train["log_price"] - proxy_train["anchor_log_pred"]
        proxy_valid["delta_target"] = proxy_valid["log_price"] - proxy_valid["anchor_log_pred"]
        proxy_delta_model = _fit_model(
            proxy_train,
            proxy_valid,
            PROXY_DELTA_FEATURE_COLUMNS,
            PROXY_DELTA_CATEGORICAL_COLUMNS,
            "delta_target",
            "RMSE",
            config.random_seed,
        )

    retriever = ComparableRetriever(train_df)

    artifact_bundle = {
        "anchor_point_model": anchor_point_model,
        "anchor_lower_model": anchor_lower_model,
        "anchor_upper_model": anchor_upper_model,
        "inspection_delta_model": inspection_delta_model,
        "proxy_delta_model": proxy_delta_model,
        "comparables": retriever,
        "calibration": {},
    }

    validation_eval = valid_df.copy()
    if len(validation_eval) > config.validation_comp_sample:
        validation_eval = validation_eval.sample(config.validation_comp_sample, random_state=config.random_seed).copy()
    validation_eval = validation_eval.reset_index(drop=True)
    prediction_df = predict_frame(validation_eval, artifact_bundle)
    calibration = _signed_error_calibration(validation_eval, prediction_df)
    artifact_bundle["calibration"] = calibration
    prediction_df = predict_frame(validation_eval, artifact_bundle)

    overall_metrics = metrics_from_predictions(
        validation_eval["price"].to_numpy(dtype=float),
        prediction_df["predicted_price"].to_numpy(dtype=float),
    )
    overall_metrics["rows_used"] = int(len(prepared))
    overall_metrics["train_rows"] = int(len(train_df))
    overall_metrics["validation_rows"] = int(len(valid_df))
    overall_metrics["validation_eval_rows"] = int(len(validation_eval))
    overall_metrics["inspection_rows"] = int(prepared["has_inspection_score"].sum())
    overall_metrics["proxy_rows"] = int(prepared["has_proxy_signal"].sum())
    overall_metrics["prediction_modes"] = prediction_df["prediction_mode"].value_counts().to_dict()
    overall_metrics["avg_range_width"] = float(
        np.mean(prediction_df["price_range_high"] - prediction_df["price_range_low"])
    )
    overall_metrics["range_coverage"] = float(
        np.mean(
            (validation_eval["price"] >= prediction_df["price_range_low"])
            & (validation_eval["price"] <= prediction_df["price_range_high"])
        )
    )

    feature_importance = anchor_point_model.get_feature_importance(prettified=True)
    feature_importance = feature_importance.rename(
        columns={"Feature Id": "feature", "Importances": "importance"}
    )
    feature_importance.to_csv(model_dir / "feature_importance.csv", index=False)

    anchor_point_model.save_model(str(model_dir / "anchor_point_model.cbm"))
    anchor_lower_model.save_model(str(model_dir / "anchor_lower_model.cbm"))
    anchor_upper_model.save_model(str(model_dir / "anchor_upper_model.cbm"))
    if inspection_delta_model is not None:
        inspection_delta_model.save_model(str(model_dir / "inspection_delta_model.cbm"))
    if inspection_isotonic_model is not None:
        with open(model_dir / "inspection_isotonic.pkl", "wb") as file:
            pickle.dump(inspection_isotonic_model, file)
    if proxy_delta_model is not None:
        proxy_delta_model.save_model(str(model_dir / "proxy_delta_model.cbm"))

    retriever.save(model_dir / "comparables.pkl")

    metadata = {
        "db_path": str(config.db_path),
        "table": config.table,
        "anchor_feature_columns": ANCHOR_FEATURE_COLUMNS,
        "delta_feature_columns": DELTA_FEATURE_COLUMNS,
        "proxy_delta_feature_columns": PROXY_DELTA_FEATURE_COLUMNS,
        "has_inspection_delta_model": inspection_delta_model is not None,
        "has_inspection_isotonic_model": inspection_isotonic_model is not None,
        "has_proxy_delta_model": proxy_delta_model is not None,
    }
    with open(model_dir / "metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    with open(model_dir / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(overall_metrics, file, indent=2)
    with open(model_dir / "calibration.json", "w", encoding="utf-8") as file:
        json.dump(calibration, file, indent=2)

    profile = {
        "rows_after_cleaning": int(len(prepared)),
        "rows_with_inspection": int(prepared["has_inspection_score"].sum()),
        "rows_with_proxy_signals": int(prepared["has_proxy_signal"].sum()),
        "median_price": float(prepared["price"].median()),
        "median_year": float(prepared["year"].median()),
        "median_mileage": float(prepared["mileage"].median()),
    }
    with open(model_dir / "data_profile.json", "w", encoding="utf-8") as file:
        json.dump(profile, file, indent=2)

    save_catalog(build_catalog(prepared), model_dir / "catalog.json")

    prediction_df.join(validation_eval[["price"]]).to_csv(
        model_dir / "validation_predictions_sample.csv",
        index=False,
    )

    return overall_metrics


if __name__ == "__main__":
    main()
