from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from catalog import build_catalog, save_catalog
from comparables import ComparableRetriever
from db_dataset import PreparedDatasetBundle, prepare_dataset_bundle
from features import (
    ANCHOR_ABLATIONS,
    ANCHOR_BASE_CATEGORICAL_COLUMNS,
    DEFAULT_ANCHOR_NAME,
    DELTA_CATEGORICAL_COLUMNS,
    DELTA_FEATURE_COLUMNS,
    PROXY_DELTA_CATEGORICAL_COLUMNS,
    PROXY_DELTA_FEATURE_COLUMNS,
    engineer_feature_frame,
)
from inspection import SECTION_WEIGHTAGE_VERSION
from serving import predict_frame


PRICE_BANDS = [
    ("budget", 0, 2_000_000),
    ("mid", 2_000_000, 6_000_000),
    ("premium", 6_000_000, 15_000_000),
    ("luxury", 15_000_000, float("inf")),
]


@dataclass
class TrainingConfig:
    db_path: str = "usedCars.db"
    table: str = "car_listings_old"
    model_dir: str = "artifacts"
    validation_date_fraction: float = 0.15
    random_seed: int = 42
    validation_comp_sample: int = 1200


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the hardened used-car pricing pipeline.")
    parser.add_argument("--db-path", default="usedCars.db", help="Path to the SQLite database.")
    parser.add_argument("--table", default="car_listings_old", help="Source table name.")
    parser.add_argument("--model-dir", default="artifacts", help="Directory for saved artifacts.")
    parser.add_argument("--validation-date-fraction", type=float, default=0.15)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--validation-comp-sample", type=int, default=1200)
    return parser.parse_args()


def build_model(loss_function: str, random_seed: int) -> CatBoostRegressor:
    return CatBoostRegressor(
        iterations=320,
        depth=6,
        learning_rate=0.06,
        loss_function=loss_function,
        eval_metric="MAE",
        random_seed=random_seed,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    )


def _categorical_columns_for(features: list[str]) -> list[str]:
    return [column for column in features if column in ANCHOR_BASE_CATEGORICAL_COLUMNS or column in {"variant", "body_type", "color"}]


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


def metrics_from_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    denominator = np.maximum(np.abs(y_true), 1.0)
    ape = np.abs(y_true - y_pred) / denominator
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "median_ae": float(median_absolute_error(y_true, y_pred)),
        "mape": float(np.mean(ape)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _run_anchor_ablations(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    random_seed: int,
) -> tuple[str, dict, CatBoostRegressor]:
    results = {}
    best_name = DEFAULT_ANCHOR_NAME
    best_model = None
    best_metric = float("inf")

    for name, feature_columns in ANCHOR_ABLATIONS.items():
        model = _fit_model(
            train_df,
            valid_df,
            feature_columns,
            _categorical_columns_for(feature_columns),
            "log_price",
            "RMSE",
            random_seed,
        )
        pred = np.expm1(model.predict(valid_df[feature_columns]))
        metrics = metrics_from_predictions(valid_df["price"].to_numpy(dtype=float), pred)
        results[name] = {
            "feature_columns": feature_columns,
            **metrics,
        }
        if metrics["mape"] < best_metric:
            best_metric = metrics["mape"]
            best_name = name
            best_model = model

    assert best_model is not None
    return best_name, results, best_model


def _signed_error_calibration(validation_df: pd.DataFrame, predictions: pd.DataFrame) -> dict:
    joined = validation_df[["price"]].join(predictions[["predicted_price", "prediction_mode", "support_tier"]])
    joined["signed_relative_error"] = (joined["price"] - joined["predicted_price"]) / np.maximum(joined["predicted_price"], 1.0)

    calibration = {}
    for mode, group in joined.groupby("prediction_mode"):
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


def _coverage_by(base_df: pd.DataFrame, predictions: pd.DataFrame, key: str) -> dict:
    merged = base_df[[key, "price"]].reset_index(drop=True).join(
        predictions[["price_range_low", "price_range_high"]].reset_index(drop=True)
    )
    output = {}
    for value, part in merged.groupby(key):
        coverage = np.mean((part["price"] >= part["price_range_low"]) & (part["price"] <= part["price_range_high"]))
        output[str(value)] = {
            "count": int(len(part)),
            "coverage": float(coverage),
        }
    return output


def _with_price_band(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    working["price_band"] = "luxury"
    for label, lower, upper in PRICE_BANDS:
        working.loc[(working["price"] >= lower) & (working["price"] < upper), "price_band"] = label
    return working


def _segment_metrics(df: pd.DataFrame, predictions: pd.DataFrame) -> dict:
    merged = df[["price", "make"]].reset_index(drop=True).join(
        predictions[["predicted_price", "prediction_mode", "support_tier"]].reset_index(drop=True)
    )
    merged = _with_price_band(merged)
    result = {
        "by_prediction_mode": {},
        "by_support_tier": {},
        "by_price_band": {},
        "by_make_top10": {},
    }
    for key in ["prediction_mode", "support_tier", "price_band"]:
        target = f"by_{key}"
        for value, part in merged.groupby(key):
            result[target][str(value)] = metrics_from_predictions(
                part["price"].to_numpy(dtype=float),
                part["predicted_price"].to_numpy(dtype=float),
            ) | {"count": int(len(part))}

    top_makes = merged["make"].value_counts().head(10).index.tolist()
    for make in top_makes:
        part = merged[merged["make"] == make]
        result["by_make_top10"][str(make)] = metrics_from_predictions(
            part["price"].to_numpy(dtype=float),
            part["predicted_price"].to_numpy(dtype=float),
        ) | {"count": int(len(part))}
    return result


def _proxy_prevalence_report(bundle: PreparedDatasetBundle, min_prevalence: float = 0.005) -> dict:
    prevalent = {}
    dropped = {}
    for key, prevalence in bundle.proxy_prevalence.items():
        target = prevalent if prevalence >= min_prevalence else dropped
        target[key] = float(prevalence)
    return {
        "kept_proxy_flags": prevalent,
        "dropped_low_prevalence_flags": dropped,
    }


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

    bundle = prepare_dataset_bundle(config.db_path, table=config.table)
    raw_df = bundle.frame
    prepared = engineer_feature_frame(raw_df)
    prepared = _with_price_band(prepared)

    train_df, valid_df = time_split(prepared, config.validation_date_fraction)
    train_df["log_price"] = np.log1p(train_df["price"])
    valid_df["log_price"] = np.log1p(valid_df["price"])

    selected_anchor_name, ablation_metrics, _ = _run_anchor_ablations(train_df, valid_df, config.random_seed)
    anchor_feature_columns = ANCHOR_ABLATIONS[selected_anchor_name]
    anchor_categorical_columns = _categorical_columns_for(anchor_feature_columns)

    anchor_point_model = _fit_model(
        train_df,
        valid_df,
        anchor_feature_columns,
        anchor_categorical_columns,
        "log_price",
        "RMSE",
        config.random_seed,
    )
    anchor_lower_model = _fit_model(
        train_df,
        valid_df,
        anchor_feature_columns,
        anchor_categorical_columns,
        "log_price",
        "Quantile:alpha=0.10",
        config.random_seed,
    )
    anchor_upper_model = _fit_model(
        train_df,
        valid_df,
        anchor_feature_columns,
        anchor_categorical_columns,
        "log_price",
        "Quantile:alpha=0.90",
        config.random_seed,
    )

    train_df["anchor_log_pred"] = anchor_point_model.predict(train_df[anchor_feature_columns])
    valid_df["anchor_log_pred"] = anchor_point_model.predict(valid_df[anchor_feature_columns])

    inspection_delta_model = None
    inspection_calibrator = None
    inspection_adjustment_weight = 0.2
    inspection_delta_clip = [-0.12, 0.12]
    inspected_subset = prepared[prepared["has_inspection_score"] == 1].copy()
    inspected_subset["log_price"] = np.log1p(inspected_subset["price"])
    inspected_subset["anchor_log_pred"] = anchor_point_model.predict(inspected_subset[anchor_feature_columns])
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
        train_stage_a = inspection_delta_model.predict(inspection_train[DELTA_FEATURE_COLUMNS])
        valid_stage_a = inspection_delta_model.predict(inspection_valid[DELTA_FEATURE_COLUMNS])
        inspection_calibrator = IsotonicRegression(increasing=True, out_of_bounds="clip")
        inspection_calibrator.fit(
            train_stage_a,
            (inspection_train["delta_target"] - train_stage_a).to_numpy(dtype=float),
        )
        inspection_valid["stage_a_calibrated"] = valid_stage_a + inspection_calibrator.predict(valid_stage_a)
        inspection_delta_clip = [
            float(np.quantile(inspection_train["delta_target"], 0.10)),
            float(np.quantile(inspection_train["delta_target"], 0.90)),
        ]
        anchor_only_price = np.expm1(inspection_valid["anchor_log_pred"].to_numpy(dtype=float))
        adjusted_price = np.expm1(inspection_valid["anchor_log_pred"].to_numpy(dtype=float) + inspection_valid["stage_a_calibrated"].to_numpy(dtype=float))
        actual_price = inspection_valid["price"].to_numpy(dtype=float)
        anchor_only_mae = float(np.mean(np.abs(actual_price - anchor_only_price)))
        adjusted_mae = float(np.mean(np.abs(actual_price - adjusted_price)))
        if anchor_only_mae > 0:
            improvement = (anchor_only_mae - adjusted_mae) / anchor_only_mae
            inspection_adjustment_weight = float(min(max(improvement, 0.18), 0.55))

    proxy_delta_model = None
    proxy_subset = prepared[(prepared["has_inspection_score"] == 0) & (prepared["has_proxy_signal"] == 1)].copy()
    proxy_subset["log_price"] = np.log1p(proxy_subset["price"])
    proxy_subset["anchor_log_pred"] = anchor_point_model.predict(proxy_subset[anchor_feature_columns])
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

    metadata = {
        "db_path": str(config.db_path),
        "table": config.table,
        "observed_max_year": bundle.observed_max_year,
        "inspection_weightage_version": SECTION_WEIGHTAGE_VERSION,
        "selected_anchor_ablation": selected_anchor_name,
        "anchor_feature_columns": anchor_feature_columns,
        "delta_feature_columns": DELTA_FEATURE_COLUMNS,
        "proxy_delta_feature_columns": PROXY_DELTA_FEATURE_COLUMNS,
        "inspection_adjustment_weight": inspection_adjustment_weight,
        "inspection_delta_clip": inspection_delta_clip,
        "has_inspection_delta_model": inspection_delta_model is not None,
        "has_inspection_calibrator": inspection_calibrator is not None,
        "has_proxy_delta_model": proxy_delta_model is not None,
    }
    artifact_bundle = {
        "metadata": metadata,
        "anchor_point_model": anchor_point_model,
        "anchor_lower_model": anchor_lower_model,
        "anchor_upper_model": anchor_upper_model,
        "inspection_delta_model": inspection_delta_model,
        "inspection_calibrator": inspection_calibrator,
        "proxy_delta_model": proxy_delta_model,
        "comparables": retriever,
        "calibration": {},
    }

    full_validation_predictions = predict_frame(valid_df.reset_index(drop=True), artifact_bundle, use_comparables=False)
    calibration = _signed_error_calibration(valid_df.reset_index(drop=True), full_validation_predictions)
    artifact_bundle["calibration"] = calibration
    full_validation_predictions = predict_frame(valid_df.reset_index(drop=True), artifact_bundle, use_comparables=False)

    overall_metrics = metrics_from_predictions(
        valid_df["price"].to_numpy(dtype=float),
        full_validation_predictions["predicted_price"].to_numpy(dtype=float),
    )
    overall_metrics["rows_used"] = int(len(prepared))
    overall_metrics["train_rows"] = int(len(train_df))
    overall_metrics["validation_rows"] = int(len(valid_df))
    overall_metrics["inspection_rows"] = int(prepared["has_inspection_score"].sum())
    overall_metrics["proxy_rows"] = int(prepared["has_proxy_signal"].sum())
    overall_metrics["prediction_modes"] = full_validation_predictions["prediction_mode"].value_counts().to_dict()
    overall_metrics["avg_range_width"] = float(np.mean(full_validation_predictions["price_range_high"] - full_validation_predictions["price_range_low"]))
    overall_metrics["range_coverage"] = float(
        np.mean((valid_df["price"] >= full_validation_predictions["price_range_low"]) & (valid_df["price"] <= full_validation_predictions["price_range_high"]))
    )
    overall_metrics["calibration_eval_mode"] = "full_validation_without_comparable_blend"

    interval_diagnostics = {
        "overall": {
            "coverage": overall_metrics["range_coverage"],
            "count": int(len(valid_df)),
        },
        "by_prediction_mode": _coverage_by(
            valid_df.reset_index(drop=True).assign(prediction_mode=full_validation_predictions["prediction_mode"].reset_index(drop=True)),
            full_validation_predictions,
            "prediction_mode",
        ),
        "by_support_tier": _coverage_by(
            valid_df.reset_index(drop=True).assign(support_tier=full_validation_predictions["support_tier"].reset_index(drop=True)),
            full_validation_predictions,
            "support_tier",
        ),
        "by_price_band": _coverage_by(valid_df.reset_index(drop=True), full_validation_predictions, "price_band"),
    }

    segment_metrics = _segment_metrics(valid_df.reset_index(drop=True), full_validation_predictions)
    reliability_report = {
        mode: segment_metrics["by_prediction_mode"].get(mode, {"count": 0})
        for mode in ["no_inspection", "score_only", "section_based", "text_proxy"]
    }

    validation_eval = valid_df.copy()
    if len(validation_eval) > config.validation_comp_sample:
        validation_eval = validation_eval.sample(config.validation_comp_sample, random_state=config.random_seed).copy()
    validation_eval = validation_eval.reset_index(drop=True)
    validation_predictions_sample = predict_frame(validation_eval, artifact_bundle).join(validation_eval[["price"]])
    sampled_blended_metrics = metrics_from_predictions(
        validation_eval["price"].to_numpy(dtype=float),
        validation_predictions_sample["predicted_price"].to_numpy(dtype=float),
    )
    sampled_blended_metrics["count"] = int(len(validation_eval))

    feature_importance = anchor_point_model.get_feature_importance(prettified=True)
    feature_importance = feature_importance.rename(columns={"Feature Id": "feature", "Importances": "importance"})
    feature_importance.to_csv(model_dir / "feature_importance.csv", index=False)

    anchor_point_model.save_model(str(model_dir / "anchor_point_model.cbm"))
    anchor_lower_model.save_model(str(model_dir / "anchor_lower_model.cbm"))
    anchor_upper_model.save_model(str(model_dir / "anchor_upper_model.cbm"))
    if inspection_delta_model is not None:
        inspection_delta_model.save_model(str(model_dir / "inspection_delta_model.cbm"))
    if inspection_calibrator is not None:
        with open(model_dir / "inspection_calibrator.pkl", "wb") as file:
            pickle.dump(inspection_calibrator, file)
    if proxy_delta_model is not None:
        proxy_delta_model.save_model(str(model_dir / "proxy_delta_model.cbm"))
    retriever.save(model_dir / "comparables.pkl")

    # Remove legacy artifact names so deployment only exposes the hardened contract.
    for legacy_name in ["point_model.cbm", "lower_model.cbm", "upper_model.cbm", "inspection_isotonic.pkl"]:
        legacy_path = model_dir / legacy_name
        if legacy_path.exists():
            legacy_path.unlink()

    profile = {
        "rows_after_cleaning": int(len(prepared)),
        "rows_with_inspection": int(prepared["has_inspection_score"].sum()),
        "rows_with_proxy_signals": int(prepared["has_proxy_signal"].sum()),
        "median_price": float(prepared["price"].median()),
        "median_year": float(prepared["year"].median()),
        "median_mileage": float(prepared["mileage"].median()),
    }

    save_catalog(build_catalog(prepared), model_dir / "catalog.json")
    validation_predictions_sample.to_csv(model_dir / "validation_predictions_sample.csv", index=False)

    for path, payload in [
        (model_dir / "metadata.json", metadata),
        (model_dir / "metrics.json", overall_metrics),
        (model_dir / "calibration.json", calibration),
        (model_dir / "data_profile.json", profile),
        (model_dir / "dataset_audit.json", bundle.audit),
        (model_dir / "proxy_prevalence.json", _proxy_prevalence_report(bundle)),
        (model_dir / "anchor_ablation_metrics.json", ablation_metrics),
        (model_dir / "interval_diagnostics.json", interval_diagnostics),
        (model_dir / "segment_metrics.json", segment_metrics),
        (model_dir / "reliability_report.json", reliability_report),
        (model_dir / "sampled_blended_metrics.json", sampled_blended_metrics),
    ]:
        with open(path, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)

    return overall_metrics


if __name__ == "__main__":
    main()
