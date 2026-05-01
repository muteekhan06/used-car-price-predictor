from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from comparables import ComparableRetriever
from features import (
    ANCHOR_FEATURE_COLUMNS,
    DELTA_FEATURE_COLUMNS,
    PROXY_DELTA_FEATURE_COLUMNS,
)


def load_optional_model(path: Path) -> CatBoostRegressor | None:
    if not path.exists():
        return None
    model = CatBoostRegressor()
    model.load_model(str(path))
    return model


def load_artifacts(model_dir: str | Path) -> dict:
    model_dir = Path(model_dir)
    with open(model_dir / "metadata.json", "r", encoding="utf-8") as file:
        metadata = json.load(file)

    calibration = {}
    calibration_path = model_dir / "calibration.json"
    if calibration_path.exists():
        with open(calibration_path, "r", encoding="utf-8") as file:
            calibration = json.load(file)

    return {
        "metadata": metadata,
        "calibration": calibration,
        "anchor_point_model": load_optional_model(model_dir / "anchor_point_model.cbm"),
        "anchor_lower_model": load_optional_model(model_dir / "anchor_lower_model.cbm"),
        "anchor_upper_model": load_optional_model(model_dir / "anchor_upper_model.cbm"),
        "inspection_delta_model": load_optional_model(model_dir / "inspection_delta_model.cbm"),
        "proxy_delta_model": load_optional_model(model_dir / "proxy_delta_model.cbm"),
        "inspection_isotonic_model": load_optional_pickle(model_dir / "inspection_isotonic.pkl"),
        "comparables": ComparableRetriever.load(model_dir / "comparables.pkl"),
    }


def load_optional_pickle(path: Path):
    if not path.exists():
        return None
    with open(path, "rb") as file:
        return pickle.load(file)


def _compute_comp_weight(summary: dict, prediction_mode: str) -> float:
    count_factor = min(summary["comp_count"] / 15.0, 1.0)
    dispersion_penalty = min(summary["iqr_ratio"] or 1.0, 1.0)
    exact_variant_bonus = min(summary["exact_variant_matches"] / 5.0, 1.0)
    base = 0.15 + 0.45 * count_factor + 0.15 * exact_variant_bonus - 0.2 * dispersion_penalty

    if prediction_mode == "no_inspection":
        base += 0.10
    elif prediction_mode == "score_only":
        base -= 0.05

    return float(min(max(base, 0.0), 0.75))


def _adjust_comp_weight_for_inspection(comp_weight: float, row: pd.Series, prediction_mode: str) -> float:
    if prediction_mode not in {"score_only", "section_based"}:
        return comp_weight
    score = float(row.get("inspection_score", np.nan))
    if np.isnan(score):
        return comp_weight

    low_score_factor = max(0.0, min((8.0 - score) / 7.0, 1.0))
    high_score_factor = max(0.0, min((score - 8.0) / 2.0, 1.0))
    adjusted = comp_weight * (1.0 - 0.5 * low_score_factor - 0.1 * high_score_factor)
    return float(min(max(adjusted, 0.0), 0.75))


def _compute_confidence(summary: dict, prediction_mode: str) -> float:
    comp_count_factor = min(summary["comp_count"] / 12.0, 1.0)
    dispersion_factor = 1.0 - min(summary["iqr_ratio"] or 1.0, 1.0)
    mode_factor = {
        "score_only": 0.9,
        "section_based": 0.85,
        "text_proxy": 0.6,
        "no_inspection": 0.4,
    }.get(prediction_mode, 0.4)
    confidence = 0.45 * comp_count_factor + 0.25 * dispersion_factor + 0.30 * mode_factor
    return float(min(max(confidence, 0.05), 0.98))


def _support_profile(row: pd.Series) -> tuple[int, float, str]:
    raw_rows = row.get("catalog_source_rows", np.nan)
    if raw_rows is None or pd.isna(raw_rows):
        return 0, 0.0, "unknown"

    source_rows = int(raw_rows)
    support_factor = float(min(max(source_rows / 100.0, 0.0), 1.0))
    if source_rows >= 100:
        return source_rows, support_factor, "strong"
    if source_rows >= 30:
        return source_rows, support_factor, "moderate"
    return source_rows, support_factor, "thin"


def predict_frame(frame: pd.DataFrame, artifacts: dict, top_k_comps: int = 20) -> pd.DataFrame:
    frame = frame.reset_index(drop=True).copy()
    anchor_point_model = artifacts["anchor_point_model"]
    anchor_lower_model = artifacts["anchor_lower_model"]
    anchor_upper_model = artifacts["anchor_upper_model"]
    inspection_delta_model = artifacts["inspection_delta_model"]
    inspection_isotonic_model = artifacts.get("inspection_isotonic_model")
    proxy_delta_model = artifacts["proxy_delta_model"]
    comparables = artifacts["comparables"]
    calibration = artifacts.get("calibration", {})

    anchor_log_pred = anchor_point_model.predict(frame[ANCHOR_FEATURE_COLUMNS])
    lower_log_pred = anchor_lower_model.predict(frame[ANCHOR_FEATURE_COLUMNS])
    upper_log_pred = anchor_upper_model.predict(frame[ANCHOR_FEATURE_COLUMNS])
    frame["anchor_log_pred"] = anchor_log_pred

    point_price = np.expm1(anchor_log_pred)
    lower_price = np.expm1(lower_log_pred)
    upper_price = np.expm1(upper_log_pred)

    residual_log_delta = np.zeros(len(frame), dtype=float)

    inspection_mask = frame["has_inspection_score"] == 1
    if inspection_mask.any():
        inspection_index = inspection_mask.to_numpy()
        if inspection_isotonic_model is not None:
            residual_log_delta[inspection_index] = inspection_isotonic_model.predict(
                frame.loc[inspection_mask, "inspection_score"].to_numpy(dtype=float)
            )
        elif inspection_delta_model is not None:
            inspection_features = frame.loc[inspection_mask, DELTA_FEATURE_COLUMNS].copy()
            residual_log_delta[inspection_index] = inspection_delta_model.predict(inspection_features)

    proxy_mask = (frame["has_inspection_score"] == 0) & (frame["has_proxy_signal"] == 1)
    if proxy_delta_model is not None and proxy_mask.any():
        proxy_features = frame.loc[proxy_mask, PROXY_DELTA_FEATURE_COLUMNS].copy()
        residual_log_delta[proxy_mask.to_numpy()] = proxy_delta_model.predict(proxy_features)

    adjusted_point = point_price * np.exp(residual_log_delta)
    adjusted_lower = lower_price * np.exp(residual_log_delta)
    adjusted_upper = upper_price * np.exp(residual_log_delta)

    results = []
    for index, row in frame.iterrows():
        summary_obj = comparables.summarize(row.to_dict(), top_k=top_k_comps)
        summary = {
            "comp_count": summary_obj.comp_count,
            "weighted_price": summary_obj.weighted_price,
            "median_price": summary_obj.median_price,
            "p25_price": summary_obj.p25_price,
            "p75_price": summary_obj.p75_price,
            "iqr_ratio": summary_obj.iqr_ratio,
            "exact_variant_matches": summary_obj.exact_variant_matches,
            "exact_city_matches": summary_obj.exact_city_matches,
            "comps": summary_obj.comps,
        }

        prediction_mode = row["prediction_mode"]
        if prediction_mode == "score_only" and str(row.get("inspection_input_source", "")).startswith("derived"):
            prediction_mode = "section_based"
        source_rows, support_factor, support_tier = _support_profile(row)

        comp_weight = 0.0
        blended_point = float(adjusted_point[index])
        blended_lower = float(adjusted_lower[index])
        blended_upper = float(adjusted_upper[index])

        if summary["comp_count"] > 0 and summary["weighted_price"] is not None:
            comp_weight = _compute_comp_weight(summary, prediction_mode)
            comp_weight = _adjust_comp_weight_for_inspection(comp_weight, row, prediction_mode)
            if source_rows > 0 and source_rows < 100:
                min_stat_weight = 0.30 + (1.0 - support_factor) * 0.30
                comp_weight = max(comp_weight, min_stat_weight)
            model_weight = 1.0 - comp_weight
            comp_point = float(summary["weighted_price"])
            comp_lower = float(summary["p25_price"] or comp_point)
            comp_upper = float(summary["p75_price"] or comp_point)
            blended_point = model_weight * blended_point + comp_weight * comp_point
            blended_lower = model_weight * blended_lower + comp_weight * comp_lower
            blended_upper = model_weight * blended_upper + comp_weight * comp_upper

        calibration_mode = calibration.get(prediction_mode) or calibration.get("default") or {
            "q10": -0.16,
            "q90": 0.16,
        }
        cal_low = blended_point * (1.0 + float(calibration_mode["q10"]))
        cal_high = blended_point * (1.0 + float(calibration_mode["q90"]))
        final_low = min(blended_lower, cal_low, blended_point)
        final_high = max(blended_upper, cal_high, blended_point)

        confidence = _compute_confidence(summary, prediction_mode)
        if prediction_mode in {"score_only", "section_based"} and float(row.get("inspection_score", 10)) < 4:
            confidence *= 0.85
        if prediction_mode == "no_inspection":
            final_low *= 0.98
            final_high *= 1.02
        if source_rows > 0 and source_rows < 100:
            width_penalty = 1.0 + ((1.0 - support_factor) * 0.14)
            final_low *= 1.0 - ((width_penalty - 1.0) * 0.6)
            final_high *= width_penalty
            confidence *= 0.70 + (0.30 * support_factor)

        results.append(
            {
                "predicted_price": float(blended_point),
                "price_range_low": float(max(0.0, min(final_low, blended_point))),
                "price_range_high": float(max(final_high, blended_point)),
                "prediction_mode": prediction_mode,
                "confidence_score": confidence,
                "catalog_source_rows": source_rows,
                "support_tier": support_tier,
                "anchor_price": float(point_price[index]),
                "condition_adjusted_price": float(adjusted_point[index]),
                "comparable_reference_price": summary["weighted_price"],
                "comparable_count": summary["comp_count"],
                "comparables": summary["comps"],
                "comp_weight": comp_weight,
            }
        )

    return pd.DataFrame(results, index=frame.index)
