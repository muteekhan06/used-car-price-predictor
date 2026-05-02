from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from comparables import ComparableRetriever
def load_optional_model(path: Path) -> CatBoostRegressor | None:
    if not path.exists():
        return None
    model = CatBoostRegressor()
    model.load_model(str(path))
    return model


def load_optional_pickle(path: Path):
    if not path.exists():
        return None
    with open(path, "rb") as file:
        return pickle.load(file)


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
        "inspection_calibrator": load_optional_pickle(model_dir / "inspection_calibrator.pkl"),
        "comparables": ComparableRetriever.load(model_dir / "comparables.pkl"),
    }


def _compute_comp_weight(summary: dict, prediction_mode: str, support_factor: float) -> float:
    count_factor = min(summary["usable_comp_count"] / 12.0, 1.0)
    exact_variant_bonus = min(summary["exact_variant_matches"] / 5.0, 1.0)
    dispersion_penalty = min(summary["iqr_ratio"] or 1.0, 1.0)
    base = 0.12 + 0.42 * count_factor + 0.12 * exact_variant_bonus - 0.18 * dispersion_penalty
    if prediction_mode == "no_inspection":
        base += 0.06
    base *= 0.55 + 0.45 * support_factor
    return float(min(max(base, 0.0), 0.65))


def _compute_confidence_index(summary: dict, prediction_mode: str, support_factor: float) -> float:
    comp_count_factor = min(summary["usable_comp_count"] / 10.0, 1.0)
    dispersion_factor = 1.0 - min(summary["iqr_ratio"] or 1.0, 1.0)
    mode_factor = {
        "score_only": 0.9,
        "section_based": 0.86,
        "text_proxy": 0.58,
        "no_inspection": 0.42,
    }.get(prediction_mode, 0.42)
    confidence = 0.35 * comp_count_factor + 0.25 * dispersion_factor + 0.20 * support_factor + 0.20 * mode_factor
    return float(min(max(confidence, 0.05), 0.97))


def _provided_score_prior_delta(score: float) -> float:
    centered = 0.015 * (float(score) - 7.0)
    return float(min(max(centered, -0.09), 0.05))


def _support_profile(row: pd.Series, summary: dict) -> tuple[int, int, float, str]:
    exact_variant_rows = int(row.get("catalog_source_rows", 0) or 0)
    exact_model_rows = int(summary["exact_model_rows"])
    support_factor = float(min(max(max(exact_variant_rows, exact_model_rows) / 100.0, 0.0), 1.0))
    if max(exact_variant_rows, exact_model_rows) >= 100:
        support_tier = "strong"
    elif max(exact_variant_rows, exact_model_rows) >= 30:
        support_tier = "moderate"
    else:
        support_tier = "thin"
    return exact_variant_rows, exact_model_rows, support_factor, support_tier


def predict_frame(frame: pd.DataFrame, artifacts: dict, top_k_comps: int = 20, use_comparables: bool = True) -> pd.DataFrame:
    frame = frame.reset_index(drop=True).copy()
    metadata = artifacts["metadata"]
    anchor_feature_columns = metadata["anchor_feature_columns"]
    delta_feature_columns = metadata["delta_feature_columns"]
    proxy_delta_feature_columns = metadata["proxy_delta_feature_columns"]
    anchor_point_model = artifacts["anchor_point_model"]
    anchor_lower_model = artifacts["anchor_lower_model"]
    anchor_upper_model = artifacts["anchor_upper_model"]
    inspection_delta_model = artifacts["inspection_delta_model"]
    inspection_calibrator = artifacts.get("inspection_calibrator")
    proxy_delta_model = artifacts["proxy_delta_model"]
    comparables = artifacts["comparables"]
    calibration = artifacts.get("calibration", {})
    inspection_adjustment_weight = float(metadata.get("inspection_adjustment_weight", 0.2))
    inspection_delta_clip = metadata.get("inspection_delta_clip", [-0.12, 0.12])

    anchor_log_pred = anchor_point_model.predict(frame[anchor_feature_columns])
    lower_log_pred = anchor_lower_model.predict(frame[anchor_feature_columns])
    upper_log_pred = anchor_upper_model.predict(frame[anchor_feature_columns])
    frame["anchor_log_pred"] = anchor_log_pred

    point_price = np.expm1(anchor_log_pred)
    lower_price = np.expm1(lower_log_pred)
    upper_price = np.expm1(upper_log_pred)

    residual_log_delta = np.zeros(len(frame), dtype=float)

    inspection_mask = frame["has_inspection_score"] == 1
    if inspection_delta_model is not None and inspection_mask.any():
        inspection_features = frame.loc[inspection_mask, delta_feature_columns].copy()
        stage_a_delta = inspection_delta_model.predict(inspection_features)
        if inspection_calibrator is not None:
            stage_a_delta = stage_a_delta + inspection_calibrator.predict(stage_a_delta)
        clip_low = float(inspection_delta_clip[0])
        clip_high = float(inspection_delta_clip[1])
        stage_a_delta = np.clip(stage_a_delta, clip_low, clip_high)

        inspection_rows = frame.loc[inspection_mask, ["inspection_input_source", "inspection_completeness"]].copy()
        source_weight = np.where(
            inspection_rows["inspection_input_source"].eq("provided_score"),
            inspection_adjustment_weight,
            np.maximum(inspection_adjustment_weight, 0.45 + 0.35 * inspection_rows["inspection_completeness"].to_numpy(dtype=float)),
        )
        stage_a_delta = stage_a_delta * source_weight
        provided_score_mask = inspection_rows["inspection_input_source"].eq("provided_score").to_numpy(dtype=bool)
        if provided_score_mask.any():
            score_values = frame.loc[inspection_mask, "inspection_score"].to_numpy(dtype=float)[provided_score_mask]
            prior = np.array([_provided_score_prior_delta(score) for score in score_values], dtype=float)
            stage_a_delta[provided_score_mask] = 0.5 * stage_a_delta[provided_score_mask] + 0.5 * prior
        residual_log_delta[inspection_mask.to_numpy()] = stage_a_delta

    proxy_mask = (frame["has_inspection_score"] == 0) & (frame["has_proxy_signal"] == 1)
    if proxy_delta_model is not None and proxy_mask.any():
        proxy_features = frame.loc[proxy_mask, proxy_delta_feature_columns].copy()
        residual_log_delta[proxy_mask.to_numpy()] = proxy_delta_model.predict(proxy_features)

    adjusted_point = point_price * np.exp(residual_log_delta)
    adjusted_lower = lower_price * np.exp(residual_log_delta)
    adjusted_upper = upper_price * np.exp(residual_log_delta)

    results = []
    for index, row in frame.iterrows():
        if use_comparables:
            summary_obj = comparables.summarize(row.to_dict(), top_k=top_k_comps)
            summary = {
                "comp_count": summary_obj.comp_count,
                "usable_comp_count": summary_obj.usable_comp_count,
                "weighted_price": summary_obj.weighted_price,
                "median_price": summary_obj.median_price,
                "p25_price": summary_obj.p25_price,
                "p75_price": summary_obj.p75_price,
                "iqr_ratio": summary_obj.iqr_ratio,
                "exact_variant_matches": summary_obj.exact_variant_matches,
                "exact_city_matches": summary_obj.exact_city_matches,
                "exact_model_rows": summary_obj.exact_model_rows,
                "exact_variant_rows": summary_obj.exact_variant_rows,
                "comp_quality_passed": summary_obj.comp_quality_passed,
                "blend_mode_hint": summary_obj.blend_mode_hint,
                "comps": summary_obj.comps,
            }
        else:
            summary = {
                "comp_count": 0,
                "usable_comp_count": 0,
                "weighted_price": None,
                "median_price": None,
                "p25_price": None,
                "p75_price": None,
                "iqr_ratio": None,
                "exact_variant_matches": 0,
                "exact_city_matches": 0,
                "exact_model_rows": 0,
                "exact_variant_rows": 0,
                "comp_quality_passed": False,
                "blend_mode_hint": "anchor_only",
                "comps": [],
            }

        prediction_mode = row["prediction_mode"]
        exact_variant_rows, exact_model_rows, support_factor, support_tier = _support_profile(row, summary)

        blended_point = float(adjusted_point[index])
        blended_lower = float(adjusted_lower[index])
        blended_upper = float(adjusted_upper[index])
        comp_weight = 0.0
        blend_mode = "anchor_only" if prediction_mode == "no_inspection" else "anchor_plus_condition"

        if summary["comp_quality_passed"] and summary["weighted_price"] is not None:
            comp_weight = _compute_comp_weight(summary, prediction_mode, support_factor)
            comp_point = float(summary["weighted_price"])
            comp_lower = float(summary["p25_price"] or comp_point)
            comp_upper = float(summary["p75_price"] or comp_point)
            model_weight = 1.0 - comp_weight
            blended_point = model_weight * blended_point + comp_weight * comp_point
            blended_lower = model_weight * blended_lower + comp_weight * comp_lower
            blended_upper = model_weight * blended_upper + comp_weight * comp_upper
            blend_mode = "anchor_plus_comparables" if prediction_mode == "no_inspection" else "full_blend"

        calibration_mode = calibration.get(prediction_mode) or calibration.get("default") or {"q10": -0.16, "q90": 0.16}
        cal_low = blended_point * (1.0 + float(calibration_mode["q10"]))
        cal_high = blended_point * (1.0 + float(calibration_mode["q90"]))
        final_low = min(blended_lower, cal_low, blended_point)
        final_high = max(blended_upper, cal_high, blended_point)

        confidence_index = _compute_confidence_index(summary, prediction_mode, support_factor)
        if prediction_mode == "section_based" and float(row.get("inspection_completeness", 0.0)) < 1.0:
            final_low *= 0.95
            final_high *= 1.06
            confidence_index *= 0.9
        if support_tier == "thin":
            final_low *= 0.93
            final_high *= 1.09
            confidence_index *= 0.84
        if not summary["comp_quality_passed"]:
            confidence_index *= 0.9

        results.append(
            {
                "predicted_price": float(blended_point),
                "price_range_low": float(max(0.0, min(final_low, blended_point))),
                "price_range_high": float(max(final_high, blended_point)),
                "prediction_mode": prediction_mode,
                "confidence_index": float(min(max(confidence_index, 0.05), 0.97)),
                "support_tier": support_tier,
                "exact_variant_rows": exact_variant_rows,
                "exact_model_rows": exact_model_rows,
                "usable_comp_count": summary["usable_comp_count"],
                "comp_quality_passed": summary["comp_quality_passed"],
                "blend_mode": blend_mode,
                "inspection_source": str(row.get("inspection_input_source", "missing")),
                "inspection_completeness": float(row.get("inspection_completeness", 0.0) or 0.0),
                "anchor_price": float(point_price[index]),
                "condition_adjusted_price": float(adjusted_point[index]),
                "comparable_reference_price": summary["weighted_price"],
                "comparable_count": summary["comp_count"],
                "comparables": summary["comps"],
                "comp_weight": comp_weight,
            }
        )

    return pd.DataFrame(results, index=frame.index)
