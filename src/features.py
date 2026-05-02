from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd

from db_dataset import (
    CRITICAL_CATEGORICAL_COLUMNS,
    OPTIONAL_CATEGORY_DEFAULTS,
    canonicalize_assembly,
    canonicalize_fuel_type,
    canonicalize_generic,
    canonicalize_transmission,
    extract_description_signals,
    extract_extra_feature_signals,
    parse_numeric_text,
)
from inspection import SECTION_COLUMNS, enrich_with_inspection_features


CURRENT_YEAR = date.today().year

CORE_INPUT_COLUMNS = [
    "make",
    "model",
    "year",
    "mileage",
    "transmission",
    "fuel_type",
    "registered_in",
    "assembly",
]

IDENTITY_REFINER_COLUMNS = [
    "variant",
    "body_type",
    "engine_capacity_cc",
]

WEAK_CONTEXT_COLUMNS = [
    "color",
]

CONDITION_SECTION_COLUMNS = [
    "section_engine_transmission_clutch_pct",
    "section_exterior_body_pct",
    "section_suspension_steering_pct",
    "section_tyres_pct",
    "section_body_frame_accident_pct",
    "inspection_completeness",
]

ANCHOR_BASE_CATEGORICAL_COLUMNS = [
    "make",
    "model",
    "transmission",
    "fuel_type",
    "registered_in",
    "assembly",
]

ANCHOR_BASE_NUMERICAL_COLUMNS = [
    "year",
    "mileage",
    "car_age",
    "log_mileage",
    "engine_capacity_cc",
]

ANCHOR_CORE_FEATURE_COLUMNS = ANCHOR_BASE_CATEGORICAL_COLUMNS + ANCHOR_BASE_NUMERICAL_COLUMNS
ANCHOR_CORE_VARIANT_FEATURE_COLUMNS = ANCHOR_CORE_FEATURE_COLUMNS[:2] + ["variant"] + ANCHOR_CORE_FEATURE_COLUMNS[2:]
ANCHOR_CORE_VARIANT_BODY_FEATURE_COLUMNS = (
    ANCHOR_CORE_VARIANT_FEATURE_COLUMNS[:7] + ["body_type", "body_type_missing_flag"] + ANCHOR_CORE_VARIANT_FEATURE_COLUMNS[7:]
)
ANCHOR_CORE_VARIANT_BODY_COLOR_FEATURE_COLUMNS = (
    ANCHOR_CORE_VARIANT_BODY_FEATURE_COLUMNS[:7] + ["color"] + ANCHOR_CORE_VARIANT_BODY_FEATURE_COLUMNS[7:]
)

ANCHOR_ABLATIONS = {
    "core_only": ANCHOR_CORE_FEATURE_COLUMNS,
    "core_plus_variant": ANCHOR_CORE_VARIANT_FEATURE_COLUMNS,
    "core_plus_variant_body": ANCHOR_CORE_VARIANT_BODY_FEATURE_COLUMNS,
    "core_plus_variant_body_color": ANCHOR_CORE_VARIANT_BODY_COLOR_FEATURE_COLUMNS,
}

DEFAULT_ANCHOR_NAME = "core_only"
ANCHOR_FEATURE_COLUMNS = ANCHOR_ABLATIONS[DEFAULT_ANCHOR_NAME]
ANCHOR_CATEGORICAL_COLUMNS = [
    "make",
    "model",
    "transmission",
    "fuel_type",
    "registered_in",
    "assembly",
]

PROXY_FLAG_COLUMNS = [
    "first_owner_flag",
    "accident_free_flag",
    "bumper_to_bumper_original_flag",
    "ac_ok_flag",
    "power_steering_flag",
    "tax_up_to_date_flag",
    "inspection_report_attached_flag",
    "pakwheels_inspected_flag",
    "like_new_flag",
    "total_genuine_flag",
    "non_accidental_flag",
]

DELTA_CATEGORICAL_COLUMNS = [
    "make",
    "model",
    "transmission",
    "fuel_type",
    "registered_in",
    "assembly",
]
DELTA_NUMERICAL_COLUMNS = [
    "anchor_log_pred",
    "inspection_score",
    "car_age",
    "log_mileage",
    "engine_capacity_cc",
    "inspection_mileage_interaction",
    "inspection_age_interaction",
    "inspection_completeness",
    "section_engine_transmission_clutch_pct",
    "section_exterior_body_pct",
    "section_suspension_steering_pct",
    "section_tyres_pct",
    "section_body_frame_accident_pct",
]
DELTA_FEATURE_COLUMNS = DELTA_CATEGORICAL_COLUMNS + DELTA_NUMERICAL_COLUMNS

PROXY_DELTA_CATEGORICAL_COLUMNS = [
    "make",
    "model",
    "transmission",
    "fuel_type",
    "registered_in",
    "assembly",
]
PROXY_DELTA_NUMERICAL_COLUMNS = [
    "anchor_log_pred",
    "car_age",
    "log_mileage",
    "engine_capacity_cc",
    "owner_count",
    "owner_count_known_flag",
    "proxy_signal_count",
    "proxy_signal_strength",
] + PROXY_FLAG_COLUMNS
PROXY_DELTA_FEATURE_COLUMNS = PROXY_DELTA_CATEGORICAL_COLUMNS + PROXY_DELTA_NUMERICAL_COLUMNS


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={column: column.strip().lower() for column in df.columns})


def _ensure_columns(df: pd.DataFrame, columns: list[str], value: object) -> None:
    for column in columns:
        if column not in df.columns:
            df[column] = value


def engineer_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df.copy())
    df = enrich_with_inspection_features(df)

    _ensure_columns(
        df,
        ["description", "extra_features", "body_type", "engine_capacity_cc", "color", "body_type_missing_flag"],
        pd.NA,
    )
    _ensure_columns(df, PROXY_FLAG_COLUMNS + ["owner_count", "owner_count_known_flag", "proxy_signal_count"], 0)

    if "vehicle_transmission" in df.columns and "transmission" not in df.columns:
        df["transmission"] = df["vehicle_transmission"]

    df["make"] = df["make"].map(canonicalize_generic)
    df["model"] = df["model"].map(canonicalize_generic)
    df["variant"] = df["variant"].map(canonicalize_generic)
    df["transmission"] = df["transmission"].map(canonicalize_transmission)
    df["fuel_type"] = df["fuel_type"].map(canonicalize_fuel_type)
    df["registered_in"] = df["registered_in"].map(canonicalize_generic)
    df["color"] = df["color"].map(canonicalize_generic)
    df["assembly"] = df["assembly"].map(canonicalize_assembly)
    df["body_type"] = df["body_type"].map(canonicalize_generic)

    for column in ["year", "mileage", "engine_capacity_cc", "inspection_score", "owner_count"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if "extra_features" in df.columns:
        extra_data = pd.DataFrame(df["extra_features"].map(extract_extra_feature_signals).tolist())
        for column in extra_data.columns:
            if column in df.columns:
                df[column] = df[column].fillna(extra_data[column])
            else:
                df[column] = extra_data[column]

    if "description" in df.columns:
        desc_data = pd.DataFrame(df["description"].fillna("").map(extract_description_signals).tolist())
        for column in desc_data.columns:
            if column in df.columns:
                df[column] = pd.to_numeric(df[column], errors="coerce").fillna(desc_data[column])
            else:
                df[column] = desc_data[column]

    for column in PROXY_FLAG_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0).astype(int)

    for column in SECTION_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["owner_count"] = pd.to_numeric(df["owner_count"], errors="coerce")
    df["owner_count_known_flag"] = df["owner_count"].notna().astype(int)
    df["owner_count"] = df["owner_count"].fillna(0)

    df["proxy_signal_count"] = df[PROXY_FLAG_COLUMNS].sum(axis=1)
    df["proxy_signal_strength"] = (df["proxy_signal_count"] / len(PROXY_FLAG_COLUMNS)).clip(lower=0, upper=1)
    df["has_proxy_signal"] = (df["proxy_signal_count"] > 0).astype(int)
    df["has_inspection_score"] = df["inspection_score"].notna().astype(int)

    df["body_type_missing_flag"] = pd.to_numeric(df["body_type_missing_flag"], errors="coerce").fillna(df["body_type"].isna().astype(int)).astype(int)
    df["body_type"] = df["body_type"].fillna(OPTIONAL_CATEGORY_DEFAULTS["body_type"])
    df["color"] = df["color"].fillna(OPTIONAL_CATEGORY_DEFAULTS["color"])

    df["year"] = df["year"].clip(lower=1985, upper=CURRENT_YEAR)
    df["mileage"] = df["mileage"].clip(lower=0, upper=500_000)
    df["engine_capacity_cc"] = df["engine_capacity_cc"].clip(lower=200, upper=10000)
    df["car_age"] = (CURRENT_YEAR - df["year"]).clip(lower=0)
    df["log_mileage"] = np.log1p(df["mileage"])
    df["inspection_score"] = pd.to_numeric(df["inspection_score"], errors="coerce").clip(lower=0, upper=10)
    df["inspection_mileage_interaction"] = df["inspection_score"].fillna(0) * df["log_mileage"]
    df["inspection_age_interaction"] = df["inspection_score"].fillna(0) * df["car_age"]

    df["prediction_mode"] = "no_inspection"
    df.loc[df["has_proxy_signal"] == 1, "prediction_mode"] = "text_proxy"
    df.loc[df["has_inspection_score"] == 1, "prediction_mode"] = "score_only"
    df.loc[
        (df["inspection_input_source"].astype(str).str.startswith("derived"))
        & (df["has_inspection_score"] == 1),
        "prediction_mode",
    ] = "section_based"

    return df


def prepare_user_input(record: dict) -> pd.DataFrame:
    normalized = {str(key).strip().lower(): value for key, value in record.items()}
    if "vehicle_transmission" in normalized and "transmission" not in normalized:
        normalized["transmission"] = normalized["vehicle_transmission"]
    if "engine_type" in normalized and "fuel_type" not in normalized:
        normalized["fuel_type"] = normalized["engine_type"]

    for column in CORE_INPUT_COLUMNS + ["variant", "color"]:
        normalized.setdefault(column, pd.NA)

    missing = [
        column
        for column in CORE_INPUT_COLUMNS
        if normalized.get(column) is None or str(normalized.get(column)).strip() == "" or str(normalized.get(column)).lower() == "<na>"
    ]
    if missing:
        raise ValueError("Missing required input fields: " + ", ".join(missing))

    if "engine_capacity_cc" not in normalized:
        normalized["engine_capacity_cc"] = parse_numeric_text(
            normalized.get("engine_capacity") or normalized.get("engine_displacement")
        )

    if "extra_features" in normalized and isinstance(normalized["extra_features"], dict):
        normalized["extra_features"] = json.dumps(normalized["extra_features"])

    frame = pd.DataFrame([normalized])
    prepared = engineer_feature_frame(frame)

    # Prediction API keeps color optional for catalog-supported flows; model input still expects a value.
    prepared["color"] = prepared["color"].fillna(OPTIONAL_CATEGORY_DEFAULTS["color"])
    prepared["body_type"] = prepared["body_type"].fillna(OPTIONAL_CATEGORY_DEFAULTS["body_type"])
    return prepared
