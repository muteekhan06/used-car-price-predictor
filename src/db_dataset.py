from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


RAW_COLUMNS = [
    "id",
    "source",
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
    "body_type",
    "engine_capacity",
    "engine_displacement",
    "rating",
    "description",
    "extra_features",
    "listing_url",
    "source_listing_hash",
    "ad_last_updated",
]

CRITICAL_CATEGORICAL_COLUMNS = [
    "make",
    "model",
    "variant",
    "transmission",
    "fuel_type",
    "assembly",
    "registered_in",
]

OPTIONAL_CATEGORY_DEFAULTS = {
    "body_type": "Unknown",
    "color": "Unknown",
}

POSITIVE_DESCRIPTION_PATTERNS = {
    "inspection_report_attached_flag": [
        "inspection report",
        "report attached",
        "inspected car",
    ],
    "pakwheels_inspected_flag": [
        "pakwheels inspected",
    ],
    "like_new_flag": [
        "like new",
        "brand new",
        "scratchless",
    ],
    "total_genuine_flag": [
        "total genuine",
        "genuine body",
        "bumper to bumper genuine",
    ],
    "non_accidental_flag": [
        "non-accidental",
        "non accidental",
        "accident free",
    ],
}

NUMERIC_PATTERN = re.compile(r"[^0-9]")
RATING_PATTERN = re.compile(r"([0-9]+(?:\.[0-9]+)?)")


@dataclass
class PreparedDatasetBundle:
    frame: pd.DataFrame
    audit: dict
    observed_max_year: int
    proxy_prevalence: dict[str, float]


def parse_numeric_text(value: object) -> int | None:
    if value is None:
        return None
    digits = NUMERIC_PATTERN.sub("", str(value))
    return int(digits) if digits else None


def parse_rating_text(value: object) -> float | None:
    if value is None:
        return None
    match = RATING_PATTERN.search(str(value))
    return float(match.group(1)) if match else None


def clean_text(value: object, default: str | None = None) -> str | None:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def canonicalize_transmission(value: object) -> str | None:
    text = clean_text(value)
    if text is None:
        return None
    lowered = text.lower()
    if "auto" in lowered or "cvt" in lowered or "ags" in lowered or "triptronic" in lowered:
        return "Automatic"
    if "manual" in lowered:
        return "Manual"
    return text.title()


def canonicalize_fuel_type(value: object) -> str | None:
    text = clean_text(value)
    if text is None:
        return None
    lowered = text.lower()
    if "hybrid" in lowered:
        return "Hybrid"
    if "diesel" in lowered:
        return "Diesel"
    if "electric" in lowered or lowered == "ev":
        return "Electric"
    if "cng" in lowered:
        return "CNG"
    if "petrol" in lowered or "gasoline" in lowered:
        return "Petrol"
    return text.title()


def canonicalize_assembly(value: object) -> str | None:
    text = clean_text(value)
    if text is None:
        return None
    lowered = text.lower()
    if "import" in lowered:
        return "Imported"
    if "local" in lowered:
        return "Local"
    return text.title()


def canonicalize_generic(value: object) -> str | None:
    text = clean_text(value)
    if text is None:
        return None
    return " ".join(part.capitalize() for part in text.split())


def extract_extra_feature_signals(raw_json: object) -> dict:
    defaults = {
        "owner_count": pd.NA,
        "first_owner_flag": 0,
        "accident_free_flag": 0,
        "bumper_to_bumper_original_flag": 0,
        "ac_ok_flag": 0,
        "power_steering_flag": 0,
        "tax_up_to_date_flag": 0,
    }
    if raw_json is None:
        return defaults
    try:
        parsed = json.loads(str(raw_json))
    except Exception:
        return defaults

    desc = parsed.get("desc_features", {})
    owner_count = desc.get("desc_owner_count")
    defaults["owner_count"] = owner_count if owner_count is not None else pd.NA
    defaults["first_owner_flag"] = int(bool(desc.get("desc_is_first_owner")))
    defaults["accident_free_flag"] = int(bool(desc.get("desc_accident_free")))
    defaults["bumper_to_bumper_original_flag"] = int(bool(desc.get("desc_bumper_to_bumper_original")))
    defaults["ac_ok_flag"] = int(bool(desc.get("desc_ac_ok")))
    defaults["power_steering_flag"] = int(bool(desc.get("desc_power_steering")))
    defaults["tax_up_to_date_flag"] = int(bool(desc.get("desc_tax_up_to_date")))
    return defaults


def extract_description_signals(description: object) -> dict:
    text = (clean_text(description, default="") or "").lower()
    result = {key: 0 for key in POSITIVE_DESCRIPTION_PATTERNS}
    for key, patterns in POSITIVE_DESCRIPTION_PATTERNS.items():
        result[key] = int(any(pattern in text for pattern in patterns))
    return result


def load_raw_dataset(db_path: str | Path, table: str = "car_listings_old") -> pd.DataFrame:
    connection = sqlite3.connect(str(db_path))
    try:
        query = f"SELECT {', '.join(RAW_COLUMNS)} FROM {table}"
        return pd.read_sql_query(query, connection)
    finally:
        connection.close()


def _drop_with_reason(df: pd.DataFrame, mask: pd.Series, audit: dict, reason: str) -> pd.DataFrame:
    dropped = int(mask.sum())
    if dropped:
        audit["drop_reasons"][reason] = audit["drop_reasons"].get(reason, 0) + dropped
    return df.loc[~mask].copy()


def prepare_dataset_bundle(db_path: str | Path, table: str = "car_listings_old") -> PreparedDatasetBundle:
    raw = load_raw_dataset(db_path, table=table)
    audit = {
        "raw_rows": int(len(raw)),
        "drop_reasons": {},
        "kept_rows": 0,
    }

    observed_years = raw["year"].apply(parse_numeric_text).dropna()
    observed_max_year = int(observed_years.max()) if not observed_years.empty else 2026

    df = raw.drop_duplicates(subset=["source_listing_hash"], keep="last").copy()
    audit["drop_reasons"]["duplicate_source_listing_hash"] = int(len(raw) - len(df))

    before_url = len(df)
    df = df.drop_duplicates(subset=["listing_url"], keep="last").copy()
    audit["drop_reasons"]["duplicate_listing_url"] = int(before_url - len(df))

    df["year"] = df["year"].apply(parse_numeric_text)
    df["price"] = df["price"].apply(parse_numeric_text)
    df["mileage"] = df["mileage"].apply(parse_numeric_text)
    df["engine_capacity_cc"] = df["engine_capacity"].where(
        df["engine_capacity"].notna(), df["engine_displacement"]
    ).apply(parse_numeric_text)
    df["inspection_score"] = df["rating"].apply(parse_rating_text)
    df["ad_last_updated_date"] = pd.to_datetime(df["ad_last_updated"], errors="coerce")

    df["make"] = df["make"].map(canonicalize_generic)
    df["model"] = df["model"].map(canonicalize_generic)
    df["variant"] = df["variant"].map(canonicalize_generic)
    df["transmission"] = df["vehicle_transmission"].map(canonicalize_transmission)
    df["fuel_type"] = df["fuel_type"].map(canonicalize_fuel_type)
    df["registered_in"] = df["registered_in"].map(canonicalize_generic)
    df["color"] = df["color"].map(canonicalize_generic)
    df["assembly"] = df["assembly"].map(canonicalize_assembly)
    df["body_type"] = df["body_type"].map(canonicalize_generic)
    df["description"] = df["description"].fillna("").astype(str)

    extra_signals = pd.DataFrame(df["extra_features"].map(extract_extra_feature_signals).tolist())
    description_signals = pd.DataFrame(df["description"].map(extract_description_signals).tolist())
    df = pd.concat([df, extra_signals, description_signals], axis=1)

    df["owner_count"] = pd.to_numeric(df["owner_count"], errors="coerce")
    df["owner_count_known_flag"] = df["owner_count"].notna().astype(int)
    df["owner_count"] = df["owner_count"].fillna(0)

    proxy_columns = [
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
    for column in proxy_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0).astype(int)

    df["proxy_signal_count"] = df[proxy_columns].sum(axis=1)
    df["has_proxy_signal"] = (df["proxy_signal_count"] > 0).astype(int)
    df["has_inspection_score"] = df["inspection_score"].notna().astype(int)

    invalid_numeric_mask = df["price"].isna() | df["year"].isna() | df["mileage"].isna()
    df = _drop_with_reason(df, invalid_numeric_mask, audit, "invalid_numeric_parse")

    outlier_mask = (
        ~df["price"].between(100_000, 100_000_000, inclusive="both")
        | ~df["year"].between(1985, observed_max_year, inclusive="both")
        | ~df["mileage"].between(0, 500_000, inclusive="both")
    )
    df = _drop_with_reason(df, outlier_mask, audit, "extreme_outlier_or_unsupported_year")

    for column in CRITICAL_CATEGORICAL_COLUMNS:
        missing_mask = df[column].isna() | df[column].astype(str).str.strip().eq("")
        df = _drop_with_reason(df, missing_mask, audit, f"missing_required_{column}")

    df["body_type_missing_flag"] = df["body_type"].isna().astype(int)
    df["body_type"] = df["body_type"].fillna(OPTIONAL_CATEGORY_DEFAULTS["body_type"])
    df["color"] = df["color"].fillna(OPTIONAL_CATEGORY_DEFAULTS["color"])
    df["engine_capacity_cc"] = pd.to_numeric(df["engine_capacity_cc"], errors="coerce")

    df["prediction_mode"] = "no_inspection"
    df.loc[df["has_proxy_signal"] == 1, "prediction_mode"] = "text_proxy"
    df.loc[df["has_inspection_score"] == 1, "prediction_mode"] = "score_only"

    audit["kept_rows"] = int(len(df))
    proxy_prevalence = {
        column: float(df[column].mean()) if len(df) else 0.0
        for column in proxy_columns
    }

    return PreparedDatasetBundle(
        frame=df.reset_index(drop=True),
        audit=audit,
        observed_max_year=observed_max_year,
        proxy_prevalence=proxy_prevalence,
    )


def load_and_prepare_dataset(db_path: str | Path, table: str = "car_listings_old") -> pd.DataFrame:
    return prepare_dataset_bundle(db_path, table=table).frame
