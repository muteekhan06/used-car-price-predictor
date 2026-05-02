from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


SECTION_WEIGHTAGES = {
    "section_interior_pct": 8.42,
    "section_engine_transmission_clutch_pct": 19.35,
    "section_electrical_electronics_pct": 12.20,
    "section_exterior_body_pct": 26.36,
    "section_ac_heater_pct": 5.10,
    "section_brakes_pct": 1.50,
    "section_suspension_steering_pct": 4.00,
    "section_tyres_pct": 2.16,
}
SECTION_WEIGHTAGE_VERSION = "pakwheels-like-v1"

SECTION_COLUMNS = [
    "section_interior_pct",
    "section_engine_transmission_clutch_pct",
    "section_electrical_electronics_pct",
    "section_body_frame_accident_pct",
    "section_exterior_body_pct",
    "section_ac_heater_pct",
    "section_brakes_pct",
    "section_suspension_steering_pct",
    "section_tyres_pct",
]

SECTION_LABELS = {
    "section_interior_pct": "Interior",
    "section_engine_transmission_clutch_pct": "Engine / Transmission / Clutch",
    "section_electrical_electronics_pct": "Electrical & Electronics",
    "section_body_frame_accident_pct": "Body Frame Accident Checklist",
    "section_exterior_body_pct": "Exterior & Body",
    "section_ac_heater_pct": "AC / Heater",
    "section_brakes_pct": "Brakes",
    "section_suspension_steering_pct": "Suspension / Steering",
    "section_tyres_pct": "Tyres",
}

TOTAL_SECTION_WEIGHT = sum(SECTION_WEIGHTAGES.values())


@dataclass
class InspectionComputation:
    score_out_of_10: float | None
    completeness_ratio: float
    source: str


def _to_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clip_percentage(value: float) -> float:
    return min(max(value, 0.0), 100.0)


def compute_weighted_inspection_score(record: dict) -> InspectionComputation:
    weighted_score = 0.0
    covered_weight = 0.0

    for column, weight in SECTION_WEIGHTAGES.items():
        value = _to_float(record.get(column))
        if value is None:
            continue
        pct = _clip_percentage(value)
        weighted_score += (pct / 100.0) * weight
        covered_weight += weight

    direct_score = _to_float(record.get("inspection_score"))
    if direct_score is not None:
        return InspectionComputation(
            score_out_of_10=min(max(direct_score, 0.0), 10.0),
            completeness_ratio=covered_weight / TOTAL_SECTION_WEIGHT,
            source="provided_score",
        )

    if covered_weight == 0:
        return InspectionComputation(
            score_out_of_10=None,
            completeness_ratio=0.0,
            source="missing",
        )

    derived_score = 10.0 * weighted_score / covered_weight
    source = "derived_from_full_sections" if covered_weight == TOTAL_SECTION_WEIGHT else "derived_from_partial_sections"
    return InspectionComputation(
        score_out_of_10=derived_score,
        completeness_ratio=covered_weight / TOTAL_SECTION_WEIGHT,
        source=source,
    )


def enrich_with_inspection_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for column in SECTION_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA
        df[column] = pd.to_numeric(df[column], errors="coerce").clip(lower=0, upper=100)

    computed_scores = []
    completeness_values = []
    sources = []
    for row in df.to_dict(orient="records"):
        result = compute_weighted_inspection_score(row)
        computed_scores.append(result.score_out_of_10)
        completeness_values.append(result.completeness_ratio)
        sources.append(result.source)

    if "inspection_score" not in df.columns:
        df["inspection_score"] = pd.Series(computed_scores, index=df.index)
    else:
        provided = pd.to_numeric(df["inspection_score"], errors="coerce")
        derived = pd.Series(computed_scores, index=df.index, dtype="float64")
        df["inspection_score"] = provided.fillna(derived)

    df["inspection_score"] = pd.to_numeric(df["inspection_score"], errors="coerce")
    df["inspection_completeness"] = pd.Series(completeness_values, index=df.index).astype(float)
    df["inspection_input_source"] = pd.Series(sources, index=df.index).astype(str)
    return df
