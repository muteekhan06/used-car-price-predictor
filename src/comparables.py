from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


COMPARABLE_COLUMNS = [
    "id",
    "make",
    "model",
    "variant",
    "year",
    "mileage",
    "transmission",
    "fuel_type",
    "registered_in",
    "color",
    "assembly",
    "body_type",
    "engine_capacity_cc",
    "inspection_score",
    "price",
    "listing_url",
]


@dataclass
class ComparableSummary:
    comp_count: int
    usable_comp_count: int
    weighted_price: float | None
    median_price: float | None
    p25_price: float | None
    p75_price: float | None
    iqr_ratio: float | None
    exact_variant_matches: int
    exact_city_matches: int
    exact_model_rows: int
    exact_variant_rows: int
    comp_quality_passed: bool
    blend_mode_hint: str
    comps: list[dict]


class ComparableRetriever:
    def __init__(self, frame: pd.DataFrame) -> None:
        base = frame[COMPARABLE_COLUMNS].copy()
        base["make_model_key"] = base["make"] + "||" + base["model"]
        self.frame = base.reset_index(drop=True)

    def save(self, path: str | Path) -> None:
        self.frame.to_pickle(path)

    @classmethod
    def load(cls, path: str | Path) -> "ComparableRetriever":
        frame = pd.read_pickle(path)
        obj = cls.__new__(cls)
        obj.frame = frame
        return obj

    def _candidate_pool(self, record: dict) -> tuple[pd.DataFrame, int, int]:
        make_model_key = f"{record['make']}||{record['model']}"
        exact_model = self.frame[self.frame["make_model_key"] == make_model_key].copy()
        exact_variant_rows = int((exact_model["variant"] == record["variant"]).sum()) if len(exact_model) else 0
        return exact_model, int(len(exact_model)), exact_variant_rows

    def summarize(self, record: dict, top_k: int = 20) -> ComparableSummary:
        pool, exact_model_rows, exact_variant_rows = self._candidate_pool(record)
        if len(pool) == 0:
            return ComparableSummary(0, 0, None, None, None, None, None, 0, 0, exact_model_rows, exact_variant_rows, False, "anchor_only", [])

        score = pd.Series(0.0, index=pool.index, dtype="float64")
        score += np.abs(pool["year"] - record["year"]) * 0.9
        score += np.abs(np.log1p(pool["mileage"]) - math.log1p(record["mileage"])) * 8.0
        score += (pool["variant"] != record["variant"]).astype(float) * 2.8
        score += (pool["transmission"] != record["transmission"]).astype(float) * 1.8
        score += (pool["fuel_type"] != record["fuel_type"]).astype(float) * 1.4
        score += (pool["assembly"] != record["assembly"]).astype(float) * 1.6
        score += (pool["registered_in"] != record["registered_in"]).astype(float) * 0.8
        score += (pool["body_type"] != record.get("body_type", "Unknown")).astype(float) * 0.4

        engine_target = record.get("engine_capacity_cc")
        if engine_target is not None and not pd.isna(engine_target):
            pool_engine = pd.to_numeric(pool["engine_capacity_cc"], errors="coerce")
            score += (np.abs(pool_engine - float(engine_target)).fillna(0) / 1000.0) * 0.5

        inspection_target = record.get("inspection_score")
        if inspection_target is not None and not pd.isna(inspection_target):
            pool_score = pd.to_numeric(pool["inspection_score"], errors="coerce")
            score += np.abs(pool_score.fillna(inspection_target) - float(inspection_target)) * 0.25

        pool = pool.assign(score=score).sort_values("score").head(top_k).copy()
        if len(pool) == 0:
            return ComparableSummary(0, 0, None, None, None, None, None, 0, 0, exact_model_rows, exact_variant_rows, False, "anchor_only", [])

        usable = pool[pool["score"] <= 6.0].copy()
        if len(usable) < 5:
            usable = pool.head(min(len(pool), 5)).copy()

        prices = usable["price"].to_numpy(dtype=float)
        weights = np.exp(-usable["score"].to_numpy(dtype=float))
        weights = np.where(weights <= 0, 1e-6, weights)

        weighted_price = float(np.average(prices, weights=weights))
        median_price = float(np.median(prices))
        p25_price = float(np.quantile(prices, 0.25))
        p75_price = float(np.quantile(prices, 0.75))
        iqr_ratio = float((p75_price - p25_price) / max(median_price, 1.0))
        exact_variant_matches = int((usable["variant"] == record["variant"]).sum())
        exact_city_matches = int((usable["registered_in"] == record["registered_in"]).sum())
        comp_quality_passed = bool(exact_model_rows >= 8 and len(usable) >= 5 and iqr_ratio <= 0.35 and float(usable["score"].median()) <= 4.5)
        blend_mode_hint = "anchor_plus_comparables" if comp_quality_passed else "anchor_only"

        display_columns = [
            "make",
            "model",
            "variant",
            "year",
            "mileage",
            "price",
            "registered_in",
            "assembly",
            "score",
            "listing_url",
        ]
        comps = usable[display_columns].head(5).to_dict(orient="records")
        return ComparableSummary(
            comp_count=int(len(pool)),
            usable_comp_count=int(len(usable)),
            weighted_price=weighted_price,
            median_price=median_price,
            p25_price=p25_price,
            p75_price=p75_price,
            iqr_ratio=iqr_ratio,
            exact_variant_matches=exact_variant_matches,
            exact_city_matches=exact_city_matches,
            exact_model_rows=exact_model_rows,
            exact_variant_rows=exact_variant_rows,
            comp_quality_passed=comp_quality_passed,
            blend_mode_hint=blend_mode_hint,
            comps=comps,
        )
