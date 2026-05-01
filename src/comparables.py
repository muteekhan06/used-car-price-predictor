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
    weighted_price: float | None
    median_price: float | None
    p25_price: float | None
    p75_price: float | None
    iqr_ratio: float | None
    exact_variant_matches: int
    exact_city_matches: int
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

    def _candidate_pool(self, record: dict) -> pd.DataFrame:
        make_model_key = f"{record['make']}||{record['model']}"
        pool = self.frame[self.frame["make_model_key"] == make_model_key].copy()
        if len(pool) < 15:
            pool = self.frame[self.frame["make"] == record["make"]].copy()
        if len(pool) < 15:
            pool = self.frame.copy()
        return pool

    def summarize(self, record: dict, top_k: int = 20) -> ComparableSummary:
        pool = self._candidate_pool(record)
        if len(pool) == 0:
            return ComparableSummary(0, None, None, None, None, None, 0, 0, [])

        score = pd.Series(0.0, index=pool.index, dtype="float64")
        score += np.abs(pool["year"] - record["year"]) * 0.7
        score += np.abs(np.log1p(pool["mileage"]) - math.log1p(record["mileage"])) * 8.0
        score += (pool["variant"] != record["variant"]).astype(float) * 2.5
        score += (pool["transmission"] != record["transmission"]).astype(float) * 1.5
        score += (pool["fuel_type"] != record["fuel_type"]).astype(float) * 1.2
        score += (pool["assembly"] != record["assembly"]).astype(float) * 1.3
        score += (pool["registered_in"] != record["registered_in"]).astype(float) * 0.8
        score += (pool["body_type"] != record.get("body_type", "Unknown")).astype(float) * 0.5

        engine_target = record.get("engine_capacity_cc")
        if engine_target is not None and not pd.isna(engine_target):
            pool_engine = pd.to_numeric(pool["engine_capacity_cc"], errors="coerce")
            score += (np.abs(pool_engine - float(engine_target)).fillna(0) / 1000.0) * 0.5

        inspection_target = record.get("inspection_score")
        if inspection_target is not None and not pd.isna(inspection_target):
            pool_score = pd.to_numeric(pool["inspection_score"], errors="coerce")
            score += np.abs(pool_score.fillna(inspection_target) - float(inspection_target)) * 0.4

        pool = pool.assign(score=score).sort_values("score").head(top_k).copy()
        if len(pool) == 0:
            return ComparableSummary(0, None, None, None, None, None, 0, 0, [])

        weights = np.exp(-pool["score"].to_numpy(dtype=float))
        weights = np.where(weights <= 0, 1e-6, weights)
        prices = pool["price"].to_numpy(dtype=float)

        weighted_price = float(np.average(prices, weights=weights))
        median_price = float(np.median(prices))
        p25_price = float(np.quantile(prices, 0.25))
        p75_price = float(np.quantile(prices, 0.75))
        iqr_ratio = float((p75_price - p25_price) / max(median_price, 1.0))
        exact_variant_matches = int((pool["variant"] == record["variant"]).sum())
        exact_city_matches = int((pool["registered_in"] == record["registered_in"]).sum())

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
        comps = pool[display_columns].head(5).to_dict(orient="records")
        return ComparableSummary(
            comp_count=int(len(pool)),
            weighted_price=weighted_price,
            median_price=median_price,
            p25_price=p25_price,
            p75_price=p75_price,
            iqr_ratio=iqr_ratio,
            exact_variant_matches=exact_variant_matches,
            exact_city_matches=exact_city_matches,
            comps=comps,
        )
