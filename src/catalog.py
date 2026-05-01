from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def build_catalog(frame: pd.DataFrame) -> dict:
    working = frame.copy()
    working["year"] = pd.to_numeric(working["year"], errors="coerce").astype("Int64")
    working["engine_capacity_cc"] = pd.to_numeric(working["engine_capacity_cc"], errors="coerce")

    catalog: dict[str, dict] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "makes": sorted(working["make"].dropna().astype(str).unique().tolist()),
        "tree": {},
    }

    for (make, model, year, variant), group in working.groupby(["make", "model", "year", "variant"], dropna=True):
        make_node = catalog["tree"].setdefault(str(make), {"models": {}})
        model_node = make_node["models"].setdefault(str(model), {"years": {}})
        year_node = model_node["years"].setdefault(str(int(year)), {"variants": {}})
        year_node["variants"][str(variant)] = {
            "spec": {
                "transmission": group["transmission"].mode().iloc[0] if group["transmission"].notna().any() else None,
                "fuel_type": group["fuel_type"].mode().iloc[0] if group["fuel_type"].notna().any() else None,
                "assembly": group["assembly"].mode().iloc[0] if group["assembly"].notna().any() else None,
                "body_type": group["body_type"].mode().iloc[0] if group["body_type"].notna().any() else None,
                "engine_capacity_cc": int(round(float(group["engine_capacity_cc"].median())))
                if group["engine_capacity_cc"].notna().any()
                else None,
            },
            "registered_in": sorted(group["registered_in"].dropna().astype(str).unique().tolist()),
            "color": sorted(group["color"].dropna().astype(str).unique().tolist()),
            "sample_count": int(len(group)),
        }

    return catalog


def save_catalog(catalog: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(catalog, file, indent=2)


def load_catalog(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def option_payload(catalog: dict, make: str | None = None, model: str | None = None, year: int | None = None, variant: str | None = None) -> dict:
    tree = catalog.get("tree", {})
    make_node = tree.get(make, {}) if make else {}
    model_node = make_node.get("models", {}).get(model, {}) if make and model else {}
    year_node = model_node.get("years", {}).get(str(year), {}) if make and model and year is not None else {}
    variant_node = year_node.get("variants", {}).get(variant, {}) if make and model and year is not None and variant else {}

    models = sorted(make_node.get("models", {}).keys()) if make else []
    years = sorted(int(value) for value in model_node.get("years", {}).keys()) if make and model else []
    variants = sorted(year_node.get("variants", {}).keys()) if make and model and year is not None else []

    location_source = variant_node if variant_node else {}

    return {
        "makes": catalog.get("makes", []),
        "models": models,
        "years": years,
        "variants": variants,
        "registered_in": location_source.get("registered_in", []),
        "color": location_source.get("color", []),
    }


def spec_payload(catalog: dict, make: str, model: str, year: int, variant: str) -> dict | None:
    tree = catalog.get("tree", {})
    try:
        variant_node = tree[make]["models"][model]["years"][str(year)]["variants"][variant]
    except KeyError:
        return None

    return {
        "spec": variant_node.get("spec", {}),
        "available_registered_in": variant_node.get("registered_in", []),
        "available_colors": variant_node.get("color", []),
        "source_rows": int(variant_node.get("sample_count", 0)),
    }
