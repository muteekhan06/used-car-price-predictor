from __future__ import annotations

import argparse
import json
from pathlib import Path

from features import prepare_user_input
from serving import load_artifacts, predict_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict production-grade used-car listing price.")
    parser.add_argument("--model-dir", default="artifacts", help="Directory containing trained artifacts.")
    parser.add_argument("--input-json", help="Single-record JSON string.")
    parser.add_argument("--input-file", help="Path to a JSON file containing one record.")
    return parser.parse_args()


def load_input_record(args: argparse.Namespace) -> dict:
    if args.input_json:
        return json.loads(args.input_json)
    if args.input_file:
        with open(args.input_file, "r", encoding="utf-8") as file:
            return json.load(file)
    raise ValueError("Provide either --input-json or --input-file.")


def predict_record(record: dict, model_dir: str | Path = "artifacts") -> dict:
    prepared = prepare_user_input(record)
    artifacts = load_artifacts(model_dir)
    result = predict_frame(prepared, artifacts).iloc[0].to_dict()

    comparable_reference_price = result.get("comparable_reference_price")
    if comparable_reference_price is not None:
        comparable_reference_price = round(float(comparable_reference_price), 2)

    return {
        "predicted_price": round(float(result["predicted_price"]), 2),
        "price_range_low": round(float(result["price_range_low"]), 2),
        "price_range_high": round(float(result["price_range_high"]), 2),
        "prediction_mode": result["prediction_mode"],
        "confidence_score": round(float(result["confidence_score"]), 4),
        "catalog_source_rows": int(result.get("catalog_source_rows", 0)),
        "support_tier": result.get("support_tier", "unknown"),
        "anchor_price": round(float(result["anchor_price"]), 2),
        "condition_adjusted_price": round(float(result["condition_adjusted_price"]), 2),
        "comparable_reference_price": comparable_reference_price,
        "comparable_count": int(result["comparable_count"]),
        "comp_weight": round(float(result["comp_weight"]), 4),
        "comparables": result["comparables"],
    }


def main() -> None:
    args = parse_args()
    record = load_input_record(args)
    result = predict_record(record, model_dir=args.model_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
