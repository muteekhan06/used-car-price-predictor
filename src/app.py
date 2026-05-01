from __future__ import annotations

import argparse

from predict import predict_record
from inspection import SECTION_COLUMNS, SECTION_LABELS, compute_weighted_inspection_score


FIELDS = [
    ("make", str),
    ("model", str),
    ("variant", str),
    ("year", int),
    ("transmission", str),
    ("fuel_type", str),
    ("mileage", float),
    ("registered_in", str),
    ("color", str),
    ("assembly", str),
    ("body_type", str),
    ("engine_capacity_cc", float),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive used-car calculator.")
    parser.add_argument(
        "--model-dir",
        default="artifacts",
        help="Directory containing trained model files.",
    )
    return parser.parse_args()


def prompt_user() -> dict:
    print("Enter the car details below.")
    print("You can either enter inspection_score directly or answer weighted inspection sections.")

    record = {}
    for field_name, caster in FIELDS:
        raw_value = input(f"{field_name}: ").strip()
        record[field_name] = caster(raw_value)

    direct_score = input("inspection_score (0-10, leave blank if unavailable): ").strip()
    if direct_score:
        record["inspection_score"] = float(direct_score)
    else:
        print("Enter inspection section percentages from 0 to 100.")
        for column in SECTION_COLUMNS:
            value = input(f"{SECTION_LABELS[column]} %: ").strip()
            if value:
                record[column] = float(value)

        derived = compute_weighted_inspection_score(record)
        if derived.score_out_of_10 is not None:
            print(
                "Calculated inspection_score: "
                f"{derived.score_out_of_10:.2f}/10 "
                f"(completeness {derived.completeness_ratio:.0%})"
            )
    return record


def main() -> None:
    args = parse_args()
    record = prompt_user()
    result = predict_record(record, model_dir=args.model_dir)

    print()
    print("Predicted price:")
    print(f"  {result['predicted_price']:.2f}")
    print("Likely price range:")
    print(f"  {result['price_range_low']:.2f} to {result['price_range_high']:.2f}")
    print(f"Prediction mode: {result['prediction_mode']}")
    print(f"Confidence: {result['confidence_score']:.2f}")


if __name__ == "__main__":
    main()
