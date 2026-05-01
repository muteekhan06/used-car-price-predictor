from __future__ import annotations

import argparse
from random import Random

import pandas as pd

from inspection import SECTION_WEIGHTAGES


CATALOG = {
    "Toyota": {
        "Corolla": {"variants": ["XLi", "GLi", "Altis"]},
        "Yaris": {"variants": ["GLI CVT", "ATIV X"]},
    },
    "Honda": {
        "Civic": {"variants": ["VTi", "Oriel", "RS"]},
        "City": {"variants": ["1.2LS", "1.5 Aspire"]},
    },
    "Suzuki": {
        "Alto": {"variants": ["VX", "VXR", "VXL AGS"]},
        "Swift": {"variants": ["DLX", "GLX CVT"]},
    },
    "Hyundai": {
        "Elantra": {"variants": ["GLS", "Ultimate"]},
        "Tucson": {"variants": ["FWD", "AWD"]},
    },
    "Kia": {
        "Sportage": {"variants": ["Alpha", "FWD", "AWD"]},
        "Picanto": {"variants": ["MT", "AT"]},
    },
}

FUEL_TYPES = ["Petrol", "Hybrid", "Diesel"]
TRANSMISSIONS = ["Manual", "Automatic"]
REGISTERED_IN = ["Karachi", "Lahore", "Islamabad", "Rawalpindi", "Faisalabad"]
COLORS = ["White", "Black", "Silver", "Grey", "Blue", "Red", "Beige", "Green"]
ASSEMBLIES = ["Local", "Imported"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic used-car data.")
    parser.add_argument("--rows", type=int, default=2000, help="Number of rows to generate.")
    parser.add_argument(
        "--output",
        default="data/demo_used_cars.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = Random(args.seed)

    rows = []
    for _ in range(args.rows):
        make = rng.choice(list(CATALOG.keys()))
        model = rng.choice(list(CATALOG[make].keys()))
        spec = CATALOG[make][model]
        variant = rng.choice(spec["variants"])
        year = rng.randint(2012, 2025)
        car_age = 2026 - year
        mileage = max(0, int(rng.gauss(18000 * max(car_age, 1), 18000)))
        inspection_score = round(min(10, max(2.5, rng.gauss(7.8, 1.2))), 1)
        fuel_type = rng.choices(FUEL_TYPES, weights=[0.78, 0.15, 0.07], k=1)[0]
        transmission = rng.choices(TRANSMISSIONS, weights=[0.35, 0.65], k=1)[0]
        registered_in = rng.choice(REGISTERED_IN)
        color = rng.choices(
            COLORS,
            weights=[0.34, 0.14, 0.16, 0.12, 0.07, 0.06, 0.07, 0.04],
            k=1,
        )[0]
        assembly = rng.choices(ASSEMBLIES, weights=[0.72, 0.28], k=1)[0]
        section_values = {
            "section_interior_pct": round(min(100, max(55, rng.gauss(88, 9))), 1),
            "section_engine_transmission_clutch_pct": round(min(100, max(50, rng.gauss(91, 8))), 1),
            "section_electrical_electronics_pct": round(min(100, max(60, rng.gauss(92, 7))), 1),
            "section_body_frame_accident_pct": round(min(100, max(30, rng.gauss(94, 11))), 1),
            "section_exterior_body_pct": round(min(100, max(40, rng.gauss(83, 13))), 1),
            "section_ac_heater_pct": round(min(100, max(50, rng.gauss(90, 8))), 1),
            "section_brakes_pct": round(min(100, max(50, rng.gauss(90, 8))), 1),
            "section_suspension_steering_pct": round(min(100, max(35, rng.gauss(84, 12))), 1),
            "section_tyres_pct": round(min(100, max(30, rng.gauss(78, 15))), 1),
        }
        weighted_total = sum(
            (section_values[column] / 100.0) * weight
            for column, weight in SECTION_WEIGHTAGES.items()
        )
        inspection_score = round(10.0 * weighted_total / sum(SECTION_WEIGHTAGES.values()), 2)

        base_price = (
            1200000
            + (year - 2012) * 190000
            + inspection_score * 210000
        )
        mileage_penalty = mileage * rng.uniform(7.5, 12.5)
        registered_adjustment = {
            "Karachi": 90000,
            "Lahore": 70000,
            "Islamabad": 110000,
            "Rawalpindi": 30000,
            "Faisalabad": -20000,
        }[registered_in]
        transmission_adjustment = {"Manual": -90000, "Automatic": 120000}[transmission]
        fuel_adjustment = {"Petrol": 0, "Hybrid": 300000, "Diesel": 110000}[fuel_type]
        assembly_adjustment = {"Local": 0, "Imported": 450000}[assembly]
        color_adjustment = {
            "White": 30000,
            "Black": 70000,
            "Silver": 45000,
            "Grey": 35000,
            "Blue": 10000,
            "Red": -10000,
            "Beige": -15000,
            "Green": -30000,
        }[color]
        noise = rng.gauss(0, 135000)

        price = max(
            450000,
            int(
                base_price
                - mileage_penalty
                + registered_adjustment
                + transmission_adjustment
                + fuel_adjustment
                + assembly_adjustment
                + color_adjustment
                + noise
            ),
        )

        rows.append(
            {
                "make": make,
                "model": model,
                "variant": variant,
                "year": year,
                "mileage": mileage,
                "transmission": transmission,
                "fuel_type": fuel_type,
                "inspection_score": inspection_score,
                "registered_in": registered_in,
                "color": color,
                "assembly": assembly,
                "price": price,
                "synthetic_data_only": True,
                **section_values,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)

    print(f"Saved {len(df)} synthetic rows to {args.output}")
    print("Use this only to test the pipeline, not to judge real-world model accuracy.")


if __name__ == "__main__":
    main()
