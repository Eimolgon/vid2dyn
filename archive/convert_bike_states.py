#!/usr/bin/env python3
"""
Convert bike_states.csv into the column names and units expected by animation.py.

Usage:
    python convert_bike_states.py bike_states.csv bike_states_animation.csv

Assumptions:
- Positions in the source CSV are in metres; animation.py expects millimetres.
- Angles in the source CSV are in degrees; animation.py expects radians.
- The existing sign conventions remain in animation.py:
      y = -yr, z = -zr, q3 = -psi, q4 = phi,
      q5 = -theta, q7 = -delta
"""
import argparse
import numpy as np
import pandas as pd


def convert(input_csv, output_csv, use_smoothed=True):
    df = pd.read_csv(input_csv)

    suffix = "" if use_smoothed else "_raw"
    required = [
        f"roll{suffix}_deg",
        f"pitch{suffix}_deg",
        f"yaw{suffix}_deg",
        f"steer{suffix}_deg",
        f"x{suffix}_m",
        f"y{suffix}_m",
        f"z{suffix}_m",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    # Keep the animation's legacy column names and units.
    out = pd.DataFrame({
        "xr": df[f"x{suffix}_m"] * 1000.0,
        "yr": df[f"y{suffix}_m"] * 1000.0,
        "zr": df[f"z{suffix}_m"] * 1000.0,
        "psi": np.deg2rad(df[f"yaw{suffix}_deg"]),
        "phi": np.deg2rad(df[f"roll{suffix}_deg"]),
        "theta": np.deg2rad(df[f"pitch{suffix}_deg"]),
        "delta": np.deg2rad(df[f"steer{suffix}_deg"]),
    })

    # Drop invalid rows if the CSV contains them.
    if "valid" in df.columns:
        out = out.loc[df["valid"].astype(bool)].reset_index(drop=True)

    out.to_csv(output_csv, index=False)
    print(f"Converted {len(out)} rows -> {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", help="Input state-estimation CSV")
    parser.add_argument("output_csv", help="Output CSV for animation.py")
    parser.add_argument(
        "--raw", action="store_true",
        help="Use *_raw_deg and *_raw_m columns instead of smoothed columns",
    )
    args = parser.parse_args()
    convert(args.input_csv, args.output_csv, use_smoothed=not args.raw)
