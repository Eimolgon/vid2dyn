#!/usr/bin/env python3
"""
Data Converter for OpenLoopBalanceControl
Translates GoPro Kalman Filter outputs into time-series trajectories for bicycle control optimization.
"""

import argparse
import os
import numpy as np
import pandas as pd


def convert_kf_data(
    input_path: str,
    output_path: str,
    fps: float = 30.0,
    filter_valid_only: bool = True,
    angles_in_radians: bool = True
) -> pd.DataFrame:
    """
    Converts raw frame-based GoPro Kalman Filter CSV to time-indexed kinematics.

    Parameters:
    -----------
    input_path : str
        Path to input CSV file (e.g. states_kf_gopro_test_1_1.csv)
    output_path : str
        Path to save converted output CSV file
    fps : float
        Frames per second of the source video (default: 30.0)
    filter_valid_only : bool
        Whether to exclude rows marked as valid == 0 (default: True)
    angles_in_radians : bool
        Convert angles and angular rates from degrees to radians (default: True)

    Returns:
    --------
    pd.DataFrame
        Transformed DataFrame
    """
    print(f"Loading raw dataset from: {input_path}")
    df = pd.read_csv(input_path)

    # Filter invalid frames if flag is set
    if filter_valid_only and 'valid' in df.columns:
        initial_count = len(df)
        df = df[df['valid'] == 1].copy()
        print(f"Filtered out {initial_count - len(df)} invalid frames. Retaining {len(df)} rows.")

    # 1. Generate Continuous Timestamp (seconds)
    # Zero-index time relative to first valid frame
    first_frame = df['frame'].iloc[0]
    df['time'] = (df['frame'] - first_frame) / fps

    # 2. Angular Conversions (Degrees to Radians)
    deg_to_rad = np.pi / 180.0 if angles_in_radians else 1.0
    unit_suffix = "_rad" if angles_in_radians else "_deg"
    rate_suffix = "_radps" if angles_in_radians else "_degps"

    df[f'roll{unit_suffix}'] = df['roll_deg'] * deg_to_rad
    df[f'pitch{unit_suffix}'] = df['pitch_deg'] * deg_to_rad
    df[f'yaw{unit_suffix}'] = df['yaw_deg'] * deg_to_rad
    df[f'steer{unit_suffix}'] = df['steer_deg'] * deg_to_rad

    df[f'steer_rate{rate_suffix}'] = df['steer_rate_degps'] * deg_to_rad
    df[f'yaw_rate{rate_suffix}'] = df['yaw_rate_degps'] * deg_to_rad

    # 3. Derive Roll Rate (phi_dot) via central numerical differentiation
    # as phi_dot is omitted in raw output but required in full state space models
    dt = 1.0 / fps
    df[f'roll_rate{rate_suffix}'] = np.gradient(df[f'roll{unit_suffix}'], dt)

    # 4. Standardizing Column Naming Convention for Bicycle Kinetics
    # Whipple/Carvallo standard: phi=roll, delta=steer, psi=yaw, v=speed
    column_mapping = {
        'time': 'time',
        'x_m': 'x',
        'y_m': 'y',
        'z_m': 'z',
        'speed_mps': 'v',
        f'roll{unit_suffix}': 'phi',
        f'steer{unit_suffix}': 'delta',
        f'yaw{unit_suffix}': 'psi',
        f'pitch{unit_suffix}': 'theta',
        f'roll_rate{rate_suffix}': 'phi_dot',
        f'steer_rate{rate_suffix}': 'delta_dot',
        f'yaw_rate{rate_suffix}': 'psi_dot',
    }

    out_df = df[list(column_mapping.keys())].rename(columns=column_mapping)

    # Save formatted dataset
    out_df.to_csv(output_path, index=False)
    print(f"Successfully exported converted dataset to: {output_path}")
    return out_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert GoPro KF state CSV into OpenLoopBalanceControl trajectory format.")
    parser.add_argument("--input", type=str, default="states_kf_gopro_test_1_1.csv", help="Input CSV filename")
    parser.add_argument("--output", type=str, default="trajectory_input.csv", help="Output CSV filename")
    parser.add_argument("--fps", type=float, default=30.0, help="Source video frame rate (e.g., 30.0, 59.94, 60.0)")
    parser.add_argument("--keep-degrees", action="store_true", help="Keep angular quantities in degrees instead of converting to radians")

    args = parser.parse_args()

    if os.path.exists(args.input):
        convert_kf_data(
            input_path=args.input,
            output_path=args.output,
            fps=args.fps,
            angles_in_radians=not args.keep_degrees
        )
    else:
        print(f"Error: Input file '{args.input}' not found.")