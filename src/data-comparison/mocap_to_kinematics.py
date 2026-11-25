#!/usr/bin/env python3
"""Convert a Qualisys-style mocap CSV export into bicycle kinematics CSV.

Extracts exactly two rigid bodies:
    Bike        -> rear frame
    front-frame -> front frame

The output uses the 13-column schema of 00000001_kinematics.csv.
Unavailable IMU/wheel channels are left empty. LWS_ANGLE is derived from the
relative front/rear quaternion using swing-twist decomposition about a
user-specified steering-axis vector expressed in the rear-frame coordinates.

Install: pip install numpy pandas scipy
Example:
  python mocap_to_kinematics.py --input "Take.csv" --output "mocap.csv"
  python mocap_to_kinematics.py --input "Take.csv" --output "mocap.csv" \
      --steer-axis 0,0,1
"""

from __future__ import annotations
import argparse
import csv
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

COLUMNS = [
    "timestamps", "gyro_z", "gyro_y", "gyro_x",
    "accel_z", "accel_y", "accel_x",
    "yaw", "pitch", "roll",
    "ws_rear", "LWS_ANGLE", "LWS_SPEED",
]


def read_export(path: Path):
    # csv.reader handles the blank second line and variable-width metadata row.
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    if len(rows) < 8:
        raise ValueError("Unexpected mocap export: not enough header/data rows.")
    width = max(map(len, rows))
    rows = [r + [""] * (width - len(r)) for r in rows]
    return rows


def body_columns(rows, body):
    type_row, name_row = rows[2], rows[3]
    prop_row, comp_row = rows[5], rows[6]
    found = {}
    for prop, components in {
        "Rotation": ("X", "Y", "Z", "W"),
        "Position": ("X", "Y", "Z"),
    }.items():
        indices = {}
        for comp in components:
            hits = [
                i for i in range(len(type_row))
                if type_row[i].strip() == "Rigid Body"
                and name_row[i].strip() == body
                and prop_row[i].strip().lower() == prop.lower()
                and comp_row[i].strip().upper() == comp
            ]
            if not hits:
                raise ValueError(f"Missing {body} {prop} {comp} column.")
            indices[comp] = hits[0]
        found[prop] = indices
    return found


def parse_capture_start(rows):
    meta = rows[0]
    for i in range(len(meta) - 1):
        if meta[i].strip().lower() == "capture start time":
            value = meta[i + 1].strip()
            for fmt in ("%Y-%m-%d %I.%M.%S.%f %p", "%Y-%m-%d %I.%M.%S %p"):
                try:
                    return datetime.strptime(value, fmt)
                except ValueError:
                    pass
    return None


def twist_angle_about_axis(q_xyzw, axis):
    """Signed swing-twist angle around axis; axis is in rear-frame coordinates."""
    axis = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm == 0:
        raise ValueError("Steering axis cannot be the zero vector.")
    axis /= norm
    qv = q_xyzw[:, :3]
    qw = q_xyzw[:, 3]
    projected = np.sum(qv * axis[None, :], axis=1)
    # Normalize the twist quaternion before extracting its signed angle.
    twist_norm = np.hypot(qw, projected)
    bad = twist_norm < 1e-12
    angle = 2.0 * np.arctan2(projected, qw)
    angle = (angle + np.pi) % (2 * np.pi) - np.pi
    angle[bad] = np.nan
    return angle


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--rear-body", default="Bike")
    ap.add_argument("--front-body", default="front-frame")
    ap.add_argument("--steer-axis", default="0,0,1",
                    help="Steering-axis vector in rear-frame coordinates; default 0,0,1.")
    ap.add_argument("--euler-seq", default="xyz",
                    help="SciPy Euler sequence used for yaw/pitch/roll (default xyz).")
    ap.add_argument("--timestamp-mode", choices=["elapsed", "capture"], default="elapsed",
                    help="'elapsed' writes elapsed seconds; 'capture' writes capture wall-clock time.")
    ap.add_argument("--steer-sign", type=float, choices=[-1.0, 1.0], default=1.0)
    args = ap.parse_args()

    rows = read_export(args.input)
    rear_cols = body_columns(rows, args.rear_body)
    front_cols = body_columns(rows, args.front_body)

    numeric = []
    for row in rows[7:]:
        try:
            frame = int(float(row[0]))
            t = float(row[1])
        except (ValueError, IndexError):
            continue
        numeric.append((frame, t, row))
    if not numeric:
        raise ValueError("No numeric mocap frame rows found.")

    frames = np.array([x[0] for x in numeric])
    times = np.array([x[1] for x in numeric], dtype=float)
    elapsed = times - times[0]

    def extract(cols, prop, keys):
        return np.column_stack([
            [float(row[cols[prop][key]]) if row[cols[prop][key]].strip() else np.nan
             for _, _, row in numeric]
            for key in keys
        ])

    q_rear = extract(rear_cols, "Rotation", "XYZW")
    q_front = extract(front_cols, "Rotation", "XYZW")
    p_rear = extract(rear_cols, "Position", "XYZ")
    p_front = extract(front_cols, "Position", "XYZ")

    # Interpolate invalid quaternions is intentionally not attempted: reject them.
    valid = (np.isfinite(q_rear).all(axis=1) & np.isfinite(q_front).all(axis=1)
             & (np.linalg.norm(q_rear, axis=1) > 1e-8)
             & (np.linalg.norm(q_front, axis=1) > 1e-8))
    if not valid.all():
        raise ValueError(f"{(~valid).sum()} frames have invalid rigid-body quaternions. "
                         "Clean/re-export these frames before processing.")
    q_rear /= np.linalg.norm(q_rear, axis=1)[:, None]
    q_front /= np.linalg.norm(q_front, axis=1)[:, None]

    rear_rot = Rotation.from_quat(q_rear)
    front_rot = Rotation.from_quat(q_front)
    seq = args.euler_seq.lower()
    if seq != "xyz":
        raise ValueError("This script currently maps xyz Euler angles to roll/pitch/yaw. "
                         "Use --euler-seq xyz or edit the mapping explicitly.")
    eul = rear_rot.as_euler(seq, degrees=False)
    roll, pitch, yaw = eul[:, 0], eul[:, 1], eul[:, 2]

    # Relative front orientation in rear-frame coordinates:
    # q_rel = inverse(q_rear) * q_front.
    q_rel = (rear_rot.inv() * front_rot).as_quat()
    axis = [float(v) for v in args.steer_axis.split(",")]
    if len(axis) != 3:
        raise ValueError("--steer-axis must contain three comma-separated numbers.")
    steer_rad = args.steer_sign * twist_angle_about_axis(q_rel, axis)
    steer_deg = np.degrees(steer_rad)

    capture_start = parse_capture_start(rows)
    if args.timestamp_mode == "capture" and capture_start is not None:
        stamps = (pd.Timestamp(capture_start) +
                  pd.to_timedelta(elapsed, unit="s")).astype(str)
    else:
        # Relative elapsed-time timestamps are portable and unambiguous.
        stamps = [f"{v:.9f}" for v in elapsed]

    out = pd.DataFrame({
        "timestamps": stamps,
        "gyro_z": np.nan, "gyro_y": np.nan, "gyro_x": np.nan,
        "accel_z": np.nan, "accel_y": np.nan, "accel_x": np.nan,
        "yaw": yaw, "pitch": pitch, "roll": roll,
        "ws_rear": np.nan,
        "LWS_ANGLE": steer_deg,
        "LWS_SPEED": np.nan,
    }, columns=COLUMNS)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    # Full pose sidecar retains the two rigid-body 3D translations/quaternions.
    pose = pd.DataFrame({
        "frame": frames, "time_s": elapsed,
        **{f"rear_pos_{a}_m": p_rear[:, i] for i, a in enumerate("xyz")},
        **{f"rear_q_{a}": q_rear[:, i] for i, a in enumerate("xyzw")},
        **{f"front_pos_{a}_m": p_front[:, i] for i, a in enumerate("xyz")},
        **{f"front_q_{a}": q_front[:, i] for i, a in enumerate("xyzw")},
        "steer_angle_rad": steer_rad, "steer_angle_deg": steer_deg,
    })
    pose.to_csv(args.output.with_name(args.output.stem + "_two_body_poses.csv"), index=False)

    print(f"Wrote {len(out)} samples: {args.output}")
    print(f"Steering angle: swing-twist about rear-frame axis {axis}; "
          f"sign={args.steer_sign:+g}; output degrees in LWS_ANGLE.")
    if capture_start:
        print(f"Mocap capture start metadata: {capture_start}")
    print("Check steering-axis definition/sign and coordinate conventions before "
          "using absolute steering-angle errors.")


if __name__ == "__main__":
    main()
