#!/usr/bin/env python3
"""Compare two or more bicycle-kinematics CSV files.

All input files should use the same column names/schema. The first file is the
default reference; every other file is compared against it. Or specify
--reference to select a reference file from the supplied inputs.

Install: pip install numpy pandas scipy matplotlib

Example:
  python compare_kinematics.py --files mocap.csv imu.csv third_source.csv \
      --outdir comparison --max-lag 60

Manual lag for a particular source (source time = reference time + lag):
  python compare_kinematics.py --files reference.csv other.csv \
      --lag other.csv=0.42

Expected units:
  yaw/pitch/roll: radians
  LWS_ANGLE: degrees
  gyro_*: rad/s (metrics use ordinary differences)
  accel_*: m/s^2 (metrics use ordinary differences)
  ws_rear and LWS_SPEED: same units across files

For angular channels, errors are wrapped to the shortest signed angular
difference. Steering angle is treated as degrees.
"""

from __future__ import annotations
import argparse
import itertools
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import correlate

ANGLE_RAD = {"yaw", "pitch", "roll"}
ANGLE_DEG = {"LWS_ANGLE"}
CANDIDATE_SIGNALS = ["LWS_ANGLE", "yaw", "roll", "pitch"]


def read_kinematics(path: Path):
    df = pd.read_csv(path, sep=None, engine="python")
    if "timestamps" not in df:
        raise ValueError(f"{path}: missing required 'timestamps' column.")
    # Timestamp may be ISO datetime or elapsed seconds.
    ts_num = pd.to_numeric(df["timestamps"], errors="coerce")
    if ts_num.notna().mean() > 0.95:
        t = ts_num.to_numpy(float)
    else:
        dt = pd.to_datetime(df["timestamps"], utc=True, errors="coerce")
        if dt.notna().sum() < 2:
            raise ValueError(f"{path}: timestamps are neither numeric nor parseable datetimes.")
        t = (dt - dt.dropna().iloc[0]).dt.total_seconds().to_numpy(float)
    df = df.copy()
    df["_time_s"] = t
    for c in df.columns:
        if c not in ("timestamps", "_time_s"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["_time_s"]).sort_values("_time_s")
    df = df.drop_duplicates("_time_s").reset_index(drop=True)
    df["_time_s"] -= df["_time_s"].iloc[0]
    return df


def wrap_difference(a, b, period):
    return (a - b + period / 2) % period - period / 2


def interpolate(t, y, grid):
    good = np.isfinite(t) & np.isfinite(y)
    if good.sum() < 5:
        return np.full(grid.shape, np.nan)
    order = np.argsort(t[good])
    tx, yy = t[good][order], y[good][order]
    tx, unique_idx = np.unique(tx, return_index=True)
    yy = yy[unique_idx]
    out = np.interp(grid, tx, yy)
    out[(grid < tx[0]) | (grid > tx[-1])] = np.nan
    return out


def lag_score(ref_t, ref_y, src_t, src_y, max_lag, hz):
    """Find lag maximizing absolute correlation of derivatives.

    Returns lag where source_time = reference_time + lag.
    """
    dt = 1.0 / hz
    # Use a shared relative-time grid extending across both recordings and lag.
    end = max(ref_t[-1], src_t[-1]) + max_lag
    grid = np.arange(-max_lag, end + dt, dt)
    yr = interpolate(ref_t, ref_y, grid)
    ys = interpolate(src_t, src_y, grid)
    # Derivatives suppress constant sensor offsets and focus on motion signature.
    dr = np.gradient(yr, dt)
    ds = np.gradient(ys, dt)
    best = (-np.inf, 0.0)
    max_n = int(round(max_lag * hz))
    for lag_n in range(-max_n, max_n + 1):
        # source at t+lag is compared with reference at t.
        if lag_n >= 0:
            a, b = dr[:-lag_n or None], ds[lag_n:]
        else:
            a, b = dr[-lag_n:], ds[:lag_n]
        valid = np.isfinite(a) & np.isfinite(b)
        if valid.sum() < max(20, int(5 * hz)):
            continue
        av, bv = a[valid], b[valid]
        av = av - av.mean()
        bv = bv - bv.mean()
        denom = np.linalg.norm(av) * np.linalg.norm(bv)
        score = abs(np.dot(av, bv) / denom) if denom > 0 else 0.0
        if score > best[0]:
            best = (float(score), lag_n * dt)
    return best[1], best[0]


def calculate_metrics(ref, src, ref_name, src_name, lag, hz, outdir):
    dt = 1.0 / hz
    common = sorted(set(ref.columns) & set(src.columns) - {"timestamps", "_time_s"})
    numeric_common = [c for c in common
                      if pd.api.types.is_numeric_dtype(ref[c])
                      and pd.api.types.is_numeric_dtype(src[c])]
    # Align source timeline: source_time = reference_time + lag.
    src_time_shifted = src["_time_s"].to_numpy(float) - lag
    start = max(ref["_time_s"].min(), src_time_shifted.min())
    end = min(ref["_time_s"].max(), src_time_shifted.max())
    if end <= start:
        raise ValueError(f"No temporal overlap for {src_name} at lag {lag:.4f}s.")
    grid = np.arange(start, end + 0.25 * dt, dt)
    aligned = pd.DataFrame({"time_s": grid - start})
    rows = []

    for col in numeric_common:
        yr = interpolate(ref["_time_s"].to_numpy(), ref[col].to_numpy(float), grid)
        ys = interpolate(src_time_shifted, src[col].to_numpy(float), grid)
        valid = np.isfinite(yr) & np.isfinite(ys)
        if valid.sum() < 5:
            continue
        a, b = yr[valid], ys[valid]
        if col in ANGLE_RAD:
            err = wrap_difference(b, a, 2 * np.pi)
        elif col in ANGLE_DEG:
            err = wrap_difference(b, a, 360.0)
        else:
            err = b - a
        bias = float(np.mean(err))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        mae = float(np.mean(np.abs(err)))
        corr = float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else np.nan
        rows.append({
            "reference": ref_name, "source": src_name, "signal": col,
            "n": int(valid.sum()), "bias_source_minus_reference": bias,
            "mae": mae, "rmse": rmse, "max_abs_error": float(np.max(np.abs(err))),
            "correlation": corr,
            "bias_corrected_rmse": float(np.sqrt(np.mean((err - bias) ** 2))),
        })
        aligned[f"{col}_reference"] = yr
        aligned[f"{col}_source"] = ys
        aligned[f"{col}_error"] = np.nan
        aligned.loc[valid, f"{col}_error"] = err

    if not rows:
        raise ValueError(f"No comparable numeric columns for {src_name}.")
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(src_name).stem)
    aligned.to_csv(outdir / f"aligned_{safe}.csv", index=False)
    pd.DataFrame(rows).to_csv(outdir / f"metrics_{safe}.csv", index=False)

    plot_cols = [r["signal"] for r in rows
                 if r["signal"] in ANGLE_RAD | ANGLE_DEG]
    if plot_cols:
        fig, axes = plt.subplots(len(plot_cols), 2,
                                 figsize=(14, max(4, 3.2 * len(plot_cols))),
                                 squeeze=False)
        for i, col in enumerate(plot_cols):
            axes[i, 0].plot(aligned["time_s"], aligned[f"{col}_reference"], label="Reference")
            axes[i, 0].plot(aligned["time_s"], aligned[f"{col}_source"], label=Path(src_name).name)
            axes[i, 0].set_ylabel(col)
            axes[i, 0].legend()
            axes[i, 0].grid(True)
            axes[i, 1].plot(aligned["time_s"], aligned[f"{col}_error"])
            axes[i, 1].set_ylabel(f"{col} error")
            axes[i, 1].grid(True)
        axes[-1, 0].set_xlabel("Time from overlap start [s]")
        axes[-1, 1].set_xlabel("Time from overlap start [s]")
        fig.suptitle(f"{Path(src_name).name} vs {Path(ref_name).name}; lag={lag:+.4f}s")
        fig.tight_layout()
        fig.savefig(outdir / f"plot_{safe}.png", dpi=170)
        plt.close(fig)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--files", nargs="+", required=True, type=Path,
                    help="Two or more kinematics CSV files.")
    ap.add_argument("--reference", type=Path, default=None,
                    help="Reference CSV; defaults to first --files entry.")
    ap.add_argument("--outdir", type=Path, default=Path("comparison"))
    ap.add_argument("--max-lag", type=float, default=60.0)
    ap.add_argument("--hz", type=float, default=100.0,
                    help="Comparison grid frequency.")
    ap.add_argument("--sync-signal", default=None,
                    help="Signal for automatic synchronization, e.g. LWS_ANGLE or roll.")
    ap.add_argument("--lag", action="append", default=[],
                    help="Manual lag as filename.csv=seconds; repeat for multiple files.")
    args = ap.parse_args()

    if len(args.files) < 2:
        ap.error("Provide at least two CSV files.")
    paths = [p.resolve() for p in args.files]
    ref_path = args.reference.resolve() if args.reference else paths[0]
    if ref_path not in paths:
        ap.error("--reference must also be included in --files.")
    args.outdir.mkdir(parents=True, exist_ok=True)

    datasets = {p: read_kinematics(p) for p in paths}
    ref = datasets[ref_path]
    manual = {}
    for item in args.lag:
        if "=" not in item:
            ap.error("--lag must be filename.csv=seconds")
        name, value = item.rsplit("=", 1)
        manual[Path(name).name] = float(value)

    all_rows = []
    for src_path in paths:
        if src_path == ref_path:
            continue
        src = datasets[src_path]
        if src_path.name in manual:
            lag, score, sync_signal = manual[src_path.name], np.nan, "manual"
        else:
            candidates = [args.sync_signal] if args.sync_signal else CANDIDATE_SIGNALS
            results = []
            for sig in candidates:
                if sig and sig in ref and sig in src:
                    lag_i, score_i = lag_score(
                        ref["_time_s"].to_numpy(), ref[sig].to_numpy(float),
                        src["_time_s"].to_numpy(), src[sig].to_numpy(float),
                        args.max_lag, args.hz)
                    results.append((score_i, lag_i, sig))
            if not results:
                raise ValueError(
                    f"No shared synchronization signal for {src_path.name}. "
                    "Use --lag filename.csv=seconds."
                )
            score, lag, sync_signal = max(results, key=lambda x: x[0])
        print(f"{src_path.name}: lag={lag:+.5f}s, sync={sync_signal}, "
              f"|corr|={score:.4f}")
        rows = calculate_metrics(ref, src, ref_path.name, src_path.name,
                                 lag, args.hz, args.outdir)
        for row in rows:
            row["estimated_lag_s"] = lag
            row["sync_signal"] = sync_signal
            row["sync_abs_correlation"] = score
        all_rows.extend(rows)

    summary = pd.DataFrame(all_rows)
    summary.to_csv(args.outdir / "all_metrics.csv", index=False)
    print("\nMetrics:")
    print(summary.to_string(index=False))
    print(f"\nResults: {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
