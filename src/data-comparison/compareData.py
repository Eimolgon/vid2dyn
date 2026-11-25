import numpy as np
import pandas as pd
from alignData import load_signals

SHARED_COLS = [
    "gyro_z",
    "gyro_y",
    "gyro_x",
    "accel_z",
    "accel_y",
    "accel_x",
    "yaw",
    "pitch",
    "roll",
    "ws_rear",
    "LWS_ANGLE",
    "LWS_SPEED",
]


def compare_aligned(
    imu: pd.DataFrame, kin: pd.DataFrame, lag: float, fs: float = 50.0
):
  dt = 1.0 / fs
  # Overlapping window in kinematics relative time t_k, where t_imu = t_k + lag
  t_k_start = max(0.0, -lag)
  t_k_end = min(kin["t_rel"].iloc[-1], imu["t_rel"].iloc[-1] - lag)
  t_k = np.arange(t_k_start, t_k_end, dt)
  t_i = t_k + lag

  rows = []
  for col in SHARED_COLS:
    a = np.interp(t_i, imu["t_rel"], imu[col])  # IMU aligned to t_k
    b = np.interp(t_k, kin["t_rel"], kin[col])  # Kinematics at t_k

    diff = b - a
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))
    bias = float(np.mean(diff))

    if np.std(a) > 0 and np.std(b) > 0:
      r = float(np.corrcoef(a, b)[0, 1])
      slope, intercept = np.polyfit(a, b, 1)
      b_fit = slope * a + intercept
      cal_rmse = float(np.sqrt(np.mean((b - b_fit) ** 2)))
    else:
      r, slope, intercept, cal_rmse = np.nan, np.nan, np.nan, rmse

    rows.append({
        "signal": col,
        "pearson_r": r,
        "raw_mae": mae,
        "raw_rmse": rmse,
        "raw_bias_kin_minus_imu": bias,
        "fit_slope": slope,
        "fit_intercept": intercept,
        "calibrated_rmse": cal_rmse,
    })

  return pd.DataFrame(rows)


if __name__ == "__main__":
  imu, kin = load_signals(
      "/home/eimolgon/Documents/PhD-Project/02-video-data/261106-lab-gt01/mocap/261106-take145759-imuConverted.csv",
      "/home/eimolgon/Documents/PhD-Project/02-video-data/261106-lab-gt01/00000001_kinematics.csv"
  )
  lag = 13.90  # Replace with output from Chunk 1
  metrics_df = compare_aligned(imu, kin, lag=lag, fs=50.0)
  metrics_df.to_csv("comparison_metrics.csv", index=False)
  print(metrics_df.to_string(index=False))