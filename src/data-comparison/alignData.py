import numpy as np
import pandas as pd


def load_signals(imu_path: str, kin_path: str):
  imu = pd.read_csv(imu_path, sep=";")
  kin = pd.read_csv(kin_path, sep=";")

  imu["t_rel"] = pd.to_numeric(imu["timestamps"], errors="coerce")
  imu["t_rel"] = imu["t_rel"] - imu["t_rel"].iloc[0]

  kin_dt = pd.to_datetime(kin["timestamps"], utc=True)
  kin["t_rel"] = (kin_dt - kin_dt.iloc[0]).dt.total_seconds()
  return imu.dropna(subset=["t_rel"]), kin.dropna(subset=["t_rel"])


def find_best_lag(
    imu: pd.DataFrame,
    kin: pd.DataFrame,
    col: str = "LWS_SPEED",
    lag_min: float = -30.0,
    lag_max: float = 30.0,
    step: float = 0.02,
):
  # Common base grid starting at 0 s up to kinematics duration
  t_end = min(kin["t_rel"].iloc[-1], imu["t_rel"].iloc[-1])
  grid = np.arange(0.0, t_end, step)
  kin_sig = np.interp(grid, kin["t_rel"], kin[col])
  kin_sig = kin_sig - np.mean(kin_sig)
  kin_norm = np.linalg.norm(kin_sig)

  best_lag = 0.0
  best_r = 0.0

  # Convention: t_imu = t_kin + lag  (positive lag means IMU event occurs later in relative time)
  for lag in np.arange(lag_min, lag_max + step, step):
    t_q = grid + lag
    valid = (t_q >= imu["t_rel"].iloc[0]) & (t_q <= imu["t_rel"].iloc[-1])
    if np.sum(valid) < 100:
      continue
    imu_sig = np.interp(t_q[valid], imu["t_rel"], imu[col])
    imu_sig = imu_sig - np.mean(imu_sig)
    k_sub = kin_sig[valid] - np.mean(kin_sig[valid])
    denom = np.linalg.norm(imu_sig) * np.linalg.norm(k_sub)
    if denom == 0:
      continue
    r = float(np.dot(imu_sig, k_sub) / denom)
    if abs(r) > abs(best_r):
      best_r = r
      best_lag = float(lag)

  return best_lag, best_r


if __name__ == "__main__":
  imu, kin = load_signals(
      "/home/eimolgon/Documents/PhD-Project/02-video-data/261106-lab-gt01/mocap/261106-take145759-imuConverted.csv",
      "/home/eimolgon/Documents/PhD-Project/02-video-data/261106-lab-gt01/00000001_kinematics.csv"
  )
  lag, r = find_best_lag(imu, kin, col="LWS_SPEED")
  print(f"Estimated lag (t_imu = t_kin + lag): {lag:+.3f} s (r = {r:+.4f})")