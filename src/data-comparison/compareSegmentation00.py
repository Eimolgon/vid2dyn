import numpy as np
import pandas as pd
from alignData import load_signals


def map_segmentation_time(
    seg: pd.DataFrame, mode: str = "kin_span", kin_duration: float = 124.08
):
  frames = seg["frame"].to_numpy(dtype=float)
  if mode == "kin_span":
    # Stretch frame indices evenly across kinematics duration
    return (frames - frames.min()) / max(1.0, (frames.max() - frames.min())) * kin_duration
  # Otherwise treat `mode` as numeric FPS string, e.g., "15", "20", "30"
  fps = float(mode)
  return (frames - frames.min()) / fps


def check_segmentation(
    seg_path: str, imu: pd.DataFrame, kin: pd.DataFrame, lag: float, mode: str = "kin_span"
):
  seg = pd.read_csv(seg_path)
  if "valid" in seg.columns:
    seg = seg[seg["valid"] == 1].copy()

  t_seg = map_segmentation_time(seg, mode=mode, kin_duration=kin["t_rel"].iloc[-1])
  t_end = min(t_seg[-1], kin["t_rel"].iloc[-1], imu["t_rel"].iloc[-1] - lag)
  grid = np.arange(max(0.0, t_seg[0], -lag), t_end, 0.05)

  s_roll = np.interp(grid, t_seg, seg["roll_deg"])
  s_pitch = np.interp(grid, t_seg, seg["pitch_deg"])
  s_yaw = np.interp(grid, t_seg, seg["yaw_deg"])
  s_steer = np.interp(grid, t_seg, seg["steer_deg"])

  k_roll = np.interp(grid, kin["t_rel"], kin["roll"])
  k_pitch = np.interp(grid, kin["t_rel"], kin["pitch"])
  k_yaw = np.interp(grid, kin["t_rel"], kin["yaw"])
  k_lws = np.interp(grid, kin["t_rel"], kin["LWS_ANGLE"])

  i_roll = np.interp(grid + lag, imu["t_rel"], imu["roll"])
  i_lws = np.interp(grid + lag, imu["t_rel"], imu["LWS_ANGLE"])

  pairs = [
      ("seg_roll vs kin_roll", s_roll, k_roll),
      ("seg_roll vs imu_roll", s_roll, i_roll),
      ("seg_pitch vs kin_pitch", s_pitch, k_pitch),
      ("seg_yaw vs kin_yaw", s_yaw, k_yaw),
      ("seg_steer vs kin_LWS_ANGLE", s_steer, k_lws),
      ("seg_steer vs imu_LWS_ANGLE", s_steer, i_lws),
  ]
  for label, x, y in pairs:
    r = float(np.corrcoef(x, y)[0, 1]) if (np.std(x) > 0 and np.std(y) > 0) else np.nan
    print(f"{label:28s}: r = {r:+.4f}")


if __name__ == "__main__":
  imu, kin = load_signals(
      "/home/eimolgon/Documents/PhD-Project/02-video-data/261106-lab-gt01/mocap/261106-take145759-imuConverted.csv",
      "/home/eimolgon/Documents/PhD-Project/02-video-data/261106-lab-gt01/00000001_kinematics.csv"
    )
  for m in ["kin_span", "15", "20", "30"]:
    print(f"\n--- Mode: {m} ---")
    check_segmentation("/home/eimolgon/Documents/PhD-Project/02-video-data/261106-lab-gt01/segmentation/segmentation.csv", imu, kin, lag=13.90, mode=m)