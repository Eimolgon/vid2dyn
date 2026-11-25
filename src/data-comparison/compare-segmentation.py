import numpy as np
import pandas as pd
from alignData import load_signals


def build_seg_time(
    seg: pd.DataFrame, target_duration: float | None = None, fps: float = 30.0
):
  frames = seg["frame"].to_numpy(dtype=float) - float(seg["frame"].min())
  span = max(1.0, float(frames.max()))
  if target_duration is not None:
    # Keep the same total trial duration and compute the implied frame rate
    t_seg = (frames / span) * float(target_duration)
    implied_fps = span / float(target_duration)
    return t_seg, implied_fps
  # Fixed frame rate (e.g., 30 fps)
  return frames / float(fps), float(fps)


def evaluate_seg_alignment(
    seg_path: str,
    imu: pd.DataFrame,
    kin: pd.DataFrame,
    lag: float = 13.90,
    target_duration: float | None = None,
    fps: float = 30.0,
):
  seg = pd.read_csv(seg_path)
  if "valid" in seg.columns:
    seg = seg[seg["valid"] == 1].copy()

  t_seg, eff_fps = build_seg_time(
      seg, target_duration=target_duration, fps=fps
  )

  # Evaluate on overlapping window in kinematics relative time t_k (t_imu = t_k + lag)
  t_start = max(0.0, float(t_seg[0]), -lag)
  t_end = min(
      float(t_seg[-1]), float(kin["t_rel"].iloc[-1]), float(imu["t_rel"].iloc[-1]) - lag
  )
  if t_end <= t_start:
    print(f"No overlap for eff_fps={eff_fps:.2f} (t_seg ends at {t_seg[-1]:.2f}s)")
    return

  grid = np.arange(t_start, t_end, 0.05)
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

  print(
      f"\n=== eff_fps={eff_fps:.2f} | seg_duration={t_seg[-1]:.2f}s |"
      f" overlap={t_end - t_start:.2f}s ==="
  )
  for name, a, b in [
      ("seg_roll vs kin_roll", s_roll, k_roll),
      ("seg_roll vs imu_roll", s_roll, i_roll),
      ("seg_pitch vs kin_pitch", s_pitch, k_pitch),
      ("seg_yaw vs kin_yaw", s_yaw, k_yaw),
      ("seg_steer vs kin_LWS_ANGLE", s_steer, k_lws),
      ("seg_steer vs imu_LWS_ANGLE", s_steer, i_lws),
  ]:
    r = (
        float(np.corrcoef(a, b)[0, 1])
        if (np.std(a) > 0 and np.std(b) > 0)
        else np.nan
    )
    print(f"  {name:28s}: r = {r:+.4f}")


if __name__ == "__main__":
  imu, kin = load_signals(
      "261106-take145759-imuConverted.csv", "00000001_kinematics.csv"
  )
  kin_dur = float(kin["t_rel"].iloc[-1])
  imu_dur = float(imu["t_rel"].iloc[-1])

  # 1. Same duration as kinematics (~18.03 fps) and IMU (~14.76 fps)
  evaluate_seg_alignment(
      "segmentation.csv", imu, kin, lag=13.90, target_duration=kin_dur
  )
  evaluate_seg_alignment(
      "segmentation.csv", imu, kin, lag=13.90, target_duration=imu_dur
  )

  # 2. Fixed 30 fps (74.57 s)
  evaluate_seg_alignment(
      "segmentation.csv", imu, kin, lag=13.90, target_duration=None, fps=30.0
  )