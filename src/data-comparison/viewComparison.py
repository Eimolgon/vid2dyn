import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from alignData import load_signals
from compareData import compare_aligned
from compareSegmentation02 import build_seg_time


def plot_imu_vs_kinematics(
    imu: pd.DataFrame,
    kin: pd.DataFrame,
    metrics_df: pd.DataFrame,
    lag: float = 13.90,
    out_path: str = "aligned_signals_comparison.png",
):
  # Align IMU onto kinematics timeline: t_imu = t_kin + lag
  t_start = max(0.0, -lag)
  t_end = min(float(kin["t_rel"].iloc[-1]), float(imu["t_rel"].iloc[-1]) - lag)
  grid = np.arange(t_start, t_end, 0.02)

  fig, axes = plt.subplots(3, 2, figsize=(14, 10), constrained_layout=True)

  overlay_pairs = [
      ("LWS_ANGLE", "Steering Angle (LWS_ANGLE)", axes[0, 0]),
      ("LWS_SPEED", "Steering Speed (LWS_SPEED)", axes[0, 1]),
      ("roll", "Roll", axes[1, 0]),
      ("yaw", "Yaw", axes[1, 1]),
  ]

  for col, title, ax in overlay_pairs:
    k_val = np.interp(grid, kin["t_rel"], kin[col])
    i_val = np.interp(grid + lag, imu["t_rel"], imu[col])
    ax.plot(grid, k_val, label="Kinematics", linewidth=1.4)
    ax.plot(grid, i_val, label=f"IMU (shifted by {lag:+.2f}s)", linewidth=1.1, alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel("Kinematics Relative Time (s)")
    ax.set_ylabel(col)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

  # Bottom-left: Pearson r across all shared channels
  ax_r = axes[2, 0]
  sorted_m = metrics_df.sort_values("pearson_r")
  ax_r.barh(sorted_m["signal"], sorted_m["pearson_r"])
  ax_r.axvline(0.0, color="black", linewidth=0.8)
  ax_r.set_xlim(-1.05, 1.05)
  ax_r.set_title("Pearson r (Aligned IMU vs. Kinematics)")
  ax_r.set_xlabel("Correlation r")
  ax_r.grid(True, axis="x", alpha=0.3)

  # Bottom-right: Raw vs Calibrated RMSE for top channels
  ax_e = axes[2, 1]
  x = np.arange(len(metrics_df))
  w = 0.38
  ax_e.bar(x - w / 2, metrics_df["raw_rmse"], width=w, label="Raw RMSE")
  ax_e.bar(x + w / 2, metrics_df["calibrated_rmse"], width=w, label="Linear-Fit RMSE")
  ax_e.set_xticks(x)
  ax_e.set_xticklabels(metrics_df["signal"], rotation=45, ha="right", fontsize=8)
  ax_e.set_title("Error Before vs. After Linear Scaling/Offset")
  ax_e.set_ylabel("RMSE")
  ax_e.legend(loc="best", fontsize=8)
  ax_e.grid(True, axis="y", alpha=0.3)

  fig.suptitle("IMU vs. Kinematics Signal Alignment & Comparison", fontsize=13)
  fig.savefig(out_path, dpi=150)
  plt.close(fig)
  print(f"Saved {out_path}")


def plot_segmentation_comparison(
    seg_path: str,
    kin: pd.DataFrame,
    out_path: str = "segmentation_comparison.png",
):
  seg = pd.read_csv(seg_path)
  if "valid" in seg.columns:
    seg = seg[seg["valid"] == 1].copy()

  kin_dur = float(kin["t_rel"].iloc[-1])
  t_span, eff_fps = build_seg_time(seg, target_duration=kin_dur)
  t_30fps, _ = build_seg_time(seg, target_duration=None, fps=30.0)

  fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)

  # Steering comparison: duration-matched vs 30 fps
  axes[0, 0].plot(kin["t_rel"], kin["LWS_ANGLE"], label="Kinematics LWS_ANGLE", linewidth=1.2)
  axes[0, 0].plot(t_span, seg["steer_deg"], label=f"Seg steer_deg ({eff_fps:.1f} fps)", linewidth=1.1, alpha=0.85)
  axes[0, 0].set_title("Steering: Duration-Matched (~18.0 fps)")
  axes[0, 0].set_xlabel("Time (s)")
  axes[0, 0].set_ylabel("Angle / Deg")
  axes[0, 0].legend(fontsize=8)
  axes[0, 0].grid(True, alpha=0.3)

  axes[0, 1].plot(kin["t_rel"], kin["LWS_ANGLE"], label="Kinematics LWS_ANGLE", linewidth=1.2)
  axes[0, 1].plot(t_30fps, seg["steer_deg"], label="Seg steer_deg (30 fps)", linewidth=1.1, alpha=0.85)
  axes[0, 1].set_title("Steering: Fixed 30 fps (74.6 s)")
  axes[0, 1].set_xlabel("Time (s)")
  axes[0, 1].set_ylabel("Angle / Deg")
  axes[0, 1].legend(fontsize=8)
  axes[0, 1].grid(True, alpha=0.3)

  # Roll comparison: duration-matched vs 30 fps
  axes[1, 0].plot(kin["t_rel"], kin["roll"], label="Kinematics roll (rad)", linewidth=1.2)
  axes[1, 0].plot(t_span, seg["roll_deg"], label=f"Seg roll_deg ({eff_fps:.1f} fps)", linewidth=1.1, alpha=0.85)
  axes[1, 0].set_title("Roll: Duration-Matched (~18.0 fps)")
  axes[1, 0].set_xlabel("Time (s)")
  axes[1, 0].set_ylabel("Roll")
  axes[1, 0].legend(fontsize=8)
  axes[1, 0].grid(True, alpha=0.3)

  axes[1, 1].plot(kin["t_rel"], kin["roll"], label="Kinematics roll (rad)", linewidth=1.2)
  axes[1, 1].plot(t_30fps, seg["roll_deg"], label="Seg roll_deg (30 fps)", linewidth=1.1, alpha=0.85)
  axes[1, 1].set_title("Roll: Fixed 30 fps (74.6 s)")
  axes[1, 1].set_xlabel("Time (s)")
  axes[1, 1].set_ylabel("Roll")
  axes[1, 1].legend(fontsize=8)
  axes[1, 1].grid(True, alpha=0.3)

  fig.suptitle("Segmentation vs. Kinematics Time-Axis Comparison", fontsize=13)
  fig.savefig(out_path, dpi=150)
  plt.close(fig)
  print(f"Saved {out_path}")


if __name__ == "__main__":
  imu, kin = load_signals(
      "/home/eimolgon/Documents/PhD-Project/02-video-data/261106-lab-gt01/mocap/261106-take145759-imuConverted.csv",
      "/home/eimolgon/Documents/PhD-Project/02-video-data/261106-lab-gt01/00000001_kinematics.csv"
    )
  lag = 13.90
  metrics_df = compare_aligned(imu, kin, lag=lag, fs=50.0)
  plot_imu_vs_kinematics(imu, kin, metrics_df, lag=lag)
  plot_segmentation_comparison("/home/eimolgon/Documents/PhD-Project/02-video-data/261106-lab-gt01/segmentation/segmentation.csv", kin)