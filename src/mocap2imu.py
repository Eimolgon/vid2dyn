import argparse
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation as R


def load_mocap_csv(filepath: str) -> pd.DataFrame:
  """Loads an OptiTrack multi-header CSV into a 5-level MultiIndex DataFrame."""
  with open(filepath, "r") as f:
    header_rows = [f.readline().strip().split(",") for _ in range(7)]

  types = header_rows[2]
  names = header_rows[3]
  ids = header_rows[4]
  components = header_rows[5]
  axes = header_rows[6]

  df = pd.read_csv(filepath, skiprows=7, header=None, low_memory=False)
  col_tuples = [
      (
          types[j] if j < len(types) else "",
          names[j] if j < len(names) else "",
          ids[j] if j < len(ids) else "",
          components[j] if j < len(components) else "",
          axes[j] if j < len(axes) else "",
      )
      for j in range(len(df.columns))
  ]
  df.columns = pd.MultiIndex.from_tuples(
      col_tuples, names=["Type", "Name", "ID", "Component", "Axis"]
  )
  return df


def lowpass_filter(
    data: np.ndarray, fs: float, cutoff: float = 15.0, order: int = 4
) -> np.ndarray:
  """Zero-phase Butterworth low-pass filter."""
  nyq = 0.5 * fs
  b, a = butter(order, cutoff / nyq, btype="low")
  return filtfilt(b, a, data, axis=0)


def main():
  parser = argparse.ArgumentParser(
      description=(
          "Process OptiTrack Mocap Data to Kinematics/IMU CSV export."
      )
  )
  parser.add_argument(
      "--input",
      type=str,
      default="/home/eimolgon/Documents/PhD-Project/02-video-data/261106-lab-gt01/mocap/261106-take145759.csv",
      help="Path to input mocap CSV file",
  )
  parser.add_argument(
      "--output",
      type=str,
      default="/home/eimolgon/Documents/PhD-Project/02-video-data/261106-lab-gt01/mocap/261106-take145759-imuConverted.csv",
      help="Path to output CSV file",
  )
  parser.add_argument(
      "--raw",
      action="store_true",
      help=(
          "Keep data raw (disable low-pass filtering on velocities,"
          " accelerations, and steering angle)"
      ),
  )
  args = parser.parse_args()

  print(f"Loading motion capture file: {args.input}...")
  df = load_mocap_csv(args.input)

  # Timestamps starting from 0
  time_raw = df[("Type", "Name", "ID", "", "Time (Seconds)")].to_numpy(
      dtype=float
  )
  timestamps = time_raw - time_raw[0]
  dt = np.mean(np.diff(timestamps))
  fs = 1.0 / dt

  # Extract Rear Frame ('Bike') and Front Frame ('front-frame')
  bike_sub = df.loc(axis=1)[:, "Bike", :, :, :]
  rear_pos = bike_sub.loc(
      axis=1
  )[:, :, :, "Position", ["X", "Y", "Z"]].to_numpy(dtype=float)
  rear_quat = bike_sub.loc(
      axis=1
  )[:, :, :, "Rotation", ["X", "Y", "Z", "W"]].to_numpy(dtype=float)

  front_sub = df.loc(axis=1)[:, "front-frame", :, :, :]
  front_quat = front_sub.loc(
      axis=1
  )[:, :, :, "Rotation", ["X", "Y", "Z", "W"]].to_numpy(dtype=float)

  # Position processing (filter unless --raw is specified)
  rear_pos_proc = (
      rear_pos if args.raw else lowpass_filter(rear_pos, fs=fs, cutoff=12.0)
  )

  r_rear = R.from_quat(rear_quat)
  r_front = R.from_quat(front_quat)

  # Orientation Euler Angles (deg)
  euler_yxz = r_rear.as_euler("yxz", degrees=True)
  yaw = euler_yxz[:, 0]
  roll = euler_yxz[:, 1]
  pitch = euler_yxz[:, 2]

  # Body Angular Rates (gyro_x, gyro_y, gyro_z in deg/s)
  r_step = r_rear[:-1].inv() * r_rear[1:]
  omega_body_rad = np.vstack(
      [r_step.as_rotvec() / dt, r_step[-1].as_rotvec() / dt]
  )
  omega_body_deg = np.degrees(omega_body_rad)
  if not args.raw:
    omega_body_deg = lowpass_filter(omega_body_deg, fs=fs, cutoff=15.0)

  gyro_x = omega_body_deg[:, 0]
  gyro_y = omega_body_deg[:, 1]
  gyro_z = omega_body_deg[:, 2]

  # Velocities and Accelerations
  vel_global = np.gradient(rear_pos_proc, dt, axis=0)
  acc_global = np.gradient(vel_global, dt, axis=0)

  vel_body = r_rear.inv().apply(vel_global)
  ws_rear = vel_body[:, 0]

  g_global = np.array([0.0, -9.80665, 0.0])
  accel_body = r_rear.inv().apply(acc_global - g_global)
  accel_x = accel_body[:, 0]
  accel_y = accel_body[:, 1]
  accel_z = accel_body[:, 2]

  # Steering Angle (LWS_ANGLE and LWS_SPEED)
  r_rel = r_rear.inv() * r_front
  r_rel_zero = r_rel * r_rel[0].inv()
  rv_rear = r_rel_zero.as_rotvec()

  _, _, vt = np.linalg.svd(rv_rear, full_matrices=False)
  steer_axis = vt[0]
  if steer_axis[1] < 0:
    steer_axis = -steer_axis

  lws_angle = np.degrees(rv_rear @ steer_axis)
  if not args.raw:
    lws_angle = lowpass_filter(lws_angle, fs=fs, cutoff=15.0)
  lws_speed = np.gradient(lws_angle, dt)

  # Assemble Export DataFrame
  out_df = pd.DataFrame({
      "timestamps": timestamps,
      "gyro_z": gyro_z,
      "gyro_y": gyro_y,
      "gyro_x": gyro_x,
      "accel_z": accel_z,
      "accel_y": accel_y,
      "accel_x": accel_x,
      "yaw": yaw,
      "pitch": pitch,
      "roll": roll,
      "ws_rear": ws_rear,
      "LWS_ANGLE": lws_angle,
      "LWS_SPEED": lws_speed,
  })

  out_df.to_csv(args.output, sep=";", index=False, float_format="%.6f")
  print(
      f"Successfully exported {len(out_df)} rows to '{args.output}' (Raw mode:"
      f" {args.raw})."
  )


if __name__ == "__main__":
  main()