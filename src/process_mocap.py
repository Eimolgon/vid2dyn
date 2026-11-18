import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R


def load_mocap_csv(filepath: str) -> pd.DataFrame:
  """Loads an OptiTrack motion capture CSV export into a structured

  pandas DataFrame with a 5-level MultiIndex column hierarchy.
  """
  with open(filepath, "r") as f:
    header_rows = [f.readline().strip().split(",") for _ in range(7)]

  types = header_rows[2]
  names = header_rows[3]
  ids = header_rows[4]
  components = header_rows[5]
  axes = header_rows[6]

  df = pd.read_csv(filepath, skiprows=7, header=None, low_memory=False)

  col_tuples = []
  for j in range(len(df.columns)):
    col_tuples.append(
        (
            types[j] if j < len(types) else "",
            names[j] if j < len(names) else "",
            ids[j] if j < len(ids) else "",
            components[j] if j < len(components) else "",
            axes[j] if j < len(axes) else "",
        )
    )

  df.columns = pd.MultiIndex.from_tuples(
      col_tuples, names=["Type", "Name", "ID", "Component", "Axis"]
  )
  return df


def get_body_data(df: pd.DataFrame, body_name: str):
  """Extracts 3D position [X, Y, Z] and orientation quaternion [Qx, Qy, Qz, Qw]

  for a specified rigid body by matching its Name in the MultiIndex.
  """
  # Filter columns where Name level matches body_name
  sub_df = df.loc(axis=1)[:, body_name, :, :, :]

  # Extract Position (X, Y, Z) where Axis matches X, Y, Z under Position component
  pos_df = sub_df.loc(axis=1)[:, :, :, :, ["X", "Y", "Z"]]
  # Filter specifically for position component columns
  pos_cols = [
      col
      for col in pos_df.columns
      if "Position" in col[3] or "Position" in col[2]
  ]
  if not pos_cols:
    # Fallback to first 3 numeric columns if component naming varies
    pos_data = sub_df.iloc[:, :3].to_numpy()
  else:
    pos_data = sub_df[pos_cols].to_numpy()

  # Extract Rotation Quaternions (X, Y, Z, W)
  rot_cols = [
      col for col in sub_df.columns if "Rotation" in col[3] or "Rotation" in col[2]
  ]
  if not rot_cols:
    rot_data = sub_df.iloc[:, 3:7].to_numpy()
  else:
    rot_data = sub_df[rot_cols].to_numpy()

  return pos_data, rot_data


if __name__ == "__main__":
  filename = (
      "/home/eimolgon/Documents/PhD-Project/02-video-data/261106-lab-gt01/mocap/"
      "261106-take145759.csv"
  )

  print(f"Loading motion capture file: {filename}...")
  df = load_mocap_csv(filename)

  # Extract Time and Frame correctly
  time_col = df[("Type", "Name", "ID", "", "Time (Seconds)")].to_numpy()
  frame_col = df[("", "", "", "", "Frame")].to_numpy()

  # Identify available rigid body names dynamically
  rigid_bodies = df.columns.get_level_values("Name").unique()
  rigid_bodies = [
      b
      for b in rigid_bodies
      if b not in ["", "Name", "Type"] and isinstance(b, str)
  ]
  print(f"Detected Rigid Bodies: {rigid_bodies}")

  # Map bodies based on names ('Bike' for rear frame, 'front-frame' for front frame)
  rear_name = next((b for b in rigid_bodies if "bike" in b.lower()), "Bike")
  front_name = next(
      (b for b in rigid_bodies if "front" in b.lower()), "front-frame"
  )
  camera_name = next((b for b in rigid_bodies if "camera" in b.lower()), "camera")

  print(f"\nTarget Mapping:")
  print(f" - Rear Frame (Bike) : {rear_name}")
  print(f" - Front Frame       : {front_name}")
  print(f" - Camera            : {camera_name}")

  # 1. Track Positions and Orientations
  rear_pos, rear_quat = get_body_data(df, rear_name)
  front_pos, front_quat = get_body_data(df, front_name)

  # 2. Convert Quaternions to Euler Angles (Yaw, Pitch, Roll) using scipy
  r_rear = R.from_quat(rear_quat)
  r_front = R.from_quat(front_quat)

  rear_euler = r_rear.as_euler("zxy", degrees=True)
  front_euler = r_front.as_euler("zxy", degrees=True)

  # Extract Yaw (Heading) around Z-axis
  yaw_rear = rear_euler[:, 0]
  yaw_front = front_euler[:, 0]

  # 3. Calculate Steering Angle
  steering_angle = yaw_front - yaw_rear
  steering_angle = (steering_angle + 180) % 360 - 180  # Normalize to [-180, 180]

  print("\n--- Kinematics & Steering Summary ---")
  print(f"Total Frames Processed        : {len(frame_col)}")
  print(f"Mean Rear Position (X, Y, Z)  : {np.mean(rear_pos, axis=0)}")
  print(f"Mean Front Position (X, Y, Z) : {np.mean(front_pos, axis=0)}")
  print(f"Max Steering Angle Recorded   : {np.max(np.abs(steering_angle)):.2f}°")