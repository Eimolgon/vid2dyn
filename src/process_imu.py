import pandas as pd
import matplotlib.pyplot as plt

data_path_imu = ('~/Documents/PhD-Project/02-video-data/261106-lab-gt01/'
             '00000001_kinematics.csv')

data_path_mocap = ('/home/eimolgon/Documents/PhD-Project/02-video-data/'
                '261106-lab-gt01/mocap/261106-take145759-imuConverted.csv')

df_imu = pd.read_csv(data_path_imu, header='infer', sep=';')
df_imu['timestamps'] = pd.to_datetime(df_imu['timestamps'])

df_mocap = pd.read_csv(data_path_mocap, header='infer', sep=';')
# df_mocap['timestamps'] = pd.to_datetime(df_mocap['timestamps'])

# headers = df.keys()
zeroedTime_imu = (df_imu['timestamps'] - df_imu['timestamps'].iloc[0]).dt.total_seconds()


yaw_imu = df_imu['yaw']
roll_imu = df_imu['roll']
pitch_imu = df_imu['pitch']
steer_imu = df_imu['LWS_ANGLE']

yaw_mocap = df_mocap['yaw']
roll_mocap = df_mocap['roll']
pitch_mocap = df_mocap['pitch']
steer_mocap = df_mocap['LWS_ANGLE']
# print(headers)
# df.plot(x='timestamps', y='LWS_SPEED')

# plt.plot(zeroedTime_imu, roll_imu, label='roll imu')
# plt.plot(df_mocap['timestamps'], roll_mocap, label='roll mocap' )

plt.plot(zeroedTime_imu, steer_imu, label='steer imu')
plt.plot(df_mocap['timestamps'], steer_mocap, label='steer mocap' )


plt.legend()
plt.grid()
plt.show()


