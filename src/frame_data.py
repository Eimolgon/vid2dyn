import json
import time
import datetime
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.optimize import least_squares
from bike_model import *
from skimage.draw import ellipse_perimeter

from utils import (
    readFile,
    fitEllipse,
    find_points,
    find_points2,
    process_directory,
    apply_filters,
    animate_Point,
    animate_Ellipse,
    data4model,
    fit_img2model
)


# ----- Paths -----

DATA_CUT = '/home/eimolgon/Documents/PhD-Project/02-video-data/gopro_test_1_1_cut/labels/Train/'
DATA_1_1 = '/home/eimolgon/Documents/PhD-Project/02-video-data/gopro_test_1_1_resize_annotation_161025/labels/Train/'
DATA_CRASH_005 = '/home/eimolgon/Documents/PhD-Project/02-video-data/yt-crash-005/labels/train/'


# ----- Parser setup -----

parser = argparse.ArgumentParser(description='Inverse kinematics from ellipses')
plot_group = parser.add_argument_group('Plotting options')


parser.add_argument('-f', '--filter', 
                    default=False, action='store_true', 
                    help='Apply low pass filter')
parser.add_argument('-d', '--data', 
                    type=str, default='0',
                    help='Choose data directory')
parser.add_argument('-sf', '--single_frame', 
                    default='0', type=str,
                    help='Extract data from a single frame, requires route to ' \
                    'file')
parser.add_argument('-p', '--plot',
                    type=str, default='0',
                    help='Allow plotting of the results')
parser.add_argument('--save', action='store_true',
                    help='Save plots')


plot_group.add_argument('-a2d', '--animate_points', 
                    default='0', type=str, 
                    help='Plot points animation')
plot_group.add_argument('-a3d', '--animate_ellipses', 
                    default='0', type=str, 
                    help='Plot ellipses animation')

args = parser.parse_args()


# ----- Run -----

# Directory used for testing /home/eimolgon/Documents/PhD-Project/02-vide-data/gopro_test_1_1_cut/labels/Train/

rawdate = datetime.datetime.now()
date =  rawdate.strftime('%Y') + rawdate.strftime('%m') + rawdate.strftime('%d')
start_time = time.time()

# ----- Some parameters -----
screen_resolution = (1920, 1080)
assumed_origin = (screen_resolution[0]*0.2, screen_resolution[1]*0.2)
wheel_diameter = 0.6604
bike_params = {
    'lr': 140,
    'lf1': 50,
    'lf2': 10,
    'r': 0 
}

lower_bound = [-np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2, 0, -np.inf, 0]
upper_bound = [np.pi/2, np.pi/2, np.pi/2, np.pi/2, 1920, np.inf, 1080]
boundaries = (lower_bound, upper_bound)


# === Initial guess ===

x0 = np.array([
    np.deg2rad(0),
    np.deg2rad(-30), 
    np.deg2rad(0),
    np.deg2rad(0),
    180,
    0,
    400
])


if args.single_frame != '0':

    # test_frame_single = '/home/eimolgon/Documents/PhD-Project/02-video-data/gopro_test_1_1_cut/labels/Train/frame_000417.txt'
    test_frame_single = '/home/eimolgon/Documents/PhD-Project/02-video-data/gopro_test_1_1_cut/labels/Train/frame_000480.txt'

    # data_points = readFile(args.single_frame)
    data_points = readFile(test_frame_single)
    front_data = data_points[0]['points']
    rear_data = data_points[1]['points']

    ellipse_front = fitEllipse(front_data, screen_resolution)
    ellipse_rear = fitEllipse(rear_data, screen_resolution)


    xf, zf, af, bf, theta_f = ellipse_front
    xr, zr, ar, br, theta_r = ellipse_rear

    upper_f, lower_f = find_points2(ellipse_front)
    upper_r, lower_r = find_points2(ellipse_rear)


    imgdata = {
        'r_Cf_Cr_x' : xf - xr,
        'r_Cf_Cr_z' : zf - zr,
        'r_Cr_O_x' : xr - assumed_origin[0],
        'r_Cr_O_z' : zr - assumed_origin[1],
        'r_P1r_Cr_x' : upper_r[0] - xr,
        'r_P1r_Cr_z' : upper_r[1] - zr,
        'r_P3r_Cr_x' : lower_r[0] - xr,
        'r_P3r_Cr_z' : lower_r[1] - zr,
        'r_P1f_Cf_x' : upper_f[0] - xf,
        'r_P1f_Cf_z' : upper_f[1] - zf,
        'r_P3f_Cf_x' : lower_f[0] - xf,
        'r_P3f_Cf_z' : lower_f[1] - zf
    }

    bike_params['r'] = 2*br


    print(f'Front wheel centre: ({ellipse_front[0]}, {ellipse_front[1]})')
    # print(f'Front wheel points: {P1f}, {P2f}, {P3f}, {P4f}')
    # print(f'Upper point front wheel: {upper_f}')
    # print(f'lower point front wheel: {lower_f}')
    print(f'Rear wheel centre: ({ellipse_rear[0]}, {ellipse_rear[1]})')

    if args.plot != '0':

        plt.plot(xf, zf, 'xk')
        plt.plot(upper_f[0], upper_f[1], 'ok')
        plt.plot(lower_f[0], lower_f[1], 'ob')
        plt.scatter(front_data[:, 0]*screen_resolution[0], (1 - front_data[:, 1])*screen_resolution[1], facecolors='none', edgecolors='red')

        plt.plot(xr, zr, 'xk')
        plt.plot(upper_r[0], upper_r[1], 'ok')
        plt.plot(lower_r[0], lower_r[1], 'ob')

        ell_patch_f = Ellipse((xf, zf), width = 2*af, height = 2*bf,
                      angle = theta_f*180/np.pi, edgecolor='blue', facecolor='none')

        ell_patch_r = Ellipse((xr, zr), width = 2*ar, height = 2*br,
                      angle = theta_r*180/np.pi, edgecolor='red', facecolor='none')

        plt.gca().add_patch(ell_patch_f)
        plt.gca().add_patch(ell_patch_r)

        plt.xlim(0, 1980)
        plt.ylim(0, 1080)
        plt.gca().set_aspect('equal')
        plt.grid()
        plt.show()

elif args.data != '0':

    # ----- Data processing -----

    if args.data == 'cut':
        data_directory = DATA_CUT
    elif args.data == '1-1':
        data_directory = DATA_1_1
    elif args.data == '05':
        data_directory = DATA_CRASH_005

    raw_data = process_directory(data_directory, screen_resolution)
    
    if args.filter:
        tracking_data = apply_filters(raw_data)
    else:
        tracking_data = raw_data

    data_model, rear_wheel, front_wheel = data4model(tracking_data, assumed_origin)
    first_frame = tracking_data['first_frame']
    
    bike_params['r'] = 2*max(rear_wheel[:, 3])

    results_history = fit_img2model(data_model, x0, bike_params, boundaries)

    roll_history = []
    pitch_history = []
    yaw_history = []
    steer_history = []
    x_history = []
    y_history = []
    z_history = []

    for i in results_history:
        roll_history.append(i['x'][0])
        pitch_history.append(i['x'][1])
        yaw_history.append(i['x'][2])
        steer_history.append(i['x'][3])
        x_history.append(i['x'][4])
        y_history.append(i['x'][5])
        z_history.append(i['x'][6])

    data_sim = {'phi':roll_history, 'theta':pitch_history, 'psi':yaw_history, 
                'delta':steer_history, 'xr':x_history, 'yr':y_history, 
                'zr':z_history}

    front_wheel_fit_x = x_history + data_model['r_Cf_Cr_x']
    front_wheel_fit_z = z_history + data_model['r_Cf_Cr_z']

    end_time = time.time()
    print(f'Execution time: {end_time - start_time}')

    # ----- Store processed data -----
    with open('/home/eimolgon/Documents/PhD-Project/vid2dyn/output/data4model_try1.json', 'w') as f:
        json.dump(data_model, f)

    # ----- Plotting -----
    if args.animate_points == 'data':
        animate_Point(tracking_data, screen_resolution, first_frame, save=args.save)
    elif args.animate_points == 'model':
        animate_Point(tracking_data, screen_resolution, first_frame, save=args.save)
    elif args.animate_ellipses == 'data':
        animate_Ellipse(tracking_data, screen_resolution, first_frame, save=args.save)
    
    if args.plot == 'all':
        plt.plot(np.rad2deg(roll_history), label = 'solver data')
        plt.title('Roll angle')
        plt.xlabel('Frame')
        plt.ylabel('Angle')
        plt.show()

        plt.plot(np.rad2deg(pitch_history), label = 'solver data')
        plt.title('Pitch angle')
        plt.xlabel('Frame')
        plt.ylabel('Angle')
        plt.show()

        plt.plot(np.rad2deg(yaw_history), label = 'solver data')
        plt.title('Yaw angle')
        plt.xlabel('Frame')
        plt.ylabel('Angle')
        plt.show()

        plt.plot(np.rad2deg(steer_history), label = 'solver data')
        plt.title('Steer angle')
        plt.xlabel('Frame')
        plt.ylabel('Angle')
        plt.show()

        plt.plot(x_history, label = 'solver data')
        plt.plot(rear_wheel[:, 0], label = 'image data')
        plt.legend()
        plt.title('x position')
        plt.xlabel('Frame')
        plt.ylabel('Pixels')
        plt.show()

        plt.plot(y_history, label = 'solver data')
        plt.title('y position')
        plt.xlabel('Frame')
        plt.ylabel('Pixels')
        plt.show()

        plt.plot(z_history, label = 'solver data')
        plt.plot(rear_wheel[:, 1], label = 'image data')
        plt.legend()
        plt.title('z position')
        plt.xlabel('Frame')
        plt.ylabel('Pixels')
        plt.show()

        plt.plot(x_history, z_history, c='C0', linestyle='dashed', label='solver data')
        plt.plot(rear_wheel[:,0], rear_wheel[:, 1], c='C1', linestyle='solid', label='Video data')
        plt.legend()
        plt.title('x vs z position')
        plt.xlabel('x position')
        plt.ylabel('z position')
        plt.show()

else:
    print('No hi')
