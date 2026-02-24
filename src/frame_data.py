import os
import json
import math
import time
import random
import datetime
import argparse
import numpy as np
import sympy as sp
import sympy.physics.mechanics as me
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Ellipse
from matplotlib import collections as mc
from skimage.measure import EllipseModel
from scipy.optimize import least_squares
from scipy.signal import butter, filtfilt
from collections import defaultdict
from bike_model import *

from utils import (
    readFile,
    fitEllipse,
    find_points,
    process_directory,
    butter_lowpass,
    lowpass_filter,
    apply_filters,
    animate_Point,
    animate_Ellipse,
    clean4json,
    data4model
)


# ----- Tests -----


# ----- Parser setup -----

parser = argparse.ArgumentParser(description='Inverse kinematics from ellipses')
plot_group = parser.add_argument_group('Plotting options')


parser.add_argument('-f', '--filter', 
                    default=False, action='store_true', 
                    help='Apply low pass filter')
parser.add_argument('-d', '--data', 
                    type=str, default='0',
                    help='Route to the data directory')
parser.add_argument('-sf', '--single_frame', 
                    default='0', type=str,
                    help='Extract data from a single frame, requires route to ' \
                    'file')
parser.add_argument('-p', '--plot',
                    default=False, action='store_true',
                    help='Allow plotting of the results')
parser.add_argument('--save', action='store_true',
                    help='Save plots')


plot_group.add_argument('-a2d', '--animate_points', 
                    default=False, action='store_true', 
                    help='Plot points animation')
plot_group.add_argument('-a3d', '--animate_ellipses', 
                    default=False, action='store_true', 
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

results_history = []

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

# === Data from image ===

img_data = {
    'r_Cf_Cr_x': 0, 
    'r_Cf_Cr_z': 0, 
    'r_Cr_O_x': 0, 
    'r_Cr_O_z': 0,
    'r_P1r_Cr_x': 0,
    'r_P1r_Cr_z': 0,
    'r_P3r_Cr_x': 0,
    'r_P3r_Cr_z': 0,
    'r_P1f_Cf_x': 0,
    'r_P1f_Cf_z': 0,
    'r_P3f_Cf_x': 0,
    'r_P3f_Cf_z': 0
}


if args.single_frame != '0':

    # Test frame /home/eimolgon/Documents/PhD-Project/02-video-data/gopro_test_1_1_cut/labels/Train/frame_000417.txt

    data_points = readFile(args.single_frame)
    front_data = data_points[0]['points']
    rear_data = data_points[1]['points']

    ellipse_front = fitEllipse(front_data, screen_resolution)
    ellipse_rear = fitEllipse(rear_data, screen_resolution)

    xf, zf, af, bf, theta_f = ellipse_front
    xr, zr, ar, br, theta_r = ellipse_rear

    P1f, P2f, P3f, P4f, upper_f, bottom_f = find_points(ellipse_front)

    P1r, P2r, P3r, P4r, upper_r, bottom_r = find_points(ellipse_rear)


    imgdata = {
        'r_Cf_Cr_x' : xf - xr,
        'r_Cf_Cr_z' : zf - zr,
        'r_Cr_O_x' : xr - assumed_origin[0],
        'r_Cr_O_z' : zr - assumed_origin[1],
        'r_P1r_Cr_x' : upper_r[0] - xr,
        'r_P1r_Cr_z' : upper_r[1] - zr,
        'r_P3r_Cr_x' : bottom_r[0] - xr,
        'r_P3r_Cr_z' : bottom_r[1] - zr,
        'r_P1f_Cf_x' : upper_f[0] - xf,
        'r_P1f_Cf_z' : upper_f[1] - zf,
        'r_P3f_Cf_x' : bottom_f[0] - xf,
        'r_P3f_Cf_z' : bottom_f[1] - zf
    }

    bike_params['r'] = 2*br


    print(f'Front wheel centre: ({ellipse_front[0]}, {ellipse_front[1]})')
    # print(f'Front wheel points: {P1f}, {P2f}, {P3f}, {P4f}')
    # print(f'Upper point front wheel: {upper_f}')
    # print(f'Bottom point front wheel: {bottom_f}')
    print(f'Rear wheel centre: ({ellipse_rear[0]}, {ellipse_rear[1]})')

    if args.plot:

        plt.plot(xf, zf, 'xk')
        plt.plot(P1f[0], P1f[1], 'ok')
        plt.plot(P2f[0], P2f[1], 'ob')
        plt.plot(P3f[0], P3f[1], 'or')
        plt.plot(P4f[0], P4f[1], 'og')

        plt.plot(xr, zr, 'xk')
        plt.plot(P1r[0], P1r[1], 'ok')
        plt.plot(P2r[0], P2r[1], 'ob')
        plt.plot(P3r[0], P3r[1], 'or')
        plt.plot(P4r[0], P4r[1], 'og')

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
    raw_data = process_directory(args.data, screen_resolution)
    
    if args.filter:
        tracking_data = apply_filters(raw_data)
    else:
        tracking_data = raw_data

    data_model, rear_wheel, front_wheel = data4model(tracking_data, assumed_origin)
    first_frame = tracking_data['first_frame']
    
    bike_params['r'] = 2*max(rear_wheel[:, 3])

    for i in range(len(rear_wheel)):
        img_data['r_Cf_Cr_x'] = data_model['r_Cf_Cr_x'][i]
        img_data['r_Cf_Cr_z'] = data_model['r_Cf_Cr_z'][i]
        img_data['r_Cr_O_x'] = data_model['r_Cr_O_x'][i]
        img_data['r_Cr_O_z'] = data_model['r_Cr_O_z'][i]
        img_data['r_P1r_Cr_x'] = data_model['r_P1r_Cr_x'][i]
        img_data['r_P1r_Cr_z'] = data_model['r_P1r_Cr_z'][i]
        img_data['r_P3r_Cr_x'] = data_model['r_P3r_Cr_x'][i]
        img_data['r_P3r_Cr_z'] = data_model['r_P3r_Cr_z'][i]
        img_data['r_P1f_Cf_x'] = data_model['r_P1f_Cf_x'][i]
        img_data['r_P1f_Cf_z'] = data_model['r_P1f_Cf_z'][i]
        img_data['r_P3f_Cf_x'] = data_model['r_P3f_Cf_x'][i]
        img_data['r_P3f_Cf_z'] = data_model['r_P3f_Cf_z'][i]

        results_iteration = least_squares(residual_eqs, x0, 
                                          args = (img_data, bike_params), 
                                          bounds = (lower_bound, upper_bound))
        results_history.append(results_iteration)

        # Update initial guess
        x0 = results_iteration['x'].copy()


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

    end_time = time.time()
    print(f'Execution time: {end_time - start_time}')

    # ----- Store processed data -----
    with open('/home/eimolgon/Documents/PhD-Project/vid2dyn/output/data4model_try1.json', 'w') as f:
        json.dump(data_model, f)

    # ----- Plotting -----
    if args.animate_points:
        animate_Point(tracking_data, screen_resolution, first_frame, save=args.save)
    elif args.animate_ellipses:
        animate_Ellipse(tracking_data, screen_resolution, first_frame, save=args.save)
    
    if args.plot:
        plt.plot(roll_history, label = 'solver data')
        plt.title('Roll angle')
        plt.xlabel('Frame')
        plt.ylabel('Angle')
        plt.show()

        plt.plot(pitch_history, label = 'solver data')
        plt.title('Pitch angle')
        plt.xlabel('Frame')
        plt.ylabel('Angle')
        plt.show()

        plt.plot(yaw_history, label = 'solver data')
        plt.title('Yaw angle')
        plt.xlabel('Frame')
        plt.ylabel('Angle')
        plt.show()

        plt.plot(steer_history, label = 'solver data')
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

        plt.plot(x_history, z_history, '-C', label = 'solver data')
        plt.plot(rear_wheel[:,0], rear_wheel[:, 1], '--C', label = 'Video data')
        plt.legend()
        plt.title('x vs z position')
        plt.xlabel('x position')
        plt.ylabel('z position')
        plt.show()

else:
    print('No hi')
