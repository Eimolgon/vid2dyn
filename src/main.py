import json
import time
import datetime
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import bike_model as ortpro
import bike_model_perspective as perpro
from matplotlib.patches import Ellipse
from scipy.optimize import least_squares
# from bike_model import *
from skimage.draw import ellipse_perimeter


from utils import (
    readFile,
    fitEllipse,
    find_points,
    process_directory,
    apply_filters,
    animate_Point,
    animate_Ellipse,
    data4model,
    fit_img2model,
    plot_mbd_model_2d,
    plot_mbd_model_3d,
    plot_dots,
    set_axes_equal,
    generate_synthetic_data
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
                    help='Choose data directory. Use \'test\' ' \
                    'for synthetic data ')
parser.add_argument('-sf', '--single_frame', 
                    default='0', type=int,
                    help='Extract data from a single frame, requires route to ' \
                    'file')
parser.add_argument('-p', '--plot',
                    type=str, default='0',
                    help='Allow plotting of the results')
parser.add_argument('--save', default='none', type=str,
                    help='Save plots')
parser.add_argument('-csv', '--save_csv',
                    default=False, action='store_true',
                    help='Store solver results in csv file')


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
image_resolution = (6020, 4024)
sensor_size = (0.0235, 0.0156)
assumed_origin = (screen_resolution[0]*0.2, screen_resolution[1]*0.2)
wheel_diameter = 0.6604
focal_length = 0.028
# bike_params = {
#     'lr': 140,
#     'lf1': 50,
#     'lf2': 10,
#     'rf': 75,
#     'rr': 75 
# }

bike_params = {
    'lr': 1.4,
    'lf1': 0.5,
    'lf2': 0.1,
    'rf': wheel_diameter/2,
    'rr': wheel_diameter/2 
}


# ----- Solver boundaries -----
lower_bound = [-np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2, -20.0, -20.0, -20.0]
upper_bound = [np.pi/2, np.pi/2, np.pi/2, np.pi/2, 20.0, 20.0, 20.0]
boundaries = (lower_bound, upper_bound)

# camera_params = {'f': 24,
#                  'cx': screen_resolution[0]*0.5, 
#                  'cy': screen_resolution[1]*0.5}

camera_params = {
        'fx': focal_length * image_resolution[0] / sensor_size[0],
        'fy': focal_length * image_resolution[1] / sensor_size[1],
        'cx': image_resolution[0] / 2,
        'cy': image_resolution[1] / 2,
        'cam_x': 0.0,
        'cam_y': 0.0,
        'cam_z': 0.45,
        'cam_yaw': 0.0,
        'cam_pitch': 0.0,
        'cam_roll': 0.0
    }

# === Initial guess ===



if args.single_frame < 400 :
    # x0 = np.array([
    #     np.deg2rad(-5),
    #     np.deg2rad(-15), 
    #     np.deg2rad(-45),
    #     np.deg2rad(15),
    #     1.0,
    #     1.0,
    #     1.0
    # ])
    x0 = np.array([
            np.deg2rad(0),
            np.deg2rad(0), 
            np.deg2rad(0),
            np.deg2rad(0),
            2.82,
            0.1,
            0
        ])
elif args.single_frame < 470 :
    x0 = np.array([
        np.deg2rad(0),
        np.deg2rad(-21.8), 
        np.deg2rad(0),
        np.deg2rad(0),
        5.0,
        5.0,
        5.0
    ])
else:
    # x0 = np.array([
    #     np.deg2rad(-5),
    #     np.deg2rad(-30), 
    #     np.deg2rad(75),
    #     np.deg2rad(15),
    #     1300,
    #     100,
    #     650
    # ])
    x0 = np.array([
        np.deg2rad(-5),
        np.deg2rad(-30), 
        np.deg2rad(75),
        np.deg2rad(15),
        10.0,   # Forward
        2.0,    # Lateral
        1.0     # Vertical
    ])

# That last comes from the AI, check later. Now the initial guess is in m and 
# not in pixels



if args.single_frame != 0 :

    framenum = args.single_frame
    # test_frame_single = f'/home/eimolgon/Documents/PhD-Project/02-video-data/gopro_test_1_1_cut/labels/Train/frame_000{framenum}.txt'

    test_frame_single = f'/home/eimolgon/Documents/PhD-Project/02-video-data/bike-base01/labels/train/DSC09360.txt'

    # Process raw data
    data_points = readFile(test_frame_single)
    front_data = data_points[0]['points']
    rear_data = data_points[1]['points']

    # Fit ellipse to data
    ellipse_front = fitEllipse(front_data, screen_resolution)
    ellipse_rear = fitEllipse(rear_data, screen_resolution)


    # Extract parameters from the fitted ellipses
    xf, zf, af, bf, theta_f = ellipse_front
    xr, zr, ar, br, theta_r = ellipse_rear

    # Check travel direction
    if xf > xr:
        travel_direction = 'right'
    elif xf < xr:
        travel_direction = 'left'

    # Obtain projected angles of the ellipses
    if af > bf:
        angle_ellipse_f = np.arccos(bf/af)
    else:
        angle_ellipse_f = 0

    if ar > br:
        angle_ellipse_r = np.arccos(br/ar)    
    else: 
        angle_ellipse_r = 0


    # ----- This is just another test ----- 
    # aff,bff,cff,dff,eff,fff = fitEllipse_conic(front_data, screen_resolution)
    # normal_f = get_normal(aff,bff,cff,dff,eff,fff)

    # arr, brr, crr, drr, err, frr = fitEllipse_conic(rear_data, screen_resolution)
    # normal_r = get_normal(arr, brr, crr, drr, err, frr)

    # print(f'normal front: {normal_f}')
    # new_angle2camera_f = normal_f/np.linalg.norm(normal_f)

    # print(f'normal rear: {normal_r}')
    # new_angle2camera_r = normal_r/np.linalg.norm(normal_r)
    
    # print(f'old angle f: {np.rad2deg(angle_ellipse_f):.2f}')
    # print(f'new angle f: {np.rad2deg(new_angle2camera_f[1]):.2f}')

    # print(f'old angle r: {np.rad2deg(angle_ellipse_r):.2f}')
    # print(f'new angle r: {np.rad2deg(new_angle2camera_r[1]):.2f}')

    # ----- ----- ----- ----- -----

    # Find extreme points of the ellipses
    p1_f, p2_f, p3_f, p4_f, p2_f_2, p4_f_2 = find_points(ellipse_front, travel_direction)
    p1_r, p2_r, p3_r, p4_r, p2_r_2, p4_r_2 = find_points(ellipse_rear, travel_direction)


    plot_points_f = (xf, zf, p1_f, p3_f)
    plot_points_r = (xr, zr, p1_r, p3_r)

    imgdata = {
        'r_Cf_Cr_u' : [xf - xr],
        'r_Cf_Cr_v' : [zf - zr],
        'u_Cr' : [xr],
        'v_Cr' : [zr],
        'r_P1r_Cr_u' : [p1_r[0] - xr],
        'r_P1r_Cr_v' : [p1_r[1] - zr],
        'r_P3r_Cr_u' : [p3_r[0] - xr],
        'r_P3r_Cr_v' : [p3_r[1] - zr],
        'r_P1f_Cf_u' : [p1_f[0] - xf],
        'r_P1f_Cf_v' : [p1_f[1] - zf],
        'r_P3f_Cf_u' : [p3_f[0] - xf],
        'r_P3f_Cf_v' : [p3_f[1] - zf],
        'r_Q_S_u' : [(np.sin(angle_ellipse_f)*np.cos(angle_ellipse_r) 
                     - np.sin(angle_ellipse_r)*np.cos(angle_ellipse_f)
                     * np.cos(theta_f - theta_r))*np.sin(theta_f)/ \
                        np.sqrt((np.sin(angle_ellipse_f)*np.cos(angle_ellipse_r) 
                                 - np.sin(angle_ellipse_r)*np.cos(angle_ellipse_f)*
                                 np.cos(theta_f - theta_r))**2 + 
                                 np.sin(angle_ellipse_r)**2*
                                 np.sin(theta_f - theta_r)**2) + \
                                    np.sin(angle_ellipse_r)*\
                                        np.sin(theta_f - theta_r)*\
                                            np.cos(angle_ellipse_f)*\
                                                np.cos(theta_f)/ \
                                                    np.sqrt((np.sin(angle_ellipse_f)*
                                                             np.cos(angle_ellipse_r) - 
                                                             np.sin(angle_ellipse_r)*
                                                             np.cos(angle_ellipse_f)*
                                                             np.cos(theta_f - theta_r))**2 + 
                                                             np.sin(angle_ellipse_r)**2*
                                                             np.sin(theta_f - theta_r)**2)],
        'r_Q_S_v' : [(np.sin(angle_ellipse_f)*np.cos(angle_ellipse_r) - 
                     np.sin(angle_ellipse_r)*np.cos(angle_ellipse_f)*
                     np.cos(theta_f - theta_r))*np.cos(theta_f)/ \
                        np.sqrt((np.sin(angle_ellipse_f)*
                                 np.cos(angle_ellipse_r) - 
                                 np.sin(angle_ellipse_r)*
                                 np.cos(angle_ellipse_f)*
                                 np.cos(theta_f - theta_r))**2 + 
                                 np.sin(angle_ellipse_r)**2*
                                 np.sin(theta_f - theta_r)**2) - \
                                    np.sin(angle_ellipse_r)*np.sin(theta_f)* \
                                        np.sin(theta_f - theta_r)* \
                                            np.cos(angle_ellipse_f)/ \
                                                np.sqrt((np.sin(angle_ellipse_f)*
                                                         np.cos(angle_ellipse_r) - 
                                                         np.sin(angle_ellipse_r)*
                                                         np.cos(angle_ellipse_f)*
                                                         np.cos(theta_f - theta_r))**2 + 
                                                         np.sin(angle_ellipse_r)**2*
                                                         np.sin(theta_f - theta_r)**2)]

    }

    # Set wheel radii
    # bike_params['rf'] = af
    # bike_params['rr'] = ar    

    results_sf, state_sf = fit_img2model(imgdata, x0, bike_params, 
                                         boundaries, camera_params, perpro)
    print(f'phi = {np.rad2deg(state_sf[-1][0]):.2f}')
    print(f'theta = {np.rad2deg(state_sf[-1][1]):.2f}')
    print(f'psi = {np.rad2deg(state_sf[-1][2]):.2f}')
    print(f'delta = {np.rad2deg(state_sf[-1][3]):.2f}')
    print(f'x = {state_sf[-1][4]:.2f}')
    print(f'y = {state_sf[-1][5]:.2f}')
    print(f'z = {state_sf[-1][6]:.2f}')


    if args.plot == 'e':
        
        plt.plot(xf, zf, 'ok')
        plt.scatter(p1_f[0], p1_f[1], marker='x', color='red')
        plt.scatter(p3_f[0], p3_f[1], marker='x', color='black')

        plt.scatter(p2_f_2[0], p2_f_2[1], marker='p', color='red')
        plt.scatter(p4_f_2[0], p4_f_2[1], marker='h', color='black')

        plt.plot(xr, zr, 'ok')
        plt.scatter(p1_r[0], p1_r[1], marker='x', color='red')
        plt.scatter(p3_r[0], p3_r[1], marker='x', color='black')

        plt.scatter(p2_r_2[0], p2_r_2[1], marker='p', color='red')
        plt.scatter(p4_r_2[0], p4_r_2[1], marker='h', color='black')

        ell_patch_f = Ellipse((xf, zf), width = 2*af, height = 2*bf,
                      angle = theta_f*180/np.pi, edgecolor='blue', 
                      facecolor='none', linestyle='solid')

        ell_patch_r = Ellipse((xr, zr), width = 2*ar, height = 2*br,
                      angle = theta_r*180/np.pi, edgecolor='red', 
                      facecolor='none', linestyle='solid')
        

        plt.gca().add_patch(ell_patch_f)
        plt.gca().add_patch(ell_patch_r)

        plt.xlim(0, 1980)
        plt.ylim(0, 1080)
        plt.gca().set_aspect('equal')
        plt.grid()

        fig, axs = plt.subplots()

        # plot_mbd_model_2d(axs,bike_params, x0, 'xz')
        plot_mbd_model_2d(axs, bike_params, state_sf[-1], 'xz')
        plt.show()

        # fig2, ax2 = plt.subplots(2,2)


    elif args.plot == 'mbd':

        # 2d plot
        fig, axs = plt.subplots()

        plot_dots(axs, plot_points_f, 'front')
        plt.scatter(front_data[:, 0]*screen_resolution[0], 
                    (1 - front_data[:, 1])*screen_resolution[1], 
                    facecolors='none', edgecolors='red', s=10)

        plot_dots(axs, plot_points_r, 'rear')
        plt.scatter(rear_data[:, 0]*screen_resolution[0], 
                    (1 - rear_data[:, 1])*screen_resolution[1], 
                    facecolors='none', edgecolors='blue', s=10)

        # plot_mbd_model_2d(axs,bike_params, x0, 'guess')
        plot_mbd_model_2d(axs, bike_params, state_sf[-1], 'model')
        axs.grid()

        plt.xlim(0, 1920)
        plt.ylim(0, 1080)
        plt.grid()
        plt.show()


        # 3d plot
        fig_3d = plt.figure()
        ax_3d = fig_3d.add_subplot(111, projection='3d')

        plot_mbd_model_3d(bike_params, state_sf[-1], 'solver', ax=ax_3d)
        plot_mbd_model_3d(bike_params, x0, 'initial_guess', ax=ax_3d)

        ax_3d.set_xlabel('x axis')
        ax_3d.set_ylabel('y axis')
        ax_3d.set_zlabel('z axis')

        set_axes_equal(ax_3d)

        plt.show()


elif args.data == 'test':

    # ----- Test parameters -----
    bike_params = {
    'lr': 1.4,
    'lf1': 0.5,
    'lf2': 0.1,
    'rf': wheel_diameter/2,
    'rr': wheel_diameter/2 
    }

    camera_params = {
            'fx': focal_length * image_resolution[0] / sensor_size[0],
            'fy': focal_length * image_resolution[1] / sensor_size[1],
            'cx': image_resolution[0] / 2,
            'cy': image_resolution[1] / 2,
            'cam_x': 0.0,
            'cam_y': 0.0,
            'cam_z': 0.0,
            'cam_yaw': 0.0,
            'cam_pitch': 0.0,
            'cam_roll': 0.0
        }

    lower_bound = [-np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2, -20.0, -20.0, -20.0]
    upper_bound = [np.pi/2, np.pi/2, np.pi/2, np.pi/2, 20.0, 20.0, 20.0]
    boundaries = (lower_bound, upper_bound)

    
    true_state = np.array([
        np.deg2rad(-5.0),      # phi
        np.deg2rad(-10.0),     # theta
        np.deg2rad(20.0),      # psi
        np.deg2rad(5.0),       # delta

        2.0,                   # x_r [m]
        8.0,                   # y_r [m]
        0.0                    # z_r [m]
    ])

    synth_data = generate_synthetic_data(true_state, bike_params, 
                                         camera_params, perpro)

    x0_test = np.array([
        np.deg2rad(5.0),
        np.deg2rad(0.0),
        np.deg2rad(0.0),
        np.deg2rad(0.0),
        1.0,
        1.0,
        1.0
    ])

    results, states = fit_img2model(
        synth_data,
        x0_test,
        bike_params,
        (lower_bound, upper_bound),
        camera_params,
        perpro
    )

    estimated_state = states[-1]

    print("True state:")
    print(true_state)

    print("\nEstimated state:")
    print(estimated_state)

    print("\nError:")
    print(estimated_state - true_state)

    print("\nPosition [m]")

    print("True:")
    print(true_state[4:])

    print("Estimated:")
    print(estimated_state[4:])



elif args.data == '1':

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
    
    bike_params['rr'] = max(rear_wheel[:, 2])
    bike_params['rf'] = max(front_wheel[:, 2])

    results_history, x0_history = fit_img2model(data_model, x0, bike_params, boundaries)

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

    # ----- Test print -----
    print('This is printing the results for frame 396')
    print(f'roll:{np.rad2deg(roll_history[7]):.2f}')
    print(f'pitch:{np.rad2deg(pitch_history[7]):.2f}')
    print(f'yaw:{np.rad2deg(yaw_history[7]):.2f}')
    print(f'steer:{np.rad2deg(steer_history[7]):.2f}')

    end_time = time.time()
    print(f'Execution time: {end_time - start_time}')

    if args.save_csv:
        df = pd.DataFrame(data_sim)
        df.to_csv('datacsv_bg.csv')

    # ----- Store processed data -----
    # with open('/home/eimolgon/Documents/PhD-Project/vid2dyn/output/data4model_try1.json', 'w') as f:
    #     json.dump(data_model, f)

    # ----- Plotting -----
    if args.animate_points == 'data':
        animate_Point(tracking_data, screen_resolution, first_frame, name=args.save)
    elif args.animate_points == 'model':
        animate_Point(tracking_data, screen_resolution, first_frame, name=args.save)
    elif args.animate_ellipses == 'data':
        animate_Ellipse(tracking_data, screen_resolution, first_frame, name=args.save)
    
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
