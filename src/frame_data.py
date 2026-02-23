import os
import json
import math
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
from collections import defaultdict
from scipy.signal import butter, filtfilt
from bike_model import *

import tests


def readFile(file_path):
    '''
    Read all the points on the file and split them by class.
    Input: .txt file with the points of an ellipse.
    Output: Dictionary with points divided by class id.
    '''

    with open(file_path, 'r') as f:
        lines = f.readlines()

    ellipses = []

    for line in lines:
        parts = list(map(float, line.strip().split()))

        class_id = int(parts[0])
        points = np.array(parts[1:]).reshape(-1, 2)

        ellipses.append({
            'class_id': class_id,
            'points': points
        })

    return ellipses
    

def fitEllipse(points, screen_res:tuple):
    '''
    Recreates the ellipse to output parameters and angle to the camera plane.
    Input: points, screen_resolution.
    Output: fitted ellipse
    '''

    ell = EllipseModel()
    ell.estimate(points)
    x, z, a, b, theta = ell.params

    x = x*screen_res[0]
    z = screen_res[1]*(1-z)
    a = a*screen_res[1]
    b = b*screen_res[0]


    ellipse_data = (x, z, a, b, theta)

    # Unit test -----
    b_test = 10
    a_test = 10
    assert np.arccos(b_test/a_test) == 0, 'Arccos(1) != 0'
    # ----- ----- -----

    return ellipse_data


def find_points(ellipse):
    '''
    Find extreme points of the ellipse.
    Input: ellipse parameters.
    Output: p1, p2, p3, and p4.
    '''

    x = ellipse[0]
    z = ellipse[1]
    a = ellipse[2]
    b = ellipse[3]
    theta = ellipse[4]
    
    p1 = (x - b*np.sin(theta), z + b*np.cos(theta))
    p2 = (x - a*np.cos(theta), z - a*np.sin(theta))
    p3 = (x + b*np.sin(theta), z - b*np.cos(theta))
    p4 = (x + a*np.cos(theta), z + a*np.sin(theta))

    point_list = [p1[1], p2[1], p3[1], p4[1]]
    bottom_point_idx = point_list.index(min(point_list))

    if bottom_point_idx == 0:
        bottom_point = p1
        upper_point = p3
    elif bottom_point_idx == 1:
        bottom_point = p2
        upper_point = p4
    elif bottom_point_idx == 2:
        bottom_point = p3
        upper_point = p1
    else:
        bottom_point = p4
        upper_point = p2


    return (p1, p2, p3, p4, upper_point, bottom_point)


def process_directory(directory_path, screen_resolution):
    '''
    Process directory to extract the data from .txt files.
    Input: Path to the directory where the .txt are located.
    Output: Dictionary with the points, frame number, and reconstructed ellipses.
    '''

    files = [f for f in os.listdir(directory_path) if f.endswith('.txt')]
    files.sort()

    comodin = files[0].split('_')
    first_frame = int((comodin[1].split('.'))[0])
    
    tracking_data = defaultdict(lambda: {
        'frames': [], 'points': [], 'ellipses': [], 'extremes': []
    })
    
    for frame_idx, filename in enumerate(files):
        file_path = os.path.join(directory_path, filename)
        for ellipse in readFile(file_path):
            fitted_ellipse = fitEllipse(ellipse['points'], screen_resolution)
            class_id = ellipse['class_id']
            tracking_data[class_id]['points'].append(ellipse['points'])
            tracking_data[class_id]['frames'].append(frame_idx)
            tracking_data[class_id]['ellipses'].append(fitted_ellipse)
            tracking_data[class_id]['extremes'].append(find_points(fitted_ellipse))

    tracking_data['first_frame'] = first_frame
    return tracking_data


def butter_lowpass(cutoff, fs, order=5):
    '''
    Butterworth lowpass filter.
    Input:
    Output:
    ''' 
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def lowpass_filter(data, cutoff=2.0, fs=30.0, order=5):
    '''
    Apply lowpass filter to data.
    Input:
    Output:
    '''

    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def apply_filters(tracking_data, cutoff=2.0, fs=30.0):
    '''
    Apply lowpass filters to all tracking data.
    Input: Unfiltered data.
    Output: Filtered data.
    '''
    
    filtered_data = {}
    for class_id, data in tracking_data.items():
        if class_id == 0 or class_id == 1:
            filtered = {
                'x_pos': lowpass_filter(data['x_pos'], cutoff, fs),
                'y_pos': lowpass_filter(data['y_pos'], cutoff, fs),
                'major_axes': lowpass_filter(data['major_axes'], cutoff, fs),
                'minor_axes': lowpass_filter(data['minor_axes'], cutoff, fs),
                'frames': data['frames']
            }
            filtered_data[class_id] = filtered

    return filtered_data


def animate_Point(data, resolution, first_frame, save=False):
    '''
    Create 2d point animation
    Input: Data of the points.
    Output: Plot the points in 2d.
    '''
    
    

    fig, ax = plt.subplots(figsize = (16,9))
    ax.axis('equal')
    # plt.gca().set_aspect('equal')

    ellipse_f = np.array(data[0]['ellipses'])
    xf = np.asarray(ellipse_f[:,0])
    zf = np.asarray(ellipse_f[:,1])

    ellipse_r = np.array(data[1]['ellipses'])
    xr = np.asarray(ellipse_r[:,0])
    zr = np.asarray(ellipse_r[:,1])

    line2 = ax.plot(xf[0], zf[0], label=f'front wheel')[0]
    dot2 = ax.plot(xf[0], zf[0], 'bo')[0]
    line3 = ax.plot(xr[0], zr[0], label=f'rear wheel')[0]
    dot3 = ax.plot(xr[0], zr[0], 'ro')[0]

    ax.set(xlim=[0, resolution[0]], ylim=[0, resolution[1]], xlabel='X [pixels]', ylabel='Y [pixels]')


    def update(frame):
        # for each frame, update the data stored on each artist.
        xf2 = xf[:frame]
        zf2 = zf[:frame]

        xr2 = xr[:frame]
        zr2 = zr[:frame]

        # update the line plot:
        line2.set_xdata(xf2[:frame])
        line2.set_ydata(zf2[:frame])
        dot2.set_data([xf[frame]], [zf[frame]])

        line3.set_xdata(xr2[:frame])
        line3.set_ydata(zr2[:frame])
        dot3.set_data([xr[frame]], [zr[frame]])

        ax.set_title(f'Frame: {first_frame + frame}, Time: {round(frame/30, 2)} s')
    
        return line2, line3


    animPoint = animation.FuncAnimation(fig=fig, func=update, frames=len(xf), interval=60)
    plt.show()

    if save == True:
        animPoint.save(filename=f"{filename}-animation-{date}.gif")

    return


def animate_Ellipse(data, resolution, first_frame, save=False):
    '''
    Create 3d animation
    '''

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.axis('equal')

    fwheel = np.array(data[0]['ellipses'])[:, 0:5]
    rwheel = np.array(data[1]['ellipses'])[:, 0:5]

    xf = np.asarray(fwheel[:,0])
    yf = np.asarray(fwheel[:,1])
    af = np.asarray(fwheel[:, 2])
    bf = np.asarray(fwheel[:, 3])
    thetaf = np.asarray(fwheel[:, 4])

    xr = rwheel[:,0]
    yr = rwheel[:,1]
    ar = np.asarray(rwheel[:, 2])
    br = np.asarray(rwheel[:, 3])
    thetar = np.asarray(rwheel[:, 4])


    # Create ellipse patches (initial position)
    front_ellipse = Ellipse((xf[0], yf[0]), width=2*bf[0], height=2*af[0], 
                            angle=thetaf[0], edgecolor='blue',  facecolor='none')
    rear_ellipse = Ellipse((xr[0], yr[0]), width=2*br[0], height=2*ar[0], 
                        angle=thetar[0], edgecolor='red', facecolor='none')

    ax.add_patch(front_ellipse)
    ax.add_patch(rear_ellipse)


    (line,) = ax.plot([xr[0], xf[0]], [yr[0], yf[0]], 'k-', lw=2)


    ax.set(xlim=[0, resolution[0]], ylim=[0, resolution[1]], xlabel='X [pixels]', ylabel='Y [pixels]')


    def init():
        front_ellipse.set_center((xf[0], yf[0]))
        front_ellipse.set_width(bf[0]*2)
        front_ellipse.set_height(af[0]*2)
        front_ellipse.set_angle(thetaf[0])

        rear_ellipse.set_center((xr[0], yr[0]))
        rear_ellipse.set_width(br[0]*2)
        rear_ellipse.set_height(ar[0]*2)
        rear_ellipse.set_angle(thetar[0])

        line.set_data([xr[0], xf[0]], [yr[0], yf[0]])
        # line_rs.set_data([xr[0], xs[0]], [yr[0], ys[0]])
        # line_sf.set_data([xs[0], xf[0]], [ys[0], yf[0]])

        return (front_ellipse, rear_ellipse, line, )


    def update(frame):
        front_ellipse.set_center((xf[frame], yf[frame]))
        front_ellipse.set_width(bf[frame]*2)
        front_ellipse.set_height(af[frame]*2)
        front_ellipse.set_angle(thetaf[frame])

        rear_ellipse.set_center((xr[frame], yr[frame]))
        rear_ellipse.set_width(br[frame]*2)
        rear_ellipse.set_height(ar[frame]*2)
        rear_ellipse.set_angle(thetar[frame])


        line.set_data([xr[frame], xf[frame]], [yr[frame], yf[frame]])

        # line_rs.set_data([xr[frame], xs_frame], [yr[frame], ys_frame])
        # line_sf.set_data([xs_frame, xf[frame]], [ys_frame, yf[frame]])    

        ax.set_title(f'Frame: {first_frame + frame}, Time: {round(frame/30, 2)} s')

        return (front_ellipse, rear_ellipse, line, )



    animEllipse = animation.FuncAnimation(fig=fig, func=update, frames=len(xf), interval=60)
    plt.show()

    if save == True:
        aniEllipse.save(filename=f"{filename}-animation-{date}.gif")

    return


def clean4json(obj):
    # numpy arrays -> lists
    if isinstance(obj, np.ndarray):
        return [clean4json(x) for x in obj.tolist()]
    # numpy scalars -> python scalars
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    # handle floats: replace NaN/Inf with None (strict JSON)
    if isinstance(obj, float):
        if not math.isfinite(obj):
            # return None
            return 0
        return obj
    # containers
    if isinstance(obj, dict):
        return {k: clean4json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean4json(x) for x in obj]
    # pass through other JSON-safe types (str, int, bool, None)
    return obj


def data4model(track_data, assumed_origin):
    '''
    Create the nested dictionary by frame for the data required in the model of the bicycle.
    '''

    data = {
        'r_Cf_Cr_x':0,
        'r_Cf_Cr_z':0,
        'r_Cr_O_x':0,
        'r_Cr_O_z':0,
        'r_P1r_Cr_x':0,
        'r_P1r_Cr_z':0,
        'r_P3r_Cr_x':0,
        'r_P3r_Cr_z':0,
        'r_P1f_Cf_x':0,
        'r_P1f_Cf_z':0,
        'r_P3f_Cf_x':0,
        'r_P3f_Cf_z':0,
    }

    rear_wheel = {
        'x': 0,
        'z': 0,
        'psi': 0
    }


    ellipse_f = np.array(track_data[0]['ellipses'])
    extremes_f = np.array(track_data[0]['extremes'])
    ellipse_r = np.array(track_data[1]['ellipses'])
    extremes_r = np.array(track_data[1]['extremes'])

    xf = np.array(ellipse_f[:,0])
    zf = np.array(ellipse_f[:,1])
    af = ellipse_f[:,2]
    bf = ellipse_f[:,3]
    theta_f = ellipse_f[:,4]
    upper_f = extremes_f[:, 4]
    bottom_f = extremes_f[:, 5]


    xr = np.array(ellipse_r[:,0])
    zr = np.array(ellipse_r[:,1])
    ar = ellipse_r[:,2]
    br = ellipse_r[:,3]
    theta_r = ellipse_r[:,4]
    upper_r = extremes_r[:, 4]
    bottom_r = extremes_r[:, 5]

    

    rear_wheel['x'] = xr
    rear_wheel['z'] = zr

    data['r_Cf_Cr_x'] = xf - xr
    data['r_Cf_Cr_z'] = zf - zr
    data['r_Cr_O_x'] = xr - assumed_origin[0]
    data['r_Cr_O_z'] = zr - assumed_origin[1]
    # data['r_P_S_x'] = (np.sin(q1_f)*np.cos(q1_r) - np.sin(q1_r)*np.cos(q1_f)*np.cos(theta_f - theta_r))*np.sin(theta_f)/np.sqrt((np.sin(q1_f)*np.cos(q1_r) - np.sin(q1_r)*np.cos(q1_f)*np.cos(theta_f - theta_r))**2 + np.sin(q1_r)**2*np.sin(theta_f - theta_r)**2) + np.sin(q1_r)*np.sin(theta_f - theta_r)*np.cos(q1_f)*np.cos(theta_f)/np.sqrt((np.sin(q1_f)*np.cos(q1_r) - np.sin(q1_r)*np.cos(q1_f)*np.cos(theta_f - theta_r))**2 + np.sin(q1_r)**2*np.sin(theta_f - theta_r)**2)
    # data['r_P_S_z'] = (np.sin(q1_f)*np.cos(q1_r) - np.sin(q1_r)*np.cos(q1_f)*np.cos(theta_f - theta_r))*np.cos(theta_f)/np.sqrt((np.sin(q1_f)*np.cos(q1_r) - np.sin(q1_r)*np.cos(q1_f)*np.cos(theta_f - theta_r))**2 + np.sin(q1_r)**2*np.sin(theta_f - theta_r)**2) - np.sin(q1_r)*np.sin(theta_f)*np.sin(theta_f - theta_r)*np.cos(q1_f)/np.sqrt((np.sin(q1_f)*np.cos(q1_r) - np.sin(q1_r)*np.cos(q1_f)*np.cos(theta_f - theta_r))**2 + np.sin(q1_r)**2*np.sin(theta_f - theta_r)**2)
    data['r_P1f_Cf_x'] = upper_r[:, 0] - xr
    data['r_P1f_Cf_z'] = upper_r[:, 1] - zr
    data['r_P3f_Cf_x'] = bottom_r[:, 0] - xr
    data['r_P3f_Cf_z'] = bottom_r[:, 1] - zr
    data['r_P1r_Cr_x'] = upper_f[:, 0] - xf
    data['r_P1r_Cr_z'] = upper_f[:, 1] - zf
    data['r_P3r_Cr_x'] = bottom_f[:, 0] - xf
    data['r_P3r_Cr_z'] = bottom_f[:, 1] - zf

    data_json = clean4json(data)
    rw_json = clean4json(rear_wheel)

    return data_json, ellipse_r, ellipse_f


def point_plot(points):
    '''
    Plot a set of points.
    Input: Points.
    Output: Plot.
    '''

# ----- Tests -----



# ----- Run -----

# Directory used for testing /home/eimolgon/Documents/PhD-Project/02-vide-data/gopro_test_1_1_cut/labels/Train/

rawdate = datetime.datetime.now()
date =  rawdate.strftime('%Y') + rawdate.strftime('%m') + rawdate.strftime('%d')


screen_resolution = (1920, 1080)
assumed_origin = (screen_resolution[0]*0.2, screen_resolution[1]*0.2)
wheel_diameter = 0.6604



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


plot_group.add_argument('-a2d', '--animate_points', 
                    default=False, action='store_true', 
                    help='Plot 2D animation')
plot_group.add_argument('-a3d', '--animate_ellipses', 
                    default=False, action='store_true', 
                    help='Plot 3D animation')



args = parser.parse_args()


if args.single_frame != '0':

    # Test frame /home/eimolgon/Documents/PhD-Project/02-video-data/gopro_test_1_1_resize_annotation_161025/labels/Train/frame_000417.txt

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
    raw_data = process_directory(args.data, screen_resolution)
    
    if args.filter:
        tracking_data = apply_filters(raw_data)
    else:
        tracking_data = raw_data

    data_model, rear_wheel, front_wheel = data4model(tracking_data, assumed_origin)
    first_frame = tracking_data['first_frame']

    with open('/home/eimolgon/Documents/PhD-Project/vid2dyn/output/data4model_try1.json', 'w') as f:
        json.dump(data_model, f)

    if args.animate_points:
        animate_Point(tracking_data, screen_resolution, first_frame)
    elif args.animate_ellipses:
        animate_Ellipse(tracking_data, screen_resolution, first_frame)
else:
    print('No hi')
