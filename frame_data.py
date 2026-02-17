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


def readEllipse(file_path):
    '''
    Read all the points on the file and split them by class.
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
    

def fitEllipse(points, screen_res):
    '''
    Recreates the ellipse to output parameters and angle to the camera plane
    a: vertical axis
    b: horizontal axis
    '''

    ell = EllipseModel()
    ell.estimate(points)
    x, z, a, b, theta = ell.params

    x = x*screen_res[0]
    z = screen_res[1]*(1-z)
    a = a*screen_res[1]
    b = b*screen_res[0]

    if b <= a:
        q1 = np.arccos(b/a)
        q2 = 0
    else:
        q1 = 0
        q2 = np.arccos(a/b)

    ellipse_data = (x, z, a, b, theta, q1, q2)

    # Unit test -----
    b_test = 10
    a_test = 10
    assert np.arccos(b_test/a_test) == 0, 'Arccos(1) != 0'
    # ----- ----- -----

    return ellipse_data


def process_directory(directory_path, screen_resolution):
    '''
    Process directory to extract the data from .txt files
    '''

    files = [f for f in os.listdir(directory_path) if f.endswith('.txt')]
    files.sort()

    comodin = files[0].split('_')
    first_frame = int((comodin[1].split('.'))[0])
    
    tracking_data = defaultdict(lambda: {
        'frames': [], 'points': [], 'ellipses': []
    })
    
    for frame_idx, filename in enumerate(files):
        file_path = os.path.join(directory_path, filename)
        for ellipse in readEllipse(file_path):
            class_id = ellipse['class_id']
            tracking_data[class_id]['points'].append(ellipse['points'])
            tracking_data[class_id]['frames'].append(frame_idx)
            tracking_data[class_id]['ellipses'].append(fitEllipse(ellipse['points'], screen_resolution))
    
    tracking_data['first_frame'] = first_frame
    return tracking_data


def butter_lowpass(cutoff, fs, order=5):
    '''
    Design a Butterworth lowpass filter 
    ''' 
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def lowpass_filter(data, cutoff=2.0, fs=30.0, order=5):
    '''
    Apply lowpass filter to data
    '''

    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def apply_filters(tracking_data, cutoff=2.0, fs=30.0):
    '''
    Apply lowpass filters to all tracking data
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


def find_points(ellipse:list):
    '''
    Find extreme points of the ellipse.
    
    input: ellipse
        x, z = center
        a = major axis (horizontal)
        b = minor axis (vertical)
        theta = angle
    output: p1, p2, p3, and p4
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
    

    return (p1, p2, p3, p4)


def animate2d(data, resolution, first_frame):
    '''
    Create 2d point animation
    '''
    
    fig, ax = plt.subplots(figsize = (16,9))
    ax.axis('equal')
    # plt.gca().set_aspect('equal')


    xf = np.asarray(data[0]['x_pos'])
    yf = np.asarray(data[0]['y_pos'])

    xr = np.asarray(data[1]['x_pos'])
    yr = np.asarray(data[1]['y_pos'])

    line2 = ax.plot(xf[0], yf[0], label=f'front wheel')[0]
    dot2 = ax.plot(xf[0], yf[0], 'bo')[0]
    line3 = ax.plot(xr[0], yr[0], label=f'rear wheel')[0]
    dot3 = ax.plot(xr[0], yr[0], 'ro')[0]

    ax.set(xlim=[0, resolution[0]], ylim=[0, resolution[1]], xlabel='X [pixels]', ylabel='Y [pixels]')


    def update(frame):
        # for each frame, update the data stored on each artist.
        xf2 = xf[:frame]
        yf2 = yf[:frame]

        xr2 = xr[:frame]
        yr2 = yr[:frame]

        # update the line plot:
        line2.set_xdata(xf2[:frame])
        line2.set_ydata(yf2[:frame])
        dot2.set_data([xf[frame]], [yf[frame]])

        line3.set_xdata(xr2[:frame])
        line3.set_ydata(yr2[:frame])
        dot3.set_data([xr[frame]], [yr[frame]])

        ax.set_title(f'Frame: {first_frame + frame}, Time: {round(frame/30, 2)} s')
    
        return line2, line3


    ani2d = animation.FuncAnimation(fig=fig, func=update, frames=len(xf), interval=60)
    plt.show()

    if save == True:
        ani2d.save(filename=f"{filename}-animation-{date}.gif")

    return


def animate3d(data, resolution, first_frame, save=False):
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



    ani3d = animation.FuncAnimation(fig=fig, func=update, frames=len(xf), interval=60)
    plt.show()

    if save == True:
        ani3d.save(filename=f"{filename}-animation-{date}.gif")

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
    Create the nested dictionary by frame for the data required in the model of the bicycle
    '''

    data = {
        'r_Cf_Cr_x':0,
        'r_Cf_Cr_z':0,
        'r_Cr_O_x':0,
        'r_Cr_O_z':0,
        'r_Q_S_x':0,
        'r_Q_S_z':0,
        'r_P1r_Cr_x':0,
        'r_P1r_Cr_z':0,
        'r_P2r_Cr_x':0,
        'r_P2r_Cr_z':0,
        'r_P1f_Cf_x':0,
        'r_P1f_Cf_z':0,
        'r_P2f_Cf_x':0,
        'r_P2f_Cf_z':0,
    }

    rear_wheel = {
        'x': 0,
        'z': 0,
        'psi': 0
    }


    ellipse_f = np.array(track_data[0]['ellipses'])
    ellipse_r = np.array(track_data[1]['ellipses'])

    x_f = np.array(ellipse_f[:,0])
    z_f = np.array(ellipse_f[:,1])
    a_f = ellipse_f[:,2]
    b_f = ellipse_f[:,3]
    theta_f = ellipse_f[:,4]

    ellipse_f_data = [x_f, z_f, a_f, b_f, theta_f]
    extremes_f = find_points(ellipse_f_data)

    x_r = np.array(ellipse_r[:,0])
    z_r = np.array(ellipse_r[:,1])
    a_r = ellipse_r[:,2]
    b_r = ellipse_r[:,3]
    theta_r = ellipse_r[:,4]

    ellipse_r_data = [x_r, z_r, a_r, b_r, theta_r]
    extremes_r = find_points(ellipse_r_data)

    # ellipses_data = [ellipse_r_data, ellipse_f_data]

    rear_wheel['x'] = x_r
    rear_wheel['z'] = z_r

    data['r_Cf_Cr_x'] = x_f - x_r
    data['r_Cf_Cr_z'] = z_f - z_r
    data['r_Cr_O_x'] = x_r - assumed_origin[0]
    data['r_Cr_O_z'] = z_r - assumed_origin[1]
    # data['r_P_S_x'] = (np.sin(q1_f)*np.cos(q1_r) - np.sin(q1_r)*np.cos(q1_f)*np.cos(theta_f - theta_r))*np.sin(theta_f)/np.sqrt((np.sin(q1_f)*np.cos(q1_r) - np.sin(q1_r)*np.cos(q1_f)*np.cos(theta_f - theta_r))**2 + np.sin(q1_r)**2*np.sin(theta_f - theta_r)**2) + np.sin(q1_r)*np.sin(theta_f - theta_r)*np.cos(q1_f)*np.cos(theta_f)/np.sqrt((np.sin(q1_f)*np.cos(q1_r) - np.sin(q1_r)*np.cos(q1_f)*np.cos(theta_f - theta_r))**2 + np.sin(q1_r)**2*np.sin(theta_f - theta_r)**2)
    # data['r_P_S_z'] = (np.sin(q1_f)*np.cos(q1_r) - np.sin(q1_r)*np.cos(q1_f)*np.cos(theta_f - theta_r))*np.cos(theta_f)/np.sqrt((np.sin(q1_f)*np.cos(q1_r) - np.sin(q1_r)*np.cos(q1_f)*np.cos(theta_f - theta_r))**2 + np.sin(q1_r)**2*np.sin(theta_f - theta_r)**2) - np.sin(q1_r)*np.sin(theta_f)*np.sin(theta_f - theta_r)*np.cos(q1_f)/np.sqrt((np.sin(q1_f)*np.cos(q1_r) - np.sin(q1_r)*np.cos(q1_f)*np.cos(theta_f - theta_r))**2 + np.sin(q1_r)**2*np.sin(theta_f - theta_r)**2)
    data['r_P1f_Cf_x'] = extremes_f[0][0]
    data['r_P1f_Cf_z'] = extremes_f[0][1]
    data['r_P3f_Cf_x'] = extremes_f[2][0]
    data['r_P3f_Cf_z'] = extremes_f[2][1]
    data['r_P1r_Cr_x'] = extremes_r[0][0]
    data['r_P1r_Cr_z'] = extremes_r[0][1]
    data['r_P3r_Cr_x'] = extremes_r[2][0]
    data['r_P3r_Cr_z'] = extremes_r[2][1]

    data_json = clean4json(data)
    rw_json = clean4json(rear_wheel)

    

    return data_json, ellipse_r, ellipse_f


# ----- Tests -----



# ----- Run -----

rawdate = datetime.datetime.now()
date =  rawdate.strftime('%Y') + rawdate.strftime('%m') + rawdate.strftime('%d')


imgdata = {
    'r_Cf_Cr_x': x_f-x_r,
    'r_Cf_Cr_z': y_f-y_r,
    'r_Cr_O_x': x_r - assumed_origin[0],
    'r_Cr_O_z': y_r - assumed_origin[1],
    'r_P_S_x': (np.sin(cam_angle_f)*np.cos(cam_angle_r) - np.sin(cam_angle_r)*np.cos(cam_angle_f)*np.cos(theta_f - theta_r))*np.sin(theta_f)/np.sqrt((np.sin(cam_angle_f)*np.cos(cam_angle_r) - np.sin(cam_angle_r)*np.cos(cam_angle_f)*np.cos(theta_f - theta_r))**2 + np.sin(cam_angle_r)**2*np.sin(theta_f - theta_r)**2) + np.sin(cam_angle_r)*np.sin(theta_f - theta_r)*np.cos(cam_angle_f)*np.cos(theta_f)/np.sqrt((np.sin(cam_angle_f)*np.cos(cam_angle_r) - np.sin(cam_angle_r)*np.cos(cam_angle_f)*np.cos(theta_f - theta_r))**2 + np.sin(cam_angle_r)**2*np.sin(theta_f - theta_r)**2),
    'r_P_S_z': (np.sin(cam_angle_f)*np.cos(cam_angle_r) - np.sin(cam_angle_r)*np.cos(cam_angle_f)*np.cos(theta_f - theta_r))*np.cos(theta_f)/np.sqrt((np.sin(cam_angle_f)*np.cos(cam_angle_r) - np.sin(cam_angle_r)*np.cos(cam_angle_f)*np.cos(theta_f - theta_r))**2 + np.sin(cam_angle_r)**2*np.sin(theta_f - theta_r)**2) - np.sin(cam_angle_r)*np.sin(theta_f)*np.sin(theta_f - theta_r)*np.cos(cam_angle_f)/np.sqrt((np.sin(cam_angle_f)*np.cos(cam_angle_r) - np.sin(cam_angle_r)*np.cos(cam_angle_f)*np.cos(theta_f - theta_r))**2 + np.sin(cam_angle_r)**2*np.sin(theta_f - theta_r)**2)
}

parser = argparse.ArgumentParser()
parser.add_argument('-f','--filter', default=False, action='store_true', 
                    help='Apply low pass filter')
parser.add_argument('-a2d', '--animate-2d', default=False, action='store_true', 
                    help='Plot 2D animation')
parser.add_argument('-a3d', '--animate-3d', default=False, action='store_true', 
                    help='Plot 3D animation')
args = parser.parse_args()



parser.add_argument('--word', type=str, required=False)
parser.add_argument('-l', '--long', action='store_true', help='sets long something')
parser.add_argument('-s', '--show', action='store_true', help='shows plots')