import os
import json
import math
import random
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
    '''

    ell = EllipseModel()
    ell.estimate(points)
    xc, yc, a, b, theta = ell.params

    xc = xc*screen_res[0]
    yc = screen_res[1]*(1-yc)
    a = a*screen_res[1]
    b = b*screen_res[0]

    if b < a:
        angle2camera = np.arccos(b/a)
    else:
        angle2camera = 0

    ellipse_data = (xc, yc, a, b, theta, angle2camera)

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

    return


def animate3d(data, resolution, first_frame):
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
    # ani3d.save(filename="yt-crash-005-animation-021125.gif")

    return




# ----- Tests -----



# ----- Run -----

imgdata = {
    'Ry_x': np.sin(cam_angle_r)*np.cos(theta_r),
    'Ry_z': np.sin(cam_angle_r)*np.sin(theta_r),
    'Fy_x': np.sin(cam_angle_f)*np.cos(theta_f),
    'Fy_z': np.sin(cam_angle_f)*np.sin(theta_f),
    'r_Cf_Cr_x': x_f-x_r,
    'r_Cf_Cr_z': y_f-y_r,
    'r_Cr_O_x': x_r - assumed_origin[0],
    'r_Cr_O_z': y_r - assumed_origin[1],
    'r_P_S_x': (np.sin(cam_angle_f)*np.cos(cam_angle_r) - np.sin(cam_angle_r)*np.cos(cam_angle_f)*np.cos(theta_f - theta_r))*np.sin(theta_f)/np.sqrt((np.sin(cam_angle_f)*np.cos(cam_angle_r) - np.sin(cam_angle_r)*np.cos(cam_angle_f)*np.cos(theta_f - theta_r))**2 + np.sin(cam_angle_r)**2*np.sin(theta_f - theta_r)**2) + np.sin(cam_angle_r)*np.sin(theta_f - theta_r)*np.cos(cam_angle_f)*np.cos(theta_f)/np.sqrt((np.sin(cam_angle_f)*np.cos(cam_angle_r) - np.sin(cam_angle_r)*np.cos(cam_angle_f)*np.cos(theta_f - theta_r))**2 + np.sin(cam_angle_r)**2*np.sin(theta_f - theta_r)**2),
    'r_P_S_z': (np.sin(cam_angle_f)*np.cos(cam_angle_r) - np.sin(cam_angle_r)*np.cos(cam_angle_f)*np.cos(theta_f - theta_r))*np.cos(theta_f)/np.sqrt((np.sin(cam_angle_f)*np.cos(cam_angle_r) - np.sin(cam_angle_r)*np.cos(cam_angle_f)*np.cos(theta_f - theta_r))**2 + np.sin(cam_angle_r)**2*np.sin(theta_f - theta_r)**2) - np.sin(cam_angle_r)*np.sin(theta_f)*np.sin(theta_f - theta_r)*np.cos(cam_angle_f)/np.sqrt((np.sin(cam_angle_f)*np.cos(cam_angle_r) - np.sin(cam_angle_r)*np.cos(cam_angle_f)*np.cos(theta_f - theta_r))**2 + np.sin(cam_angle_r)**2*np.sin(theta_f - theta_r)**2)
}