import os
import json
import math
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from collections import defaultdict
from bike_model import residual_eqs
from matplotlib.patches import Ellipse
import matplotlib.animation as animation
from matplotlib import collections as mc
from skimage.measure import EllipseModel
from scipy.optimize import least_squares
from scipy.signal import butter, filtfilt
from skimage.draw import ellipse_perimeter



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

    rr, cc = ellipse_perimeter(int(z), int(x), int(b), int(a), orientation=theta)
    points = np.column_stack((cc, rr))

    z_col = points[:, 1]
    idx_min = np.argmin(z_col)
    idx_max = np.argmax(z_col)

    upper_point = points[idx_max]
    lower_point = points[idx_min]

    return (upper_point, lower_point)

def find_points2(ellipse):
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

    d_z = np.sqrt(a**2*np.sin(theta)**2 + b**2*np.cos(theta)**2)

    z_max = z + d_z
    z_min = z - d_z

    x_max = x + ((a**2 - b**2)*np.sin(theta)*np.cos(theta))/d_z
    x_min = x - ((a**2 - b**2)*np.sin(theta)*np.cos(theta))/d_z

    upper_point = (x_max, z_max)
    lower_point = (x_min, z_min)

    return (upper_point, lower_point)



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


def lowpass_filter(data, cutoff=1.5, fs=30.0, order=5):
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
                'x': lowpass_filter(np.array(data['ellipses'])[:,0], cutoff, fs),
                'z': lowpass_filter(np.array(data['ellipses'])[:,1], cutoff, fs),
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
        filename = str(input('Type file name to save'))
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
        filename = str(input('Type file name to save'))
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
    
    upper_f = extremes_f[:, 0]
    bottom_f = extremes_f[:, 1]


    xr = np.array(ellipse_r[:,0])
    zr = np.array(ellipse_r[:,1])
    
    upper_r = extremes_r[:, 0]
    bottom_r = extremes_r[:, 1]


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


def fit_img2model(real_data, initial_guess, bike_parameters, boundaries):
    '''
    Least_squares fitting from image data to multibody model.
    Input: real_data, initial guess, bicycle parameters, and boundaries for the
    solver.
    Output: solver history.
    '''
    x0 = initial_guess
    results_history = []

    lower_bound, upper_bound = boundaries

    image_data = {
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

    for i in range(len(real_data['r_Cf_Cr_x'])):
        image_data['r_Cf_Cr_x'] = real_data['r_Cf_Cr_x'][i]
        image_data['r_Cf_Cr_z'] = real_data['r_Cf_Cr_z'][i]
        image_data['r_Cr_O_x'] = real_data['r_Cr_O_x'][i]
        image_data['r_Cr_O_z'] = real_data['r_Cr_O_z'][i]
        image_data['r_P1r_Cr_x'] = real_data['r_P1r_Cr_x'][i]
        image_data['r_P1r_Cr_z'] = real_data['r_P1r_Cr_z'][i]
        image_data['r_P3r_Cr_x'] = real_data['r_P3r_Cr_x'][i]
        image_data['r_P3r_Cr_z'] = real_data['r_P3r_Cr_z'][i]
        image_data['r_P1f_Cf_x'] = real_data['r_P1f_Cf_x'][i]
        image_data['r_P1f_Cf_z'] = real_data['r_P1f_Cf_z'][i]
        image_data['r_P3f_Cf_x'] = real_data['r_P3f_Cf_x'][i]
        image_data['r_P3f_Cf_z'] = real_data['r_P3f_Cf_z'][i]

        results_iteration = least_squares(residual_eqs, x0, 
                                            args = (image_data, bike_parameters), 
                                            bounds = (lower_bound, upper_bound))
        results_history.append(results_iteration)

        # Update initial guess
        x0 = results_iteration['x'].copy()

    return results_history
