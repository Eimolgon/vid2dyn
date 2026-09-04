import os
import json
import math
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from collections import defaultdict
from bike_model import *
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
    

def fitEllipse_old(points, screen_res:tuple):
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


def fitEllipse_old2(points, screen_resolution:tuple):
    x_i = points[:, 0]*screen_resolution[0]
    y_i = (1 - points[:, 1])*screen_resolution[1]

    x = x_i[:, np.newaxis]
    y = y_i[:, np.newaxis]
    
    # Design matrix
    D = np.hstack((x*x, x*y, y*y, x, y, np.ones_like(x)))
    
    # Scatter matrix
    S = np.dot(D.T, D)
    
    # Constraint matrix
    C = np.zeros((6, 6))
    C[0, 2] = 2
    C[1, 1] = -1
    C[2, 0] = 2
    
    # Solve generalized eigenvalue problem: S * v = lambda * C * v
    eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(S).dot(C))
    
    # The eigenvector corresponding to the only positive eigenvalue is our solution
    pos_eval_idx = np.where(eigenvalues > 0)[0]
    if len(pos_eval_idx) == 0:
        return None  # Should not happen with valid data
        
    coeffs = eigenvectors[:, pos_eval_idx[0]]

    a, b, c, d, e, f = coeffs

    num = b**2 - 4*a*c
    xc = (2*c*d - b*e) / num
    yc = (2*a*e - b*d) / num

    # 2. Calculate Semi-axes
    # We use absolute values inside the sqrt to handle sign flips
    # and floating point noise.
    up = 2 * (a*e**2 + c*d**2 + f*b**2 - b*d*e - 4*a*c*f)
    
    # Term in the denominator
    term = np.sqrt((a - c)**2 + b**2)
    
    # Use abs() to prevent RuntimeWarning from tiny negative numbers 
    # caused by precision limits.
    res_a = np.sqrt(np.abs(up / (num * (a + c - term))))
    res_b = np.sqrt(np.abs(up / (num * (a + c + term))))

    # 3. Calculate Rotation Angle
    theta = 0.5 * np.arctan2(b, (a - c))


    ellipse_coeffs = (xc, yc, res_b, res_a, theta)

    return ellipse_coeffs


def fitEllipse(points, resolution):
    """
    Fits an ellipse to normalized points and rescales it to screen resolution.
    
    Args:
        points: (N, 2) array of normalized (x, y) coordinates.
        resolution: (sx, sy) tuple representing the scale factors (e.g., screen width/height).
        
    Returns:
        x, y: Center coordinates in scaled space.
        a, b: Semi-major and semi-minor axes in scaled space.
        theta: Rotation angle in radians (standardized range).
    """
    sx, sy = resolution
    xn = points[:, 0]
    yn = points[:, 1]

    # 1. Fit in normalized space
    D = np.stack([xn**2, xn*yn, yn**2, xn, yn, np.ones_like(xn)], axis=1)
    S = D.T @ D
    C = np.zeros((6, 6))
    C[0, 2], C[2, 0], C[1, 1] = 2, 2, -1
    
    try:
        evals, evecs = np.linalg.eig(np.linalg.inv(S) @ C)
        coeffs = evecs[:, np.argmax(evals)]
    except:
        return None

    # 2. Map to screen: x = sx*xn, y = sy*(1 - yn)
    a_n, b_n, c_n, d_n, e_n, f_n = coeffs
    
    # Transformation algebra
    a = a_n / (sx**2)
    b = -b_n / (sx * sy) 
    c = c_n / (sy**2)
    d = (b_n + d_n) / sx
    e = -(2 * c_n + e_n) / sy
    f = c_n + e_n + f_n

    # 3. Robust Extraction using the Matrix Form
    # Center (standard solution for linear system of partial derivatives)
    num = b**2 - 4*a*c
    xc = (2*c*d - b*e) / num
    yc = (2*a*e - b*d) / num

    # Re-scale the constant term relative to the new center
    # This value 'q' determines the size of the axes
    q = a*xc**2 + b*xc*yc + c*yc**2 - f
    
    # Quadratic form matrix
    A_mat = np.array([[a, b/2], [b/2, c]])
    eigvals, eigvecs = np.linalg.eigh(A_mat)

    # Semi-axes: sqrt(q / eigenvalue)
    # We use abs to ensure no sqrt of negative due to float noise
    axis_lengths = np.sqrt(np.abs(q / eigvals))
    
    # Identify which eigenvalue corresponds to which axis
    # We force axis_1 to be the MAJOR axis
    if axis_lengths[0] >= axis_lengths[1]:
        axis_1, axis_2 = axis_lengths[0], axis_lengths[1]
        # Angle of the first eigenvector
        theta = np.arctan2(eigvecs[1, 0], eigvecs[0, 0])
    else:
        axis_1, axis_2 = axis_lengths[1], axis_lengths[0]
        theta = np.arctan2(eigvecs[1, 1], eigvecs[0, 1])

    # 4. Final Angle Normalization
    # Ensure theta is in a consistent range
    theta = (theta + np.pi/2) % np.pi - np.pi/2

    return xc, yc, axis_1, axis_2, theta


def fitEllipse_conic(points, resolution):
    """
    Fits an ellipse to normalized points and rescales it to screen resolution.
    
    Args:
        points: (N, 2) array of normalized (x, y) coordinates.
        resolution: (sx, sy) tuple representing the scale factors (e.g., screen width/height).
        
    Returns:
        x, y: Center coordinates in scaled space.
        a, b: Semi-major and semi-minor axes in scaled space.
        theta: Rotation angle in radians (standardized range).
    """
    sx, sy = resolution
    xn = points[:, 0]
    yn = points[:, 1]

    # 1. Fit in normalized space
    D = np.stack([xn**2, xn*yn, yn**2, xn, yn, np.ones_like(xn)], axis=1)
    S = D.T @ D
    C = np.zeros((6, 6))
    C[0, 2], C[2, 0], C[1, 1] = 2, 2, -1
    
    try:
        evals, evecs = np.linalg.eig(np.linalg.inv(S) @ C)
        coeffs = evecs[:, np.argmax(evals)]
    except:
        return None

    # 2. Map to screen: x = sx*xn, y = sy*(1 - yn)
    a_n, b_n, c_n, d_n, e_n, f_n = coeffs
    
    # Transformation algebra
    a = a_n / (sx**2)
    b = -b_n / (sx * sy) 
    c = c_n / (sy**2)
    d = (b_n + d_n) / sx
    e = -(2 * c_n + e_n) / sy
    f = c_n + e_n + f_n

    return a, b, c, d, e, f


def find_points(ellipse, direction):
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

    d_x = np.sqrt(a**2*np.cos(theta)**2 + b**2*np.sin(theta)**2)
    d_z = np.sqrt(a**2*np.sin(theta)**2 + b**2*np.cos(theta)**2)

    w_x = -d_z
    w_y = ((a**2 - b**2)*np.sin(theta)*np.cos(theta))/d_z

    # ----- Gemini -----
    A = (np.cos(theta)**2)/(a**2) + (np.sin(theta)**2)/(b**2)
    B = 2*np.sin(theta)*np.cos(theta)*((1/a**2) - (1/b**2))
    C = (np.sin(theta)**2/(a**2)) + (np.cos(theta)**2/(b**2))

    k = 1.0 / np.sqrt(np.abs(A*w_x**2 + B*w_x*w_y + C*w_y**2))

    # p2_2 = (x - k*w_x, z - k*w_y)
    # p4_2 = (x + k*w_x, z + k*w_y)

    p2_2 = (x + k*w_x, z + k*w_y)
    p4_2 = (x - k*w_x, z - k*w_y)
    # ----- ----- -----

    p1_z = z + d_z
    p3_z = z - d_z

    p2_x = x - d_x
    p4_x = x + d_x

    p1_x = x + ((a**2 - b**2)*np.sin(theta)*np.cos(theta))/d_z
    p3_x = x - ((a**2 - b**2)*np.sin(theta)*np.cos(theta))/d_z

    p2_z = z - ((a**2 - b**2)*np.sin(theta)*np.cos(theta))/d_x
    p4_z = z + ((a**2 - b**2)*np.sin(theta)*np.cos(theta))/d_x


    # I just flip points 2 and 4 because those are the two that are 'in front'
    # and 'behind' of the wheels and will change depending on the direction
    # of travel.
    if direction == 'right':
        p2 = (p2_x, p2_z)
        p4 = (p4_x, p4_z)
        p2_2 = (x + k*w_x, z + k*w_y)
        p4_2 = (x - k*w_x, z - k*w_y)
    elif direction == 'left':
        p4 = (p2_x, p2_z)
        p2 = (p4_x, p4_z)
        p4_2 = (x + k*w_x, z + k*w_y)
        p2_2 = (x - k*w_x, z - k*w_y)

    p1 = (p1_x, p1_z)
    p3 = (p3_x, p3_z)
    

    return p1, p2, p3, p4, p2_2, p4_2


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


def animate_Ellipse(data, resolution, first_frame, name='none'):
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

    if name != 'none':
        writer = animation.HTMLWriter()
        animation_file = './tmp/animation.html'
        animEllipse.save(animation_file, writer=writer)
        filename = str(input('Type file name to save'))
        animEllipse.save(filename=f"{name}-animation.gif")

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
        'r_Q_S_x':0,
        'r_Q_S_z':0
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

    p1_f = extremes_f[:, 0]
    p3_f = extremes_f[:, 2]

    xf = np.array(ellipse_f[:,0])
    zf = np.array(ellipse_f[:,1])
    af = np.array(ellipse_f[:,2])
    bf = np.array(ellipse_f[:,3])
    theta_f = np.array(ellipse_f[:,4])
    

    p1_r = extremes_r[:, 0]
    p3_r = extremes_r[:, 2]

    xr = np.array(ellipse_r[:,0])
    zr = np.array(ellipse_r[:,1])
    ar = np.array(ellipse_r[:,2]) 
    br = np.array(ellipse_r[:,3])
    theta_r = np.array(ellipse_r[:,4])
    
    
    angle_ellipse_f = np.array([
        np.arccos(b / a) if a > b else 0
        for a, b in zip(af, bf)
    ])

    angle_ellipse_r = np.array([
        np.arccos(b / a) if a > b else 0
        for a, b in zip(ar, br)
    ])


    rear_wheel['x'] = xr
    rear_wheel['z'] = zr

    data['r_Cf_Cr_x'] = xf - xr
    data['r_Cf_Cr_z'] = zf - zr
    data['r_Cr_O_x'] = xr
    data['r_Cr_O_z'] = zr
    data['r_P1f_Cf_x'] = p1_r[:, 0] - xr
    data['r_P1f_Cf_z'] = p1_r[:, 1] - zr
    data['r_P3f_Cf_x'] = p3_r[:, 0] - xr
    data['r_P3f_Cf_z'] = p3_r[:, 1] - zr
    data['r_P1r_Cr_x'] = p1_f[:, 0] - xf
    data['r_P1r_Cr_z'] = p1_f[:, 1] - zf
    data['r_P3r_Cr_x'] = p3_f[:, 0] - xf
    data['r_P3r_Cr_z'] = p3_f[:, 1] - zf
    data['r_Q_S_x'] = (np.sin(angle_ellipse_f)*np.cos(angle_ellipse_r) - 
                     np.sin(angle_ellipse_r)*np.cos(angle_ellipse_f)*
                     np.cos(theta_f - theta_r))*np.sin(theta_f)/ \
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
                                                             np.sin(theta_f - theta_r)**2)    
    data['r_Q_S_z'] = (np.sin(angle_ellipse_f)*np.cos(angle_ellipse_r) - 
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
                                                         np.sin(theta_f - theta_r)**2)

    data_json = clean4json(data)
    rw_json = clean4json(rear_wheel)

    return data_json, ellipse_r, ellipse_f


def fit_img2model(real_data, initial_guess, bike_parameters, boundaries, camera_parameters, model):
    '''
    Least_squares fitting from image data to multibody model.
    Input: real_data, initial guess, bicycle parameters, and boundaries for the
    solver.
    Output: solver history.
    '''
    x0 = initial_guess
    results_history = []
    x0_history = [x0]

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
    'r_P3f_Cf_z': 0,
    'r_Q_S_x': 0,
    'r_Q_S_z': 0
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
        image_data['r_Q_S_x'] = real_data['r_Q_S_x'][i]
        image_data['r_Q_S_z'] = real_data['r_Q_S_z'][i]

        results_iteration = least_squares(model.residual_eqs, x0, 
                                            args = (image_data, bike_parameters, camera_parameters), 
                                            bounds = (lower_bound, upper_bound))
        results_history.append(results_iteration)
        x0_history.append(results_iteration['x'])

        # Update initial guess
        x0 = results_iteration['x'].copy()

    return results_history, x0_history


def points2plot(bike_parameters:dict, state):
    '''
    Find the points to plot the model in 2d and 3d.
    Input: bicycle parameters and state.
    Output: vector with points.
    '''

    subs = (state[0], state[1], state[2], state[3], state[4], state[5], 
            state[6], bike_parameters['lr'], bike_parameters['lf1'],
            bike_parameters['lf2'], bike_parameters['rr'], bike_parameters['rf'])
    
    
    Cr_point = (eval_f03(*subs), eval_p01(*subs), eval_f04(*subs))
    S_point = (eval_p02(*subs), eval_p03(*subs), eval_p04(*subs))
    Q_point = (eval_p05(*subs), eval_p06(*subs), eval_p07(*subs))
    Cf_point = (eval_p08(*subs), eval_p09(*subs), eval_p10(*subs))

    p1r_point = (eval_p11(*subs), eval_p12(*subs), eval_p13(*subs))
    p3r_point = (eval_p23(*subs), eval_p24(*subs), eval_p25(*subs))

    p1f_point = (eval_p14(*subs), eval_p15(*subs), eval_p16(*subs))
    p3f_point = (eval_p26(*subs), eval_p27(*subs), eval_p28(*subs))

    p2r_point = (eval_p17(*subs), eval_p18(*subs), eval_p19(*subs))
    p4r_point = (eval_p29(*subs), eval_p30(*subs), eval_p31(*subs))
    
    p2f_point = (eval_p20(*subs), eval_p21(*subs), eval_p22(*subs))
    p4f_point = (eval_p32(*subs), eval_p33(*subs), eval_p34(*subs))

    
    return [(Cr_point, S_point, Q_point, Cf_point), 
            (p1r_point, p3r_point), (p1f_point, p3f_point),
            (p2r_point, p4r_point), (p2f_point, p4f_point)]


def pt2circle(p1, p2, p3, p4):
    pts = np.array([
        [p1[0], p1[1], p1[2]],
        [p2[0], p2[1], p2[2]],
        [p3[0], p3[1], p3[2]],
        [p4[0], p4[1], p4[2]],
    ])

    center_circle = np.mean(pts, axis=0)
    radius_circle = np.linalg.norm(pts[0] - center_circle)

    u = (pts[0] - center_circle)/np.linalg.norm(pts[0] - center_circle)
    w = np.cross(pts[0] - center_circle, pts[1] - center_circle) 
    w /= np.linalg.norm(w)

    v = np.cross(w, u)
    theta_circle = np.linspace(0, 2 * np.pi, 100)

    circle_pts = np.array([center_circle + radius_circle * np.cos(t) * u + 
                         radius_circle * np.sin(t) * v for t in theta_circle])

    cx, cy, cz = circle_pts[:, 0], circle_pts[:, 1], circle_pts[:, 2]

    return cx, cy, cz


def set_axes_equal(ax):
    '''
    Used to plot equal axes in the 3d view.
    '''
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    x_middle = sum(x_limits) / 2
    y_middle = sum(y_limits) / 2
    z_middle = sum(z_limits) / 2

    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def plot_mbd_model_2d(ax, bike_parameters:dict, state, source):
    '''
    Create 2-dimensional plot of the model at the given state.
    Input: state-space vector, plane of the projection. 
    (phi, theta, psi, delta, x_r, y_r, z_r, lr, lf1, lf2, r)
    Output: plot of the model.
    '''

    if source == 'guess':
        colors={'rear_frame': 'blue',
                'front_frame': 'green',
                'trail': 'red',
                'p1': 'purple',
                'p3': 'yellow'}

    frame_points, rw_points, fw_points, r_pts_2, f_pts2 = \
        points2plot(bike_parameters, state)
    Cr_point, S_point, Q_point, Cf_point = frame_points
    p1r_point, p3r_point = rw_points
    p1f_point, p3f_point = fw_points

    rear_frame_x, rear_frame_y, rear_frame_z = [Cr_point[0], S_point[0]], \
        [Cr_point[0], S_point[0]], [Cr_point[2], S_point[2]]
    
    front_frame_x, front_frame_y, front_frame_z = [S_point[0], Q_point[0]], \
        [S_point[1], Q_point[1]], [S_point[2], Q_point[2]]
    
    trail_x, trail_y, trail_z = [Q_point[0], Cf_point[0]], \
        [Q_point[1], Cf_point[1]], [Q_point[2], Cf_point[2]]
    
    ax.plot(rear_frame_x, rear_frame_z, marker='o', color='C0')
    ax.plot(front_frame_x, front_frame_z, marker='o', color='C2')
    ax.plot(trail_x, trail_z, marker='o', color='C1')

    ax.scatter(Cr_point[0], Cr_point[2], marker='x', color='C0')
    ax.scatter(Cf_point[0], Cf_point[2], marker='x', color='C1')
    
    ax.scatter(p1r_point[0], p1r_point[2], facecolors='none', edgecolors='C9', s=50)
    ax.scatter(p3r_point[0], p3r_point[2], facecolors='none', edgecolors='C8', s=50)
    ax.scatter(p1f_point[0], p1f_point[2], facecolors='none', edgecolors='C9', s=50)
    ax.scatter(p3f_point[0], p3f_point[2], facecolors='none', edgecolors='C8', s=50)

    ax.grid()
    ax.axis('equal')

    return


def plot_mbd_model_3d(bike_parameters:dict, state, source, ax=None):
    '''
    Create 3-dimensional plot of the model at the given state.
    Input: state-space vector.
    Output: plot of the model.
    '''

    if source == 'initial_guess':
        line_type = 'dashed'
    elif source == 'solver':
        line_type = 'solid'


    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    

    frame_points, rw_points, fw_points, r_pts_2, f_pts_2 = \
        points2plot(bike_parameters, state)
    
    Cr_point, S_point, Q_point, Cf_point = frame_points

    p1r_point, p3r_point = rw_points
    p1f_point, p3f_point = fw_points

    p2r_point, p4r_point = r_pts_2
    p2f_point, p4f_point = f_pts_2

    cx_r, cy_r, cz_r = pt2circle(p1r_point, p2r_point, p3r_point, p4r_point)
    cx_f, cy_f, cz_f = pt2circle(p1f_point, p2f_point, p3f_point, p4f_point)

    

    ax.plot([Cr_point[0], S_point[0]],
            [Cr_point[1], S_point[1]],
            [Cr_point[2], S_point[2]], marker='o', color='C0', label='Frame',
            linestyle=line_type)

    ax.plot([S_point[0], Q_point[0]],
            [S_point[1], Q_point[1]],
            [S_point[2], Q_point[2]], marker='o', color='C2', label='Steering',
            linestyle=line_type)
    
    ax.plot([Q_point[0], Cf_point[0]],
            [Q_point[1], Cf_point[1]],
            [Q_point[2], Cf_point[2]], marker='o', color='C1', label='Fork',
            linestyle=line_type)
    
    ax.scatter(p1r_point[0], p1r_point[1], p1r_point[2], marker='x', color='red')
    ax.scatter(p3r_point[0], p3r_point[1], p3r_point[2], marker='x', color='black')

    ax.scatter(p1f_point[0], p1f_point[1], p1f_point[2], marker='^', color='red')
    ax.scatter(p3f_point[0], p3f_point[1], p3f_point[2], marker='^', color='black')
    
    ax.plot(cx_r, cy_r, cz_r, color='black', linestyle=line_type)
    ax.plot(cx_f, cy_f, cz_f, color='black', linestyle=line_type)

    # ax.set_xlabel('X axis')
    # ax.set_ylabel('Y axis')
    # ax.set_zlabel('Z axis')
    # ax.set_box_aspect([1, 1, 1])
    # set_axes_equal(ax)
    # plt.show()

    return ax


def plot_raw_data(ax, raw_data, screen_resolution, plot_color):
    '''
    Plot original dots in 2d plane.
    '''

    ax.scatter(raw_data[:, 0]*screen_resolution[0], (1 - raw_data[:, 1])*screen_resolution[1], 
               facecolors='none', edgecolors=plot_color)
    ax.grid()

    return


def plot_dots(ax, dots, wheel):
    '''
    Plot objective points.
    '''

    if wheel == 'front':
        colors = {'p1': 'red', 'p3': 'black', 'center': 'C1', 'mark':'^'}
    elif wheel == 'rear':
        colors = {'p1': 'red', 'p3': 'black', 'center': 'C0', 'mark':'x'}

    ax.scatter(dots[0], dots[1], edgecolor=colors['center'], facecolors='none', marker='s')
    ax.scatter(dots[2][0], dots[2][1], color=colors['p1'], marker=colors['mark'])
    ax.scatter(dots[3][0], dots[3][1], color=colors['p3'], marker=colors['mark'])
    ax.grid()

    return


# ----- This is made by copilot so needs checking -----

def get_normal(a, b, c, d, e, f):
    # extract quadratic form
    Q = np.array([[a, b/2],
                  [b/2, c]])

    
    if np.linalg.det(Q) < 0:
        Q = -Q

    vals, vecs = np.linalg.eigh(Q)

    eps = 1e-12
    vals = np.maximum(vals, eps)

    # semi-axes ratios (scale irrelevant)
    alpha = 1.0 / np.sqrt(vals[1])
    beta  = 1.0 / np.sqrt(vals[0])

    # major axis direction
    v = vecs[:, 1]
    theta = np.arctan2(v[1], v[0])

    u1 = np.array([
        np.cos(theta) / alpha,
        np.sin(theta) / alpha,
        1.0
    ])

    u2 = np.array([
        -np.sin(theta) / beta,
         np.cos(theta) / beta,
         1.0
    ])

    n = np.cross(u1, u2)
    n /= np.linalg.norm(n)

    # Optional: enforce positive Z
    if n[2] < 0:
        n = -n

    normal = n

    return normal

# ----- ------ ----- ----- -----

