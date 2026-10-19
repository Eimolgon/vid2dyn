# src/interactive_viewer.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import sympy as sp
import sympy.physics.mechanics as me

# Import your model functions
from bike_model_perspective import perspective_projection
from utils import _build_frames, _wheel_plane_axes, _to_xyz, points2plot
from utils import readFile, fitEllipse, find_points

def launch_interactive_viewer(bike_params, camera_params, state0, image_points=None, ellipses=None, screen_resolution=(1920, 1080)):
    """
    Interactive viewer with 3D model view and 2D perspective projection view,
    complete with sliders for state variables and camera parameters.
    """
    fig = plt.figure(figsize=(16, 9))
    plt.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.35)

    # 3D Subplot
    ax3d = fig.add_subplot(121, projection='3d')
    ax3d.set_title("3D Bike Model View")
    ax3d.set_xlabel("X (m)")
    ax3d.set_ylabel("Y (m)")
    ax3d.set_zlabel("Z (m)")

    # 2D Image Plane Subplot
    ax2d = fig.add_subplot(122)
    ax2d.set_title("2D Perspective Projection & Real Data")
    ax2d.set_xlim(0, screen_resolution[0])
    ax2d.set_ylim(screen_resolution[1], 0) # Image coordinates: 0 at top
    ax2d.set_aspect('equal')
    ax2d.grid(True)

    current_state = np.array(state0, dtype=float)
    current_cam = camera_params.copy()

    # Plot real data points if provided
    if image_points is not None:
        for cid, pts in image_points.items():
            color = 'red' if cid == 0 else 'blue'
            ax2d.scatter(pts[:, 0] * screen_resolution[0], pts[:, 1] * screen_resolution[1], 
                         color=color, alpha=0.5, s=15, label=f'Real Data (Class {cid})')

    # Plot fitted ellipses if provided
    if ellipses is not None:
        for cid, ell in ellipses.items():
            if ell is not None:
                xc, yc, a, b, theta = ell
                ellipse_patch = plt.matplotlib.patches.Ellipse(
                    (xc, yc), width=2*a, height=2*b, angle=np.rad2deg(theta),
                    edgecolor='orange' if cid==0 else 'cyan', facecolor='none', linestyle='--', lw=1.5
                )
                ax2d.add_patch(ellipse_patch)

    # Drawing placeholders
    line_3d_frame, = ax3d.plot([], [], [], 'o-', color='C0', lw=2, label='Rear Frame')
    line_3d_steer, = ax3d.plot([], [], [], 'o-', color='C2', lw=2, label='Steering')
    line_3d_fork, = ax3d.plot([], [], [], 'o-', color='C1', lw=2, label='Fork')
    
    line_2d_frame, = ax2d.plot([], [], 'o-', color='C0', lw=2, label='Model Frame')
    line_2d_steer, = ax2d.plot([], [], 'o-', color='C2', lw=2, label='Model Steer')
    line_2d_fork, = ax2d.plot([], [], 'o-', color='C1', lw=2, label='Model Fork')

    ax3d.legend(loc='upper left')
    ax2d.legend(loc='upper right')

    # Slider axes definition
    axcolor = 'lightgoldenrodyellow'
    slider_ax_color = [0.10, 0.05, 0.35, 0.22] # [left, bottom, width, height]
    
    # We create axes for State Sliders (Left) and Camera Sliders (Right)
    ax_state = plt.axes([0.10, 0.05, 0.35, 0.22], facecolor=axcolor)
    ax_cam = plt.axes([0.55, 0.05, 0.35, 0.22], facecolor=axcolor)

    # State Sliders
    s_phi_ax   = plt.axes([0.12, 0.24, 0.30, 0.02])
    s_theta_ax = plt.axes([0.12, 0.21, 0.30, 0.02])
    s_psi_ax   = plt.axes([0.12, 0.18, 0.30, 0.02])
    s_delta_ax = plt.axes([0.12, 0.15, 0.30, 0.02])
    s_xr_ax    = plt.axes([0.12, 0.12, 0.30, 0.02])
    s_yr_ax    = plt.axes([0.12, 0.09, 0.30, 0.02])
    s_zr_ax    = plt.axes([0.12, 0.06, 0.30, 0.02])

    s_phi   = Slider(s_phi_ax, 'phi (roll)', -np.pi, np.pi, valinit=current_state[0])
    s_theta = Slider(s_theta_ax, 'theta (pitch)', -np.pi, np.pi, valinit=current_state[1])
    s_psi   = Slider(s_psi_ax, 'psi (yaw)', -np.pi, np.pi, valinit=current_state[2])
    s_delta = Slider(s_delta_ax, 'delta (steer)', -np.pi/2, np.pi/2, valinit=current_state[3])
    s_xr    = Slider(s_xr_ax, 'x_r (m)', -10.0, 20.0, valinit=current_state[4])
    s_yr    = Slider(s_yr_ax, 'y_r (m)', -10.0, 10.0, valinit=current_state[5])
    s_zr    = Slider(s_zr_ax, 'z_r (m)', -2.0, 2.0, valinit=current_state[6])

    # Camera Sliders
    s_camz_ax   = plt.axes([0.57, 0.18, 0.30, 0.02])
    s_fx_ax     = plt.axes([0.57, 0.15, 0.30, 0.02])
    s_camyaw_ax = plt.axes([0.57, 0.12, 0.30, 0.02])

    s_camz   = Slider(s_camz_ax, 'Cam Z (m)', -2.0, 5.0, valinit=current_cam['cam_z'])
    s_fx     = Slider(s_fx_ax, 'Fx (px)', 500.0, 5000.0, valinit=current_cam['fx'])
    s_camyaw = Slider(s_camyaw_ax, 'Cam Yaw', -np.pi, np.pi, valinit=current_cam['cam_yaw'])

    def update_plots(val):
        current_state[0] = s_phi.val
        current_state[1] = s_theta.val
        current_state[2] = s_psi.val
        current_state[3] = s_delta.val
        current_state[4] = s_xr.val
        current_state[5] = s_yr.val
        current_state[6] = s_zr.val

        current_cam['cam_z'] = s_camz.val
        current_cam['fx'] = s_fx.val
        current_cam['fy'] = s_fx.val # Keep square pixels roughly

        # Compute 3D points
        frame_pts, rw_pts, fw_pts, _, _ = points2plot(bike_params, current_state)
        Cr, S, Q, Cf = frame_pts

        # Update 3D lines
        line_3d_frame.set_data_3d([Cr[0], S[0]], [Cr[1], S[1]], [Cr[2], S[2]])
        line_3d_steer.set_data_3d([S[0], Q[0]], [S[1], Q[1]], [S[2], Q[2]])
        line_3d_fork.set_data_3d([Q[0], Cf[0]], [Q[1], Cf[1]], [Q[2], Cf[2]])

        # Compute Perspective Projection for 2D View
        # Define camera frame and position
        N, R, F = _build_frames(current_state, bike_params)
        C_frame = me.ReferenceFrame('C')
        C_frame.orient_body_fixed(N, (current_cam['cam_yaw'], current_cam['cam_pitch'], current_cam['cam_roll']), 'ZYX')
        P_cam = me.Point('P')
        O = me.Point('O'); O.set_pos(O, 0)
        P_cam.set_pos(O, current_cam['cam_x']*N.x + current_cam['cam_y']*N.y + current_cam['cam_z']*N.z)

        # Helper projection function
        def proj(pt_obj):
            r = pt_obj.pos_from(P_cam)
            Xc = r.dot(C_frame.x)
            Yc = r.dot(C_frame.y)
            Zc = r.dot(C_frame.z)
            # Apply correction sign for vertical pixel axis if needed
            u = current_cam['fx'] * Xc / Yc + current_cam['cx']
            v = current_cam['cy'] - current_cam['fy'] * Zc / Yc
            return float(u), float(v)

        # Reconstruct SymPy points for projection
        Cr_pt = me.Point('Cr'); Cr_pt.set_pos(O, current_state[4]*N.x + current_state[5]*N.y + current_state[6]*N.z)
        S_pt = me.Point('S'); S_pt.set_pos(Cr_pt, bike_params['lr'] * R.x)
        Q_pt = me.Point('Q'); Q_pt.set_pos(S_pt, bike_params['lf1'] * -F.z)
        Cf_pt = me.Point('Cf'); Cf_pt.set_pos(Q_pt, bike_params['lf2'] * F.x)

        u_Cr, v_Cr = proj(Cr_pt)
        u_S, v_S = proj(S_pt)
        u_Q, v_Q = proj(Q_pt)
        u_Cf, v_Cf = proj(Cf_pt)

        line_2d_frame.set_data([u_Cr, u_S], [v_Cr, v_S])
        line_2d_steer.set_data([u_S, u_Q], [v_S, v_Q])
        line_2d_fork.set_data([u_Q, u_Cf], [v_Q, v_Cf])

        ax3d.set_xlim(current_state[4] - 2, current_state[4] + 2)
        ax3d.set_ylim(current_state[5] - 2, current_state[5] + 2)
        ax3d.set_zlim(-1, 2)

        fig.canvas.draw_idle()

        # Output current guess string to console for easy copying
        print(f"Current X0 Guess: np.array([{current_state[0]:.4f}, {current_state[1]:.4f}, {current_state[2]:.4f}, {current_state[3]:.4f}, {current_state[4]:.4f}, {current_state[5]:.4f}, {current_state[6]:.4f}])")

    s_phi.on_changed(update_plots)
    s_theta.on_changed(update_plots)
    s_psi.on_changed(update_plots)
    s_delta.on_changed(update_plots)
    s_xr.on_changed(update_plots)
    s_yr.on_changed(update_plots)
    s_zr.on_changed(update_plots)
    s_camz.on_changed(update_plots)
    s_fx.on_changed(update_plots)
    s_camyaw.on_changed(update_plots)

    update_plots(None)
    plt.show()
    return current_state