# src/interactive.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import bike_model_perspective as perpro
from utils import (
    readFile, fitEllipse, find_points, plot_mbd_model_3d,
    points2plot, pt2circle, set_axes_equal
)


class BikeInteractiveViewer:
    """
    Interactive 3D visualization of the bicycle multibody model.
    """

    STATE_LABELS = [
        ('phi',   'Roll [deg]',    -90.0,  90.0),
        ('theta', 'Pitch [deg]',   -90.0,  90.0),
        ('psi',   'Yaw [deg]',     -90.0,  90.0),
        ('delta', 'Steer [deg]',   -90.0,  90.0),
        ('x_r',   'X rear [m]',    -20.0,  20.0),
        ('y_r',   'Y rear [m]',    -20.0,  20.0),
        ('z_r',   'Z rear [m]',    -20.0,  20.0),
    ]

    def __init__(self, bike_params, camera_params, state0,
                 image_points=None, ellipses=None, screen_resolution=(1920, 1080),
                 focal_length=0.028, sensor_size=(0.0235, 0.0156),
                 image_resolution=(6020, 4024)):
        self.bike_params = bike_params
        self.camera_params = camera_params
        self.state = np.array(state0, dtype=float)
        self.image_points = image_points
        self.ellipses = ellipses
        self.screen_resolution = screen_resolution

        # Back-project image points onto the ground plane z=0
        self._init_geometry(focal_length, sensor_size, image_resolution)

        # ----- Figure / axes -----
        self.fig = plt.figure(figsize=(15, 8))

        self.ax = self.fig.add_axes([0.32, 0.05, 0.66, 0.90], projection='3d')
        self.ax.set_xlabel('X [m]')
        self.ax.set_ylabel('Y [m]')
        self.ax.set_zlabel('Z [m]')
        self.ax.set_title('Bicycle model – drag the sliders to change the state')
        self.ax.set_box_aspect([1, 1, 1])

        # Fixed camera angle for the 3D view (set once, never touched again)
        self.ax.view_init(elev=18, azim=-90)

        # Registry of artists that must be removed on each redraw
        self._artists = []
        self._axes_initialised = False

        # ----- Sliders -----
        self.sliders = []
        self.slider_axs = []
        n = len(self.STATE_LABELS)
        bottom = 0.35
        h = 0.03
        gap = 0.012
        for i, (key, label, lo, hi) in enumerate(self.STATE_LABELS):
            ax_s = self.fig.add_axes([0.05, bottom + (n - 1 - i) * (h + gap),
                                      0.20, h])
            init_val = (np.rad2deg(self.state[i])
                        if key in ('phi', 'theta', 'psi', 'delta')
                        else self.state[i])
            s = Slider(ax_s, label, lo, hi, valinit=init_val)
            s.on_changed(self._on_slider_change)
            self.sliders.append(s)
            self.slider_axs.append(ax_s)

        # ----- Buttons -----
        ax_reset = self.fig.add_axes([0.05, 0.25, 0.09, 0.04])
        self.btn_reset = Button(ax_reset, 'Reset')
        self.btn_reset.on_clicked(self._on_reset)

        ax_fit = self.fig.add_axes([0.16, 0.25, 0.09, 0.04])
        self.btn_fit = Button(ax_fit, 'Fit')
        self.btn_fit.on_clicked(self._on_fit)

        ax_copy = self.fig.add_axes([0.05, 0.19, 0.20, 0.04])
        self.btn_copy = Button(ax_copy, 'Copy state to clipboard')
        self.btn_copy.on_clicked(self._on_copy)

        # ----- Radio -----
        ax_radio = self.fig.add_axes([0.05, 0.05, 0.20, 0.12])
        self.radio = RadioButtons(ax_radio, ('front', 'rear', 'both'),
                                  active=2)
        self.radio.on_clicked(self._on_radio)

        # ----- Draw -----
        self._draw()
        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------ #
    # Geometry helpers
    # ------------------------------------------------------------------ #
    def _init_geometry(self, focal_length, sensor_size, image_resolution):
        fx = focal_length * image_resolution[0] / sensor_size[0]
        fy = focal_length * image_resolution[1] / sensor_size[1]
        cx = image_resolution[0] / 2.0
        cy = image_resolution[1] / 2.0
        self.K = np.array([[fx, 0, cx],
                           [0, fy, cy],
                           [0, 0, 1]])

        cy_, cp_, cr_ = (self.camera_params['cam_yaw'],
                         self.camera_params['cam_pitch'],
                         self.camera_params['cam_roll'])
        Rz = np.array([[np.cos(cy_), -np.sin(cy_), 0],
                       [np.sin(cy_),  np.cos(cy_), 0],
                       [0, 0, 1]])
        Ry = np.array([[ np.cos(cp_), 0, np.sin(cp_)],
                       [0, 1, 0],
                       [-np.sin(cp_), 0, np.cos(cp_)]])
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(cr_), -np.sin(cr_)],
                       [0, np.sin(cr_),  np.cos(cr_)]])
        self.R_cam = Rz @ Ry @ Rx

        self.cam_pos = np.array([self.camera_params['cam_x'],
                                 self.camera_params['cam_y'],
                                 self.camera_params['cam_z']])

    # ------------------------------------------------------------------ #
    # Drawing
    # ------------------------------------------------------------------ #
    def _draw(self):
        # ---- Remove previous artists -------------------------------------
        for art in self._artists:
            try:
                art.remove()
            except (ValueError, AttributeError):
                pass
        self._artists = []

        # ---- Model --------------------------------------------------------
        before = set(id(a) for a in (self.ax.lines
                                     + self.ax.collections
                                     + self.ax.patches))
        plot_mbd_model_3d(self.bike_params, self.state, 'solver', ax=self.ax)
        self._collect_new_artists(before)

        # ---- Raw image points --------------------------------------------
        if self.image_points is not None:
            self._draw_image_points()

        # ---- Fitted wheel circles ----------------------------------------
        if self.ellipses is not None:
            self._draw_ellipse_circles()

        # ---- Axes: set ONCE, then freeze --------------------------------
        if not self._axes_initialised:
            self._set_fixed_axes()
            self._axes_initialised = True

        self.fig.canvas.draw_idle()

    def _set_fixed_axes(self, margin=0.5):
        """
        Compute a 1:1:1 bounding box from the bike geometry at the current
        state and freeze the axes limits. Called only once.
        """
        try:
            frame_pts, rw_pts, fw_pts, r2, f2 = points2plot(self.bike_params,
                                                            self.state)
        except Exception as e:
            print(f"[interactive] points2plot failed for axes: {e}")
            self.ax.set_xlim3d(-2, 2)
            self.ax.set_ylim3d(-2, 2)
            self.ax.set_zlim3d( 0, 2)
            return

        all_pts = list(frame_pts) + list(rw_pts) + list(fw_pts) \
                  + list(r2) + list(f2)
        pts = np.array(all_pts, dtype=float)

        mins = pts.min(axis=0)
        maxs = pts.max(axis=0)
        center = 0.5 * (mins + maxs)
        half   = 0.5 * np.max(maxs - mins)
        half   = max(half, 0.5) + margin

        self.ax.set_xlim3d(center[0] - half, center[0] + half)
        self.ax.set_ylim3d(center[1] - half, center[1] + half)
        self.ax.set_zlim3d(center[2] - half, center[2] + half)

    def _draw_image_points(self):
        sr = self.screen_resolution
        for cid, pts in self.image_points.items():
            u = pts[:, 0] * sr[0]
            v = (1.0 - pts[:, 1]) * sr[1]

            fx_s = self.camera_params['fx'] * (sr[0] / (self.K[0, 2] * 2))
            fy_s = self.camera_params['fy'] * (sr[1] / (self.K[1, 2] * 2))
            cx_s, cy_s = sr[0] / 2.0, sr[1] / 2.0
            K_s = np.array([[fx_s, 0, cx_s],
                            [0, fy_s, cy_s],
                            [0, 0, 1]])
            invK = np.linalg.inv(K_s)

            world_pts = []
            for ui, vi in zip(u, v):
                d_cam = invK @ np.array([ui, vi, 1.0])
                d_world = self.R_cam @ d_cam
                denom = d_world[2]
                if abs(denom) < 1e-9:
                    continue
                t = (0.0 - self.cam_pos[2]) / denom
                if t <= 0:
                    continue
                world_pts.append(self.cam_pos + t * d_world)

            if len(world_pts):
                wp = np.array(world_pts)
                color = 'red' if cid == 0 else 'blue'
                sc = self.ax.scatter(wp[:, 0], wp[:, 1], wp[:, 2],
                                     c=color, s=2, alpha=0.5)
                self._artists.append(sc)

    def _draw_ellipse_circles(self):
        try:
            frame_pts, rw_pts, fw_pts, r2, f2 = points2plot(self.bike_params,
                                                            self.state)
            (Cr, S, Q, Cf) = frame_pts
            (p1r, p3r) = rw_pts
            (p1f, p3f) = fw_pts
            (p2r, p4r) = r2
            (p2f, p4f) = f2

            cx_r, cy_r, cz_r = pt2circle(p1r, p2r, p3r, p4r)
            cx_f, cy_f, cz_f = pt2circle(p1f, p2f, p3f, p4f)

            l_r, = self.ax.plot(cx_r, cy_r, cz_r,
                                color='blue', lw=1.0, alpha=0.7)
            l_f, = self.ax.plot(cx_f, cy_f, cz_f,
                                color='red',  lw=1.0, alpha=0.7)
            self._artists.append(l_r)
            self._artists.append(l_f)
        except Exception as e:
            print(f"[interactive] Could not draw ellipse circles: {e}")

    def _collect_new_artists(self, before_ids):
        for coll in self.ax.collections:
            if id(coll) not in before_ids:
                self._artists.append(coll)
        for line in self.ax.lines:
            if id(line) not in before_ids:
                self._artists.append(line)
        for patch in self.ax.patches:
            if id(patch) not in before_ids:
                self._artists.append(patch)

    # ------------------------------------------------------------------ #
    # Callbacks
    # ------------------------------------------------------------------ #
    def _on_slider_change(self, _):
        for i, s in enumerate(self.sliders):
            v = s.val
            if i < 4:
                v = np.deg2rad(v)
            self.state[i] = v
        self._draw()

    def _on_reset(self, _):
        self._on_slider_change(None)

    def _on_copy(self, _):
        txt = (f"phi={self.state[0]:.6f}, theta={self.state[1]:.6f}, "
               f"psi={self.state[2]:.6f}, delta={self.state[3]:.6f}, "
               f"x_r={self.state[4]:.6f}, y_r={self.state[5]:.6f}, "
               f"z_r={self.state[6]:.6f}")
        try:
            import pyperclip
            pyperclip.copy(txt)
            print("State copied to clipboard:")
        except Exception:
            print("State (pyperclip unavailable):")
        print(txt)

    def _on_radio(self, label):
        self._draw()

    def _on_fit(self, _):
        from scipy.optimize import least_squares

        if self.image_points is None:
            print("No image data available for fitting.")
            return

        data = self._build_model_data()
        if data is None:
            return

        lower = [-np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2, -20, -20, -20]
        upper = [ np.pi/2,  np.pi/2,  np.pi/2,  np.pi/2,  20,  20,  20]

        try:
            res = least_squares(
                perpro.residual_eqs,
                self.state,
                args=(data, self.bike_params, self.camera_params),
                bounds=(lower, upper),
                max_nfev=200,
            )
            self.state = res.x
            for i, s in enumerate(self.sliders):
                v = self.state[i]
                if i < 4:
                    v = np.rad2deg(v)
                s.set_val(v)
            print("Fit converged. Residual norm:", res.cost)
        except Exception as e:
            print("Fit failed:", e)

        self._draw()

    def _build_model_data(self):
        if (self.image_points is None
                or 0 not in self.image_points
                or 1 not in self.image_points):
            print("Need both front (0) and rear (1) point sets.")
            return None

        sr = self.screen_resolution
        front = self.image_points[0]
        rear = self.image_points[1]

        ef = fitEllipse(front, sr)
        er = fitEllipse(rear, sr)
        if ef is None or er is None:
            print("Ellipse fit failed.")
            return None
        xf, zf, af, bf, thf = ef
        xr, zr, ar, br, thr = er

        p1f, p2f, p3f, p4f = find_points(ef)
        p1r, p2r, p3r, p4r = find_points(er)

        ang_f = np.arccos(bf / af) if af > bf else 0.0
        ang_r = np.arccos(br / ar) if ar > br else 0.0

        denom_f = np.sqrt(
            (np.sin(ang_f) * np.cos(ang_r)
             - np.sin(ang_r) * np.cos(ang_f) * np.cos(thf - thr)) ** 2
            + np.sin(ang_r) ** 2 * np.sin(thf - thr) ** 2
        )
        num_u = ((np.sin(ang_f) * np.cos(ang_r)
                  - np.sin(ang_r) * np.cos(ang_f) * np.cos(thf - thr))
                 * np.sin(thf)
                 + np.sin(ang_r) * np.sin(thf - thr)
                 * np.cos(ang_f) * np.cos(thf))
        num_v = ((np.sin(ang_f) * np.cos(ang_r)
                  - np.sin(ang_r) * np.cos(ang_f) * np.cos(thf - thr))
                 * np.cos(thf)
                 - np.sin(ang_r) * np.sin(thf)
                 * np.sin(thf - thr) * np.cos(ang_f))

        return {
            'r_Cf_Cr_u':  np.array([xf - xr]),
            'r_Cf_Cr_v':  np.array([zf - zr]),
            'u_Cr':       np.array([xr]),
            'v_Cr':       np.array([zr]),
            'r_P1r_Cr_u': np.array([p1r[0] - xr]),
            'r_P1r_Cr_v': np.array([p1r[1] - zr]),
            'r_P3r_Cr_u': np.array([p3r[0] - xr]),
            'r_P3r_Cr_v': np.array([p3r[1] - zr]),
            'r_P1f_Cf_u': np.array([p1f[0] - xf]),
            'r_P1f_Cf_v': np.array([p1f[1] - zf]),
            'r_P3f_Cf_u': np.array([p3f[0] - xf]),
            'r_P3f_Cf_v': np.array([p3f[1] - zf]),
            'r_Q_S_u':    np.array([num_u / denom_f]),
            'r_Q_S_v':    np.array([num_v / denom_f]),
        }


# ---------------------------------------------------------------------- #
# Convenience launcher
# ---------------------------------------------------------------------- #
def launch_interactive(bike_params, camera_params, state0,
                       image_points=None, ellipses=None,
                       screen_resolution=(1920, 1080)):
    viewer = BikeInteractiveViewer(
        bike_params=bike_params,
        camera_params=camera_params,
        state0=state0,
        image_points=image_points,
        ellipses=ellipses,
        screen_resolution=screen_resolution,
    )
    plt.show()
    return viewer