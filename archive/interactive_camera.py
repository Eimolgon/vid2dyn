# src/interactive_camera.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

import bike_model_perspective as perpro
from utils import points2plot, fitEllipse, find_points


# Default camera parameters – override via the `camera_params` argument.
DEFAULT_CAMERA = {
    'fx': 8000.0, 'fy': 8000.0,
    'cx': 3000.0, 'cy': 2000.0,
    'cam_x': 0.0, 'cam_y': 0.0, 'cam_z': 0.45,
    'cam_yaw': 0.0, 'cam_pitch': 0.0, 'cam_roll': 0.0,
}


class CameraInteractiveViewer:
    """
    2D projection viewer with sliders for the camera parameters.

    The state of the bike is fixed (passed at construction). The sliders
    modify the camera only. Two projections are shown side by side:

    * the raw image points (as read from the label files),
    * the model's projected points for the current camera parameters.
    """

    CAMERA_LABELS = [
        ('fx',        'fx [px]',     500.0,   20000.0),
        ('fy',        'fy [px]',     500.0,   20000.0),
        ('cx',        'cx [px]',       0.0,   10000.0),
        ('cy',        'cy [px]',       0.0,   10000.0),
        ('cam_x',     'cam_x [m]',  -10.0,      10.0),
        ('cam_y',     'cam_y [m]',  -10.0,      10.0),
        ('cam_z',     'cam_z [m]',   -5.0,       5.0),
        ('cam_yaw',   'yaw [deg]',  -180.0,    180.0),
        ('cam_pitch', 'pitch [deg]',-180.0,    180.0),
        ('cam_roll',  'roll [deg]', -180.0,    180.0),
    ]

    def __init__(self, bike_params, camera_params, state,
                 image_points=None, screen_resolution=(1920, 1080)):
        self.bike_params = dict(bike_params)
        self.camera_params = dict(DEFAULT_CAMERA)
        self.camera_params.update(camera_params or {})
        self.state = np.array(state, dtype=float)
        self.image_points = image_points
        self.screen_resolution = screen_resolution

        # --- Figure ---
        self.fig = plt.figure(figsize=(14, 8))
        self.ax = self.fig.add_axes([0.30, 0.28, 0.66, 0.66])
        self.ax.set_aspect('equal')
        self.ax.set_xlabel('u [px]')
        self.ax.set_ylabel('v [px]')
        self.ax.set_title('2D projection – camera sliders')
        self.ax.invert_yaxis()  # image convention: v grows downward
        self.ax.grid(True)

        # Artists to redraw
        self._artists = []

        # --- Sliders ---
        self.sliders = []
        n = len(self.CAMERA_LABELS)
        bottom, h, gap = 0.05, 0.025, 0.008
        for i, (key, label, lo, hi) in enumerate(self.CAMERA_LABELS):
            ax_s = self.fig.add_axes(
                [0.05, bottom + (n - 1 - i) * (h + gap), 0.20, h])
            val = self.camera_params[key]
            if key in ('cam_yaw', 'cam_pitch', 'cam_roll'):
                val = np.rad2deg(val)
            s = Slider(ax_s, label, lo, hi, valinit=val)
            s.on_changed(self._on_slider_change)
            self.sliders.append(s)

        # --- Buttons ---
        ax_reset = self.fig.add_axes([0.05, 0.005, 0.09, 0.03])
        self.btn_reset = Button(ax_reset, 'Reset')
        self.btn_reset.on_clicked(self._on_reset)

        ax_copy = self.fig.add_axes([0.16, 0.005, 0.11, 0.03])
        self.btn_copy = Button(ax_copy, 'Copy camera')
        self.btn_copy.on_clicked(self._on_copy)

        # --- First draw ---
        self._draw()
        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------ #
    # Drawing
    # ------------------------------------------------------------------ #
    def _draw(self):
        # --- Remove old artists ---
        for art in self._artists:
            try:
                art.remove()
            except (ValueError, AttributeError):
                pass
        self._artists = []

        # --- Raw image points (if provided) ---
        if self.image_points is not None:
            sr = self.screen_resolution
            for cid, pts in self.image_points.items():
                u = pts[:, 0] * sr[0]
                v = (1.0 - pts[:, 1]) * sr[1]
                color = 'red' if cid == 0 else 'blue'
                sc = self.ax.scatter(u, v, s=4, c=color, alpha=0.4,
                                     label=f'class {cid}')
                self._artists.append(sc)

        # --- Model projection ---
        proj = self._project_model()
        if proj is not None:
            for cid, pts in proj.items():
                color = 'darkred' if cid == 0 else 'darkblue'
                mark = 'x' if cid == 0 else '+'
                sc = self.ax.scatter(pts[:, 0], pts[:, 1],
                                     s=60, c=color, marker=mark,
                                     label=f'model {cid}')
                self._artists.append(sc)

        # --- Axis bounds: keep the sensor frame in view ---
        cx = self.camera_params['cx']
        cy = self.camera_params['cy']
        half_w = max(cx, 1.0)
        half_h = max(cy, 1.0)
        self.ax.set_xlim(cx - 1.3 * half_w, cx + 1.3 * half_w)
        self.ax.set_ylim(cy + 1.3 * half_h, cy - 1.3 * half_h)  # inverted

        # A small legend
        self.ax.legend(loc='upper right', fontsize=8)

    def _project_model(self):
        """
        Project the model's wheel contact points and rim points using
        bike_model_perspective.eval_f03..eval_f12.
        Returns a dict {0: front pts (Nx2), 1: rear pts (Nx2)} in pixels.
        """
        try:
            frame_pts, rw_pts, fw_pts, r2, f2 = points2plot(self.bike_params,
                                                            self.state)
        except Exception as e:
            print(f"[camera_view] points2plot failed: {e}")
            return None

        # Reuse the same lambdified functions used by perpro.residual_eqs
        subs = (
            self.state[0], self.state[1], self.state[2], self.state[3],
            self.state[4], self.state[5], self.state[6],
            self.bike_params['lr'], self.bike_params['lf1'],
            self.bike_params['lf2'], self.bike_params['rr'],
            self.bike_params['rf'],
            self.camera_params['fx'], self.camera_params['fy'],
            self.camera_params['cx'], self.camera_params['cy'],
            self.camera_params['cam_x'], self.camera_params['cam_y'],
            self.camera_params['cam_z'], self.camera_params['cam_yaw'],
            self.camera_params['cam_pitch'], self.camera_params['cam_roll'],
        )

        def safe_eval(fn):
            try:
                val = fn(*subs)
                return float(np.asarray(val).ravel()[0])
            except Exception:
                return np.nan

        # Wheel-contact / frame points (u_Cr, v_Cr, u_Cf, v_Cf, ...)
        u_Cr = safe_eval(perpro.eval_f03)
        v_Cr = safe_eval(perpro.eval_f04)
        u_Cf = u_Cr + safe_eval(perpro.eval_f01)
        v_Cf = v_Cr + safe_eval(perpro.eval_f02)

        # Rear rim points (relative to Cr)
        u_P1r = u_Cr + safe_eval(perpro.eval_f05)
        v_P1r = v_Cr + safe_eval(perpro.eval_f06)
        u_P3r = u_Cr + safe_eval(perpro.eval_f07)
        v_P3r = v_Cr + safe_eval(perpro.eval_f08)

        # Front rim points (relative to Cf)
        u_P1f = u_Cf + safe_eval(perpro.eval_f09)
        v_P1f = v_Cf + safe_eval(perpro.eval_f10)
        u_P3f = u_Cf + safe_eval(perpro.eval_f11)
        v_P3f = v_Cf + safe_eval(perpro.eval_f12)

        front = np.array([
            [u_Cf,  v_Cf],
            [u_P1f, v_P1f],
            [u_P3f, v_P3f],
        ])
        rear = np.array([
            [u_Cr,  v_Cr],
            [u_P1r, v_P1r],
            [u_P3r, v_P3r],
        ])
        return {0: front, 1: rear}

    # ------------------------------------------------------------------ #
    # Callbacks
    # ------------------------------------------------------------------ #
    def _on_slider_change(self, _):
        for i, (key, *_rest) in enumerate(self.CAMERA_LABELS):
            v = self.sliders[i].val
            if key in ('cam_yaw', 'cam_pitch', 'cam_roll'):
                v = np.deg2rad(v)
            self.camera_params[key] = v
        self._draw()

    def _on_reset(self, _):
        for i, (key, *_rest) in enumerate(self.CAMERA_LABELS):
            v = DEFAULT_CAMERA[key]
            if key in ('cam_yaw', 'cam_pitch', 'cam_roll'):
                v = np.rad2deg(v)
            self.sliders[i].set_val(v)
        self._on_slider_change(None)

    def _on_copy(self, _):
        txt = ",\n".join(
            f"    '{k}': {self.camera_params[k]:.6f}"
            for k, *_ in self.CAMERA_LABELS
        )
        txt = "camera_params = {\n" + txt + "\n}"
        try:
            import pyperclip
            pyperclip.copy(txt)
            print("Camera parameters copied to clipboard:")
        except Exception:
            print("Camera parameters (pyperclip unavailable):")
        print(txt)


# ---------------------------------------------------------------------- #
# Launcher
# ---------------------------------------------------------------------- #
def launch_camera_view(bike_params, camera_params, state,
                       image_points=None, screen_resolution=(1920, 1080)):
    viewer = CameraInteractiveViewer(
        bike_params=bike_params,
        camera_params=camera_params,
        state=state,
        image_points=image_points,
        screen_resolution=screen_resolution,
    )
    plt.show()
    return viewer