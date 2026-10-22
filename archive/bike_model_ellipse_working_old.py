# src/bike_model_ellipse.py
"""
Replacement for bike_model_perspective.py + the data side of the residuals.
Numpy only (no sympy). Same conventions as bike_model_perspective.py.

What was wrong before
---------------------
1. P1/P3 mismatch. The data (fitEllipse + find_points) gives the ellipse centre and its
   top/bottom points in the image. The model compared them with the projection of the hub
   and of the 3D point straight above the hub. Perspective moves the ellipse centre away
   from the projected hub, and the top of the image ellipse is not the projection of the
   top of the 3D wheel, so the residuals did not vanish at the true pose (~5 px at 6 m).
2. The ellipse WIDTH was never used. P1 and P3 are the same information (P3 = -P1 about
   the centre) and only constrain the ellipse height and tilt, so roll and steer were
   practically unobservable in a near side view (the solver wandered to +-90 deg).
3. The Q-S equations mixed pixels and a unit vector (and the unit vector was not the same
   direction as the model's Q-S), so they are dropped.

What this module does
---------------------
Per wheel it compares the full image ellipse (5 numbers) with the model:
    centre (u, v), top point P1 relative to the centre, right-most point P2 relative to
    the centre.
The model side uses the exact image-space extremes of the projected rim circle
c + r(cos t A + sin t B). For the image coordinate q (v for the top, u for the right):
    dq/dt = 0  <=>  alpha cos t + beta sin t = gamma
    alpha = y0 qb - q0 yb,  beta = q0 ya - y0 qa,  gamma = r (qa yb - qb ya)
(camera-frame components of the hub c_c and of A, B), two solutions = max and min of q.
The centre of the image ellipse is the midpoint of an extreme and its opposite.

Geometry note
-------------
bike_model_perspective.py puts the steering head S at the SAME height as the rear hub and
then goes lf1 down the steering axis, so the front hub ends up lf1 BELOW the rear hub
(0.5 m with the numbers in main.py). With two equal wheels that is impossible on level
ground and the solver compensates with a wrong yaw/pitch. Set bike_params['lh'] = lf1
(head height above the rear hub; default 0 = old behaviour) so both hubs are level.
lr + lf2 is then the hub-to-hub distance: MEASURE IT on your bike (1.4 + 0.1 m is far too
long for a normal bicycle; the frame DSC09360 fits ~1.07 m).

Usage
-----
    import bike_model_ellipse as be
    bike_params['lh'] = bike_params['lf1']
    ef = fitEllipse(front_pts, res); er = fitEllipse(rear_pts, res)       # utils.py
    imgdata = be.image_data(ef, er)                  # or lists of ellipses for a video
    results, states = be.fit_frames(imgdata, x0, bike_params, boundaries, camera_params)

Use |psi| bounds of pi (not pi/2) so a bike travelling to the left of the image is possible.
"""

import numpy as np
from scipy.optimize import least_squares

KEYS = ["r_Cf_Cr_u", "r_Cf_Cr_v", "u_Cr", "v_Cr",
        "r_P1r_Cr_u", "r_P1r_Cr_v", "r_P2r_Cr_u", "r_P2r_Cr_v",     # P1 = top, P2 = right-most
        "r_P1f_Cf_u", "r_P1f_Cf_v", "r_P2f_Cf_u", "r_P2f_Cf_v"]


def _rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _project(P, C, pc, cam):
    r = (np.asarray(P, float) - pc) @ C
    return (cam["fx"] * r[0] / r[1] + cam["cx"],
            cam["fy"] * r[2] / r[1] + cam["cy"])


def _extremes(c, A, B, r, C, pc, cam, axis):
    """(max, min) image points of the projected rim along v (axis='v') or u (axis='u')."""
    c_c, A_c, B_c = C.T @ (c - pc), C.T @ A, C.T @ B
    k = 2 if axis == "v" else 0                 # camera component that becomes v / u
    y0, q0 = c_c[1], c_c[k]
    ya, qa, yb, qb = A_c[1], A_c[k], B_c[1], B_c[k]
    alpha = y0 * qb - q0 * yb
    beta = q0 * ya - y0 * qa
    gamma = r * (qa * yb - qb * ya)
    kk = np.hypot(alpha, beta)
    phase = np.arctan2(beta, alpha)
    d = np.arccos(np.clip(gamma / kk, -1.0, 1.0))
    pts = [_project(c + r * (np.cos(t) * A + np.sin(t) * B), C, pc, cam)
           for t in (phase + d, phase - d)]
    i = 1 if axis == "v" else 0
    return (pts[0], pts[1]) if pts[0][i] >= pts[1][i] else (pts[1], pts[0])


def model_features(x, bike_params, camera_params):
    """The 12 model quantities that correspond to KEYS (pixels)."""
    phi, theta, psi, delta, xr, yr, zr = (float(v) for v in x[:7])
    bp, cam = bike_params, camera_params

    R = _rz(psi) @ _rx(phi) @ _ry(theta)        # orient_body_fixed(N,(psi,phi,theta),'ZXY')
    F = R @ _rz(delta)                          # orient_axis(R, delta, R.z)
    Cr = np.array([xr, yr, zr])
    S = Cr + bp["lr"] * R[:, 0] + bp.get("lh", 0.0) * R[:, 2]
    Q = S - bp["lf1"] * F[:, 2]
    Cf = Q + bp["lf2"] * F[:, 0]

    C = _rz(cam["cam_yaw"]) @ _ry(cam["cam_pitch"]) @ _rx(cam["cam_roll"])
    pc = np.array([cam["cam_x"], cam["cam_y"], cam["cam_z"]])

    out = {}
    for w, (c, M, rad) in {"r": (Cr, R, bp["rr"]), "f": (Cf, F, bp["rf"])}.items():
        top, bot = _extremes(c, M[:, 0], M[:, 2], rad, C, pc, cam, "v")
        rgt, lft = _extremes(c, M[:, 0], M[:, 2], rad, C, pc, cam, "u")
        out[w] = (((top[0] + bot[0]) / 2, (top[1] + bot[1]) / 2),          # ellipse centre
                  ((top[0] - bot[0]) / 2, (top[1] - bot[1]) / 2),          # top - centre
                  ((rgt[0] - lft[0]) / 2, (rgt[1] - lft[1]) / 2))          # right - centre
    (cr, tr, rr), (cf, tf, rf) = out["r"], out["f"]
    return np.array([cf[0] - cr[0], cf[1] - cr[1], cr[0], cr[1], *tr, *rr, *tf, *rf])


def residual_eqs(x, data, bike_params, camera_params):
    """Same signature as bike_model_perspective.residual_eqs; 12 residuals in pixels."""
    d = np.array([data[k] for k in KEYS], float)
    return model_features(x, bike_params, camera_params) - d


# ---------------------------------------------------------------------------
# Data side: fitEllipse output -> the 12 numbers (replaces find_points + the Q-S formula)
# ---------------------------------------------------------------------------
def ellipse_keypoints(e):
    """e = (xc, zc, a, b, theta) from utils.fitEllipse (pixels, v up, theta = major axis).
    Returns centre, top-minus-centre and right-most-minus-centre."""
    xc, zc, a, b, th = (float(v) for v in e[:5])
    s, c = np.sin(th), np.cos(th)
    dz = np.sqrt(a * a * s * s + b * b * c * c)       # half height
    dx = np.sqrt(a * a * c * c + b * b * s * s)       # half width
    k = (a * a - b * b) * s * c
    return (xc, zc), (k / dz, dz), (dx, k / dx)


def image_data(ellipse_front, ellipse_rear):
    """
    One frame: pass two 5-tuples. Video: pass two sequences of 5-tuples (e.g. the 'ellipses'
    lists of process_directory). Returns {key: [value per frame]} as fit_img2model expects.
    """
    ef, er = np.atleast_2d(np.asarray(ellipse_front, float)), np.atleast_2d(np.asarray(ellipse_rear, float))
    rows = []
    for f, r in zip(ef, er):
        (cf, tf, rf), (cr, tr, rr) = ellipse_keypoints(f), ellipse_keypoints(r)
        rows.append([cf[0] - cr[0], cf[1] - cr[1], cr[0], cr[1], *tr, *rr, *tf, *rf])
    rows = np.array(rows)
    return {k: rows[:, i].tolist() for i, k in enumerate(KEYS)}


def synthetic_data(x, bike_params, camera_params):
    """Noise-free data for a state, same format as image_data()."""
    m = model_features(x, bike_params, camera_params)
    return {k: [float(v)] for k, v in zip(KEYS, m)}


# ---------------------------------------------------------------------------
# Solver loop (same as utils.fit_img2model but generic in the keys, + optional multistart)
# ---------------------------------------------------------------------------
def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def fit_frames(data, x0, bike_params, boundaries, camera_params, multistart_first=True):
    """
    Frame by frame least squares, warm-started from the previous frame.
    For the first frame the yaw sign / direction is ambiguous from a single ellipse pair,
    so a few mirrored yaw starts are tried and the lowest cost is kept.
    Returns (results_history, x_history) like fit_img2model.
    """
    lb, ub = (np.asarray(b, float) for b in boundaries)
    x_prev = np.clip(np.asarray(x0, float), lb + 1e-9, ub - 1e-9)
    n = len(data[KEYS[0]])
    results, x_hist = [], [x_prev.copy()]
    for i in range(n):
        frame = {k: data[k][i] for k in KEYS}
        starts = [x_prev]
        if i == 0 and multistart_first:
            p = x_prev[2]
            for psi in (-p, _wrap(np.pi - p), _wrap(p - np.pi)):
                g = x_prev.copy()
                g[2] = psi
                starts.append(np.clip(g, lb + 1e-9, ub - 1e-9))
        best = None
        for g in starts:
            s = least_squares(residual_eqs, g, args=(frame, bike_params, camera_params), bounds=(lb, ub))
            if best is None or s.cost < best.cost:
                best = s
        results.append(best)
        x_prev = best.x.copy()
        x_hist.append(x_prev.copy())
    return results, x_hist
