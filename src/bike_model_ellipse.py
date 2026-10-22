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


def _Rx(a):
    a = np.asarray(a, float)
    c, s = np.cos(a), np.sin(a)
    M = np.zeros(a.shape + (3, 3))
    M[..., 0, 0] = 1
    M[..., 1, 1], M[..., 1, 2], M[..., 2, 1], M[..., 2, 2] = c, -s, s, c
    return M


def _Ry(a):
    a = np.asarray(a, float)
    c, s = np.cos(a), np.sin(a)
    M = np.zeros(a.shape + (3, 3))
    M[..., 1, 1] = 1
    M[..., 0, 0], M[..., 0, 2], M[..., 2, 0], M[..., 2, 2] = c, s, -s, c
    return M


def _Rz(a):
    a = np.asarray(a, float)
    c, s = np.cos(a), np.sin(a)
    M = np.zeros(a.shape + (3, 3))
    M[..., 2, 2] = 1
    M[..., 0, 0], M[..., 0, 1], M[..., 1, 0], M[..., 1, 1] = c, -s, s, c
    return M


def camera_matrix(cam):
    """Columns = camera x (right), y (optical axis), z (up) in the world frame."""
    return _Rz(cam["cam_yaw"]) @ _Ry(cam["cam_pitch"]) @ _Rx(cam["cam_roll"])


def _extremes_batch(c, A, B, r, C, pc, cam, comp):
    """
    Image-space extremes of the projected rims c + r(cos t A + sin t B), for N poses at once.
    comp=2 -> along v (top, bottom), comp=0 -> along u (right, left).
    Returns ((u_hi, v_hi), (u_lo, v_lo)), each an (N,) array.
    """
    c_c, A_c, B_c = (c - pc) @ C, A @ C, B @ C
    y0, q0 = c_c[:, 1], c_c[:, comp]
    ya, qa, yb, qb = A_c[:, 1], A_c[:, comp], B_c[:, 1], B_c[:, comp]
    alpha = y0 * qb - q0 * yb
    beta = q0 * ya - y0 * qa
    gamma = r * (qa * yb - qb * ya)
    kk = np.hypot(alpha, beta)
    phase = np.arctan2(beta, alpha)
    d = np.arccos(np.clip(gamma / kk, -1.0, 1.0))
    pts = []
    for t in (phase + d, phase - d):
        P = c + r * (np.cos(t)[:, None] * A + np.sin(t)[:, None] * B)
        rc = (P - pc) @ C
        pts.append((cam["fx"] * rc[:, 0] / rc[:, 1] + cam["cx"],
                    cam["fy"] * rc[:, 2] / rc[:, 1] + cam["cy"]))
    i = 1 if comp == 2 else 0
    swap = pts[0][i] < pts[1][i]
    hi = (np.where(swap, pts[1][0], pts[0][0]), np.where(swap, pts[1][1], pts[0][1]))
    lo = (np.where(swap, pts[0][0], pts[1][0]), np.where(swap, pts[0][1], pts[1][1]))
    return hi, lo


def model_features_batch(X, bike_params, camera_params):
    """X: (N,7) states -> (N,12) model quantities corresponding to KEYS (pixels)."""
    X = np.atleast_2d(np.asarray(X, float))
    bp, cam = bike_params, camera_params
    phi, theta, psi, delta = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    Cr = X[:, 4:7]
    R = _Rz(psi) @ _Rx(phi) @ _Ry(theta)        # orient_body_fixed(N,(psi,phi,theta),'ZXY')
    F = R @ _Rz(delta)                          # orient_axis(R, delta, R.z)
    S = Cr + bp["lr"] * R[:, :, 0] + bp.get("lh", 0.0) * R[:, :, 2]
    Q = S - bp["lf1"] * F[:, :, 2]
    Cf = Q + bp["lf2"] * F[:, :, 0]
    C = camera_matrix(cam)
    pc = np.array([cam["cam_x"], cam["cam_y"], cam["cam_z"]])

    out = {}
    for w, (c, M, rad) in {"r": (Cr, R, bp["rr"]), "f": (Cf, F, bp["rf"])}.items():
        A, B = M[:, :, 0], M[:, :, 2]
        (tu, tv), (bu, bv) = _extremes_batch(c, A, B, rad, C, pc, cam, 2)
        (ru, rv), (lu, lv) = _extremes_batch(c, A, B, rad, C, pc, cam, 0)
        out[w] = ((tu + bu) / 2, (tv + bv) / 2,                  # ellipse centre
                  (tu - bu) / 2, (tv - bv) / 2,                  # top - centre
                  (ru - lu) / 2, (rv - lv) / 2)                  # right - centre
    cu_r, cv_r, tu_r, tv_r, ru_r, rv_r = out["r"]
    cu_f, cv_f, tu_f, tv_f, ru_f, rv_f = out["f"]
    return np.column_stack([cu_f - cu_r, cv_f - cv_r, cu_r, cv_r,
                            tu_r, tv_r, ru_r, rv_r, tu_f, tv_f, ru_f, rv_f])


def model_features(x, bike_params, camera_params):
    """The 12 model quantities that correspond to KEYS (pixels), one pose."""
    return model_features_batch(np.asarray(x, float)[None, :7], bike_params, camera_params)[0]


def residual_eqs(x, data, bike_params, camera_params):
    """Same signature as bike_model_perspective.residual_eqs; 12 residuals in pixels."""
    d = np.array([data[k] for k in KEYS], float)
    return model_features(x, bike_params, camera_params) - d


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------
def make_camera(res, focal_mm=28.0, sensor_w_mm=23.5, orig_res=None, sensor_h_mm=None,
                cam_pos=(0.0, 0.0, 0.45), cam_angles_deg=(0.0, 0.0, 0.0)):
    """
    Pinhole intrinsics for frames of resolution res=(W,H) (the resolution the labels are
    normalised to).
        fx = focal_mm / sensor_w_mm * W                       [px]
    Only the RATIO focal_mm/sensor_w_mm matters: if the sensor is unknown, pass any sensor
    width together with a focal length that is consistent with it (e.g. the 35 mm
    equivalent focal length with sensor_w_mm = 36), or pass the focal length in pixels
    through focal_px.
        fy : if the original resolution orig_res=(W0,H0) is given (square pixels there):
                 fy = fx * (H/W) / (H0/W0)       <- handles a non-uniform resize, no sensor height
             elif sensor_h_mm:  fy = focal_mm / sensor_h_mm * H
             else:              fy = fx
    """
    W, H = res
    fx = focal_mm / sensor_w_mm * W
    if orig_res is not None:
        fy = fx * (H / W) / (orig_res[1] / orig_res[0])
    elif sensor_h_mm:
        fy = focal_mm / sensor_h_mm * H
    else:
        fy = fx
    return {"fx": fx, "fy": fy, "cx": W / 2.0, "cy": H / 2.0,
            "cam_x": cam_pos[0], "cam_y": cam_pos[1], "cam_z": cam_pos[2],
            "cam_yaw": np.deg2rad(cam_angles_deg[0]), "cam_pitch": np.deg2rad(cam_angles_deg[1]),
            "cam_roll": np.deg2rad(cam_angles_deg[2])}


def scale_focal(cam, factor):
    """Copy of cam with fx, fy multiplied by factor (aspect fy/fx is kept)."""
    c = dict(cam)
    c["fx"], c["fy"] = cam["fx"] * factor, cam["fy"] * factor
    return c


def hfov_deg(cam, res):
    return float(np.rad2deg(2 * np.arctan(res[0] / 2 / cam["fx"])))


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


def initial_guess(ellipse_front, ellipse_rear, bike_params, camera_params):
    """
    Closed-form rough pose from one pair of ellipses (camera ~ looking along +y):
    depth from the apparent size of the rear wheel, x/z from its centre, yaw from the
    aspect ratio (|cos psi| = b/a, corrected for fy/fx) and the travel direction.
    """
    xr, zr, ar, br, _ = (float(v) for v in ellipse_rear[:5])
    xf, zf, af, bf, _ = (float(v) for v in ellipse_front[:5])
    cam, bp = camera_params, bike_params
    asp = cam["fy"] / cam["fx"]                      # a non-uniform resize squashes b by fy/fx
    Y = cam["fx"] * bp["rr"] / max(ar, 1e-6)
    x = (xr - cam["cx"]) * Y / cam["fx"] + cam["cam_x"]
    z = (zr - cam["cy"]) * Y / cam["fy"] + cam["cam_z"]
    alpha = np.mean([np.arccos(np.clip(bf / (af * asp), 0, 1)), np.arccos(np.clip(br / (ar * asp), 0, 1))])
    front_farther = af < ar
    if xf >= xr:                                     # travelling to the right of the image
        psi = alpha if front_farther else -alpha
    else:                                            # travelling to the left
        psi = (np.pi - alpha) if front_farther else -(np.pi - alpha)
    return np.array([0.0, 0.0, psi, 0.0, x, Y + cam["cam_y"], z])


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
