# src/bike_model_ellipse_symbolic.py
"""
Symbolic (sympy) twin of bike_model_ellipse.py
==============================================

Same model, same 12 features, same API and the same numbers as the numpy module
`bike_model_ellipse.py` -- but written the way `bike_model_perspective.py` was: with
sympy.physics.mechanics frames and points, then lambdified. Use whichever you find easier
to read, check or extend; they are interchangeable (`import bike_model_ellipse_symbolic as be`).

What the model computes
-----------------------
For every pose x = [phi, theta, psi, delta, x_r, y_r, z_r] it predicts what the image
ellipse fit of each wheel would measure:

    ellipse centre (u, v)                         -> u_Cr, v_Cr and  r_Cf_Cr (front - rear)
    top-most point P1, relative to the centre     -> r_P1r_Cr_*  (rear),  r_P1f_Cf_*  (front)
    right-most point P2, relative to the centre   -> r_P2r_Cr_*  (rear),  r_P2f_Cf_*  (front)

(P1..P4 keep the numbering of the original file: P1 top, P2 right, P3 bottom, P4 left,
 but they are now the IMAGE extremes of the projected rim, see below.)

Frames and conventions (identical to bike_model_perspective.py)
---------------------------------------------------------------
    N  world        x right in the image, y depth (away from the camera), z up
    R  rear frame   x forward, y left, z up      R = orient_body_fixed(N, (psi, phi, theta), 'ZXY')
    F  front frame  R rotated by delta about R.z F = orient_axis(R, delta, R.z)
    C  camera       x right, y optical axis, z up C = orient_body_fixed(N, (yaw, pitch, roll), 'ZYX')
    pixels          u to the right, v UP (labels: v = (1 - y_yolo) * H, as fitEllipse does)
    projection      u = fx Xc/Yc + cx,  v = fy Zc/Yc + cy      (Xc, Yc, Zc: camera components)

Points (all positions in N):
    Cr  rear wheel CENTRE (hub)  = (x_r, y_r, z_r)
    S   steering head            = Cr + lr R.x + lh R.z
    Q   fork bottom              = S - lf1 F.z
    Cf  front wheel centre       = Q + lf2 F.x
    P   camera pinhole           = (cam_x, cam_y, cam_z)

Differences with bike_model_perspective.py (the reasons are in the numpy module's docstring)
    1. lh: head height above the rear hub. lh = 0 is your old geometry, lh = lf1 puts both
       hubs level (then lf1 cancels exactly: F.z = R.z, so only lr and lf2 matter).
    2. 12 features instead of 14, and they are the quantities an ellipse fit really measures:
       centre, top point and right-most point of the PROJECTED rim (not the projection of
       hand-picked 3D rim points). The Q-S equations are gone.
    3. residual_eqs returns 12 residuals in pixels.

How the image extremes of the rim are found (the only non-obvious step)
-----------------------------------------------------------------------
The rim is the circle  X(t) = c + r (cos t A + sin t B)  (A, B: the wheel-plane axes, i.e.
R.x, R.z for the rear and F.x, F.z for the front wheel). In camera components
(c0, c1, c2) = centre - P, (a0, a1, a2) = A, (b0, b1, b2) = B, an image coordinate is
    q(t) = f * X_q(t) / X_depth(t)           q = u for component 0, q = v for component 2
Setting dq/dt = 0 gives a single trigonometric equation
    alpha cos t + beta sin t = gamma
    alpha = c1 b_q - c_q b1,    beta = c_q a1 - c1 a_q,    gamma = r (a_q b1 - b_q a1)
with the two solutions
    t = atan2(beta, alpha) +- acos(gamma / sqrt(alpha^2 + beta^2))
which are the maximum and the minimum of q along the rim. Evaluating q for both and keeping
the larger/smaller one gives the top/bottom (q = v) or right/left (q = u) rim point.
The projection of a circle is an ellipse, which is centrally symmetric, so the ellipse
centre is the midpoint of the top and the bottom point -- exactly the centre fitEllipse finds.
(Not the projection of the hub: perspective shifts the two apart.)

Usage (same as bike_model_ellipse)
-----
    import bike_model_ellipse_symbolic as be
    bike_params['lh'] = bike_params['lf1']
    ef = fitEllipse(front_pts, res); er = fitEllipse(rear_pts, res)
    imgdata = be.image_data(ef, er)
    results, states = be.fit_frames(imgdata, x0, bike_params, boundaries, camera_params)

Because the model is symbolic you can also inspect or differentiate it:
    be.FEATURES['r_P1r_Cr_v']                      # sympy expression of one feature
    sp.diff(be.FEATURES['u_Cr'], be.psi)           # e.g. an analytic derivative
Building the expressions and compiling them takes ~3 s at import; one evaluation of the
12 features for 300 poses takes a few milliseconds.
"""

import numpy as np
import sympy as sp
import sympy.physics.mechanics as me
from scipy.optimize import least_squares

# Order of the 12 data / model quantities (same as bike_model_ellipse.KEYS)
KEYS = ["r_Cf_Cr_u", "r_Cf_Cr_v", "u_Cr", "v_Cr",
        "r_P1r_Cr_u", "r_P1r_Cr_v", "r_P2r_Cr_u", "r_P2r_Cr_v",     # P1 = top, P2 = right-most
        "r_P1f_Cf_u", "r_P1f_Cf_v", "r_P2f_Cf_u", "r_P2f_Cf_v"]


# =============================================================================
# 1. Symbols
# =============================================================================
# State
phi, theta, psi, delta = sp.symbols('varphi theta psi delta')   # roll, pitch, yaw, steer
x_r, y_r, z_r = sp.symbols('x_r y_r z_r')                       # rear hub position in N

# Bike parameters
lr, lf1, lf2, lh = sp.symbols('l_r l_f1 l_f2 l_h')
rr, rf = sp.symbols('rr rf')                                    # rear / front wheel radius

# Camera intrinsics and extrinsics
fx, fy, cx, cy = sp.symbols('fx fy cx cy')
cam_x, cam_y, cam_z = sp.symbols('cam_x cam_y cam_z')
cam_yaw, cam_pitch, cam_roll = sp.symbols('cam_yaw cam_pitch cam_roll')

# The order in which the lambdified function takes its arguments (and the dict keys that feed it)
STATE_SYMBOLS = (phi, theta, psi, delta, x_r, y_r, z_r)
BIKE_KEYS = ("lr", "lf1", "lf2", "lh", "rr", "rf")
BIKE_SYMBOLS = (lr, lf1, lf2, lh, rr, rf)
CAMERA_KEYS = ("fx", "fy", "cx", "cy", "cam_x", "cam_y", "cam_z", "cam_yaw", "cam_pitch", "cam_roll")
CAMERA_SYMBOLS = (fx, fy, cx, cy, cam_x, cam_y, cam_z, cam_yaw, cam_pitch, cam_roll)
variables = STATE_SYMBOLS + BIKE_SYMBOLS + CAMERA_SYMBOLS


# =============================================================================
# 2. Reference frames and points (the kinematic chain)
# =============================================================================
N, R, F, C = sp.symbols('N, R, F, C', cls=me.ReferenceFrame)

R.orient_body_fixed(N, (psi, phi, theta), 'ZXY')     # rear frame
F.orient_axis(R, delta, R.z)                         # front frame: steering about R.z
C.orient_body_fixed(N, (cam_yaw, cam_pitch, cam_roll), 'ZYX')   # camera frame

O = me.Point('O')       # world origin
Cr = me.Point('C_r')    # rear wheel centre
S = me.Point('S')       # steering head
Q = me.Point('Q')       # fork bottom
Cf = me.Point('C_f')    # front wheel centre
P = me.Point('P')       # camera pinhole

O.set_vel(N, 0)
Cr.set_pos(O, x_r * N.x + y_r * N.y + z_r * N.z)
S.set_pos(Cr, lr * R.x + lh * R.z)                   # lh = 0 -> your original geometry
Q.set_pos(S, -lf1 * F.z)
Cf.set_pos(Q, lf2 * F.x)
P.set_pos(O, cam_x * N.x + cam_y * N.y + cam_z * N.z)


# =============================================================================
# 3. Perspective projection
# =============================================================================
def camera_components(vector):
    """Components (Xc, Yc, Zc) of a vector along the camera axes C.x (right), C.y (depth), C.z (up)."""
    return vector.dot(C.x), vector.dot(C.y), vector.dot(C.z)


def perspective_projection(point):
    """Pixel coordinates (u, v) of a point: u = fx Xc/Yc + cx,  v = fy Zc/Yc + cy."""
    Xc, Yc, Zc = camera_components(point.pos_from(P))
    return fx * Xc / Yc + cx, fy * Zc / Yc + cy


# =============================================================================
# 4. Image extremes of a projected rim
# =============================================================================
def rim_extreme_angles(centre, axis_a, axis_b, radius, comp):
    """
    Rim angles t of the two points of the circle  centre + radius (cos t axis_a + sin t axis_b)
    where the image coordinate q is largest and smallest.
        comp = 2 -> q = v : returns (t_top,   t_bottom)
        comp = 0 -> q = u : returns (t_right, t_left)
    See the module docstring for the derivation.
    """
    c = camera_components(centre.pos_from(P))
    c1, cq = c[1], c[comp]                            # depth and q-component of the wheel centre
    a = camera_components(axis_a)
    b = camera_components(axis_b)
    a1, aq, b1, bq = a[1], a[comp], b[1], b[comp]

    alpha = c1 * bq - cq * b1
    beta = cq * a1 - c1 * aq
    gamma = radius * (aq * b1 - bq * a1)
    ratio = sp.Min(1, sp.Max(-1, gamma / sp.sqrt(alpha**2 + beta**2)))   # clip like numpy does
    phase = sp.atan2(beta, alpha)
    spread = sp.acos(ratio)
    t_a, t_b = phase + spread, phase - spread

    def image_q(t):                                   # q(t) of the rim point at angle t
        point = me.Point('tmp')
        point.set_pos(centre, radius * (sp.cos(t) * axis_a + sp.sin(t) * axis_b))
        return perspective_projection(point)[1 if comp == 2 else 0]

    a_is_lower = image_q(t_a) < image_q(t_b)
    t_hi = sp.Piecewise((t_b, a_is_lower), (t_a, True))
    t_lo = sp.Piecewise((t_a, a_is_lower), (t_b, True))
    return t_hi, t_lo


def rim_point(name, centre, axis_a, axis_b, radius, t):
    """3D rim point at angle t as a mechanics Point."""
    point = me.Point(name)
    point.set_pos(centre, radius * (sp.cos(t) * axis_a + sp.sin(t) * axis_b))
    return point


def wheel_keypoints(tag, centre, axis_a, axis_b, radius):
    """
    The four image-extreme rim points of one wheel (tag 'r' or 'f'), named like in the original file:
        P1 top, P2 right-most, P3 bottom, P4 left-most
    Returns the points and the pixel-space features
        (u_centre, v_centre, P1 - centre, P2 - centre)
    """
    t_top, t_bottom = rim_extreme_angles(centre, axis_a, axis_b, radius, comp=2)
    t_right, t_left = rim_extreme_angles(centre, axis_a, axis_b, radius, comp=0)
    P1 = rim_point(f'P_1{tag}', centre, axis_a, axis_b, radius, t_top)
    P2 = rim_point(f'P_2{tag}', centre, axis_a, axis_b, radius, t_right)
    P3 = rim_point(f'P_3{tag}', centre, axis_a, axis_b, radius, t_bottom)
    P4 = rim_point(f'P_4{tag}', centre, axis_a, axis_b, radius, t_left)

    (u1, v1), (u2, v2) = perspective_projection(P1), perspective_projection(P2)
    (u3, v3), (u4, v4) = perspective_projection(P3), perspective_projection(P4)

    u_centre, v_centre = (u1 + u3) / 2, (v1 + v3) / 2      # centre of the image ellipse
    features = (u_centre, v_centre,
                (u1 - u3) / 2, (v1 - v3) / 2,               # P1 relative to the centre (top - centre)
                (u2 - u4) / 2, (v2 - v4) / 2)               # P2 relative to the centre (right - centre)
    return (P1, P2, P3, P4), features


# =============================================================================
# 5. The 12 features
# =============================================================================
_, (u_Cr, v_Cr, d_P1r_u, d_P1r_v, d_P2r_u, d_P2r_v) = wheel_keypoints('r', Cr, R.x, R.z, rr)
_, (u_Cf, v_Cf, d_P1f_u, d_P1f_v, d_P2f_u, d_P2f_v) = wheel_keypoints('f', Cf, F.x, F.z, rf)

r_Cf_Cr_u = u_Cf - u_Cr
r_Cf_Cr_v = v_Cf - v_Cr
r_P1r_Cr_u, r_P1r_Cr_v = d_P1r_u, d_P1r_v
r_P2r_Cr_u, r_P2r_Cr_v = d_P2r_u, d_P2r_v
r_P1f_Cf_u, r_P1f_Cf_v = d_P1f_u, d_P1f_v
r_P2f_Cf_u, r_P2f_Cf_v = d_P2f_u, d_P2f_v

# {key: sympy expression}, in the order of KEYS
FEATURES = {"r_Cf_Cr_u": r_Cf_Cr_u, "r_Cf_Cr_v": r_Cf_Cr_v, "u_Cr": u_Cr, "v_Cr": v_Cr,
            "r_P1r_Cr_u": r_P1r_Cr_u, "r_P1r_Cr_v": r_P1r_Cr_v,
            "r_P2r_Cr_u": r_P2r_Cr_u, "r_P2r_Cr_v": r_P2r_Cr_v,
            "r_P1f_Cf_u": r_P1f_Cf_u, "r_P1f_Cf_v": r_P1f_Cf_v,
            "r_P2f_Cf_u": r_P2f_Cf_u, "r_P2f_Cf_v": r_P2f_Cf_v}

# =============================================================================
# 6. Lambdify (one compiled function for all 12 features; cse = share common sub-expressions)
# =============================================================================
_eval_features = sp.lambdify(variables, [FEATURES[k] for k in KEYS], modules="numpy", cse=True)


# =============================================================================
# 7. Numeric interface (same signatures as bike_model_ellipse)
# =============================================================================
def model_features_batch(X, bike_params, camera_params):
    """X: (N,7) states -> (N,12) model quantities corresponding to KEYS (pixels)."""
    X = np.atleast_2d(np.asarray(X, float))
    bp = {"lh": 0.0, **bike_params}                  # lh defaults to 0 (original geometry)
    args = (*X[:, :7].T,
            *(bp[k] for k in BIKE_KEYS),
            *(camera_params[k] for k in CAMERA_KEYS))
    cols = np.broadcast_arrays(*_eval_features(*args))
    return np.column_stack(cols)


def model_features(x, bike_params, camera_params):
    """The 12 model quantities that correspond to KEYS (pixels), one pose."""
    return model_features_batch(np.asarray(x, float)[None, :7], bike_params, camera_params)[0]


def residual_eqs(x, data, bike_params, camera_params):
    """Same signature as bike_model_perspective.residual_eqs; 12 residuals in pixels (model - data)."""
    d = np.array([data[k] for k in KEYS], float)
    return model_features(x, bike_params, camera_params) - d


# =============================================================================
# 8. Numeric helpers (identical to bike_model_ellipse.py; used by the camera,
#    the viewer and the synthetic-data generators)
# =============================================================================
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



# =============================================================================
# 9. Self-check:  python bike_model_ellipse_symbolic.py
# =============================================================================
if __name__ == "__main__":
    import time

    res = (1920, 1080)
    cam = make_camera(res, focal_mm=28.0, sensor_w_mm=23.5, orig_res=(6020, 4024))
    bp = dict(lr=1.0, lf1=0.5, lf2=0.1, lh=0.5, rr=0.6604 / 2, rf=0.6604 / 2)
    rng = np.random.default_rng(0)
    n = 300
    X = np.column_stack([rng.uniform(-.4, .4, n), rng.uniform(-.2, .2, n), rng.uniform(-np.pi, np.pi, n),
                         rng.uniform(-.6, .6, n), rng.uniform(-2, 2, n), rng.uniform(3, 9, n),
                         rng.uniform(.2, .5, n)])

    t0 = time.time()
    feats = model_features_batch(X, bp, cam)
    print(f"[speed]   12 features for {n} poses: {1e3 * (time.time() - t0):.1f} ms")

    try:                                              # 1. agree with the numpy implementation?
        import bike_model_ellipse as numeric
        diff = np.abs(feats - numeric.model_features_batch(X, bp, cam)).max()
        print(f"[numpy]   max |symbolic - numpy| over {n} random poses: {diff:.2e} px")
    except ImportError:
        print("[numpy]   bike_model_ellipse.py not found next to this file: comparison skipped")

    x_true = np.array([np.deg2rad(-3), np.deg2rad(1), np.deg2rad(160), np.deg2rad(8), 0.6, 5.0, 0.33])
    data = synthetic_data(x_true, bp, cam)            # 2. noise-free data -> zero residual at the truth
    frame = {k: v[0] for k, v in data.items()}
    print(f"[truth]   max |residual| at the true pose: {np.abs(residual_eqs(x_true, frame, bp, cam)).max():.1e} px")

    lb = np.array([-np.pi / 2, -np.pi / 2, -np.pi, -np.pi / 2, -20, 0.1, -20])
    ub = np.array([np.pi / 2, np.pi / 2, np.pi, np.pi / 2, 20, 20, 20])
    x0 = x_true + np.array([.05, .03, .15, .1, .2, .3, .03])      # 3. recover it from a nearby start
    out, _ = fit_frames(data, x0, bp, (lb, ub), cam, multistart_first=False)
    print("[fit]     true     :", np.round(np.r_[np.rad2deg(x_true[:4]), x_true[4:]], 3))
    print("          recovered:", np.round(np.r_[np.rad2deg(out[0].x[:4]), out[0].x[4:]], 3))
