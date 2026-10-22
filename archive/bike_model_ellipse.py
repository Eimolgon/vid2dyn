# src/bike_model_ellipse.py
"""
Drop-in replacement for bike_model_perspective.py that fixes the P1/P3 mismatch.

Problem in bike_model_perspective.py
    The data side (fitEllipse + find_points) gives, per wheel, the ellipse centre and
    the TOP / BOTTOM points of the ellipse in the image. The model side projected the
    3D point  C +- r*u_ver  (the point straight above the hub) and the projected hub.
    Neither is the top of the image ellipse, nor the ellipse centre (perspective moves
    the centre of the ellipse away from the projection of the hub), so the residuals
    never vanish at the true pose (~5 px at 6 m) and biased the estimated pose.

Fix
    Compute on the model side exactly what the data side measures:
      * P1/P3 = image-space top / bottom of the projected rim circle (closed form below)
      * centre = midpoint of P1 and P3 (an ellipse is centrally symmetric)

    Closed form. Rim: c + r(cos t A + sin t B) with A = R.x, B = R.z (wheel plane).
    In camera coordinates (c_c, A_c, B_c) the image row is v = fy Z(t)/Y(t) + cy and
        dv/dt = 0  <=>  alpha cos t + beta sin t = gamma
        alpha = y0 zb - z0 yb,  beta = z0 ya - y0 za,  gamma = r (za yb - zb ya)
    giving two solutions: the topmost and the bottommost image point.

Usage (same interface as bike_model_perspective, no sympy needed):
    import bike_model_ellipse as perpro
    fit_img2model(imgdata, x0, bike_params, boundaries, camera_params, perpro)
It uses the same 12 data keys as before (r_Q_S_* are ignored).
"""

import numpy as np

KEYS = ["r_Cf_Cr_u", "r_Cf_Cr_v", "u_Cr", "v_Cr",
        "r_P1r_Cr_u", "r_P1r_Cr_v", "r_P3r_Cr_u", "r_P3r_Cr_v",
        "r_P1f_Cf_u", "r_P1f_Cf_v", "r_P3f_Cf_u", "r_P3f_Cf_v"]


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


def _top_bottom(c, A, B, r, C, pc, cam):
    """Image (u, v) of the topmost and bottommost points of the projected rim."""
    c_c, A_c, B_c = C.T @ (c - pc), C.T @ A, C.T @ B
    y0, z0 = c_c[1], c_c[2]
    ya, za, yb, zb = A_c[1], A_c[2], B_c[1], B_c[2]
    alpha = y0 * zb - z0 * yb
    beta = z0 * ya - y0 * za
    gamma = r * (za * yb - zb * ya)
    k = np.hypot(alpha, beta)
    phase = np.arctan2(beta, alpha)
    d = np.arccos(np.clip(gamma / k, -1.0, 1.0))
    pts = [_project(c + r * (np.cos(t) * A + np.sin(t) * B), C, pc, cam)
           for t in (phase + d, phase - d)]
    return (pts[0], pts[1]) if pts[0][1] >= pts[1][1] else (pts[1], pts[0])


def model_features(x, bike_params, camera_params):
    """The 12 model quantities that correspond to KEYS."""
    phi, theta, psi, delta, xr, yr, zr = (float(v) for v in x[:7])
    bp, cam = bike_params, camera_params

    R = _rz(psi) @ _rx(phi) @ _ry(theta)        # same conventions as bike_model_perspective.py
    F = R @ _rz(delta)
    Cr = np.array([xr, yr, zr])
    S = Cr + bp["lr"] * R[:, 0]
    Q = S - bp["lf1"] * F[:, 2]
    Cf = Q + bp["lf2"] * F[:, 0]

    C = _rz(cam["cam_yaw"]) @ _ry(cam["cam_pitch"]) @ _rx(cam["cam_roll"])
    pc = np.array([cam["cam_x"], cam["cam_y"], cam["cam_z"]])

    cen, off = {}, {}
    for w, (c, M, rad) in {"r": (Cr, R, bp["rr"]), "f": (Cf, F, bp["rf"])}.items():
        top, bot = _top_bottom(c, M[:, 0], M[:, 2], rad, C, pc, cam)
        cen[w] = ((top[0] + bot[0]) / 2, (top[1] + bot[1]) / 2)
        off[w] = ((top[0] - bot[0]) / 2, (top[1] - bot[1]) / 2)

    return np.array([cen["f"][0] - cen["r"][0], cen["f"][1] - cen["r"][1],
                     cen["r"][0], cen["r"][1],
                     off["r"][0], off["r"][1], -off["r"][0], -off["r"][1],
                     off["f"][0], off["f"][1], -off["f"][0], -off["f"][1]])


def residual_eqs(x, data, bike_params, camera_params):
    """Same signature as bike_model_perspective.residual_eqs; 12 residuals in pixels."""
    d = np.array([data[k] for k in KEYS], float)
    return model_features(x, bike_params, camera_params) - d


def synthetic_data(x, bike_params, camera_params):
    """Noise-free data for a given state, in the format fit_img2model() expects."""
    m = model_features(x, bike_params, camera_params)
    return {k: [float(v)] for k, v in zip(KEYS, m)}
