# src/model_viewer.py
"""
Interactive visual debugger for the bicycle perspective model.

Left   : 3D view of the bike + camera frustum in the world frame N
         (x -> right, y -> depth / optical axis at zero yaw, z -> up).
Centre : 2D image plane (pixels, v pointing UP, same convention as the model and
         as fitEllipse) with the projected model and, on top of it, the real data
         (annotation points, fitted ellipses, the keypoints the solver is asked
         to match, and the residual lines between both).
Right  : the 14 residuals exactly as the solver sees them (model, data, residual,
         share of the cost).
Bottom : sliders for the 7 state variables (-> solver initial guess x0) and for
         the camera parameters, plus buttons.

The sliders of the state are the initial guess: "Export x0" prints a ready to
paste `x0 = np.array([...])` and writes `x0_guess.json`. "Solve" runs the same
least_squares call as fit_img2model() from the current slider state so you can
watch where it ends up (the starting pose is kept as a grey dashed ghost).

Usage (from src/):
    python model_viewer.py --label /path/to/DSC09360.txt          # real frame
    python model_viewer.py                                        # synthetic frame
    python model_viewer.py --label f.txt --res 1920 1080 --x0 0 0 20 0 1 6 0.3
    python model_viewer.py --screenshot out.png                   # headless render

Camera/bike defaults are the ones in main.py (single-frame branch).
"""

import argparse
import json
import os
import sys
import traceback
import warnings

if "--screenshot" in sys.argv:                     # headless rendering
    import matplotlib
    matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider
from scipy.optimize import least_squares

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Project modules (the model that the solver really uses + the data pipeline)
# ---------------------------------------------------------------------------
try:
    import bike_model_perspective as perpro
except Exception as exc:                            # pragma: no cover
    perpro = None
    print(f"[warn] could not import bike_model_perspective ({exc}); the 'as coded' "
          f"Solve will use the numpy copy of the model instead.")
try:
    from utils import readFile, fitEllipse, find_points
except Exception as exc:                            # pragma: no cover
    print(f"[error] could not import utils.py ({exc}). It is needed for "
          f"readFile / fitEllipse / find_points.")
    raise


# ---------------------------------------------------------------------------
# Defaults (copied from main.py)
# ---------------------------------------------------------------------------
SCREEN_RES = (1920, 1080)
SENSOR_SIZE = (0.0235, 0.0156)
FOCAL_LENGTH = 0.028
WHEEL_DIAMETER = 0.6604

BIKE_PARAMS = {"lr": 1.4, "lf1": 0.5, "lf2": 0.1, "lh": 0.0,      # lh: head height above rear hub (0 = as in the sympy model)
               "rf": WHEEL_DIAMETER / 2, "rr": WHEEL_DIAMETER / 2}
BP_KEYS = ("lr", "lf1", "lf2", "lh", "rr", "rf")

STATE_NAMES = ("phi", "theta", "psi", "delta", "x_r", "y_r", "z_r")
X0_MAIN = np.array([0.0, 0.0, 0.0, 0.0, 2.82, 0.1, 0.0])       # main.py, frame < 400

FEATURE_NAMES = ["r_Cf_Cr_u", "r_Cf_Cr_v", "u_Cr", "v_Cr",
                 "r_P1r_Cr_u", "r_P1r_Cr_v", "r_P3r_Cr_u", "r_P3r_Cr_v",
                 "r_P1f_Cf_u", "r_P1f_Cf_v", "r_P3f_Cf_u", "r_P3f_Cf_v",
                 "r_Q_S_u", "r_Q_S_v"]

EPS_DEPTH = 1e-3


def default_camera(res):
    return {"fx": FOCAL_LENGTH * res[0] / SENSOR_SIZE[0],
            "fy": FOCAL_LENGTH * res[1] / SENSOR_SIZE[1],
            "cx": res[0] / 2.0, "cy": res[1] / 2.0,
            "cam_x": 0.0, "cam_y": 0.0, "cam_z": 0.45,
            "cam_yaw": 0.0, "cam_pitch": 0.0, "cam_roll": 0.0}


def make_bounds(psi_deg=180.0, roll_deg=90.0, steer_deg=90.0):
    """main.py uses psi +-90 (the bike can then only travel to the RIGHT of the image)."""
    p, r, d = np.deg2rad(psi_deg), np.deg2rad(roll_deg), np.deg2rad(steer_deg)
    lb = np.array([-r, -np.pi / 2, -p, -d, -20.0, 0.1, -20.0])
    ub = np.array([r, np.pi / 2, p, d, 20.0, 20.0, 20.0])
    return lb, ub


# ---------------------------------------------------------------------------
# Numpy copy of the model (verified against the lambdified sympy model, see
# verify_against_sympy()).  Fast enough to be evaluated on every slider move.
# ---------------------------------------------------------------------------
def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def bike_frames(phi, theta, psi, delta):
    """Columns are the unit vectors of R and F expressed in N."""
    R = rot_z(psi) @ rot_x(phi) @ rot_y(theta)       # orient_body_fixed(N,(psi,phi,theta),'ZXY')
    F = R @ rot_z(delta)                              # orient_axis(R, delta, R.z)
    return R, F


def camera_matrix(cam):
    """Columns are C.x (right), C.y (optical axis), C.z (up) expressed in N."""
    return rot_z(cam["cam_yaw"]) @ rot_y(cam["cam_pitch"]) @ rot_x(cam["cam_roll"])


def camera_position(cam):
    return np.array([cam["cam_x"], cam["cam_y"], cam["cam_z"]])


def _plane_axes(M):
    """u_hor, u_ver exactly as in bike_model_perspective.py (NaN when singular)."""
    y = M[:, 1]
    w = np.cross(y, np.cross(y, np.array([1.0, 0.0, 0.0])))
    n = np.linalg.norm(w)
    if n < 1e-9:
        nan = np.full(3, np.nan)
        return nan, nan
    u_hor = w / n
    return u_hor, np.cross(y, u_hor)


def bike_points(x, bp):
    phi, theta, psi, delta, xr, yr, zr = (float(v) for v in x[:7])
    R, F = bike_frames(phi, theta, psi, delta)
    Cr = np.array([xr, yr, zr])
    S = Cr + bp["lr"] * R[:, 0] + bp.get("lh", 0.0) * R[:, 2]
    Q = S - bp["lf1"] * F[:, 2]
    Cf = Q + bp["lf2"] * F[:, 0]
    hr, vr = _plane_axes(R)
    hf, vf = _plane_axes(F)
    pts = {"Cr": Cr, "S": S, "Q": Q, "Cf": Cf,
           "P1r": Cr + bp["rr"] * vr, "P2r": Cr + bp["rr"] * hr,
           "P3r": Cr - bp["rr"] * vr, "P4r": Cr - bp["rr"] * hr,
           "P1f": Cf + bp["rf"] * vf, "P2f": Cf + bp["rf"] * hf,
           "P3f": Cf - bp["rf"] * vf, "P4f": Cf - bp["rf"] * hf}
    return pts, R, F


def rim_circle(center, ax_x, ax_z, radius, n=120):
    t = np.linspace(0, 2 * np.pi, n)
    return center[None, :] + radius * (np.cos(t)[:, None] * ax_x[None, :]
                                       + np.sin(t)[:, None] * ax_z[None, :])


def project(P, cam, mask=False):
    """Pinhole projection used by the model: u = fx Xc/Yc + cx, v = fy Zc/Yc + cy."""
    P = np.atleast_2d(np.asarray(P, float))
    rc = (P - camera_position(cam)) @ camera_matrix(cam)
    Xc, Yc, Zc = rc[:, 0], rc[:, 1], rc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = cam["fx"] * Xc / Yc + cam["cx"]
        v = cam["fy"] * Zc / Yc + cam["cy"]
    if mask:
        bad = Yc <= EPS_DEPTH
        u = np.where(bad, np.nan, u)
        v = np.where(bad, np.nan, v)
    return u, v, Yc


def wheel_image_extremes(c, A, B, r, cam, axis="v"):
    """
    Exact image-space top / bottom points of the projected rim circle
    c + r(cos t A + sin t B). d/dt of v(t)=fy Z/Y+cy reduces to
    alpha cos t + beta sin t = gamma, so it is closed form.
    Returns ((u,v)_max, (u,v)_min) along v (top, bottom) or u (right, left), NaN if the circle cannot be handled.
    """
    C = camera_matrix(cam)
    c_c = C.T @ (c - camera_position(cam))
    A_c, B_c = C.T @ A, C.T @ B
    kc = 2 if axis == "v" else 0                    # camera component that becomes v (top) / u (right)
    y0, z0 = c_c[1], c_c[kc]
    ya, za, yb, zb = A_c[1], A_c[kc], B_c[1], B_c[kc]
    al = y0 * zb - z0 * yb
    be = z0 * ya - y0 * za
    ga = r * (za * yb - zb * ya)
    k = np.hypot(al, be)
    if not np.isfinite(k) or k < 1e-12:
        return (np.nan, np.nan), (np.nan, np.nan)
    ph = np.arctan2(be, al)
    d = np.arccos(np.clip(ga / k, -1, 1))
    out = []
    for t in (ph + d, ph - d):
        p = c + r * (np.cos(t) * A + np.sin(t) * B)
        u, v, _ = project(p, cam)
        out.append((float(u[0]), float(v[0])))
    i = 1 if axis == "v" else 0
    return (out[0], out[1]) if out[0][i] >= out[1][i] else (out[1], out[0])


def model_features(x, bp, cam, p1_mode="as coded"):
    """
    The 14 model quantities of residual_eqs() (before subtracting the data).
    p1_mode = 'as coded'       -> P1/P3 = Cr +- rr*u_ver (what bike_model_perspective.py does)
    p1_mode = 'image extreme'  -> P1/P3 = top/bottom of the projected rim, centre = their
                                  midpoint (the same definition fitEllipse+find_points give)
    """
    pts, R, F = bike_points(x, bp)
    uv = {}
    for k, p in pts.items():
        u, v, _ = project(p, cam)
        uv[k] = (float(u[0]), float(v[0]))
    qs = (uv["Q"][0] - uv["S"][0], uv["Q"][1] - uv["S"][1])

    if p1_mode == "extremes + width":
        o = {}
        for w, (c, M, rad) in {"r": (pts["Cr"], R, bp["rr"]), "f": (pts["Cf"], F, bp["rf"])}.items():
            top, bot = wheel_image_extremes(c, M[:, 0], M[:, 2], rad, cam, "v")
            rgt, lft = wheel_image_extremes(c, M[:, 0], M[:, 2], rad, cam, "u")
            o[w] = (((top[0] + bot[0]) / 2, (top[1] + bot[1]) / 2),
                    ((top[0] - bot[0]) / 2, (top[1] - bot[1]) / 2),
                    ((rgt[0] - lft[0]) / 2, (rgt[1] - lft[1]) / 2))
        (cr, tr, rr_), (cf, tf, rf_) = o["r"], o["f"]
        return np.array([cf[0] - cr[0], cf[1] - cr[1], cr[0], cr[1], *tr, *rr_, *tf, *rf_], float)
    if p1_mode == "as coded":
        f = [uv["Cf"][0] - uv["Cr"][0], uv["Cf"][1] - uv["Cr"][1], uv["Cr"][0], uv["Cr"][1]]
        for w in ("r", "f"):
            c = "C" + w
            for p in ("P1", "P3"):
                f += [uv[p + w][0] - uv[c][0], uv[p + w][1] - uv[c][1]]
    else:
        cen, off = {}, {}
        for w, (c, A, B, r) in {"r": (pts["Cr"], R[:, 0], R[:, 2], bp["rr"]),
                                "f": (pts["Cf"], F[:, 0], F[:, 2], bp["rf"])}.items():
            top, bot = wheel_image_extremes(c, A, B, r, cam)
            cen[w] = ((top[0] + bot[0]) / 2, (top[1] + bot[1]) / 2)
            off[w] = ((top[0] - bot[0]) / 2, (top[1] - bot[1]) / 2)
        f = [cen["f"][0] - cen["r"][0], cen["f"][1] - cen["r"][1], cen["r"][0], cen["r"][1],
             off["r"][0], off["r"][1], -off["r"][0], -off["r"][1],
             off["f"][0], off["f"][1], -off["f"][0], -off["f"][1]]
    return np.array(f + [qs[0], qs[1]], float)


FULL_NAMES = ["r_Cf_Cr_u", "r_Cf_Cr_v", "u_Cr", "v_Cr", "r_P1r_Cr_u", "r_P1r_Cr_v", "r_P2r_Cr_u", "r_P2r_Cr_v",
              "r_P1f_Cf_u", "r_P1f_Cf_v", "r_P2f_Cf_u", "r_P2f_Cf_v"]
P1_MODES = ["as coded", "image extreme", "extremes + width"]


def residual_vector(x, data, bp, cam, qs_mode="as coded", p1_mode="as coded"):
    m = model_features(x, bp, cam, p1_mode)
    if p1_mode == "extremes + width":          # 12 equations, no Q-S
        return m - data["vec_full"]
    data_vec = data["vec"]
    if qs_mode == "normalised":
        n = np.hypot(m[12], m[13])
        m = m.copy()
        if n > 0:
            m[12:] = m[12:] / n
    r = m - data_vec
    return r[:12] if qs_mode == "off" else r


# ---------------------------------------------------------------------------
# Data: annotation file -> ellipses -> the 14 numbers of the solver
# ---------------------------------------------------------------------------
def qs_from_ellipses(ang_f, ang_r, th_f, th_r):
    """r_Q_S_u / r_Q_S_v formula copied from main.py / data4model()."""
    with np.errstate(all="ignore"):
        A = np.sin(ang_f) * np.cos(ang_r) - np.sin(ang_r) * np.cos(ang_f) * np.cos(th_f - th_r)
        B = np.sin(ang_r) * np.sin(th_f - th_r)
        D = np.sqrt(A ** 2 + B ** 2)
        u = (A * np.sin(th_f) + B * np.cos(ang_f) * np.cos(th_f)) / D
        v = (A * np.cos(th_f) - B * np.cos(ang_f) * np.sin(th_f)) / D
    return float(u), float(v)


def ellipse_keypoints(e):
    """centre, (top - centre), (right-most - centre) of the ellipse (xc, zc, a, b, theta)."""
    xc, zc, a, b, th = (float(v) for v in e[:5])
    s_, c_ = np.sin(th), np.cos(th)
    dz = np.sqrt(a * a * s_ * s_ + b * b * c_ * c_)
    dx = np.sqrt(a * a * c_ * c_ + b * b * s_ * s_)
    k = (a * a - b * b) * s_ * c_
    return (xc, zc), (k / dz, dz), (dx, k / dx)


def build_data(front_n, rear_n, res):
    """front_n / rear_n: (N,2) normalised annotation points (YOLO convention, y down)."""
    ef = fitEllipse(front_n, res)
    er = fitEllipse(rear_n, res)
    if ef is None or er is None:
        raise RuntimeError("fitEllipse failed on the annotation points")
    xf, zf, af, bf, tf = (float(v) for v in ef)
    xr, zr, ar, br, tr = (float(v) for v in er)
    p1f, _, p3f, _ = find_points(ef)
    p1r, _, p3r, _ = find_points(er)
    ang_f = np.arccos(bf / af) if af > bf else 0.0
    ang_r = np.arccos(br / ar) if ar > br else 0.0
    qu, qv = qs_from_ellipses(ang_f, ang_r, tf, tr)
    img = {"r_Cf_Cr_u": xf - xr, "r_Cf_Cr_v": zf - zr, "u_Cr": xr, "v_Cr": zr,
           "r_P1r_Cr_u": p1r[0] - xr, "r_P1r_Cr_v": p1r[1] - zr,
           "r_P3r_Cr_u": p3r[0] - xr, "r_P3r_Cr_v": p3r[1] - zr,
           "r_P1f_Cf_u": p1f[0] - xf, "r_P1f_Cf_v": p1f[1] - zf,
           "r_P3f_Cf_u": p3f[0] - xf, "r_P3f_Cf_v": p3f[1] - zf,
           "r_Q_S_u": qu, "r_Q_S_v": qv}
    img = {k: float(v) for k, v in img.items()}
    px = lambda p: np.column_stack([p[:, 0] * res[0], (1 - p[:, 1]) * res[1]])
    (cf_, tf_, rf_), (cr_, tr_, rr_) = ellipse_keypoints(ef), ellipse_keypoints(er)
    vec_full = np.array([cf_[0] - cr_[0], cf_[1] - cr_[1], cr_[0], cr_[1], *tr_, *rr_, *tf_, *rf_])
    return {"img": img, "vec": np.array([img[k] for k in FEATURE_NAMES]), "vec_full": vec_full,
            "rgt_f": (xf + rf_[0], zf + rf_[1]), "lft_f": (xf - rf_[0], zf - rf_[1]),
            "rgt_r": (xr + rr_[0], zr + rr_[1]), "lft_r": (xr - rr_[0], zr - rr_[1]),
            "ef": (xf, zf, af, bf, tf), "er": (xr, zr, ar, br, tr),
            "p1f": p1f, "p3f": p3f, "p1r": p1r, "p3r": p3r,
            "raw_f": px(front_n), "raw_r": px(rear_n)}


def synthetic_points(x, bp, cam, res, n=48, noise=0.5, seed=0):
    rng = np.random.default_rng(seed)
    pts, R, F = bike_points(x, bp)
    out = {}
    for name, (c, A, B, r) in {"front": (pts["Cf"], F[:, 0], F[:, 2], bp["rf"]),
                               "rear": (pts["Cr"], R[:, 0], R[:, 2], bp["rr"])}.items():
        u, v, _ = project(rim_circle(c, A, B, r, n), cam)
        u = u + noise * rng.standard_normal(n)
        v = v + noise * rng.standard_normal(n)
        out[name] = np.column_stack([u / res[0], 1 - v / res[1]])
    return out["front"], out["rear"]


def load_label(path):
    ell = readFile(path)
    by_class = {e["class_id"]: e["points"] for e in ell}
    if 0 in by_class and 1 in by_class:                 # class 0 = front, 1 = rear
        return by_class[0], by_class[1]
    print("[warn] class ids are not {0,1}: using file order (first line = front, second = rear)")
    return ell[0]["points"], ell[1]["points"]


def auto_guess(data, bp, cam):
    """
    Closed-form rough guess from the two ellipses (pinhole, camera ~ looking along +y):
      depth from the apparent size of the rear wheel, lateral/vertical from its centre,
      yaw from the aspect ratio b/a (|cos psi| = b/a) and the travel direction.
    """
    xr, zr, ar, br, _ = data["er"]
    xf, zf, af, bf, _ = data["ef"]
    Y = cam["fx"] * bp["rr"] / max(ar, 1e-6)
    x = (xr - cam["cx"]) * Y / cam["fx"] + cam["cam_x"]
    z = (zr - cam["cy"]) * Y / cam["fy"] + cam["cam_z"]
    y = Y + cam["cam_y"]
    alpha = np.mean([np.arccos(np.clip(bf / af, 0, 1)), np.arccos(np.clip(br / ar, 0, 1))])
    front_farther = af < ar                       # smaller front ellipse -> front is farther
    if xf >= xr:                                  # travelling to the right
        psi = alpha if front_farther else -alpha
    else:                                         # travelling to the left
        psi = (np.pi - alpha) if front_farther else -(np.pi - alpha)
    return np.array([0.0, 0.0, psi, 0.0, x, y, z])


# ---------------------------------------------------------------------------
# Sanity check: numpy copy == lambdified sympy model that the solver uses
# ---------------------------------------------------------------------------
def verify_against_sympy(bp, cam, n=6, seed=1):
    if perpro is None:
        return None
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(n):
        x = np.array([rng.uniform(-.4, .4), rng.uniform(-.4, .4), rng.uniform(-1.4, 1.4),
                      rng.uniform(-.4, .4), rng.uniform(-2, 2), rng.uniform(3, 9), rng.uniform(0, .7)])
        zero = {k: 0.0 for k in FEATURE_NAMES}
        r_sym = np.asarray(perpro.residual_eqs(x, zero, bp, cam))   # = model features
        r_np = model_features(x, bp, cam, "as coded")[:len(r_sym)]
        worst = max(worst, float(np.nanmax(np.abs(r_sym - r_np))))
    return worst


PERPRO_NEQ = 14          # number of residuals returned by perpro.residual_eqs (set in main)


def perpro_n_eqs(bp, cam):
    """12 if you removed the Q-S equations from bike_model_perspective.py, else 14."""
    if perpro is None:
        return 14
    zero = {k: 0.0 for k in FEATURE_NAMES}
    return len(perpro.residual_eqs(np.array([0, 0, 0.3, 0, 0, 5.0, 0.3]), zero, bp, cam))


def solver_matches_perpro(qs_mode, p1_mode, bp=None):
    """True when the selected formulation is exactly what perpro.residual_eqs computes."""
    if perpro is None or p1_mode != "as coded" or (bp is not None and bp.get("lh", 0.0) != 0.0):
        return False
    return qs_mode == ("off" if PERPRO_NEQ == 12 else "as coded")


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
class BikeLayer:
    """Artists for one bike pose, in the 3D axes and in the image axes."""

    def __init__(self, ax3, ax2, ghost=False):
        self.ghost = ghost
        if ghost:
            kw = lambda c: dict(color="0.55", ls="--", lw=1.0, alpha=0.9)
        else:
            kw = lambda c: dict(color=c, lw=2.0)
        self.parts = []
        e3 = lambda *a, **k: ax3.plot([], [], [], *a, **k)[0]
        e2 = lambda *a, **k: ax2.plot([], [], *a, **k)[0]
        mk = dict(marker="o", ms=3.5) if not ghost else dict(marker="o", ms=2)

        label = (lambda s: s) if not ghost else (lambda s: "_nolegend_")
        # 3D
        self.f3 = [e3(label=label("rear frame Cr-S"), **kw("C0"), **mk),
                   e3(label=label("steering axis S-Q"), **kw("C2"), **mk),
                   e3(label=label("fork Q-Cf"), **kw("C1"), **mk)]
        self.rim3 = [e3(label=label("rear rim"), **(kw("C0") | {"lw": 1.2})),
                     e3(label=label("front rim"), **(kw("C1") | {"lw": 1.2}))]
        # 2D
        self.f2 = [e2(**kw("C0"), **mk), e2(**kw("C2"), **mk), e2(**kw("C1"), **mk)]
        self.rim2 = [e2(label=label("model rim (rear)"), **(kw("C0") | {"lw": 1.4})),
                     e2(label=label("model rim (front)"), **(kw("C1") | {"lw": 1.4}))]
        if not ghost:
            ms = dict(ls="", ms=7, mfc="none", mew=1.5)
            self.p13_3 = [e3(marker="o", color="red", **ms), e3(marker="o", color="k", **ms)]
            self.p13_2 = [e2(marker="o", color="red", label="model P1 (as coded)", **ms),
                          e2(marker="o", color="k", label="model P3 (as coded)", **ms)]
            self.ext_2 = e2(ls="", marker="D", ms=6, mfc="none", mec="#009E73", mew=1.4,
                            label="model: image extremes of rim (top/bottom/left/right)")
            self.hub_2 = e2(ls="", marker="+", ms=11, mew=1.8, color="#7A00B8",
                            label="model hubs Cr, Cf")
            self.qs_model = e2(color="#D55E00", lw=2.2, label="Q-S model (normalised, 60 px)")

    def set(self, sc, cam):
        pts = sc["pts"]
        segs = [("Cr", "S"), ("S", "Q"), ("Q", "Cf")]
        for ln3, ln2, (a, b) in zip(self.f3, self.f2, segs):
            ln3.set_data_3d([pts[a][0], pts[b][0]], [pts[a][1], pts[b][1]], [pts[a][2], pts[b][2]])
            ln2.set_data([sc["uv"][a][0], sc["uv"][b][0]], [sc["uv"][a][1], sc["uv"][b][1]])
        for ln3, ln2, key in zip(self.rim3, self.rim2, ("rim_r", "rim_f")):
            P = sc[key]
            ln3.set_data_3d(P[:, 0], P[:, 1], P[:, 2])
            ln2.set_data(sc[key + "_uv"][0], sc[key + "_uv"][1])
        if not self.ghost:
            for ln, k in zip(self.p13_3, (("P1r", "P1f"), ("P3r", "P3f"))):
                ln.set_data_3d([pts[k[0]][0], pts[k[1]][0]], [pts[k[0]][1], pts[k[1]][1]],
                               [pts[k[0]][2], pts[k[1]][2]])
            for ln, k in zip(self.p13_2, (("P1r", "P1f"), ("P3r", "P3f"))):
                ln.set_data([sc["uv"][k[0]][0], sc["uv"][k[1]][0]],
                            [sc["uv"][k[0]][1], sc["uv"][k[1]][1]])
            e = sc["ext"]
            self.ext_2.set_data([p[0] for w in e for p in w], [p[1] for w in e for p in w])
            self.hub_2.set_data([sc["uv"]["Cr"][0], sc["uv"]["Cf"][0]],
                                [sc["uv"]["Cr"][1], sc["uv"]["Cf"][1]])
            q = np.array(sc["uv"]["Q"]) - np.array(sc["uv"]["S"])
            n = np.hypot(*q)
            if np.isfinite(n) and n > 0:
                q = 60 * q / n
                s = sc["uv"]["S"]
                self.qs_model.set_data([s[0], s[0] + q[0]], [s[1], s[1] + q[1]])
            else:
                self.qs_model.set_data([], [])

    def clear(self):
        for a in self.f3 + self.rim3 + self.f2 + self.rim2:
            a.set_data([], [])
            if hasattr(a, "set_3d_properties"):
                a.set_3d_properties([])


def compute_scene(x, bp, cam):
    pts, R, F = bike_points(x, bp)
    sc = {"pts": pts, "R": R, "F": F}
    sc["uv"] = {}
    for k, p in pts.items():
        u, v, _ = project(p, cam, mask=True)
        sc["uv"][k] = (float(u[0]), float(v[0]))
    sc["rim_r"] = rim_circle(pts["Cr"], R[:, 0], R[:, 2], bp["rr"])
    sc["rim_f"] = rim_circle(pts["Cf"], F[:, 0], F[:, 2], bp["rf"])
    for k in ("rim_r", "rim_f"):
        u, v, _ = project(sc[k], cam, mask=True)
        sc[k + "_uv"] = (u, v)
    sc["ext"] = (wheel_image_extremes(pts["Cr"], R[:, 0], R[:, 2], bp["rr"], cam),
                 wheel_image_extremes(pts["Cf"], F[:, 0], F[:, 2], bp["rf"], cam),
                 wheel_image_extremes(pts["Cr"], R[:, 0], R[:, 2], bp["rr"], cam, "u"),
                 wheel_image_extremes(pts["Cf"], F[:, 0], F[:, 2], bp["rf"], cam, "u"))
    return sc


# ---------------------------------------------------------------------------
# The viewer
# ---------------------------------------------------------------------------
STATE_SLIDERS = [  # key, label, min, max, step   (angles in deg)
    ("phi", "phi   roll [deg]", -90, 90, 0.1),
    ("theta", "theta pitch [deg]", -90, 90, 0.1),
    ("psi", "psi   yaw [deg]", -180, 180, 0.1),
    ("delta", "delta steer [deg]", -90, 90, 0.1),
    ("x_r", "x_r [m]", -15, 15, 0.01),
    ("y_r", "y_r depth [m]", -2, 30, 0.01),
    ("z_r", "z_r [m]", -3, 3, 0.01),
]


class Viewer:
    def __init__(self, data, bp, cam0, state0, res, bounds, out_path,
                 true_state=None, qs_mode="as coded", p1_mode="as coded"):
        self.data, self.bp, self.cam0, self.res = data, dict(bp), dict(cam0), res
        self.bounds, self.out_path, self.true_state = bounds, out_path, true_state
        self.qs_mode, self.p1_mode = qs_mode, p1_mode
        self.suspend = False
        self.ghost_state = None
        self.note = ""
        self.show_data, self.show_lines, self.fit_bike = True, True, False

        W, H = res
        self.fig = plt.figure(figsize=(19, 10.5))
        self.fig.canvas.manager.set_window_title("Bike model viewer") if self.fig.canvas.manager else None
        self.ax3 = self.fig.add_axes([0.005, 0.385, 0.355, 0.60], projection="3d")
        self.ax2 = self.fig.add_axes([0.375, 0.385, 0.405, 0.58])
        self.axt = self.fig.add_axes([0.785, 0.335, 0.21, 0.65])
        self.axt.axis("off")

        # ---- 2D axes
        ax2 = self.ax2
        ax2.set_aspect("equal")
        ax2.set_xlim(-0.08 * W, 1.08 * W)
        ax2.set_ylim(-0.12 * H, 1.12 * H)
        ax2.add_patch(Rectangle((0, 0), W, H, fill=False, ec="0.3", lw=1.2))
        ax2.plot(self.cam0["cx"], self.cam0["cy"], "+", color="0.6", ms=8)
        ax2.set_xlabel("u [px]  (right)", fontsize=8)
        ax2.set_ylabel("v [px]  (up, origin bottom-left, as fitEllipse)", fontsize=8)
        ax2.tick_params(labelsize=7)
        ax2.grid(alpha=0.25)
        ax2.set_title("Image plane: model (lines) vs data (annotation points / ellipses / keypoints)",
                      fontsize=9)

        # ---- static data artists
        d = data
        self.data_artists = []
        for raw, col, lab in ((d["raw_r"], "C0", "data: rear annotation"),
                              (d["raw_f"], "C1", "data: front annotation")):
            self.data_artists.append(ax2.scatter(raw[:, 0], raw[:, 1], s=14, facecolors="none",
                                                 edgecolors=col, lw=0.9, label=lab, zorder=3))
        for e, col in ((d["er"], "C0"), (d["ef"], "C1")):
            xc, zc, a, b, th = e
            patch = Ellipse((xc, zc), 2 * a, 2 * b, angle=np.rad2deg(th), fill=False,
                            ec=col, ls=":", lw=1.3, zorder=3)
            ax2.add_patch(patch)
            self.data_artists.append(patch)
        kp = dict(ls="", zorder=5)
        self.data_artists += [
            ax2.plot([d["er"][0], d["ef"][0]], [d["er"][1], d["ef"][1]], marker="s", ms=7, mfc="none",
                     mec="k", mew=1.5, label="data: ellipse centres", **kp)[0],
            ax2.plot([d["p1r"][0], d["p1f"][0]], [d["p1r"][1], d["p1f"][1]], marker="x", ms=8, mew=2,
                     color="red", label="data: P1 (ellipse top)", **kp)[0],
            ax2.plot([d["p3r"][0], d["p3f"][0]], [d["p3r"][1], d["p3f"][1]], marker="x", ms=8, mew=2,
                     color="k", label="data: P3 (ellipse bottom)", **kp)[0],
        ]
        self.data_artists.append(ax2.plot(
            [d["rgt_r"][0], d["rgt_f"][0]], [d["rgt_r"][1], d["rgt_f"][1]], marker=">", ms=7, mfc="#E69F00",
            mec="k", label="data: ellipse right extreme", **kp)[0])
        self.data_artists.append(ax2.plot(
            [d["lft_r"][0], d["lft_f"][0]], [d["lft_r"][1], d["lft_f"][1]], marker="<", ms=7, mfc="#E69F00",
            mec="k", label="data: ellipse left extreme", **kp)[0])
        # data Q-S direction (unit vector, drawn 60 px long from the model S)
        self.qs_data, = ax2.plot([], [], color="k", lw=2.2, label="Q-S data (unit vector, 60 px)", zorder=6)
        self.data_artists.append(self.qs_data)
        # residual lines
        self.res_lines = [ax2.plot([], [], color="0.35", ls=(0, (1, 2)), lw=1.0, zorder=2)[0]
                          for _ in range(6)]

        # ---- layers
        self.main = BikeLayer(self.ax3, ax2)
        self.ghost = BikeLayer(self.ax3, ax2, ghost=True)

        # ---- 3D camera artists
        a3 = self.ax3
        self.cam_pt, = a3.plot([], [], [], "s", color="k", ms=7, label="camera")
        self.cam_axes = [a3.plot([], [], [], color=c, lw=2)[0] for c in ("#D55E00", "#009E73", "#0072B2")]
        self.frustum = [a3.plot([], [], [], color="0.45", lw=0.9)[0] for _ in range(8)]
        a3.set_xlabel("x (right)", fontsize=8)
        a3.set_ylabel("y (depth)", fontsize=8)
        a3.set_zlabel("z (up)", fontsize=8)
        a3.tick_params(labelsize=7)
        a3.view_init(elev=24, azim=-62)
        self.fig.text(0.18, 0.975, "World frame N  (camera axes: Xc orange, Yc optical axis green, Zc blue)", fontsize=9, ha="center")

        self._build_widgets(state0)

        # legends
        h, l = ax2.get_legend_handles_labels()
        extra = [self.main.p13_2[0], self.main.p13_2[1], self.main.ext_2, self.main.hub_2, self.main.qs_model]
        keep = [(x, y) for x, y in zip(h, l) if not y.startswith("model P") and "image extremes" not in y
                and "hubs" not in y and not y.startswith("Q-S model")]
        keep += [(x, x.get_label()) for x in extra]
        ax2.legend([k[0] for k in keep], [k[1] for k in keep], fontsize=6.5, loc="upper left",
                   ncol=2, framealpha=0.9)
        self.ax3.legend([self.main.f3[0], self.main.f3[1], self.main.f3[2], self.main.rim3[0],
                         self.main.rim3[1], self.cam_pt],
                        ["rear frame Cr-S", "steering axis S-Q", "fork Q-Cf", "rear rim", "front rim",
                         "camera"], fontsize=6.5, loc="upper left", framealpha=0.85)

        self.update()
        self.fig.canvas.mpl_connect("close_event", lambda e: self.export_x0(quiet_file=True))

    # ------------------------------------------------------------------ widgets
    def _build_widgets(self, state0):
        fig = self.fig
        ROW = 0.0375
        self.sliders = {}
        c = self.cam0
        W, H = self.res

        def add_slider(x, y, w, key, label, vmin, vmax, init, step):
            ax = fig.add_axes([x, y, w, 0.0215])
            init = float(np.clip(init, vmin, vmax))
            s = Slider(ax, label, vmin, vmax, valinit=init, valstep=step, color="#6FA8DC")
            s.label.set_fontsize(7.5)
            s.valtext.set_fontsize(7.5)
            s.on_changed(self.update)
            self.sliders[key] = s

        fig.text(0.012, 0.358, "BIKE STATE  (= solver initial guess x0)", fontsize=9, weight="bold")
        s0 = np.asarray(state0, float)
        init = np.r_[np.rad2deg(s0[:4]), s0[4:]]
        for i, (k, lab, lo, hi, st) in enumerate(STATE_SLIDERS):
            add_slider(0.095, 0.325 - i * ROW, 0.15, k, lab, lo, hi, init[i], st)

        fig.text(0.285, 0.358, "CAMERA  intrinsics", fontsize=9, weight="bold")
        intr = [("fx", "fx [px]", c["fx"] * 0.25, c["fx"] * 3, 1.0), ("fy", "fy [px]", c["fy"] * 0.25, c["fy"] * 3, 1.0),
                ("cx", "cx [px]", 0, 2 * c["cx"], 1.0), ("cy", "cy [px]", 0, 2 * c["cy"], 1.0)]
        for i, (k, lab, lo, hi, st) in enumerate(intr):
            add_slider(0.335, 0.325 - i * ROW, 0.13, k, lab, lo, hi, c[k], st)
        fig.text(0.285, 0.1745, "BIKE GEOMETRY [m]  (hub-to-hub = lr + lf2)", fontsize=9, weight="bold")
        geo = [("lr", "lr  rear hub -> head", 0.3, 2.0, 0.005), ("lf1", "lf1 head -> fork bottom", -0.2, 1.0, 0.005),
               ("lf2", "lf2 fork offset", -0.2, 0.5, 0.005), ("lh", "lh head height > hub", 0.0, 1.0, 0.005)]
        for i, (k, lab, lo, hi, st) in enumerate(geo):
            add_slider(0.335, 0.325 - (5 + i) * ROW, 0.13, k, lab, lo, hi, self.bp[k], st)
        fig.text(0.50, 0.1015, "WHEELS [m]", fontsize=9, weight="bold")
        for i, (k, lab) in enumerate((("rr", "rr rear radius"), ("rf", "rf front radius"))):
            add_slider(0.555, 0.325 - (7 + i) * ROW, 0.13, k, lab, 0.2, 0.5, self.bp[k], 0.001)

        fig.text(0.50, 0.358, "CAMERA  extrinsics (pose in N)", fontsize=9, weight="bold")
        ext = [("cam_x", "cam_x [m]", -5, 5, 0.01), ("cam_y", "cam_y [m]", -5, 5, 0.01),
               ("cam_z", "cam_z [m]", -2, 3, 0.01),
               ("cam_yaw", "yaw [deg]", -180, 180, 0.1), ("cam_pitch", "pitch [deg]", -60, 60, 0.1),
               ("cam_roll", "roll [deg]", -45, 45, 0.1)]
        for i, (k, lab, lo, hi, st) in enumerate(ext):
            v = c[k] if k in ("cam_x", "cam_y", "cam_z") else np.rad2deg(c[k])
            add_slider(0.555, 0.325 - i * ROW, 0.13, k, lab, lo, hi, v, st)

        # buttons ----------------------------------------------------------
        def add_button(rect, text, cb, color="#E8EEF7"):
            b = Button(fig.add_axes(rect), text, color=color, hovercolor="#CFE0F5")
            b.label.set_fontsize(8)
            b.on_clicked(cb)
            return b

        self.buttons = [
            add_button([0.725, 0.300, 0.125, 0.032], "Export x0  (print + json)", self.export_x0, "#D9EAD3"),
            add_button([0.855, 0.300, 0.125, 0.032], "Solve from sliders", self.solve, "#FCE5CD"),
            add_button([0.725, 0.262, 0.125, 0.032], "Auto-guess from data", self.on_autoguess),
            add_button([0.855, 0.262, 0.125, 0.032], "main.py's x0", self.on_main_x0),
            add_button([0.725, 0.224, 0.125, 0.032], "Reset camera", self.on_reset_cam),
            add_button([0.855, 0.224, 0.125, 0.032], "Clear ghost", self.on_clear_ghost),
        ]
        ax = fig.add_axes([0.725, 0.085, 0.125, 0.105])
        ax.set_title("Q-S eqs f13/f14 (table + Solve)", fontsize=7, pad=2)
        self.radio_qs = RadioButtons(ax, ["as coded", "normalised", "off"],
                                     active=["as coded", "normalised", "off"].index(self.qs_mode))
        ax = fig.add_axes([0.855, 0.085, 0.125, 0.105])
        ax.set_title("residual set (table + Solve)", fontsize=7, pad=2)
        self.radio_p1 = RadioButtons(ax, P1_MODES, active=P1_MODES.index(self.p1_mode))
        for r in (self.radio_qs, self.radio_p1):
            for t in r.labels:
                t.set_fontsize(7.5)
        self.radio_qs.on_clicked(lambda v: self._set_mode("qs", v))
        self.radio_p1.on_clicked(lambda v: self._set_mode("p1", v))

        ax = fig.add_axes([0.725, 0.012, 0.255, 0.065])
        self.checks = CheckButtons(ax, ["show data", "residual lines", "3D: fit to bike only"],
                                   [True, True, False])
        for t in self.checks.labels:
            t.set_fontsize(7.5)
        self.checks.on_clicked(self._on_checks)

    # ------------------------------------------------------------------ getters / setters
    def get_state(self):
        s = self.sliders
        return np.array([np.deg2rad(s["phi"].val), np.deg2rad(s["theta"].val),
                         np.deg2rad(s["psi"].val), np.deg2rad(s["delta"].val),
                         s["x_r"].val, s["y_r"].val, s["z_r"].val])

    def get_cam(self):
        s = self.sliders
        return {"fx": s["fx"].val, "fy": s["fy"].val, "cx": s["cx"].val, "cy": s["cy"].val,
                "cam_x": s["cam_x"].val, "cam_y": s["cam_y"].val, "cam_z": s["cam_z"].val,
                "cam_yaw": np.deg2rad(s["cam_yaw"].val), "cam_pitch": np.deg2rad(s["cam_pitch"].val),
                "cam_roll": np.deg2rad(s["cam_roll"].val)}

    def _set_slider(self, key, val):
        s = self.sliders[key]
        s.set_val(float(np.clip(val, s.valmin, s.valmax)))

    def set_state(self, x):
        self.suspend = True
        vals = np.r_[np.rad2deg(x[:4]), x[4:]]
        for (k, *_), v in zip(STATE_SLIDERS, vals):
            self._set_slider(k, v)
        self.suspend = False
        self.update()

    def set_cam(self, cam):
        self.suspend = True
        for k, v in cam.items():
            self._set_slider(k, v if k in ("fx", "fy", "cx", "cy", "cam_x", "cam_y", "cam_z") else np.rad2deg(v))
        self.suspend = False
        self.update()

    # ------------------------------------------------------------------ callbacks
    def _set_mode(self, which, value):
        if which == "qs":
            self.qs_mode = value
        else:
            self.p1_mode = value
        self.update()

    def _on_checks(self, label):
        st = dict(zip(["show data", "residual lines", "3D: fit to bike only"], self.checks.get_status()))
        self.show_data, self.show_lines, self.fit_bike = (st["show data"], st["residual lines"],
                                                          st["3D: fit to bike only"])
        self.update()

    def on_autoguess(self, _=None):
        x = auto_guess(self.data, self.bp, self.get_cam())
        self.note = "auto-guess applied (closed-form from the ellipses)"
        self.set_state(x)

    def on_main_x0(self, _=None):
        self.note = "main.py initial guess"
        self.set_state(X0_MAIN)

    def on_reset_cam(self, _=None):
        self.set_cam(self.cam0)

    def on_clear_ghost(self, _=None):
        self.ghost_state = None
        self.update()

    def solver_fun(self, cam):
        data = self.data
        if solver_matches_perpro(self.qs_mode, self.p1_mode, self.bp):
            return lambda x: perpro.residual_eqs(x, data["img"], self.bp, cam)
        return lambda x: residual_vector(x, data, self.bp, cam, self.qs_mode, self.p1_mode)

    def solve(self, _=None):
        lb, ub = self.bounds
        x_start = self.get_state()
        x0 = np.clip(x_start, lb + 1e-9, ub - 1e-9)
        cam = self.get_cam()
        src = "perpro.residual_eqs" if solver_matches_perpro(self.qs_mode, self.p1_mode, self.bp) else "numpy copy"
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sol = least_squares(self.solver_fun(cam), x0, bounds=(lb, ub))
        except Exception as exc:                                   # same failure the user would see
            self.note = f"SOLVER EXCEPTION: {exc}"
            print("[solve]", self.note)
            self.update()
            return
        self.ghost_state = x_start
        fmt = lambda x: np.round(np.r_[np.rad2deg(x[:4]), x[4:]], 3).tolist()
        print(f"\n[solve] residuals from {src}  (Q-S: {self.qs_mode}, P1/P3: {self.p1_mode})")
        print(f"   start (deg,deg,deg,deg,m,m,m): {fmt(x0)}")
        print(f"   end                          : {fmt(sol.x)}")
        print(f"   cost={sol.cost:.3f}  nfev={sol.nfev}  status={sol.status}  ({sol.message})")
        at = [n for n, v, lo, hi in zip(STATE_NAMES, sol.x, lb, ub)
              if v <= lo + 1e-6 or v >= hi - 1e-6]
        if at:
            print(f"   !! at a bound: {at}")
        self.note = f"solve: cost={sol.cost:.1f} nfev={sol.nfev}" + (f"  AT BOUND: {at}" if at else "")
        self.set_state(sol.x)

    # ------------------------------------------------------------------ export
    def x0_text(self, x=None):
        x = self.get_state() if x is None else x
        d = np.rad2deg(x[:4])
        return ("x0 = np.array([\n"
                f"    np.deg2rad({d[0]:.2f}),   # phi   (roll)\n"
                f"    np.deg2rad({d[1]:.2f}),   # theta (pitch)\n"
                f"    np.deg2rad({d[2]:.2f}),   # psi   (yaw)\n"
                f"    np.deg2rad({d[3]:.2f}),   # delta (steer)\n"
                f"    {x[4]:.3f},   # x_r [m]\n"
                f"    {x[5]:.3f},   # y_r [m]\n"
                f"    {x[6]:.3f},   # z_r [m]\n"
                "])")

    def export_x0(self, _=None, quiet_file=False):
        x, cam = self.get_state(), self.get_cam()
        payload = {"x0": [float(v) for v in x],
                   "x0_units": "phi,theta,psi,delta in rad; x_r,y_r,z_r in m",
                   "x0_deg_m": {n: float(v) for n, v in zip(STATE_NAMES, np.r_[np.rad2deg(x[:4]), x[4:]])},
                   "camera_params": {k: float(v) for k, v in cam.items()},
                   "bike_params": {k: float(v) for k, v in self.bp.items()}}
        with open(self.out_path, "w") as fh:
            json.dump(payload, fh, indent=2)
        if quiet_file:
            return
        print("\n" + self.x0_text(x))
        print("camera_params = " + json.dumps({k: round(float(v), 6) for k, v in cam.items()}))
        print(f"(also written to {os.path.abspath(self.out_path)})")
        self.note = f"x0 exported to {self.out_path}"
        self.update()

    # ------------------------------------------------------------------ main update
    def update(self, _=None):
        if self.suspend:
            return
        for k in BP_KEYS:
            self.bp[k] = float(self.sliders[k].val)
        x, cam, bp = self.get_state(), self.get_cam(), self.bp
        sc = compute_scene(x, bp, cam)
        self.main.set(sc, cam)
        if self.ghost_state is not None:
            self.ghost.set(compute_scene(self.ghost_state, bp, cam), cam)
        else:
            self.ghost.clear()

        # camera in 3D -------------------------------------------------------
        C, pc = camera_matrix(cam), camera_position(cam)
        self.cam_pt.set_data_3d([pc[0]], [pc[1]], [pc[2]])
        for ln, i in zip(self.cam_axes, range(3)):
            e = pc + 0.6 * C[:, i]
            ln.set_data_3d([pc[0], e[0]], [pc[1], e[1]], [pc[2], e[2]])
        W, H = self.res
        depth = float(np.clip((sc["pts"]["Cr"] - pc) @ C[:, 1], 1.0, 20.0))
        corners = [C @ (depth * np.array([(u - cam["cx"]) / cam["fx"], 1.0, (v - cam["cy"]) / cam["fy"]])) + pc
                   for u, v in ((0, 0), (W, 0), (W, H), (0, H))]
        for k in range(4):
            a, b = corners[k], corners[(k + 1) % 4]
            self.frustum[k].set_data_3d([pc[0], a[0]], [pc[1], a[1]], [pc[2], a[2]])
            self.frustum[4 + k].set_data_3d([a[0], b[0]], [a[1], b[1]], [a[2], b[2]])

        # 3D limits (equal aspect cube) -----------------------------------------
        cloud = [sc["rim_r"], sc["rim_f"], np.array([sc["pts"][k] for k in ("Cr", "S", "Q", "Cf")])]
        if not self.fit_bike:
            cloud.append(pc[None, :])
            cloud.append(np.array(corners))
        P = np.vstack(cloud)
        P = P[np.all(np.isfinite(P), axis=1)]
        lo, hi = P.min(0), P.max(0)
        mid = (lo + hi) / 2
        half = max(0.55 * np.max(hi - lo), 1.0)
        self.ax3.set_xlim3d(mid[0] - half, mid[0] + half)
        self.ax3.set_ylim3d(mid[1] - half, mid[1] + half)
        self.ax3.set_zlim3d(mid[2] - half, mid[2] + half)
        self.ax3.set_box_aspect((1, 1, 1))

        # data layer visibility + residual lines --------------------------------
        for a in self.data_artists:
            a.set_visible(self.show_data)
        d = self.data
        q = np.array([d["img"]["r_Q_S_u"], d["img"]["r_Q_S_v"]])
        s = sc["uv"]["S"]
        qs_used = self.p1_mode != "extremes + width"
        self.main.qs_model.set_visible(qs_used)
        if qs_used and self.show_data and np.all(np.isfinite(q)) and np.all(np.isfinite(s)):
            self.qs_data.set_data([s[0], s[0] + 60 * q[0]], [s[1], s[1] + 60 * q[1]])
        else:
            self.qs_data.set_data([], [])
        pairs = [(sc["uv"]["Cr"], (d["er"][0], d["er"][1])), (sc["uv"]["Cf"], (d["ef"][0], d["ef"][1])),
                 (sc["uv"]["P1r"], d["p1r"]), (sc["uv"]["P3r"], d["p3r"]),
                 (sc["uv"]["P1f"], d["p1f"]), (sc["uv"]["P3f"], d["p3f"])]
        for ln, (m, dd) in zip(self.res_lines, pairs):
            if self.show_lines and self.show_data and np.all(np.isfinite(m)):
                ln.set_data([m[0], dd[0]], [m[1], dd[1]])
            else:
                ln.set_data([], [])

        self._update_text(x, cam, sc)
        self.fig.canvas.draw_idle()

    def _update_text(self, x, cam, sc):
        d, lb, ub = self.data, *self.bounds
        res = residual_vector(x, d, self.bp, cam, self.qs_mode, self.p1_mode)
        m = model_features(x, self.bp, cam, self.p1_mode)
        if self.qs_mode == "normalised" and self.p1_mode != "extremes + width":
            n = np.hypot(m[12], m[13])
            if n > 0:
                m = m.copy()
                m[12:] /= n
        full = self.p1_mode == "extremes + width"
        names = FULL_NAMES if full else FEATURE_NAMES[:len(res)]
        dvec = d["vec_full"] if full else d["vec"]
        tot = np.nansum(res ** 2)
        cost = 0.5 * tot
        deg = np.rad2deg(x[:4])
        lines = ["x0 = [phi %.1f, theta %.1f, psi %.1f, delta %.1f] deg" % tuple(deg),
                 "     [x_r %.2f, y_r %.2f, z_r %.2f] m" % tuple(x[4:]),
                 "",
                 "bike: " + " ".join(f"{k}={self.bp[k]:.3f}" for k in ("lr", "lf1", "lf2", "lh")),
                 f"       wheelbase(d=0) = lr+lf2 = {self.bp['lr'] + self.bp['lf2']:.3f} m",
                 f"Q-S eqs : {'(unused)' if full else self.qs_mode:11s} residuals : {self.p1_mode}",
                 f"cost = 0.5*sum(res^2) = {cost:,.1f}",
                 "",
                 f"{'eq':3s} {'name':11s} {'model':>9s} {'data':>9s} {'res':>9s} {'share':>5s}"]
        for i, (n, mv, dv, r) in enumerate(zip(names, m, dvec, res)):
            share = 100 * r * r / tot if tot > 0 else 0
            lines.append(f"f{i + 1:02d} {n[:11]:11s} {mv:9.2f} {dv:9.2f} {r:9.2f} {share:4.0f}%")
        warn = []
        if not np.all(np.isfinite(res)):
            warn.append("NaN/inf residual -> least_squares raises 'Residuals are not finite'")
        if self.true_state is not None:
            t = self.true_state
            lines += ["", "synthetic truth: " + ", ".join(f"{v:.1f}" for v in np.rad2deg(t[:4])) + " deg",
                      "                 " + ", ".join(f"{v:.2f}" for v in t[4:]) + " m"]
        for nme, v, lo_, hi_ in zip(STATE_NAMES, x, lb, ub):
            if v < lo_ - 1e-9 or v > hi_ + 1e-9:
                shown = np.rad2deg(v) if nme in STATE_NAMES[:4] else v
                warn.append(f"{nme}={shown:.2f} outside solver bounds")
        behind = [k for k in ("Cr", "Cf", "S", "Q") if np.isnan(sc["uv"][k][0])]
        if behind:
            warn.append("behind the camera: " + ",".join(behind))
        if not full and (np.any(np.isnan(sc["pts"]["P1r"])) or np.any(np.isnan(sc["pts"]["P1f"]))):
            warn.append("wheel axle || N.x: u_hor undefined (NaN in the model)")
        if not full and self.qs_mode == "as coded" and np.isfinite(res).all():
            q = np.hypot(d["img"]["r_Q_S_u"], d["img"]["r_Q_S_v"])
            mq = np.hypot(m[12], m[13])
            if mq > 5 * max(q, 1e-9) and q <= 1.5:
                warn.append(f"f13/f14: data |Q-S|={q:.2f} (unit vec), model |Q-S|={mq:.0f} px")
        if warn:
            lines += ["", "WARNINGS:"] + ["  - " + w for w in warn]
        if self.note:
            lines += ["", self.note]
        self.axt.clear()
        self.axt.axis("off")
        self.axt.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", family="monospace",
                      fontsize=7.4, transform=self.axt.transAxes)

    def show(self):
        plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", help="YOLO-seg .txt of one frame: 'class x1 y1 x2 y2 ...' normalised; "
                                    "class 0 = front wheel, 1 = rear wheel. Omit for a synthetic frame.")
    ap.add_argument("--res", nargs=2, type=int, default=list(SCREEN_RES), metavar=("W", "H"))
    ap.add_argument("--x0", nargs=7, type=float, default=None,
                    metavar=("PHI", "THETA", "PSI", "DELTA", "X", "Y", "Z"),
                    help="initial slider state (angles in degrees, positions in m). "
                         "Default: closed-form auto-guess from the data")
    ap.add_argument("--out", default="x0_guess.json", help="where 'Export x0' writes the json")
    ap.add_argument("--psi-bound-deg", type=float, default=180.0,
                    help="|psi| bound used by Solve (main.py uses 90, which cannot represent a bike "
                         "travelling to the left of the image)")
    ap.add_argument("--roll-bound-deg", type=float, default=90.0)
    ap.add_argument("--steer-bound-deg", type=float, default=90.0)
    ap.add_argument("--bike", nargs=6, type=float, default=None, metavar=("LR", "LF1", "LF2", "LH", "RR", "RF"),
                    help="bike geometry in m (lh = head height above the rear hub; use lh=lf1 for level hubs)")
    ap.add_argument("--noise", type=float, default=0.5, help="px noise of the synthetic annotation")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--qs", choices=["as coded", "normalised", "off"], default="as coded",
                    help="Q-S equations; auto-set to 'off' if your residual_eqs has only 12 residuals")
    ap.add_argument("--p1", choices=P1_MODES, default="extremes + width",
                    help="residual formulation (default: the recommended one, see bike_model_ellipse.py)")
    ap.add_argument("--solve", action="store_true", help="press 'Solve' once at start-up")
    ap.add_argument("--screenshot", help="save a PNG and exit (headless)")
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()

    res = tuple(args.res)
    bp = dict(BIKE_PARAMS)
    if args.bike:
        bp.update(dict(zip(("lr", "lf1", "lf2", "lh", "rr", "rf"), args.bike)))
    cam = default_camera(res)

    true_state = None
    if args.label:
        front_n, rear_n = load_label(args.label)
    else:
        true_state = np.array([np.deg2rad(-6), 0.0, np.deg2rad(35), np.deg2rad(12), 1.0, 6.0, 0.33])
        front_n, rear_n = synthetic_points(true_state, bp, cam, res, noise=args.noise, seed=args.seed)
        print("[info] no --label given: using a synthetic frame generated from the model "
              f"(truth: {np.round(np.r_[np.rad2deg(true_state[:4]), true_state[4:]], 2).tolist()})")
    data = build_data(front_n, rear_n, res)

    global PERPRO_NEQ
    PERPRO_NEQ = perpro_n_eqs(bp, cam)
    if PERPRO_NEQ == 12 and args.qs == "as coded":
        args.qs = "off"
        print("[info] bike_model_perspective.residual_eqs has 12 residuals (no Q-S): using --qs off")

    if not args.no_verify and perpro is not None:
        worst = verify_against_sympy({**bp, "lh": 0.0}, cam)   # the sympy model has no lh
        print(f"[check] numpy model vs lambdified sympy model: max |diff| = {worst:.2e} px "
              f"({'OK' if worst < 1e-6 else 'MISMATCH - the viewer does not show what the solver sees!'})")

    if args.x0 is not None:
        a = np.array(args.x0, float)
        state0 = np.r_[np.deg2rad(a[:4]), a[4:]]
    else:
        state0 = auto_guess(data, bp, cam)
        print("[info] starting from the closed-form auto-guess "
              f"{np.round(np.r_[np.rad2deg(state0[:4]), state0[4:]], 2).tolist()}  "
              "(use the 'main.py's x0' button to see your old guess)")

    if data["ef"][0] < data["er"][0] and args.psi_bound_deg < 180:
        print("[warn] the front wheel is LEFT of the rear wheel (bike travelling left) but |psi| is bounded "
              f"to {args.psi_bound_deg:.0f} deg: the model cannot represent this pose.")
    viewer = Viewer(data, bp, cam, state0, res, make_bounds(args.psi_bound_deg, args.roll_bound_deg, args.steer_bound_deg), args.out,
                    true_state=true_state, qs_mode=args.qs, p1_mode=args.p1)
    if args.solve:
        viewer.solve()
    if args.screenshot:
        viewer.fig.savefig(args.screenshot, dpi=110)
        print(f"[info] saved {args.screenshot}")
        return
    viewer.show()


if __name__ == "__main__":
    main()
