"""Numeric helpers shared by the tests (not collected by pytest)."""
import numpy as np
import sympy as sp

import src.bike_model_perspective as mdl

_V = mdl.variables[:22]          # the 22 symbols used by the lambdified model
_cache = {}


def _f(expr):
    return sp.lambdify(_V, expr, "numpy")


def _get():
    if not _cache:
        m = mdl
        pos = lambda p: _f(p.pos_from(m.O).to_matrix(m.N))
        dirn = lambda v: _f(v.to_matrix(m.N))
        _cache.update(
            Cr=pos(m.Cr), Cf=pos(m.Cf), P=pos(m.P),
            P1r=pos(m.P1r), P3r=pos(m.P3r), P1f=pos(m.P1f), P3f=pos(m.P3f),
            Rx=dirn(m.R.x), Ry=dirn(m.R.y), Rz=dirn(m.R.z),
            Fx=dirn(m.F.x), Fy=dirn(m.F.y), Fz=dirn(m.F.z),
            Cx=dirn(m.C.x), Cy=dirn(m.C.y), Cz=dirn(m.C.z),
        )
    return _cache


def pack(state, bike, cam):
    """Substitution tuple in the exact order of `bike_model_perspective.variables`."""
    return (*map(float, state),
            bike["lr"], bike["lf1"], bike["lf2"], bike["rr"], bike["rf"],
            cam["fx"], cam["fy"], cam["cx"], cam["cy"],
            cam["cam_x"], cam["cam_y"], cam["cam_z"],
            cam["cam_yaw"], cam["cam_pitch"], cam["cam_roll"])


def vec(name, s):
    return np.array(_get()[name](*s), dtype=float).ravel()


def wheel_circle(s, wheel, n=720):
    """3D points of the wheel rim in world coordinates."""
    if wheel == "rear":
        c, e1, e2, r = vec("Cr", s), vec("Rx", s), vec("Rz", s), s[10]
    else:
        c, e1, e2, r = vec("Cf", s), vec("Fx", s), vec("Fz", s), s[11]
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return c + r * (np.cos(t)[:, None] * e1 + np.sin(t)[:, None] * e2)


def project(pts, s):
    """Pinhole projection, same convention as perspective_projection()."""
    rel = pts - vec("P", s)
    Xc, Yc, Zc = rel @ vec("Cx", s), rel @ vec("Cy", s), rel @ vec("Cz", s)
    fx, fy, cx, cy = s[12:16]
    return np.column_stack([fx * Xc / Yc + cx, fy * Zc / Yc + cy])


def ellipse_points_norm(xc, yc, a, b, theta, res, n=40, noise=0.0, seed=0):
    """Points on an ellipse given in (y-up) pixels, returned as normalized
    label-file coordinates (x/W, 1 - y/H)."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = xc + a * np.cos(t) * np.cos(theta) - b * np.sin(t) * np.sin(theta)
    y = yc + a * np.cos(t) * np.sin(theta) + b * np.sin(t) * np.cos(theta)
    if noise:
        rng = np.random.default_rng(seed)
        x = x + rng.normal(0, noise, n)
        y = y + rng.normal(0, noise, n)
    return np.column_stack([x / res[0], 1 - y / res[1]])


def angle_diff_mod_pi(a, b):
    """Smallest difference between two undirected angles (period pi)."""
    return (a - b + np.pi / 2) % np.pi - np.pi / 2