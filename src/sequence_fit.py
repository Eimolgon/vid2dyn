# src/sequence_fit.py
"""
Pose of the bicycle over a whole video: directory of per-frame annotation .txt files
-> smoothed trajectory (roll, pitch, yaw, steer, x, y, z) as CSV + a QA figure.

    python sequence_fit.py DIR --focal-mm 28 --sensor-w-mm 23.5 --orig-res 6020 4024 \
                           --res 1920 1080 --wheelbase 1.08 --out results.csv

What makes a trajectory smooth (and right) instead of N independent guesses
---------------------------------------------------------------------------
 1. one consistent camera and bike for the whole video (shared parameters, see
    --calibrate: wheelbase / focal scale estimated ONCE from many frames, then fixed);
 2. a temporal model: all frames are solved together (one sparse least-squares problem)
    with a penalty on the second difference of every state ("constant velocity" prior).
    The data term is in pixels (/ sigma_px), the smoothness term in units of the
    expected change of velocity per frame (--smooth scales it);
 3. continuous angles: yaw is unwrapped (a bike travelling left sits at ~180 deg and
    crosses +-180 all the time; bounds of +-pi would clamp it) and wide bounds are used;
 4. a robust loss (soft-L1) so a frame with a bad annotation does not drag the neighbours,
    and frames without both wheels are simply bridged by the smoothness term;
 5. a good start: frame-by-frame warm-started fits (mirrored yaw starts on frame 1).

Only the following inputs are needed: the directory, the image size the labels are
normalised to (--res), the camera (--focal-mm + --sensor-w-mm, or --focal-px or --hfov-deg),
the wheel diameter and, optionally, the wheelbase (otherwise it is estimated).
"""

import argparse
import csv
import os
import re
import sys
import time
import warnings

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bike_model_ellipse as be

STATE_NAMES = ("phi", "theta", "psi", "delta", "x_r", "y_r", "z_r")
DEG_STATES = (0, 1, 2, 3)


# ---------------------------------------------------------------------------
# Ellipse fit (Halir & Flusser 1998: numerically stable, no complex eigenvectors)
# ---------------------------------------------------------------------------
def fit_ellipse(points_n, res):
    """
    points_n : (N,2) annotation points normalised to [0,1], y pointing DOWN (YOLO).
    Returns (xc, zc, a, b, theta) in pixels with v pointing UP (same convention as
    utils.fitEllipse), a >= b the semi axes and theta the angle of the major axis
    in [-pi/2, pi/2). Returns None if the fit is not a valid ellipse.
    """
    p = np.asarray(points_n, float)
    if p.ndim != 2 or len(p) < 6:
        return None
    u = p[:, 0] * res[0]
    v = (1.0 - p[:, 1]) * res[1]
    m = np.array([u.mean(), v.mean()])
    sc = np.sqrt(np.mean((u - m[0]) ** 2 + (v - m[1]) ** 2))
    if not np.isfinite(sc) or sc < 1e-9:
        return None
    x, y = (u - m[0]) / sc, (v - m[1]) / sc

    D1 = np.column_stack([x * x, x * y, y * y])
    D2 = np.column_stack([x, y, np.ones_like(x)])
    S1, S2, S3 = D1.T @ D1, D1.T @ D2, D2.T @ D2
    try:
        T = -np.linalg.solve(S3, S2.T)
        M = S1 + S2 @ T
        M = np.array([M[2] / 2, -M[1], M[0] / 2])
        w, vec = np.linalg.eig(M)
    except np.linalg.LinAlgError:
        return None
    vec = np.real(vec)
    cond = 4 * vec[0] * vec[2] - vec[1] ** 2
    ok = np.where(cond > 0)[0]
    if len(ok) == 0:
        return None
    a1 = vec[:, ok[0]]
    a2 = T @ a1
    a, b, c, d, e, f = a1[0], a1[1], a1[2], a2[0], a2[1], a2[2]

    den = b * b - 4 * a * c
    if abs(den) < 1e-14:
        return None
    xc = (2 * c * d - b * e) / den
    yc = (2 * a * e - b * d) / den
    q = -(f + (d * xc + e * yc) / 2)
    ev, evec = np.linalg.eigh(np.array([[a, b / 2], [b / 2, c]]))
    with np.errstate(all="ignore"):
        ax = np.sqrt(q / ev)
    if not np.all(np.isfinite(ax)) or np.any(ax <= 0):
        return None
    k = int(np.argmax(ax))
    a_len, b_len = ax[k], ax[1 - k]
    theta = np.arctan2(evec[1, k], evec[0, k])
    theta = (theta + np.pi / 2) % np.pi - np.pi / 2
    return (xc * sc + m[0], yc * sc + m[1], a_len * sc, b_len * sc, theta)


# ---------------------------------------------------------------------------
# Directory loading
# ---------------------------------------------------------------------------
def _natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def parse_label(path):
    """{class_id: (N,2) points}; blank lines are ignored; the longest polygon per class wins."""
    out = {}
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 7:
                continue
            try:
                cid = int(float(parts[0]))
                pts = np.array(parts[1:], float)
            except ValueError:
                continue
            if len(pts) % 2:
                pts = pts[:-1]
            pts = pts.reshape(-1, 2)
            if cid not in out or len(pts) > len(out[cid]):
                out[cid] = pts
    return out


def load_sequence(directory, res, front_class=0, rear_class=1, min_points=8, verbose=True):
    """
    Returns a dict with, per file: names, frame_id (integer in the name, else the order),
    front_pts / rear_pts (normalised polygons or None), ef / er (N,5) ellipses (NaN if
    missing or invalid) and valid (bool).
    """
    files = sorted((f for f in os.listdir(directory) if f.lower().endswith(".txt")), key=_natural_key)
    if not files:
        raise FileNotFoundError(f"no .txt files in {directory}")
    n = len(files)
    ids = []
    for i, f in enumerate(files):
        nums = re.findall(r"\d+", os.path.splitext(f)[0])
        ids.append(int(nums[-1]) if nums else i)
    if len(set(ids)) != n:
        ids = list(range(n))
    seq = {"files": files, "frame_id": np.array(ids), "front_pts": [None] * n, "rear_pts": [None] * n,
           "ef": np.full((n, 5), np.nan), "er": np.full((n, 5), np.nan), "valid": np.zeros(n, bool),
           "res": tuple(res), "directory": directory}
    bad = 0
    for i, f in enumerate(files):
        lab = parse_label(os.path.join(directory, f))
        pf, pr = lab.get(front_class), lab.get(rear_class)
        seq["front_pts"][i], seq["rear_pts"][i] = pf, pr
        if pf is None or pr is None or len(pf) < min_points or len(pr) < min_points:
            bad += 1
            continue
        ef, er = fit_ellipse(pf, res), fit_ellipse(pr, res)
        if ef is None or er is None:
            bad += 1
            continue
        if min(ef[3], er[3]) < 3 or max(ef[2], er[2]) > 2 * max(res):      # absurd sizes
            bad += 1
            continue
        seq["ef"][i], seq["er"][i], seq["valid"][i] = ef, er, True
    if verbose:
        print(f"[load] {n} files, {seq['valid'].sum()} with two valid wheel ellipses, {bad} unusable")
    return seq


def to_grid(seq):
    """Re-index onto the contiguous frame-number grid min..max; absent files become invalid frames,
    so the smoothness term bridges the gap with the right time spacing."""
    ids = seq["frame_id"]
    full = np.arange(ids.min(), ids.max() + 1)
    if len(full) == len(ids):
        return seq
    pos = {int(i): k for k, i in enumerate(ids)}
    n = len(full)
    out = dict(seq)
    out["frame_id"] = full
    out["files"] = [seq["files"][pos[int(i)]] if int(i) in pos else None for i in full]
    for key in ("front_pts", "rear_pts"):
        out[key] = [seq[key][pos[int(i)]] if int(i) in pos else None for i in full]
    for key, shape in (("ef", (n, 5)), ("er", (n, 5))):
        a = np.full(shape, np.nan)
        for k, i in enumerate(full):
            if int(i) in pos:
                a[k] = seq[key][pos[int(i)]]
        out[key] = a
    v = np.zeros(n, bool)
    for k, i in enumerate(full):
        v[k] = bool(seq["valid"][pos[int(i)]]) if int(i) in pos else False
    out["valid"] = v
    return out


# ---------------------------------------------------------------------------
# Synthetic video (for tests / demos)
# ---------------------------------------------------------------------------
def simulate_states(T=120, fps=30.0, left=True):
    """A bike going across the image with small oscillations of roll, steer and yaw."""
    s = np.linspace(0, 1, T)
    X = np.zeros((T, 7))
    X[:, 0] = np.deg2rad(3.0) * np.sin(2 * np.pi * 1.5 * s)                      # roll
    X[:, 1] = np.deg2rad(0.6) * np.sin(2 * np.pi * 2.0 * s + 0.5)                # pitch
    base = np.pi if left else 0.0
    X[:, 2] = base + np.deg2rad(-8 + 16 * s)                                     # yaw crosses 180 deg
    X[:, 3] = np.deg2rad(6.0) * np.sin(2 * np.pi * 2.0 * s + 1.0)                # steer
    X[:, 4] = (1.1 - 2.4 * s) if left else (-1.1 + 2.4 * s)
    X[:, 5] = 3.0 + 0.6 * s
    X[:, 6] = 0.33 + 0.004 * np.sin(2 * np.pi * 3 * s)
    return X


def write_synthetic_sequence(directory, states, bp, cam, res, noise_px=0.7, n_pts=90, seed=0,
                             outlier_frames=(), missing_frames=(), one_wheel_frames=()):
    """Writes frame_000000.txt ... in the same format as the real labels."""
    os.makedirs(directory, exist_ok=True)
    rng = np.random.default_rng(seed)
    cam_ = cam
    for t, x in enumerate(states):
        if t in missing_frames:
            continue
        phi, theta, psi, delta = x[:4]
        R = be._Rz(psi) @ be._Rx(phi) @ be._Ry(theta)
        F = R @ be._Rz(delta)
        Cr = x[4:7]
        S = Cr + bp["lr"] * R[:, 0] + bp.get("lh", 0.0) * R[:, 2]
        Cf = S - bp["lf1"] * F[:, 2] + bp["lf2"] * F[:, 0]
        C = be.camera_matrix(cam_)
        pc = np.array([cam_["cam_x"], cam_["cam_y"], cam_["cam_z"]])
        lines = []
        for cid, (c, M, rad) in ((0, (Cf, F, bp["rf"])), (1, (Cr, R, bp["rr"]))):
            if cid == 0 and t in one_wheel_frames:
                continue
            ang = np.sort(rng.uniform(0, 2 * np.pi, n_pts))
            P = c + rad * (np.cos(ang)[:, None] * M[:, 0] + np.sin(ang)[:, None] * M[:, 2])
            rc = (P - pc) @ C
            u = cam_["fx"] * rc[:, 0] / rc[:, 1] + cam_["cx"] + noise_px * rng.standard_normal(n_pts)
            v = cam_["fy"] * rc[:, 2] / rc[:, 1] + cam_["cy"] + noise_px * rng.standard_normal(n_pts)
            if t in outlier_frames and cid == 1:                         # a sloppy annotation
                u = u + rng.uniform(25, 45)
                v = v - rng.uniform(15, 35)
                u = cam_["cx"] + (u - cam_["cx"]) * 1.12
            xy = np.column_stack([u / res[0], 1.0 - v / res[1]]).ravel()
            lines.append(f"{cid} " + " ".join(f"{q:.6f}" for q in xy))
        with open(os.path.join(directory, f"frame_{t:06d}.txt"), "w") as fh:
            fh.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Solving
# ---------------------------------------------------------------------------
def default_bounds(psi_range=3 * np.pi, pos=20.0):
    """Wide yaw range: the model is periodic in yaw and an unwrapped yaw can pass +-pi."""
    lb = np.array([-np.pi / 2, -np.pi / 2, -psi_range, -np.pi / 2, -pos, 0.1, -pos])
    ub = np.array([np.pi / 2, np.pi / 2, psi_range, np.pi / 2, pos, 50.0, pos])
    return lb, ub


def make_bike(wheel_diameter=0.6604, wheelbase=1.1, lf2=0.1, lf1=0.5, front_wheel_diameter=None):
    """lh = lf1 puts both hubs on the same level; only (lr + lf2) = wheelbase matters then."""
    return {"lr": wheelbase - lf2, "lf1": lf1, "lf2": lf2, "lh": lf1,
            "rr": wheel_diameter / 2, "rf": (front_wheel_diameter or wheel_diameter) / 2}


def observations(seq):
    """(N,12) data vectors (the KEYS of bike_model_ellipse); NaN rows for unusable frames."""
    obs = np.full((len(seq["files"]), len(be.KEYS)), np.nan)
    v = np.where(seq["valid"])[0]
    if len(v):
        d = be.image_data(seq["ef"][v], seq["er"][v])
        obs[v] = np.column_stack([d[k] for k in be.KEYS])
    return obs


def interpolate_states(X, valid):
    """Fill rows of X where valid is False by linear interpolation (yaw is already unwrapped)."""
    X = X.copy()
    idx = np.arange(len(X))
    for j in range(X.shape[1]):
        X[~valid, j] = np.interp(idx[~valid], idx[valid], X[valid, j])
    return X


def initial_states(seq, obs, bp, cam, bounds, x0=None, prior_deg=10.0, sigma_px=1.0):
    """Frame-by-frame fits, warm-started, for every valid frame. Returns (N,7) (gaps interpolated)."""
    v = np.where(seq["valid"])[0]
    if len(v) == 0:
        raise RuntimeError("no frame with two valid ellipses")
    if x0 is None:
        x0 = be.initial_guess(seq["ef"][v[0]], seq["er"][v[0]], bp, cam)
    lb, ub = bounds
    x0 = np.clip(x0, lb + 1e-6, ub - 1e-6)
    sp = np.deg2rad(prior_deg)
    starts = [x0]
    Xv, costs = [], []
    for j, t in enumerate(v):
        def fun(x, o=obs[t]):
            r = (be.model_features_batch(x[None], bp, cam)[0] - o) / sigma_px
            return np.r_[r, x[0] / sp, x[3] / sp]
        cands = [starts[-1]]
        if j == 0:                                    # mirrored yaw starts on the first frame only
            for dpsi in (np.pi, -np.pi / 2, np.pi / 2):
                c = x0.copy(); c[2] = x0[2] + dpsi; cands.append(np.clip(c, lb + 1e-6, ub - 1e-6))
        best = None
        for c in cands:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r = least_squares(fun, c, bounds=(lb, ub), x_scale="jac")
            if best is None or r.cost < best.cost:
                best = r
        starts.append(best.x)
        Xv.append(best.x); costs.append(best.cost)
    Xv = np.array(Xv)
    Xv[:, 2] = np.unwrap(Xv[:, 2])
    X = np.zeros((len(obs), 7))
    X[v] = Xv
    valid = seq["valid"].copy()
    return interpolate_states(X, valid), np.array(costs)


def calibrate(obs, X_init, bp, cam, bounds, fit_L=True, fit_f=False, n_frames=20, sigma_px=1.0,
              prior_L=None, sigma_L=0.2, sigma_f=0.2, robust=3.0, seed=0):
    """
    Estimate SHARED parameters once, from many frames at once: the wheelbase L (hub-to-hub at
    zero steer) and/or a focal-length factor. The states of the chosen frames are free as well.
    Weak log-normal priors keep the unobservable directions from running away
    (the focal length is only observable through foreshortening, i.e. NOT in a side view).
    Returns (bp_new, cam_new, info).
    """
    valid = np.where(np.isfinite(obs[:, 0]))[0]
    K = min(n_frames, len(valid))
    sel = valid[np.unique(np.linspace(0, len(valid) - 1, K).round().astype(int))]
    K = len(sel)
    L0 = bp["lr"] + bp["lf2"]
    G = int(fit_L) + int(fit_f)
    lb, ub = bounds
    p0 = np.r_[X_init[sel].ravel(), np.zeros(G)]           # globals in log space (0 = nominal)
    lo = np.r_[np.tile(lb, K), np.full(G, np.log(0.5))]
    hi = np.r_[np.tile(ub, K), np.full(G, np.log(2.0))]
    gi = {}
    c = 0
    if fit_L:
        gi["L"] = c; c += 1
    if fit_f:
        gi["f"] = c; c += 1

    def unpack(p):
        X = p[:7 * K].reshape(K, 7)
        g = p[7 * K:]
        b2, c2 = dict(bp), cam
        if fit_L:
            b2["lr"] = L0 * np.exp(g[gi["L"]]) - bp["lf2"]
        if fit_f:
            c2 = be.scale_focal(cam, np.exp(g[gi["f"]]))
        return X, b2, c2, g

    def fun(p):
        X, b2, c2, g = unpack(p)
        r = ((be.model_features_batch(X, b2, c2) - obs[sel]) / sigma_px).ravel()
        pri = []
        if fit_L and prior_L is not None:
            pri.append(g[gi["L"]] / sigma_L)
        if fit_f:
            pri.append(g[gi["f"]] / sigma_f)
        return np.r_[r, pri]

    nr = 12 * K + (int(fit_L and prior_L is not None) + int(fit_f))
    S = lil_matrix((nr, 7 * K + G), dtype=int)
    for k in range(K):
        S[12 * k:12 * k + 12, 7 * k:7 * k + 7] = 1
        if G:
            S[12 * k:12 * k + 12, 7 * K:] = 1
    r0 = 12 * K
    if fit_L and prior_L is not None:
        S[r0, 7 * K + gi["L"]] = 1; r0 += 1
    if fit_f:
        S[r0, 7 * K + gi["f"]] = 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sol = least_squares(fun, p0, bounds=(lo, hi), jac_sparsity=S, loss="soft_l1", f_scale=robust,
                            x_scale="jac", method="trf", tr_solver="lsmr")
    X, b2, c2, g = unpack(sol.x)
    rms = float(np.sqrt(np.mean(((be.model_features_batch(X, b2, c2) - obs[sel])) ** 2)))
    info = {"frames_used": int(K), "rms_px": rms, "wheelbase": b2["lr"] + b2["lf2"],
            "focal_factor": float(np.exp(g[gi["f"]])) if fit_f else 1.0}
    return b2, c2, info


SIG_ACC_30FPS = np.array([0.4, 0.3, 0.5, 1.0, 0.01, 0.01, 0.004])      # per frame^2: deg, deg, deg, deg, m, m, m


def smooth_solve(obs, X0, bp, cam, bounds, sigma_px=1.0, smooth=1.0, fps=30.0, prior_deg=30.0,
                 robust=3.0, sig_acc=None, max_nfev=60):
    """
    One sparse least-squares problem for ALL frames:
        data      :  (model(x_t) - observation_t) / sigma_px                       [12 per valid frame]
        smoothness:  smooth * (x_{t+1} - 2 x_t + x_{t-1}) / sig_acc                [7 per interior frame]
        prior     :  roll, steer / prior_deg  (weak, only to stop unobservable wandering)
    with a soft-L1 loss so isolated bad frames (and genuine abrupt events) cannot dominate.
    Frames with NaN observations just have no data term and are bridged by the smoothness term.
    """
    N = len(X0)
    vi = np.where(np.isfinite(obs[:, 0]))[0]
    nv = len(vi)
    sa = (SIG_ACC_30FPS if sig_acc is None else np.asarray(sig_acc, float)) * (30.0 / fps) ** 2
    sa = sa.copy()
    sa[:4] = np.deg2rad(sa[:4])
    lb, ub = bounds
    use_prior = prior_deg and prior_deg > 0
    sp = np.deg2rad(prior_deg) if use_prior else 1.0

    def fun(p):
        X = p.reshape(N, 7)
        parts = [((be.model_features_batch(X[vi], bp, cam) - obs[vi]) / sigma_px).ravel()]
        if smooth > 0 and N > 2:
            parts.append((smooth * (X[2:] - 2 * X[1:-1] + X[:-2]) / sa).ravel())
        if use_prior:
            parts.append(np.r_[X[:, 0], X[:, 3]] / sp)
        return np.concatenate(parts)

    nrow = 12 * nv + (7 * (N - 2) if smooth > 0 and N > 2 else 0) + (2 * N if use_prior else 0)
    S = lil_matrix((nrow, 7 * N), dtype=int)
    for j, t in enumerate(vi):
        S[12 * j:12 * j + 12, 7 * t:7 * t + 7] = 1
    r0 = 12 * nv
    if smooth > 0 and N > 2:
        for t in range(1, N - 1):
            for s in range(7):
                S[r0 + 7 * (t - 1) + s, [7 * (t - 1) + s, 7 * t + s, 7 * (t + 1) + s]] = 1
        r0 += 7 * (N - 2)
    if use_prior:
        for t in range(N):
            S[r0 + t, 7 * t + 0] = 1
            S[r0 + N + t, 7 * t + 3] = 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sol = least_squares(fun, np.clip(X0, lb + 1e-6, ub - 1e-6).ravel(),
                            bounds=(np.tile(lb, N), np.tile(ub, N)), jac_sparsity=S, loss="soft_l1",
                            f_scale=robust, x_scale="jac", method="trf", tr_solver="lsmr",
                            max_nfev=max_nfev)
    return sol.x.reshape(N, 7), sol


def frame_rms(X, obs, bp, cam):
    """Per-frame RMS of the 12 data residuals in pixels (NaN where there is no observation)."""
    out = np.full(len(X), np.nan)
    vi = np.where(np.isfinite(obs[:, 0]))[0]
    out[vi] = np.sqrt(np.mean((be.model_features_batch(X[vi], bp, cam) - obs[vi]) ** 2, axis=1))
    return out


def run_pipeline(seq, bp, cam, calibrate_what="L", smooth=1.0, sigma_px=1.0, fps=30.0, prior_deg=15.0,
                 robust=3.0, x0=None, prior_L=None, verbose=True, max_nfev=60):
    """Everything after loading. Returns a dict with raw / smoothed states, geometry and QA."""
    t0 = time.time()
    seq = to_grid(seq)
    bounds = default_bounds()
    obs = observations(seq)
    log = (lambda *a: print(*a)) if verbose else (lambda *a: None)
    X_raw, costs = initial_states(seq, obs, bp, cam, bounds, x0)
    log(f"[init] per-frame fits done ({np.isfinite(obs[:, 0]).sum()} frames, {time.time() - t0:.1f}s)")
    info = {}
    if calibrate_what and calibrate_what != "none":
        fit_L, fit_f = "L" in calibrate_what, "f" in calibrate_what
        bp2, cam2, info = calibrate(obs, X_raw, bp, cam, bounds, fit_L=fit_L, fit_f=fit_f,
                                    prior_L=prior_L, sigma_px=sigma_px, robust=robust)
        log(f"[calibrate] {info['frames_used']} frames: wheelbase {bp['lr'] + bp['lf2']:.3f} -> "
            f"{info['wheelbase']:.3f} m, focal factor {info['focal_factor']:.3f}, rms {info['rms_px']:.2f} px")
        bp, cam = bp2, cam2
        X_raw, costs = initial_states(seq, obs, bp, cam, bounds, x0)
    X_s, sol = smooth_solve(obs, X_raw, bp, cam, bounds, sigma_px=sigma_px, smooth=smooth, fps=fps,
                            prior_deg=prior_deg, robust=robust, max_nfev=max_nfev)
    log(f"[smooth] cost {sol.cost:.1f}, nfev {sol.nfev}, {time.time() - t0:.1f}s total")
    return {"seq": seq, "X_raw": X_raw, "X": X_s, "obs": obs, "bp": bp, "cam": cam, "calib": info,
            "rms_raw": frame_rms(X_raw, obs, bp, cam), "rms": frame_rms(X_s, obs, bp, cam)}


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
CSV_COLS = ["frame", "file", "valid", "roll_deg", "pitch_deg", "yaw_deg", "steer_deg", "x_m", "y_m", "z_m",
            "rms_px_smooth", "rms_px_raw", "roll_raw_deg", "pitch_raw_deg", "yaw_raw_deg", "steer_raw_deg",
            "x_raw_m", "y_raw_m", "z_raw_m"]


def write_csv(path, out):
    seq, X, Xr = out["seq"], out["X"], out["X_raw"]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(CSV_COLS)
        for t in range(len(X)):
            def conv(x):
                return list(np.round(np.r_[np.rad2deg(x[:2]), np.rad2deg(x[2]) % 360.0, np.rad2deg(x[3]), x[4:]], 4))
            w.writerow([int(seq["frame_id"][t]), seq["files"][t] or "", int(seq["valid"][t])] + conv(X[t]) +
                       [round(out["rms"][t], 3), round(out["rms_raw"][t], 3)] + conv(Xr[t]))


def save_qa_plot(path, out, fps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    seq, X, Xr = out["seq"], out["X"], out["X_raw"]
    t = seq["frame_id"] / fps
    fig, ax = plt.subplots(4, 2, figsize=(13, 9), sharex=True)
    items = [("roll [deg]", 0, True), ("pitch [deg]", 1, True), ("yaw [deg]", 2, True), ("steer [deg]", 3, True),
             ("x [m]", 4, False), ("depth y [m]", 5, False), ("z [m]", 6, False)]
    for k, (lab, j, deg) in enumerate(items):
        a = ax.flat[k]
        f = np.rad2deg if deg else (lambda q: q)
        a.plot(t, f(Xr[:, j]), ".", ms=3, color="0.6", label="per-frame")
        a.plot(t, f(X[:, j]), "-", lw=1.6, color="C0", label="smoothed")
        a.set_ylabel(lab)
        a.grid(alpha=.3)
    a = ax.flat[7]
    a.plot(t, out["rms_raw"], ".", ms=3, color="0.6")
    a.plot(t, out["rms"], "-", color="C3")
    a.set_ylabel("RMS residual [px]")
    a.set_yscale("log")
    a.grid(alpha=.3)
    ax.flat[0].legend()
    for a in ax[-1]:
        a.set_xlabel("time [s]")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def camera_from_args(a):
    res = tuple(int(q) for q in a.res.split("x"))
    orig = tuple(int(q) for q in a.orig_res.split("x")) if a.orig_res else None
    if a.focal_px:
        f = a.focal_px
        cam = be.make_camera(res, focal_mm=28.0, sensor_w_mm=23.5, orig_res=orig)
        cam = dict(cam, fx=f, fy=f * ((res[1] / res[0]) / (orig[1] / orig[0]) if orig else 1.0))
    elif a.hfov_deg:
        f = res[0] / 2 / np.tan(np.deg2rad(a.hfov_deg) / 2)
        cam = be.make_camera(res, focal_mm=28.0, sensor_w_mm=23.5, orig_res=orig)
        cam = dict(cam, fx=f, fy=f * ((res[1] / res[0]) / (orig[1] / orig[0]) if orig else 1.0))
    else:
        cam = be.make_camera(res, focal_mm=a.focal_mm, sensor_w_mm=a.sensor_w_mm, orig_res=orig,
                             cam_pos=(0, 0, a.cam_height))
    cam["cam_z"] = a.cam_height
    return res, cam


def main(argv=None):
    ap = argparse.ArgumentParser(description="Bike pose over a directory of per-frame label .txt files, smoothed jointly.")
    ap.add_argument("directory", nargs="?", help="directory with the sequenced .txt labels")
    ap.add_argument("--res", default="1920x1080", help="image size the labels are normalised to (WxH)")
    ap.add_argument("--orig-res", default=None, help="original capture size WxH if the image was resized/stretched, e.g. 6020x4024")
    ap.add_argument("--focal-mm", type=float, default=28.0)
    ap.add_argument("--sensor-w-mm", type=float, default=23.5, help="SENSOR WIDTH in mm (only the ratio focal/sensor matters)")
    ap.add_argument("--focal-px", type=float, default=None, help="focal length in pixels (overrides mm/sensor)")
    ap.add_argument("--hfov-deg", type=float, default=None, help="horizontal field of view (overrides mm/sensor)")
    ap.add_argument("--cam-height", type=float, default=0.45)
    ap.add_argument("--wheel-diameter", type=float, default=0.6604, help="metres (sets the metric scale)")
    ap.add_argument("--wheelbase", type=float, default=1.1, help="initial hub-to-hub distance [m]")
    ap.add_argument("--calibrate", default="L", help="shared parameters to estimate: none | L | f | Lf")
    ap.add_argument("--smooth", type=float, default=1.0, help="smoothness multiplier (0 = none, 3 = stiffer)")
    ap.add_argument("--sigma-px", type=float, default=1.0)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--out", default="bike_states.csv")
    ap.add_argument("--plot", default=None, help="QA figure path (default: next to the CSV)")
    ap.add_argument("--make-demo", default=None, metavar="DIR", help="write a synthetic label sequence to DIR and exit")
    a = ap.parse_args(argv)
    res, cam = camera_from_args(a)
    if a.make_demo:
        bp = make_bike(a.wheel_diameter, a.wheelbase)
        write_synthetic_sequence(a.make_demo, simulate_states(), bp, cam, res, outlier_frames=[20, 50],
                                 missing_frames=[30, 31, 32], one_wheel_frames=[60])
        print("demo labels written to", a.make_demo)
        return 0
    if not a.directory:
        ap.error("directory required")
    seq = load_sequence(a.directory, res)
    bp = make_bike(a.wheel_diameter, a.wheelbase)
    out = run_pipeline(seq, bp, cam, calibrate_what=a.calibrate, smooth=a.smooth, sigma_px=a.sigma_px, fps=a.fps)
    write_csv(a.out, out)
    plot = a.plot or os.path.splitext(a.out)[0] + "_qa.png"
    save_qa_plot(plot, out, a.fps)
    print(f"[done] {a.out}, {plot}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
