#!/usr/bin/env python3
"""
Kalman smoother (EKF forward pass + Rauch-Tung-Striebel backward pass) for the bike pose,
STAGE 1: kinematic bicycle process model.   Input: the CSV written by sequence_fit.py.

    python kalman_smoother.py bike_states.csv --wheelbase 1.1 --lf2 0.1 --out bike_kf.csv

Frames (world N: x right in the image, y depth, z up;  bike body R: x forward, y left, z up)
-------------------------------------------------------------------------------------------
State  s = [phi, theta, psi, delta, x, y, z, v, delta_dot]      (rad, rad, rad, rad, m, m, m, m/s, rad/s)
         phi roll (about R.x, + = leaning to the bike's RIGHT), theta pitch, psi yaw (0 = moving to +x),
         delta steer, (x, y, z) rear hub in N, v forward speed of the rear hub, delta_dot steer rate.

Process model (what makes the motion "bicycle-like")
-----------------------------------------------------
  x'     = v cos(psi)                      the hub moves along the heading (no side slip)
  y'     = v sin(psi)
  psi'   = v sin(delta) / (lr cos(delta) + lf2)     exact planar kinematics of a bike with a vertical
                                                    steering axis and fork offset lf2 (= v tan(delta)/L
                                                    for lf2 = 0)
  phi    relaxes towards the steady-state lean  phi_ss = -atan(v psi' / g)  with time constant tau_lean
         (a left turn, psi' > 0, leans left = negative phi).  NOT strict: the rider can lean off it, the
         process noise on phi allows that.   --lean-tau 0 switches the coupling off.
  delta' = delta_dot,  delta_dot decays with tau_steer and is driven by white acceleration noise
  v      random walk (acceleration noise), theta mean-reverting, z random walk (small).
Speed is NOT measured. It follows from the position changes (metric thanks to the wheel size) and
from the yaw-steer coupling. It is only as good as the depth/position scale (see README notes).

Measurements: the 7 pose states [phi theta psi delta x y z] from the per-frame fit (*_raw columns), each
frame with its own sigma (sig_* columns; defaults if absent). Frames with valid=0 get no update (the
filter predicts through the gap). Innovation gating inflates R of frames that disagree with the dynamics.
"""
import argparse
import csv
import os
import sys
from dataclasses import dataclass

import numpy as np

G = 9.81
NS = 9
IDX = dict(phi=0, theta=1, psi=2, delta=3, x=4, y=5, z=6, v=7, ddot=8)


@dataclass
class Params:
    lr: float = 1.0                 # rear hub -> steering axis [m]
    lf2: float = 0.1                # fork offset [m]  (hub-to-hub = lr + lf2)
    tau_lean: float = 0.5           # s; 0 = no lean coupling
    tau_steer: float = 0.4          # s, decay of the steer rate
    tau_pitch: float = 3.0          # s, pitch mean reversion
    # white-noise intensities (sigma per sqrt(second) unless stated)
    q_phi_deg: float = 4.0
    q_theta_deg: float = 0.5
    q_psi_deg: float = 1.0          # sideslip / model error in the heading
    q_steer_acc_deg: float = 400.0  # deg/s^2 per sqrt(s) : how violently the rider steers
    q_xy: float = 0.03
    q_z: float = 0.01
    q_v: float = 1.0                # m/s^2 per sqrt(s): how fast the speed can change
    gate_chi2: float = 24.3         # 99.9% of chi2(7)
    sub_dt: float = 1 / 60          # integration sub-step [s]
    theta_mean: float = 0.0


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# ---------------------------------------------------------------------------------- model
def _deriv(s, p):
    phi, th, psi, de, x, y, z, v, dd = s
    om = v * np.sin(de) / (p.lr * np.cos(de) + p.lf2)
    return om


def f_step(s, dt, p):
    """Deterministic transition over dt (sub-stepped; midpoint rule for the planar motion)."""
    s = s.copy()
    n = max(1, int(np.ceil(dt / p.sub_dt)))
    h = dt / n
    for _ in range(n):
        phi, th, psi, de, x, y, z, v, dd = s
        om = _deriv(s, p)
        psim = psi + 0.5 * h * om
        s[IDX["x"]] = x + h * v * np.cos(psim)
        s[IDX["y"]] = y + h * v * np.sin(psim)
        s[IDX["psi"]] = psi + h * om
        s[IDX["delta"]] = de + h * dd
        s[IDX["ddot"]] = dd * np.exp(-h / p.tau_steer)
        if p.tau_lean > 0:
            phi_ss = -np.arctan(v * om / G)
            s[IDX["phi"]] = phi_ss + (phi - phi_ss) * np.exp(-h / p.tau_lean)
        s[IDX["theta"]] = p.theta_mean + (th - p.theta_mean) * np.exp(-h / p.tau_pitch)
    return s


def jac_f(s, dt, p, eps=1e-6):
    F = np.zeros((NS, NS))
    for j in range(NS):
        d = np.zeros(NS); d[j] = eps
        F[:, j] = (f_step(s + d, dt, p) - f_step(s - d, dt, p)) / (2 * eps)
    return F


def Q_of(dt, p):
    d2r = np.deg2rad
    Q = np.zeros((NS, NS))
    for k, q in (("phi", d2r(p.q_phi_deg)), ("theta", d2r(p.q_theta_deg)), ("psi", d2r(p.q_psi_deg)),
                 ("x", p.q_xy), ("y", p.q_xy), ("z", p.q_z), ("v", p.q_v)):
        Q[IDX[k], IDX[k]] = q ** 2 * dt
    qa = d2r(p.q_steer_acc_deg) ** 2
    i, j = IDX["delta"], IDX["ddot"]
    Q[i, i] = qa * dt ** 3 / 3; Q[i, j] = Q[j, i] = qa * dt ** 2 / 2; Q[j, j] = qa * dt
    return Q


# ---------------------------------------------------------------------------------- filter
def run_filter(Z, SIG, valid, dts, p, s0=None, P0=None):
    """
    Z (N,7) measurements [phi theta psi delta x y z] (rad / m), SIG (N,7) their sigmas, valid (N,),
    dts (N,) time since the previous row (dts[0] unused). Returns smoothed (N,9) states, covariances and
    the forward-filtered states.
    """
    N = len(Z)
    H = np.zeros((7, NS)); H[:7, :7] = np.eye(7)
    xs_f, Ps_f, xs_p, Ps_p, Fs = (np.zeros((N, NS)), np.zeros((N, NS, NS)), np.zeros((N, NS)),
                                  np.zeros((N, NS, NS)), np.zeros((N, NS, NS)))
    gate_scale = np.ones(N)
    vi = np.where(valid)[0]
    if s0 is None:
        s0 = np.zeros(NS)
        s0[:7] = Z[vi[0]]
        k = vi[:6]
        if len(k) > 2:                                           # initial speed from the first position changes
            dpos = np.diff(Z[k][:, 4:6], axis=0)
            hd = np.array([np.cos(Z[k[0], 2]), np.sin(Z[k[0], 2])])
            s0[IDX["v"]] = max(0.0, np.median(dpos @ hd / (np.diff(np.cumsum(np.r_[0, dts[k[1:]]])) + 1e-9)))
    if P0 is None:
        P0 = np.diag(np.r_[SIG[vi[0]] ** 2 + 1e-6, 3.0 ** 2, np.deg2rad(200) ** 2])
        P0[IDX["theta"], IDX["theta"]] += np.deg2rad(2) ** 2
    s, P = s0.copy(), P0.copy()
    for t in range(N):
        if t == 0:
            sp, Pp, F = s, P, np.eye(NS)
        else:
            dt = dts[t]
            F = jac_f(s, dt, p)
            sp, Pp = f_step(s, dt, p), F @ P @ F.T + Q_of(dt, p)
        xs_p[t], Ps_p[t], Fs[t] = sp, Pp, F
        if valid[t] and np.all(np.isfinite(SIG[t])):
            z = Z[t].copy()
            z[2] = sp[2] + wrap(z[2] - sp[2])                  # nearest yaw branch
            y = z - H @ sp
            R = np.diag(SIG[t] ** 2 + 1e-12)
            S = H @ Pp @ H.T + R
            d2 = float(y @ np.linalg.solve(S, y))
            if d2 > p.gate_chi2:                                # outlier frame: down-weight, don't discard
                gate_scale[t] = d2 / p.gate_chi2
                S = H @ Pp @ H.T + R * gate_scale[t]
            K = Pp @ H.T @ np.linalg.inv(S)
            s = sp + K @ y
            IKH = np.eye(NS) - K @ H
            P = IKH @ Pp @ IKH.T + K @ (R * gate_scale[t]) @ K.T
        else:
            s, P = sp, Pp
        xs_f[t], Ps_f[t] = s, P
    xs, Ps = xs_f.copy(), Ps_f.copy()                           # RTS
    for t in range(N - 2, -1, -1):
        C = Ps_f[t] @ Fs[t + 1].T @ np.linalg.inv(Ps_p[t + 1] + 1e-12 * np.eye(NS))
        xs[t] = xs_f[t] + C @ (xs[t + 1] - xs_p[t + 1])
        Ps[t] = Ps_f[t] + C @ (Ps[t + 1] - Ps_p[t + 1]) @ C.T
    return xs, Ps, xs_f, gate_scale


# ---------------------------------------------------------------------------------- IO
DEF_SIG = np.array([np.deg2rad(0.5), np.deg2rad(0.4), np.deg2rad(0.5), np.deg2rad(2.0), 0.03, 0.08, 0.01])


def load_csv(path):
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    g = lambda k: np.array([float(r[k]) if r.get(k, "") not in ("", "nan") else np.nan for r in rows])
    ids = np.array([int(r["frame"]) for r in rows])
    Z = np.column_stack([np.deg2rad(g("roll_raw_deg")), np.deg2rad(g("pitch_raw_deg")),
                         np.deg2rad(g("yaw_raw_deg")), np.deg2rad(g("steer_raw_deg")),
                         g("x_raw_m"), g("y_raw_m"), g("z_raw_m")])
    cols = ["sig_roll_deg", "sig_pitch_deg", "sig_yaw_deg", "sig_steer_deg", "sig_x_m", "sig_y_m", "sig_z_m"]
    if all(c in rows[0] for c in cols):
        SIG = np.column_stack([np.deg2rad(g(c)) if c.endswith("deg") else g(c) for c in cols])
    else:
        print("[warn] no sig_* columns in the CSV: using default measurement sigmas")
        SIG = np.tile(DEF_SIG, (len(rows), 1))
    valid = np.array([int(r["valid"]) for r in rows], bool) & np.all(np.isfinite(Z), axis=1)
    SIG = np.maximum(SIG, DEF_SIG * 0.05)                       # never trust a frame absolutely
    return ids, Z, SIG, valid


def smooth_csv(ids, Z, SIG, valid, fps, p):
    dts = np.r_[0.0, np.diff(ids)] / fps
    p.theta_mean = float(np.mean(Z[valid, 1]))
    xs, Ps, xf, gs = run_filter(Z, SIG, valid, dts, p)
    return xs, Ps, xf, gs


OUT_COLS = ["frame", "valid", "roll_deg", "pitch_deg", "yaw_deg", "steer_deg", "x_m", "y_m", "z_m",
            "speed_mps", "speed_kmh", "steer_rate_degps", "yaw_rate_degps",
            "sig_roll_deg", "sig_pitch_deg", "sig_yaw_deg", "sig_steer_deg", "sig_x_m", "sig_y_m", "sig_z_m",
            "sig_speed_mps", "gate_scale"]


def write_out(path, ids, valid, xs, Ps, gs, p):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(OUT_COLS)
        for t in range(len(xs)):
            s = xs[t]
            om = _deriv(s, p)
            sg = np.sqrt(np.clip(np.diag(Ps[t]), 0, None))
            w.writerow([ids[t], int(valid[t])] + [round(v, 4) for v in
                       [np.rad2deg(s[0]), np.rad2deg(s[1]), np.rad2deg(s[2]) % 360, np.rad2deg(s[3]), *s[4:7],
                        s[7], s[7] * 3.6, np.rad2deg(s[8]), np.rad2deg(om),
                        np.rad2deg(sg[0]), np.rad2deg(sg[1]), np.rad2deg(sg[2]), np.rad2deg(sg[3]), *sg[4:7], sg[7],
                        gs[t]]])


# ---------------------------------------------------------------------------------- test simulator
def simulate_truth(T=150, fps=30.0, p=None, seed=0, v0=4.0, psi0=np.pi - 0.15, steer_acc_deg=60.0):
    """Bike obeying the Stage-1 model driven by random steer accelerations (for filter tests)."""
    p = p or Params()
    rng = np.random.default_rng(seed)
    s = np.zeros(NS)
    s[IDX["psi"]], s[IDX["x"]], s[IDX["y"]], s[IDX["z"]], s[IDX["v"]] = psi0, 3.0, 5.0, 0.33, v0
    out = [s.copy()]
    dt = 1 / fps
    for t in range(1, T):
        s = f_step(s, dt, p)
        s[IDX["ddot"]] += np.deg2rad(steer_acc_deg) * np.sqrt(dt) * rng.standard_normal() - 2.0 * s[IDX["delta"]] * dt * 4
        s[IDX["v"]] += 0.6 * np.sqrt(dt) * rng.standard_normal()
        s[IDX["phi"]] += np.deg2rad(1.5) * np.sqrt(dt) * rng.standard_normal()
        out.append(s.copy())
    return np.array(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="bike_states.csv from sequence_fit.py")
    ap.add_argument("--out", default=None)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--wheelbase", type=float, default=1.1, help="hub-to-hub [m]; use the value sequence_fit calibrated")
    ap.add_argument("--lf2", type=float, default=0.1)
    ap.add_argument("--lean-tau", type=float, default=0.5, help="s; 0 = no roll<->turn coupling")
    ap.add_argument("--steer-acc-deg", type=float, default=400.0, help="process noise on the steer rate [deg/s^2/sqrt(s)]")
    ap.add_argument("--speed-acc", type=float, default=1.0, help="process noise on speed [m/s^2/sqrt(s)]")
    ap.add_argument("--yaw-noise-deg", type=float, default=1.0)
    ap.add_argument("--roll-noise-deg", type=float, default=4.0)
    ap.add_argument("--gate-chi2", type=float, default=24.3)
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args(argv)
    p = Params(lr=a.wheelbase - a.lf2, lf2=a.lf2, tau_lean=a.lean_tau, q_steer_acc_deg=a.steer_acc_deg,
               q_v=a.speed_acc, q_psi_deg=a.yaw_noise_deg, q_phi_deg=a.roll_noise_deg, gate_chi2=a.gate_chi2)
    ids, Z, SIG, valid = load_csv(a.csv)
    xs, Ps, xf, gs = smooth_csv(ids, Z, SIG, valid, a.fps, p)
    out = a.out or os.path.splitext(a.csv)[0] + "_kf.csv"
    write_out(out, ids, valid, xs, Ps, gs, p)
    print(f"[done] {valid.sum()}/{len(ids)} frames used, {int((gs > 1).sum())} down-weighted by the gate -> {out}")
    print(f"       speed: median {np.median(xs[:, IDX['v']]):.2f} m/s ({np.median(xs[:, IDX['v']]) * 3.6:.1f} km/h)")
    if a.plot:
        plot(os.path.splitext(out)[0] + ".png", ids, Z, valid, xs, Ps, a.fps)
    return 0


def plot(path, ids, Z, valid, xs, Ps, fps, truth=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = ids / fps
    fig, ax = plt.subplots(4, 2, figsize=(13, 9), sharex=True)
    names = ["roll [deg]", "pitch [deg]", "yaw [deg]", "steer [deg]", "x [m]", "depth y [m]", "z [m]", "speed [m/s]"]
    for k, a in enumerate(ax.flat):
        f = np.rad2deg if k < 4 else (lambda q: q)
        sg = np.sqrt(Ps[:, k, k])
        a.fill_between(t, f(xs[:, k] - 2 * sg), f(xs[:, k] + 2 * sg), color="C0", alpha=.2)
        a.plot(t, f(xs[:, k]), color="C0", label="Kalman smoother")
        if k < 7:
            a.plot(t[valid], f(Z[valid, k]) if k != 2 else np.rad2deg(np.unwrap(Z[valid, 2])), ".", ms=3, color="0.5", label="per-frame fit")
        if truth is not None:
            a.plot(t, f(truth[:, k]), "k--", lw=1, label="truth")
        a.set_ylabel(names[k]); a.grid(alpha=.3)
    ax.flat[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
