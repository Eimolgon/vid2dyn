#!/usr/bin/env python3
"""
3D animation of the bike pose from the CSV written by sequence_fit.py.

    python animate_csv.py states.csv --out bike.mp4                # smoothed states (default)
    python animate_csv.py states.csv --raw --out bike_raw.gif      # per-frame (unsmoothed) states
    python animate_csv.py states.csv --wheelbase 1.1 --wheel-diameter 0.66 --cam-height 0.45

Shows: rear/front rim, frame (hub -> head -> fork), a fading trail of the rear hub, the ground
plane, the camera with its field-of-view pyramid, and a live read-out of the states.
Geometry must match what you passed to sequence_fit.py (--wheelbase, --wheel-diameter,
--cam-height; the calibrated wheelbase is printed by sequence_fit.py).
Output: .mp4 (needs ffmpeg) or .gif (Pillow); falls back to gif if ffmpeg is missing.
Frames missing from the CSV timeline (valid=0) are drawn greyed-out (interpolated).
"""
import argparse
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation


def Rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def Ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def load_csv(path, raw=False):
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    suf = "_raw" if raw else ""
    g = lambda k: np.array([float(r[k]) for r in rows])
    X = np.column_stack([np.deg2rad(g(f"roll{suf}_deg")), np.deg2rad(g(f"pitch{suf}_deg")),
                         np.unwrap(np.deg2rad(g(f"yaw{suf}_deg"))), np.deg2rad(g(f"steer{suf}_deg")),
                         g(f"x{suf}_m"), g(f"y{suf}_m"), g(f"z{suf}_m")])
    return np.array([int(r["frame"]) for r in rows]), X, np.array([int(r["valid"]) for r in rows], bool)


def bike_geometry(x, bp, n=80):
    phi, theta, psi, delta = x[:4]
    R = Rz(psi) @ Rx(phi) @ Ry(theta)
    F = R @ Rz(delta)
    Cr = x[4:7]
    S = Cr + bp["lr"] * R[:, 0] + bp["lh"] * R[:, 2]
    Q = S - bp["lf1"] * F[:, 2]
    Cf = Q + bp["lf2"] * F[:, 0]
    t = np.linspace(0, 2 * np.pi, n)
    rim = lambda c, M, r: c + r * (np.cos(t)[:, None] * M[:, 0] + np.sin(t)[:, None] * M[:, 2])
    return {"Cr": Cr, "S": S, "Q": Q, "Cf": Cf, "rim_r": rim(Cr, R, bp["rr"]), "rim_f": rim(Cf, F, bp["rf"])}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--out", default="bike_anim.mp4")
    ap.add_argument("--raw", action="store_true", help="animate the per-frame (unsmoothed) columns")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--wheelbase", type=float, default=1.1)
    ap.add_argument("--lf2", type=float, default=0.1)
    ap.add_argument("--lf1", type=float, default=0.5)
    ap.add_argument("--wheel-diameter", type=float, default=0.6604)
    ap.add_argument("--cam-height", type=float, default=0.45)
    ap.add_argument("--hfov-deg", type=float, default=None, help="draw the camera FOV pyramid (e.g. 45.5)")
    ap.add_argument("--aspect", type=float, default=16 / 9)
    ap.add_argument("--view", nargs=2, type=float, default=(22, -62), metavar=("ELEV", "AZIM"))
    ap.add_argument("--follow", action="store_true", help="camera of the plot follows the bike")
    ap.add_argument("--trail", type=int, default=40, help="trail length in frames (0 = off)")
    ap.add_argument("--step", type=int, default=1, help="use every n-th frame")
    ap.add_argument("--dpi", type=int, default=100)
    a = ap.parse_args(argv)

    ids, X, valid = load_csv(a.csv, a.raw)
    ids, X, valid = ids[::a.step], X[::a.step], valid[::a.step]
    bp = {"lr": a.wheelbase - a.lf2, "lf1": a.lf1, "lf2": a.lf2, "lh": a.lf1,
          "rr": a.wheel_diameter / 2, "rf": a.wheel_diameter / 2}
    G = [bike_geometry(x, bp) for x in X]
    N = len(X)

    allp = np.vstack([np.vstack([g["rim_r"], g["rim_f"]]) for g in G])
    lo, hi = allp.min(0), allp.max(0)
    ctr = (lo + hi) / 2
    ZMAX = max(1.2, hi[2] + 0.3)
    half = max(1.5, (hi - lo).max() / 2 + 0.3)
    fig = plt.figure(figsize=(9, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("x (right) [m]"); ax.set_ylabel("y (depth) [m]"); ax.set_zlabel("z (up) [m]")
    try:
        ax.set_box_aspect((2 * half, 2 * half, ZMAX))
    except Exception:
        pass
    ax.view_init(*a.view)
    ax.set_title(("per-frame" if a.raw else "smoothed") + " bike pose")

    def set_limits(c):
        ax.set_xlim(c[0] - half, c[0] + half); ax.set_ylim(c[1] - half, c[1] + half)
        ax.set_zlim(0, ZMAX)

    set_limits(ctr)
    gx = np.array([[ctr[0] - 3 * half, ctr[0] + 3 * half]] * 2)
    gy = np.array([[ctr[1] - 3 * half] * 2, [ctr[1] + 3 * half] * 2])
    ax.plot_surface(gx, gy, np.zeros_like(gx), alpha=0.08, color="0.5")

    pc = np.array([0.0, 0.0, a.cam_height])
    ax.plot([pc[0]], [pc[1]], [pc[2]], "ks", ms=7)
    if a.hfov_deg:
        d = half
        w = d * np.tan(np.deg2rad(a.hfov_deg) / 2)
        h = w / a.aspect
        cs = [pc + np.array(c) for c in ((-w, d, -h), (w, d, -h), (w, d, h), (-w, d, h))]
        for c in cs:
            ax.plot(*zip(pc, c), color="0.4", lw=0.6)
        ax.plot(*zip(*(cs + [cs[0]])), color="0.4", lw=0.6)
    else:
        ax.plot(*zip(pc, pc + [0, half, 0]), color="g", lw=1)      # optical axis

    ln_r, = ax.plot([], [], [], color="C0", lw=2)
    ln_f, = ax.plot([], [], [], color="C1", lw=2)
    ln_fr, = ax.plot([], [], [], color="k", lw=2.5, marker="o", ms=3)
    ln_fk, = ax.plot([], [], [], color="C3", lw=2.5)
    ln_tr, = ax.plot([], [], [], color="C0", lw=1, alpha=0.5, ls=":")
    txt = fig.text(0.02, 0.97, "", va="top", family="monospace", fontsize=9)
    hub = np.array([g["Cr"] for g in G])

    def upd(i):
        g = G[i]
        col = "C0" if valid[i] else "0.6"
        ln_r.set_data_3d(*g["rim_r"].T); ln_r.set_color(col)
        ln_f.set_data_3d(*g["rim_f"].T); ln_f.set_color("C1" if valid[i] else "0.6")
        ln_fr.set_data_3d(*np.array([g["Cr"], g["S"]]).T)
        ln_fk.set_data_3d(*np.array([g["S"], g["Q"], g["Cf"]]).T)
        if a.trail:
            j = max(0, i - a.trail)
            ln_tr.set_data_3d(*hub[j:i + 1].T)
        if a.follow:
            set_limits(g["Cr"])
        x = X[i]
        txt.set_text(f"frame {ids[i]:5d}{'' if valid[i] else '  (no data: interpolated)'}\n"
                     f"roll {np.rad2deg(x[0]):6.1f}  pitch {np.rad2deg(x[1]):5.1f}\n"
                     f"yaw  {np.rad2deg(x[2]) % 360:6.1f}  steer {np.rad2deg(x[3]):5.1f}\n"
                     f"x {x[4]:5.2f}  depth {x[5]:5.2f}  z {x[6]:4.2f} m")
        return ln_r, ln_f, ln_fr, ln_fk, ln_tr, txt

    ani = animation.FuncAnimation(fig, upd, frames=N, interval=1000 / a.fps * a.step, blit=False)
    out = a.out
    if out.lower().endswith(".mp4") and not animation.writers.is_available("ffmpeg"):
        out = os.path.splitext(out)[0] + ".gif"
        print("[warn] ffmpeg not available -> writing a gif instead")
    writer = animation.FFMpegWriter(fps=a.fps / a.step) if out.lower().endswith(".mp4") else animation.PillowWriter(fps=min(25, a.fps / a.step))
    ani.save(out, writer=writer, dpi=a.dpi)
    print(f"[done] {N} frames -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
