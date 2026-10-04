"""Tests for the image-processing side (utils.py)."""
import inspect
import json

import numpy as np
import pytest

import src.utils as utils
from helpers import ellipse_points_norm, angle_diff_mod_pi

RES = (1920, 1080)


def implicit(p, e):
    """
    (x'/a)^2 + (z'/b)^2 for point p on ellipse e=(x,z,a,b,theta); 1 == on it.
    """
    x, z, a, b, th = e
    dx, dz = p[0] - x, p[1] - z
    xp = dx * np.cos(th) + dz * np.sin(th)
    zp = -dx * np.sin(th) + dz * np.cos(th)
    return (xp / a) ** 2 + (zp / b) ** 2


# ------------------------------------------------------------------ readFile
def test_readFile_splits_classes(tmp_path):
    f = tmp_path / "frame_000001.txt"
    f.write_text("0 0.1 0.2 0.3 0.4 0.5 0.6\n1 0.7 0.8 0.9 1.0\n")
    out = utils.readFile(str(f))
    assert [o["class_id"] for o in out] == [0, 1]
    assert out[0]["points"].shape == (3, 2)
    assert out[1]["points"].shape == (2, 2)
    np.testing.assert_allclose(out[0]["points"][0], [0.1, 0.2])


# ---------------------------------------------------------------- fitEllipse
@pytest.mark.parametrize("xc,yc,a,b,th", [
    (900, 500, 200, 80, np.deg2rad(30)),
    (500, 700, 150, 120, np.deg2rad(-60)),
    (1200, 300, 90, 30, np.deg2rad(5)),
    (960, 540, 300, 100, 0.0),
])
def test_fitEllipse_recovers_parameters(xc, yc, a, b, th):
    pts = ellipse_points_norm(xc, yc, a, b, th, RES)
    fx, fy, fa, fb, fth = utils.fitEllipse(pts, RES)
    assert (fx, fy) == pytest.approx((xc, yc), abs=1e-4)
    assert (fa, fb) == pytest.approx((a, b), rel=1e-6)
    assert abs(angle_diff_mod_pi(fth, th)) < 1e-6


def test_fitEllipse_with_pixel_noise():
    pts = ellipse_points_norm(900, 500, 200, 80, 0.5, RES, n=60, noise=1.0)
    fx, fy, fa, fb, fth = utils.fitEllipse(pts, RES)
    assert (fx, fy) == pytest.approx((900, 500), abs=3)
    assert (fa, fb) == pytest.approx((200, 80), abs=4)
    assert abs(angle_diff_mod_pi(fth, 0.5)) < np.deg2rad(2)


def test_fitEllipse_major_axis_is_first_and_theta_range():
    fx, fy, fa, fb, fth = utils.fitEllipse(
        ellipse_points_norm(900, 500, 60, 180, 0.3, RES), RES)   # b > a on input
    assert fa >= fb
    assert -np.pi / 2 <= fth <= np.pi / 2


def test_fitEllipse_degenerate_input_does_not_return_None_silently():
    """Collinear points -> singular scatter matrix. Callers unpack the result,
    so returning None crashes later. Prefer raising a clear error."""
    pts = np.column_stack([np.linspace(0.1, 0.9, 10), np.full(10, 0.5)])
    try:
        out = utils.fitEllipse(pts, RES)
    except Exception:
        return
    assert out is not None


# --------------------------------------------------------------- find_points
ELLIPSES = [(900, 500, 200, 80, 0.5), (900, 500, 200, 80, -0.9),
            (400, 300, 100, 100, 0.0), (700, 200, 150, 40, 0.0)]


@pytest.mark.parametrize("e", ELLIPSES)
@pytest.mark.parametrize("direction", ["right", "left"])
def test_find_points_lie_on_ellipse(e, direction):
    for p in utils.find_points(e, direction):
        assert implicit(p, e) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("e", ELLIPSES)
def test_find_points_p1_top_p3_bottom_symmetric(e):
    x, z, a, b, th = e
    p1, p2, p3, p4, *_ = utils.find_points(e, "right")
    d_z = np.sqrt(a**2 * np.sin(th)**2 + b**2 * np.cos(th)**2)
    assert p1[1] == pytest.approx(z + d_z)
    assert p3[1] == pytest.approx(z - d_z)
    assert (p1[0] + p3[0]) / 2 == pytest.approx(x)
    assert (p2[0] + p4[0]) / 2 == pytest.approx(x)


@pytest.mark.parametrize("e", ELLIPSES)
def test_find_points_direction_swaps_p2_p4(e):
    r = utils.find_points(e, "right")
    l = utils.find_points(e, "left")
    assert r[1] == pytest.approx(l[3]) and r[3] == pytest.approx(l[1])
    assert r[0] == pytest.approx(l[0]) and r[2] == pytest.approx(l[2])
    assert r[1][0] < e[0] < r[3][0]          # 'right': p2 left of centre


def test_find_points_bad_direction_is_reported():
    """Anything other than 'left'/'right' currently raises UnboundLocalError."""
    with pytest.raises(Exception):
        utils.find_points((900, 500, 200, 80, 0.5), "up")


# ---------------------------------------------------------- process_directory
def _write_frames(dirpath, n=3, first=400):
    for i in range(n):
        rear = ellipse_points_norm(600 + 10 * i, 700, 150, 120, 0.1, RES)
        front = ellipse_points_norm(1200 + 10 * i, 700, 150, 130, 0.0, RES)
        lines = [f"1 " + " ".join(f"{v:.8f}" for v in rear.ravel()),
                 f"0 " + " ".join(f"{v:.8f}" for v in front.ravel())]
        (dirpath / f"frame_{first + i:06d}.txt").write_text("\n".join(lines) + "\n")


@pytest.mark.xfail(strict=True, raises=TypeError, reason=(
    "process_directory() calls find_points(fitted_ellipse) without the "
    "required `direction` argument"))
def test_process_directory(tmp_path):
    _write_frames(tmp_path)
    td = utils.process_directory(str(tmp_path), RES)
    assert td["first_frame"] == 400
    assert len(td[0]["ellipses"]) == 3 and len(td[1]["ellipses"]) == 3


# ------------------------------------------------------------------- filters
def test_lowpass_keeps_slow_signal_and_removes_fast_noise():
    fs = 30.0
    t = np.arange(300) / fs
    slow = np.sin(2 * np.pi * 0.2 * t)
    y = utils.lowpass_filter(slow + 0.5 * np.sin(2 * np.pi * 10 * t), cutoff=2.0, fs=fs)
    assert np.sqrt(np.mean((y - slow) ** 2)) < 0.05


def _tracking(n=60):
    rng = np.random.default_rng(0)
    t = np.arange(n)
    base = lambda x0: [(x0 + 2 * k + rng.normal(0, 2), 500 + rng.normal(0, 2), 100, 80, 0.1)
                       for k in t]
    return {0: {"ellipses": base(1200), "frames": list(t), "points": [], "extremes": []},
            1: {"ellipses": base(600), "frames": list(t), "points": [], "extremes": []},
            "first_frame": 400}


def test_apply_filters_smooths_centres():
    td = _tracking()
    raw = np.array(td[0]["ellipses"])[:, 0]
    out = utils.apply_filters(td)
    ideal = 1200 + 2 * np.arange(60)
    assert np.std(out[0]["x"] - ideal) < np.std(raw - ideal)


@pytest.mark.xfail(strict=True, reason=(
    "apply_filters() returns only {'x','z','frames'}: it drops 'ellipses', "
    "'extremes' and 'first_frame', which data4model/animate_* and main.py need"))
def test_apply_filters_output_is_usable_downstream():
    out = utils.apply_filters(_tracking())
    assert "first_frame" in out
    assert "ellipses" in out[0] and "extremes" in out[0]


# --------------------------------------------------------------- data4model
def _tracking_with_extremes(n=3):
    front, rear = [], []
    for k in range(n):
        rear.append((600 + 10 * k, 500, 150, 120, 0.1))
        front.append((1200 + 10 * k, 520, 150, 130, 0.0))
    ext = lambda es: [utils.find_points(e, "right") for e in es]
    return {0: {"ellipses": front, "extremes": ext(front)},
            1: {"ellipses": rear, "extremes": ext(rear)}}


def test_data4model_centre_terms():
    td = _tracking_with_extremes()
    data, ell_r, ell_f = utils.data4model(td, (0, 0))
    xr = np.array([e[0] for e in td[1]["ellipses"]])
    xf = np.array([e[0] for e in td[0]["ellipses"]])
    np.testing.assert_allclose(data["u_Cr"], xr)
    np.testing.assert_allclose(data["r_Cf_Cr_u"], xf - xr)
    json.dumps(data)                                  # must be JSON-serialisable


@pytest.mark.xfail(strict=True, reason=(
    "data4model() swaps wheels: r_P*r_* use the FRONT extremes and r_P*f_* "
    "use the REAR extremes"))
def test_data4model_wheel_points_belong_to_the_right_wheel():
    td = _tracking_with_extremes()
    data, _, _ = utils.data4model(td, (0, 0))
    p1_rear = np.array([e[0] for e in td[1]["extremes"]])     # (n, 2)
    xr = np.array([e[0] for e in td[1]["ellipses"]])
    np.testing.assert_allclose(data["r_P1r_Cr_u"], p1_rear[:, 0] - xr)


# ------------------------------------------------------------------ misc
def test_clean4json():
    out = utils.clean4json({"a": np.array([1.0, np.nan, np.inf]),
                            "b": np.float32(2.5), "c": (np.int64(3),)})
    assert out == {"a": [1.0, 0, 0], "b": 2.5, "c": [3]}
    json.dumps(out)


def test_pt2circle_points_lie_on_circle():
    c, r = np.array([1.0, 2.0, 3.0]), 0.5
    p = [c + r * v for v in ([0, 0, 1], [1, 0, 0], [0, 0, -1], [-1, 0, 0])]
    cx, cy, cz = utils.pt2circle(*p)
    pts = np.column_stack([cx, cy, cz])
    np.testing.assert_allclose(np.linalg.norm(pts - c, axis=1), r, atol=1e-12)
    np.testing.assert_allclose(cy, 2.0, atol=1e-12)


@pytest.mark.xfail(strict=True, reason="animate_Point has `save`, main.py passes `name=`")
def test_animate_functions_accept_the_arguments_main_passes():
    """main.py calls animate_Point(..., name=args.save)."""
    assert "name" in inspect.signature(utils.animate_Point).parameters


@pytest.mark.xfail(strict=True, reason="`date` is only defined in main.py, not in utils")
def test_utils_defines_everything_it_uses():
    """animate_Point uses a global `date` that is never defined in utils."""
    assert hasattr(utils, "date")