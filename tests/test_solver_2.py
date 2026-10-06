# tests/test_solver_2.py

"""Tests for the least-squares solver (fit_img2model + residual_eqs).

Strategy: generate synthetic image data from a KNOWN state with the model
itself (generate_synthetic_data), then check that the solver gets it back.
Tests marked xfail(strict=True) document known bugs: they flip to a failure
("XPASS strict") the moment the bug is fixed, so you remember to remove the mark.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import src.bike_model_perspective as mdl
import src.utils as utils
from helpers import (pack, wheel_circle, project, ellipse_points_norm)
from conftest import IMAGE_RES

KEYS = ["r_Cf_Cr_u", "r_Cf_Cr_v", "u_Cr", "v_Cr",
        "r_P1r_Cr_u", "r_P1r_Cr_v", "r_P3r_Cr_u", "r_P3r_Cr_v",
        "r_P1f_Cf_u", "r_P1f_Cf_v", "r_P3f_Cf_u", "r_P3f_Cf_v",
        "r_Q_S_u", "r_Q_S_v"]


def state(deg4, pos):
    return np.r_[np.deg2rad(deg4), pos].astype(float)


TRUE_STATES = [
    pytest.param(state([-5, -10, 20, 5], [2, 8, 0]), id="baseline"),
    pytest.param(state([0, 0, 0, 0], [1, 6, 0]), id="upright_straight"),
    pytest.param(state([10, 5, -30, -8], [-1.5, 10, -0.2]), id="lean_right_steer_left"),
    pytest.param(state([-15, 0, 45, 10], [0.5, 5, 0.1]), id="strong_lean_close"),
]

# Initial-guess perturbation: ~2 deg on angles, 0.1/0.3/0.05 m on position
PERTURB = np.r_[np.deg2rad([2, 2, 2, 2]), 0.1, 0.3, 0.05]


# --------------------------------------------------------------- helpers
def synth(x, bike, cam):
    return utils.generate_synthetic_data(np.asarray(x), bike, cam, mdl)


def scalars(data):
    return {k: v[0] for k, v in data.items()}


def concat(frames):
    return {k: [f[k][0] for f in frames] for k in frames[0]}


def numeric_jacobian(fun, x, h=1e-6):
    f0 = fun(x)
    J = np.zeros((f0.size, x.size))
    for i in range(x.size):
        d = np.zeros_like(x)
        d[i] = h
        J[:, i] = (fun(x + d) - fun(x - d)) / (2 * h)
    return J


def assert_state_close(est, true, ang_tol=1e-3, pos_tol=5e-3):
    np.testing.assert_allclose(est[:4], true[:4], atol=ang_tol,
                               err_msg="angles [rad]")
    np.testing.assert_allclose(est[4:], true[4:], atol=pos_tol,
                               err_msg="position [m]")


# --------------------------------------------------------------- residuals
def test_synthetic_data_has_all_keys(bike_params, camera_params):
    data = synth(TRUE_STATES[0].values[0], bike_params, camera_params)
    assert sorted(data) == sorted(KEYS)
    assert all(np.isfinite(v[0]) for v in data.values())


@pytest.mark.parametrize("true", TRUE_STATES)
def test_residual_is_zero_at_truth(true, bike_params, camera_params):
    data = scalars(synth(true, bike_params, camera_params))
    r = mdl.residual_eqs(true, data, bike_params, camera_params)
    assert r.shape == (14,)
    np.testing.assert_allclose(r, 0.0, atol=1e-6)


def test_residual_is_nonzero_away_from_truth(bike_params, camera_params):
    true = TRUE_STATES[0].values[0]
    data = scalars(synth(true, bike_params, camera_params))
    r = mdl.residual_eqs(true + PERTURB, data, bike_params, camera_params)
    assert np.linalg.norm(r) > 1.0


def test_residual_depends_on_each_state_variable(bike_params, camera_params):
    """Every one of the 7 unknowns must influence at least one residual."""
    true = TRUE_STATES[0].values[0]
    data = scalars(synth(true, bike_params, camera_params))
    f = lambda x: mdl.residual_eqs(x, data, bike_params, camera_params)
    J = numeric_jacobian(f, true)
    assert np.all(np.linalg.norm(J, axis=0) > 1e-6)


# --------------------------------------------------------------- recovery
@pytest.mark.parametrize("true", TRUE_STATES)
def test_recovers_state_from_perturbed_guess(true, bike_params, camera_params, bounds):
    data = synth(true, bike_params, camera_params)
    _, hist = utils.fit_img2model(data, true + PERTURB, bike_params,
                                  bounds, camera_params, mdl)
    assert_state_close(hist[-1], true)


@pytest.mark.parametrize("true", TRUE_STATES)
def test_solver_reports_success_and_small_cost(true, bike_params, camera_params, bounds):
    data = synth(true, bike_params, camera_params)
    res, _ = utils.fit_img2model(data, true + PERTURB, bike_params,
                                 bounds, camera_params, mdl)
    assert res[0].success
    assert res[0].cost < 1e-6


@pytest.mark.parametrize("scale", [
    1,
    pytest.param(3, marks=pytest.mark.xfail(strict=False, reason="basin of convergence")),
    pytest.param(6, marks=pytest.mark.xfail(strict=False, reason="basin of convergence")),
])
def test_basin_of_convergence(scale, bike_params, camera_params, bounds):
    """Diagnostic: how far can the initial guess be from the truth?
    XFAIL rows tell you where the solver stops converging."""
    true = TRUE_STATES[0].values[0]
    data = synth(true, bike_params, camera_params)
    x0 = np.clip(true + scale * PERTURB, bounds[0], bounds[1])
    _, hist = utils.fit_img2model(data, x0, bike_params, bounds, camera_params, mdl)
    assert_state_close(hist[-1], true)


def test_zero_initial_guess_like_main(bike_params, camera_params, bounds):
    """main.py starts from angles=0 and depth=0.1 m. Documents what happens."""
    true = TRUE_STATES[0].values[0]
    data = synth(true, bike_params, camera_params)
    x0 = np.array([0, 0, 0, 0, 2.82, 0.6, 0.0])   # 0.1 m depth is infeasible here
    _, hist = utils.fit_img2model(data, x0, bike_params, bounds, camera_params, mdl)
    assert_state_close(hist[-1], true, ang_tol=1e-2, pos_tol=5e-2)


# --------------------------------------------------------------- robustness
def test_pixel_noise_robustness(bike_params, camera_params, bounds):
    true = TRUE_STATES[0].values[0]
    data = synth(true, bike_params, camera_params)
    rng = np.random.default_rng(0)
    noisy = {k: [v[0] + rng.normal(0, 0.5)] for k, v in data.items()}   # 0.5 px
    _, hist = utils.fit_img2model(noisy, true + PERTURB, bike_params,
                                  bounds, camera_params, mdl)
    est = hist[-1]
    assert np.all(np.abs(np.rad2deg(est[:4] - true[:4])) < 5.0)
    assert np.all(np.abs(est[4:] - true[4:]) < 0.2)


def test_solution_respects_bounds(bike_params, camera_params, bounds):
    true = TRUE_STATES[0].values[0].copy()
    true[4] = 25.0                                  # outside the [-20, 20] box
    data = synth(true, bike_params, camera_params)
    x0 = true.copy(); x0[4] = 10.0
    res, hist = utils.fit_img2model(data, x0, bike_params, bounds, camera_params, mdl)

    lo = np.asarray(bounds[0], dtype=float)
    hi = np.asarray(bounds[1], dtype=float)
    est = np.asarray(hist[-1], dtype=float)

    # assert np.all(hist[-1] >= bounds[0] - 1e-9)
    # assert np.all(hist[-1] <= bounds[1] + 1e-9)
    assert np.all(est >= lo - 1e-9)
    assert np.all(est <= hi + 1e-9)
    assert hist[-1][4] <= 20.0 + 1e-9


def test_infeasible_initial_guess_raises(bike_params, camera_params, bounds):
    true = TRUE_STATES[0].values[0]
    data = synth(true, bike_params, camera_params)
    x0 = true.copy(); x0[5] = 0.1                   # depth below lower bound 0.5
    with pytest.raises(ValueError):
        utils.fit_img2model(data, x0, bike_params, bounds, camera_params, mdl)


def test_problem_is_observable(bike_params, camera_params):
    """Column-normalised Jacobian at the truth must have full rank 7 and be
    reasonably conditioned, otherwise some state cannot be recovered."""
    true = TRUE_STATES[0].values[0]
    data = scalars(synth(true, bike_params, camera_params))
    f = lambda x: mdl.residual_eqs(x, data, bike_params, camera_params)
    J = numeric_jacobian(f, true)
    Jn = J / np.linalg.norm(J, axis=0)
    s = np.linalg.svd(Jn, compute_uv=False)
    assert np.linalg.matrix_rank(Jn, tol=1e-8 * s[0]) == 7
    assert s[-1] / s[0] > 1e-6, f"badly conditioned, singular values: {s}"


def test_residual_groups_have_comparable_scale(bike_params, camera_params):
    """A group with tiny Jacobian rows is ignored by the solver."""
    true = TRUE_STATES[0].values[0]
    data = scalars(synth(true, bike_params, camera_params))
    f = lambda x: mdl.residual_eqs(x, data, bike_params, camera_params)
    J = numeric_jacobian(f, true)
    rows = np.linalg.norm(J, axis=1)
    assert rows.max() / rows.min() < 1e3, f"row norms: {rows}"


# --------------------------------------------------------------- sequences
def test_multi_frame_tracking(bike_params, camera_params, bounds):
    base = TRUE_STATES[0].values[0]
    step = np.r_[np.deg2rad([0.5, 0.2, 1.0, 0.5]), 0.05, 0.03, 0.0]
    truth = [base + k * step for k in range(12)]
    data = concat([synth(t, bike_params, camera_params) for t in truth])

    results, hist = utils.fit_img2model(data, truth[0] + PERTURB, bike_params,
                                        bounds, camera_params, mdl)
    assert len(results) == 12
    # hist[0] is the initial guess, hist[1:] are the per-frame solutions
    assert len(hist) == 13
    for est, t in zip(hist[1:], truth):
        assert_state_close(est, t)


def test_warm_start_does_not_leak_between_frames(bike_params, camera_params, bounds):
    """Fitting frame k alone must give the same answer as inside a sequence."""
    a, b = TRUE_STATES[0].values[0], TRUE_STATES[2].values[0]
    seq = concat([synth(a, bike_params, camera_params),
                  synth(b, bike_params, camera_params)])
    _, hist_seq = utils.fit_img2model(seq, a + PERTURB, bike_params,
                                      bounds, camera_params, mdl)
    _, hist_b = utils.fit_img2model(concat([synth(b, bike_params, camera_params)]),
                                    b + PERTURB, bike_params, bounds, camera_params, mdl)
    np.testing.assert_allclose(hist_seq[-1], hist_b[-1], atol=5e-3)


# --------------------------------------------------------------- known bugs
@pytest.mark.xfail(strict=True, reason=(
    "main.py/utils.data4model build r_Q_S as a UNIT vector while the model "
    "returns u_Q - u_S in PIXELS. Normalise one of them."))
def test_qs_data_and_model_use_same_units(bike_params, camera_params):
    true = TRUE_STATES[0].values[0]
    d = synth(true, bike_params, camera_params)
    assert np.hypot(d["r_Q_S_u"][0], d["r_Q_S_v"][0]) == pytest.approx(1.0, rel=1e-6)


def test_ellipse_pipeline_matches_model_projection(bike_params, camera_params):
    """
    Render both wheels with the model -> normalised label points ->
    fitEllipse -> find_points, and compare with what the model predicts for
    centre / top / bottom points. Any large mismatch is model error that the
    solver cannot fix (perspective bias of ellipse centre and extremes).
    """
    true = TRUE_STATES[0].values[0]
    s = pack(true, bike_params, camera_params)
    W, H = IMAGE_RES
    expected = {                                      # pixel coordinates
        "rear": dict(c=(mdl.eval_f03(*s), mdl.eval_f04(*s)),
                     p1=(mdl.eval_f03(*s) + mdl.eval_f05(*s), mdl.eval_f04(*s) + mdl.eval_f06(*s)),
                     p3=(mdl.eval_f03(*s) + mdl.eval_f07(*s), mdl.eval_f04(*s) + mdl.eval_f08(*s))),
        "front": dict(c=(mdl.eval_f03(*s) + mdl.eval_f01(*s), mdl.eval_f04(*s) + mdl.eval_f02(*s)),
                      p1=(mdl.eval_f03(*s) + mdl.eval_f01(*s) + mdl.eval_f09(*s),
                          mdl.eval_f04(*s) + mdl.eval_f02(*s) + mdl.eval_f10(*s)),
                      p3=(mdl.eval_f03(*s) + mdl.eval_f01(*s) + mdl.eval_f11(*s),
                          mdl.eval_f04(*s) + mdl.eval_f02(*s) + mdl.eval_f12(*s))),
    }
    for wheel in ("rear", "front"):
        uv = project(wheel_circle(s, wheel), s)
        pts = np.column_stack([uv[:, 0] / W, 1 - uv[:, 1] / H])[::8]   # label format
        xc, yc, a, b, th = utils.fitEllipse(pts, (W, H))
        p1, _, p3, *_ = utils.find_points((xc, yc, a, b, th))
        e = expected[wheel]
        tol_c, tol_p = 0.01 * a, 0.05 * a
        assert np.hypot(xc - e["c"][0], yc - e["c"][1]) < tol_c, f"{wheel} centre"
        assert np.hypot(p1[0] - e["p1"][0], p1[1] - e["p1"][1]) < tol_p, f"{wheel} p1"
        assert np.hypot(p3[0] - e["p3"][0], p3[1] - e["p3"][1]) < tol_p, f"{wheel} p3"


def test_cli_synthetic_smoke():
    main = Path(__file__).resolve().parents[1] / "src" / "main.py"
    if not main.exists():
        pytest.skip("src/main.py not found")
    r = subprocess.run([sys.executable, str(main), "-d", "test"],
                       capture_output=True, text=True, timeout=600, cwd=main.parent)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "Estimated state" in r.stdout