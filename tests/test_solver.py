import numpy as np
import pytest
from importlib import import_module
import src.bike_model_perspective as perpro

from src.bike_model_perspective import (
    variables, eval_f01, eval_f02, eval_f03, eval_f04,
    eval_f05, eval_f06, eval_f07, eval_f08,
    eval_f09, eval_f10, eval_f11, eval_f12,
    eval_f13, eval_f14,
)
# Adjust to wherever fit_img2model lives:
from src.utils import fit_img2model


# ---------------------------------------------------------------- helpers

def build_subs(state, bike_params, camera_params):
    """Reproduce the ordering used by residual_eqs."""
    return (
        state[0], state[1], state[2], state[3],
        state[4], state[5], state[6],
        bike_params['lr'], bike_params['lf1'], bike_params['lf2'],
        bike_params['rr'], bike_params['rf'],
        camera_params['fx'], camera_params['fy'],
        camera_params['cx'], camera_params['cy'],
        camera_params['cam_x'], camera_params['cam_y'], camera_params['cam_z'],
        camera_params['cam_yaw'], camera_params['cam_pitch'], camera_params['cam_roll'],
    )


def make_synthetic_data(state_true, bike_params, camera_params):
    """Generate an `imgdata` dict from a known state using the model itself."""
    s = build_subs(state_true, bike_params, camera_params)
    keys = [
        ('r_Cf_Cr_u', eval_f01), ('r_Cf_Cr_v', eval_f02),
        ('u_Cr',      eval_f03), ('v_Cr',      eval_f04),
        ('r_P1r_Cr_u', eval_f05), ('r_P1r_Cr_v', eval_f06),
        ('r_P3r_Cr_u', eval_f07), ('r_P3r_Cr_v', eval_f08),
        ('r_P1f_Cf_u', eval_f09), ('r_P1f_Cf_v', eval_f10),
        ('r_P3f_Cf_u', eval_f11), ('r_P3f_Cf_v', eval_f12),
        ('r_Q_S_u',   eval_f13), ('r_Q_S_v',   eval_f14),
    ]
    return {k: np.array([float(f(*s))]) for k, f in keys}


def convergence_residual(data, state, bike_params, camera_params):
    r = perpro.residual_eqs(state, data, bike_params, camera_params)
    return float(np.linalg.norm(r))


def final_state(states):
    """Extract the final state whatever the shape of the returned list."""
    arr = np.asarray(states)
    # if it is (n_iter, 7), take last row; if it is (7,), take as is
    return arr[-1] if arr.ndim > 1 else arr

# ---------------------------------------------------------------- fixtures

@pytest.fixture
def bike_params():
    return dict(lr=0.5, lf1=0.1, lf2=1.0, rr=0.35, rf=0.35)


@pytest.fixture
def camera_params():
    return dict(
        fx=1000, fy=1000, cx=960, cy=540,
        cam_x=0.0, cam_y=-10.0, cam_z=1.0,
        cam_yaw=np.deg2rad(-90), cam_pitch=0.0, cam_roll=0.0,
    )


@pytest.fixture
def state_true():
    return np.array([
        np.deg2rad(5),    # phi
        np.deg2rad(10),   # theta
        np.deg2rad(-15),  # psi
        np.deg2rad(20),   # delta
        5.0, 2.0, 0.0,    # x_r, y_r, z_r
    ])


@pytest.fixture
def boundaries():
    # Adjust to match your real bounds
    lo = np.array([-np.pi/2, -np.pi/2, -np.pi, -np.pi/2, -1e6, -1e6, -1e6])
    hi = np.array([ np.pi/2,  np.pi/2,  np.pi,  np.pi/2,  1e6,  1e6,  1e6])
    return (lo, hi)


# ---------------------------------------------------------------- residual

def test_residual_zero_at_true_state(state_true, bike_params, camera_params):
    mod = import_module("src.bike_model_perspective")
    residual_eqs = mod.residual_eqs

    data = make_synthetic_data(state_true, bike_params, camera_params)
    r = residual_eqs(state_true, data, bike_params, camera_params)
    np.testing.assert_allclose(r, np.zeros((14,1)), atol=1e-9)


def test_residual_shape(state_true, bike_params, camera_params):
    mod = import_module("src.bike_model_perspective")
    residual_eqs = mod.residual_eqs

    data = make_synthetic_data(state_true, bike_params, camera_params)
    r = residual_eqs(state_true, data, bike_params, camera_params)
    assert r.shape == (14, 1)
    assert np.all(np.isfinite(r))


# ---------------------------------------------------------------- solver

def test_solver_recovers_synthetic_state(
    state_true, bike_params, camera_params, boundaries
):
    """Noise-free synthetic data -> solver recovers the true state."""
    data = make_synthetic_data(state_true, bike_params, camera_params)
    x0 = state_true + np.array([0.05, -0.05, 0.05, -0.05, 0.5, -0.5, 0.1])

    results, states = fit_img2model(
        data, x0, bike_params, boundaries, camera_params, model=perpro
    )
    est = final_state(states)
    np.testing.assert_allclose(states[-1], state_true, atol=1e-4)
    assert convergence_residual(data, est, bike_params, camera_params) < 1e-6


@pytest.mark.parametrize("seed", range(5))
def test_solver_multiple_seeds(
    seed, state_true, bike_params, camera_params, boundaries
):
    rng = np.random.default_rng(seed)
    data = make_synthetic_data(state_true, bike_params, camera_params)
    x0 = state_true + rng.normal(0, 0.1, size=7)

    results, states = fit_img2model(
        data, x0, bike_params, boundaries, camera_params, model=perpro
    )
    np.testing.assert_allclose(states[-1], state_true, atol=1e-3)


def test_solver_under_pixel_noise(
    state_true, bike_params, camera_params, boundaries
):
    rng = np.random.default_rng(0)
    data = make_synthetic_data(state_true, bike_params, camera_params)
    data = {k: v + rng.normal(0, 0.5, size=v.shape) for k, v in data.items()}
    x0 = state_true + np.array([0.05, -0.05, 0.05, -0.05, 0.5, -0.5, 0.1])

    results, states = fit_img2model(
        data, x0, bike_params, boundaries, camera_params, model=perpro
    )
    est = final_state(states)
    # Under pixel noise, allow a looser tolerance
    np.testing.assert_allclose(est, state_true, atol=0.2)


def test_solver_respects_boundaries(
    state_true, bike_params, camera_params, boundaries
):
    data = make_synthetic_data(state_true, bike_params, camera_params)
    x0 = state_true.copy()
    results, states = fit_img2model(
        data, x0, bike_params, boundaries, camera_params, model=perpro
    )
    lo, hi = boundaries
    assert np.all(states[-1] >= lo - 1e-9)
    assert np.all(states[-1] <= hi + 1e-9)


def test_solver_degenerate_scale(
    state_true, bike_params, camera_params, boundaries
):
    """
    Scale ambiguity: if fx and fy are also unknowns, many states fit the same
    image. Here we keep them fixed and just check the solver still works when
    the true state is far away.
    """
    far_state = state_true.copy()
    far_state[4] = 50.0   # x_r = 50 m
    far_state[5] = 20.0   # y_r = 20 m
    data = make_synthetic_data(far_state, bike_params, camera_params)
    x0 = state_true.copy()
    results, states = fit_img2model(
        data, x0, bike_params, boundaries, camera_params, model=perpro
    )
    np.testing.assert_allclose(states[-1], far_state, atol=1e-3)