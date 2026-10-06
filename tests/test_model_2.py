# test_model_2.py

import numpy as np
import pytest

import src.bike_model_perspective as mdl
from helpers import pack, vec, wheel_circle, project


def _cam(camera_params, **kw):
    return dict(camera_params, **kw)


@pytest.mark.parametrize("seed", range(5))
def test_frames_are_orthonormal_for_random_angles(seed, bike_params, camera_params):
    rng = np.random.default_rng(seed)
    state = np.r_[rng.uniform(-1, 1, 4), rng.uniform(-3, 3, 3)]
    cam = _cam(camera_params, cam_yaw=rng.uniform(-1, 1),
               cam_pitch=rng.uniform(-1, 1), cam_roll=rng.uniform(-1, 1))
    s = pack(state, bike_params, cam)
    for names in (("Rx", "Ry", "Rz"), ("Fx", "Fy", "Fz"), ("Cx", "Cy", "Cz")):
        M = np.column_stack([vec(n, s) for n in names])
        np.testing.assert_allclose(M.T @ M, np.eye(3), atol=1e-12)
        assert np.linalg.det(M) == pytest.approx(1.0)


def test_image_features_invariant_to_common_translation(bike_params, camera_params):
    """Moving bike and camera by the same vector must not change the image."""
    state = np.r_[np.deg2rad([-5, -10, 20, 5]), 2.0, 8.0, 0.3]
    cam = _cam(camera_params, cam_x=0.4, cam_y=-0.2, cam_z=0.5)
    d = np.array([3.0, -2.0, 1.0])
    state2 = state.copy(); state2[4:] += d
    cam2 = _cam(cam, cam_x=cam["cam_x"] + d[0], cam_y=cam["cam_y"] + d[1],
                cam_z=cam["cam_z"] + d[2])
    s1, s2 = pack(state, bike_params, cam), pack(state2, bike_params, cam2)
    for k in range(1, 15):
        f = getattr(mdl, f"eval_f{k:02d}")
        assert f(*s1) == pytest.approx(f(*s2), abs=1e-6), f"f{k:02d}"


@pytest.mark.parametrize("state", [
    np.r_[np.deg2rad([0, 0, 0, 0]), 1, 6, 0],
    np.r_[np.deg2rad([-5, -10, 20, 5]), 2, 8, 0],
])
def test_p1_p3_are_highest_and_lowest_rim_points(state, bike_params, camera_params):
    s = pack(state, bike_params, camera_params)
    for wheel, p1, p3 in (("rear", "P1r", "P3r"), ("front", "P1f", "P3f")):
        z = wheel_circle(s, wheel, n=7200)[:, 2]
        assert vec(p1, s)[2] == pytest.approx(z.max(), abs=1e-3)
        assert vec(p3, s)[2] == pytest.approx(z.min(), abs=1e-3)


def test_wheel_rim_projects_to_closed_curve_in_front_of_camera(bike_params, camera_params):
    state = np.r_[np.deg2rad([-5, -10, 20, 5]), 2.0, 8.0, 0.0]
    s = pack(state, bike_params, camera_params)
    for wheel in ("rear", "front"):
        pts = wheel_circle(s, wheel)
        assert np.all((pts - vec("P", s)) @ vec("Cy", s) > 0), "rim behind camera"
        assert np.all(np.isfinite(project(pts, s)))


# @pytest.mark.xfail(strict=True, reason=(
#     "cam_pitch rotates about C.y, which is the OPTICAL AXIS, so it behaves like "
#     "a camera roll. A tilt up/down needs a rotation about C.x."))
# def test_camera_pitch_tilts_image_vertically(bike_params, camera_params):
#     cam = _cam(camera_params, cam_pitch=np.deg2rad(10))
#     # cam = _cam(camera_params, cam_roll=np.deg2rad(10))
#     s = pack(np.zeros(7), bike_params, cam)
#     uv = project(np.array([[0.0, 10.0, 0.0]]), s)[0]      # point straight ahead
#     assert abs(uv[1] - camera_params["cy"]) == pytest.approx(
#         camera_params["fy"] * np.tan(np.deg2rad(10)), rel=1e-6)


# def test_camera_roll_currently_acts_as_pitch(bike_params, camera_params):
#     """Documents the swapped naming (see test above)."""
#     cam = _cam(camera_params, cam_roll=np.deg2rad(10))
#     s = pack(np.zeros(7), bike_params, cam)
#     uv = project(np.array([[0.0, 10.0, 0.0]]), s)[0]
#     assert abs(uv[1] - camera_params["cy"]) == pytest.approx(
#         camera_params["fy"] * np.tan(np.deg2rad(10)), rel=1e-6)


# @pytest.mark.xfail(strict=True, reason=(
#     "Bounds in main.py allow roll = +-pi/2 where R.y || N.z and the normalize() "
#     "in P1r..P4r divides by ~0: the rim points become numerically unstable."))
# def test_rim_points_are_stable_near_roll_90deg(bike_params, camera_params):
#     def p1(roll):
#         s = pack([roll, 0, 0, 0, 0, 5, 0], bike_params, camera_params)
#         return vec("P1r", s) - vec("Cr", s)
#     r0 = np.pi / 2 - 1e-9
#     assert np.linalg.norm(p1(r0) - p1(r0 + 1e-12)) < 1e-6