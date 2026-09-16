import pytest
import matplotlib.pyplot as plt
from src.bike_model_perspective import *

idx_frame_roll  = 0
idx_frame_pitch = 1
idx_frame_yaw   = 2
idx_frame_steer = 3

idx_frame_x = 4
idx_frame_y = 5
idx_frame_z = 6

idx_lr  = 7
idx_lf1 = 8
idx_lf2 = 9

idx_rear_radius  = 10
idx_front_radius = 11

idx_fx = 12
idx_fy = 13
idx_cx = 14
idx_cy = 15

idx_cam_x = 16
idx_cam_y = 17
idx_cam_z = 18

idx_cam_yaw   = 19
idx_cam_pitch = 20
idx_cam_roll  = 21

idx_test_x = 22
idx_test_y = 23
idx_test_z = 24



def make_test_subs():
    """
    Create a default numerical substitution vector.
    """

    return [
        0, 0, 0, 0,       # phi, theta, psi, delta
        0, 0, 0,          # x_r, y_r, z_r
        0, 0, 0,          # lr, lf1, lf2
        0, 0,             # rr, rf
        0, 0, 0, 0,       # fx, fy, cx, cy
        0, 0, 0,          # cam_x, cam_y, cam_z
        0, 0, 0,          # yaw, pitch, roll
        0, 0, 0           # x_test, y_test, z_test
    ]


def set_camera_orientation(subs, yaw=0, pitch=0, roll=0):

    subs = list(subs)

    subs[idx_cam_yaw] = yaw
    subs[idx_cam_pitch] = pitch
    subs[idx_cam_roll] = roll

    return tuple(subs)


def evaluate_vector(vector, frame, subs):

    expression = vector.to_matrix(frame)

    evaluator = sp.lambdify(variables, expression)

    return np.array(evaluator(*subs), dtype=float).flatten()


def test_point_world():

    subs = make_test_subs()

    subs[idx_test_x] = 5
    subs[idx_test_y] = 7
    subs[idx_test_z] = 9
    subs = tuple(subs)

    position = testP.pos_from(O)

    eval_position = sp.lambdify(variables, position.to_matrix(N))

    result = np.array(eval_position(*subs), dtype=float).flatten()

    expected = np.array([5, 7,9])

    np.testing.assert_allclose(result, expected, atol=1e-12)


def test_point_rotated_frame():

    subs = make_test_subs()

    subs[idx_test_x] = 5
    subs[idx_test_y] = 0
    subs[idx_test_z] = 0

    position = testP.pos_from(O)

    position_in_N = position.to_matrix(N)

    eval_position = sp.lambdify(variables, position_in_N)

    result_N = np.array(eval_position(*subs),dtype=float).flatten()

    np.testing.assert_allclose(result_N, [5, 0, 0], atol=1e-12)


def test_rear_frame_zero_orientation():

    subs = make_test_subs()

    result_x = evaluate_vector(R.x, N, subs)
    result_y = evaluate_vector(R.y, N, subs)
    result_z = evaluate_vector(R.z, N, subs)

    np.testing.assert_allclose(result_x, [1, 0, 0])
    np.testing.assert_allclose(result_y, [0, 1, 0])
    np.testing.assert_allclose(result_z, [0, 0, 1])


def test_camera_orientation_zero():

    subs = make_test_subs()

    subs = set_camera_orientation(subs, yaw=0, pitch=0, roll=0)

    x = evaluate_vector(C.x, N, subs)
    y = evaluate_vector(C.y, N, subs)
    z = evaluate_vector(C.z, N, subs)

    np.testing.assert_allclose(x, [1, 0, 0])
    np.testing.assert_allclose(y, [0, 1, 0])
    np.testing.assert_allclose(z, [0, 0, 1])


# Check this one on the model and create the same for more rotations
def test_camera_yaw_90():

    subs = make_test_subs()

    subs = set_camera_orientation(subs, yaw=np.pi/2, pitch=0, roll=0)

    x = evaluate_vector(C.x, N, subs)
    y = evaluate_vector(C.y, N, subs)
    z = evaluate_vector(C.z, N, subs)

    np.testing.assert_allclose(x, [0, 1, 0], atol=1e-12)
    np.testing.assert_allclose(y, [-1, 0, 0], atol=1e-12)
    np.testing.assert_allclose(z, [0, 0, 1], atol=1e-12)


def test_camera_roll_90():

    subs = make_test_subs()

    subs = set_camera_orientation(subs, yaw=0, pitch=0, roll=np.pi/2)

    x = evaluate_vector(C.x, N, subs)
    y = evaluate_vector(C.y, N, subs)
    z = evaluate_vector(C.z, N, subs)

    np.testing.assert_allclose(x, [1, 0, 0], atol=1e-12)
    np.testing.assert_allclose(y, [0, 0, 1], atol=1e-12)
    np.testing.assert_allclose(z, [0, -1, 0], atol=1e-12)    


def test_camera_pitch_90():

    subs = make_test_subs()

    subs = set_camera_orientation(subs, yaw=0, pitch=np.pi/2, roll=0)

    x = evaluate_vector(C.x, N, subs)
    y = evaluate_vector(C.y, N, subs)
    z = evaluate_vector(C.z, N, subs)

    np.testing.assert_allclose(x, [0, 0, -1], atol=1e-12)
    np.testing.assert_allclose(y, [0, 1, 0], atol=1e-12)
    np.testing.assert_allclose(z, [1, 0, 0], atol=1e-12) 


def test_camera_translation():

    subs = make_test_subs()

    subs[idx_cam_x] = 100
    subs[idx_cam_y] = 200
    subs[idx_cam_z] = 300

    position = P.pos_from(O)

    result = evaluate_vector(position, N, subs)

    np.testing.assert_allclose(result, [100, 200, 300], atol=1e-12)


@pytest.mark.parametrize('radius', [100, 250, 300, 350, 500])
def test_wheel_radius(radius):

    subs = make_test_subs()

    subs[idx_rear_radius] = radius 

    vector = P1r.pos_from(Cr)
    magnitude = sp.sqrt(vector.dot(vector))
    evaluator = sp.lambdify(variables, magnitude)
    result = float(evaluator(*subs))

    np.testing.assert_allclose(result, radius, atol=1e-12)


def test_rear_wheel_radius_symbolic():

    vector = P1r.pos_from(Cr)

    radius_squared = sp.simplify(vector.dot(vector))

    print("P1r-Cr =", vector)
    print("|P1r-Cr|² =", radius_squared)

    assert sp.simplify(radius_squared - rr**2) == 0


def test_rear_wheel_plane():

    for point in [P1r, P2r, P3r, P4r]:

        vector = point.pos_from(Cr)

        result = sp.simplify(vector.dot(R.y))

        assert result == 0


def test_front_wheel_plane():

    for point in [P1f, P2f, P3f, P4f]:

        vector = point.pos_from(Cf)

        result = sp.simplify(vector.dot(F.y))

        assert result == 0


def test_rear_wheel_position():

    subs = make_test_subs()

    subs[idx_frame_x] = 100
    subs[idx_frame_y] = 200
    subs[idx_frame_z] = 300

    position = Cr.pos_from(O)

    result = evaluate_vector(position, N, subs)

    np.testing.assert_allclose(result, [100, 200, 300], atol=1e-12)


def test_steering_distance():

    subs = make_test_subs()

    subs[idx_lr] = 500

    vector = S.pos_from(Cr)
    magnitude = vector.magnitude()
    evaluator = sp.lambdify(variables, magnitude)
    result = float(evaluator(*subs))

    np.testing.assert_allclose(result, 500, atol=1e-12)


def test_fork_distance():

    subs = make_test_subs()

    subs[idx_lf1] = 100

    vector = Q.pos_from(S)
    magnitude = vector.magnitude()
    evaluator = sp.lambdify(variables, magnitude)
    result = float(evaluator(*subs))

    np.testing.assert_allclose(result, 100, atol=1e-12)


def test_trail_distance():

    subs = make_test_subs()

    subs[9] = 1000

    vector = Cf.pos_from(Q)
    magnitude = vector.magnitude()
    evaluator = sp.lambdify(variables, magnitude)
    result = float(evaluator(*subs))

    np.testing.assert_allclose(result, 1000, atol=1e-12)


def test_projection():

    subs = make_test_subs()

    subs[idx_test_x] = 2
    subs[idx_test_y] = 10
    subs[idx_test_z] = 3

    subs[idx_fx] = 1000     # fx
    subs[idx_fy] = 1000     # fy
    subs[idx_cx] = 500      # cx
    subs[idx_cy] = 400      # cy

    u, v = perspective_projection(testP, P, C, fx, fy, cx, cy)

    eval_u = sp.lambdify(variables, u)
    eval_v = sp.lambdify(variables, v)

    result_u = float(eval_u(*subs))
    result_v = float(eval_v(*subs))

    np.testing.assert_allclose(result_u, 700, atol=1e-12)

    np.testing.assert_allclose(result_v, 700, atol=1e-12)


@pytest.mark.parametrize('depth', [1, 5, 10, 20, 40])
def test_projection_depth(depth):

    subs = make_test_subs()

    subs[idx_test_x] = 2
    subs[idx_test_y] = depth
    subs[idx_test_z] = 3

    subs[idx_fx] = 1000
    subs[idx_fy] = 1000
    subs[idx_cx] = 500
    subs[idx_cy] = 400

    u, v = perspective_projection(testP, P, C, fx, fy, cx, cy)

    eval_u = sp.lambdify(variables, u)
    eval_v = sp.lambdify(variables, v)

    result_u = float(eval_u(*subs))
    result_v = float(eval_v(*subs))

    expected_u = 1000 * 2 / depth + 500
    expected_v = 1000 * 3 / depth + 400

    np.testing.assert_allclose(result_u, expected_u, atol=1e-12)

    np.testing.assert_allclose(result_v, expected_v, atol=1e-12)


# ----- Don't remove any of these lines -----
x_test, y_test, z_test = sp.symbols('x_t, y_t, z_t')

testP = me.Point('P_t')
testP.set_pos(O, x_test*N.x + y_test*N.y + z_test*N.z)

variables = (phi, theta, psi, delta,
             x_r, y_r, z_r,
             lr, lf1, lf2,
             rr, rf,
             fx, fy, cx, cy,
             cam_x, cam_y, cam_z,
             cam_yaw, cam_pitch, cam_roll,
             x_test, y_test, z_test)