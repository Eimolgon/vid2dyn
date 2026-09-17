import pytest
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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


def test_front_wheel_radius_symbolic():

    for point in [P1f, P2f, P3f, P4f]:

        vector = point.pos_from(Cf)

        radius_squared = sp.simplify(
            vector.dot(vector) - rf**2
        )

        assert radius_squared == 0


def test_rear_wheel_opposite_points():

    v1 = P1r.pos_from(Cr)
    v3 = P3r.pos_from(Cr)

    assert sp.simplify(v1 + v3) == sp.zeros(3, 1)


def test_front_wheel_opposite_points():
    # Check this function
    v1 = P1f.pos_from(Cf)
    v3 = P3f.pos_from(Cf)

    assert sp.simplify(v1 + v3) == sp.zeros(3, 1)


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


@pytest.mark.parametrize(
    "phi,theta,psi,delta",
    [
        (0, 0, 0, 0),
        (0.1, 0.0, 0.0, 0),
        (0.0, 0.2, 0.0, 0),
        (0.0, 0.0, 0.3, 0),
        (0.1, 0.2, 0.3, 0.15),
        (-0.2, 0.3, -0.4, -0.2),
    ],
)
def test_bicycle_distances_under_rotation(phi, theta, psi, delta):
    subs = make_test_subs()

    subs[idx_frame_roll] = phi
    subs[idx_frame_pitch] = theta
    subs[idx_frame_yaw] = psi
    subs[idx_frame_steer] = delta

    subs[idx_lr] = 500
    subs[idx_lf1] = 100
    subs[idx_lf2] = 1000

    evaluator_lr = sp.lambdify(variables, S.pos_from(Cr).magnitude())
    evaluator_lf1 = sp.lambdify(variables, Q.pos_from(S).magnitude())
    evaluator_lf2 = sp.lambdify(variables, Cf.pos_from(Q).magnitude())

    np.testing.assert_allclose(evaluator_lr(*subs), 500, atol=1e-12)
    np.testing.assert_allclose(evaluator_lf1(*subs), 100, atol=1e-12)
    np.testing.assert_allclose(evaluator_lf2(*subs), 1000, atol=1e-12)


def test_front_frame_zero_steering():

    subs = make_test_subs()

    x = evaluate_vector(F.x, N, subs)
    y = evaluate_vector(F.y, N, subs)
    z = evaluate_vector(F.z, N, subs)

    np.testing.assert_allclose(x, [1, 0, 0], atol=1e-12)
    np.testing.assert_allclose(y, [0, 1, 0], atol=1e-12)
    np.testing.assert_allclose(z, [0, 0, 1], atol=1e-12)


def test_front_frame_steering_90():

    subs = make_test_subs()
    subs[idx_frame_steer] = np.pi / 2

    x = evaluate_vector(F.x, N, subs)
    y = evaluate_vector(F.y, N, subs)
    z = evaluate_vector(F.z, N, subs)

    np.testing.assert_allclose(x, [0, 1, 0], atol=1e-12)
    np.testing.assert_allclose(y, [-1, 0, 0], atol=1e-12)
    np.testing.assert_allclose(z, [0, 0, 1], atol=1e-12)


def test_front_frame_steering_45():

    subs = make_test_subs()
    subs[idx_frame_steer] = np.pi / 4

    x = evaluate_vector(F.x, N, subs)

    expected = [np.sqrt(2) / 2, np.sqrt(2) / 2, 0]

    np.testing.assert_allclose(x, expected, atol=1e-12)


def test_rear_to_steering_direction():

    subs = make_test_subs()
    subs[idx_lr] = 500

    vector = S.pos_from(Cr)

    result = evaluate_vector(vector, N, subs)

    np.testing.assert_allclose(result, [500, 0, 0], atol=1e-12)


def test_steering_to_trail_direction():

    subs = make_test_subs()
    subs[idx_lf1] = 100

    vector = Q.pos_from(S)

    result = evaluate_vector(vector, N, subs)

    np.testing.assert_allclose(result, [0, 0, -100], atol=1e-12)


def test_trail_to_front_direction():

    subs = make_test_subs()
    subs[idx_lf2] = 1000

    vector = Cf.pos_from(Q)

    result = evaluate_vector(vector, N, subs)

    np.testing.assert_allclose(result, [1000, 0, 0], atol=1e-12)


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


def test_projection_optical_axis():

    subs = make_test_subs()

    subs[idx_test_x] = 0
    subs[idx_test_y] = 10
    subs[idx_test_z] = 0

    subs[idx_fx] = 1000
    subs[idx_fy] = 1000
    subs[idx_cx] = 500
    subs[idx_cy] = 400

    u, v = perspective_projection(testP, P, C, fx, fy, cx, cy)

    eval_u = sp.lambdify(variables, u)
    eval_v = sp.lambdify(variables, v)

    np.testing.assert_allclose(eval_u(*subs), 500, atol=1e-12)
    np.testing.assert_allclose(eval_v(*subs), 400, atol=1e-12)


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


def visualize_square_projection():

    x_t2, y_t2, z_t2, ls = sp.symbols('x_t2, y_t2, z_t2, ls')
    x_t3, y_t3, z_t3 = sp.symbols('x_t3, y_t3, z_t3')
    q1_t, q2_t, q3_t = sp.symbols('q1_t, q2_t, q3_t')
    s1, s2, s3, s4 = sp.symbols('s1, s2, s3, s4', cls=me.Point)
    t1, t2, t3, t4 = sp.symbols('t1, t2, t3, t4', cls=me.Point)
    
    square_vars = (ls, q1_t, q2_t, q3_t,
                    x_t2, y_t2, z_t2,
                    x_t3, y_t3, z_t3,
                    cam_yaw, cam_pitch, cam_roll,
                    cam_x, cam_y, cam_z)
    
    A = me.ReferenceFrame('A')
    
    A.orient_body_fixed(N, (q3_t, q2_t, q1_t), 'zyx')
    
    s1.set_pos(O, x_t2*N.x + y_t2*N.y + z_t2*N.z)
    s2.set_pos(s1, ls*A.x)
    s3.set_pos(s2, ls*A.z)
    s4.set_pos(s1, ls*A.z)
    
    t1.set_pos(O, x_t3*N.x + y_t3*N.y + z_t3*N.z)
    t2.set_pos(t1, ls*A.x)
    t3.set_pos(t2, ls*A.z)
    t4.set_pos(t1, ls*A.z)

    subs = make_test_subs()

    # ----- camera intrinsics -----
    fx = 28
    fy = fx
    cx = 500
    cy = cx

    square_values = {
        ls: 2,
        q1_t: 0,
        q2_t: 0,
        q3_t: -np.pi/2,

        x_t2: 5,
        y_t2: -5,
        z_t2: 1,

        x_t3: 10,
        y_t3: -10,
        z_t3: 1,

        cam_yaw: np.deg2rad(-90),
        cam_pitch: np.deg2rad(0),
        cam_roll: np.deg2rad(0),

        cam_x: 0,
        cam_y: -10,
        cam_z: 0
    }


    us1, vs1 = perspective_projection(s1, P, C, fx, fy, cx, cy)
    us2, vs2 = perspective_projection(s2, P, C, fx, fy, cx, cy)
    us3, vs3 = perspective_projection(s3, P, C, fx, fy, cx, cy)
    us4, vs4 = perspective_projection(s4, P, C, fx, fy, cx, cy)

    ut1, vt1 = perspective_projection(t1, P, C, fx, fy, cx, cy)
    ut2, vt2 = perspective_projection(t2, P, C, fx, fy, cx, cy)
    ut3, vt3 = perspective_projection(t3, P, C, fx, fy, cx, cy)
    ut4, vt4 = perspective_projection(t4, P, C, fx, fy, cx, cy)


    projections = [
        (us1, vs1),
        (us2, vs2),
        (us3, vs3),
        (us4, vs4),
        (ut1, vt1),
        (ut2, vt2),
        (ut3, vt3),
        (ut4, vt4),
    ]

    eval_projections = [
        (
            sp.lambdify(square_vars, u),
            sp.lambdify(square_vars, v)
        )
        for u, v in projections
    ]

    values = [
        square_values[ls],
        square_values[q1_t],
        square_values[q2_t],
        square_values[q3_t],
        square_values[x_t2],
        square_values[y_t2],
        square_values[z_t2],
        square_values[x_t3],
        square_values[y_t3],
        square_values[z_t3],
        square_values[cam_yaw],
        square_values[cam_pitch],
        square_values[cam_roll],
        square_values[cam_x],
        square_values[cam_y],
        square_values[cam_z]
    ]

    projected_points = [
        (float(eval_u(*values)), float(eval_v(*values)))
        for eval_u, eval_v in eval_projections
    ]

    s_points = projected_points[:4]
    t_points = projected_points[4:]


    fig, ax = plt.subplots()

    s_x, s_y = zip(*(s_points + [s_points[0]]))
    t_x, t_y = zip(*(t_points + [t_points[0]]))

    ax.plot(s_x, s_y, marker='o', label='s')
    ax.plot(t_x, t_y, marker='o', label='t')

    ax.set_aspect('equal')
    ax.set_xlabel('u [px]')
    ax.set_ylabel('v [px]')
    ax.invert_yaxis()

    ax.legend()
    ax.grid(True)

    plt.xlim([300, 650])
    plt.ylim([650, 400])
    # plt.show()


    # ----- 3d plot -----
    fig3d = plt.figure()
    ax3d = fig3d.add_subplot(111, projection='3d')

    def world_components(pt):
        v = pt.pos_from(O).express(N)
        return v.dot(N.x), v.dot(N.y), v.dot(N.z)

    all_syms = list(square_vars)

    def eval_point(pt):
        cx, cy, cz = world_components(pt)
        fx = sp.lambdify(all_syms, cx)
        fy = sp.lambdify(all_syms, cy)
        fz = sp.lambdify(all_syms, cz)
        return (float(fx(*values)), float(fy(*values)), float(fz(*values)))

    s_pts_3d = [eval_point(pt) for pt in (s1, s2, s3, s4)]
    t_pts_3d = [eval_point(pt) for pt in (t1, t2, t3, t4)]

    s_pts_3d_closed = s_pts_3d + [s_pts_3d[0]]
    t_pts_3d_closed = t_pts_3d + [t_pts_3d[0]]

    sx, sy, sz = zip(*s_pts_3d_closed)
    tx, ty, tz = zip(*t_pts_3d_closed)

    ax3d.plot(sx, sy, sz, marker='o', label='s')
    ax3d.plot(tx, ty, tz, marker='o', label='t')

    cam_pos = (
        float(square_values[cam_x]),
        float(square_values[cam_y]),
        float(square_values[cam_z])
    )
    ax3d.scatter(*cam_pos, color='red', s=60, label='camera')

    # --- Reference frame arrows ---
    def draw_frame(origin, axes_exprs, length, colors, labels):
        """axes_exprs: list of 3 sympy Vectors (already in N frame)."""
        for vec_expr, color, lab in zip(axes_exprs, colors, labels):
            v = vec_expr.express(N)
            fx = sp.lambdify(all_syms, v.dot(N.x))
            fy = sp.lambdify(all_syms, v.dot(N.y))
            fz = sp.lambdify(all_syms, v.dot(N.z))
            d = np.array([float(fx(*values)), float(fy(*values)), float(fz(*values))])
            d = d / (np.linalg.norm(d) + 1e-12) * length
            ax3d.quiver(
                origin[0], origin[1], origin[2],
                d[0], d[1], d[2],
                color=color, arrow_length_ratio=0.2
            )
            ax3d.text(
                origin[0] + d[0], origin[1] + d[1], origin[2] + d[2],
                lab, color=color
            )

    # World frame at origin
    draw_frame(
        origin=(0.0, 0.0, 0.0),
        axes_exprs=[N.x, N.y, N.z],
        length=1.5,
        colors=['black', 'black', 'black'],
        labels=['Nx', 'Ny', 'Nz']
    )

    # Camera frame at camera position
    # C is presumably the camera reference frame already used in perspective_projection
    draw_frame(
        origin=cam_pos,
        axes_exprs=[C.x, C.y, C.z],
        length=1.5,
        colors=['red', 'green', 'blue'],
        labels=['Cx', 'Cy', 'Cz']
    )

    ax3d.set_xlabel('X')
    ax3d.set_ylabel('Y')
    ax3d.set_zlabel('Z')
    ax3d.legend()
    ax3d.set_title('3D view of squares and camera')

    plt.gca().set_aspect('equal')
    plt.show()


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


if __name__ == "__main__":
    visualize_square_projection()