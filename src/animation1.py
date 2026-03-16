def generate_symbolic_bicycle_kinematics():
    import sympy as sm
    import sympy.physics.mechanics as me

    x, y, z, q3, q4, q5, q6, q7, q8, d1, d2, d3, rr, rf = sm.symbols(
        'x, y, z, q3, q4, q5, q6, q7, q8, d1, d2, d3, rr, rf')

    # Newtonian Frame
    N = me.ReferenceFrame('N')
    # Yaw Frame
    A = me.ReferenceFrame('A')
    # Roll Frame
    B = me.ReferenceFrame('B')
    # Rear Frame
    C = me.ReferenceFrame('C')
    # Rear Wheel Frame
    D = me.ReferenceFrame('D')
    # Front Frame
    E = me.ReferenceFrame('E')
    # Front Wheel Frame
    F = me.ReferenceFrame('F')

    # The following defines a 3-1-2 Tait-Bryan rotation with yaw (q3), roll
    # (q4), pitch (q5) angles to orient the rear frame relative to the ground
    # (Newtonian frame). The front frame is then rotated through the steer
    # angle (q7) about the rear frame's 3 axis. The wheels are not oriented, as
    # q6 and q8 end up being ignorable coordinates.

    # rear frame yaw
    A.orient(N, 'Axis', (q3, N.z))
    # rear frame roll
    B.orient(A, 'Axis', (q4, A.x))
    # rear frame pitch
    C.orient(B, 'Axis', (q5, B.y))
    # rear wheel rotation
    D.orient(C, 'Axis', (q6, C.y))
    # front frame steer
    E.orient(C, 'Axis', (q7, C.z))
    # front wheel rotation
    F.orient(E, 'Axis', (q8, E.y))

    # point fixed on the ground
    o = me.Point('o')

    # origin to rear wheel center
    do = me.Point('do')
    do.set_pos(o, x*N.x + y*N.y + z*N.z)

    # rear wheel center to steer axis point
    ce = me.Point('ce')
    ce.set_pos(do, d1*C.x)

    # steer axis point to the ?
    ff = me.Point('ff')
    ff.set_pos(ce, d2*E.z)

    # ? to the front wheel center
    fo = me.Point('fo')
    fo.set_pos(ff, d3*E.x)

    return (N, A, B, C, D, E, F, o, do, ce, ff, fo, x, y, z, q3, q4, q5, q6,
            q7, q8, d1, d2, d3, rr, rf)


def plot_3d_bicycle(x, y, z, q3, q4, q5, q6, q7, q8, d1, d2, d3, rr, rf):
    """

    .. plot::
       :context: reset
       :include-source:

       import numpy as np
       from dtk.bicycle import (benchmark_parameters, benchmark_to_moore,
                                plot_3d_bicycle)
       bpar = benchmark_parameters()
       mpar = benchmark_to_moore(bpar)
       plot_3d_bicycle(
           0.2,
           0.2,
           -0.2,
           -np.deg2rad(10.0),
           -np.deg2rad(20.0),
           bpar['lam'],
           np.deg2rad(90.0),
           np.deg2rad(40.0),
           np.deg2rad(135.0),
           mpar['d1'],
           mpar['d2'],
           mpar['d3'],
           mpar['rr'],
           mpar['rf'],
       )

    """

    import matplotlib.pyplot as plt
    from symmeplot.matplotlib import Scene3D
    import sympy.physics.mechanics as me

    (N, A, B, C, D, E, F,
     o, do, ce, ff, fo,
     x_s, y_s, z_s, q3_s, q4_s, q5_s, q6_s, q7_s, q8_s,
     d1_s, d2_s, d3_s, rr_s, rf_s) = generate_symbolic_bicycle_kinematics()

    fig, ax = plt.subplots(subplot_kw={'projection': '3d'},
                           figsize=(12.0, 6.0))

    scene = Scene3D(N, o, ax=ax, scale=1.0)

    rwheel = me.RigidBody('rwheel', do, D, 0.0, (me.inertia(D, 1, 1, 1), do))
    rear_wheel_plot = scene.add_body(rwheel,
                                     plot_frame_properties={"scale": 0.3})
    rear_wheel_plot.attach_circle(do, rr_s, C.y, facecolor="black", alpha=0.4,
                                  edgecolor="black")

    fwheel = me.RigidBody('rwheel', fo, F, 0.0, (me.inertia(F, 1, 1, 1), fo))
    front_wheel_plot = scene.add_body(fwheel,
                                      plot_frame_properties={"scale": 0.3})
    front_wheel_plot.attach_circle(fo, rf_s, E.y, facecolor="black", alpha=0.4,
                                   edgecolor="black")

    scene.add_line([do, ce, ff, fo], color='k', linewidth=2)

    scene.lambdify_system((x_s, y_s, z_s, q3_s, q4_s, q5_s, q6_s, q7_s, q8_s,
                           d1_s, d2_s, d3_s, rr_s, rf_s))
    scene.evaluate_system(x, y, z, q3, q4, q5, q6, q7, q8, d1, d2, d3, rr, rf)

    scene.plot(prettify=False)
    ax.set_xlim((200.0, 900.0))
    ax.set_ylim((-50.0, 50.0))
    ax.set_zlim((-700.0, 100.0))
    ax.invert_zaxis()
    ax.set_aspect('equal')
    fig.subplots_adjust(left=-0.3, right=1.3, bottom=-0.3, top=1.3)

    return scene


def animate_3d_bicycle(time, x_v, y_v, z_v, q3_v, q4_v, q5_v, q6_v, q7_v, q8_v,
                       d1_v, d2_v, d3_v, rr_v, rf_v, save=False):
    """
    .. plot::
       :context: reset
       :include-source:

       import numpy as np
       from scipy.signal import lti, lsim
       from dtk.bicycle import (benchmark_matrices, benchmark_state_space,
                                animate_3d_bicycle, benchmark_parameters,
                                benchmark_to_moore)
       t = np.linspace(0.0, 5.0)
       x0 = np.zeros(10)
       x0[8] = 1.5
       speed = 4.6  # m/s

       bpar = benchmark_parameters()
       #q1' = v
       #q2' = v*q3
       #q3' = (v*q7 + c*u7)/w/cos(lam)
       #q5' = 0
       #q6' = v/rr
       #q8' = v/rf

       #q4, q7, u4 u7
       #A = [0, 0, 1, 0]
           #[0, 0, 0, 1]
           #[          ]
           #[          ]
       A_, B_ = benchmark_state_space(*benchmark_matrices(), speed, 9.81)

       A = np.zeros((10, 10))
       B = np.zeros((10, 2))

       A[1, 2] = speed  # q2' = v*q3
       A[2, 6] = speed/bpar['w']/np.cos(bpar['lam'])
       A[2, 9] = bpar['c']/bpar['w']/np.cos(bpar['lam'])
       A[3, 8] = 1.0
       A[6, 9] = 1.0
       A[3, 3] = A_[2, 0]
       A[3, 6] = A_[2, 1]
       A[3, 8] = A_[2, 2]
       A[3, 9] = A_[2, 3]
       A[6, 3] = A_[3, 0]
       A[6, 6] = A_[3, 1]
       A[6, 8] = A_[3, 2]
       A[6, 9] = A_[3, 3]

       C, D = np.eye(10), np.zeros((10, 2))

       system = lti(A, B, C, D)
       t, y, _ = lsim(system, 0.0, t, X0=x0)

       mpar = benchmark_to_moore(bpar)
       ani = animate_3d_bicycle(
           t,
           speed*t,
           speed*np.sin(y[:, 2]*t,
           -bpar['rR']*np.cos(y[:, 3]),
           np.zeros_like(y[:, 2]),  # q3
           y[:, 3],  # q4
           bpar['lam']*np.ones_like(t),
           speed/bpar['rR']*t,
           y[:, 6],  # q7
           speed/bpar['rF']*t,
           mpar['d1'],
           mpar['d2'],
           mpar['d3'],
           mpar['rr'],
           mpar['rf'],
           save=True,
       )

    """
    FPS = int(1/(time[1] - time[0]))

    scene = plot_3d_bicycle(x_v[0], y_v[0], z_v[0], q3_v[0], q4_v[0], q5_v[0],
                            q6_v[0], q7_v[0], q8_v[0], d1_v, d2_v, d3_v, rr_v,
                            rf_v)

    def update(i):
        return (x_v[i], y_v[i], z_v[i], q3_v[i], q4_v[i], q5_v[i], q6_v[i],
                q7_v[i], q8_v[i], d1_v, d2_v, d3_v, rr_v, rf_v)

    slow_factor = 1  # int
    ani = scene.animate(update, frames=len(time),
                        interval=slow_factor/FPS*1000)
    if save:
        #ani.save("animation.gif", fps=FPS//slow_factor)
        ani.save("animation.mp4", fps=60)
                 #fps=FPS//slow_factor)

    return ani
