import sympy as sp
import numpy as np
import sympy.physics.mechanics as me

def residual_eqs(x, data, bike_params, camera_params):
    subs = (x[0], x[1], x[2], x[3], x[4], x[5], x[6], 

            bike_params['lr'],bike_params['lf1'], bike_params['lf2'], 
            bike_params['rr'], bike_params['rf'], 

            camera_params['fx'], camera_params['fy'],
            camera_params['cx'], camera_params['cy'],
            camera_params['cam_x'], camera_params['cam_y'], 
            camera_params['cam_z'], camera_params['cam_yaw'], 
            camera_params['cam_pitch'],camera_params['cam_roll']
            )

    f01 = eval_f01(*subs) - data['r_Cf_Cr_u']
    f02 = eval_f02(*subs) - data['r_Cf_Cr_v']
    f03 = eval_f03(*subs) - data['u_Cr']
    f04 = eval_f04(*subs) - data['v_Cr']
    f05 = eval_f05(*subs) - data['r_P1r_Cr_u']
    f06 = eval_f06(*subs) - data['r_P1r_Cr_v']
    f07 = eval_f07(*subs) - data['r_P3r_Cr_u']
    f08 = eval_f08(*subs) - data['r_P3r_Cr_v']
    f09 = eval_f09(*subs) - data['r_P1f_Cf_u']
    f10 = eval_f10(*subs) - data['r_P1f_Cf_v']
    f11 = eval_f11(*subs) - data['r_P3f_Cf_u']
    f12 = eval_f12(*subs) - data['r_P3f_Cf_v']
    f13 = eval_f13(*subs) - data['r_Q_S_u']
    f14 = eval_f14(*subs) - data['r_Q_S_v']

    return np.array([f01, f02, f03, f04, f05, f06, f07, 
                     f08, f09, f10, f11, f12, f13, f14])

def perspective_projection(point, camera_center, camera_frame,
                           fx, fy=0, cx=0, cy=0):
    """
    Project a point into the image plane.

    Camera convention:
        camera_frame.x -> image horizontal axis
        camera_frame.z -> image vertical axis
        camera_frame.y -> optical axis / depth
    """
    if fy == 0:
        fy = fx

    r = point.pos_from(camera_center)

    Xc = r.dot(camera_frame.x)
    Yc = r.dot(camera_frame.y)
    Zc = r.dot(camera_frame.z)

    u = fx * Xc / Yc + cx
    v = fy * Zc / Yc + cy
    
    return u, v


phi, theta, psi, delta = sp.symbols('varphi, theta, psi, delta')
lr, lf1, lf2 = sp.symbols('l_r, l_f1, l_f2')
x_r, y_r, z_r = sp.symbols('x_r, y_r, z_r')
rr, rf = sp.symbols('rr, rf')

# ===== Camera =====

"""
Camera frame convention:
    C.x : image horizontal, positive right
    C.y : optical axis / depth, positive forward
    C.z : image vertical, positive up

Perspective projection:
    u = fx * Xc / Yc + cx
    v = fy * Zc / Yc + cy
"""

# Camera intrinsics
fx, fy, cx, cy = sp.symbols('fx, fy, cx, cy') 

# Camera extrinsics
cam_x, cam_y, cam_z = sp.symbols('cam_x cam_y cam_z')
cam_yaw, cam_pitch, cam_roll = sp.symbols('cam_yaw cam_pitch cam_roll')

# The camera reference frame has x right, y up, z depth
# The bike reference frame has x depth, y left, z up


variables = (phi, theta, psi, delta, 
             x_r, y_r, z_r, 
             lr, lf1, lf2, 
             rr, rf,
             fx, fy, cx, cy,
             cam_x, cam_y, cam_z,
             cam_yaw, cam_pitch, cam_roll)

N, R, F, C = sp.symbols('N, R, F, C', cls=me.ReferenceFrame)


R.orient_body_fixed(N, (psi, phi, theta), 'ZXY')
F.orient_axis(R, delta, R.z)
C.orient_body_fixed(N, (cam_yaw, cam_pitch, cam_roll), 'ZYX')

Cr = me.Point('C_r')    # Rear wheel contact point
Cf = me.Point('C_f')    # Front wheel contact point
S = me.Point('S')       # Steering
Q = me.Point('Q')       # Trail
O = me.Point('O')       # Origin of the world coordinate system
P = me.Point('P')       # Pinhole

P1r = me.Point('P_1r') 
P2r = me.Point('P_2r')
P3r = me.Point('P_3r')
P4r = me.Point('P_4r')

P1f = me.Point('P_1f')
P2f = me.Point('P_2f')
P3f = me.Point('P_3f')
P4f = me.Point('P_4f')


O.set_pos(O, 0)
O.set_vel(N, 0)

Cr.set_pos(O, x_r*N.x + y_r*  N.y + z_r*N.z) # assuming that the bicycle is never in front of the set origin.
S.set_pos(Cr, lr * R.x)
Q.set_pos(S, lf1 * -F.z)
Cf.set_pos(Q, lf2 * F.x)
P.set_pos(O, cam_x*N.x + cam_y*N.y + cam_z*N.z)

P1r.set_pos(Cr, rr*me.cross(R.y, -me.cross(R.y, N.z)).normalize())
P2r.set_pos(Cr, rr*me.cross(R.y, me.cross(R.y, N.x)).normalize())
P3r.set_pos(Cr, rr*me.cross(R.y, me.cross(R.y, N.z)).normalize())
P4r.set_pos(Cr, rr*me.cross(R.y, me.cross(R.y, -N.x)).normalize())

P1f.set_pos(Cf, rf*me.cross(F.y, -me.cross(F.y, N.z)).normalize())
P2f.set_pos(Cf, rf*me.cross(F.y, me.cross(F.y, N.x)).normalize())
P3f.set_pos(Cf, rf*me.cross(F.y, me.cross(F.y, N.z)).normalize())
P4f.set_pos(Cf, rf*me.cross(F.y, me.cross(F.y, -N.x)).normalize())

# Perspective projections
u_Cr, v_Cr = perspective_projection(Cr, P, C, fx, fy, cx, cy)
u_Cf, v_Cf = perspective_projection(Cf, P, C, fx, fy, cx, cy)

u_S, v_S = perspective_projection(S, P, C, fx, fy, cx, cy)
u_Q, v_Q = perspective_projection(Q, P, C, fx, fy, cx, cy)

u_P1r, v_P1r = perspective_projection(P1r, P, C, fx, fy, cx, cy)
u_P3r, v_P3r = perspective_projection(P3r, P, C, fx, fy, cx, cy)

u_P1f, v_P1f = perspective_projection(P1f, P, C, fx, fy, cx, cy)
u_P3f, v_P3f = perspective_projection(P3f, P, C, fx, fy, cx, cy)


r_Cf_Cr_u = u_Cf - u_Cr
r_Cf_Cr_v = v_Cf - v_Cr

r_Q_S_u = u_Q - u_S
r_Q_S_v = v_Q - v_S

r_P1r_Cr_u = u_P1r - u_Cr
r_P1r_Cr_v = v_P1r - v_Cr

r_P3r_Cr_u = u_P3r - u_Cr
r_P3r_Cr_v = v_P3r - v_Cr

r_P1f_Cf_u = u_P1f - u_Cf
r_P1f_Cf_v = v_P1f - v_Cf

r_P3f_Cf_u = u_P3f - u_Cf
r_P3f_Cf_v = v_P3f - v_Cf



# Lambdify functions

eval_f01 = sp.lambdify(variables, r_Cf_Cr_u)
eval_f02 = sp.lambdify(variables, r_Cf_Cr_v)

eval_f03 = sp.lambdify(variables, u_Cr)
eval_f04 = sp.lambdify(variables, v_Cr)

eval_f05 = sp.lambdify(variables, r_P1r_Cr_u)
eval_f06 = sp.lambdify(variables, r_P1r_Cr_v)

eval_f07 = sp.lambdify(variables, r_P3r_Cr_u)
eval_f08 = sp.lambdify(variables, r_P3r_Cr_v)

eval_f09 = sp.lambdify(variables, r_P1f_Cf_u)
eval_f10 = sp.lambdify(variables, r_P1f_Cf_v)

eval_f11 = sp.lambdify(variables, r_P3f_Cf_u)
eval_f12 = sp.lambdify(variables, r_P3f_Cf_v)

eval_f13 = sp.lambdify(variables, r_Q_S_u)
eval_f14 = sp.lambdify(variables, r_Q_S_v)
