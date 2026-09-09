import sympy as sp
import numpy as np
import sympy.physics.mechanics as me

def residual_eqs(x, data, bike_params, camera_params):
    subs = (x[0], x[1], x[2], x[3], x[4], x[5], x[6], bike_params['lr'],
            bike_params['lf1'], bike_params['lf2'], bike_params['rr'], 
            bike_params['rf'])

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

    return np.array([f01, f02, f03, f04, f05, f06, f07, f08, f09, f10, f11, f12, f13, f14])

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

    # if cx == 0:
    #     cx = screen_resolution[0]/2

    # if cy == 0:
    #     cy = screen_resolution[1]/2

    r = point.pos_from(camera_center)

    Xc = r.dot(camera_frame.x)
    Yc = r.dot(camera_frame.y)
    Zc = r.dot(camera_frame.z)

    u = fx * Xc / Zc + cx
    v = fy * Yc / Zc + cy

    return u, v



phi, theta, psi, delta = sp.symbols('varphi, theta, psi, delta')
lr, lf1, lf2 = sp.symbols('l_r, l_f1, l_f2')
x_r, y_r, z_r = sp.symbols('x_r, y_r, z_r')
rr, rf = sp.symbols('rr, rf')

# Camera intrinsics
f, cx, cy = sp.symbols('f, cx, cy') 

# Camera extrinsics
cam_x, cam_y, cam_z = sp.symbols('cam_x cam_y cam_z')
cam_yaw, cam_pitch, cam_roll = sp.symbols('cam_yaw cam_pitch cam_roll')

# The camera reference frame has x right, y up, z depth
# The bike reference frame has x depth, y left, z up


variables = (phi, theta, psi, delta, 
             x_r, y_r, z_r, 
             lr, lf1, lf2, 
             rr, rf,
             f, cx, cy,
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

P1r.set_pos(Cr, rr*me.cross(R.y, -me.cross(R.y, N.z)))
P2r.set_pos(Cr, rr*me.cross(R.y, me.cross(R.y, N.x)))
P3r.set_pos(Cr, rr*me.cross(R.y, me.cross(R.y, N.z)))
P4r.set_pos(Cr, rr*me.cross(R.y, me.cross(R.y, -N.x)))

P1f.set_pos(Cf, rf*me.cross(F.y, -me.cross(F.y, N.z)))
P2f.set_pos(Cf, rf*me.cross(F.y, me.cross(F.y, N.x)))
P3f.set_pos(Cf, rf*me.cross(F.y, me.cross(F.y, N.z)))
P4f.set_pos(Cf, rf*me.cross(F.y, me.cross(F.y, -N.x)))

# Perspective projections
u_Cr, v_Cr = perspective_projection(Cr, P, C, f)
u_Cf, v_Cf = perspective_projection(Cf, P, C, f)

u_S, v_S = perspective_projection(S, P, C, f)
u_Q, v_Q = perspective_projection(Q, P, C, f)

u_P1r, v_P1r = perspective_projection(P1r, P, C, f)
u_P3r, v_P3r = perspective_projection(P3r, P, C, f)

u_P1f, v_P1f = perspective_projection(P1f, P, C, f)
u_P3f, v_P3f = perspective_projection(P3f, P, C, f)


r_Cf_Cr_u = u_Cf - u_Cr
r_Cf_Cr_v = v_Cf - v_Cr

# r_Cr_O_u = u_Cr - cx
# r_Cr_O_v = v_Cr - cy

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

# eval_f15 = sp.lambdify(variables, r_Cf_Cr_y)
# eval_f16 = sp.lambdify(variables, r_Cr_O_y)
# eval_f17 = sp.lambdify(variables, r_Q_S_y)
# eval_f18 = sp.lambdify(variables, r_P1r_Cr_y)
# eval_f19 = sp.lambdify(variables, r_P3r_Cr_y)
# eval_f20 = sp.lambdify(variables, r_P1f_Cf_y)
# eval_f21 = sp.lambdify(variables, r_P3f_Cf_y)


# # ----- Segments for plotting -----

# r_Cr_O_y = Cr.pos_from(O).dot(N.y)

# r_S_O_x = S.pos_from(O).dot(N.x)
# r_S_O_y = S.pos_from(O).dot(N.y)
# r_S_O_z = S.pos_from(O).dot(N.z)

# r_Q_O_x = Q.pos_from(O).dot(N.x)
# r_Q_O_y = Q.pos_from(O).dot(N.y)
# r_Q_O_z = Q.pos_from(O).dot(N.z)

# r_Cf_O_x = Cf.pos_from(O).dot(N.x)
# r_Cf_O_y = Cf.pos_from(O).dot(N.y)
# r_Cf_O_z = Cf.pos_from(O).dot(N.z)

# r_P1r_O_x = P1r.pos_from(O).dot(N.x)
# r_P1r_O_y = P1r.pos_from(O).dot(N.y)
# r_P1r_O_z = P1r.pos_from(O).dot(N.z)

# r_P1f_O_x = P1f.pos_from(O).dot(N.x)
# r_P1f_O_y = P1f.pos_from(O).dot(N.y)
# r_P1f_O_z = P1f.pos_from(O).dot(N.z)

# r_P2r_O_x = P2r.pos_from(O).dot(N.x)
# r_P2r_O_y = P2r.pos_from(O).dot(N.y)
# r_P2r_O_z = P2r.pos_from(O).dot(N.z)

# r_P2f_O_x = P2f.pos_from(O).dot(N.x)
# r_P2f_O_y = P2f.pos_from(O).dot(N.y)
# r_P2f_O_z = P2f.pos_from(O).dot(N.z)

# r_P3r_O_x = P3r.pos_from(O).dot(N.x)
# r_P3r_O_y = P3r.pos_from(O).dot(N.y)
# r_P3r_O_z = P3r.pos_from(O).dot(N.z)

# r_P3f_O_x = P3f.pos_from(O).dot(N.x)
# r_P3f_O_y = P3f.pos_from(O).dot(N.y)
# r_P3f_O_z = P3f.pos_from(O).dot(N.z)

# r_P4r_O_x = P4r.pos_from(O).dot(N.x)
# r_P4r_O_y = P4r.pos_from(O).dot(N.y)
# r_P4r_O_z = P4r.pos_from(O).dot(N.z)

# r_P4f_O_x = P4f.pos_from(O).dot(N.x)
# r_P4f_O_y = P4f.pos_from(O).dot(N.y)
# r_P4f_O_z = P4f.pos_from(O).dot(N.z)


# eval_p01 = sp.lambdify(variables, r_Cr_O_y)
# eval_p02 = sp.lambdify(variables, r_S_O_x)
# eval_p03 = sp.lambdify(variables, r_S_O_y)
# eval_p04 = sp.lambdify(variables, r_S_O_z)
# eval_p05 = sp.lambdify(variables, r_Q_O_x)
# eval_p06 = sp.lambdify(variables, r_Q_O_y)
# eval_p07 = sp.lambdify(variables, r_Q_O_z)
# eval_p08 = sp.lambdify(variables, r_Cf_O_x)
# eval_p09 = sp.lambdify(variables, r_Cf_O_y)
# eval_p10 = sp.lambdify(variables, r_Cf_O_z)
# eval_p11 = sp.lambdify(variables, r_P1r_O_x)
# eval_p12 = sp.lambdify(variables, r_P1r_O_y)
# eval_p13 = sp.lambdify(variables, r_P1r_O_z)
# eval_p14 = sp.lambdify(variables, r_P1f_O_x)
# eval_p15 = sp.lambdify(variables, r_P1f_O_y)
# eval_p16 = sp.lambdify(variables, r_P1f_O_z)
# eval_p17 = sp.lambdify(variables, r_P2r_O_x)
# eval_p18 = sp.lambdify(variables, r_P2r_O_y)
# eval_p19 = sp.lambdify(variables, r_P2r_O_z)
# eval_p20 = sp.lambdify(variables, r_P2f_O_x)
# eval_p21 = sp.lambdify(variables, r_P2f_O_y)
# eval_p22 = sp.lambdify(variables, r_P2f_O_z)
# eval_p23 = sp.lambdify(variables, r_P3r_O_x)
# eval_p24 = sp.lambdify(variables, r_P3r_O_y)
# eval_p25 = sp.lambdify(variables, r_P3r_O_z)
# eval_p26 = sp.lambdify(variables, r_P3f_O_x)
# eval_p27 = sp.lambdify(variables, r_P3f_O_y)
# eval_p28 = sp.lambdify(variables, r_P3f_O_z)
# eval_p29 = sp.lambdify(variables, r_P4r_O_x)
# eval_p30 = sp.lambdify(variables, r_P4r_O_y)
# eval_p31 = sp.lambdify(variables, r_P4r_O_z)
# eval_p32 = sp.lambdify(variables, r_P4f_O_x)
# eval_p33 = sp.lambdify(variables, r_P4f_O_y)
# eval_p34 = sp.lambdify(variables, r_P4f_O_z)