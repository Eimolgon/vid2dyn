import sympy as sp
import sympy.physics.mechanics as me

phi, theta, psi, delta = sp.symbols('varphi, theta, psi, delta')
lr, lf1, lf2 = sp.symbols('l_r, l_f1, l_f2')
x_r, y_r, z_r = sp.symbols('x_r, y_r, z_r')
r = sp.symbols('r')

N, R, F = sp.symbols('N, R, F', cls=me.ReferenceFrame)


R.orient_body_fixed(N, (psi, phi, theta), 'zxy')
F.orient_axis(R, delta, R.z)

Cr = me.Point('C_r')
Cf = me.Point('C_f')
S = me.Point('S')
Q = me.Point('Q')
O = me.Point('O')

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

Cr.set_pos(O, x_r*N.x + y_r*N.y + z_r*N.z) # assuming that the bicycle is never in front of the set origin.
S.set_pos(Cr, lr*R.x)
Q.set_pos(S, -lf1 * F.z)
Cf.set_pos(Q, lf2 * F.x)

P1r.set_pos(Cr, r*me.cross(R.y, -me.cross(R.y, N.z)))
P2r.set_pos(Cr, r*me.cross(R.y, me.cross(R.y, N.x)))
P3r.set_pos(Cr, r*me.cross(R.y, me.cross(R.y, N.z)))
P4r.set_pos(Cr, r*me.cross(R.y, me.cross(R.y, -N.x)))

P1f.set_pos(Cf, r*me.cross(F.y, -me.cross(F.y, N.z)))
P2f.set_pos(Cf, r*me.cross(F.y, me.cross(F.y, N.x)))
P3f.set_pos(Cf, r*me.cross(F.y, me.cross(F.y, N.z)))
P4f.set_pos(Cf, r*me.cross(F.y, me.cross(F.y, -N.x)))

# Projections into the screen plane

r_Cf_Cr_xz_x = Cf.pos_from(Cr).dot(N.x)
r_Cf_Cr_xz_z = Cf.pos_from(Cr).dot(N.z)

r_Cr_O_xz_x = Cr.pos_from(O).dot(N.x)
r_Cr_O_xz_z = Cr.pos_from(O).dot(N.z)

r_Q_S_x = R.x.dot(N.x)
r_Q_S_z = R.z.dot(N.z)

r_P1r_Cr_x = P1r.pos_from(Cr).dot(N.x)
r_P1r_Cr_z = P1r.pos_from(Cr).dot(N.z)

r_P3r_Cr_x = P3r.pos_from(Cr).dot(N.x)
r_P3r_Cr_z = P3r.pos_from(Cr).dot(N.z)

r_P1f_Cf_x = P1f.pos_from(Cf).dot(N.x)
r_P1f_Cf_z = P1f.pos_from(Cf).dot(N.z)

r_P3f_Cf_x = P3f.pos_from(Cf).dot(N.x)
r_P3f_Cf_z = P3f.pos_from(Cf).dot(N.z)

# Lambdify functions

eval_f1 = sp.lambdify((phi, theta, psi, delta, x_r, y_r, z_r, lr, lf1, lf2, r), r_Cf_Cr_xz_x)
eval_f2 = sp.lambdify((phi, theta, psi, delta, x_r, y_r, z_r, lr, lf1, lf2, r), r_Cf_Cr_xz_z)
eval_f3 = sp.lambdify((phi, theta, psi, delta, x_r, y_r, z_r, lr, lf1, lf2, r), r_Cr_O_xz_x)
eval_f4 = sp.lambdify((phi, theta, psi, delta, x_r, y_r, z_r, lr, lf1, lf2, r), r_Cr_O_xz_z)
eval_f5 = sp.lambdify((phi, theta, psi, delta, x_r, y_r, z_r, lr, lf1, lf2, r), r_P1r_Cr_x)
eval_f6 = sp.lambdify((phi, theta, psi, delta, x_r, y_r, z_r, lr, lf1, lf2, r), r_P1r_Cr_z)
eval_f7 = sp.lambdify((phi, theta, psi, delta, x_r, y_r, z_r, lr, lf1, lf2, r), r_P3r_Cr_x)
eval_f8 = sp.lambdify((phi, theta, psi, delta, x_r, y_r, z_r, lr, lf1, lf2, r), r_P3r_Cr_z)
eval_f9 = sp.lambdify((phi, theta, psi, delta, x_r, y_r, z_r, lr, lf1, lf2, r), r_P1f_Cf_x)
eval_f10 = sp.lambdify((phi, theta, psi, delta, x_r, y_r, z_r, lr, lf1, lf2, r), r_P1f_Cf_z)
eval_f11 = sp.lambdify((phi, theta, psi, delta, x_r, y_r, z_r, lr, lf1, lf2, r), r_P3f_Cf_x)
eval_f12 = sp.lambdify((phi, theta, psi, delta, x_r, y_r, z_r, lr, lf1, lf2, r), r_P3f_Cf_z)