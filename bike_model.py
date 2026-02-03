import sympy as sp
import sympy.physics.mechanics as me

phi, theta, psi, delta = sp.symbols('varphi, theta, psi, delta')
lr, lf1, lf2 = sp.symbols('l_r, l_f1, l_f2')
x_r, y_r, z_r = sp.symbols('x_r, y_r, z_r')

N, R, Rs, F = sp.symbols('N, R, Rs, F', cls=me.ReferenceFrame)


R.orient_body_fixed(N, (psi, phi, theta), 'zxy')
F.orient_axis(R, delta, R.z)

Cr = me.Point('C_r')
Cf = me.Point('C_f')
S = me.Point('S')
P = me.Point('P')
O = me.Point('O')

O.set_pos(O, 0)
O.set_vel(N, 0)

Cr.set_pos(O, x_r*N.x + y_r*N.y + z_r*N.z) # assuming that the bicycle is never in front of the set origin.
S.set_pos(Cr, lr*R.x)
P.set_pos(S, -lf1 * F.z)
Cf.set_pos(P, lf2 * F.x)

Cf.pos_from(Cr)

r_Cf_Cr_xz_x = Cf.pos_from(Cr).dot(N.x)
r_Cf_Cr_xz_z = Cf.pos_from(Cr).dot(N.z)

r_Cr_O_xz_x = Cr.pos_from(O).dot(N.x)
r_Cr_O_xz_z = Cr.pos_from(O).dot(N.z)

r_P_S_x = R.x.dot(N.x)
r_P_S_z = R.z.dot(N.z)