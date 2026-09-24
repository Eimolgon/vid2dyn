# coding: utf-8
import sympy as sp
import sympy.physics.mechanics as me
N = me.ReferenceFrame('N')
A = me.ReferenceFrame('A')
C = me.Point('C')
P1 = me.Point('P1')
x, y, z = sp.symbols('x, y, z')
q1, q2, q3 = sp.symbols('q1, q2, q3')
A.orient_body_fixed(N, (q1, q2, q3), 'xyz')
O = me.Point('O')
O.set_pos(O, 0)
C.set_pos(O, x*N.x + y*N.y + z*N.z)
x2, y2, z2 = sp.symbols('x2, y2, z2')
P1.set_pos(C, x2*N.x + y2*N.y + z2*N.z)
v = P1.pos_from(C)
v
u = P1.pos_form(O)
v2 = P1.pos_from(O)
v2
v3 = v + v2
v3
sp.simplify(v3)
sp.simplify(v + v2)
sp.simplify(v)
v0 = sp.zeros(3,1)
v0
v4 = v3.to_matrix()
v4 = v3.to_matrix(N)
v4
v5 = -v - v2
v5
v6 = v5.to_matrix(N)
v4 - v6
v4 + v6
assert (v4 + v6) == v0
(v4 + v6) == v0
