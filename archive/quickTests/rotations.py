# coding: utf-8
import sympy as sp
import sympy.physics.mechanics as me
N = me.ReferenceFrame('N')
A = me.ReferenceFrame('A')
q1, q2, q3 = sp.symbols('q1, q2, q3')
A.orient_body_fixed(N, (q1, q2, q3), 'xyz')
eval_frame_x = sp.lambdify((q1, q2, q3), A.x.to_matrix(N)) 
import numpy as np
subs_x = (np.pi/2, 0, 0)
f_x = eval_frame_x(*subs_x)
f_x
eval_frame_y = sp.lambdify((q1, q2, q3), A.y.to_matrix(N))
eval_frame_z = sp.lambdify((q1, q2, q3), A.z.to_matrix(N))
fx2 = eval_frame_y(*subs_x)
fx2
subs_y = (0, np.pi/2, 0)
fy1 = eval_frame_x(*subs_y)
fy1
subs_z = (0, 0, np.pi/2)
fz1 = eval_frame_x(subs_z)
fz1 = eval_frame_x(*subs_z)
fzq
fz1
fz2 = eval_frame_y(*subs_z)
fz3 = eval_frame_z(*subs_z)
z_vectors = [fz1, fz2, fz3]
fzStack = np.stack(z_vectors)
fzStack
fx3 = eval_frame_z(*subs_x)
x_vectors = [f_x, fx2, fx3]
fxStack = np.stack(x_vectors)
fxStack
fy2 = eval_frame_y(*subs_y)
fy3 = eval_frame_z(*subs_y)
y_vectors = [fy1, fy2, fy3]
fyStack = np.stack(y_vectors)
fyStack
