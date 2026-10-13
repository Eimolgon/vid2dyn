"""Perspective bicycle model and least-squares residuals.

Camera convention follows the supplied project: camera x is image-right,
y is optical depth (positive in front), and z is image-up.
"""
import numpy as np
import sympy as sp
import sympy.physics.mechanics as me

def perspective_projection(point, camera_center, camera_frame, fx, fy, cx, cy):
    r = point.pos_from(camera_center)
    Xc, Yc, Zc = (r.dot(camera_frame.x), r.dot(camera_frame.y), r.dot(camera_frame.z))
    u = fx * Xc / Yc + cx
    v = fy * Zc / Yc + cy  # model uses image-up v; plotting flips to raster coordinates
    return u, v

phi, theta, psi, delta = sp.symbols('varphi theta psi delta')
lr, lf1, lf2 = sp.symbols('l_r l_f1 l_f2')
x_r, y_r, z_r = sp.symbols('x_r y_r z_r')
rr, rf = sp.symbols('rr rf')
fx, fy, cx, cy = sp.symbols('fx fy cx cy')
cam_x, cam_y, cam_z = sp.symbols('cam_x cam_y cam_z')
cam_yaw, cam_pitch, cam_roll = sp.symbols('cam_yaw cam_pitch cam_roll')
variables = (phi, theta, psi, delta, x_r, y_r, z_r, lr, lf1, lf2, rr, rf, fx, fy, cx, cy, cam_x, cam_y, cam_z, cam_yaw, cam_pitch, cam_roll)
N, R, F, C = sp.symbols('N R F C', cls=me.ReferenceFrame)
R.orient_body_fixed(N, (psi, phi, theta), 'ZXY')
F.orient_axis(R, delta, R.z)
C.orient_body_fixed(N, (cam_yaw, cam_pitch, cam_roll), 'ZYX')
O = me.Point('O'); O.set_pos(O, 0)
Cr, S, Q, Cf, P = [me.Point(s) for s in ('Cr','S','Q','Cf','P')]
Cr.set_pos(O, x_r*N.x + y_r*N.y + z_r*N.z)
S.set_pos(Cr, lr*R.x); Q.set_pos(S, -lf1*F.z); Cf.set_pos(Q, lf2*F.x)
P.set_pos(O, cam_x*N.x + cam_y*N.y + cam_z*N.z)
def wheel_axes(frame):
    h = me.cross(frame.y, me.cross(frame.y, N.x)).normalize()
    return h, me.cross(frame.y, h)
uhr, uvr = wheel_axes(R); uhf, uvf = wheel_axes(F)
P1r, P3r, P1f, P3f = [me.Point(s) for s in ('P1r','P3r','P1f','P3f')]
P1r.set_pos(Cr, rr*uvr); P3r.set_pos(Cr, -rr*uvr)
P1f.set_pos(Cf, rf*uvf); P3f.set_pos(Cf, -rf*uvf)
def proj(point): return perspective_projection(point, P, C, fx, fy, cx, cy)
ucr,vcr=proj(Cr); ucf,vcf=proj(Cf); us,vs=proj(S); uq,vq=proj(Q)
u1r,v1r=proj(P1r); u3r,v3r=proj(P3r); u1f,v1f=proj(P1f); u3f,v3f=proj(P3f)
exprs=(ucf-ucr, vcf-vcr, ucr, vcr, u1r-ucr, v1r-vcr, u3r-ucr, v3r-vcr, u1f-ucf, v1f-vcf, u3f-ucf, v3f-vcf, uq-us, vq-vs)
_evals=[sp.lambdify(variables,e,'numpy') for e in exprs]
KEYS=('r_Cf_Cr_u','r_Cf_Cr_v','u_Cr','v_Cr','r_P1r_Cr_u','r_P1r_Cr_v','r_P3r_Cr_u','r_P3r_Cr_v','r_P1f_Cf_u','r_P1f_Cf_v','r_P3f_Cf_u','r_P3f_Cf_v','r_Q_S_u','r_Q_S_v')
def _subs(x,b,c):
    return tuple(x)+tuple(b[k] for k in ('lr','lf1','lf2','rr','rf'))+tuple(c[k] for k in ('fx','fy','cx','cy','cam_x','cam_y','cam_z','cam_yaw','cam_pitch','cam_roll'))
def predict(x,bike_params,camera_params):
    args=_subs(x,bike_params,camera_params)
    return dict(zip(KEYS, (float(f(*args)) for f in _evals)))
def residual_eqs(x,data,bike_params,camera_params):
    pred=predict(x,bike_params,camera_params)
    return np.asarray([pred[k]-float(np.asarray(data[k]).reshape(-1)[0]) for k in KEYS])

def points3d(bike_params,state):
    phi,theta,psi,delta,xr,yr,zr=map(float,state)
    n,r,f=me.ReferenceFrame('N'),me.ReferenceFrame('R'),me.ReferenceFrame('F')
    r.orient_body_fixed(n,(psi,phi,theta),'ZXY'); f.orient_axis(r,delta,r.z)
    o=me.Point('O'); o.set_pos(o,0); cr=me.Point('Cr'); cr.set_pos(o,xr*n.x+yr*n.y+zr*n.z)
    s=me.Point('S'); s.set_pos(cr,bike_params['lr']*r.x)
    q=me.Point('Q'); q.set_pos(s,-bike_params['lf1']*f.z)
    cf=me.Point('Cf'); cf.set_pos(q,bike_params['lf2']*f.x)
    def xyz(pt):
        v=pt.pos_from(o); return np.array([float(v.dot(a)) for a in (n.x,n.y,n.z)])
    def axes(fr):
        h=me.cross(fr.y,me.cross(fr.y,n.x)).normalize(); return h,me.cross(fr.y,h)
    hr,vr=axes(r); hf,vf=axes(f)
    return {'Cr':xyz(cr),'S':xyz(s),'Q':xyz(q),'Cf':xyz(cf),
      'P1r':xyz(cr)+bike_params['rr']*np.array([float(vr.dot(a)) for a in (n.x,n.y,n.z)]),
      'P3r':xyz(cr)-bike_params['rr']*np.array([float(vr.dot(a)) for a in (n.x,n.y,n.z)]),
      'P2r':xyz(cr)+bike_params['rr']*np.array([float(hr.dot(a)) for a in (n.x,n.y,n.z)]),
      'P4r':xyz(cr)-bike_params['rr']*np.array([float(hr.dot(a)) for a in (n.x,n.y,n.z)]),
      'P1f':xyz(cf)+bike_params['rf']*np.array([float(vf.dot(a)) for a in (n.x,n.y,n.z)]),
      'P3f':xyz(cf)-bike_params['rf']*np.array([float(vf.dot(a)) for a in (n.x,n.y,n.z)]),
      'P2f':xyz(cf)+bike_params['rf']*np.array([float(hf.dot(a)) for a in (n.x,n.y,n.z)]),
      'P4f':xyz(cf)-bike_params['rf']*np.array([float(hf.dot(a)) for a in (n.x,n.y,n.z)])}
