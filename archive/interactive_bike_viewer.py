"""Interactive 3D bicycle and calibrated pinhole projection viewer.
Run: python interactive_bike_viewer.py [--data frame.txt]
Frame labels are YOLO-style: class-id followed by normalized x y pairs;
class 0=front wheel and class 1=rear wheel.
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from bike_model_perplexity import points3d, predict

def load_points(path):
    if not path: return None
    out={}
    with open(path) as f:
        for line in f:
            a=np.fromstring(line,sep=' ')
            if a.size>=5 and int(a[0]) in (0,1): out[int(a[0])]=a[1:].reshape(-1,2)
    return out or None

def launch(bike=None,camera=None,state=None,image_points=None,screen_resolution=(1920,1080),show=True):
    bike=dict(bike or {'lr':1.4,'lf1':.5,'lf2':.1,'rr':.3302,'rf':.3302})
    cam=dict(camera or {'fx':7169.0,'fy':7169.0,'cx':960.0,'cy':540.0,'cam_x':0.,'cam_y':0.,'cam_z':.45,'cam_yaw':0.,'cam_pitch':0.,'cam_roll':0.})
    x0=np.array(state if state is not None else [0,0,0,0,2.82,.1,0.],float)
    labels=('phi','theta','psi','delta','x_r [m]','y_r [m]','z_r [m]')
    bounds=[(-np.pi/2,np.pi/2)]*4+[(-20,20),(-20,20),(-20,20)]
    fig=plt.figure(figsize=(14,9)); ax3=fig.add_axes([.04,.31,.44,.64],projection='3d'); ax2=fig.add_axes([.55,.31,.42,.64])
    fig.subplots_adjust(bottom=.29)
    sliders={}
    vals=x0.tolist()
    def make_artists():
        p=points3d(bike,vals)
        for ax in (ax3,ax2): ax.clear()
        ax3.set_title('Bike model (world coordinates)'); ax3.set_xlabel('X [m]'); ax3.set_ylabel('Y [m]'); ax3.set_zlabel('Z [m]'); ax3.set_proj_type('persp')
        for a,b,c in [('Cr','S','C0'),('S','Q','C2'),('Q','Cf','C1')]:
            ax3.plot(*np.array([p[a],p[b]]).T,color=c,marker='o',lw=3)
        for wheel,col in [('r','C0'),('f','C1')]:
            keys=[f'P{i}{wheel}' for i in (1,2,3,4)]
            pts=np.array([p[k] for k in keys]); center=pts.mean(axis=0); u=pts[0]-center; v=pts[1]-center
            t=np.linspace(0,2*np.pi,100); ring=center[:,None]+np.linalg.norm(u)* (u[:,None]/np.linalg.norm(u)*np.cos(t)+v[:,None]/np.linalg.norm(v)*np.sin(t))
            ax3.plot(*ring,color=col); ax3.scatter(*center,color=col,s=20)
        xyz=np.array(list(p.values())); lo=xyz.min(axis=0)-.6; hi=xyz.max(axis=0)+.6
        ax3.set_xlim(lo[0],hi[0]); ax3.set_ylim(lo[1],hi[1]); ax3.set_zlim(lo[2],hi[2]); ax3.set_box_aspect(np.maximum(hi-lo,.1))
        ax2.set_title('Pinhole projection (pixels)'); ax2.set_xlabel('u [px]'); ax2.set_ylabel('v [px]'); ax2.set_xlim(0,screen_resolution[0]); ax2.set_ylim(screen_resolution[1],0); ax2.set_aspect('equal',adjustable='box'); ax2.grid(True,alpha=.25)
        ax2.axvline(cam['cx'],color='.8',lw=.8); ax2.axhline(cam['cy'],color='.8',lw=.8)
        camera_pos=np.array([cam['cam_x'],cam['cam_y'],cam['cam_z']]); yaw,pitch,roll=cam['cam_yaw'],cam['cam_pitch'],cam['cam_roll']
        # Match the model camera axes: C.x right, C.y forward/depth, C.z up.
        n=__import__('sympy').physics.mechanics.ReferenceFrame('Nplot'); c=__import__('sympy').physics.mechanics.ReferenceFrame('Cplot'); c.orient_body_fixed(n,(yaw,pitch,roll),'ZYX')
        basis=np.array([[float(c.x.dot(a)) for a in (n.x,n.y,n.z)],[float(c.y.dot(a)) for a in (n.x,n.y,n.z)],[float(c.z.dot(a)) for a in (n.x,n.y,n.z)]])
        def project(pt):
            X,Y,Z=basis@(np.asarray(pt)-camera_pos)
            if Y<=1e-8: return np.nan,np.nan
            return cam['fx']*X/Y+cam['cx'], cam['cy']-cam['fy']*Z/Y
        for a,b,c0 in [('Cr','S','C0'),('S','Q','C2'),('Q','Cf','C1')]:
            uv=np.array([project(p[a]),project(p[b])]); ax2.plot(uv[:,0],uv[:,1],'-o',color=c0,lw=2)
        for wheel,col in [('r','C0'),('f','C1')]:
            pts=np.array([p[f'P{i}{wheel}'] for i in (1,2,3,4,1)]); uv=np.array([project(v) for v in pts]); ax2.plot(uv[:,0],uv[:,1],color=col,lw=1.5)
        if image_points:
            for cls,col,name in [(0,'red','measured front'),(1,'blue','measured rear')]:
                if cls in image_points:
                    q=np.asarray(image_points[cls],float).copy()
                    # Inputs from project are normalized annotations (origin top-left); pixel arrays remain pixel coordinates.
                    if q.size and np.nanmax(q)<=1.5: q[:,0]*=screen_resolution[0]; q[:,1]=(1-q[:,1])*screen_resolution[1]
                    ax2.scatter(q[:,0],q[:,1],s=13,facecolors='none',edgecolors=col,label=name)
            ax2.legend(loc='best',fontsize=8)
        try:
            res=predict(vals,bike,cam); text='  '.join(f'{k}: {res[k]:.1f}' for k in ('u_Cr','v_Cr','r_Cf_Cr_u','r_Cf_Cr_v'))
        except Exception as e: text=f'Projection invalid: {e}'
        ax2.text(.01,.02,text,transform=ax2.transAxes,fontsize=8,va='bottom')
        fig.canvas.draw_idle()
    rows=[]
    for i,(label,v,(mn,mx)) in enumerate(zip(labels,vals,bounds)):
        y=.255-i*.027; a=fig.add_axes([.18,y,.30,.015]); sliders[label]=Slider(a,label,mn,mx,valinit=float(np.clip(v,mn,mx)),valfmt='%1.3f')
        sliders[label].on_changed(lambda z,idx=i: state_update(idx,z))
    state_update_ref=None
    def state_update(idx,z):
        vals[idx]=float(z); make_artists(); state_text.set_text('state = np.array(['+', '.join(f'{v:.8g}' for v in vals)+'])')
    cam_defs=[('cam_x',-10,10),('cam_y',-10,10),('cam_z',-10,10),('cam_yaw',-np.pi,np.pi),('cam_pitch',-np.pi/2,np.pi/2),('cam_roll',-np.pi,np.pi),('fx',100,20000),('fy',100,20000),('cx',0,screen_resolution[0]),('cy',0,screen_resolution[1])]
    # Compact camera controls in a second column beneath the plots.
    for i,(k,mn,mx) in enumerate(cam_defs):
        y=.255-i*.025; a=fig.add_axes([.62,y,.29,.014]); s=Slider(a,k,mn,mx,valinit=float(np.clip(cam[k],mn,mx)),valfmt='%1.2f'); sliders['cam_'+k]=s
        s.on_changed(lambda z,key=k: (cam.__setitem__(key,float(z)),make_artists()))
    state_text=fig.text(.05,.035,'state = np.array(['+', '.join(f'{v:.8g}' for v in vals)+'])',family='monospace',fontsize=9)
    def copy(event):
        text='state = np.array(['+', '.join(f'{v:.10g}' for v in vals)+'])'; state_text.set_text(text); fig.canvas.draw_idle()
        try:
            import tkinter as tk
            root=tk.Tk(); root.withdraw(); root.clipboard_clear(); root.clipboard_append(text); root.update(); root.destroy()
        except Exception: pass
    b_ax=fig.add_axes([.83,.025,.13,.04]); button=Button(b_ax,'Copy state') ; button.on_clicked(copy)
    make_artists(); fig._bike_sliders=sliders; fig._bike_button=button
    if show: plt.show()
    return fig,sliders

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--data'); args=ap.parse_args(); launch(image_points=({k:v for k,v in load_points(args.data).items()} if args.data else None))
