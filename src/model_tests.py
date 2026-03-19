from bike_model import *
import matplotlib.pyplot as plt
import numpy as np


r_S_O_x = S.pos_from(O).dot(N.x)
r_S_O_y = S.pos_from(O).dot(N.y)
r_S_O_z = S.pos_from(O).dot(N.z)

r_Q_O_x = Q.pos_from(O).dot(N.x)
r_Q_O_y = Q.pos_from(O).dot(N.y)
r_Q_O_z = Q.pos_from(O).dot(N.z)

r_Cf_O_x = Cf.pos_from(O).dot(N.x)
r_Cf_O_y = Cf.pos_from(O).dot(N.y)
r_Cf_O_z = Cf.pos_from(O).dot(N.z)

r_P1r_O_x = P1r.pos_from(O).dot(N.x)
r_P1r_O_y = P1r.pos_from(O).dot(N.y)
r_P1r_O_z = P1r.pos_from(O).dot(N.z)

r_P1f_O_x = P1f.pos_from(O).dot(N.x)
r_P1f_O_y = P1f.pos_from(O).dot(N.y)
r_P1f_O_z = P1f.pos_from(O).dot(N.z)

r_P2r_O_x = P2r.pos_from(O).dot(N.x)
r_P2r_O_y = P2r.pos_from(O).dot(N.y)
r_P2r_O_z = P2r.pos_from(O).dot(N.z)

r_P2f_O_x = P2f.pos_from(O).dot(N.x)
r_P2f_O_y = P2f.pos_from(O).dot(N.y)
r_P2f_O_z = P2f.pos_from(O).dot(N.z)

r_P3r_O_x = P3r.pos_from(O).dot(N.x)
r_P3r_O_y = P3r.pos_from(O).dot(N.y)
r_P3r_O_z = P3r.pos_from(O).dot(N.z)

r_P3f_O_x = P3f.pos_from(O).dot(N.x)
r_P3f_O_y = P3f.pos_from(O).dot(N.y)
r_P3f_O_z = P3f.pos_from(O).dot(N.z)

r_P4r_O_x = P4r.pos_from(O).dot(N.x)
r_P4r_O_y = P4r.pos_from(O).dot(N.y)
r_P4r_O_z = P4r.pos_from(O).dot(N.z)

r_P4f_O_x = P4f.pos_from(O).dot(N.x)
r_P4f_O_y = P4f.pos_from(O).dot(N.y)
r_P4f_O_z = P4f.pos_from(O).dot(N.z)



r_Cr_O_y = Cr.pos_from(O).dot(N.y)

eval_rear_wheel_y = sp.lambdify((phi, theta, psi, delta, x_r, y_r, z_r, lr, lf1, lf2, r), r_Cr_O_y)

eval_steering_x = sp.lambdify((phi, theta, psi, delta, x_r, y_r, z_r, lr, lf1, lf2, r), r_S_O_x)
eval_steering_y = sp.lambdify((phi, theta, psi, delta, x_r, y_r, z_r, lr, lf1, lf2, r), r_S_O_y)
eval_steering_z = sp.lambdify((phi, theta, psi, delta, x_r, y_r, z_r, lr, lf1, lf2, r), r_S_O_z)

eval_fork_x = sp.lambdify((phi, theta, psi, delta, x_r, y_r, z_r, lr, lf1, lf2, r), r_Q_O_x)
eval_fork_y = sp.lambdify((phi, theta, psi, delta, x_r, y_r, z_r, lr, lf1, lf2, r), r_Q_O_y)
eval_fork_z = sp.lambdify((phi, theta, psi, delta, x_r, y_r, z_r, lr, lf1, lf2, r), r_Q_O_z)

eval_trail_x = sp.lambdify((phi, theta, psi, delta, x_r, y_r, z_r, lr, lf1, lf2, r), r_Cf_O_x)
eval_trail_y = sp.lambdify((phi, theta, psi, delta, x_r, y_r, z_r, lr, lf1, lf2, r), r_Cf_O_y)
eval_trail_z = sp.lambdify((phi, theta, psi, delta, x_r, y_r, z_r, lr, lf1, lf2, r), r_Cf_O_z)

eval_rear_contact_x = sp.lambdify((phi, theta, psi, delta, x_r, y_r, z_r, lr, lf1, lf2, r), r_P3r_O_x)
eval_rear_contact_y = sp.lambdify((phi, theta, psi, delta, x_r, y_r, z_r, lr, lf1, lf2, r), r_P3r_O_y)
eval_rear_contact_z = sp.lambdify((phi, theta, psi, delta, x_r, y_r, z_r, lr, lf1, lf2, r), r_P3r_O_z)

eval_front_contact_x = sp.lambdify((phi, theta, psi, delta, x_r, y_r, z_r, lr, lf1, lf2, r), r_P3f_O_x)
eval_front_contact_y = sp.lambdify((phi, theta, psi, delta, x_r, y_r, z_r, lr, lf1, lf2, r), r_P3f_O_y)
eval_front_contact_z = sp.lambdify((phi, theta, psi, delta, x_r, y_r, z_r, lr, lf1, lf2, r), r_P3f_O_z)

eval_rear_top_x = sp.lambdify((phi, theta, psi, delta, x_r, y_r, z_r, lr, lf1, lf2, r), r_P1r_O_x)
eval_rear_top_y = sp.lambdify((phi, theta, psi, delta, x_r, y_r, z_r, lr, lf1, lf2, r), r_P1r_O_y)
eval_rear_top_z = sp.lambdify((phi, theta, psi, delta, x_r, y_r, z_r, lr, lf1, lf2, r), r_P1r_O_z)

eval_front_top_x = sp.lambdify((phi, theta, psi, delta, x_r, y_r, z_r, lr, lf1, lf2, r), r_P1f_O_x)
eval_front_top_y = sp.lambdify((phi, theta, psi, delta, x_r, y_r, z_r, lr, lf1, lf2, r), r_P1f_O_y)
eval_front_top_z = sp.lambdify((phi, theta, psi, delta, x_r, y_r, z_r, lr, lf1, lf2, r), r_P1f_O_z)

eval_rear_p2_x = sp.lambdify((phi, theta, psi, delta, x_r, y_r, z_r, lr, lf1, lf2, r), r_P2r_O_x)
eval_rear_p2_y = sp.lambdify((phi, theta, psi, delta, x_r, y_r, z_r, lr, lf1, lf2, r), r_P2r_O_y)
eval_rear_p2_z = sp.lambdify((phi, theta, psi, delta, x_r, y_r, z_r, lr, lf1, lf2, r), r_P2r_O_z)

eval_front_p2_x = sp.lambdify((phi, theta, psi, delta, x_r, y_r, z_r, lr, lf1, lf2, r), r_P2f_O_x)
eval_front_p2_y = sp.lambdify((phi, theta, psi, delta, x_r, y_r, z_r, lr, lf1, lf2, r), r_P2f_O_y)
eval_front_p2_z = sp.lambdify((phi, theta, psi, delta, x_r, y_r, z_r, lr, lf1, lf2, r), r_P2f_O_z)

eval_rear_p4_x = sp.lambdify((phi, theta, psi, delta, x_r, y_r, z_r, lr, lf1, lf2, r), r_P4r_O_x)
eval_rear_p4_y = sp.lambdify((phi, theta, psi, delta, x_r, y_r, z_r, lr, lf1, lf2, r), r_P4r_O_y)
eval_rear_p4_z = sp.lambdify((phi, theta, psi, delta, x_r, y_r, z_r, lr, lf1, lf2, r), r_P4r_O_z)

eval_front_p4_x = sp.lambdify((phi, theta, psi, delta, x_r, y_r, z_r, lr, lf1, lf2, r), r_P4f_O_x)
eval_front_p4_y = sp.lambdify((phi, theta, psi, delta, x_r, y_r, z_r, lr, lf1, lf2, r), r_P4f_O_y)
eval_front_p4_z = sp.lambdify((phi, theta, psi, delta, x_r, y_r, z_r, lr, lf1, lf2, r), r_P4f_O_z)


# subs = (0, 0, 0, 0, 500, 0, 355.6, 1000, 500, 100, 355.6)
# subs = (0, 0, 0, 0, 0, 0, 0, 1000, 500, 100, 500)


subs = (np.deg2rad(-10), 
        np.deg2rad(-21.8014125), 
        np.deg2rad(10), 
        np.deg2rad(25), 
        5, 
        0, 
        5, 
        4, 
        2, 
        1, 
        2)

# subs = (0, 
#         np.deg2rad(-21.8014125), 
#         0, 
#         0, 
#         5, 
#         0, 
#         5, 
#         4, 
#         2, 
#         1, 
#         2)



repl = {phi:0, theta:0, psi:0, delta:0, x_r:0, y_r:0, z_r:0, lr:4, lf1:2, 
        lf2:1, r:1}


rear_wheel = (eval_f03(*subs), eval_rear_wheel_y(*subs), eval_f04(*subs))
front_wheel = (eval_f01(*subs), eval_f02(*subs))

rw_contact = (eval_rear_contact_x(*subs), eval_rear_contact_y(*subs), eval_rear_contact_z(*subs))
fw_contact = (eval_front_contact_x(*subs), eval_front_contact_y(*subs), eval_front_contact_z(*subs))

rw_top = (eval_rear_top_x(*subs), eval_rear_top_y(*subs), eval_rear_top_z(*subs))
fw_top = (eval_front_top_x(*subs), eval_front_top_y(*subs), eval_front_top_z(*subs))

rw_p2 = (eval_rear_p2_x(*subs), eval_rear_p2_y(*subs), eval_rear_p2_z(*subs))
fw_p2 = (eval_front_p2_x(*subs), eval_front_p2_y(*subs), eval_front_p2_z(*subs))

rw_p4 = (eval_rear_p4_x(*subs), eval_rear_p4_y(*subs), eval_rear_p4_z(*subs))
fw_p4 = (eval_front_p4_x(*subs), eval_front_p4_y(*subs), eval_front_p4_z(*subs))

steering = (eval_steering_x(*subs), eval_steering_y(*subs), eval_steering_z(*subs))
fork = (eval_fork_x(*subs), eval_fork_y(*subs), eval_fork_z(*subs))
trail = (eval_trail_x(*subs), eval_trail_y(*subs), eval_trail_z(*subs))

x1, y1 = [rear_wheel[0], steering[0]], [rear_wheel[2], steering[2]]
x2, y2 = [steering[0], fork[0]], [steering[2], fork[2]]
x3, y3 = [fork[0], trail[0]], [fork[2], trail[2]]


# ----- Gemini plot -----

pts_r = np.array([
    [rw_top[0], rw_top[1], rw_top[2]],
    [rw_p2[0], rw_p2[1], rw_p2[2]],
    [rw_contact[0], rw_contact[1], rw_contact[2]],
    [rw_p4[0], rw_p4[1], rw_p4[2]]
])

# 2. Find the center (mean of the points)
center_circle_r = np.mean(pts_r, axis=0)

# 3. Find the radius (distance from center to the first point)
radius_circle_r = np.linalg.norm(pts_r[0] - center_circle_r)

# 4. Define the plane vectors
# We use the vector from center to point 1 as our first direction (u)
u_r = (pts_r[0] - center_circle_r) / np.linalg.norm(pts_r[0] - center_circle_r)

# We need a second vector (v) perpendicular to u and in the same plane.
# We can get the plane normal by crossing two vectors from the center to points.
w_r = np.cross(pts_r[0] - center_circle_r, pts_r[1] - center_circle_r) 
w_r /= np.linalg.norm(w_r) # Normal to the plane

v_r = np.cross(w_r, u_r) # Perpendicular to u within the plane

theta_circle_r = np.linspace(0, 2 * np.pi, 100)

# Calculate the circle points
# We multiply the unit vectors u and v by cos and sin to sweep the circle
circle_pts_r = np.array([center_circle_r + radius_circle_r * np.cos(t) * u_r + 
                         radius_circle_r * np.sin(t) * v_r for t in theta_circle_r])

# Extract X, Y, Z for plotting
cx_r, cy_r, cz_r = circle_pts_r[:, 0], circle_pts_r[:, 1], circle_pts_r[:, 2]


pts_f = np.array([
    [fw_top[0], fw_top[1], fw_top[2]],
    [fw_p2[0], fw_p2[1], fw_p2[2]],
    [fw_contact[0], fw_contact[1], fw_contact[2]],
    [fw_p4[0], fw_p4[1], fw_p4[2]]
])

center_circle_f = np.mean(pts_f, axis=0)

radius_circle_f = np.linalg.norm(pts_f[0] - center_circle_f)

u_f = (pts_f[0] - center_circle_f) / np.linalg.norm(pts_f[0] - center_circle_f)

w_f = np.cross(pts_f[0] - center_circle_f, pts_f[1] - center_circle_f) 
w_f /= np.linalg.norm(w_f)

v_f = np.cross(w_f, u_f)

theta_circle_f = np.linspace(0, 2 * np.pi, 100)

circle_pts_f = np.array([center_circle_f + radius_circle_f * np.cos(t) * u_f + 
                         radius_circle_f * np.sin(t) * v_f for t in theta_circle_f])

cx_f, cy_f, cz_f = circle_pts_f[:, 0], circle_pts_f[:, 1], circle_pts_f[:, 2]





# ----- 2d plot -----
plt.plot(x1, y1, marker='o', color='C0')
plt.plot(x2, y2, marker='o', color='C2')
plt.plot(x3, y3, marker='o', color='C1')
plt.scatter(rear_wheel[0], rear_wheel[2], facecolors='none', edgecolors='C0', s=100)
plt.scatter(trail[0], trail[2], facecolors='none', edgecolors='C1', s=100)
plt.scatter(rw_contact[0], rw_contact[2], facecolors='none', edgecolors='C8', s=50)
plt.scatter(fw_contact[0], fw_contact[2], facecolors='none', edgecolors='C8', s=50)
plt.scatter(rw_top[0], rw_top[2], facecolors='none', edgecolors='C9', s=50)
plt.scatter(fw_top[0], fw_top[2], facecolors='none', edgecolors='C9', s=50)

plt.xlim(0, 20)
plt.ylim(0, 20)
plt.gca().set_aspect('equal')
plt.grid()
plt.show()


# ----- 3d plot -----
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot([rear_wheel[0], steering[0]], 
        [rear_wheel[1], steering[1]], 
        [rear_wheel[2], steering[2]], marker='o', color='C0', label='Frame')
ax.plot([steering[0], fork[0]], 
        [steering[1], fork[1]], 
        [steering[2], fork[2]], marker='o', color='C2', label='Steering')
ax.plot([fork[0], trail[0]], 
        [fork[1], trail[1]], 
        [fork[2], trail[2]], marker='o', color='C1', label='Fork')

ax.scatter(rw_contact[0], rw_contact[1], rw_contact[2], marker='x', color='C0')
ax.scatter(fw_contact[0], fw_contact[1], fw_contact[2], marker='x', color='C1')
ax.scatter(rw_top[0], rw_top[1], rw_top[2], marker='x', color='C9')
ax.scatter(fw_top[0], fw_top[1], fw_top[2], marker='x', color='C9')
ax.scatter(rw_p2[0], rw_p2[1], rw_p2[2], marker='s', color='C3')
ax.scatter(rw_p4[0], rw_p4[1], rw_p4[2], marker='^', color='C8')
ax.scatter(fw_p2[0], fw_p2[1], fw_p2[2], marker='s', color='C3')
ax.scatter(fw_p4[0], fw_p4[1], fw_p4[2], marker='^', color='C8')

ax.plot(cx_r, cy_r, cz_r, color='black', linestyle='-')
ax.plot(cx_f, cy_f, cz_f, color='black', linestyle='-')



ax.set_xlim(-5, 10)
ax.set_ylim(-5, 10)
ax.set_zlim(-5, 10)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
plt.show()