# This is the code when I assumed the angles and the last two lines are to find
# the intersection of both frames.

imgdata = {
    'r_Cf_Cr_x': x_f-x_r,
    'r_Cf_Cr_z': y_f-y_r,
    'r_Cr_O_x': x_r - assumed_origin[0],
    'r_Cr_O_z': y_r - assumed_origin[1],
    'r_P_S_x': (np.sin(cam_angle_f)*np.cos(cam_angle_r) - np.sin(cam_angle_r)*np.cos(cam_angle_f)*np.cos(theta_f - theta_r))*np.sin(theta_f)/np.sqrt((np.sin(cam_angle_f)*np.cos(cam_angle_r) - np.sin(cam_angle_r)*np.cos(cam_angle_f)*np.cos(theta_f - theta_r))**2 + np.sin(cam_angle_r)**2*np.sin(theta_f - theta_r)**2) + np.sin(cam_angle_r)*np.sin(theta_f - theta_r)*np.cos(cam_angle_f)*np.cos(theta_f)/np.sqrt((np.sin(cam_angle_f)*np.cos(cam_angle_r) - np.sin(cam_angle_r)*np.cos(cam_angle_f)*np.cos(theta_f - theta_r))**2 + np.sin(cam_angle_r)**2*np.sin(theta_f - theta_r)**2),
    'r_P_S_z': (np.sin(cam_angle_f)*np.cos(cam_angle_r) - np.sin(cam_angle_r)*np.cos(cam_angle_f)*np.cos(theta_f - theta_r))*np.cos(theta_f)/np.sqrt((np.sin(cam_angle_f)*np.cos(cam_angle_r) - np.sin(cam_angle_r)*np.cos(cam_angle_f)*np.cos(theta_f - theta_r))**2 + np.sin(cam_angle_r)**2*np.sin(theta_f - theta_r)**2) - np.sin(cam_angle_r)*np.sin(theta_f)*np.sin(theta_f - theta_r)*np.cos(cam_angle_f)/np.sqrt((np.sin(cam_angle_f)*np.cos(cam_angle_r) - np.sin(cam_angle_r)*np.cos(cam_angle_f)*np.cos(theta_f - theta_r))**2 + np.sin(cam_angle_r)**2*np.sin(theta_f - theta_r)**2)
}