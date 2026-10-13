"""Launch viewer using constants from the project; optionally pass one label file."""
import argparse
from interactive_bike_viewer import launch, load_points

p=argparse.ArgumentParser()
p.add_argument('--data', help='One YOLO-format frame label file')
p.add_argument('--width',type=int,default=1920)
p.add_argument('--height',type=int,default=1080)
a=p.parse_args()
resolution=(a.width,a.height)
bike={'lr':1.4,'lf1':0.5,'lf2':0.1,'rr':0.3302,'rf':0.3302}
image_resolution=(6020,4024); sensor_size=(0.0235,0.0156); focal_length=0.028
camera={'fx':focal_length*image_resolution[0]/sensor_size[0],
'fy':focal_length*image_resolution[1]/sensor_size[1],
'cx':image_resolution[0]/2,'cy':image_resolution[1]/2,
'cam_x':0.0,'cam_y':0.0,'cam_z':0.45,'cam_yaw':0.0,'cam_pitch':0.0,'cam_roll':0.0}
launch(bike,camera,[0,0,0,0,2.82,0.1,0],load_points(a.data),resolution)
