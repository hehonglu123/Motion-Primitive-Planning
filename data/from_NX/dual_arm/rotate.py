import sys, yaml, copy
import numpy as np
from general_robotics_toolbox import *
from pandas import read_csv, DataFrame

original_dir='diffevo_pose2/'
with open(original_dir+'abb1200.yaml') as file:
	abb1200_pose = np.array(yaml.safe_load(file)['H'],dtype=np.float64)
curve_js = read_csv(original_dir+'arm1.csv',header=None).values


rotate_angle=-np.pi/3

H_R=np.eye(4)
H_R[:3,:3]=rot([0,0,1],rotate_angle)
abb1200_pose_new=H_R@abb1200_pose

new_dir='diffevo_pose2_2/'
with open(new_dir+'abb1200.yaml', 'w') as file:
	documents = yaml.dump({'H':abb1200_pose_new.tolist()}, file)

curve_js_new=copy.deepcopy(curve_js)
curve_js_new[:,0]=curve_js[:,0]+rotate_angle


df=DataFrame({'q0':curve_js_new[:,0],'q1':curve_js_new[:,1],'q2':curve_js_new[:,2],'q3':curve_js_new[:,3],'q4':curve_js_new[:,4],'q5':curve_js_new[:,5]})
df.to_csv(new_dir+'arm1.csv',header=False,index=False)
