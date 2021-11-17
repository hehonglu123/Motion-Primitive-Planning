from general_robotics_toolbox import *
from pandas import *
import sys, traceback
import numpy as np
sys.path.append('../toolbox')
from robot_def import *

def rotate_curve(curve,curve_direction,R,p):
	curve_new=np.dot(R,curve.T).T+p
	curve_direction_new=np.dot(R,curve_direction.T).T
	return curve_new,curve_direction_new

def main():
	col_names=['X', 'Y', 'Z','direction_x','direction_y','direction_z'] 
	data = read_csv("curve_poses/curve_pose0/Curve_backproj_in_base_frame.csv", names=col_names)
	curve_x=data['X'].tolist()
	curve_y=data['Y'].tolist()
	curve_z=data['Z'].tolist()
	curve_direction_x=data['direction_x'].tolist()
	curve_direction_y=data['direction_y'].tolist()
	curve_direction_z=data['direction_z'].tolist()

	curve=np.vstack((curve_x, curve_y, curve_z)).T
	curve_direction=np.vstack((curve_direction_x, curve_direction_y, curve_direction_z)).T

	curve_new,curve_direction_new=rotate_curve(curve,curve_direction,Rx(np.pi/2),np.array([0,0,0]))

	df=DataFrame({'x':curve_new[:,0],'y':curve_new[:,1], 'z':curve_new[:,2],'x_direction':curve_direction_new[:,0],'y_direction':curve_direction_new[:,1],'z_direction':curve_direction_new[:,2]})
	df.to_csv("curve_poses/curve_pose3/Curve_backproj_in_base_frame.csv",header=False,index=False)


if __name__ == "__main__":
	main()