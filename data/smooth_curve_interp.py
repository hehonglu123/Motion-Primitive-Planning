import numpy as np
from pandas import *
import sys, traceback
from general_robotics_toolbox import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
sys.path.append('../toolbox')
from robot_def import *



def main():

	col_names=['X', 'Y', 'Z','direction_x','direction_y','direction_z'] 
	data = read_csv("Curve_dense_new_mm.csv", names=col_names)
	curve_x=data['X'].tolist()
	curve_y=data['Y'].tolist()
	curve_z=data['Z'].tolist()

	curve_new=np.dot(np.dot(Rz(np.pi/2),Rx(np.pi/2)),np.vstack((curve_x, curve_y, curve_z))).T+np.array([2700,-800,500])
	curve_direction_new=[]

	col_names=['X', 'Y', 'Z','direction_x','direction_y','direction_z'] 
	data = read_csv("original/Curve_in_base_frame.csv", names=col_names)
	curve_x=data['X'].tolist()
	curve_y=data['Y'].tolist()
	curve_z=data['Z'].tolist()
	curve_direction_x=data['direction_x'].tolist()
	curve_direction_y=data['direction_y'].tolist()
	curve_direction_z=data['direction_z'].tolist()

	curve=np.vstack((curve_x, curve_y, curve_z)).T
	curve_direction=np.vstack((curve_direction_x,curve_direction_y,curve_direction_z)).T

	###find point with corresponding curve normal
	direction_idx=np.zeros(len(curve))
	for i in range(len(curve)):
		direction_idx[i]=np.argmin(np.linalg.norm(curve_new-curve[i],axis=1))

	idx=0
	###interpolate curve normal
	for i in range(len(curve_new)):
		if i in direction_idx:
			curve_direction_new.append(curve_direction[idx])
			if i==len(curve_new)-1:
				break
			idx+=1
			axis=np.cross(curve_direction[idx-1],curve_direction[idx])
			axis=axis/np.linalg.norm(axis)
			angle=np.arctan2(np.linalg.norm(np.cross(curve_direction[idx-1],curve_direction[idx])), np.dot(curve_direction[idx-1],curve_direction[idx]))
			
			continue

		angle_interp=angle*(i-direction_idx[idx-1])/(direction_idx[idx]-direction_idx[idx-1])
		R_interp=rot(axis,angle_interp)
		curve_direction_new.append(np.dot(R_interp,curve_direction[idx-1]))


	curve_direction_new=np.array(curve_direction_new)
	df=DataFrame({'x':curve_new[:,0],'y':curve_new[:,1], 'z':curve_new[:,2],'x_direction':curve_direction_new[:,0],'y_direction':curve_direction_new[:,1],'z_direction':curve_direction_new[:,2]})
	df.to_csv("from_ge/Curve_in_base_frame.csv",header=False,index=False)

if __name__ == "__main__":
	main()