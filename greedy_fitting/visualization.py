import matplotlib.pyplot as plt
from general_robotics_toolbox import *
from pandas import *
import sys, traceback
import numpy as np
sys.path.append('../toolbox')
from robot_def import *
from error_check import *
sys.path.append('../circular_fit')
from toolbox_circular_fit import *

def main():
	col_names=['X', 'Y', 'Z','direction_x', 'direction_y', 'direction_z'] 
	data = read_csv("../data/from_cad/Curve_backproj_in_base_frame.csv", names=col_names)
	curve_x=data['X'].tolist()
	curve_y=data['Y'].tolist()
	curve_z=data['Z'].tolist()
	curve=np.vstack((curve_x, curve_y, curve_z)).T

	col_names=['X', 'Y', 'Z','direction_x', 'direction_y', 'direction_z'] 
	data = read_csv("comparison/moveL+moveC/threshold05/curve_fit_backproj.csv")
	curve_x=data['x'].tolist()
	curve_y=data['y'].tolist()
	curve_z=data['z'].tolist()
	curve_fit=np.vstack((curve_x, curve_y, curve_z)).T

	###read in points backprojected
	col_names=['timestamp', 'J1', 'J2','J3', 'J4', 'J5', 'J6'] 
	data = read_csv("comparison/moveL+moveC/threshold1/v500_fine.csv",names=col_names)
	data = data.apply(to_numeric, errors='coerce')
	q1=data['J1'].tolist()[1:]
	q2=data['J2'].tolist()[1:]
	q3=data['J3'].tolist()[1:]
	q4=data['J4'].tolist()[1:]
	q5=data['J5'].tolist()[1:]
	q6=data['J6'].tolist()[1:]
	timestamp=data['timestamp'].tolist()[1:]
	q_all=np.vstack((q1,q2,q3,q4,q5,q6)).T

	###find start configuration (RS recording start when button pressed)
	# dist=np.linalg.norm(q_all-np.tile(np.degrees([ 0.62750007,  0.17975177,  0.51961085,  1.60530199, -0.89342989,
	# 	0.91741297]),(len(q_all),1)),axis=1)
	# start_idx=np.argsort(dist)[0]
	start_idx=0
	q_all=np.radians(q_all)
	curve_exe=[]
	for q in q_all[start_idx:]:
		pose=fwd(q)
		curve_exe.append(pose.p)
	curve_exe=np.array(curve_exe)


	###plane projection visualization
	curve_mean = curve.mean(axis=0)
	curve_centered = curve - curve_mean
	U,s,V = np.linalg.svd(curve_centered)
	# Normal vector of fitting plane is given by 3rd column in V
	# Note linalg.svd returns V^T, so we need to select 3rd row from V^T
	normal = V[2,:]

	curve_2d_vis = rodrigues_rot(curve_centered, normal, [0,0,1])[:,:2]
	curve_fit_2d_vis = rodrigues_rot(curve_fit-curve_mean, normal, [0,0,1])[:,:2]
	curve_exe_2d_vis = rodrigues_rot(curve_exe-curve_mean, normal, [0,0,1])[:,:2]
	plt.plot(curve_2d_vis[:,0],curve_2d_vis[:,1])
	plt.plot(curve_fit_2d_vis[:,0],curve_fit_2d_vis[:,1])
	plt.plot(curve_exe_2d_vis[:,0],curve_exe_2d_vis[:,1])
	plt.legend(['original curve','curve fit','curve execution'])



	# fig = plt.figure()
	# ax = plt.axes(projection='3d')
	# ax.plot3D(curve[:,0], curve[:,1], curve[:,2], 'gray')
	# ax.plot3D(curve_fit[:,0], curve_fit[:,1], curve_fit[:,2], 'red')
	# ax.plot3D(curve_exe[:,0], curve_exe[:,1], curve_exe[:,2], 'blue')
	


	plt.show()


if __name__ == "__main__":
	main()