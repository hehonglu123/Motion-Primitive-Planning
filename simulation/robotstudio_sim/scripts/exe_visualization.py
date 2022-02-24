import matplotlib.pyplot as plt
from general_robotics_toolbox import *
from pandas import *
import sys, traceback
import numpy as np
sys.path.append('../../../toolbox')
from robots_def import *
from error_check import *
sys.path.append('../../../circular_fit')
from toolbox_circular_fit import *

def main():
	col_names=['X', 'Y', 'Z','direction_x', 'direction_y', 'direction_z'] 
	data = read_csv("../../../data/from_ge/Curve_backproj_in_base_frame.csv", names=col_names)
	curve_x=data['X'].tolist()
	curve_y=data['Y'].tolist()
	curve_z=data['Z'].tolist()
	curve=np.vstack((curve_x, curve_y, curve_z)).T

	data_dir="fitting_output/movej/"
	col_names=['X', 'Y', 'Z','direction_x', 'direction_y', 'direction_z'] 
	data = read_csv(data_dir+"curve_fit_backproj.csv")
	curve_x=data['x'].tolist()
	curve_y=data['y'].tolist()
	curve_z=data['z'].tolist()
	curve_fit=np.vstack((curve_x, curve_y, curve_z)).T

	###read in points backprojected
	col_names=['timestamp', 'cmd_num', 'J1', 'J2','J3', 'J4', 'J5', 'J6'] 
	data = read_csv(data_dir+"curve_exe_v50_fine.csv",names=col_names)
	q1=data['J1'].tolist()[1:]
	q2=data['J2'].tolist()[1:]
	q3=data['J3'].tolist()[1:]
	q4=data['J4'].tolist()[1:]
	q5=data['J5'].tolist()[1:]
	q6=data['J6'].tolist()[1:]
	
	cmd_num=np.array(data['cmd_num'].tolist()[1:]).astype(float)
	start_idx=np.where(cmd_num==3)[0][0]
	timestamp=np.array(data['timestamp'].tolist()[1:]).astype(float)[start_idx:]
	curve_exe_js=np.vstack((q1,q2,q3,q4,q5,q6)).T.astype(float)[start_idx:]

	data = read_csv(data_dir+"command_backproj.csv")
	breakpoints=np.array(data['breakpoints'].tolist())
	primitives=data['primitives'].tolist()
	points=data['points'].tolist()
	####only every 100 points
	steps=10
	curve=curve[::steps]
	curve_fit=curve_fit[::steps]
	breakpoints=breakpoints/steps

	curve_exe=[]
	curve_exe_R=[]
	robot=abb6640()
	for i in range(len(curve_exe_js)):
		robot_pose=robot.fwd(np.radians(curve_exe_js[i]))
		curve_exe.append(robot_pose.p)
		curve_exe_R.append(robot_pose.R)
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
	plt.scatter(curve_fit_2d_vis[breakpoints.astype(int),0],curve_fit_2d_vis[breakpoints.astype(int),1])
	plt.plot(curve_exe_2d_vis[:,0],curve_exe_2d_vis[:,1])
	plt.legend(['original curve','curve fit','curve execution'])



	# fig = plt.figure()
	# ax = plt.axes(projection='3d')
	# ax.plot3D(curve[:,0], curve[:,1], curve[:,2],label='original',c='gray')
	# ax.plot3D(curve_fit[:,0], curve_fit[:,1], curve_fit[:,2],label='curve_fit',c='red')
	# # ax.plot3D(curve_exe[:,0], curve_exe[:,1], curve_exe[:,2], 'blue')
	# ax.legend()

	


	plt.show()


if __name__ == "__main__":
	main()