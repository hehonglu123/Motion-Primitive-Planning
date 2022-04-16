import numpy as np
from pandas import *
import sys, traceback
from general_robotics_toolbox import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
sys.path.append('../toolbox')
from robots_def import *



def main():
	robot=abb6640(d=50)
	# col_names=['X', 'Y', 'Z','direction_x','direction_y','direction_z'] 
	# data = read_csv("from_ge/Curve_backproj_in_base_frame.csv", names=col_names)
	# curve_x=data['X'].tolist()
	# curve_y=data['Y'].tolist()
	# curve_z=data['Z'].tolist()
	# curve_direction_x=data['direction_x'].tolist()
	# curve_direction_y=data['direction_y'].tolist()
	# curve_direction_z=data['direction_z'].tolist()

	# curve_new=np.dot(np.dot(Rz(np.pi/2),Rx(np.pi/2)),np.vstack((curve_x, curve_y, curve_z))).T+np.array([2700,-800,500])

	# col_names=['X', 'Y', 'Z','direction_x','direction_y','direction_z'] 
	# data = read_csv("original/Curve_in_base_frame.csv", names=col_names)
	# curve_x=data['X'].tolist()
	# curve_y=data['Y'].tolist()
	# curve_z=data['Z'].tolist()
	# curve_direction_x=data['direction_x'].tolist()
	# curve_direction_y=data['direction_y'].tolist()
	# curve_direction_z=data['direction_z'].tolist()

	# curve=np.vstack((curve_x, curve_y, curve_z)).T

	col_names=['q1', 'q2', 'q3','q4', 'q5', 'q6'] 
	data = read_csv('from_ge/qsol.csv', names=col_names)
	# data = read_csv('../greedy_fitting/curve_fit_js.csv',names=col_names)
	curve_q1=data['q1'].tolist()
	curve_q2=data['q2'].tolist()
	curve_q3=data['q3'].tolist()
	curve_q4=data['q4'].tolist()
	curve_q5=data['q5'].tolist()
	curve_q6=data['q6'].tolist()
	curve_js=np.vstack((curve_q1, curve_q2, curve_q3,curve_q4,curve_q5,curve_q6)).T
	curve=[]
	curve_normal=[]
	for i in range(len(curve_js)):
		pose_temp=robot.fwd(curve_js[i])
		curve.append(pose_temp.p)
		curve_normal.append(pose_temp.R[:,-1])
	curve=np.array(curve)
	curve_normal=np.array(curve_normal)




	plt.figure()
	ax = plt.axes(projection='3d')
	# ax.plot3D(curve_new[:,0], curve_new[:,1],curve_new[:,2], 'gray')
	ax.plot3D(curve[:,0], curve[:,1],curve[:,2], 'red')
	plt.show()

if __name__ == "__main__":
	main()