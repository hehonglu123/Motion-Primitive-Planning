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
	data = read_csv("Curve_new_mm.csv", names=col_names)
	curve_x=data['X'].tolist()
	curve_y=data['Y'].tolist()
	curve_z=data['Z'].tolist()
	curve_direction_x=data['direction_x'].tolist()
	curve_direction_y=data['direction_y'].tolist()
	curve_direction_z=data['direction_z'].tolist()

	curve_new=np.dot(np.dot(Rz(np.pi/2),Rx(np.pi/2)),np.vstack((curve_x, curve_y, curve_z))).T+np.array([2700,-800,500])

	col_names=['X', 'Y', 'Z','direction_x','direction_y','direction_z'] 
	data = read_csv("original/Curve_in_base_frame.csv", names=col_names)
	curve_x=data['X'].tolist()
	curve_y=data['Y'].tolist()
	curve_z=data['Z'].tolist()
	curve_direction_x=data['direction_x'].tolist()
	curve_direction_y=data['direction_y'].tolist()
	curve_direction_z=data['direction_z'].tolist()

	curve=np.vstack((curve_x, curve_y, curve_z)).T




	plt.figure()
	ax = plt.axes(projection='3d')
	ax.plot3D(curve_new[:,0], curve_new[:,1],curve_new[:,2], 'gray')
	ax.plot3D(curve[:,0], curve[:,1],curve[:,2], 'red')
	plt.show()

if __name__ == "__main__":
	main()