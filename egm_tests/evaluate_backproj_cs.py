import matplotlib.pyplot as plt
from general_robotics_toolbox import *
from pandas import *
import sys, traceback
import numpy as np
sys.path.append('../toolbox')
from robot_def import *
from error_check import *

def eval(curve_exe,curve_exe_ori,curve,curve_backproj):
	d=50
	curve_proj=[]
	for i in range(len(curve_exe)):
		curve_proj.append(curve_exe[i]+d*q2R(curve_exe_ori[i])[:,-1])
	max_error1=calc_max_error(curve_exe,curve_backproj)
	max_error2=calc_max_error(curve_proj,curve)
	avg_error1=calc_avg_error(curve_exe,curve_backproj)
	avg_error2=calc_avg_error(curve_proj,curve)
	return max_error1, max_error2, avg_error1,avg_error2


def main():
	col_names=['timestamp','x', 'y','z','q1','q2','q3','q4'] 
	data = read_csv("execution_egm_cs.csv",names=col_names)
	data = data.apply(to_numeric, errors='coerce')
	timestamp=data['timestamp'].tolist()
	x=data['x'].tolist()
	y=data['y'].tolist()
	z=data['z'].tolist()
	q1=data['q1'].tolist()
	q2=data['q2'].tolist()
	q3=data['q3'].tolist()
	q4=data['q4'].tolist()
	curve_exe=np.vstack((x,y,z)).T
	curve_exe_ori=np.vstack((q1,q2,q3,q4)).T

	col_names=['X', 'Y', 'Z','direction_x', 'direction_y', 'direction_z'] 
	data = read_csv("../data/from_cad/Curve_in_base_frame.csv", names=col_names)
	curve_x=data['X'].tolist()
	curve_y=data['Y'].tolist()
	curve_z=data['Z'].tolist()
	curve=np.vstack((curve_x, curve_y, curve_z)).T

	###read in points backprojected
	col_names=['X', 'Y', 'Z','direction_x', 'direction_y', 'direction_z'] 
	data = read_csv("../data/from_cad/Curve_backproj_in_base_frame.csv", names=col_names)
	curve_backproj_x=data['X'].tolist()
	curve_backproj_y=data['Y'].tolist()
	curve_backproj_z=data['Z'].tolist()
	curve_backproj=np.vstack((curve_backproj_x, curve_backproj_y, curve_backproj_z)).T


	###find start configuration (RS recording start when button pressed)
	start_idx=0

	max_error1, max_error2, avg_error1,avg_error2=eval(curve_exe,curve_exe_ori,curve,curve_backproj)
	print(max_error1,max_error2, avg_error1, avg_error2)
	print('time: ',timestamp[-1]-timestamp[start_idx])

if __name__ == "__main__":
	main()