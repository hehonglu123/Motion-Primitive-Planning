import matplotlib.pyplot as plt
from general_robotics_toolbox import *
from pandas import *
import sys, traceback
import numpy as np
sys.path.append('../toolbox')
from robot_def import *
from error_check import *

def eval(q_all,curve,curve_backproj):
	q_all=np.radians(q_all)
	d=50
	curve_proj=[]
	curve_exe=[]
	for q in q_all:
		pose=fwd(q)
		curve_exe.append(pose.p)
		curve_proj.append(pose.p+d*pose.R[:,-1])
	max_error1,max_error_idx1=calc_max_error(curve_exe,curve_backproj)
	max_error2,max_error_idx2=calc_max_error(curve_proj,curve)
	avg_error1=calc_avg_error(curve_exe,curve_backproj)
	avg_error2=calc_avg_error(curve_proj,curve)
	return max_error1, max_error2, avg_error1,avg_error2


def main():
	col_names=['timestamp', 'J1', 'J2','J3', 'J4', 'J5', 'J6'] 
	data = read_csv("comparison/moveL+moveC/threshold05/v500_fine.csv",names=col_names)
	data = data.apply(to_numeric, errors='coerce')
	q1=data['J1'].tolist()[1:]
	q2=data['J2'].tolist()[1:]
	q3=data['J3'].tolist()[1:]
	q4=data['J4'].tolist()[1:]
	q5=data['J5'].tolist()[1:]
	q6=data['J6'].tolist()[1:]
	timestamp=data['timestamp'].tolist()[1:]
	q_all=np.vstack((q1,q2,q3,q4,q5,q6)).T

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

	col_names=['q1', 'q2', 'q3','q4', 'q5', 'q6'] 
	data = read_csv("../data/from_cad/Curve_backproj_js.csv", names=col_names)
	curve_backproj_q1=data['q1'].tolist()
	curve_backproj_q2=data['q2'].tolist()
	curve_backproj_q3=data['q3'].tolist()
	curve_backproj_q4=data['q4'].tolist()
	curve_backproj_q5=data['q5'].tolist()
	curve_backproj_q6=data['q6'].tolist()
	curve_backproj_js=np.vstack((curve_backproj_q1, curve_backproj_q2, curve_backproj_q3,curve_backproj_q4,curve_backproj_q5,curve_backproj_q6)).T

	###find start configuration (RS recording start when button pressed)
	dist=np.linalg.norm(q_all-np.tile(np.array([ 35.952876472045666,10.291001528494611,29.77535782467298,91.98023628868943,-51.18973047515866,52.558538106882104]),(len(q_all),1)),axis=1)
	start_idx=np.argsort(dist)[0]

	max_error1, max_error2, avg_error1,avg_error2=eval(q_all[start_idx:],curve,curve_backproj)
	print(max_error1,max_error2, avg_error1, avg_error2)
	print('time: ',timestamp[-1]-timestamp[start_idx])

if __name__ == "__main__":
	main()