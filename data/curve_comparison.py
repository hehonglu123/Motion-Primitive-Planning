import numpy as np
from pandas import *
import sys, traceback
from general_robotics_toolbox import *
sys.path.append('../toolbox')
from robots_def import *
from utils import *

def main():

	col_names=['X', 'Y', 'Z','direction_x','direction_y','direction_z'] 
	data = read_csv("from_ge/Curve_in_base_frame2.csv", names=col_names)
	curve_x=data['X'].tolist()
	curve_y=data['Y'].tolist()
	curve_z=data['Z'].tolist()
	curve_direction_x=data['direction_x'].tolist()
	curve_direction_y=data['direction_y'].tolist()
	curve_direction_z=data['direction_z'].tolist()

	curve=np.vstack((curve_x, curve_y, curve_z)).T
	curve_direction=np.vstack((curve_direction_x, curve_direction_y, curve_direction_z)).T


	abb6640_obj=abb6640(d=50)

	col_names=['q1', 'q2', 'q3','q4', 'q5', 'q6'] 
	# data = read_csv("../data/from_ge/Curve_js2.csv", names=col_names)
	data = read_csv("from_Jon/qbestcurve_new.csv", names=col_names)
	curve_q1=data['q1'].tolist()
	curve_q2=data['q2'].tolist()
	curve_q3=data['q3'].tolist()
	curve_q4=data['q4'].tolist()
	curve_q5=data['q5'].tolist()
	curve_q6=data['q6'].tolist()
	curve_js=np.vstack((curve_q1, curve_q2, curve_q3,curve_q4,curve_q5,curve_q6)).T

	dis_error=[]
	ori_error=[]
	for i in range(len(curve_js)):
		pose=abb6640_obj.fwd(curve_js[i])
		dis_error.append(np.linalg.norm(pose.p-curve[i]))
		ori_error.append(get_angle(pose.R[:,-1],curve_direction[i]))

	print(np.max(dis_error),np.max(ori_error))


if __name__ == "__main__":
	main()