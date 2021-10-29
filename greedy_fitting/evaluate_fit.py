import matplotlib.pyplot as plt
from general_robotics_toolbox import *
from pandas import *
import sys, traceback
import numpy as np
sys.path.append('../toolbox')
from robot_def import *
from error_check import *

def eval(q_all,curve):
	q_all=np.radians(q_all)
	d=50
	curve_fit=[]
	for q in q_all:
		pose=fwd(q)
		curve_fit.append(pose.p)
	max_error=calc_max_error(curve_fit,curve)

	print('max error: ',max_error)


def main():

	data = read_excel("fit_recordings.xlsx")
	q1=data['J1'].tolist()
	q2=data['J2'].tolist()
	q3=data['J3'].tolist()
	q4=data['J4'].tolist()
	q5=data['J5'].tolist()
	q6=data['J6'].tolist()
	q_all=np.vstack((q1,q2,q3,q4,q5,q6)).T

	col_names=['X', 'Y', 'Z','direction_x', 'direction_y', 'direction_z'] 
	data = read_csv("../data/from_cad/Curve_in_base_frame.csv", names=col_names)
	curve_x=data['X'].tolist()
	curve_y=data['Y'].tolist()
	curve_z=data['Z'].tolist()
	curve=np.vstack((curve_x, curve_y, curve_z)).T

	###read interpolated curves in joint space
	col_names=['q1', 'q2', 'q3','q4', 'q5', 'q6'] 
	data = read_csv("../data/from_cad/Curve_js.csv", names=col_names)
	curve_q1=data['q1'].tolist()
	curve_q2=data['q2'].tolist()
	curve_q3=data['q3'].tolist()
	curve_q4=data['q4'].tolist()
	curve_q5=data['q5'].tolist()
	curve_q6=data['q6'].tolist()
	curve_js=np.vstack((curve_q1, curve_q2, curve_q3,curve_q4,curve_q5,curve_q6)).T


	###find start configuration (RS recording start when button pressed)
	dist=np.linalg.norm(q_all-np.tile(np.degrees([ 0.61807527,  0.21784529,  0.48718426,  1.58392987, -0.89707819,
        0.90722028]),(len(q_all),1)),axis=1)
	start_idx=np.argsort(dist)[0]

	eval(q_all[start_idx:],curve)

if __name__ == "__main__":
	main()