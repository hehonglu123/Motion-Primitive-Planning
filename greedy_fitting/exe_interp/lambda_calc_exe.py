import matplotlib.pyplot as plt
from general_robotics_toolbox import *
from pandas import *
import sys, traceback
import numpy as np
sys.path.append('../../toolbox')
from robot_def import *
from lambda_calc import *
import matplotlib.pyplot as plt


def main():

	col_names=['q1', 'q2', 'q3','q4', 'q5', 'q6'] 
	data = read_csv("Curve_js.csv", names=col_names)
	curve_q1=data['q1'].tolist()
	curve_q2=data['q2'].tolist()
	curve_q3=data['q3'].tolist()
	curve_q4=data['q4'].tolist()
	curve_q5=data['q5'].tolist()
	curve_q6=data['q6'].tolist()
	curve_js=np.vstack((curve_q1, curve_q2, curve_q3,curve_q4,curve_q5,curve_q6)).T

	data = read_csv("../comparison/moveL+moveC/threshold1/curve_fit_backproj.csv")
	curve_x=data['x'].tolist()
	curve_y=data['y'].tolist()
	curve_z=data['z'].tolist()
	curve_fit=np.vstack((curve_x, curve_y, curve_z)).T

	lam=calc_lam(curve_fit)
	joint_vel_limit=np.radians([110,90,90,150,120,235])
	dlam_max,idx=calc_lamdot_acc_constraints(curve_js,lam,joint_vel_limit,10*joint_vel_limit,np.array([1,9064,15496,19997,25085,31478,38876,44918,47784,50007])-1,100)

	plt.plot(lam[idx],dlam_max,label="lambda_dot_max")
	plt.xlabel("lambda")
	plt.ylabel("lambda_dot")
	plt.ylim([0.5,3.5])
	plt.title("max lambda_dot vs lambda (path index)")
	plt.show()

if __name__ == "__main__":
	main()