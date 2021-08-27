import numpy as np
from pandas import *
import sys, traceback
from general_robotics_toolbox import *
import matplotlib.pyplot as plt

sys.path.append('../toolbox')
from robot_def import *

def find_dlam_max(start,end):
	global joint_vel_limit, curve_js, breakpoints
	ps=fwd(curve_js[start])
	pe=fwd(curve_js[end])

	R_diff=np.dot(ps.R.T,pe.R)
	k,theta=R2rot(R_diff)
	k=np.array(k)
	moving_direction=np.hstack((theta*k,pe.p-ps.p))

	dlam_max=[]
	ddlam_max=[]
	dlam_act=[0]

	for r in range(start,end):

		J=jacobian(curve_js[r])

		# dq=(curve_js[r+1]-curve_js[r])
		# t=np.max(dq/joint_vel_limit)
		# qd_max=dq/t

		dlam_max.append(np.min(joint_vel_limit/np.abs(np.dot(np.linalg.inv(J),moving_direction))))

	return dlam_max

def main():
	col_names=['X', 'Y', 'Z','direction_x','direction_y','direction_z'] 
	data = read_csv("../data/execution/Curve_moveL.csv", names=col_names)
	curve_x=data['X'].tolist()
	curve_y=data['Y'].tolist()
	curve_z=data['Z'].tolist()
	curve_direction_x=data['direction_x'].tolist()
	curve_direction_y=data['direction_y'].tolist()
	curve_direction_z=data['direction_z'].tolist()
	curve=np.vstack((curve_x, curve_y, curve_z)).T
	curve_direction=np.vstack((curve_direction_x, curve_direction_y, curve_direction_z))

	col_names=['q1', 'q2', 'q3','q4', 'q5', 'q6'] 
	data = read_csv("../data/execution/Curve_moveL_js.csv", names=col_names)
	curve_q1=data['q1'].tolist()
	curve_q2=data['q2'].tolist()
	curve_q3=data['q3'].tolist()
	curve_q4=data['q4'].tolist()
	curve_q5=data['q5'].tolist()
	curve_q6=data['q6'].tolist()
	global curve_js, breakpoints, joint_vel_limit
	curve_js=np.vstack((curve_q1, curve_q2, curve_q3,curve_q4,curve_q5,curve_q6)).T
	breakpoints=[0, 427, 853, 1278, 1714, 2166, 2643, 3154, 3713, 4344, 5095, 5985, 7404, 8367, 9128, 9780, 10360, 10890, 11383, 11850, 12300, 12740, 13180, 13581, 13990, 14415, 14860, 15332, 15835, 16376, 16944, 17580]


	joint_vel_limit=np.radians([110,90,90,150,120,235])
	dlam_max=[]

	for i in range(len(breakpoints)-1):
		dlam_max+=find_dlam_max(breakpoints[i],breakpoints[i+1])

	lam=np.arange(0,len(curve_js)-1)/len(curve_js)
	plt.plot(lam,dlam_max)
	plt.xlabel("lambda")
	plt.ylabel("lambda_dot")
	plt.title("lambda_dot vs lambda (path index)")
	plt.savefig("velocity-constraint_cs.png")
	plt.show()



if __name__ == "__main__":
	main()