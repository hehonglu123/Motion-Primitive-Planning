import numpy as np
from pandas import *
import sys, traceback
from general_robotics_toolbox import *
import matplotlib.pyplot as plt

sys.path.append('../toolbox')
from robot_def import *

def find_dlam_max(start,end):
	global joint_vel_limit, curve, curve_js, curve_R, breakpoints,lam

	R_diff=np.dot(curve_R[end],curve_R[start].T)
	k,theta=R2rot(R_diff)
	k=np.array(k)
	###constant during each segment

	moving_direction=np.hstack((theta*k,(curve[end]-curve[start])))/(lam[end]-lam[start])

	dlam_max=[]
	ddlam_max=[]
	dlam_act=[0]

	for r in range(start,end):

		J=jacobian(curve_js[r])

		dq=np.abs(curve_js[r+1]-curve_js[r])
		t=np.max(dq/joint_vel_limit)
		qd_max=(curve_js[r+1]-curve_js[r])/t

		temp=np.dot(J,qd_max)
		temp[3:]*=1000.
		### a little variation here, so choose the average
		dlam_max.append(np.average(temp[3:]/moving_direction[3:]))

	return dlam_max

def main():
	global curve, curve_R, curve_js, breakpoints, joint_vel_limit, lam

	col_names=['X', 'Y', 'Z','direction_x','direction_y','direction_z'] 
	data = read_csv("../data/execution/Curve_moveL.csv", names=col_names)
	curve_x=data['X'].tolist()
	curve_y=data['Y'].tolist()
	curve_z=data['Z'].tolist()
	curve=np.vstack((curve_x, curve_y, curve_z)).T
	curve_R=np.load("../data/execution/Curve_moveL.npy")

	col_names=['q1', 'q2', 'q3','q4', 'q5', 'q6'] 
	data = read_csv("../data/execution/Curve_moveL_js.csv", names=col_names)
	curve_q1=data['q1'].tolist()
	curve_q2=data['q2'].tolist()
	curve_q3=data['q3'].tolist()
	curve_q4=data['q4'].tolist()
	curve_q5=data['q5'].tolist()
	curve_q6=data['q6'].tolist()
	curve_js=np.vstack((curve_q1, curve_q2, curve_q3,curve_q4,curve_q5,curve_q6)).T
	breakpoints=[0, 427, 853, 1278, 1714, 2166, 2643, 3154, 3713, 4344, 5095, 5985, 7404, 8367, 9128, 9780, 10360, 10890, 11383, 11850, 12300, 12740, 13180, 13581, 13990, 14415, 14860, 15332, 15835, 16376, 16944, 17580]

	###find path length
	lam=[0]
	for i in range(len(curve)-1):
		lam.append(lam[-1]+np.linalg.norm(curve[i+1]-curve[i]))
	###normalize lam
	lam=np.array(lam)/lam[-1]

	joint_vel_limit=np.radians([110,90,90,150,120,235])
	dlam_max=[]

	for i in range(len(breakpoints)-1):
		dlam_max+=find_dlam_max(breakpoints[i],breakpoints[i+1])

	plt.plot(lam[1:],dlam_max)
	plt.xlabel("lambda")
	plt.ylabel("lambda_dot")
	plt.title("lambda_dot vs lambda (path index)")
	plt.savefig("velocity-constraint_cs.png")
	plt.show()



if __name__ == "__main__":
	main()