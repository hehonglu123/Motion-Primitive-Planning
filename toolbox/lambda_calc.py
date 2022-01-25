import numpy as np
from pandas import *
import sys, traceback
from general_robotics_toolbox import *
import matplotlib.pyplot as plt
from robot_def import *

def calc_lam(curve):
	###find path length
	lam=[0]
	for i in range(len(curve)-1):
		lam.append(lam[-1]+np.linalg.norm(curve[i+1]-curve[i]))
	###normalize lam, 
	return np.array(lam)/lam[-1]

def calc_lamdot(curve_js,lam,joint_vel_limit,step):
	############find maximum lambda dot vs lambda
	###curve_js: curve expressed in joint space in radians
	###lam: discrete lambda (path length), same shape as curve_js, from 0 to 1
	###joint_vel_limit: joint velocity limit in radians
	###step: step size used for dense curve

	dlam_max=[]

	for i in range(0,len(lam)-step,step):
		dq=np.abs(curve_js[i+step]-curve_js[i])
		dqdlam=dq/(lam[i+step]-lam[i])
		t=np.max(np.divide(dq,joint_vel_limit))

		qdot_max=dq/t 		###approximated max qdot
		dlam_max.append(qdot_max[0]/dqdlam[0])



	return dlam_max

def calc_lamdot_acc_constraints(curve_js,lam,joint_vel_limit,joint_acc_limit,breakpoints,step):
	############find maximum lambda dot vs lambda
	###curve_js: curve expressed in joint space in radians
	###lam: discrete lambda (path length), same shape as curve_js, from 0 to 1
	###joint_vel_limit: joint velocity limit in radians

	dlam_max=[]

	joint_vel_prev=np.zeros(6)
	num_steps=int(len(curve_js)/step)
	idx=[]
	for i in range(len(breakpoints)-1):
		for j in range(breakpoints[i],1+breakpoints[i]+step*int((breakpoints[i+1]-breakpoints[i])/step),step):
			idx.append(j)

			next_step=min(breakpoints[i+1],j+step)

			dq=np.abs(curve_js[next_step]-curve_js[j])
			dqdlam=dq/(lam[next_step]-lam[j])
			t=np.max(np.divide(dq,joint_vel_limit))
			qdot_max=dq/t 		###approximated max qdot
			if j in breakpoints or next_step in breakpoints:
				t_acc=np.max(np.divide(joint_vel_prev,joint_acc_limit))
				if np.linalg.norm(dq-t_acc*qdot_max/2)>0:
					t_rem=np.max(np.divide(dq-t_acc*qdot_max/2,joint_vel_limit))
					qdot_act=dq/(t_acc+t_rem)
					dlam_max.append(qdot_act[0]/dqdlam[0])
					joint_vel_prev=qdot_max
				else:
					qddot_max=qdot_max/t_acc
					t_act=np.sqrt(2*dq/qddot_max)[0]
					dlam_max.append((qdot_max[0]/2)/dqdlam[0])
					joint_vel_prev=qddot_max*t_act
			else:
				dlam_max.append(qdot_max[0]/dqdlam[0])

	return dlam_max,idx


def calc_lamdot_dual(curve_js1,curve_js2,lam,joint_vel_limit1,joint_vel_limit2,step):
	############find maximum lambda dot vs lambda
	###curve_js1,2: curve expressed in joint space in radians for both robots
	###lam: discrete lambda (relative path length), same shape as curve_js, from 0 to 1
	###joint_vel_limit1,2: joint velocity limit in radians for both robots
	###step: step size used for dense curve

	dlam_max=[]

	for i in range(0,len(lam)-step,step):
		dq=np.abs(curve_js[i+step]-curve_js[i])
		dqdlam=dq/(lam[i+step]-lam[i])
		t=np.max(np.divide(dq,joint_vel_limit))

		qdot_max=dq/t 		###approximated max qdot
		dlam_max.append(qdot_max[0]/dqdlam[0])



	return dlam_max