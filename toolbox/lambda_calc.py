import numpy as np
from pandas import *
import sys, traceback
from general_robotics_toolbox import *
import matplotlib.pyplot as plt
from robot_def import *


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