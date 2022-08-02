########
# This module utilized https://github.com/johnwason/abb_motion_program_exec
# and send whatever the motion primitives that algorithms generate
# to RobotStudio
########

import numpy as np
from general_robotics_toolbox import *
from pandas import read_csv
import sys
from io import StringIO
from scipy.signal import find_peaks
# sys.path.append('../abb_motion_program_exec')
from abb_motion_program_exec_client import *
# sys.path.append('../../toolbox')


from robots_def import *
from error_check import *
from MotionSend import *
from lambda_calc import *
from blending import *

def main():
	robot1=abb6640(d=50)
	robot2=abb1200()
	ms=MotionSend()

	dataset='wood/'
	solution_dir='qp1/'
	data_dir="../../data/"+dataset
	curve = read_csv(data_dir+"Curve_in_base_frame.csv",header=None).values
	relative_path = read_csv(data_dir+"/Curve_dense.csv", header=None).values
	lam_relative_path=calc_lam_cs(relative_path)

	with open(data_dir+'dual_arm/'+solution_dir+'abb1200.yaml') as file:
		H_1200 = np.array(yaml.safe_load(file)['H'],dtype=np.float64)

	base2_R=H_1200[:3,:3]
	base2_p=1000*H_1200[:-1,-1]

	###read in curve_exe
	df = read_csv('curve1_500_dual_qp1_50/dual_iteration_29.csv')
	lam, curve_exe1,curve_exe2,curve_exe_R1,curve_exe_R2,curve_exe_js1,curve_exe_js2, speed, timestamp, relative_path_exe, relative_path_exe_R = ms.logged_data_analysis_multimove(df,base2_R,base2_p,realrobot=True)
	#############################chop extension off##################################
	lam, curve_exe1,curve_exe2,curve_exe_R1,curve_exe_R2,curve_exe_js1,curve_exe_js2, speed, timestamp, relative_path_exe, relative_path_exe_R=\
			ms.chop_extension_dual(lam, curve_exe1,curve_exe2,curve_exe_R1,curve_exe_R2,curve_exe_js1,curve_exe_js2, speed, timestamp, relative_path_exe,relative_path_exe_R,relative_path[0,:3],relative_path[-1,:3])


	qdot1_all=np.gradient(curve_exe_js1,axis=0)/np.tile([np.gradient(timestamp)],(6,1)).T
	qddot1_all=np.gradient(qdot1_all,axis=0)/np.tile([np.gradient(timestamp)],(6,1)).T

	for i in range(len(curve_exe_js1[0])):
		qddot_violate_idx=np.argwhere(np.abs(qddot1_all[:,i])>robot1.joint_acc_limit[i])

		plt.scatter(lam[qddot_violate_idx],qdot1_all[qddot_violate_idx,i],label='acc limit')
		plt.plot(lam,qdot1_all[:,i],label='joint '+str(i+1))
		plt.ylim([-robot1.joint_vel_limit[i]-0.1, robot1.joint_vel_limit[i]+0.1])
		plt.xlabel('lambda (mm)')
		plt.ylabel('qdot (rad/s)')
		plt.title('joint '+str(i+1))
		plt.legend()
		plt.show()

	qdot2_all=np.gradient(curve_exe_js2,axis=0)/np.tile([np.gradient(timestamp)],(6,1)).T
	qddot2_all=np.gradient(qdot2_all,axis=0)/np.tile([np.gradient(timestamp)],(6,1)).T

	qdot2_all=np.gradient(curve_exe_js2,axis=0)/np.tile([np.gradient(timestamp)],(6,1)).T
	qddot2_all=np.gradient(qdot2_all,axis=0)/np.tile([np.gradient(timestamp)],(6,1)).T

	for i in range(len(curve_exe_js2[0])):
		qddot_violate_idx=np.argwhere(np.abs(qddot2_all[:,i])>robot2.joint_acc_limit[i])

		plt.scatter(lam[qddot_violate_idx],qdot2_all[qddot_violate_idx,i],label='acc limit')
		plt.plot(lam,qdot2_all[:,i],label='joint '+str(i+1))
		plt.ylim([-robot2.joint_vel_limit[i]-0.1, robot2.joint_vel_limit[i]+0.1])
		plt.xlabel('lambda (mm)')
		plt.ylabel('qdot (rad/s)')
		plt.title('joint '+str(i+1))
		plt.legend()
		plt.show()

	qdot2_all=np.gradient(curve_exe_js2,axis=0)/np.tile([np.gradient(timestamp)],(6,1)).T
	qddot2_all=np.gradient(qdot2_all,axis=0)/np.tile([np.gradient(timestamp)],(6,1)).T


if __name__ == "__main__":
	main()