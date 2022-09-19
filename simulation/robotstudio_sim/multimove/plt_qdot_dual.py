import numpy as np
from general_robotics_toolbox import *
from pandas import read_csv
import sys
from io import StringIO
from scipy.signal import find_peaks
from abb_motion_program_exec_client import *

from robots_def import *
from error_check import *
from MotionSend import *
from lambda_calc import *
from blending import *
from dual_arm import *

def main():
	dataset='wood/'
	data_dir="../../../data/"+dataset
	solution_dir=data_dir+'dual_arm/'+'diffevo_pose1/'
	cmd_dir=solution_dir+'50L/'
	
	relative_path,robot1,robot2,base2_R,base2_p,lam_relative_path,lam1,lam2,curve_js1,curve_js2=initialize_data(dataset,data_dir,solution_dir,cmd_dir)

	ms = MotionSend(robot1=robot1,robot2=robot2,base2_R=base2_R,base2_p=base2_p)


	robot1=abb6640(d=50,acc_dict_path='../../../toolbox/robot_info/6640acc.pickle')
	robot2=abb1200(acc_dict_path='../../../toolbox/robot_info/1200acc.pickle')


	###read in curve_exe
	df = read_csv('recorded_data/curve_exe_v1000_z100.csv')

	lam_exe, curve_exe1,curve_exe2,curve_exe_R1,curve_exe_R2,curve_exe_js1,curve_exe_js2, speed, timestamp, relative_path_exe,relative_path_exe_R =ms.logged_data_analysis_multimove(df,base2_R,base2_p,realrobot=True)
	lam_exe, curve_exe1,curve_exe2,curve_exe_R1,curve_exe_R2,curve_exe_js1,curve_exe_js2, speed, timestamp, relative_path_exe, relative_path_exe_R=\
		ms.chop_extension_dual(lam_exe, curve_exe1,curve_exe2,curve_exe_R1,curve_exe_R2,curve_exe_js1,curve_exe_js2, speed, timestamp, relative_path_exe,relative_path_exe_R,relative_path[0,:3],relative_path[-1,:3])

	qdot1_all=np.gradient(curve_exe_js1,axis=0)/np.tile([np.gradient(timestamp)],(6,1)).T
	qddot1_all=np.gradient(qdot1_all,axis=0)/np.tile([np.gradient(timestamp)],(6,1)).T

	qdot2_all=np.gradient(curve_exe_js2,axis=0)/np.tile([np.gradient(timestamp)],(6,1)).T
	qddot2_all=np.gradient(qdot2_all,axis=0)/np.tile([np.gradient(timestamp)],(6,1)).T

	print(qddot1_all[:,3])
	joint_acc_limit1=robot1.get_acc(curve_exe_js1)
	joint_acc_limit2=robot2.get_acc(curve_exe_js2)

	for i in range(len(curve_exe_js1[0])):
		qddot1_violate_idx=np.argwhere(np.abs(qddot1_all[:,i])>joint_acc_limit1[:,i])
		qddot2_violate_idx=np.argwhere(np.abs(qddot2_all[:,i])>joint_acc_limit2[:,i])

		plt.scatter(lam_exe[qddot1_violate_idx],qdot1_all[qddot1_violate_idx,i],label='acc1 limit')
		plt.plot(lam_exe,qdot1_all[:,i],label='robot1 joint '+str(i+1))
		plt.scatter(lam_exe[qddot2_violate_idx],qdot2_all[qddot2_violate_idx,i],label='acc2 limit')
		plt.plot(lam_exe,qdot2_all[:,i],label='robot2 joint '+str(i+1))

		plt.ylim([-max(robot1.joint_vel_limit[i],robot2.joint_vel_limit[i])-0.1, max(robot1.joint_vel_limit[i],robot2.joint_vel_limit[i])+0.1])
		plt.xlabel('lambda (mm)')
		plt.ylabel('qdot (rad/s)')
		plt.title('joint '+str(i+1))
		plt.legend()
		plt.show()




if __name__ == "__main__":
	main()