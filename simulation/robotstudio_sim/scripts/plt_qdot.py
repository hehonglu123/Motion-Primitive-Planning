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
	robot1=abb6640(d=50,acc_dict_path='../../../toolbox/robot_info/6640acc.pickle')
	robot2=abb1200(acc_dict_path='../../../toolbox/robot_info/1200acc.pickle')

	dataset='wood/'
	data_dir="../../../data/"+dataset
	solution_dir=data_dir+'dual_arm/'+'diffevo_pose1/'
	cmd_dir=solution_dir+'50L/'
	
	


	robot=robot1
	ms=MotionSend()

	###read in curve_exe
	df = read_csv('recorded_data/curve_exe_v1000_z50.csv')

	breakpoints1,primitives1,p_bp1,q_bp1=ms.extract_data_from_cmd(cmd_dir+'command1.csv')
	p_bp_start=copy.deepcopy(p_bp1[0])
	p_bp_end=copy.deepcopy(p_bp1[-1])

	lam, curve_exe, curve_exe_R,curve_exe_js, speed, timestamp=ms.logged_data_analysis(robot,df,realrobot=True)
	lam, curve_exe,curve_exe_R,curve_exe_js, speed, timestamp=ms.chop_extension(curve_exe,curve_exe_R,curve_exe_js, speed, timestamp, p_bp_start,p_bp_end)

	qdot_all=np.gradient(curve_exe_js,axis=0)/np.tile([np.gradient(timestamp)],(6,1)).T
	qddot_all=np.gradient(qdot_all,axis=0)/np.tile([np.gradient(timestamp)],(6,1)).T

	joint_acc_limit=robot.get_acc(curve_exe_js)

	for i in range(len(curve_exe_js[0])):
		qddot_violate_idx=np.argwhere(np.abs(qddot_all[:,i])>joint_acc_limit[:,i])

		plt.scatter(lam[qddot_violate_idx],qdot_all[qddot_violate_idx,i],label='acc limit')
		plt.plot(lam,qdot_all[:,i],label='joint '+str(i+1))
		plt.ylim([-robot.joint_vel_limit[i]-0.1, robot.joint_vel_limit[i]+0.1])
		plt.xlabel('lambda (mm)')
		plt.ylabel('qdot (rad/s)')
		plt.title('joint '+str(i+1))
		plt.legend()
		plt.show()



if __name__ == "__main__":
	main()