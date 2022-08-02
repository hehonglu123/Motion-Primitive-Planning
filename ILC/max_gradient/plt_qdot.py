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
	robot=abb6640(d=50)
	ms=MotionSend()

	dataset='wood/'
	data_dir="../../data/"+dataset
	curve = read_csv(data_dir+"Curve_in_base_frame.csv",header=None).values
	###read in curve_exe
	df = read_csv('curve1_300_qp_multipeak/curve_exe_v300_z10.csv')
	lam, curve_exe, curve_exe_R,curve_exe_js, speed, timestamp=ms.logged_data_analysis(robot,df,realrobot=True)
	#############################chop extension off##################################
	lam, curve_exe, curve_exe_R,curve_exe_js, speed, timestamp=ms.chop_extension(curve_exe, curve_exe_R,curve_exe_js, speed, timestamp,curve[0,:3],curve[-1,:3])

	qdot_all=np.gradient(curve_exe_js,axis=0)/np.tile([np.gradient(timestamp)],(6,1)).T
	qddot_all=np.gradient(qdot_all,axis=0)/np.tile([np.gradient(timestamp)],(6,1)).T

	for i in range(len(curve_exe_js[0])):
		qddot_violate_idx=np.argwhere(np.abs(qddot_all[:,i])>robot.joint_acc_limit[i])

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