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
sys.path.append('../')
sys.path.append('../../toolbox')
from ilc_toolbox import *

from robots_def import *
from error_check import *
from MotionSend import *
from lambda_calc import *
from blending import *

def main():
	dataset='wood/'
	solution_dir='curve_pose_opt7/'
	data_dir="../../data/"+dataset+solution_dir
	cmd_dir="../../data/"+dataset+solution_dir+'100L/'



	curve = read_csv(data_dir+"Curve_in_base_frame.csv",header=None).values


	multi_peak_threshold=0.2
	robot=abb6640(d=50)

	v=500
	s = speeddata(v,9999999,9999999,999999)
	z = z10


	ms = MotionSend()
	breakpoints,primitives,p_bp,q_bp=ms.extract_data_from_cmd(cmd_dir+'command.csv')

	###extension
	p_bp,q_bp=ms.extend(robot,q_bp,primitives,breakpoints,p_bp,extension_start=100,extension_end=100)

	ms = MotionSend()
	###execution with plant
	logged_data=ms.exec_motions(robot,primitives,breakpoints,p_bp,q_bp,s,z)

	StringData=StringIO(logged_data)
	df = read_csv(StringData, sep =",")
	##############################data analysis#####################################
	lam, curve_exe, curve_exe_R,curve_exe_js, speed, timestamp=ms.logged_data_analysis(robot,df,realrobot=True)
	#############################chop extension off##################################
	lam, curve_exe, curve_exe_R,curve_exe_js, speed, timestamp=ms.chop_extension(curve_exe, curve_exe_R,curve_exe_js, speed, timestamp,curve[0,:3],curve[-1,:3])

	bp_idx=50
	for bp_idx in range(3,50):
		_,peak=calc_error(p_bp[bp_idx][0],curve_exe) 

		######gradient calculation related to nearest 3 points from primitive blended trajectory, not actual one
		_,peak_error_curve_idx=calc_error(curve_exe[peak],curve[:,:3])  # index of original curve closest to max error point

		error_vector=np.reshape(curve[peak_error_curve_idx,:3]-curve_exe[peak],(3,1))
		# print('error vector normalized: ',error_vector/np.linalg.norm(error_vector))
		grad=np.zeros((3,3))
		delta=0.1
		###get gradient in 3 direction
		for m in range(3):
			p_bp_temp=copy.deepcopy(p_bp)
			q_bp_temp=copy.deepcopy(q_bp)

			p_bp_temp[bp_idx][0][m]+=delta
			q_bp_temp[bp_idx][0]=car2js(robot,q_bp[bp_idx][0],np.array(p_bp_temp[bp_idx][0]),robot.fwd(q_bp[bp_idx][0]).R)[0]

			###exe temp trajectory to get gradient
			###execution with plant
			logged_data=ms.exec_motions(robot,primitives,breakpoints,p_bp_temp,q_bp_temp,s,z)

			StringData=StringIO(logged_data)
			df = read_csv(StringData, sep =",")
			##############################data analysis#####################################
			lam_temp, curve_exe_temp, curve_exe_R_temp,curve_exe_js_temp, speed_temp, timestamp_temp=ms.logged_data_analysis(robot,df,realrobot=True)
			#############################chop extension off##################################
			lam_temp, curve_exe_temp, curve_exe_R_temp,curve_exe_js_temp, speed_temp, timestamp_temp=ms.chop_extension(curve_exe_temp, curve_exe_R_temp,curve_exe_js_temp, speed_temp, timestamp_temp,curve[0,:3],curve[-1,:3])

			_,peak_temp=calc_error(p_bp_temp[bp_idx][0],curve_exe_temp) 
			d=curve_exe_temp[peak_temp]-curve_exe[peak]

			grad[:,m]=d/delta

		# print('gradient matrix: ',grad)
		print(error_vector.T@grad@error_vector)

if __name__ == "__main__":
	main()