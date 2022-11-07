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

from ilc_toolbox import *
from robots_def import *
from error_check import *
from MotionSend import *
from lambda_calc import *
from blending import *
from dual_arm import *
from realrobot import *

def main():
	# SAFE_Q1=np.radians([33.89,7.77,57.5,-144.03,55.32,0])
	# SAFE_Q2=np.radians([109.45,23.78,54,-38,-66.55,348.77])
	SAFE_Q1=None
	SAFE_Q2=None
	dataset='curve_1/'
	data_dir="../../data/"+dataset
	solution_dir=data_dir+'dual_arm/'+'diffevo_pose4_2/'
	cmd_dir=solution_dir+'50J/'
	
	robot1=robot_obj('ABB_6640_180_255','../../config/abb_6640_180_255_robot_default_config.yml',tool_file_path='../../config/paintgun.csv',d=50,acc_dict_path='')
	robot2=robot_obj('ABB_1200_5_90','../../config/abb_1200_5_90_robot_default_config.yml',tool_file_path=solution_dir+'tcp.csv',base_transformation_file=solution_dir+'base.csv',acc_dict_path='')

	relative_path,lam_relative_path,lam1,lam2,curve_js1,curve_js2=initialize_data(dataset,data_dir,solution_dir,robot1,robot2)


	ms = MotionSend(url='http://192.168.55.1:80')
	# ms = MotionSend()

	breakpoints1,primitives1,p_bp1,q_bp1=ms.extract_data_from_cmd(cmd_dir+'command1.csv')
	breakpoints2,primitives2,p_bp2,q_bp2=ms.extract_data_from_cmd(cmd_dir+'command2.csv')

	###get lambda at each breakpoint
	lam_bp=lam_relative_path[np.append(breakpoints1[0],breakpoints1[1:]-1)]

	vd_relative=500

	s1_all,s2_all=calc_individual_speed(vd_relative,lam1,lam2,lam_relative_path,breakpoints1)
	v2_all=[]
	for i in range(len(breakpoints1)):
		v2_all.append(speeddata(s2_all[i],9999999,9999999,999999))
		# v2_all.append(v5000)
	


	###extension
	p_bp1,q_bp1,p_bp2,q_bp2=ms.extend_dual(robot1,p_bp1,q_bp1,primitives1,robot2,p_bp2,q_bp2,primitives2,breakpoints1)


	###ilc toolbox def
	ilc=ilc_toolbox([robot1,robot2],[primitives1,primitives2])

	multi_peak_threshold=0.2
	###TODO: extension fix start point, moveC support
	max_error=999
	inserted_points=[]
	iteration=100
	for i in range(iteration):

		###execution with real robots
		curve_js_all_new, avg_curve_js, timestamp_d=average_5_exe_multimove(ms,breakpoints1,robot1,primitives1,p_bp1,q_bp1,vmax,z50,robot2,primitives2,p_bp2,q_bp2,v2_all,z50,relative_path,SAFE_Q1,SAFE_Q2)
		###calculat data with average curve
		lam, curve_exe1,curve_exe2,curve_exe_R1,curve_exe_R2,curve_exe_js1,curve_exe_js2, speed, timestamp, relative_path_exe, relative_path_exe_R =\
			logged_data_analysis_multimove(robot1,robot2,timestamp_d,avg_curve_js)
		#############################chop extension off##################################
		lam, curve_exe1,curve_exe2,curve_exe_R1,curve_exe_R2,curve_exe_js1,curve_exe_js2, speed, timestamp, relative_path_exe, relative_path_exe_R=\
			ms.chop_extension_dual(lam, curve_exe1,curve_exe2,curve_exe_R1,curve_exe_R2,curve_exe_js1,curve_exe_js2, speed, timestamp, relative_path_exe,relative_path_exe_R,relative_path[0,:3],relative_path[-1,:3])

		###save commands
		ms.write_data_to_cmd('recorded_data/command1.csv',breakpoints1,primitives1, p_bp1,q_bp1)
		ms.write_data_to_cmd('recorded_data/command2.csv',breakpoints2,primitives2, p_bp2,q_bp2)

		##############################calcualte error########################################
		error,angle_error=calc_all_error_w_normal(relative_path_exe,relative_path[:,:3],relative_path_exe_R[:,:,-1],relative_path[:,3:])
		print(max(error))

		#############################error peak detection###############################
		peaks,_=find_peaks(error,height=multi_peak_threshold,prominence=0.05,distance=20/(lam[int(len(lam)/2)]-lam[int(len(lam)/2)-1]))		###only push down peaks higher than height, distance between each peak is 20mm, threshold to filter noisy peaks
		if len(peaks)==0 or np.argmax(error) not in peaks:
			peaks=np.append(peaks,np.argmax(error))

		# peaks=np.array([np.argmax(error)])
		##############################plot error#####################################

		try:
			plt.close(fig)
		except:
			pass
		fig, ax1 = plt.subplots(figsize=(6,4))
		ax2 = ax1.twinx()
		ax1.plot(lam, speed, 'g-', label='Speed')
		ax2.plot(lam, error, 'b-',label='Error')
		ax2.scatter(lam[peaks],error[peaks],label='peaks')
		ax2.plot(lam, np.degrees(angle_error), 'y-',label='Normal Error')
		ax1.axis(ymin=0,ymax=2.*vd_relative)
		ax2.axis(ymin=0,ymax=4)

		ax1.set_xlabel('lambda (mm)')
		ax1.set_ylabel('Speed/lamdot (mm/s)', color='g')
		ax2.set_ylabel('Error/Normal Error (mm/deg)', color='b')
		plt.title("Speed and Error Plot, Iteration: "+str(i))
		h1, l1 = ax1.get_legend_handles_labels()
		h2, l2 = ax2.get_legend_handles_labels()
		ax1.legend(h1+h2, l1+l2, loc=1)

		fig.canvas.manager.window.move(50,350)
		plt.show(block=False)
		plt.pause(0.1)

		error_bps_v1,error_bps_w1,error_bps_v2,error_bps_w2=ilc.get_error_direction_dual(relative_path,p_bp1,q_bp1,p_bp2,q_bp2,relative_path_exe,relative_path_exe_R,curve_exe1,curve_exe_R1,curve_exe2,curve_exe_R2)

		# error_bps_w1=np.zeros(error_bps_w1.shape)
		# error_bps_w2=np.zeros(error_bps_w2.shape)
		# error_bps_v1=np.zeros(error_bps_v1.shape)
		# error_bps_v2=np.zeros(error_bps_v2.shape)
		p_bp1, q_bp1, p_bp2, q_bp2=ilc.update_error_direction_dual(relative_path,p_bp1,q_bp1,p_bp2,q_bp2,error_bps_v1,error_bps_w1,error_bps_v2,error_bps_w2,gamma_v=0.3,gamma_w=0.05)

		if max(error)<0.5:
			break

if __name__ == "__main__":
	main()