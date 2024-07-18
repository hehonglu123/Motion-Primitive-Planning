########
# This module utilized https://github.com/johnwason/abb_motion_program_exec
# and send whatever the motion primitives that algorithms generate
# to RobotStudio
########

import numpy as np
from general_robotics_toolbox import *
from pandas import read_csv
import sys
from scipy.signal import find_peaks

sys.path.append('../../../../../../../toolbox')
sys.path.append('../../../../../../../ilc')
from ilc_toolbox import *
from robot_def import *
from error_check import *
from MotionSend import *
from lambda_calc import *
from blending import *
from dual_arm import *
from realrobot import *

def main():
	dataset='curve_2/'
	data_dir="../../../../../../"+dataset
	solution_dir=data_dir+'dual_arm/'+'diffevo_pose6_3/'
	cmd_dir=solution_dir+'30L/'
	exe_dir=''

	SAFE_Q1=None
	SAFE_Q2=None
	
	robot1=robot_obj('ABB_6640_180_255','../../../../../../../config/abb_6640_180_255_robot_default_config.yml',tool_file_path='../../../../../../../config/paintgun.csv',d=50,acc_dict_path='')
	robot2=robot_obj('ABB_1200_5_90','../../../../../../../config/abb_1200_5_90_robot_default_config.yml',tool_file_path=solution_dir+'tcp.csv',base_transformation_file=solution_dir+'base.csv',acc_dict_path='')

	relative_path,lam_relative_path,lam1,lam2,curve_js1,curve_js2=initialize_data(dataset,data_dir,solution_dir,robot1,robot2)


	ms = MotionSend()

	breakpoints1,primitives1,p_bp1,q_bp1=ms.extract_data_from_cmd(cmd_dir+'command1.csv')
	breakpoints2,primitives2,p_bp2,q_bp2=ms.extract_data_from_cmd(cmd_dir+'command2.csv')

	N=5 		###N-run average
	curve_exe_js_all=[]
	timestamp_all=[]
	total_time_all=[]
	multi_peak_threshold=0.4
	for i in range(N):
		data = np.loadtxt(exe_dir+'run_'+str(i)+'.csv',delimiter=',', skiprows=1)
		##############################data analysis#####################################
		lam, curve_exe1,curve_exe2,curve_exe_R1,curve_exe_R2,curve_exe_js1,curve_exe_js2, speed, timestamp, relative_path_exe, relative_path_exe_R=ms.logged_data_analysis_multimove(MotionProgramResultLog(None, None, data),robot1,robot2,realrobot=True)
		###throw bad curves
		_, _,_,_,_,_,_, _, timestamp_temp, _, _=\
			ms.chop_extension_dual(lam, curve_exe1,curve_exe2,curve_exe_R1,curve_exe_R2,curve_exe_js1,curve_exe_js2, speed, timestamp, relative_path_exe,relative_path_exe_R,relative_path[0,:3],relative_path[-1,:3])

		curve_exe_js_dual=np.hstack((curve_exe_js1,curve_exe_js2))
		total_time_all.append(timestamp_temp[-1]-timestamp_temp[0])

		timestamp=timestamp-timestamp[0]

		curve_exe_js_all.append(curve_exe_js_dual)
		timestamp_all.append(timestamp)

	###trajectory outlier detection, based on chopped time
	curve_exe_js_all,timestamp_all=remove_traj_outlier(curve_exe_js_all,timestamp_all,total_time_all)

	###infer average curve from linear interplateion
	curve_js_all_new, avg_curve_js, timestamp_d=average_curve(curve_exe_js_all,timestamp_all)

	###calculat data with average curve
	lam, curve_exe1,curve_exe2,curve_exe_R1,curve_exe_R2,curve_exe_js1,curve_exe_js2, speed, timestamp, relative_path_exe, relative_path_exe_R =\
		logged_data_analysis_multimove(robot1,robot2,timestamp_d,avg_curve_js)

	#############################chop extension off##################################
	lam, curve_exe1,curve_exe2,curve_exe_R1,curve_exe_R2,curve_exe_js1,curve_exe_js2, speed, timestamp, relative_path_exe, relative_path_exe_R=\
		ms.chop_extension_dual(lam, curve_exe1,curve_exe2,curve_exe_R1,curve_exe_R2,curve_exe_js1,curve_exe_js2, speed, timestamp, relative_path_exe,relative_path_exe_R,relative_path[0,:3],relative_path[-1,:3])


	##############################calcualte error########################################
	error,angle_error=calc_all_error_w_normal(relative_path_exe,relative_path[:,:3],relative_path_exe_R[:,:,-1],relative_path[:,3:])
	#############################error peak detection###############################
	peaks,_=find_peaks(error,height=multi_peak_threshold,prominence=0.05,distance=20/(lam[int(len(lam)/2)]-lam[int(len(lam)/2)-1]))		###only push down peaks higher than height, distance between each peak is 20mm, threshold to filter noisy peaks
	if len(peaks)==0 or np.argmax(error) not in peaks:
		peaks=np.append(peaks,np.argmax(error))


	##############################plot error#####################################

	fig, ax1 = plt.subplots()
	ax2 = ax1.twinx()
	ax1.plot(lam, speed, 'g-', label='Speed')
	ax2.plot(lam, error, 'b-',label='Error')
	ax2.scatter(lam[peaks],error[peaks],label='peaks')
	ax2.plot(lam, np.degrees(angle_error), 'y-',label='Normal Error')
	ax1.axis(ymin=0,ymax=1.2*np.max(speed))
	ax2.axis(ymin=0,ymax=4)
	ax1.set_xlabel('lambda (mm)')
	ax1.set_ylabel('Speed/lamdot (mm/s)', color='g')
	ax2.set_ylabel('Error/Normal Error (mm/deg)', color='b')
	plt.title("Speed and Error Plot")
	h1, l1 = ax1.get_legend_handles_labels()
	h2, l2 = ax2.get_legend_handles_labels()
	ax1.legend(h1+h2, l1+l2, loc='center right')
	# #move legend to middle right
	# fig.tight_layout()
	# fig.legend(loc='center right')


	plt.show()


if __name__ == "__main__":
	main()