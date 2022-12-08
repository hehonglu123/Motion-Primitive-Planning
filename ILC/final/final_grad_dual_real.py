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
	dataset='curve_1/'
	data_dir="../../data/"+dataset
	solution_dir=data_dir+'dual_arm/'+'diffevo_pose4/'
	cmd_dir=solution_dir+'50J/'

	SAFE_Q1=None
	SAFE_Q2=None
	
	robot1=robot_obj('ABB_6640_180_255','../../config/abb_6640_180_255_robot_default_config.yml',tool_file_path='../../config/paintgun.csv',d=50,acc_dict_path='')
	robot2=robot_obj('ABB_1200_5_90','../../config/abb_1200_5_90_robot_default_config.yml',tool_file_path=solution_dir+'tcp.csv',base_transformation_file=solution_dir+'base.csv',acc_dict_path='')

	relative_path,lam_relative_path,lam1,lam2,curve_js1,curve_js2=initialize_data(dataset,data_dir,solution_dir,robot1,robot2)


	ms = MotionSend(url='http://192.168.55.1:80')

	breakpoints1,primitives1,p_bp1,q_bp1=ms.extract_data_from_cmd(cmd_dir+'command1.csv')
	breakpoints2,primitives2,p_bp2,q_bp2=ms.extract_data_from_cmd(cmd_dir+'command2.csv')

	###get lambda at each breakpoint
	lam_bp=lam_relative_path[np.append(breakpoints1[0],breakpoints1[1:]-1)]

	vd_relative=450

	s1_all,s2_all=calc_individual_speed(vd_relative,lam1,lam2,lam_relative_path,breakpoints1)
	v2_all=[]
	for i in range(len(breakpoints1)):
		v2_all.append(speeddata(s2_all[i],9999999,9999999,999999))
		# v2_all.append(v5000)

	###extension
	p_bp1,q_bp1,p_bp2,q_bp2=ms.extend_dual(robot1,p_bp1,q_bp1,primitives1,robot2,p_bp2,q_bp2,primitives2,breakpoints1)

	###ilc toolbox def
	ilc=ilc_toolbox([robot1,robot2],[primitives1,primitives2])

	multi_peak_threshold=0.4
	max_error_prev=999
	max_grad=False
	iteration=100

	
	for i in range(iteration):
		
		###execution with real robots
		curve_js_all_new, avg_curve_js, timestamp_d=average_N_exe_multimove(ms,breakpoints1,robot1,primitives1,p_bp1,q_bp1,vmax,z50,robot2,primitives2,p_bp2,q_bp2,v2_all,z50,relative_path,SAFE_Q1,SAFE_Q2,log_path='recorded_data',N=10)

		###save commands
		ms.write_data_to_cmd('recorded_data/command1.csv',breakpoints1,primitives1, p_bp1,q_bp1)
		ms.write_data_to_cmd('recorded_data/command2.csv',breakpoints2,primitives2, p_bp2,q_bp2)

		###calculat data with average curve
		lam, curve_exe1,curve_exe2,curve_exe_R1,curve_exe_R2,curve_exe_js1,curve_exe_js2, speed, timestamp, relative_path_exe, relative_path_exe_R =\
			logged_data_analysis_multimove(robot1,robot2,timestamp_d,avg_curve_js)
		#############################chop extension off##################################
		lam, curve_exe1,curve_exe2,curve_exe_R1,curve_exe_R2,curve_exe_js1,curve_exe_js2, speed, timestamp, relative_path_exe, relative_path_exe_R=\
			ms.chop_extension_dual(lam, curve_exe1,curve_exe2,curve_exe_R1,curve_exe_R2,curve_exe_js1,curve_exe_js2, speed, timestamp, relative_path_exe,relative_path_exe_R,relative_path[0,:3],relative_path[-1,:3])

		##############################calcualte error########################################
		error,angle_error=calc_all_error_w_normal(relative_path_exe,relative_path[:,:3],relative_path_exe_R[:,:,-1],relative_path[:,3:])
		print(max(error))

		#############################error peak detection###############################
		peaks,_=find_peaks(error,height=multi_peak_threshold,prominence=0.05,distance=20/(lam[int(len(lam)/2)]-lam[int(len(lam)/2)-1]))		###only push down peaks higher than height, distance between each peak is 20mm, threshold to filter noisy peaks
		if len(peaks)==0 or np.argmax(error) not in peaks:
			peaks=np.append(peaks,np.argmax(error))

		# peaks=np.array([np.argmax(error)])
		##############################plot error#####################################

		fig, ax1 = plt.subplots()
		ax2 = ax1.twinx()
		ax1.plot(lam, speed, 'g-', label='Speed')
		ax2.plot(lam, error, 'b-',label='Error')
		ax2.scatter(lam[peaks],error[peaks],label='peaks')
		ax2.plot(lam, np.degrees(angle_error), 'y-',label='Normal Error')
		ax1.axis(ymin=0,ymax=1.2*vd_relative)
		ax2.axis(ymin=0,ymax=4)

		ax1.set_xlabel('lambda (mm)')
		ax1.set_ylabel('Speed/lamdot (mm/s)', color='g')
		ax2.set_ylabel('Error/Normal Error (mm/deg)', color='b')
		plt.title("Speed and Error Plot")
		h1, l1 = ax1.get_legend_handles_labels()
		h2, l2 = ax2.get_legend_handles_labels()
		ax1.legend(h1+h2, l1+l2, loc=1)

		plt.savefig('recorded_data/iteration_'+str(i))
		plt.clf()
		# plt.show()

		###########################plot for verification###################################
		# p_bp_relative,_=ms.form_relative_path(np.squeeze(q_bp1),np.squeeze(q_bp2),base2_R,base2_p)
		# plt.figure()
		# ax = plt.axes(projection='3d')
		# ax.plot3D(relative_path[:,0], relative_path[:,1], relative_path[:,2], c='gray',label='original')
		# ax.plot3D(relative_path_exe[:,0], relative_path_exe[:,1], relative_path_exe[:,2], c='red',label='execution')
		# ax.scatter3D(p_bp_relative[:,0], p_bp_relative[:,1], p_bp_relative[:,2], c=p_bp_relative[:,2], cmap='Greens',label='breakpoints')
		# ax.scatter(relative_path_exe[peaks,0], relative_path_exe[peaks,1], relative_path_exe[peaks,2],c='orange',label='worst case')
		
		# plt.show()

		if max(error)<max_error_prev and not max_grad:
			print('all bps adjustment')
			##########################################move towards error direction######################################
			error_bps_v1,error_bps_w1,error_bps_v2,error_bps_w2=ilc.get_error_direction_dual(relative_path,p_bp1,q_bp1,p_bp2,q_bp2,relative_path_exe,relative_path_exe_R,curve_exe1,curve_exe_R1,curve_exe2,curve_exe_R2)
			# error_bps_w1=np.zeros(error_bps_w1.shape)
			error_bps_w2=np.zeros(error_bps_w2.shape)
			# error_bps_v1=np.zeros(error_bps_v1.shape)
			# error_bps_v2=np.zeros(error_bps_v2.shape)

			p_bp1_new, q_bp1_new, p_bp2_new, q_bp2_new=ilc.update_error_direction_dual(relative_path,p_bp1,q_bp1,p_bp2,q_bp2,error_bps_v1,error_bps_w1,error_bps_v2,error_bps_w2,gamma_v=0.4)

		else:
			if not max_grad:
				print('switch to max gradient, restore prev iteration')
				max_grad=True
				###restore
				p_bp1=p_bp1_prev
				q_bp1=q_bp1_prev
				p_bp2=p_bp2_prev
				q_bp2=q_bp2_prev
				###execution with real robots
				curve_js_all_new, avg_curve_js, timestamp_d=average_N_exe_multimove(ms,breakpoints1,robot1,primitives1,p_bp1,q_bp1,vmax,z50,robot2,primitives2,p_bp2,q_bp2,v2_all,z50,relative_path,SAFE_Q1,SAFE_Q2,N=10)

				###calculat data with average curve
				lam, curve_exe1,curve_exe2,curve_exe_R1,curve_exe_R2,curve_exe_js1,curve_exe_js2, speed, timestamp, relative_path_exe, relative_path_exe_R =\
					logged_data_analysis_multimove(robot1,robot2,timestamp_d,avg_curve_js)
				#############################chop extension off##################################
				lam, curve_exe1,curve_exe2,curve_exe_R1,curve_exe_R2,curve_exe_js1,curve_exe_js2, speed, timestamp, relative_path_exe, relative_path_exe_R=\
					ms.chop_extension_dual(lam, curve_exe1,curve_exe2,curve_exe_R1,curve_exe_R2,curve_exe_js1,curve_exe_js2, speed, timestamp, relative_path_exe,relative_path_exe_R,relative_path[0,:3],relative_path[-1,:3])

				##############################calcualte error########################################
				error,angle_error=calc_all_error_w_normal(relative_path_exe,relative_path[:,:3],relative_path_exe_R[:,:,-1],relative_path[:,3:])
				print(max(error))

				#############################error peak detection###############################
				peaks,_=find_peaks(error,height=multi_peak_threshold,prominence=0.05,distance=20/(lam[int(len(lam)/2)]-lam[int(len(lam)/2)-1]))		###only push down peaks higher than height, distance between each peak is 20mm, threshold to filter noisy peaks
				if len(peaks)==0 or np.argmax(error) not in peaks:
					peaks=np.append(peaks,np.argmax(error))

			
			##########################################calculate gradient for peaks######################################
			###restore trajectory from primitives
			curve_interp1, curve_R_interp1, curve_js_interp1, breakpoints_blended=form_traj_from_bp(q_bp1,primitives1,robot1)
			curve_interp2, curve_R_interp2, curve_js_interp2, breakpoints_blended=form_traj_from_bp(q_bp2,primitives2,robot2)
			curve_js_blended1,curve_blended1,curve_R_blended1=blend_js_from_primitive(curve_interp1, curve_js_interp1, breakpoints_blended, primitives1,robot1,zone=10)
			curve_js_blended2,curve_blended2,curve_R_blended2=blend_js_from_primitive(curve_interp2, curve_js_interp2, breakpoints_blended, primitives2,robot2,zone=10)

			###establish relative trajectory from blended trajectory
			_,_,_,_,relative_path_blended,relative_path_blended_R=form_relative_path(curve_js_blended1,curve_js_blended2,robot1,robot2)

			###create copy to modify each peak individually
			p_bp1_new=copy.deepcopy(p_bp1)
			q_bp1_new=copy.deepcopy(q_bp1)
			p_bp2_new=copy.deepcopy(p_bp2)
			q_bp2_new=copy.deepcopy(q_bp2)
			for peak in peaks:
				######gradient calculation related to nearest 3 points from primitive blended trajectory, not actual one
				_,peak_error_curve_idx=calc_error(relative_path_exe[peak],relative_path[:,:3])  # index of original curve closest to max error point

				###get closest to worst case point on blended trajectory
				_,peak_error_curve_blended_idx=calc_error(relative_path_exe[peak],relative_path_blended)

				###############get numerical gradient#####
				###find closest 3 breakpoints
				order=np.argsort(np.abs(breakpoints_blended-peak_error_curve_blended_idx))
				breakpoint_interp_2tweak_indices=order[:3]

				de_dp=ilc.get_gradient_from_model_xyz_dual(\
					[p_bp1,p_bp2],[q_bp1,q_bp2],breakpoints_blended,[curve_blended1,curve_blended2],peak_error_curve_blended_idx,[curve_exe_js1[peak],curve_exe_js2[peak]],relative_path[peak_error_curve_idx,:3],breakpoint_interp_2tweak_indices)


				p_bp1_new, q_bp1_new,p_bp2_new,q_bp2_new=ilc.update_bp_xyz_dual([p_bp1_new,p_bp2_new],[q_bp1_new,q_bp2_new],de_dp,error[peak],breakpoint_interp_2tweak_indices)


				#########plot adjusted breakpoints
				_,_,_,_,p_bp_relative_new,_=form_relative_path(np.squeeze(q_bp1_new),np.squeeze(q_bp2_new),robot1,robot2)


		p_bp1_prev=copy.deepcopy(p_bp1)
		q_bp1_prev=copy.deepcopy(q_bp1)
		p_bp2_prev=copy.deepcopy(p_bp2)
		q_bp2_prev=copy.deepcopy(q_bp2)
		###update
		p_bp1=p_bp1_new
		q_bp1=q_bp1_new
		p_bp2=p_bp2_new
		q_bp2=q_bp2_new


		###cmd speed adjustment
		# speed_alpha=0.1

		# for m in range(1,len(lam_bp)):
		# 	###get segment average speed
		# 	segment_avg=np.average(speed[np.argmin(np.abs(lam-lam_bp[m-1])):np.argmin(np.abs(lam-lam_bp[m]))])
		# 	###cap above 100m/s for robot2
		# 	s2_all[m]+=speed_alpha*(vd_relative-segment_avg)
		# 	s2_all[m]=max(s2_all[m],100)
		# 	v2_all[m]=speeddata(s2_all[m],9999999,9999999,999999)

		if max(error)<0.5:
			break

		max_error_prev=max(error)


if __name__ == "__main__":
	main()