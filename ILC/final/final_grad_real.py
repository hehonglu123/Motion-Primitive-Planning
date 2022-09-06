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
	cmd_dir="../../data/"+dataset+solution_dir+'greedy0.02/'



	curve = read_csv(data_dir+"Curve_in_base_frame.csv",header=None).values


	multi_peak_threshold=0.2
	robot=abb6640(d=50)

	v=400
	s = speeddata(v,9999999,9999999,999999)
	zone=20
	z = zonedata(False,zone,1.5*zone,1.5*zone,0.15*zone,1.5*zone,0.15*zone)

	all_bp_threshold=0.8

	
	ms = MotionSend()
	# breakpoints,primitives,p_bp,q_bp=ms.extract_data_from_cmd(cmd_dir+'command.csv')

	# ###extension
	# p_bp,q_bp=ms.extend(robot,q_bp,primitives,breakpoints,p_bp,extension_start=100,extension_end=100)
	breakpoints,primitives,p_bp,q_bp=ms.extract_data_from_cmd('curve1_pose_opt7_v400/command.csv')

	
	###ilc toolbox def
	ilc=ilc_toolbox(robot,primitives)

	###TODO: extension fix start point, moveC support
	max_error_prev=999
	max_grad=False
	inserted_points=[]
	iteration=50
	for i in range(iteration):

		ms = MotionSend(url='http://192.168.55.1:80')
		curve_js_all_new, avg_curve_js, timestamp_d=average_5_exe(ms,robot,primitives,breakpoints,p_bp,q_bp,s,z,curve,"recorded_data")
		###calculat data with average curve
		lam, curve_exe, curve_exe_R, speed=logged_data_analysis(robot,timestamp_d,avg_curve_js)
		#############################chop extension off##################################
		lam, curve_exe, curve_exe_R,curve_exe_js, speed, timestamp=ms.chop_extension(curve_exe, curve_exe_R,avg_curve_js, speed, timestamp_d,curve[0,:3],curve[-1,:3])

		ms.write_data_to_cmd('recorded_data/command.csv',breakpoints,primitives, p_bp,q_bp)

		##############################calcualte error########################################
		error,angle_error=calc_all_error_w_normal(curve_exe,curve[:,:3],curve_exe_R[:,:,-1],curve[:,3:])
		print(max(error))

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
		ax2.axis(ymin=0,ymax=5)
		ax1.axis(ymin=0,ymax=1.2*v)

		ax1.set_xlabel('lambda (mm)')
		ax1.set_ylabel('Speed/lamdot (mm/s)', color='g')
		ax2.set_ylabel('Error/Normal Error (mm/deg)', color='b')
		plt.title("Speed and Error Plot")
		ax1.legend(loc="upper right")

		ax2.legend(loc="upper left")

		plt.legend()
		plt.savefig('recorded_data/iteration_'+str(i))
		plt.clf()
		# plt.show()

		

		###########################plot for verification###################################
		# plt.figure()
		# ax = plt.axes(projection='3d')
		# ax.plot3D(curve[:,0], curve[:,1], curve[:,2], c='gray',label='original')
		# ax.plot3D(curve_exe[:,0], curve_exe[:,1], curve_exe[:,2], c='red',label='execution')
		# p_bp_np=[]
		# for m in range(len(p_bp)):
		# 	for bp_sub_idx in range(len(p_bp[m])):
		# 		p_bp_np.append(p_bp[m][bp_sub_idx])
		# p_bp_np=np.array(p_bp_np)      ###np version
		# ax.scatter(p_bp_np[:,0], p_bp_np[:,1], p_bp_np[:,2], c='green',label='breakpoints')
		# ax.scatter(curve_exe[peaks,0], curve_exe[peaks,1], curve_exe[peaks,2],c='orange',label='worst case')
		
		
		
		if max(error)<max_error_prev and not max_grad:
			print('all bps adjustment')
			##########################################adjust bp's toward error direction######################################
			error_bps_v,error_bps_w=ilc.get_error_direction(curve,p_bp,q_bp,curve_exe,curve_exe_R)
			p_bp, q_bp=ilc.update_error_direction(curve,p_bp,q_bp,error_bps_v,error_bps_w)
		else:
			max_grad=True
			print('max gradient')
			##########################################Multipeak Max Gradient######################################
			###restore trajectory from primitives
			curve_interp, curve_R_interp, curve_js_interp, breakpoints_blended=form_traj_from_bp(q_bp,primitives,robot)

			curve_js_blended,curve_blended,curve_blended_R=blend_js_from_primitive(curve_interp, curve_js_interp, breakpoints_blended, primitives,robot,zone=10)

			for peak in peaks:
				######gradient calculation related to nearest 3 points from primitive blended trajectory, not actual one
				_,peak_error_curve_idx=calc_error(curve_exe[peak],curve[:,:3])  # index of original curve closest to max error point

				###get closest to worst case point on blended trajectory
				_,peak_error_curve_blended_idx=calc_error(curve_exe[peak],curve_blended)

				###############get numerical gradient#####
				###find closest 3 breakpoints
				order=np.argsort(np.abs(breakpoints_blended-peak_error_curve_blended_idx))
				breakpoint_interp_2tweak_indices=order[:2]

				peak_pose=robot.fwd(curve_exe_js[peak])
				##################################################################XYZ Gradient######################################################################
				de_dp=ilc.get_gradient_from_model_xyz(p_bp,q_bp,breakpoints_blended,curve_blended,peak_error_curve_blended_idx,peak_pose,curve[peak_error_curve_idx,:3],breakpoint_interp_2tweak_indices)
				p_bp, q_bp=ilc.update_bp_xyz(p_bp,q_bp,de_dp,error[peak],breakpoint_interp_2tweak_indices)


				##################################################################Ori Gradient######################################################################
				de_ori_dp=ilc.get_gradient_from_model_ori(p_bp,q_bp,breakpoints_blended,curve_blended_R,peak_error_curve_blended_idx,peak_pose,curve[peak_error_curve_idx,3:],breakpoint_interp_2tweak_indices)
				q_bp=ilc.update_bp_ori(p_bp,q_bp,de_ori_dp,angle_error[peak],breakpoint_interp_2tweak_indices)

			# for m in breakpoint_interp_2tweak_indices:
			# 	for bp_sub_idx in range(len(p_bp[m]))
			# 		ax.scatter(p_bp[m][bp_sub_idx][0], p_bp[m][bp_sub_idx][1], p_bp[m][bp_sub_idx][2],c='blue')
			# plt.legend()
			# plt.show()

		max_error_prev=max(error)

if __name__ == "__main__":
	main()