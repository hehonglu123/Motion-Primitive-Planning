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
from realrobot import *

def main():
	dataset='from_NX/'
	solution_dir='curve_pose_opt2/'
	data_dir="../../data/"+dataset+solution_dir
	cmd_dir="../../data/"+dataset+solution_dir+'100L/'



	curve = read_csv(data_dir+"Curve_in_base_frame.csv",header=None).values


	multi_peak_threshold=0.2
	robot=abb6640(d=50)

	alpha_default=1.
	skip=False

	v=1200
	s = speeddata(v,9999999,9999999,999999)
	z = z10


	ms = MotionSend()
	breakpoints,primitives,p_bp,q_bp=ms.extract_data_from_cmd(cmd_dir+'command.csv')

	###extension
	p_bp,q_bp=ms.extend(robot,q_bp,primitives,breakpoints,p_bp,extension_start=100,extension_end=100)
	###ilc toolbox def
	ilc=ilc_toolbox(robot,primitives)

	###TODO: extension fix start point, moveC support
	max_error=999
	inserted_points=[]
	iteration=50
	for i in range(iteration):
		if skip:
			lam=lam_temp
			timestamp=timestamp_temp
			curve_exe=curve_exe_temp
			curve_exe_js=curve_exe_js_temp

			error=error_temp
			angle_error=angle_error_temp
			speed=speed_temp
		else:


			ms = MotionSend()
			curve_js_all_new, avg_curve_js, timestamp_d=average_5_exe(ms,robot,primitives,breakpoints,p_bp,q_bp,s,z,"recorded_data/curve_exe_v")
			###calculat data with average curve
			lam, curve_exe, curve_exe_R, speed=logged_data_analysis(robot,timestamp_d,avg_curve_js)
			#############################chop extension off##################################
			lam, curve_exe, curve_exe_R,curve_exe_js, speed, timestamp=ms.chop_extension(curve_exe, curve_exe_R,curve_exe_js, speed, timestamp_d,curve[0,:3],curve[-1,:3])

			ms.write_data_to_cmd('recorded_data/command.csv',breakpoints,primitives, p_bp,q_bp)


			##############################calcualte error########################################
			error,angle_error=calc_all_error_w_normal(curve_exe,curve[:,:3],curve_exe_R[:,:,-1],curve[:,3:])

		print('avg traj worst error: ',max(error))

		##############################plot error#####################################

		fig, ax1 = plt.subplots()
		ax2 = ax1.twinx()
		ax1.plot(lam, speed, 'g-', label='Speed')
		ax2.plot(lam, error, 'b-',label='Error')
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
		# p_bp_np=np.array([item[0] for item in p_bp])      ###np version, avoid first and last extended points
		# ax.scatter3D(p_bp_np[:,0], p_bp_np[:,1], p_bp_np[:,2], c=p_bp_np[:,2], cmap='Greens',label='breakpoints')
		# ax.scatter(curve_exe[max_error_idx,0], curve_exe[max_error_idx,1], curve_exe[max_error_idx,2],c='orange',label='worst case')
		
		
		##########################################calculate gradient for all errors at bp's######################################
		###restore trajectory from primitives
		curve_interp, curve_R_interp, curve_js_interp, breakpoints_blended=form_traj_from_bp(q_bp,primitives,robot)
		curve_js_blended,curve_blended,curve_R_blended=blend_js_from_primitive(curve_interp, curve_js_interp, breakpoints_blended, primitives,robot,zone=10)

		###find points on curve_exe closest to all p_bp's, and closest point on blended trajectory to all p_bp's
		bp_exe_indices=[]		###breakpoints on curve_exe
		curve_original_indices=[]
		error_bps=[]
		v_d=[]
		p_bp_temp=copy.deepcopy(p_bp)
		q_bp_temp=copy.deepcopy(q_bp)
		bp_blended_indices=[]
		for bp_idx in range(len(p_bp)):
			_,bp_exe_idx=calc_error(p_bp[bp_idx][0],curve_exe)							###find closest point on curve_exe to bp
			bp_exe_indices.append(bp_exe_idx)
			error_bp,curve_original_idx=calc_error(curve_exe[bp_exe_idx],curve[:,:3])	###find closest point on curve_original to curve_exe[bp]
			curve_original_indices.append(curve_original_idx)			
			error_bps.append(error_bp)
			###gradient direction
			vd_temp=(curve[curve_original_idx,:3]-curve_exe[bp_exe_idx])/2
			v_d.append(vd_temp)
			###p_bp temp for numerical gradient
			p_bp_temp[bp_idx][0]+=vd_temp
			q_bp_temp[bp_idx][0]=car2js(robot,q_bp[bp_idx][0],p_bp_temp[bp_idx][0],robot.fwd(q_bp[bp_idx][0]).R)[0]
			###on blended trajectory
			_,blended_idx=calc_error(p_bp[bp_idx][0],curve_blended)
			bp_blended_indices.append(blended_idx)
		
		###restore trajectory from primitives with temp bp's
		curve_interp_temp, curve_R_interp_temp, curve_js_interp_temp, breakpoints_blended_temp=form_traj_from_bp(q_bp_temp,primitives,robot)
		curve_js_blended_temp,curve_blended_temp,curve_R_blended_temp=blend_js_from_primitive(curve_interp_temp, curve_js_interp_temp, breakpoints_blended_temp, primitives,robot,zone=10)
		###find the gradient
		shift_vectors=[]
		de_dps=[]
		for bp_idx in range(len(p_bp)):
			shift_vector=curve_blended_temp[bp_blended_indices[bp_idx]]-curve_blended[bp_blended_indices[bp_idx]]
			new_error=np.linalg.norm(curve[curve_original_indices[bp_idx],:3]-(curve_exe[bp_exe_indices[bp_idx]]+shift_vector))
			de_dps.append(new_error-error_bps[bp_idx])

		###find correct gamma
		de_dps=np.reshape(de_dps,(-1,1))
		gamma=np.linalg.pinv(de_dps)@error_bps


		#########################################adaptive step size######################
		alpha=alpha_default
		skip=False
		for x in range(4):
			p_bp_temp=copy.deepcopy(p_bp)
			q_bp_temp=copy.deepcopy(q_bp)

			###update all bp's
			for bp_idx in range(len(p_bp)):
				###p_bp temp for numerical gradient
				p_bp_temp[bp_idx][0]-=gamma*v_d[bp_idx]
				q_bp_temp[bp_idx][0]=car2js(robot,q_bp[bp_idx][0],p_bp[bp_idx][0],robot.fwd(q_bp[bp_idx][0]).R)[0]

			##############################execution##################################################
			curve_js_all_new_temp, avg_curve_js_temp, timestamp_d_temp=average_5_exe(ms,robot,primitives,breakpoints,p_bp,q_bp,s,z,"recorded_data/curve_exe_v")
			###calculat data with average curve
			lam_temp, curve_exe_temp, curve_exe_R_temp, speed_temp=logged_data_analysis(robot,timestamp_d_temp,avg_curve_js_temp)
			#############################chop extension off##################################
			lam_temp, curve_exe_temp, curve_exe_R_temp,curve_exe_js_temp, speed_temp, timestamp_temp=ms.chop_extension(curve_exe_temp, curve_exe_R_temp,curve_exe_js_temp, speed_temp, timestamp_d_temp,curve[0,:3],curve[-1,:3])
			##############################calcualte error########################################
			error_temp,angle_error_temp=calc_all_error_w_normal(curve_exe_temp,curve[:,:3],curve_exe_R_temp[:,:,-1],curve[:,3:])

			if np.max(error_temp)>np.max(error):
				alpha/=2
			else:
				skip=True
				break

		p_bp=p_bp_temp
		q_bp=q_bp_temp

		print('step size: ',alpha)



		# for m in breakpoint_interp_2tweak_indices:
		# 	ax.scatter(p_bp[m][0][0], p_bp[m][0][1], p_bp[m][0][2],c='blue',label='adjusted breakpoints')
		# plt.legend()
		# plt.show()


if __name__ == "__main__":
	main()