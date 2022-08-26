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
	dataset='from_NX/'
	solution_dir='curve_pose_opt2/'
	data_dir="../../data/"+dataset+solution_dir
	cmd_dir="../../data/"+dataset+solution_dir+'100L/'



	curve = read_csv(data_dir+"Curve_in_base_frame.csv",header=None).values

	robot=abb6640(d=50)

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

		ms = MotionSend()
		###execution with plant
		logged_data=ms.exec_motions(robot,primitives,breakpoints,p_bp,q_bp,s,z)
		# Write log csv to file
		with open("recorded_data/curve_exe_v"+str(v)+"_z10.csv","w") as f:
			f.write(logged_data)

		ms.write_data_to_cmd('recorded_data/command.csv',breakpoints,primitives, p_bp,q_bp)

		StringData=StringIO(logged_data)
		df = read_csv(StringData, sep =",")
		##############################data analysis#####################################
		lam, curve_exe, curve_exe_R,curve_exe_js, speed, timestamp=ms.logged_data_analysis(robot,df,realrobot=True)
		#############################chop extension off##################################
		lam, curve_exe, curve_exe_R,curve_exe_js, speed, timestamp=ms.chop_extension(curve_exe, curve_exe_R,curve_exe_js, speed, timestamp,curve[0,:3],curve[-1,:3])

		##############################calcualte error########################################
		error,angle_error=calc_all_error_w_normal(curve_exe,curve[:,:3],curve_exe_R[:,:,-1],curve[:,3:])
		print(max(error))

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
		bp_exe_indices=np.zeros((len(p_bp),2,3))		###breakpoints on curve_exe
		curve_original_indices=np.zeros((len(p_bp),2))	###breakpoint index on original curve
		error_bps=np.zeros((len(p_bp),2))				###error distance
		error_bps_v=np.zeros((len(p_bp),2,3))			###error vector
		
		grad=np.zeros((len(p_bp),2))
		gamma=0.5
		for bp_idx in range(len(p_bp)):
			for bp_sub_idx in range(len(p_bp[bp_idx])):
				p_bp_temp=copy.deepcopy(p_bp)
				q_bp_temp=np.array(copy.deepcopy(q_bp))

				if bp_idx==0 and bp_sub_idx==0:
					_,blended_idx=calc_error(curve[0,:3],curve_blended)
					curve_original_idx=0
					error_bp,bp_exe_idx=calc_error(curve[0,:3],curve_exe)							###find closest point on curve_exe to bp
				elif bp_idx==len(p_bp)-1 and bp_sub_idx==len(p_bp[bp_idx])-1:
					_,blended_idx=calc_error(curve[-1,:3],curve_blended)
					curve_original_idx=len(curve)-1
					error_bp,bp_exe_idx=calc_error(curve[-1,:3],curve_exe)
				else:
					###on blended trajectory
					_,blended_idx=calc_error(p_bp[bp_idx][bp_sub_idx],curve_blended)
					_,bp_exe_idx=calc_error(p_bp[bp_idx][bp_sub_idx],curve_exe)							###find closest point on curve_exe to bp
					error_bp,curve_original_idx=calc_error(curve_exe[bp_exe_idx],curve[:,:3])	###find closest point on curve_original to curve_exe[bp]
				
				bp_exe_indices[bp_idx][bp_sub_idx]=bp_exe_idx
				curve_original_indices[bp_idx][bp_sub_idx]=curve_original_idx			
				error_bps[bp_idx][bp_sub_idx]=error_bp
				###error direction
				vd=(curve[curve_original_idx,:3]-curve_exe[bp_exe_idx])
				vd_temp=vd*gamma
				error_bps_v[bp_idx][bp_sub_idx]=vd

				###adjust breakpoint in error direction
				p_bp_temp[bp_idx][bp_sub_idx]+=vd_temp
				q_bp_temp[bp_idx][bp_sub_idx]=car2js(robot,q_bp[bp_idx][bp_sub_idx],p_bp_temp[bp_idx][bp_sub_idx],robot.fwd(q_bp[bp_idx][bp_sub_idx]).R)[0]
				
				#restore new trajectory, only for adjusted breakpoint, 1-bp change requires traj interp from 5 bp
				short_version=range(max(bp_idx-2,0),min(bp_idx+3,len(breakpoints_blended)))
				###start & end idx, choose points in the middle of breakpoints to avoid affecting previous/next blending segments, unless at the boundary (star/end of all curve)
				###guard 5 breakpoints for short blending
				if short_version[0]==0:
					short_version=range(0,5)
					start_idx=breakpoints_blended[short_version[0]]
				else:
					start_idx=int((breakpoints_blended[short_version[0]]+breakpoints_blended[short_version[1]])/2)
				if short_version[-1]==len(breakpoints_blended)-1:
					short_version=range(len(breakpoints_blended)-5,len(breakpoints_blended))
					end_idx = breakpoints_blended[short_version[-1]]+1
				else:
					end_idx=int((breakpoints_blended[short_version[-1]]+breakpoints_blended[short_version[-2]])/2)+1


				curve_interp_temp, curve_R_interp_temp, curve_js_interp_temp, breakpoints_blended_temp=form_traj_from_bp(q_bp_temp[short_version],[primitives[i] for i in short_version],robot)

				curve_js_blended_temp,curve_blended_temp,curve_R_blended_temp=blend_js_from_primitive(curve_interp_temp, curve_js_interp_temp, breakpoints_blended_temp, [primitives[i] for i in short_version],robot,zone=10)
				
				curve_blended_new=copy.deepcopy(curve_blended)

				curve_blended_new[start_idx:end_idx]=curve_blended_temp[start_idx-breakpoints_blended[short_version[0]]:len(curve_blended_temp)-(breakpoints_blended[short_version[-1]]+1-end_idx)]

				###calculate relative gradient
				worst_case_point_shift=curve_blended_new[blended_idx]-curve_blended[blended_idx]

				###get new error - prev error
				de=np.linalg.norm(curve[curve_original_idx,:3]-curve_exe[bp_exe_idx]+worst_case_point_shift)-error_bp
			
				grad[bp_idx][bp_sub_idx]=de/gamma


		###update all bp's
		for bp_idx in range(len(p_bp)):
			for bp_sub_idx in range(len(p_bp[bp_idx])):
				gamma=error_bps[bp_idx][bp_sub_idx]/grad[bp_idx][bp_sub_idx]
				###p_bp temp for numerical gradient
				p_bp[bp_idx][bp_sub_idx]+=gamma*error_bps_v[bp_idx]
				q_bp[bp_idx][bp_sub_idx]=car2js(robot,q_bp[bp_idx][bp_sub_idx],p_bp[bp_idx][bp_sub_idx],robot.fwd(q_bp[bp_idx][bp_sub_idx]).R)[0]


		# for m in breakpoint_interp_2tweak_indices:
		# 	ax.scatter(p_bp[m][0][0], p_bp[m][0][1], p_bp[m][0][2],c='blue',label='adjusted breakpoints')
		# plt.legend()
		# plt.show()


if __name__ == "__main__":
	main()