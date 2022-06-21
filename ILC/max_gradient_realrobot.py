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

# sys.path.append('../abb_motion_program_exec')
from abb_motion_program_exec_client import *
from ilc_toolbox import *
sys.path.append('../toolbox')
from robots_def import *
from error_check import *
from MotionSend import *
from lambda_calc import *
from blending import *

def main():
	ms = MotionSend(url='http://192.168.55.1:80')

	# data_dir="fitting_output_new/python_qp_movel/"
	dataset='from_NX/'
	data_dir="../data/"+dataset
	fitting_output="../data/"+dataset+'baseline/100L/'


	curve_js=read_csv(data_dir+'Curve_js.csv',header=None).values
	curve = read_csv(data_dir+"Curve_in_base_frame.csv",header=None).values


	max_error_threshold=0.5
	robot=abb6640(d=50)

	s = speeddata(1100,9999999,9999999,999999)
	z = z10


	###########################################get cmd from original cmd################################
	# breakpoints,primitives,p_bp,q_bp=ms.extract_data_from_cmd(fitting_output+'command.csv')
	# ###extension
	# primitives,p_bp,q_bp=ms.extend(robot,q_bp,primitives,breakpoints,p_bp)
	###########################################get cmd from simulation improved cmd################################
	breakpoints,primitives,p_bp,q_bp=ms.extract_data_from_cmd('recorded_data/curve2_1100/command.csv')

	###ilc toolbox def
	ilc=ilc_toolbox(robot,primitives)

	###TODO: extension fix start point, moveC support
	max_error=999
	inserted_points=[]
	i=0
	iteration=0
	while max_error>max_error_threshold:
		i+=1
		ms = MotionSend(url='http://192.168.55.1:80')
		###5 run execute
		curve_exe_all=[]
		curve_exe_js_all=[]
		timestamp_all=[]
		for r in range(5):
			logged_data=ms.exec_motions(robot,primitives,breakpoints,p_bp,q_bp,s,z)

			StringData=StringIO(logged_data)
			df = read_csv(StringData, sep =",")
			##############################data analysis#####################################
			lam, curve_exe, curve_exe_R,curve_exe_js, speed, timestamp=ms.logged_data_analysis(robot,df,realrobot=True)
			
			timestamp=timestamp-timestamp[0]

			curve_exe_all.append(curve_exe)
			curve_exe_js_all.append(curve_exe_js)
			timestamp_all.append(timestamp)


			# ax.plot3D(curve_exe[:,0], curve_exe[:,1], curve_exe[:,2], c=np.random.rand(3,),label=str(i+1)+'th trajectory')

		###infer average curve from linear interplateion
		curve_js_all_new, avg_curve_js, timestamp_d=average_curve(curve_exe_js_all,timestamp_all)
		###calculat error with average curve
		lam, curve_exe, curve_exe_R, speed=logged_data_analysis(robot,timestamp_d,avg_curve_js)
		#############################chop extension off##################################
		lam, curve_exe, curve_exe_R,curve_exe_js, speed, timestamp=ms.chop_extension(curve_exe, curve_exe_R,curve_exe_js, speed, timestamp_d,curve[0,:3],curve[-1,:3])

		error,angle_error=calc_all_error_w_normal(curve_exe,curve[:,:3],curve_exe_R[:,:,-1],curve[:,3:])


		max_error=max(error)
		print(max_error)
		max_angle_error=max(angle_error)
		max_error_idx=np.argmax(error)#index of exe curve with max error
		_,max_error_curve_idx=calc_error(curve_exe[max_error_idx],curve[:,:3])  # index of original curve closest to max error point
		exe_error_vector=(curve[max_error_curve_idx,:3]-curve_exe[max_error_idx])           # shift vector
		
		##############################plot error#####################################
		if i>iteration:
			fig, ax1 = plt.subplots()
			ax2 = ax1.twinx()
			ax1.plot(lam, speed, 'g-', label='Speed')
			ax2.plot(lam, error, 'b-',label='Error')
			ax2.plot(lam, np.degrees(angle_error), 'y-',label='Normal Error')

			ax1.set_xlabel('lambda (mm)')
			ax1.set_ylabel('Speed/lamdot (mm/s)', color='g')
			ax2.set_ylabel('Error/Normal Error (mm/deg)', color='b')
			plt.title("Speed and Error Plot")
			ax1.legend(loc=0)

			ax2.legend(loc=0)

			plt.legend()
			###########################find closest bp####################################
			bp_idx=np.absolute(breakpoints-max_error_curve_idx).argmin()
			###########################plot for verification###################################
			plt.figure()
			ax = plt.axes(projection='3d')
			ax.plot3D(curve[:,0], curve[:,1], curve[:,2], c='gray',label='original')
			ax.plot3D(curve_exe[:,0], curve_exe[:,1], curve_exe[:,2], c='red',label='execution')
			p_bp_np=np.array(p_bp[1:-1])      ###np version, avoid first and last extended points
			ax.scatter3D(p_bp_np[:,0,0], p_bp_np[:,0,1], p_bp_np[:,0,2], c=p_bp_np[:,0,2], cmap='Greens',label='breakpoints')
			ax.scatter(curve_exe[max_error_idx,0], curve_exe[max_error_idx,1], curve_exe[max_error_idx,2],c='orange',label='worst case')
		
		
		##########################################calculate gradient######################################
		######gradient calculation related to nearest 3 points from primitive blended trajectory, not actual one
		###restore trajectory from primitives
		curve_interp, curve_R_interp, curve_js_interp, breakpoints_blended=form_traj_from_bp(q_bp,primitives,robot)

		curve_js_blended,curve_blended,curve_R_blended=blend_js_from_primitive(curve_interp, curve_js_interp, breakpoints_blended, primitives,robot,zone=10)

		###get closest to worst case point on blended trajectory
		_,max_error_curve_blended_idx=calc_error(curve_exe[max_error_idx],curve_blended)
		curve_blended_point=copy.deepcopy(curve_blended[max_error_curve_blended_idx])

		###############get numerical gradient#####
		###find closest 3 breakpoints
		order=np.argsort(np.abs(breakpoints_blended-max_error_curve_blended_idx))
		breakpoint_interp_2tweak_indices=order[:3]

		de_dp=ilc.get_gradient_from_model_xyz(q_bp,p_bp,breakpoints_blended,curve_blended,max_error_curve_blended_idx,robot.fwd(curve_exe_js[max_error_idx]),curve[max_error_curve_idx,:3],breakpoint_interp_2tweak_indices)
		p_bp, q_bp=ilc.update_bp_xyz(p_bp,q_bp,de_dp,max_error,breakpoint_interp_2tweak_indices)

		if i>iteration:
			for m in breakpoint_interp_2tweak_indices:
				ax.scatter(p_bp[m][0][0], p_bp[m][0][1], p_bp[m][0][2],c='blue',label='adjusted breakpoints')
			plt.legend()
			plt.show()


if __name__ == "__main__":
	main()