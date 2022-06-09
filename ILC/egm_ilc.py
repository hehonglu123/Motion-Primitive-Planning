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
	# data_dir="fitting_output_new/python_qp_movel/"
	dataset='from_NX/'
	data_dir="../data/"+dataset
	fitting_output="../data/"+dataset+'baseline/100L/'


	curve_js=read_csv(data_dir+'Curve_js.csv',header=None).values
	curve = read_csv(data_dir+"Curve_in_base_frame.csv",header=None).values


	max_error_threshold=0.1
	robot=abb6640(d=50)

	s = speeddata(800,9999999,9999999,999999)
	z = z10


	curve_fit_js=read_csv(fitting_output+'curve_fit_js.csv',header=None).values

	ms = MotionSend()
	breakpoints,primitives,p_bp,q_bp=ms.extract_data_from_cmd(fitting_output+'command.csv')

	###assume bp on original curve
	curve_bp=copy.deepcopy(p_bp)

	###extension
	primitives,p_bp,q_bp=ms.extend(robot,q_bp,primitives,breakpoints,p_bp)

	###ilc toolbox def
	ilc=ilc_toolbox(robot,primitives)

	###TODO: extension fix start point, moveC support
	max_error=999
	inserted_points=[]
	i=0
	iteration=0
	while max_error>max_error_threshold:
		i+=1
		ms = MotionSend()
		###execute,curve_fit_js only used for orientation
		logged_data=ms.exec_motions(robot,primitives,breakpoints,p_bp,q_bp,s,z)

		StringData=StringIO(logged_data)
		df = read_csv(StringData, sep =",")
		##############################data analysis#####################################
		lam, curve_exe, curve_exe_R,curve_exe_js, speed, timestamp=ms.logged_data_analysis(robot,df)
		#############################chop extension off##################################
		start_idx=np.argmin(np.linalg.norm(curve[0,:3]-curve_exe,axis=1))
		end_idx=np.argmin(np.linalg.norm(curve[-1,:3]-curve_exe,axis=1))

		#make sure extension doesn't introduce error
		if np.linalg.norm(curve_exe[start_idx]-curve[0,:3])>0.5:
			start_idx+=1
		if np.linalg.norm(curve_exe[end_idx]-curve[-1,:3])>0.5:
			end_idx-=1

		curve_exe=curve_exe[start_idx:end_idx+1]
		curve_exe_js=curve_exe_js[start_idx:end_idx+1]
		curve_exe_R=curve_exe_R[start_idx:end_idx+1]
		speed=speed[start_idx:end_idx+1]
		speed=replace_outliers(np.array(speed))
		lam=calc_lam_cs(curve_exe)

		##############################calcualte error########################################
		error,angle_error=calc_all_error_w_normal(curve_exe,curve[:,:3],curve_exe_R[:,:,-1],curve[:,3:])
		max_error=max(error)
		print(max_error)
		max_angle_error=max(angle_error)
		max_error_idx=np.argmax(error)#index of exe curve with max error
		_,max_error_curve_idx=calc_error(curve_exe[max_error_idx],curve[:,:3])  # index of original curve closest to max error point
		exe_error_vector=(curve[max_error_curve_idx,:3]-curve_exe[max_error_idx])           # shift vector
		
		#######################################ILC######################################
		###calcualte error at each breakpoints
		ep_flip, eR_flip, exe_bp_p, exe_bp_R=ilc.get_error_bp(p_bp,q_bp,curve_exe,curve_exe_R,curve_bp,None)
		ep_flip.reverse()
		eR_flip.reverse()

		###ILC, add flipped error to current u
		p_bp_temp=copy.deepcopy(p_bp)
		q_bp_temp=copy.deepcopy(q_bp)
		for m in range(1,len(p_bp)-1):
			p_bp_temp[m]=p_bp_temp[m]+ep_flip[m-1]

			q_bp_temp[m][0]=car2js(robot,q_bp_temp[m][0],p_bp_temp[-1],robot.fwd(q_bp[m][0]).R)[0]

		###execute augmented input
		logged_data=ms.exec_motions(robot,primitives,breakpoints,p_bp_temp,q_bp_temp,s,z)

		StringData=StringIO(logged_data)
		df = read_csv(StringData, sep =",")
		###get new tracking error
		_, curve_exe_temp, curve_exe_R_temp,_, _, _=ms.logged_data_analysis(robot,df)
		_, _,exe_bp_p_new, exe_bp_R_new=ilc.get_error_bp(p_bp_temp,q_bp_temp,curve_exe_temp,curve_exe_R_temp,curve_bp,None)



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

			###visualize breakpoints
			# for l in lam:
			# 	ax1.axvline(x=l)

			plt.legend()

			###########################plot for verification###################################
			# plt.figure()
			# ax = plt.axes(projection='3d')
			# ax.plot3D(curve[:,0], curve[:,1], curve[:,2], c='gray',label='original')
			# ax.plot3D(curve_exe[:,0], curve_exe[:,1], curve_exe[:,2], c='red',label='execution')
			# p_bp_np=np.array(p_bp[1:-1])      ###np version, avoid first and last extended points
			# ax.scatter3D(p_bp_np[:,0], p_bp_np[:,1], p_bp_np[:,2], c=p_bp_np[:,2], cmap='Greens',label='breakpoints')
			# ax.scatter(curve_exe[max_error_idx,0], curve_exe[max_error_idx,1], curve_exe[max_error_idx,2],c='orange',label='worst case')


		###update all breakpoints
		p_bp, q_bp=ilc.update_bp_ilc(p_bp,q_bp,exe_bp_p,exe_bp_R,exe_bp_p_new, exe_bp_R_new)


		if i>iteration:
			# p_bp_np=np.array(p_bp[1:-1])      ###np version, avoid first and last extended points
			# ax.scatter3D(p_bp_np[:,0], p_bp_np[:,1], p_bp_np[:,2], c=p_bp_np[:,2], cmap='Blues',label='new breakpoints')
			# plt.legend()
			plt.savefig('iteration '+str(i))
			# plt.show()


if __name__ == "__main__":
	main()