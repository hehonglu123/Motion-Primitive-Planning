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
sys.path.append('../../toolbox/')
from ilc_toolbox import *
from robots_def import *
from error_check import *
from MotionSend import *
from lambda_calc import *
from blending import *

def main():
	dataset='wood/'
	data_dir="../../data/"+dataset
	relative_path = read_csv(data_dir+"/Curve_dense.csv", header=None).values

	lam_relative_path=calc_lam_cs(relative_path)

	with open(data_dir+'dual_arm/abb1200.yaml') as file:
		H_1200 = np.array(yaml.safe_load(file)['H'],dtype=np.float64)

	base2_R=H_1200[:3,:3]
	base2_p=1000*H_1200[:-1,-1]

	with open(data_dir+'dual_arm/tcp.yaml') as file:
		H_tcp = np.array(yaml.safe_load(file)['H'],dtype=np.float64)
	robot1=abb6640(d=50)
	robot2=abb1200(R_tool=H_tcp[:3,:3],p_tool=H_tcp[:-1,-1])

	ms = MotionSend(robot1=robot1,robot2=robot2,base2_R=base2_R,base2_p=base2_p)

	breakpoints1,primitives1,p_bp1,q_bp1=ms.extract_data_from_cmd(data_dir+'/dual_arm/command1.csv')
	breakpoints2,primitives2,p_bp2,q_bp2=ms.extract_data_from_cmd(data_dir+'/dual_arm/command2.csv')

	###get lambda at each breakpoint
	lam_bp=lam_relative_path[np.append(breakpoints1[0],breakpoints1[1:]-1)]

	vd=500

	s1=9999
	s2=[1000]*len(primitives2)
	z=10
	v1 = speeddata(s1,9999999,9999999,999999)
	


	###extension
	p_bp1,q_bp1,p_bp2,q_bp2=ms.extend_dual(ms.robot1,p_bp1,q_bp1,primitives1,ms.robot2,p_bp2,q_bp2,primitives2,breakpoints1)


	###ilc toolbox def
	ilc=ilc_toolbox([robot1,robot2],[primitives1,primitives2],base2_R,base2_p)

	multi_peak_threshold=0.2
	###TODO: extension fix start point, moveC support
	max_error=999
	inserted_points=[]
	iteration=100
	for i in range(iteration):
		###format speed for second arm
		v2=[]
		for m in range(len(s2)):
			v2.append(speeddata(s2[m],9999999,9999999,999999))

		ms = MotionSend(robot1=robot1,robot2=robot2,base2_R=base2_R,base2_p=base2_p)
		###execution with plant
		logged_data=ms.exec_motions_multimove(breakpoints1,primitives1,primitives2,p_bp1,p_bp2,q_bp1,q_bp2,v1,v2,z10,z10)
		with open('recorded_data/dual_iteration_'+str(i)+'.csv',"w") as f:
		    f.write(logged_data)
		StringData=StringIO(logged_data)
		df = read_csv(StringData, sep =",")
		##############################data analysis#####################################
		lam, curve_exe1,curve_exe2,curve_exe_R1,curve_exe_R2,curve_exe_js1,curve_exe_js2, speed, timestamp, relative_path_exe, relative_path_exe_R = ms.logged_data_analysis_multimove(df,base2_R,base2_p,realrobot=True)
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
		ax1.axis(ymin=0,ymax=2.*vd)
		ax2.axis(ymin=0,ymax=4)

		ax1.set_xlabel('lambda (mm)')
		ax1.set_ylabel('Speed/lamdot (mm/s)', color='g')
		ax2.set_ylabel('Error/Normal Error (mm/deg)', color='b')
		plt.title("Speed and Error Plot")
		ax1.legend(loc=0)

		ax2.legend(loc=0)

		plt.legend()
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

		##########################################calculate gradient for peaks######################################
		###restore trajectory from primitives
		curve_interp1, curve_R_interp1, curve_js_interp1, breakpoints_blended=form_traj_from_bp(q_bp1,primitives1,robot1)
		curve_interp2, curve_R_interp2, curve_js_interp2, breakpoints_blended=form_traj_from_bp(q_bp2,primitives2,robot2)
		curve_js_blended1,curve_blended1,curve_R_blended1=blend_js_from_primitive(curve_interp1, curve_js_interp1, breakpoints_blended, primitives1,robot1,zone=10)
		curve_js_blended2,curve_blended2,curve_R_blended2=blend_js_from_primitive(curve_interp2, curve_js_interp2, breakpoints_blended, primitives2,robot2,zone=10)

		###establish relative trajectory from blended trajectory
		relative_path_blended,relative_path_blended_R=ms.form_relative_path(curve_js_blended1,curve_js_blended2,base2_R,base2_p)

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


			p_bp1_new, q_bp1_new,p_bp2_new,q_bp2_new=ilc.update_bp_xyz_dual([p_bp1,p_bp2],[q_bp1,q_bp2],de_dp,error[peak],breakpoint_interp_2tweak_indices)
			


			#########plot adjusted breakpoints
			p_bp_relative_new,_=ms.form_relative_path(np.squeeze(q_bp1_new),np.squeeze(q_bp2_new),base2_R,base2_p)


			# ax.scatter3D(p_bp_relative_new[breakpoint_interp_2tweak_indices,0], p_bp_relative_new[breakpoint_interp_2tweak_indices,1], p_bp_relative_new[breakpoint_interp_2tweak_indices,2], c='blue',label='new breakpoints')
			# plt.legend()
			# plt.show()

			###update
			p_bp1=p_bp1_new
			q_bp1=q_bp1_new
			p_bp2=p_bp2_new
			q_bp2=q_bp2_new

		###cmd speed adjustment
		speed_alpha=0.5

		for m in range(1,len(lam_bp)):
			###get segment average speed
			segment_avg=np.average(speed[np.argmin(np.abs(lam-lam_bp[m-1])):np.argmin(np.abs(lam-lam_bp[m]))])
			###cap above 100m/s for robot2
			s2[m]+=speed_alpha*(vd-segment_avg)
			s2[m]=max(s2[m],100)



if __name__ == "__main__":
	main()