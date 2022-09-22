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

def main():
	dataset='from_NX/'
	data_dir="../../data/"+dataset
	solution_dir=data_dir+'dual_arm/'+'diffevo3/'
	cmd_dir=solution_dir+'30J/'
	
	relative_path,robot1,robot2,base2_R,base2_p,lam_relative_path,lam1,lam2,curve_js1,curve_js2=initialize_data(dataset,data_dir,solution_dir)


	ms = MotionSend(robot1=robot1,robot2=robot2,base2_R=base2_R,base2_p=base2_p)

	breakpoints1,primitives1,p_bp1,q_bp1=ms.extract_data_from_cmd(cmd_dir+'command1.csv')
	breakpoints2,primitives2,p_bp2,q_bp2=ms.extract_data_from_cmd(cmd_dir+'command2.csv')

	# breakpoints1[1:]=breakpoints1[1:]-1
	# breakpoints2[2:]=breakpoints2[2:]-1

	###get lambda at each breakpoint
	lam_bp=lam_relative_path[np.append(breakpoints1[0],breakpoints1[1:]-1)]

	vd_relative=2000

	s1_all,s2_all=calc_individual_speed(vd_relative,lam1,lam2,lam_relative_path,breakpoints1)
	v2_all=[]
	for i in range(len(breakpoints1)):
		v2_all.append(speeddata(s2_all[i],9999999,9999999,999999))
		# v2_all.append(v5000)
	


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


		ms = MotionSend(robot1=robot1,robot2=robot2,base2_R=base2_R,base2_p=base2_p)
		###execution with plant
		logged_data=ms.exec_motions_multimove(breakpoints1,primitives1,primitives2,p_bp1,p_bp2,q_bp1,q_bp2,vmax,v2_all,z50,z50)
		with open('recorded_data/dual_iteration_'+str(i)+'.csv',"w") as f:
			f.write(logged_data)
		###save commands
		ms.write_data_to_cmd('recorded_data/command1.csv',breakpoints1,primitives1, p_bp1,q_bp1)
		ms.write_data_to_cmd('recorded_data/command2.csv',breakpoints2,primitives2, p_bp2,q_bp2)

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
		ax1.axis(ymin=0,ymax=2.*vd_relative)
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

		##########################################move towards error direction######################################
		error_bps_v1,error_bps_w1,error_bps_v2,error_bps_w2=ilc.get_error_direction_dual(relative_path,p_bp1,q_bp1,p_bp2,q_bp2,relative_path_exe,relative_path_exe_R,curve_exe1,curve_exe_R1,curve_exe2,curve_exe_R2)

		error_bps_w1=np.zeros(error_bps_w1.shape)
		error_bps_w2=np.zeros(error_bps_w2.shape)
		# error_bps_v1=np.zeros(error_bps_v1.shape)
		# error_bps_v2=np.zeros(error_bps_v2.shape)
		p_bp1, q_bp1, p_bp2, q_bp2=ilc.update_error_direction_dual(relative_path,p_bp1,q_bp1,p_bp2,q_bp2,error_bps_v1,error_bps_w1,error_bps_v2,error_bps_w2)

		###cmd speed adjustment
		# speed_alpha=0.1

		# for m in range(1,len(lam_bp)):
		# 	###get segment average speed
		# 	segment_avg=np.average(speed[np.argmin(np.abs(lam-lam_bp[m-1])):np.argmin(np.abs(lam-lam_bp[m]))])
		# 	###cap above 100m/s for robot2
		# 	s2_all[m]+=speed_alpha*(vd_relative-segment_avg)
		# 	s2_all[m]=max(s2_all[m],100)
		# 	v2_all[m]=speeddata(s2_all[m],9999999,9999999,999999)

		if max(error)<0.2:
			break

if __name__ == "__main__":
	main()