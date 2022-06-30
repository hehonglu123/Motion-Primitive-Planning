########
# This module utilized https://github.com/johnwason/abb_motion_program_exec
# and send whatever the motion primitives that algorithms generate
# to RobotStudio
########

import numpy as np
from general_robotics_toolbox import *
from pandas import read_csv
import sys, os
from io import StringIO
from scipy.signal import find_peaks
import matplotlib.animation as animation

# sys.path.append('../abb_motion_program_exec')
from abb_motion_program_exec_client import *
from ilc_toolbox import *
sys.path.append('../toolbox')
from robots_def import *
from error_check import *
from MotionSend import *
from lambda_calc import *
from blending import *


fig = plt.figure()
#creating a subplot 
ax1 = fig.add_subplot(1,1,1)
start=time.time()
def animate(i):
	global lam_temp_all,error_temp_all,angle_error_temp_all,error, lam
	ax1.clear()
	if len(error)==0:
		for traj_i in range(len(lam_temp_all)):
			ax1.plot(lam_temp_all[traj_i],error_temp_all[traj_i],label='traj_'+str(traj_i+1))
	else:
		ax1.plot(lam,error,label='Averaged Trajectory')

	plt.legend()
	plt.xlabel('lambda (mm)')
	plt.ylabel('Error/Normal Error (mm/deg)')
	plt.title("Speed and Error Plot")	


def main():
	ms = MotionSend(url='http://192.168.55.1:80')

	# data_dir="fitting_output_new/python_qp_movel/"
	dataset='wood/'
	data_dir="../data/"+dataset
	fitting_output="../data/"+dataset+'baseline/100L/'


	curve_js=read_csv(data_dir+'Curve_js.csv',header=None).values
	curve = read_csv(data_dir+"Curve_in_base_frame.csv",header=None).values


	multi_peak_threshold=0.2
	robot=abb6640(d=50)

	v=250
	s = speeddata(v,9999999,9999999,999999)
	z = z10

	global lam_temp_all,error_temp_all,angle_error_temp_all, error, lam
	lam_temp_all=[]
	error_temp_all=[]
	angle_error_temp_all=[]
	error=[]
	lam=[]
	ani = animation.FuncAnimation(fig, animate, interval=1000) 
	plt.show(block=False)	

	###########################################get cmd from original cmd################################
	# breakpoints,primitives,p_bp,q_bp=ms.extract_data_from_cmd(fitting_output+'command.csv')
	# ###extension
	# primitives,p_bp,q_bp=ms.extend(robot,q_bp,primitives,breakpoints,p_bp)
	###########################################get cmd from simulation improved cmd################################
	breakpoints,primitives,p_bp,q_bp=ms.extract_data_from_cmd('max_gradient/curve1_250_100L_multipeak/command.csv')

	###ilc toolbox def
	ilc=ilc_toolbox(robot,primitives)

	###TODO: extension fix start point, moveC support
	max_error=999
	inserted_points=[]

	iteration=50
	for i in range(iteration):

		ms = MotionSend(url='http://192.168.55.1:80')
		#write current command
		ms.write_data_to_cmd('recorded_data/command.csv',breakpoints,primitives, p_bp,q_bp)
		path='recorded_data/iteration_'+str(i)
		if not os.path.isdir(path):
			os.mkdir(path)
		###5 run execute
		curve_exe_all=[]
		curve_exe_js_all=[]
		timestamp_all=[]
		total_time_all=[]

		###clear globals
		lam_temp_all=[]
		error_temp_all=[]
		angle_error_temp_all=[]
		error=[]
		lam=[]

		for r in range(5):
			logged_data=ms.exec_motions(robot,primitives,breakpoints,p_bp,q_bp,s,z)
			###save 5 runs
			# Write log csv to file
			with open(path+'/run_'+str(r)+'.csv',"w") as f:
				f.write(logged_data)

			StringData=StringIO(logged_data)
			df = read_csv(StringData, sep =",")
			##############################data analysis#####################################
			lam, curve_exe, curve_exe_R,curve_exe_js, speed, timestamp=ms.logged_data_analysis(robot,df,realrobot=True)

			###throw bad curves, error calc for individual traj for demo
			lam_temp, curve_exe_temp, curve_exe_R_temp,curve_exe_js_temp, speed_temp, timestamp_temp=ms.chop_extension(curve_exe, curve_exe_R,curve_exe_js, speed, timestamp,curve[0,:3],curve[-1,:3])
			error_temp,angle_error_temp=calc_all_error_w_normal(curve_exe_temp,curve[:,:3],curve_exe_R_temp[:,:,-1],curve[:,3:])
			lam_temp_all.append(lam_temp)
			error_temp_all.append(error_temp_all)
			angle_error_temp_all.append(angle_error_temp_all)

			total_time_all.append(timestamp_temp[-1]-timestamp_temp[0])

			###TODO, avoid corner path failure

			timestamp=timestamp-timestamp[0]

			curve_exe_all.append(curve_exe)
			curve_exe_js_all.append(curve_exe_js)
			timestamp_all.append(timestamp)

		###trajectory outlier detection, based on chopped time
		curve_exe_all,curve_exe_js_all,timestamp_all=remove_traj_outlier(curve_exe_all,curve_exe_js_all,timestamp_all,total_time_all)

		###infer average curve from linear interplateion
		curve_js_all_new, avg_curve_js, timestamp_d=average_curve(curve_exe_js_all,timestamp_all)
		###calculat data with average curve
		lam, curve_exe, curve_exe_R, speed=logged_data_analysis(robot,timestamp_d,avg_curve_js)
		#############################chop extension off##################################
		lam, curve_exe, curve_exe_R,curve_exe_js, speed, timestamp=ms.chop_extension(curve_exe, curve_exe_R,curve_exe_js, speed, timestamp_d,curve[0,:3],curve[-1,:3])


		##############################calcualte error########################################
		error,angle_error=calc_all_error_w_normal(curve_exe,curve[:,:3],curve_exe_R[:,:,-1],curve[:,3:])
		print('avg traj worst error: ',max(error))
		#############################error peak detection###############################
		peaks,_=find_peaks(error,height=multi_peak_threshold,prominence=0.05,distance=20/(lam[int(len(lam)/2)]-lam[int(len(lam)/2)-1]))		###only push down peaks higher than height, distance between each peak is 20mm, threshold to filter noisy peaks
		
		if len(peaks)==0 or np.argmax(error) not in peaks:
			peaks=np.append(peaks,np.argmax(error))
		
		
		##########################################calculate gradient######################################
		######gradient calculation related to nearest 3 points from primitive blended trajectory, not actual one
		###restore trajectory from primitives
		curve_interp, curve_R_interp, curve_js_interp, breakpoints_blended=form_traj_from_bp(q_bp,primitives,robot)

		curve_js_blended,curve_blended,curve_R_blended=blend_js_from_primitive(curve_interp, curve_js_interp, breakpoints_blended, primitives,robot,zone=10)

		for peak in peaks:
			######gradient calculation related to nearest 3 points from primitive blended trajectory, not actual one
			_,peak_error_curve_idx=calc_error(curve_exe[peak],curve[:,:3])  # index of original curve closest to max error point

			###get closest to worst case point on blended trajectory
			_,peak_error_curve_blended_idx=calc_error(curve_exe[peak],curve_blended)
			curve_blended_point=copy.deepcopy(curve_blended[peak_error_curve_blended_idx])

			###############get numerical gradient#####
			###find closest 3 breakpoints
			order=np.argsort(np.abs(breakpoints_blended-peak_error_curve_blended_idx))
			breakpoint_interp_2tweak_indices=order[:3]

			de_dp=ilc.get_gradient_from_model_xyz(p_bp,q_bp,breakpoints_blended,curve_blended,peak_error_curve_blended_idx,robot.fwd(curve_exe_js[peak]),curve[peak_error_curve_idx,:3],breakpoint_interp_2tweak_indices)
			p_bp, q_bp=ilc.update_bp_xyz(p_bp,q_bp,de_dp,error[peak],breakpoint_interp_2tweak_indices)


if __name__ == "__main__":
	main()