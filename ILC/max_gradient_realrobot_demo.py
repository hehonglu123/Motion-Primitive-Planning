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

	lam_temp_all=[]
	error_temp_all=[]
	angle_error_temp_all=[]
	error=[]
	lam=[]

	###########################################get cmd from original cmd################################
	# breakpoints,primitives,p_bp,q_bp=ms.extract_data_from_cmd(fitting_output+'command.csv')
	# ###extension
	# primitives,p_bp,q_bp=ms.extend(robot,q_bp,primitives,breakpoints,p_bp)
	###########################################get cmd from simulation improved cmd################################
	breakpoints,primitives,p_bp,q_bp=ms.extract_data_from_cmd('max_gradient/curve1_250_100L_multipeak/command.csv')

	###########################3D plot###################################
	fig_3d=plt.figure()
	ax = plt.axes(projection='3d')
	ax.plot3D(curve[:,0], curve[:,1], curve[:,2], c='gray',label='original')
	p_bp_np=np.array([item[0] for item in p_bp])      ###np version, avoid first and last extended points
	ax.scatter3D(p_bp_np[:,0], p_bp_np[:,1], p_bp_np[:,2], c=p_bp_np[:,2], cmap='Greens',label='breakpoints')
	plt.legend()
	plt.title("Spatial Curve and Simulation Tuned Breakpoints")
	plt.show(block=False)
	fig_3d.canvas.manager.window.move(888,0)
	plt.pause(0.1)

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
		speed_temp_all=[]
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
			error_temp_all.append(error_temp)
			speed_temp_all.append(speed_temp)
			angle_error_temp_all.append(angle_error_temp)

			total_time_all.append(timestamp_temp[-1]-timestamp_temp[0])

			###################################real time plotting 5 run ##########################################
			try:
				plt.close(fig_error_speed)
			except:
				pass
			fig_error_speed, ax1 = plt.subplots()
			ax2 = ax1.twinx()
			for traj_i in range(len(lam_temp_all)):
				ax1.plot(lam_temp_all[traj_i], speed_temp_all[traj_i], 'g-',label='Error' if traj_i==0 else None)
				ax2.plot(lam_temp_all[traj_i],error_temp_all[traj_i],'b-',label='Speed' if traj_i==0 else None)
				ax2.plot(lam_temp_all[traj_i], np.degrees(angle_error_temp_all[traj_i]), 'y-',label='Normal Error' if traj_i==0 else None)

			ax2.axis(ymin=0,ymax=2)

			ax1.set_xlabel('lambda (mm)')
			ax1.set_ylabel('Speed/lamdot (mm/s)', color='g')
			ax2.set_ylabel('Error/Normal Error (mm/deg)', color='b')
			plt.title("Iteration "+str(i+1)+": Individual-Run Speed and Error Plot")
			ax1.legend(loc=0)

			ax2.legend(loc=0)

			plt.legend()
			fig_error_speed.canvas.manager.window.move(0,0)
			plt.show(block=False)
			plt.pause(0.1)	

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
		
		##############################plot averaged curve error#####################################
		plt.close(fig_error_speed)
		fig_error_speed, ax1 = plt.subplots()
		ax2 = ax1.twinx()
		ax1.plot(lam, speed, 'g-', label='Speed')
		ax2.plot(lam, error, 'b-',label='Error')
		ax2.scatter(lam[peaks],error[peaks],label='peaks')
		ax2.plot(lam, np.degrees(angle_error), 'y-',label='Normal Error')
		ax2.axis(ymin=0,ymax=2)

		ax1.set_xlabel('lambda (mm)')
		ax1.set_ylabel('Speed/lamdot (mm/s)', color='g')
		ax2.set_ylabel('Error/Normal Error (mm/deg)', color='b')
		plt.title("Iteration "+str(i+1)+": Trajectory-Average Speed and Error Plot")
		ax1.legend(loc=0)
		ax2.legend(loc=0)
		plt.legend()
		fig_error_speed.canvas.manager.window.move(0,0)
		plt.show(block=False)
		plt.pause(0.1)
		
		###########################3D plot###################################
		plt.close(fig_3d)
		fig_3d=plt.figure()
		ax = plt.axes(projection='3d')
		ax.plot3D(curve[:,0], curve[:,1], curve[:,2], c='gray',label='original')
		ax.plot3D(curve_exe[:,0], curve_exe[:,1], curve_exe[:,2], c='red',label='execution')
		p_bp_np=np.array([item[0] for item in p_bp])      ###np version, avoid first and last extended points
		ax.scatter3D(p_bp_np[:,0], p_bp_np[:,1], p_bp_np[:,2], c=p_bp_np[:,2], cmap='Greens',label='breakpoints')
		curve_exe_peaks=curve_exe[peaks]
		ax.scatter(curve_exe_peaks[:,0], curve_exe_peaks[:,1], curve_exe_peaks[:,2],c='orange',label='peaks')
		plt.legend()
		plt.title("Iteration "+str(i+1)+": 3D Plot")
		fig_3d.canvas.manager.window.move(888,0)
		plt.show(block=False)
		plt.pause(0.1)
		##########################################calculate gradient######################################
		######gradient calculation related to nearest 3 points from primitive blended trajectory, not actual one
		###restore trajectory from primitives
		curve_interp, curve_R_interp, curve_js_interp, breakpoints_blended=form_traj_from_bp(q_bp,primitives,robot)

		curve_js_blended,curve_blended,curve_R_blended=blend_js_from_primitive(curve_interp, curve_js_interp, breakpoints_blended, primitives,robot,zone=10)

		adjusted_bp_idx=[]
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
			adjusted_bp_idx.append(breakpoint_interp_2tweak_indices)

			de_dp=ilc.get_gradient_from_model_xyz(p_bp,q_bp,breakpoints_blended,curve_blended,peak_error_curve_blended_idx,robot.fwd(curve_exe_js[peak]),curve[peak_error_curve_idx,:3],breakpoint_interp_2tweak_indices)
			p_bp, q_bp=ilc.update_bp_xyz(p_bp,q_bp,de_dp,error[peak],breakpoint_interp_2tweak_indices)

		adjusted_bp_idx=np.array(adjusted_bp_idx).flatten()
		###########################3D plot###################################
		plt.close(fig_3d)
		fig_3d=plt.figure()
		ax = plt.axes(projection='3d')
		ax.plot3D(curve[:,0], curve[:,1], curve[:,2], c='gray',label='original')
		ax.plot3D(curve_exe[:,0], curve_exe[:,1], curve_exe[:,2], c='red',label='execution')
		ax.scatter3D(p_bp_np[:,0], p_bp_np[:,1], p_bp_np[:,2], c=p_bp_np[:,2], cmap='Greens',label='breakpoints')
		for m in adjusted_bp_idx:
			ax.scatter(p_bp[m][0][0], p_bp[m][0][1], p_bp[m][0][2],c='blue',label='adjusted breakpoints' if m==breakpoint_interp_2tweak_indices[0] else None)
		ax.scatter(curve_exe_peaks[:,0], curve_exe_peaks[:,1], curve_exe_peaks[:,2],c='orange',label='peaks')
		plt.title("Iteration "+str(i+1)+": 3D Plot")
		fig_3d.canvas.manager.window.move(888,0)
		plt.legend()
		plt.show(block=False)
		plt.pause(0.1)

if __name__ == "__main__":
	main()