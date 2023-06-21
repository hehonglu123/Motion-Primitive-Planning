import numpy as np
from general_robotics_toolbox import *
from pandas import read_csv
import sys
from scipy.signal import find_peaks

sys.path.append('../../toolbox')
from robots_def import *
from error_check import *
from MotionSend_motoman import *
from MocapPoseListener import *
from lambda_calc import *
from blending import *
from realrobot import *
from PH_interp import *

sys.path.append('../')
from ilc_toolbox import *

def main():
	curve_name='curve_2'
	dataset=curve_name+'/'
	solution_dir='baseline_motoman/'
	data_dir="../../data/"+dataset+solution_dir
	# cmd_dir="../../data/"+dataset+solution_dir+'100L/'

	cmd_dir=curve_name+"_baseline_100L_nPL/"

	curve = read_csv(data_dir+"Curve_in_base_frame.csv",header=None).values

	multi_peak_threshold=0.4
	# robot=robot_obj('MA2010_A0',def_path='../../config/MA2010_A0_robot_default_config.yml',tool_file_path='../../config/weldgun2.csv',\
    # 	pulse2deg_file_path='../../config/MA2010_A0_pulse2deg_real.csv',d=50)
	
	config_dir='../../config/'
	robot=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',tool_file_path=config_dir+'torch.csv',d=50,\
		pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv',\
		base_marker_config_file=config_dir+'MA2010_0620_marker_config.yaml',tool_marker_config_file=config_dir+'weldgun_0620_marker_config.yaml')
	
	ph_param_lin=None
	## load calibrated PH with q config
	# PH_data_dir=config_dir+'ph_param/test0516_R1/'
	# with open(PH_data_dir+'calib_PH_q.pickle','rb') as file:
	# 	PH_q=pickle.load(file)
	# ph_param_lin=PH_Param()
	# ph_param_lin.fit(PH_q,method='linear')
	## using rotated PH
	# robot.robot.P=deepcopy(robot.calib_P)
	# robot.robot.H=deepcopy(robot.calib_H)

	## add tool
	# robot.robot.R_tool = robot.T_tool_toolmarker.R
	# robot.robot.p_tool = robot.T_tool_toolmarker.p
	
	mocap_url = 'rr+tcp://192.168.55.10:59823?service=optitrack_mocap'
	mocap_url = mocap_url
	mocap_cli = RRN.ConnectService(mocap_url)

	mpl_obj = MocapPoseListener(mocap_cli,[robot],collect_base_window=240)

	v=300
	# v=150
	z=None

	gamma_v_max=1
	gamma_v_min=0.2
	
	ms = MotionSend(robot)
	# breakpoints,primitives,p_bp,q_bp=ms.extract_data_from_cmd(cmd_dir+'command.csv')
	breakpoints,primitives,p_bp,q_bp=ms.extract_data_from_cmd(cmd_dir+'command1.csv')
	###extension
	# p_bp,q_bp=ms.extend(robot,q_bp,primitives,breakpoints,p_bp,extension_start=60,extension_end=60)
	p_bp,q_bp=ms.extend(robot,q_bp,primitives,breakpoints,p_bp,extension_start=50,extension_end=50)

	###ilc toolbox def
	ilc=ilc_toolbox(robot,primitives)

	###TODO: extension fix start point, moveC support
	max_error_prev=999
	max_grad=False
	inserted_points=[]
	iteration=50

	max_error_draw=None
	max_angerror_draw=None

	N=5 	###N-run average

	for i in range(iteration):

		curve_js_all_new, avg_curve_js, timestamp_d, curve_exe_mocap_avg, curve_exe_R_mocap_avg, timestamp_d_mocap=\
			average_N_exe_mocap_log(mpl_obj,ms,robot,primitives,breakpoints,p_bp,q_bp,v,z,curve,log_path="recorded_data/iteration_%i" % i,N=N)

		###calculat data with average curve
		lam, curve_exe, curve_exe_R, speed=logged_data_analysis(robot,timestamp_d,avg_curve_js)
		#############################chop extension off (using nominal PH param)##################################
		lam_nom, curve_exe_nom, curve_exe_R_nom,curve_exe_js, speed_nom, timestamp=ms.chop_extension(curve_exe, curve_exe_R,avg_curve_js, speed, timestamp_d,curve[0,:3],curve[-1,:3])

		############# get error using calibrated PH ########################
		lam, curve_exe, curve_exe_R, speed=logged_data_analysis_phparam(robot,timestamp,curve_exe_js,ph_param_lin)

		############# mocap info #################
		lam_mocap=calc_lam_cs(curve_exe_mocap_avg)

		ms.write_data_to_cmd('recorded_data/command%i.csv'%i,breakpoints,primitives, p_bp,q_bp)

		##############################calcualte error########################################
		error_nominal,angle_error_nominal=calc_all_error_w_normal(curve_exe_nom,curve[:,:3],curve_exe_R_nom[:,:,-1],curve[:,3:]) ## error using nominal PH
		error,angle_error=calc_all_error_w_normal(curve_exe,curve[:,:3],curve_exe_R[:,:,-1],curve[:,3:]) ## error using calibrated PH
		error_mocap,angle_error_mocap=calc_all_error_w_normal(curve_exe_mocap_avg,curve[:,:3],curve_exe_R_mocap_avg[:,:,-1],curve[:,3:]) ## error using mocap
		print(max(error),np.std(speed)/np.average(speed))

		#############################error peak detection###############################
		peaks,_=find_peaks(error,height=multi_peak_threshold,prominence=0.2,distance=20/(lam[int(len(lam)/2)]-lam[int(len(lam)/2)-1]))		###only push down peaks higher than height, distance between each peak is 20mm, threshold to filter noisy peaks
		if len(peaks)==0 or np.argmax(error) not in peaks:
			peaks=np.append(peaks,np.argmax(error))
		peaks=np.array(peaks).astype(int)
		##############################plot error#####################################
		lam=np.array(lam)
		error=np.array(error)

		fig, ax1 = plt.subplots()
		ax2 = ax1.twinx()
		ax1.plot(lam, speed, 'g-', label='Speed (Calib PH)')
		ax2.plot(lam_nom, error_nominal, 'm-',label='Error (Nominal PH)')
		ax2.plot(lam, error, 'b-',label='Error (Calib PH)')
		ax2.plot(lam_mocap, error_mocap, 'c-',label='Error (Mocap)')
		ax2.scatter(lam[peaks],error[peaks],label='Peaks (Calib PH)')
		
		if max_error_draw is None:
			max_error_draw=1.05*max(error)
		ax2.axis(ymin=0,ymax=max_error_draw)
		
		ax1.axis(ymin=0,ymax=1.2*v)

		ax1.set_xlabel('lambda (mm)')
		ax1.set_ylabel('Speed/lamdot (mm/s)', color='g')
		ax2.set_ylabel('Error/Normal Error (mm/deg)', color='b')
		plt.title("Speed and Error Plot v=%f"%v)
		h1, l1 = ax1.get_legend_handles_labels()
		h2, l2 = ax2.get_legend_handles_labels()
		ax1.legend(h1+h2, l1+l2, loc=3)

		plt.savefig('recorded_data/iteration_'+str(i)+'_error')
		plt.clf()

		fig, ax1 = plt.subplots()
		ax2 = ax1.twinx()
		ax1.plot(lam, speed, 'g-', label='Speed (Calib PH)')
		ax2.plot(lam_nom, np.degrees(angle_error_nominal), 'm-',label='Normal Error (Nominal PH)')
		ax2.plot(lam, np.degrees(angle_error), 'b-',label='Normal Error (Calib PH)')
		ax2.plot(lam_mocap, np.degrees(angle_error_mocap), 'c-',label='Normal Error (Mocap)')
		
		if max_angerror_draw is None:
			max_angerror_draw=1.05*max(np.degrees(angle_error))
		ax2.axis(ymin=0,ymax=max_angerror_draw)

		ax1.axis(ymin=0,ymax=1.2*v)

		ax1.set_xlabel('lambda (mm)')
		ax1.set_ylabel('Speed/lamdot (mm/s)', color='g')
		ax2.set_ylabel('Error/Normal Error (mm/deg)', color='b')
		plt.title("Speed and Error Plot v=%f"%v)
		h1, l1 = ax1.get_legend_handles_labels()
		h2, l2 = ax2.get_legend_handles_labels()
		ax1.legend(h1+h2, l1+l2, loc=3)

		plt.savefig('recorded_data/iteration_'+str(i)+'_angerror')
		plt.clf()
		
		###terminate condition
		if max(error)<0.5:
			break

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
		
		
		p_bp_old=copy.deepcopy(p_bp)
		q_bp_old=copy.deepcopy(q_bp)
		if max(error)<1.2*max_error_prev and not max_grad:
			print('ALL BPs ADJUSTMENT')
			##########################################adjust bp's toward error direction######################################
			error_bps_v,error_bps_w=ilc.get_error_direction(curve,p_bp,q_bp,curve_exe,curve_exe_R)
			p_bp, q_bp=ilc.update_error_direction(curve,p_bp,q_bp,error_bps_v,error_bps_w,gamma_v=0.5,gamma_w=0.1)
			# for m in range(len(p_bp)):
			# 	print(np.linalg.norm(q_bp[m][0]-q_bp_old[m][0]),np.linalg.norm(p_bp[m][0]-p_bp_old[m][0]))
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
				p_bp, q_bp=ilc.update_bp_xyz(p_bp,q_bp,de_dp,error[peak],breakpoint_interp_2tweak_indices,alpha=0.25)


				##################################################################Ori Gradient######################################################################
				de_ori_dp=ilc.get_gradient_from_model_ori(p_bp,q_bp,breakpoints_blended,curve_blended_R,peak_error_curve_blended_idx,peak_pose,curve[peak_error_curve_idx,3:],breakpoint_interp_2tweak_indices)
				q_bp=ilc.update_bp_ori(p_bp,q_bp,de_ori_dp,angle_error[peak],breakpoint_interp_2tweak_indices,alpha=0.1)

			# for m in breakpoint_interp_2tweak_indices:
			# 	for bp_sub_idx in range(len(p_bp[m]))
			# 		ax.scatter(p_bp[m][bp_sub_idx][0], p_bp[m][bp_sub_idx][1], p_bp[m][bp_sub_idx][2],c='blue')
			# plt.legend()
			# plt.show()

		max_error_prev=max(error)

		if max(error)<0.5:
			break

if __name__ == "__main__":
	main()