from pandas import read_csv
from sklearn.cluster import KMeans
import numpy as np
import scipy, os, time
from utils import *

def average_curve(curve_all,timestamp_all):
	###get desired synced timestamp first
	max_length=[]
	max_time=[]
	for i in range(len(timestamp_all)):
		max_length.append(len(timestamp_all[i]))
		max_time.append(timestamp_all[i][-1])
	max_length=np.max(max_length)
	max_time=np.max(max_time)
	timestamp_d=np.linspace(0,max_time,num=max_length)

	###linear interpolate each curve with synced timestamp
	curve_all_new=[]
	for i in range(len(timestamp_all)):
		curve_all_new.append(interplate_timestamp(curve_all[i],timestamp_all[i],timestamp_d))

	curve_all_new=np.array(curve_all_new)

	return curve_all_new, np.average(curve_all_new,axis=0),timestamp_d

def average_curve_mocap(curve_exe_all,curve_exe_w_all,timestamp_all):
	###get desired synced timestamp first
	max_length=[]
	max_time=[]
	for i in range(len(timestamp_all)):
		max_length.append(len(timestamp_all[i]))
		max_time.append(timestamp_all[i][-1])
	max_length=np.max(max_length)
	max_time=np.max(max_time)
	timestamp_d=np.linspace(0,max_time,num=max_length)

	###linear interpolate each curve with synced timestamp
	curve_exe_all_new=[]
	curve_exe_w_all_new=[]
	for i in range(len(timestamp_all)):
		curve_exe_all_new.append(interplate_timestamp(curve_exe_all[i],timestamp_all[i],timestamp_d))
		curve_exe_w_all_new.append(interplate_timestamp(curve_exe_w_all[i],timestamp_all[i],timestamp_d))

	curve_exe_all_new=np.array(curve_exe_all_new)
	curve_exe_w_all_new=np.array(curve_exe_w_all_new)

	return np.average(curve_exe_all_new,axis=0), np.average(curve_exe_w_all_new,axis=0), timestamp_d


def remove_traj_outlier(curve_exe_js_all,timestamp_all,total_time_all):

	km = KMeans(n_clusters=2)
	index=km.fit_predict(np.array(total_time_all).reshape(-1,1))
	cluster=km.cluster_centers_
	major_index=scipy.stats.mode(index)[0][0]       ###mostly appeared index
	major_indices=np.where(index==major_index)[0]
	time_mode_avg=cluster[major_index]

	# threshold=0.2 ###ms 
	threshold=0.02*time_mode_avg
	if abs(cluster[0][0]-cluster[1][0])>threshold:
		curve_exe_js_all=[curve_exe_js_all[iii] for iii in major_indices]
		timestamp_all=[timestamp_all[iii] for iii in major_indices]
		print('outlier traj detected')

	return curve_exe_js_all,timestamp_all

def remove_traj_outlier_mocap(curve_exe_all,curve_exe_R_all,timestamp_all,total_time_all):

	km = KMeans(n_clusters=2)
	index=km.fit_predict(np.array(total_time_all).reshape(-1,1))
	cluster=km.cluster_centers_
	major_index=scipy.stats.mode(index)[0][0]       ###mostly appeared index
	major_indices=np.where(index==major_index)[0]
	time_mode_avg=cluster[major_index]

	# threshold=0.2 ###ms 
	threshold=0.02*time_mode_avg
	if abs(cluster[0][0]-cluster[1][0])>threshold:
		curve_exe_all=[curve_exe_all[iii] for iii in major_indices]
		curve_exe_R_all=[curve_exe_R_all[iii] for iii in major_indices]
		timestamp_all=[timestamp_all[iii] for iii in major_indices]
		print('outlier traj detected')

	return curve_exe_all,curve_exe_R_all,timestamp_all

def average_N_exe(ms,robot,primitives,breakpoints,p_bp,q_bp,v,z,curve,log_path='',N=5,safe_q=None):

	if not os.path.exists(log_path):
		os.makedirs(log_path)

	###N run execute
	curve_exe_js_all=[]
	timestamp_all=[]
	total_time_all=[]

	for r in range(N):
		# if safe_q is not None:
		# 	primitives_exe=copy.deepcopy(primitives)
		# 	breakpoints_exe=copy.deepcopy(breakpoints)
		# 	p_bp_exe=copy.deepcopy(p_bp)
		# 	q_bp_exe=copy.deepcopy(q_bp)

		# 	# ms.jog_joint(safe_q)
		# 	primitives_exe.insert(0, "moveabsj")
		# 	breakpoints_exe=np.insert(breakpoints_exe, 0, 0.)
		# 	p_bp_exe.insert(0,[robot.fwd(safe_q).p])
		# 	q_bp_exe.insert(0,[safe_q])
		# 	log_results=ms.exec_motions(robot,primitives_exe,breakpoints_exe,p_bp_exe,q_bp_exe,v,z)

		# else:
		# 	log_results=ms.exec_motions(robot,primitives,breakpoints,p_bp,q_bp,v,z)

		log_results=ms.exec_motions(robot,primitives,breakpoints,p_bp,q_bp,v,z)
		if safe_q is not None:
			ms.jog_joint(safe_q)

		###save 5 runs
		if len(log_path)>0:
			# Write log csv to file
			timestamp,curve_exe_js,cmd_num=ms.parse_logged_data(log_results)
			np.savetxt(log_path+'/run_'+str(r)+'.csv',np.hstack((timestamp.reshape((-1,1)),cmd_num.reshape((-1,1)),curve_exe_js)),delimiter=',',comments='')

		##############################data analysis#####################################
		lam, curve_exe, curve_exe_R,curve_exe_js, speed, timestamp=ms.logged_data_analysis(robot,log_results,realrobot=True)

		###throw bad curves
		_, _, _,_, _, timestamp_temp=ms.chop_extension(curve_exe, curve_exe_R,curve_exe_js, speed, timestamp,curve[0,:3],curve[-1,:3])
		total_time_all.append(timestamp_temp[-1]-timestamp_temp[0])

		timestamp=timestamp-timestamp[0]

		curve_exe_js_all.append(curve_exe_js)
		timestamp_all.append(timestamp)
		time.sleep(0.5)
	###trajectory outlier detection, based on chopped time
	curve_exe_js_all,timestamp_all=remove_traj_outlier(curve_exe_js_all,timestamp_all,total_time_all)

	###infer average curve from linear interplateion
	curve_js_all_new, avg_curve_js, timestamp_d=average_curve(curve_exe_js_all,timestamp_all)

	return curve_js_all_new, avg_curve_js, timestamp_d

def average_N_exe_mocap_log(mpl_obj,ms,robot,primitives,breakpoints,p_bp,q_bp,v,z,curve,log_path='',N=5,safe_q=None):

	if not os.path.exists(log_path):
		os.makedirs(log_path)

	###N run execute
	## robot info
	curve_exe_js_all=[]
	timestamp_all=[]
	total_time_all=[]
	## mocap info
	curve_exe_mocap_all=[]
	curve_exe_w_mocap_all=[]
	timestamp_mocap_all=[]
	total_time_mocap_all=[]
	for r in range(N):

		mpl_obj.run_pose_listener()
		log_results=ms.exec_motions(robot,primitives,breakpoints,p_bp,q_bp,v,z)
		mpl_obj.stop_pose_listener()

		if safe_q is not None:
			ms.jog_joint(safe_q)

		###save 5 runs
		if len(log_path)>0:
			# Write log csv to file
			timestamp,curve_exe_js,cmd_num=ms.parse_logged_data(log_results)
			np.savetxt(log_path+'/run_'+str(r)+'.csv',np.hstack((timestamp.reshape((-1,1)),cmd_num.reshape((-1,1)),curve_exe_js)),delimiter=',',comments='')

		##############################data analysis#####################################
		lam, curve_exe, curve_exe_R,curve_exe_js, speed, timestamp=ms.logged_data_analysis(robot,log_results,realrobot=True)

		###throw bad curves
		_, _, _,_, _, timestamp_temp=ms.chop_extension(curve_exe, curve_exe_R,curve_exe_js, speed, timestamp,curve[0,:3],curve[-1,:3])
		total_time_all.append(timestamp_temp[-1]-timestamp_temp[0])

		timestamp=timestamp-timestamp[0]

		curve_exe_js_all.append(curve_exe_js)
		timestamp_all.append(timestamp)

		##############################mocap data########################################
		curve_exe_dict,curve_exe_R_dict,timestamp_dict = mpl_obj.get_robots_traj()
		curve_exe_mocap,curve_exe_w_mocap,timestamp_mocap=ms.logged_data_analysis_mocap(robot,curve_exe_dict,curve_exe_R_dict,timestamp_dict)
		curve_exe_R_mocap=w2R(curve_exe_w_mocap,np.eye(3))

		###save 5 runs
		if len(log_path)>0:
			# Write log csv to file
			np.savetxt(log_path+'/run_'+str(r)+'_mocap.csv',np.hstack((timestamp_mocap.reshape((-1,1)),curve_exe_mocap,curve_exe_w_mocap)),delimiter=',',comments='')

		###throw bad curves
		speed_mocap=get_speed(curve_exe_mocap,timestamp_mocap)
		_, curve_exe_mocap_temp, curve_exe_R_mocap_temp, _, timestamp_mocap_temp=ms.chop_extension_mocap(curve_exe_mocap, curve_exe_R_mocap, speed_mocap, timestamp_mocap,curve[0,:3],curve[-1,:3],p_bp[0][0])
		curve_exe_w_mocap_temp=smooth_w(R2w(curve_exe_R_mocap_temp,np.eye(3)))		###deal with Singularity

		total_time_mocap_all.append(timestamp_mocap_temp[-1]-timestamp_mocap_temp[0])

		timestamp_mocap=timestamp_mocap-timestamp_mocap[0]

		curve_exe_mocap_all.append(curve_exe_mocap_temp)
		curve_exe_w_mocap_all.append(curve_exe_w_mocap_temp)
		timestamp_mocap_all.append(timestamp_mocap_temp)
		
		time.sleep(0.5)
	###trajectory outlier detection, based on chopped time
	curve_exe_js_all,timestamp_all=remove_traj_outlier(curve_exe_js_all,timestamp_all,total_time_all)
	###infer average curve from linear interplateion
	curve_js_all_new, avg_curve_js, timestamp_d=average_curve(curve_exe_js_all,timestamp_all)

	###trajectory outlier detection, based on chopped time
	curve_exe_mocap_all,curve_exe_w_mocap_all,timestamp_mocap_all=remove_traj_outlier_mocap(curve_exe_mocap_all,curve_exe_w_mocap_all,timestamp_mocap_all,total_time_mocap_all)
	###infer average curve from linear interplateion
	curve_exe_mocap_avg, curve_exe_w_mocap_avg, timestamp_d_mocap=average_curve_mocap(curve_exe_mocap_all,curve_exe_w_mocap_all,timestamp_mocap_all)

	return curve_js_all_new, avg_curve_js, timestamp_d, curve_exe_mocap_avg, curve_exe_w_mocap_avg, timestamp_d_mocap

def average_N_exe_mocap(mpl_obj,ms,robot,primitives,breakpoints,p_bp,q_bp,v,z,curve,log_path='',N=5,safe_q=None):

	if not os.path.exists(log_path):
		os.makedirs(log_path)

	###N run execute
	timestamp_all=[]
	total_time_all=[]
	curve_exe_all=[]
	curve_exe_w_all=[]
	for r in range(N):
		mpl_obj.run_pose_listener()
		log_results=ms.exec_motions(robot,primitives,breakpoints,p_bp,q_bp,v,z)
		mpl_obj.stop_pose_listener()
		
		curve_exe_dict,curve_exe_R_dict,timestamp_dict = mpl_obj.get_robots_traj()
		curve_exe,curve_exe_w,timestamp=ms.logged_data_analysis_mocap(robot,curve_exe_dict,curve_exe_R_dict,timestamp_dict)
		curve_exe_R=w2R(curve_exe_w,np.eye(3))
		
		if safe_q is not None:
			ms.jog_joint(safe_q)

		###save 5 runs
		if len(log_path)>0:
			# Write log csv to file
			np.savetxt(log_path+'/run_'+str(r)+'.csv',np.hstack((timestamp.reshape((-1,1)),curve_exe,curve_exe_w)),delimiter=',',comments='')

		###throw bad curves
		speed=get_speed(curve_exe,timestamp)
		_, curve_exe_temp, curve_exe_R_temp, _, timestamp_temp=ms.chop_extension_mocap(curve_exe, curve_exe_R, speed, timestamp,curve[0,:3],curve[-1,:3],p_bp[0][0])
		curve_exe_w_temp=smooth_w(R2w(curve_exe_R_temp,np.eye(3)))		###deal with Singularity

		total_time_all.append(timestamp_temp[-1]-timestamp_temp[0])

		timestamp=timestamp-timestamp[0]

		curve_exe_all.append(curve_exe_temp)
		curve_exe_w_all.append(curve_exe_w_temp)
		timestamp_all.append(timestamp_temp)
		time.sleep(0.1)

	###trajectory outlier detection, based on chopped time
	curve_exe_all,curve_exe_w_all,timestamp_all=remove_traj_outlier_mocap(curve_exe_all,curve_exe_w_all,timestamp_all,total_time_all)
	###infer average curve from linear interplateion
	curve_exe_avg, curve_exe_w_avg, timestamp_d=average_curve_mocap(curve_exe_all,curve_exe_w_all,timestamp_all)

	return curve_exe_avg, w2R(curve_exe_w_avg,np.eye(3)), timestamp_d

def average_N_exe_multimove(ms,breakpoints,robot1,primitives1,p_bp1,q_bp1,v1_all,z1_all,robot2,primitives2,p_bp2,q_bp2,v2_all,z2_all,relative_path,safeq1=None,safeq2=None,log_path='',N=5):
	###N run execute
	curve_exe_js_all=[]
	timestamp_all=[]
	total_time_all=[]

	for r in range(N):
		if safeq1:
			ms.jog_joint_multimove(safeq1,safeq2)

		log_results=ms.exec_motions_multimove(robot1,robot2,primitives1,primitives2,p_bp1,p_bp2,q_bp1,q_bp2,v1_all,v2_all,z1_all,z2_all)
		###save 5 runs
		if len(log_path)>0:
			# Write log csv to file
			np.savetxt(log_path+'/run_'+str(r)+'.csv',log_results.data,delimiter=',',comments='',header='timestamp,cmd_num,J1,J2,J3,J4,J5,J6,J1_2,J2_2,J3_2,J4_2,J5_2,J6_2')

		##############################data analysis#####################################
		lam, curve_exe1,curve_exe2,curve_exe_R1,curve_exe_R2,curve_exe_js1,curve_exe_js2, speed, timestamp, relative_path_exe, relative_path_exe_R=ms.logged_data_analysis_multimove(log_results,robot1,robot2,realrobot=True)

		curve_exe_js_dual=np.hstack((curve_exe_js1,curve_exe_js2))
		###throw bad curves
		_, _,_,_,_,_,_, _, timestamp_temp, _, _=\
			ms.chop_extension_dual(lam, curve_exe1,curve_exe2,curve_exe_R1,curve_exe_R2,curve_exe_js1,curve_exe_js2, speed, timestamp, relative_path_exe,relative_path_exe_R,relative_path[0,:3],relative_path[-1,:3])
		total_time_all.append(timestamp_temp[-1]-timestamp_temp[0])

		timestamp=timestamp-timestamp[0]

		curve_exe_js_all.append(curve_exe_js_dual)
		timestamp_all.append(timestamp)
		time.sleep(0.5)

	###trajectory outlier detection, based on chopped time
	curve_exe_js_all,timestamp_all=remove_traj_outlier(curve_exe_js_all,timestamp_all,total_time_all)

	###infer average curve from linear interplateion
	curve_js_all_new, avg_curve_js, timestamp_d=average_curve(curve_exe_js_all,timestamp_all)

	return curve_js_all_new, avg_curve_js, timestamp_d


def average_5_egm_car_exe(et,curve_cmd,curve_cmd_R):
	###5 run execute egm Cartesian
	curve_exe_js_all=[]
	timestamp_all=[]
	for r in range(5):
		###move to start first
		print('moving to start point')
		et.jog_joint_cartesian(curve_cmd[0],curve_cmd_R[0])
		
		###traverse the curve
		timestamp,curve_exe_js=et.traverse_curve_cartesian(curve_cmd,curve_cmd_R)

		timestamp=timestamp-timestamp[0]
		curve_exe_js_all.append(curve_exe_js)
		timestamp_all.append(timestamp)
		time.sleep(0.5)

	###infer average curve from linear interplateion
	curve_js_all_new, avg_curve_js, timestamp_d=average_curve(curve_exe_js_all,timestamp_all)

	return curve_js_all_new,avg_curve_js, timestamp_d
