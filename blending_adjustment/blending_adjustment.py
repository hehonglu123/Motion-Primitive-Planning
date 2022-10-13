########
# This module utilized https://github.com/johnwason/abb_motion_program_exec
# and send whatever the motion primitives that algorithms generate
# to RobotStudio
########

import numpy as np
from general_robotics_toolbox import *
from pandas import read_csv
import sys, copy
from io import StringIO
from abb_motion_program_exec_client import *
from scipy.signal import find_peaks

from MotionSend import *
from dual_arm import *
from utils import *


Z_MIN=5
def failure_detection(curve_exe,speed,p_bp):
	###find blending failure breakpoint index
	peaks,_=find_peaks(-speed,height=-100)
	bp_failure_indices=[]
	p_bp_np=np.array([p_bp[i][0] for i in range(len(p_bp))])
	for peak in peaks:
		bp_failure_indices.append(np.argmin(np.linalg.norm(p_bp_np-curve_exe[peak],axis=1)))

	return np.array(bp_failure_indices)
def fix_blending_error(ms,robot,primitives,breakpoints,p_bp,q_bp,v_all,z_all,start,end):
	########################################################################fix blending failure first#################################################################
	iteration=0
	bp_failure_indices=[None]
	while len(bp_failure_indices)>0:
		logged_data= ms.exec_motions(robot,primitives,breakpoints,p_bp,q_bp,v_all,z_all)

		StringData=StringIO(logged_data)
		df = read_csv(StringData, sep =",")
		##############################data analysis#####################################
		lam, curve_exe, curve_exe_R,curve_exe_js, speed, timestamp=ms.logged_data_analysis(robot,df,realrobot=True)
		#############################chop extension off##################################
		lam, curve_exe, curve_exe_R,curve_exe_js, speed, timestamp=ms.chop_extension(curve_exe, curve_exe_R,curve_exe_js, speed, timestamp,start,end)

		bp_failure_indices=failure_detection(curve_exe,speed,p_bp)
		if len(bp_failure_indices):
			print('blending failure detected')

		for bp_failure_index in bp_failure_indices:
			###decrease blending zone for failure breakpoints & surrounding bp's
			for i in range(-1,2):
				zone=max(z_all[bp_failure_index+i].pzone_tcp-10,Z_MIN)
				z_all[bp_failure_index+i]= zonedata(False,zone,1.5*zone,1.5*zone,0.15*zone,1.5*zone,0.15*zone)

		plot_speed_error(lam,speed,[],[],np.max([v.v_tcp for v in v_all]),path='recorded_data/iteration_'+str(iteration))

	return v_all,z_all

def fix_blending_error_multimove(ms,breakpoints,primitives1,p_bp1,q_bp1,v1_all,z1_all,primitives2,p_bp2,q_bp2,v2_all,z2_all,relative_path,vd_relative,start1,end_1,start_2,end_2):
	########################################################################fix blending failure multimove#################################################################
	logged_data=ms.exec_motions_multimove(primitives1,primitives2,p_bp1,p_bp2,q_bp1,q_bp2,v1_all,v2_all,z1_all,z2_all)

	StringData=StringIO(logged_data)
	df = read_csv(StringData, sep =",")
	##############################data analysis#####################################
	lam, curve_exe1,curve_exe2,curve_exe_R1,curve_exe_R2,curve_exe_js1,curve_exe_js2, speed, timestamp, relative_path_exe,relative_path_exe_R = ms.logged_data_analysis_multimove(df,ms.base2_R,ms.base2_p,realrobot=True)
	#############################chop extension off##################################
	lam, curve_exe1,curve_exe2,curve_exe_R1,curve_exe_R2,curve_exe_js1,curve_exe_js2, speed, timestamp, relative_path_exe, relative_path_exe_R=\
		ms.chop_extension_dual(lam, curve_exe1,curve_exe2,curve_exe_R1,curve_exe_R2,curve_exe_js1,curve_exe_js2, speed, timestamp, relative_path_exe,relative_path_exe_R,relative_path[0,:3],relative_path[-1,:3])
	
	speed1=get_speed(curve_exe1,timestamp)
	speed2=get_speed(curve_exe2,timestamp)
	###calculate error
	error,angle_error=calc_all_error_w_normal(relative_path_exe,relative_path[:,:3],relative_path_exe_R[:,:,-1],relative_path[:,3:])

	fig, ax1 = plt.subplots()

	ax2 = ax1.twinx()
	ax1.plot(lam,speed, 'g-', label='Relative Speed')
	ax1.plot(lam,speed1, 'r-', label='TCP1 Speed')
	# ax1.plot(lam_relative_path[2:],s1_cmd,'p-',label='TCP1 cmd Speed')
	ax1.plot(lam,speed2, 'm-', label='TCP2 Speed')
	# ax1.plot(lam_relative_path[2:],s2_cmd,'p-',label='TCP2 cmd Speed')
	ax2.plot(lam, error, 'b-',label='Error')
	ax2.plot(lam, np.degrees(angle_error), 'y-',label='Normal Error')

	ax1.set_xlabel('lambda (mm)')
	ax1.set_ylabel('Speed/lamdot (mm/s)', color='g')
	ax2.set_ylabel('Error (mm)', color='b')
	plt.title('Speed: v'+str(vd_relative))
	h1, l1 = ax1.get_legend_handles_labels()
	h2, l2 = ax2.get_legend_handles_labels()
	ax1.legend(h1+h2, l1+l2, loc=1)

	plt.show()


	bp_failure_indices=failure_detection(curve_exe1,speed1,p_bp1)
	if len(bp_failure_indices):
		print('blending failure detected, fixing individual robots')
		###execute individually to see which robot 
		v1_all,z1_all=fix_blending_error_multimove_single6640(ms,breakpoints,primitives1,p_bp1,q_bp1,v1_all,z1_all,primitives2,p_bp2,q_bp2,v2_all,z2_all,start1,end_1,vd_relative)
		print('6640 blending error fixed')
		v2_all,z2_all=fix_blending_error_multimove_single1200(ms,breakpoints,primitives1,p_bp1,q_bp1,v1_all,z1_all,primitives2,p_bp2,q_bp2,v2_all,z2_all,start_2,end_2,vd_relative)
		print('1200 blending error fixed')
	return v1_all,z1_all,v2_all,z2_all

def fix_blending_error_multimove_single6640(ms,breakpoints,primitives1,p_bp1,q_bp1,v1_all,z1_all,primitives2,p_bp2,q_bp2,v2_all,z2_all,start,end,vd_relative):
	########################################################################fix blending failure first#################################################################
	iteration=0
	bp_failure_indices=[None]

	###set other robot not moving
	q_bp2_temp=copy.deepcopy(q_bp2)
	p_bp2_temp=copy.deepcopy(p_bp2)
	for x in range(1,len(q_bp1)):
		q_bp2_temp[x]=q_bp2[0]
		p_bp2_temp[x]=p_bp2[0]

	while len(bp_failure_indices)>0:
		logged_data=ms.exec_motions_multimove(primitives1,primitives2,p_bp1,p_bp2_temp,q_bp1,q_bp2_temp,v1_all,v2_all,z1_all,z2_all)

		StringData=StringIO(logged_data)
		df = read_csv(StringData, sep =",")
		##############################data analysis#####################################
		lam, curve_exe1,_,curve_exe_R1,_,curve_exe_js1,_, speed, timestamp, _,_ = ms.logged_data_analysis_multimove(df,ms.base2_R,ms.base2_p,realrobot=True)
		#############################chop extension off##################################
		lam, curve_exe1, curve_exe_R,curve_exe_js, speed, timestamp=ms.chop_extension(curve_exe1, curve_exe_R1,curve_exe_js1, speed, timestamp,start,end)

		bp_failure_indices=failure_detection(curve_exe1,speed,p_bp1)
		if len(bp_failure_indices):
			print('blending failure detected')
		
		for bp_failure_index in bp_failure_indices:
			min_zone_count=0

			###decrease blending zone for failure breakpoints & surrounding bp's
			for i in range(-1,2):
				min_zone_count+=z1_all[bp_failure_index+i].pzone_tcp
				zone1=max(z1_all[bp_failure_index+i].pzone_tcp-10,Z_MIN)
				z1_all[bp_failure_index+i]= zonedata(False,zone1,1.5*zone1,1.5*zone1,0.15*zone1,1.5*zone1,0.15*zone1)

			if min_zone_count==3*Z_MIN:
				print('adjusting speed')
				for i in range(-1,1):
					v1_all[bp_failure_index+i]= speeddata(v1_all[bp_failure_index+i].v_tcp,9999999,9999999,999999)

		# plot_speed_error(lam,speed,[],[],np.max([v.v_tcp for v in v1_all]),path='recorded_data/iteration_'+str(iteration))

	return v1_all,z1_all

def fix_blending_error_multimove_single1200(ms,breakpoints,primitives1,p_bp1,q_bp1,v1_all,z1_all,primitives2,p_bp2,q_bp2,v2_all,z2_all,start,end,vd_relative):
	########################################################################fix blending failure first#################################################################
	iteration=0
	bp_failure_indices=[None]

	q_bp1_temp=copy.deepcopy(q_bp1)
	p_bp1_temp=copy.deepcopy(p_bp1)
	###set other robot not moving
	for x in range(1,len(q_bp1)):
		q_bp1_temp[x]=q_bp1[0]
		p_bp1_temp[x]=p_bp1[0]

	while len(bp_failure_indices)>0:
		logged_data=ms.exec_motions_multimove(primitives1,primitives2,p_bp1_temp,p_bp2,q_bp1_temp,q_bp2,v1_all,v2_all,z1_all,z2_all)

		StringData=StringIO(logged_data)
		df = read_csv(StringData, sep =",")
		##############################data analysis#####################################
		lam, curve_exe1,curve_exe2,curve_exe_R1,curve_exe_R2,curve_exe_js1,curve_exe_js2, speed, timestamp, relative_path_exe,relative_path_exe_R = ms.logged_data_analysis_multimove(df,ms.base2_R,ms.base2_p,realrobot=True)
		 
		curve_exe=curve_exe2
		curve_exe_R=curve_exe_R2
		curve_exe_js=curve_exe_js2
		#############################chop extension off##################################

		lam, curve_exe,curve_exe_R,curve_exe_js, speed, timestamp=ms.chop_extension(curve_exe,curve_exe_R,curve_exe_js, speed, timestamp, start,end)
		speed=get_speed(curve_exe,timestamp)

		bp_failure_indices=failure_detection(curve_exe,speed,p_bp2)
		if len(bp_failure_indices):
			print('blending failure detected')
		
		for bp_failure_index in bp_failure_indices:
			min_zone_count=0
			###decrease blending zone for failure breakpoints & surrounding bp's
			for i in range(-1,2):
				min_zone_count+=z2_all[bp_failure_index+i].pzone_tcp
				zone2=max(z2_all[bp_failure_index+i].pzone_tcp-10,Z_MIN)
				z2_all[bp_failure_index+i]= zonedata(False,zone2,1.5*zone2,1.5*zone2,0.15*zone2,1.5*zone2,0.15*zone2)

			if min_zone_count==3*Z_MIN:
				print('adjusting speed')
				for i in range(-1,1):
					v2_all[bp_failure_index+i]= speeddata(v2_all[bp_failure_index+i].v_tcp,9999999,9999999,999999)

		# plot_speed_error(lam,speed,[],[],np.max([v.v_tcp for v in v2_all]),path='recorded_data/iteration_'+str(iteration))
		# for line in ms.client.read_event_log():
		# 	print(line)

	return v2_all,z2_all

def main6640():
	dataset='from_NX/'
	data_dir="../data/"+dataset
	solution_dir=data_dir+'dual_arm/'+'diffevo_pose2/'
	cmd_dir=solution_dir+'50L/'
	
	relative_path,robot1,robot2,base2_R,base2_p,lam_relative_path,lam1,lam2,curve_js1,curve_js2=initialize_data(dataset,data_dir,solution_dir)

	ms = MotionSend()

	breakpoints1,primitives1,p_bp1,q_bp1=ms.extract_data_from_cmd(cmd_dir+'command1.csv')
	breakpoints2,primitives2,p_bp2,q_bp2=ms.extract_data_from_cmd(cmd_dir+'command2.csv')

	start=copy.deepcopy(p_bp1[0])
	end=copy.deepcopy(p_bp1[-1])

	###extension
	p_bp1,q_bp1,p_bp2,q_bp2=ms.extend_dual(ms.robot1,p_bp1,q_bp1,primitives1,ms.robot2,p_bp2,q_bp2,primitives2,breakpoints1)

	###get lambda at each breakpoint
	lam_bp=lam_relative_path[np.append(breakpoints1[0],breakpoints1[1:]-1)]

	vd_relative=2500

	s1_all,s2_all=calc_individual_speed(vd_relative,lam1,lam2,lam_relative_path,breakpoints1)
	v1_all=[]
	for i in range(len(breakpoints1)):
		v1_all.append(speeddata(s1_all[i],9999999,9999999,999999))

	# s1_cmd,s2_cmd=cmd_speed_profile(breakpoints1,s1_all,s2_all)


	z_all=[z50]*len(breakpoints1)

	
	v_all,z_all=fix_blending_error(ms,robot1,primitives1,breakpoints1,p_bp1,q_bp1,v1_all,z_all,start,end)

def main1200():
	dataset='from_NX/'
	data_dir="../data/"+dataset
	solution_dir=data_dir+'dual_arm/'+'diffevo_pose2/'
	cmd_dir=solution_dir+'30L/'
	
	relative_path,robot1,robot2,base2_R,base2_p,lam_relative_path,lam1,lam2,curve_js1,curve_js2=initialize_data(dataset,data_dir,solution_dir)

	ms = MotionSend()

	breakpoints1,primitives1,p_bp1,q_bp1=ms.extract_data_from_cmd(cmd_dir+'command1.csv')
	breakpoints2,primitives2,p_bp2,q_bp2=ms.extract_data_from_cmd(cmd_dir+'command2.csv')

	start=copy.deepcopy(p_bp2[0])
	end=copy.deepcopy(p_bp2[-1])

	###extension
	p_bp1,q_bp1,p_bp2,q_bp2=ms.extend_dual(ms.robot1,p_bp1,q_bp1,primitives1,ms.robot2,p_bp2,q_bp2,primitives2,breakpoints1,extension_start2=100,extension_end2=100)

	###get lambda at each breakpoint
	lam_bp=lam_relative_path[np.append(breakpoints1[0],breakpoints1[1:]-1)]

	vd_relative=2500

	s1_all,s2_all=calc_individual_speed(vd_relative,lam1,lam2,lam_relative_path,breakpoints1)
	v2_all=[]
	for i in range(len(breakpoints2)):
		v2_all.append(speeddata(s2_all[i],9999999,9999999,999999))

	# s1_cmd,s2_cmd=cmd_speed_profile(breakpoints1,s1_all,s2_all)


	z_all=[z50]*len(breakpoints2)

	
	v_all,z_all=fix_blending_error(ms,robot2,primitives2,breakpoints2,p_bp2,q_bp2,v2_all,z_all,start,end)

def main_multimove():
	dataset='from_NX/'
	data_dir="../data/"+dataset
	solution_dir=data_dir+'dual_arm/'+'diffevo_pose2/'
	cmd_dir=solution_dir+'30L/'
	
	relative_path,robot1,robot2,base2_R,base2_p,lam_relative_path,lam1,lam2,curve_js1,curve_js2=initialize_data(dataset,data_dir,solution_dir)

	ms = MotionSend(robot1=robot1,robot2=robot2,base2_R=base2_R,base2_p=base2_p)

	breakpoints1,primitives1,p_bp1,q_bp1=ms.extract_data_from_cmd(cmd_dir+'command1.csv')
	breakpoints2,primitives2,p_bp2,q_bp2=ms.extract_data_from_cmd(cmd_dir+'command2.csv')

	start1=copy.deepcopy(p_bp1[0])
	end_1=copy.deepcopy(p_bp1[-1])
	start_2=copy.deepcopy(p_bp2[0])
	end_2=copy.deepcopy(p_bp2[-1])
	###extension
	p_bp1,q_bp1,p_bp2,q_bp2=ms.extend_dual(ms.robot1,p_bp1,q_bp1,primitives1,ms.robot2,p_bp2,q_bp2,primitives2,breakpoints1)

	###get lambda at each breakpoint
	lam_bp=lam_relative_path[np.append(breakpoints1[0],breakpoints1[1:]-1)]

	vd_relative=2500

	s1_all,s2_all=calc_individual_speed(vd_relative,lam1,lam2,lam_relative_path,breakpoints1)
	v1_all=[]
	v2_all=[]
	for i in range(len(breakpoints1)):
		v1_all.append(speeddata(s1_all[i],9999999,9999999,999999))
		v2_all.append(speeddata(s2_all[i],9999999,9999999,999999))

	# s1_cmd,s2_cmd=cmd_speed_profile(breakpoints1,s1_all,s2_all)


	z1_all=[z50]*len(breakpoints1)
	z2_all=[z30]*len(breakpoints1)

	
	v1_all,z1_all,v2_all,z2_all=fix_blending_error_multimove(ms,breakpoints1,primitives1,p_bp1,q_bp1,v1_all,z1_all,primitives2,p_bp2,q_bp2,v2_all,z2_all,relative_path,vd_relative,start1,end_1,start_2,end_2)

	#####################################################execute blending failure free commands##################################################################
	logged_data=ms.exec_motions_multimove(primitives1,primitives2,p_bp1,p_bp2,q_bp1,q_bp2,v1_all,v2_all,z1_all,z2_all)

	StringData=StringIO(logged_data)
	df = read_csv(StringData, sep =",")
	##############################data analysis#####################################
	lam, curve_exe1,curve_exe2,curve_exe_R1,curve_exe_R2,curve_exe_js1,curve_exe_js2, speed, timestamp, relative_path_exe,relative_path_exe_R = ms.logged_data_analysis_multimove(df,ms.base2_R,ms.base2_p,realrobot=True)
	#############################chop extension off##################################
	lam, curve_exe1,curve_exe2,curve_exe_R1,curve_exe_R2,curve_exe_js1,curve_exe_js2, speed, timestamp, relative_path_exe, relative_path_exe_R=\
		ms.chop_extension_dual(lam, curve_exe1,curve_exe2,curve_exe_R1,curve_exe_R2,curve_exe_js1,curve_exe_js2, speed, timestamp, relative_path_exe,relative_path_exe_R,relative_path[0,:3],relative_path[-1,:3])
	
	speed1=get_speed(curve_exe1,timestamp)
	speed2=get_speed(curve_exe2,timestamp)
	###calculate error
	error,angle_error=calc_all_error_w_normal(relative_path_exe,relative_path[:,:3],relative_path_exe_R[:,:,-1],relative_path[:,3:])

	fig, ax1 = plt.subplots()

	ax2 = ax1.twinx()
	ax1.plot(lam,speed, 'g-', label='Relative Speed')
	ax1.plot(lam,speed1, 'r-', label='TCP1 Speed')
	# ax1.plot(lam_relative_path[2:],s1_cmd,'p-',label='TCP1 cmd Speed')
	ax1.plot(lam,speed2, 'm-', label='TCP2 Speed')
	# ax1.plot(lam_relative_path[2:],s2_cmd,'p-',label='TCP2 cmd Speed')
	ax2.plot(lam, error, 'b-',label='Error')
	ax2.plot(lam, np.degrees(angle_error), 'y-',label='Normal Error')

	ax1.set_xlabel('lambda (mm)')
	ax1.set_ylabel('Speed/lamdot (mm/s)', color='g')
	ax2.set_ylabel('Error (mm)', color='b')
	plt.title("Speed: "+dataset+'v'+str(vd_relative))
	h1, l1 = ax1.get_legend_handles_labels()
	h2, l2 = ax2.get_legend_handles_labels()
	ax1.legend(h1+h2, l1+l2, loc=1)

	plt.show()

if __name__ == "__main__":
	main_multimove()