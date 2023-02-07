from fanuc_motion_program_exec_client import *
from math import degrees,radians
from pandas import *
from io import StringIO
from pathlib import Path
import pickle

import copy
import sys
sys.path.append('../../../toolbox/')
from utils import *
from robots_def import *
import matplotlib.pyplot as plt

###### put IP address here ######
IP_ADDRESS='127.0.0.2'
#################################

client = FANUCClient(robot_ip=IP_ADDRESS)

###### check your group,uframe number and utool number here ######
group_num = [2]
uframe = [2]
utool = [2]
##################################################################

zone=95
resolution=0.05 ###rad
displacement=radians(5)
waves_num=8

def robot_joint_limits(q,robot,joint):
	
	### robot joint limits
	q2_up_sample=np.radians([20,90])
	# q3_low_sample=np.radians(np.array([-70,0]))
	q3_low_sample=np.radians(np.array([-65,0]))
	# q2_low_sample=np.radians([-90,-65])
	# q3_up_sample=np.radians(np.array([180,205]))
	q2_low_sample=np.radians([-45,45])
	q3_up_sample=np.radians(np.array([10,180]))

	fq3low_q2=interp1d(q2_up_sample,q3_low_sample)
	fq2up_q3=interp1d(q3_low_sample,q2_up_sample)
	fq3up_q2=interp1d(q2_low_sample,q3_up_sample)
	fq2low_q3=interp1d(q3_up_sample,q2_low_sample)

	lower_limit=robot.lower_limit[joint]
	upper_limit=robot.upper_limit[joint]
	if joint==1: # joint 2
		if q<= np.max(q3_low_sample):
			if q>np.min(q3_low_sample):
				upper_limit=fq2up_q3(q)
			else:
				upper_limit=np.min(q2_up_sample)
		if q>=np.min(q3_up_sample):
			if q<np.max(q3_up_sample):
				lower_limit=fq2low_q3(q)
			else:
				lower_limit=np.max(q2_low_sample)
	if joint==2: # joint 3
		if q<= np.max(q2_low_sample):
			if q>np.min(q2_low_sample):
				upper_limit=fq3up_q2(q)
			else:
				upper_limit=np.min(q3_up_sample)
		if q>=np.min(q2_up_sample):
			if q<np.max(q2_up_sample):
				lower_limit=fq3low_q2(q)
			else:
				lower_limit=np.max(q3_low_sample)
	
	return lower_limit,upper_limit

def get_acc(log_results,qds,joint):
	StringData=StringIO(log_results.decode('utf-8'))
	df = read_csv(StringData, sep =",")

	# print(df)
	# exit()

	q1_1=df['J11'].tolist()[1:]
	q1_2=df['J12'].tolist()[1:]
	q1_3=df['J13'].tolist()[1:]
	q1_4=df['J14'].tolist()[1:]
	q1_5=df['J15'].tolist()[1:]
	q1_6=df['J16'].tolist()[1:]
	q2_1=df['J21'].tolist()[1:]
	q2_2=df['J22'].tolist()[1:]
	q2_3=df['J23'].tolist()[1:]
	q2_4=df['J24'].tolist()[1:]
	q2_5=df['J25'].tolist()[1:]
	q2_6=df['J26'].tolist()[1:]
	timestamp=np.round(np.array(df['timestamp'].tolist()[1:]).astype(float)*1e-3,3) # from msec to sec
	curve_exe_js1=np.radians(np.vstack((q1_1,q1_2,q1_3,q1_4,q1_5,q1_6)).T.astype(float))
	curve_exe_js2=np.radians(np.vstack((q2_1,q2_2,q2_3,q2_4,q2_5,q2_6)).T.astype(float))

	###filter
	curve_exe_js1_act=[]
	curve_exe_js2_act=[]
	timestamp_act = []
	dont_show_id=[]
	for i in range(len(curve_exe_js1)):
		if i>5 and i<len(curve_exe_js1)-5:
			# if the recording is not fast enough
			# then having to same logged joint angle
			# do interpolation for estimation
			if np.all(curve_exe_js1[i]==curve_exe_js1[i+1]) and np.all(curve_exe_js2[i]==curve_exe_js2[i+1]):
				dont_show_id=np.append(dont_show_id,i).astype(int)
				continue

		curve_exe_js1_act.append(curve_exe_js1[i])
		curve_exe_js2_act.append(curve_exe_js2[i])
		timestamp_act.append(timestamp[i])

	curve_exe_js1=np.array(curve_exe_js1_act)
	curve_exe_js2=np.array(curve_exe_js2_act)
	timestamp = np.array(timestamp_act)

	# timestamp1, joint_data1=lfilter(timestamp, curve_exe_js1[:,joint])
	# timestamp2, joint_data2=lfilter(timestamp, curve_exe_js2[:,joint])

	joint_data1=curve_exe_js1[:,joint]
	joint_data2=curve_exe_js2[:,joint]
	timestamp1=timestamp
	timestamp2=timestamp

	all_qddot_max_p=[]
	all_qddot_max_n=[]
	###get qdot, qddot
	joint_vel=np.gradient(joint_data1)/np.gradient(timestamp1)
	joint_acc=np.gradient(joint_vel)/np.gradient(timestamp1)
	timestampacc, joint_acc_filter=lfilter(timestamp1[5:-5], joint_acc[5:-5],n=4)
	qddot_sorted=np.sort(joint_acc_filter)
	qddot_max_p=np.average(qddot_sorted[-5:])
	qddot_max_n=-np.average(qddot_sorted[:5])
	all_qddot_max_p.append(qddot_max_p)
	all_qddot_max_n.append(qddot_max_n)

	# plt.plot(timestamp1,joint_data1,label='position')
	# plt.plot(timestamp1,joint_vel,label='velocity')
	# plt.plot(timestamp1[5:-5],joint_acc[5:-5],label='acceleration')
	# plt.plot(timestampacc,joint_acc_filter,label='acceleration filter')
	# plt.legend()
	# plt.show()
	print(qddot_max_p,qddot_max_n)

	joint_vel=np.gradient(joint_data2)/np.gradient(timestamp2)
	joint_acc=np.gradient(joint_vel)/np.gradient(timestamp2)
	timestampacc, joint_acc_filter=lfilter(timestamp2[5:-5], joint_acc[5:-5],n=4)
	qddot_sorted=np.sort(joint_acc_filter)
	qddot_max_p=np.average(qddot_sorted[-5:])
	qddot_max_n=-np.average(qddot_sorted[:5])
	all_qddot_max_p.append(qddot_max_p)
	all_qddot_max_n.append(qddot_max_n)

	# plt.plot(timestamp2,joint_data2,label='position')
	# plt.plot(timestamp2,joint_vel,label='velocity')
	# plt.plot(timestamp2[5:-5],joint_acc[5:-5],label='acceleration')
	# plt.plot(timestampacc,joint_acc_filter,label='acceleration filter')
	# plt.legend()
	# plt.show()
	print(qddot_max_p,qddot_max_n)

	return all_qddot_max_p, all_qddot_max_n

def exec(qds,joint,displacement):
	###move joint at q_d configuration
	
	tps=[]
	for i in range(len(qds)):
		q_init=copy.deepcopy(qds[i])
		q_end=copy.deepcopy(qds[i])
		q_init[joint]+=displacement
		q_end[joint]-=displacement
	
		tp = TPMotionProgram(utool[i],uframe[i])
		jd = jointtarget(group_num[i],uframe[i],utool[i],np.degrees(qds[i]),[0]*6)
		tp.moveJ(jd,100,'%',-1)

		j_init=jointtarget(group_num[i],uframe[i],utool[i],np.degrees(q_init),[0]*6)
		j_end=jointtarget(group_num[i],uframe[i],utool[i],np.degrees(q_end),[0]*6)
		for i in range(waves_num):
			tp.moveJ(j_init,100,'%',zone)
			tp.moveJ(j_end,100,'%',zone)
		tp.moveJ(j_end,100,'%',-1)
		tps.append(tp)
	
	# log_results=client.execute_motion_program_multi(tps[0],tps[1])
	log_results=client.execute_motion_program(tps[0])

	return get_acc(log_results,qds,joint)

# robot1=m10ia(Ry(np.radians(90)),p_tool=np.array([0,0,0]))
robot2=lrmate200id(Ry(np.radians(90)),p_tool=np.array([0,0,0]))
robots=[]
# robots.append(robot1)
robots.append(robot2)

q2_test_lower=radians(-45)
q2_test_upper=radians(70)

### arrange all q2 q3 pairs between robot1 and robot2
robots_qds=[]
total_runs=0
for i in range(len(robots)):
	qds=[]
	for q2 in np.arange(q2_test_lower,q2_test_upper,resolution):

		q3_test_lower,q3_test_upper=robot_joint_limits(q2,robots[i],2)

		for q3 in np.arange(q3_test_lower,q3_test_upper,resolution):
			
			q_d=np.array([radians(90),q2,q3,0,radians(0),0])

			travel_lower_limit = copy.deepcopy(robots[i].lower_limit)
			travel_upper_limit = copy.deepcopy(robots[i].upper_limit)

			travel_lower_limit[2],travel_upper_limit[2]=robot_joint_limits(q_d[1],robots[i],2)
			travel_lower_limit[1],travel_upper_limit[1]=robot_joint_limits(q_d[2],robots[i],1)
			
			if np.any((q_d-displacement)<travel_lower_limit) or np.any((q_d+displacement)>travel_upper_limit):
				continue
			
			qds.append(q_d)
	robots_qds.append(qds)
	if len(qds)>total_runs:
		total_runs=len(qds)

print("Total runs:",total_runs)

# qds=np.array(robots_qds[0])
# plt.scatter(np.degrees(qds[:,1]),np.degrees(qds[:,2]))
# plt.xlabel('q2 (deg)')
# plt.ylabel('q3 (deg)')
# plt.title("m10ia Sampling Points")
# plt.show()
# exit()

#### check all waypoints
show_range=True
if show_range:
	for i in range(1):
		tp = TPMotionProgram(utool[i],uframe[i])
		jd = jointtarget(group_num[i],uframe[i],utool[i],np.degrees(robots_qds[0][0]),[0]*6)
		tp.moveJ(jd,100,'%',-1)

		for qd_tar_i in range(total_runs):
			qd_tar=robots_qds[0][qd_tar_i]
			
			if qd_tar_i>0 and qd_tar_i<total_runs-1:
				if qd_tar[1]!=robots_qds[0][qd_tar_i-1][1] or qd_tar[1]!=robots_qds[0][qd_tar_i+1][1]:

					# joint 1 shake
					jd = jointtarget(group_num[i],uframe[i],utool[i],np.degrees(qd_tar)+np.degrees([displacement,0,0,0,0,0]),[0]*6)
					tp.moveJ(jd,100,'%',-1)
					jd = jointtarget(group_num[i],uframe[i],utool[i],np.degrees(qd_tar)-np.degrees([displacement,0,0,0,0,0]),[0]*6)
					tp.moveJ(jd,100,'%',-1)
					# joint 2 shake
					jd = jointtarget(group_num[i],uframe[i],utool[i],np.degrees(qd_tar)+np.degrees([0,displacement,0,0,0,0]),[0]*6)
					tp.moveJ(jd,100,'%',-1)
					jd = jointtarget(group_num[i],uframe[i],utool[i],np.degrees(qd_tar)-np.degrees([0,displacement,0,0,0,0]),[0]*6)
					tp.moveJ(jd,100,'%',-1)
					# joint 3 shake
					jd = jointtarget(group_num[i],uframe[i],utool[i],np.degrees(qd_tar)+np.degrees([0,0,displacement,0,0,0]),[0]*6)
					tp.moveJ(jd,100,'%',-1)
					jd = jointtarget(group_num[i],uframe[i],utool[i],np.degrees(qd_tar)-np.degrees([0,0,displacement,0,0,0]),[0]*6)
					tp.moveJ(jd,100,'%',-1)
			else:
				continue

		jd = jointtarget(group_num[i],uframe[i],utool[i],np.degrees(robots_qds[0][-1]),[0]*6)
		tp.moveJ(jd,100,'%',-1)	

		log_results=client.execute_motion_program(tp)