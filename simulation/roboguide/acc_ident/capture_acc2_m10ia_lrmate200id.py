from fanuc_motion_program_exec_client import *
from math import degrees,radians
from pandas import *
from io import StringIO
import json
import pickle

import copy
import sys
sys.path.append('../../../toolbox/')
from utils import *
from robots_def import *
import matplotlib.pyplot as plt

client = FANUCClient()

group_num = [1,2]
uframe = [1,1]
utool = [4,4]

zone=95
resolution=0.05 ###rad
displacement=radians(5)
waves_num=8

def get_acc(log_results,qds,joint):
	StringData=StringIO(log_results.decode('utf-8'))
	df = read_csv(StringData, sep =",")

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
	
	log_results=client.execute_motion_program_multi(tps[0],tps[1])

	return get_acc(log_results,qds,joint)

robot1=m10ia(Ry(np.radians(90)),p_tool=np.array([0,0,0]))
robot2=lrmate200id(Ry(np.radians(90)),p_tool=np.array([0,0,0]))
robots=[]
robots.append(robot1)
robots.append(robot2)

### robot joint limits
q2_up_sample=[]
q3_low_sample =[]
q2_low_sample = []
q3_up_sample = []
fq3low_q2=[]
fq2up_q3=[]
fq3up_q2=[]
fq2low_q3=[]

# robot1 limits
q2_up_sample.append(np.radians(np.arange(40,160+0.1,10)))
q3_low_sample.append(np.radians(np.array([-89,-87.5,-84.5,-79,-71.5,-63.5,-57.5,-51,-45,-38.5,-32.5,-26,-19.5])))
q2_low_sample.append(np.radians(np.array([-80,-85,-90])))
q3_up_sample.append(np.radians(np.array([180,177.5,174.5])))
# robot2 limits
q2_up_sample.append(np.radians(np.arange(60,145+0.1,5)))
q3_low_sample.append(np.radians(np.array([-70,-68,-65.5,-62.5,-59,-54.5,-50,-44.5,-39,-32.5,-25.5,-18,-10,-2,6.5,15,24,33])))
q2_low_sample.append(np.radians(np.arange(-100,-75+0.1,5)))
q3_up_sample.append(np.radians(np.array([180,185,190,195,200,205])))

for i in range(len(q2_up_sample)):
	fq3low_q2.append(interp1d(q2_up_sample[i],q3_low_sample[i]))
	fq2up_q3.append(interp1d(q3_low_sample[i],q2_up_sample[i]))
	fq3up_q2.append(interp1d(q2_low_sample[i],q3_up_sample[i]))
	fq2low_q3.append(interp1d(q3_up_sample[i],q2_low_sample[i]))

### arrange all q2 q3 pairs between robot1 and robot2
robots_qds=[]
total_runs=0
for i in range(len(robots)):
	qds=[]
	q2_test_lower = robots[i].lower_limit[1]
	q2_test_upper = robots[i].upper_limit[1]
	for q2 in np.arange(q2_test_lower,q2_test_upper,resolution):

		if q2 < np.max(q2_low_sample[i]):
			q3_test_upper =fq3up_q2[i](q2)
		else:
			q3_test_upper = np.max(q3_up_sample[i])
		if q2 > np.min(q2_up_sample[i]):
			q3_test_lower = fq3low_q2[i](q2)
		else:
			q3_test_lower = np.min(q3_low_sample[i])

		for q3 in np.arange(q3_test_lower,q3_test_upper,resolution):
			q_d=np.array([0,q2,q3,0,0,0])

			travel_lower_limit = copy.deepcopy(robots[i].lower_limit)
			travel_upper_limit = copy.deepcopy(robots[i].upper_limit)

			## get actual travel limits
			if q_d[2] > np.min(q3_up_sample[i]):
				if q_d[2] <= np.max(q3_up_sample[i]):
					travel_lower_limit[1] = fq2low_q3[i](q_d[2])
				else:
					travel_lower_limit[1] = np.max(q2_low_sample[i])
			if q_d[2] < np.max(q3_low_sample[i]):
				if q_d[2] >= np.min(q3_low_sample[i]):
					travel_upper_limit[1] = fq2up_q3[i](q_d[2])
				else:
					travel_upper_limit[1] = np.min(q2_up_sample[i])
			if q_d[1] < np.max(q2_low_sample[i]):
				if q_d[1] >=  np.min(q2_low_sample[i]):
					travel_upper_limit[2] = fq3up_q2[i](q_d[1])
				else:
					travel_upper_limit[2] = np.min(q3_up_sample[i])
			if q_d[1] > np.min(q2_up_sample[i]):
				if q_d[1] <= np.max(q2_up_sample[i]):
					travel_lower_limit[2] = fq3low_q2[i](q_d[1])
				else:
					travel_lower_limit[2] = np.max(q3_low_sample[i])
			
			if np.any((q_d-displacement)<travel_lower_limit) or np.any((q_d+displacement)>travel_upper_limit):
				continue
			
			qds.append(q_d)
	robots_qds.append(qds)
	if len(qds)>total_runs:
		total_runs=len(qds)

print("robot1 total:",len(robots_qds[0]))
print("robot2 total:",len(robots_qds[1]))
print("Total runs:",total_runs)

dict_tables=[{},{}]

#####################first & second joint acc both depends on second and third joint#####################################
for i_run in range(4681,total_runs):
	print(i_run,'in total',total_runs)

	qds=[]
	for i in range(len(robots)):
		qds.append(robots_qds[i][i_run%len(robots_qds[i])])
		dict_tables[i][(qds[i][1],qds[i][2])]=[0]*6 ### dict_table[(q2,q3)]=[0]*6, [+j1,-j1,+j2,-j2,+j3,-j3]

	for joint in range(3):
		print("joint",joint)
		all_qddot_max_p,all_qddot_max_n=exec(qds,joint,displacement)
		for i in range(len(robots)):
			dict_tables[i][(qds[i][1],qds[i][2])][2*joint]=all_qddot_max_p[i]
			dict_tables[i][(qds[i][1],qds[i][2])][2*joint+1]=all_qddot_max_n[i]
	print("================================")
	
	# save when a qd is finished
	with open('m10ia/acc_shake3.txt','w+') as f:
		f.write(str(dict_tables[0]))
	pickle.dump(dict_tables[0], open('m10ia/acc_shake3.pickle','wb'))
	with open('lrmate200id/acc_shake3.txt','w+') as f:
		f.write(str(dict_tables[1]))
	pickle.dump(dict_tables[1], open('lrmate200id/acc_shake3.pickle','wb'))

# for q2 in np.arange(robot.lower_limit[1]+displacement+0.01,robot.upper_limit[1]-displacement-0.01,resolution):
# 	for q3 in np.arange(robot.lower_limit[2]+displacement+0.01,robot.upper_limit[2]-displacement-0.01,resolution):
# 		###initialize keys, and desired pose

# 		dict_table[(q2,q3)]=[0]*6 		###[+j1,-j1,+j2,-j2,+j3,-j3]
# 		q_d=[0,q2,q3,0,0,0]

# 		#measure first joint first
# 		qddot_max,_=exec(q_d,0,displacement)
# 		###update dict
# 		dict_table[(q2,q3)][0]=qddot_max
# 		dict_table[(q2,q3)][1]=qddot_max

# 		for joint in range(1,3):
# 			###move first q2 and q3
# 			qddot_max_p,qddot_max_n=exec(q_d,joint,displacement)
# 			###update dict
# 			dict_table[(q2,q3)][2*joint]=qddot_max_p
# 			dict_table[(q2,q3)][2*joint+1]=qddot_max_n