# from abb_motion_program_exec_client import *
from re import L
from fanuc_motion_program_exec_client import *
from io import StringIO
from pandas import *
import json
import pickle
from math import radians,degrees

import copy
import sys
sys.path.append('../../../toolbox/')
from robots_def import *
import matplotlib.pyplot as plt

############################JOINT ACCELERATION RECORDING FROM ROBOTSTUDIO#################################
###simple version
#(q2,q3)->(q1,q2,q3) acceleration
#q4, q5, q6 constant acceleration




# client = MotionProgramExecClient(base_url="http://127.0.0.1:80")
client = FANUCClient()

group_num = 1
uframe = 1
utool = 2

def jog(q):
	###takes radians
	q=np.degrees(q)
	# mp = MotionProgram()
	# j=jointtarget(q,[0]*6)
	# mp = MotionProgram()
	# mp.MoveAbsJ(j,vmax,fine)
	# mp.MoveAbsJ(j,v100,fine)
	# mp.MoveAbsJ(j,v100,fine)
	tp = TPMotionProgram()
	j = jointtarget(group_num,uframe,utool,q,[0]*6)
	tp.moveJ(j,100,'%',-1)
	tp.moveJ(j,10,'%',-1)
	tp.moveJ(j,10,'%',-1)
	return client.execute_motion_program(tp)

def get_acc(log_results,q,joint,clipped=False):
	StringData=StringIO(log_results.decode('utf-8'))
	df = read_csv(StringData, sep =",")
	
	q1=df['J1'].tolist()[1:]
	q2=df['J2'].tolist()[1:]
	q3=df['J3'].tolist()[1:]
	q4=df['J4'].tolist()[1:]
	q5=df['J5'].tolist()[1:]
	q6=df['J6'].tolist()[1:]
	curve_exe_js=np.radians(np.vstack((q1,q2,q3,q4,q5,q6)).T.astype(float))
	timestamp=np.array(df['timestamp'].tolist()[1:]).astype(float)*1e-3 # from msec to sec

	curve_exe_js_act=[]
	timestamp_act = []
	dont_show_id=[]
	for i in range(len(curve_exe_js)):
		this_q = curve_exe_js[i]
		if i>5 and i<len(curve_exe_js)-5:
			# if the recording is not fast enough
			# then having to same logged joint angle
			# do interpolation for estimation
			if np.all(this_q==curve_exe_js[i+1]):
				dont_show_id=np.append(dont_show_id,i).astype(int)
				continue

		curve_exe_js_act.append(this_q)
		timestamp_act.append(timestamp[i])

	curve_exe_js=np.array(curve_exe_js_act)
	timestamp = np.array(timestamp_act)

	###find closet idx
	idx=np.argmin(np.abs(curve_exe_js[:,joint]-q[joint]))

	if clipped:
		idx=max(np.argmin(np.abs(curve_exe_js[:,joint]-(q[joint]-0.05))),np.argmin(np.abs(curve_exe_js[:,joint]-(q[joint]+0.05))))
		# if idx==0:
		# 	print(curve_exe_js[:,joint])
		# 	print(q,joint)

	###get qdot, qddot
	qdot_all=np.gradient(curve_exe_js,axis=0)/np.tile([np.gradient(timestamp)],(6,1)).T
	qddot_all=np.gradient(qdot_all,axis=0)/np.tile([np.gradient(timestamp)],(6,1)).T

	max_idx=np.argmax(np.abs(qddot_all[0:2*idx,joint]))
	qddot_max=np.average(np.abs(qddot_all[max(max_idx-2,0):max_idx+3,joint]))

	###plot profile
	# plt.plot(curve_exe_js[:,joint],qdot_all[:,joint],label='qdot')
	# plt.plot(curve_exe_js[:,joint],qddot_all[:,joint],label='qddot')
	# plt.legend()
	# plt.xlabel('q')
	# plt.ylabel('speed & acceleration')
	# plt.title('joint 1 profile')
	# plt.show()

	return qddot_max

def exec(q_d,joint):
	###move joint at q_d configuration
	q_init=copy.deepcopy(q_d)
	q_end=copy.deepcopy(q_d)
	q_init[joint]-=0.1
	q_end[joint]+=1

	### configuration depend upper/lower limits
	lower_limit = copy.deepcopy(robot.lower_limit)
	upper_limit = copy.deepcopy(robot.upper_limit)

	if q_init[1] < radians(-54):
		upper_limit[2] = radians(180)
	if q_end[1] > radians(20):
		lower_limit[2] = radians(-80)+((q_end[1]-radians(20))/2)

	if q_end[2] > radians(180):
		lower_limit[1] = radians(-54)
	if q_init[2] < radians(-45):
		upper_limit[1] = radians(90)-((radians(-45)-q_init[2])*2)

	###if end outside boundary, move in other direction
	# if q_end[joint]>upper_limit[joint]:
	if np.any(np.array(q_end)>np.array(upper_limit)) or np.any(np.array(q_end)<np.array(lower_limit)) or \
		 np.any(np.array(q_init)>np.array(upper_limit)) or np.any(np.array(q_init)<np.array(lower_limit)):
		q_init[joint]=q_d[joint]+0.1
		q_end[joint]=q_d[joint]-1
		
		# ensure the ending pose is smaller than limit
		dang = 1
		while q_end[joint]<lower_limit[joint]:
			dang = dang*0.9
			q_end[joint]=q_d[joint]-dang
			
			if dang < 0.3:
				print("dang too small")
				raise AssertionError

	###clip start within limits
	if q_init[joint]<lower_limit[joint] or q_init[joint]>upper_limit[joint]:
		q_init=np.clip(q_init,lower_limit,upper_limit)
		clipped=True
	else:
		clipped=False
	jog(q_init)
	time.sleep(0.1)
	log_results=jog(q_end)
	qdot_max=get_acc(log_results,q_d,joint,clipped)

	return qdot_max

# robot=abb6640()
# robot=abb1200()
robot=m710ic(d=50)
resolution=0.05 ###rad

dict_table={}

#####################first & second joint acc both depends on second and third joint#####################################
# jog([0,0,0,0,0,0])
# q2_test_lower = robot.lower_limit[1]+resolution
q2_test_lower = robot.lower_limit[1]+resolution*45
q2_test_upper = robot.upper_limit[1]
for q2 in np.arange(q2_test_lower,q2_test_upper,resolution):
	
	if q2 < radians(-54):
		q3_test_upper = radians(180)
	else:
		q3_test_upper = robot.upper_limit[2]
	
	if q2 > radians(20):
		q3_test_lower = radians(-80)+((q2-radians(20))/2)
	else:
		q3_test_lower = robot.lower_limit[2]

	for q3 in np.arange(q3_test_lower,q3_test_upper,resolution):
		###initialize keys, and desired pose
		dict_table[(q2,q3)]=[0]*3
		q_d=[0,q2,q3,0,0,0]

		print(q_d)
		for joint in range(3):
			print(joint)
			###move first q1, q2 and q3
			qdot_max=exec(q_d,joint)
			###update dict
			dict_table[(q2,q3)][joint]=copy.deepcopy(qdot_max)
		print("===================================")
		
		# save when a qd is finished
		with open('m710ic/acc4.txt','w+') as f:
			f.write(str(dict_table))
		pickle.dump(dict_table, open('m710ic/acc4.pickle','wb'))






###############################joint 4,5,6 constant acceleration measurement##############################################
# for joint in range(3,6):
# 	q_d=np.zeros(6)
# 	qdot_max=exec(q_d,joint)
# 	print('joint '+str(joint+1),qdot_max)