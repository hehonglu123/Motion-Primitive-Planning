from abb_motion_program_exec_client import *
from io import StringIO
from pandas import *
import json
import pickle

import copy
from utils import *
from robots_def import *
import matplotlib.pyplot as plt

############################JOINT ACCELERATION RECORDING FROM ROBOTSTUDIO#################################
###simple version
#(q2,q3)->(q1,q2,q3) acceleration
#q4, q5, q6 constant acceleration




client = MotionProgramExecClient(base_url="http://127.0.0.1:80")



def jog(q):
	###takes radians
	q=np.degrees(q)
	mp = MotionProgram()
	j=jointtarget(q,[0]*6)
	mp = MotionProgram()
	mp.MoveAbsJ(j,vmax,fine)
	mp.MoveAbsJ(j,v100,fine)
	mp.MoveAbsJ(j,v100,fine)
	log_results=client.execute_motion_program(mp)
	log_results_str = log_results.decode('ascii')
	return log_results_str

def get_acc(log_results_str,q,joint,clipped=False):
	StringData=StringIO(log_results_str)
	df = read_csv(StringData, sep =",")
	q1=df[' J1'].tolist()
	q2=df[' J2'].tolist()
	q3=df[' J3'].tolist()
	q4=df[' J4'].tolist()
	q5=df[' J5'].tolist()
	q6=df[' J6'].tolist()
	curve_exe_js=np.radians(np.vstack((q1,q2,q3,q4,q5,q6)).T.astype(float))
	# start_idx=np.argwhere(curve_exe_js[:,joint]!=curve_exe_js[0,joint])[0][0]
	# print(start_idx)
	timestamp=np.array(df['timestamp'].tolist()).astype(float)

	###find closet idx
	idx=np.argmin(np.abs(curve_exe_js[:,joint]-q[joint]))

	if clipped:
		idx=max(np.argmin(np.abs(curve_exe_js[:,joint]-(q[joint]-0.05))),np.argmin(np.abs(curve_exe_js[:,joint]-(q[joint]+0.05))))
		# if idx==0:
		# 	print(curve_exe_js[:,joint])
		# 	print(q,joint)


	###filter
	timestamp, curve_exe_js=lfilter(timestamp, curve_exe_js)
	###get qdot, qddot
	qdot_all=np.gradient(curve_exe_js,axis=0)/np.tile([np.gradient(timestamp)],(6,1)).T
	qddot_all=np.gradient(qdot_all,axis=0)/np.tile([np.gradient(timestamp)],(6,1)).T


	###plot profile
	# plt.plot(curve_exe_js[:,joint],qdot_all[:,joint],label='qdot')
	# plt.plot(curve_exe_js[:,joint],qddot_all[:,joint],label='qddot')
	# plt.legend()
	# plt.xlabel('q')
	# plt.ylabel('speed & acceleration')
	# plt.title('joint 1 profile')
	# plt.show()

	return np.max(np.abs(qddot_all[0:2*idx,joint]))

def exec(q_d,joint):
	###move joint at q_d configuration
	q_init=copy.deepcopy(q_d)
	q_end=copy.deepcopy(q_d)
	q_init[joint]-=0.1
	q_end[joint]+=1
	###if end outside boundary, move in other direction
	if q_end[joint]>robot.upper_limit[joint]:
		q_init[joint]=q_d[joint]+0.1
		q_end[joint]=q_d[joint]-1
	###clip start within limits
	if q_init[joint]<robot.lower_limit[joint] or q_init[joint]>robot.upper_limit[joint]:
		q_init=np.clip(q_init,robot.lower_limit,robot.upper_limit)
		clipped=True
	else:
		clipped=False
	jog(q_init)
	time.sleep(0.1)
	log_results_str=jog(q_end)
	qdot_max=get_acc(log_results_str,q_d,joint,clipped)

	return qdot_max

# robot=abb6640()
robot=abb1200()
resolution=0.02 ###rad

dict_table={}


#####################first & second joint acc both depends on second and third joint#####################################
for q2 in np.arange(robot.lower_limit[1],robot.upper_limit[1],resolution):
	for q3 in np.arange(robot.lower_limit[2],robot.upper_limit[2],resolution):
		###initialize keys, and desired pose
		dict_table[(q2,q3)]=[0]*3
		q_d=[0,q2,q3,0,0,0]

		for joint in range(3):
			###move first q1, q2 and q3
			qdot_max=exec(q_d,joint)
			###update dict
			dict_table[(q2,q3)][joint]=copy.deepcopy(qdot_max)


with open(r'test.txt','w+') as f:
	f.write(str(dict_table))
pickle.dump(dict_table, open('test.pickle','wb'))



###############################joint 4,5,6 constant acceleration measurement##############################################
# for joint in range(3,6):
# 	q_d=np.zeros(6)
# 	qdot_max=exec(q_d,joint)
# 	print('joint '+str(joint+1),qdot_max)