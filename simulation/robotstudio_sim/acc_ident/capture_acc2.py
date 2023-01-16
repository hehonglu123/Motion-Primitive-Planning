from abb_motion_program_exec import *
from pandas import *
import json
import pickle

import copy
from utils import *
from robots_def import *
import matplotlib.pyplot as plt

client = MotionProgramExecClient(base_url="http://127.0.0.1:80")


def get_acc(log_results,q,joint):
	curve_exe_js=np.radians(log_results.data[:,2:8])
	timestamp=log_results.data[:,0]

	###filter
	timestamp, curve_exe_js=lfilter(timestamp, curve_exe_js)
	###get qdot, qddot
	qdot_all=np.gradient(curve_exe_js,axis=0)/np.tile([np.gradient(timestamp)],(6,1)).T
	qddot_all=np.gradient(qdot_all,axis=0)/np.tile([np.gradient(timestamp)],(6,1)).T

	qddot_sorted=np.sort(qddot_all[:,joint])

	qddot_max_p=np.average(qddot_sorted[-5:])
	qddot_max_n=-np.average(qddot_sorted[:5])

	return qddot_max_p, qddot_max_n

def exec(q_d,joint,displacement):
	###move joint at q_d configuration
	q_init=copy.deepcopy(q_d)
	q_end=copy.deepcopy(q_d)
	q_init[joint]+=displacement
	q_end[joint]-=displacement
	
	mp = MotionProgram()
	mp.MoveAbsJ(jointtarget(np.degrees(q_d),[0]*6),vmax,fine)

	j_init=jointtarget(np.degrees(q_init),[0]*6)
	j_end=jointtarget(np.degrees(q_end),[0]*6)
	for i in range(4):
		mp.MoveAbsJ(j_init,vmax,z10)
		mp.MoveAbsJ(j_end,vmax,z10)
	
	log_results=client.execute_motion_program(mp)

	return get_acc(log_results,q_d,joint)

robot=robot_obj('ABB_6640_180_255','../../../config/abb_6640_180_255_robot_default_config.yml',tool_file_path='../../../config/paintgun.csv',d=50,acc_dict_path='')
resolution=0.05 ###rad
displacement=0.02

dict_table={}
directions=[-1,1]

#####################first & second joint acc both depends on second and third joint#####################################
for q2 in np.arange(robot.lower_limit[1]+displacement+0.01,robot.upper_limit[1]-displacement-0.01,resolution):
	for q3 in np.arange(robot.lower_limit[2]+displacement+0.01,robot.upper_limit[2]-displacement-0.01,resolution):
		###initialize keys, and desired pose

		dict_table[(q2,q3)]=[0]*6 		###[+j1,-j1,+j2,-j2,+j3,-j3]
		q_d=[0,q2,q3,0,0,0]

		#measure first joint first
		qddot_max,_=exec(q_d,0,displacement)
		###update dict
		dict_table[(q2,q3)][0]=qddot_max
		dict_table[(q2,q3)][1]=qddot_max

		for joint in range(1,3):
			###move first q2 and q3
			qddot_max_p,qddot_max_n=exec(q_d,joint,displacement)
			###update dict
			dict_table[(q2,q3)][2*joint]=qddot_max_p
			dict_table[(q2,q3)][2*joint+1]=qddot_max_n


with open(r'test.txt','w+') as f:
	f.write(str(dict_table))
pickle.dump(dict_table, open('test.pickle','wb'))