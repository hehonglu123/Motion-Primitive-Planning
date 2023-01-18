# from abb_motion_program_exec import *
from fanuc_motion_program_exec_client import *
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

group_num1 = 1
uframe1 = 1
utool1 = 4
group_num2 = 2
uframe2 = 1
utool2 = 4

tp1 = TPMotionProgram(utool1,uframe1)
j10=jointtarget(group_num1,uframe1,utool1,[0,-1.1915417547223457,-0.19154175472231888,0,0,0],[0]*6)
tp1.moveJ(j10,100,'%',-1)
tp2 = TPMotionProgram(utool2,uframe2)
j20=jointtarget(group_num2,uframe2,utool2,[0,-1.1915417547223457,-0.19154175472231888,0,0,0],[0]*6)
tp2.moveJ(j20,100,'%',-1)
# client.execute_motion_program_coord(tp2,tp1)
# client.execute_motion_program_multi(tp1,tp2)

## oscillation test
displacement=5

# tp1 = TPMotionProgram(utool1,uframe1)
# tp2 = TPMotionProgram(utool2,uframe2)

zone=95
for i in range(8):
	# mp.MoveAbsJ(jointtarget([-displacement]+[0]*5,[0]*6),vmax,z10)
	# mp.MoveAbsJ(jointtarget([displacement]+[0]*5,[0]*6),vmax,z10)

	j1s=jointtarget(group_num1,uframe1,utool1,[0,-1.1915417547223457,-0.19154175472231888-displacement,0,0,0],[0]*6)
	j1e=jointtarget(group_num1,uframe1,utool1,[0,-1.1915417547223457,-0.19154175472231888+displacement,0,0,0],[0]*6)
	tp1.moveJ(j1s,100,'%',zone)
	tp1.moveJ(j1e,100,'%',zone)
	j2s=jointtarget(group_num2,uframe2,utool2,[0,-1.1915417547223457,-0.19154175472231888-displacement,0,0,0],[0]*6)
	j2e=jointtarget(group_num2,uframe2,utool2,[0,-1.1915417547223457,-0.19154175472231888+displacement,0,0,0],[0]*6)
	tp2.moveJ(j2s,100,'%',zone)
	tp2.moveJ(j2e,100,'%',zone)

# log_results=client.execute_motion_program_coord(tp2,tp1)
log_results=client.execute_motion_program_multi(tp1,tp2)
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

joint_data=curve_exe_js1[:,2]
joint_vel=np.gradient(joint_data)/np.gradient(timestamp)
joint_acc=np.gradient(joint_vel)/np.gradient(timestamp)
plt.plot(timestamp,joint_data,label='position')
plt.plot(timestamp,joint_vel,label='velocity')
plt.plot(timestamp,joint_acc,label='acceleration')
plt.legend()
plt.show()

joint_data=curve_exe_js2[:,2]
joint_vel=np.gradient(joint_data)/np.gradient(timestamp)
joint_acc=np.gradient(joint_vel)/np.gradient(timestamp)
plt.plot(timestamp,joint_data,label='position')
plt.plot(timestamp,joint_vel,label='velocity')
plt.plot(timestamp,joint_acc,label='acceleration')
plt.legend()
plt.show()