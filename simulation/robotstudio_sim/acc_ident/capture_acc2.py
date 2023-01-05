from abb_motion_program_exec import *
from pandas import *
import json
import pickle

import copy
from utils import *
from robots_def import *
import matplotlib.pyplot as plt

client = MotionProgramExecClient(base_url="http://127.0.0.1:80")


mp = MotionProgram()
j0=jointtarget([0]*6,[0]*6)
mp = MotionProgram()
mp.MoveAbsJ(j0,vmax,fine)
client.execute_motion_program(mp)

mp = MotionProgram()

displacement=1

for i in range(5):
	mp.MoveAbsJ(jointtarget([-displacement]+[0]*5,[0]*6),vmax,z10)
	mp.MoveAbsJ(jointtarget([displacement]+[0]*5,[0]*6),vmax,z10)

log_results=client.execute_motion_program(mp)
timestamp=log_results.data[:,0]
joint_data=np.radians(log_results.data[:,2])
joint_vel=np.gradient(joint_data)/np.gradient(timestamp)
joint_acc=np.gradient(joint_vel)/np.gradient(timestamp)
plt.plot(timestamp,joint_data,label='position')
# plt.plot(timestamp,joint_vel,label='velocity')
# plt.plot(timestamp,joint_acc,label='acceleration')
plt.legend()
plt.show()