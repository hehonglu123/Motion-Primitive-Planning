import numpy as np
import time, sys
from pandas import *


sys.path.append('../../../toolbox')
from robots_def import *
from error_check import *
from lambda_calc import *
sys.path.append('../../../toolbox/egm_toolbox')

import rpi_abb_irc5

egm = rpi_abb_irc5.EGM()

dataset='wood/'
data_dir='../../../data/'
curve_js = read_csv(data_dir+dataset+'Curve_js.csv',header=None).values
curve = read_csv(data_dir+dataset+'Curve_in_base_frame.csv',header=None).values
vd=200

robot=abb6640(d=50)
lam=calc_lam_cs(curve[:,:3])
ts=0.004

steps=int((lam[-1]/vd)/ts)
idx=np.linspace(0.,len(curve_js)-1,num=steps).astype(int)
curve_js=curve_js[idx]

res, state = egm.receive_from_robot(.1)
q_cur=np.radians(state.joint_angles)
num=int(np.linalg.norm(curve_js[0]-q_cur)/ts)
curve2start=np.linspace(q_cur,curve_js[0],num=num)

###move to start first
print('moving to start point')
try:
	for i in range(len(curve2start)):
		res_i, state_i = egm.receive_from_robot(ts)
		send_res = egm.send_to_robot(curve2start[i])

	for i in range(500):
		while True:
			res_i, state_i = egm.receive_from_robot()
			if res_i:
				send_res = egm.send_to_robot(curve_js[0])
				break

except KeyboardInterrupt:
	raise

# # Clear queue
# while True:
# 	res_i, state_i = egm.receive_from_robot()
# 	if not res_i:
# 		break

curve_exe_js=[]
timestamp=[]
###traverse curve
print('traversing trajectory')
try:
	for i in range(len(curve_js)):
		while True:
			res_i, state_i = egm.receive_from_robot()
			if res_i:
				send_res = egm.send_to_robot(curve_js[i])
				#save joint angles
				curve_exe_js.append(np.radians(state_i.joint_angles))
				#TODO: replace with controller time
				timestamp.append(state_i.robot_message.header.tm)
				break
except KeyboardInterrupt:
	raise

plt.plot(timestamp)
plt.show()
# DataFrame(np.hstack((np.array(timestamp).reshape((-1,1)),curve_exe_js))).to_csv(dataset+'curve_exe_v'+str(vd)+'.csv',header=False,index=False)