import numpy as np
import time, sys
from pandas import *
sys.path.append('egm_toolbox')
import rpi_abb_irc5

sys.path.append('../../../toolbox')
from robots_def import *
from error_check import *
from lambda_calc import *

egm = rpi_abb_irc5.EGM()
robot=abb6640(d=50)

dataset='wood/'
data_dir='../../../data/'
curve_js = read_csv(data_dir+dataset+'Curve_js.csv',header=None).values
###get cartesian pose
curve=[]
curve_R=[]
for q in curve_js:
	pose=robot.fwd(q)
	curve.append(pose.p)
	curve_R.append(pose.R)

curve=np.array(curve)
curve_R=np.array(curve_R)


vd=50

lam=calc_lam_cs(curve[:,:3])
ts=0.004

steps=int((lam[-1]/vd)/ts)
idx=np.linspace(0.,len(curve_js)-1,num=steps).astype(int)
curve_cmd_js=curve_js[idx]
curve_cmd=curve[idx]
curve_cmd_R=curve_R[idx]

res, state = egm.receive_from_robot(.1)

q_cur=np.radians(state.joint_angles)
pose_cur=robot.fwd(q_cur)

num=int(np.linalg.norm(curve[0]-pose_cur.p)/(1000*ts))

curve2start=np.linspace(pose_cur.p,curve[0],num=num)
R2start=orientation_interp(pose_cur.R,curve_R[0],num)
quat2start=[]
for R in R2start:
	quat2start.append(R2q(R))

###move to start first
print('moving to start point')
try:
	for i in range(len(curve2start)):
		while True:
			res_i, state_i = egm.receive_from_robot()
			if res_i:
				send_res = egm.send_to_robot_cart(curve2start[i], quat2start[i])
				break

	for i in range(500):
		while True:
			res_i, state_i = egm.receive_from_robot()
			if res_i:
				send_res = egm.send_to_robot_cart(curve[0], R2q(curve_R[0]))
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
	for i in range(len(curve_cmd_js)):
		while True:
			res_i, state_i = egm.receive_from_robot()
			if res_i:
				send_res = egm.send_to_robot_cart(curve_cmd[i], R2q(curve_cmd_R[i]))
				#save joint angles
				curve_exe_js.append(np.radians(state_i.joint_angles))
				#TODO: replace with controller time
				timestamp.append(state_i.robot_message.header.tm)
				break
except KeyboardInterrupt:
	raise

# DataFrame(np.hstack((np.array(timestamp).reshape((-1,1)),curve_exe_js))).to_csv(dataset+'curve_exe_v'+str(vd)+'.csv',header=False,index=False)