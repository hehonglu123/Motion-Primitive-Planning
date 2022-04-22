import rpi_abb_irc5
import numpy as np
import time
from pandas import *

egm = rpi_abb_irc5.EGM()

col_names=['q1', 'q2', 'q3','q4', 'q5', 'q6'] 
data = read_csv('../../../data/from_ge/Curve_js2.csv', names=col_names)
curve_q1=data['q1'].tolist()
curve_q2=data['q2'].tolist()
curve_q3=data['q3'].tolist()
curve_q4=data['q4'].tolist()
curve_q5=data['q5'].tolist()
curve_q6=data['q6'].tolist()
curve_js=np.vstack((curve_q1, curve_q2, curve_q3,curve_q4,curve_q5,curve_q6)).T

lam_des=500
lam_f=1758
steps=lam_f/lam_des/0.004
step_size=len(curve_js)/steps

q_cur=np.zeros(6)
###move to start first
try:
	while np.linalg.norm(q_cur-curve_js[0])>0.001:
		res, state = egm.receive_from_robot(.1)
		if res:
			# Clear queue
			i = 0
			while True:
				res_i, state_i = egm.receive_from_robot()
				if res_i: # there was another msg waiting
					state = state_i
					i += 1
				else: # previous msg was end of queue
					break

			if i > 0:
				print("Number of extra msgs in queue: ", i)

			send_res = egm.send_to_robot(curve_js[0])
			q_cur=np.radians(state.joint_angles)
except KeyboardInterrupt:
	raise


try:
	while True:
		res, state = egm.receive_from_robot(.1)

		if res:
			# Clear queue
			i = 0
			while True:
				res_i, state_i = egm.receive_from_robot()
				if res_i: # there was another msg waiting
					state = state_i
					i += 1
				else: # previous msg was end of queue
					break

			if i > 0:
				print("Number of extra msgs in queue: ", i)

			send_res = egm.send_to_robot(curve_js[int(step_size*i)])
			print(state.joint_angles)

except KeyboardInterrupt:
	raise