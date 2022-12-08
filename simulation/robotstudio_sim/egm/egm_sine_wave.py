import numpy as np
from general_robotics_toolbox import *
from pandas import read_csv
import sys
from io import StringIO
sys.path.append('egm_toolbox')
import rpi_abb_irc5
from EGM_toolbox import *
from robots_def import *


def main():
	robot=abb6640(d=50)

	egm = rpi_abb_irc5.EGM()

	et=EGM_toolbox(egm,robot)
	et.jog_joint(np.zeros(6))

	curve_exe_js=[]
	timestamp=[]

	amp=0.3#rad

	for freq in range(1,10):
		time_traj=np.arange(0,1,0.004)
		joint_traj=amp*np.sin(2*np.pi*freq*time_traj)
		print(joint_traj.reshape((-1,1)))
		joint_traj_1to6=np.zeros((len(joint_traj),5))
		
		joint_traj_all=np.hstack((joint_traj.reshape((-1,1)),joint_traj_1to6))
		timestamp,curve_exe_js=et.traverse_curve_js(joint_traj_all)


		# try:
		# 	while True:
		# 		res_i, state_i = egm.receive_from_robot()
		# 		if res_i:
		# 			send_res = egm.send_to_robot(curve_cmd_js_ext[0])
		# 			#save joint angles
		# 			curve_exe_js.append(np.radians(state_i.joint_angles))
		# 			#TODO: replace with controller time
		# 			timestamp.append(state_i.robot_message.header.tm)
		# 			break

		# except KeyboardInterrupt:
		# 	raise


if __name__ == "__main__":
	main()