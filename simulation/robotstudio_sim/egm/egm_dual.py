import numpy as np
from general_robotics_toolbox import *
import sys
sys.path.append('../../../toolbox')
from robots_def import *
from error_check import *
from MotionSend import *
from lambda_calc import *
sys.path.append('../../../toolbox/egm_toolbox')
from EGM_toolbox import *
import rpi_abb_irc5

def main():
	data_dir='../../../data/wood/'

	robot1=abb6640(d=50)
	with open(data_dir+'dual_arm/abb1200.yaml') as file:
		H_1200 = np.array(yaml.safe_load(file)['H'],dtype=np.float64)

	base2_R=H_1200[:3,:3]
	base2_p=1000*H_1200[:-1,-1]

	with open(data_dir+'dual_arm/tcp.yaml') as file:
		H_tcp = np.array(yaml.safe_load(file)['H'],dtype=np.float64)
	robot2=abb1200(R_tool=H_tcp[:3,:3],p_tool=H_tcp[:-1,-1])

	egm1 = rpi_abb_irc5.EGM(port=6510)
	egm2 = rpi_abb_irc5.EGM(port=6511)

	EGM_robot1=EGM_toolbox(egm1,robot1)
	EGM_robot2=EGM_toolbox(egm2,robot2)

	curve_js1=read_csv(data_dir+"dual_arm/arm1.csv",header=None).values
	curve_js2=read_csv(data_dir+"dual_arm/arm2.csv",header=None).values
	relative_path=read_csv(data_dir+"Curve_dense.csv",header=None).values

	vd=2000
	idx=EGM_robot1.downsample24ms(relative_path,vd)

	###jog both arm to start pose
	pose1=robot1.fwd(curve_js1[0])
	EGM_robot1.jog_joint_cartesian(pose1.p,pose1.R)
	EGM_robot2.jog_joint(curve_js2[0])
	# pose2=robot2.fwd(curve_js2[0])
	# EGM_robot2.jog_joint_cartesian(pose2.p,pose2.R)

	################################traverse curve for both arms#####################################

	curve_exe_js1=[]
	timestamp1=[]
	curve_exe_js2=[]
	timestamp2=[]
	print('traversing trajectory')
	try:
		for i in idx:
			while True:
				res_i1, state_i1 = egm1.receive_from_robot()
				res_i2, state_i2 = egm2.receive_from_robot()
				if res_i1 and res_i2:
					pose1=robot1.fwd(curve_js1[i])
					send_res1 = egm1.send_to_robot_cart(pose1.p, R2q(pose1.R))
					#save joint angles
					curve_exe_js1.append(np.radians(state_i1.joint_angles))
					timestamp1.append(state_i1.robot_message.header.tm)

					

					# pose2=robot2.fwd(curve_js2[i])
					# send_res2 = egm2.send_to_robot_cart(pose2.p, R2q(pose2.R))
					send_res2 = egm2.send_to_robot(curve_js2[i])
					#save joint angles
					curve_exe_js2.append(np.radians(state_i2.joint_angles))
					timestamp2.append(state_i2.robot_message.header.tm)
					break
	except KeyboardInterrupt:
		raise
	
	timestamp1=np.array(timestamp1)/1000
	timestamp2=np.array(timestamp2)/1000
if __name__ == '__main__':
	main()