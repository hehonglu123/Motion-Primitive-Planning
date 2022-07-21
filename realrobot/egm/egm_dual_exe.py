import numpy as np
from general_robotics_toolbox import *
import sys
sys.path.append('../../toolbox')
from robots_def import *
from error_check import *
from MotionSend import *
from lambda_calc import *
sys.path.append('../../toolbox/egm_toolbox')
from EGM_toolbox import *
import rpi_abb_irc5

def main():
	data_dir='../../data/wood/'

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

	et1=EGM_toolbox(egm1,robot1)
	et2=EGM_toolbox(egm2,robot2)

	curve_js1=read_csv(data_dir+"dual_arm/arm1.csv",header=None).values
	curve_js2=read_csv(data_dir+"dual_arm/arm2.csv",header=None).values
	relative_path=read_csv(data_dir+"Curve_dense.csv",header=None).values

	vd=50
	idx=et1.downsample24ms(relative_path,vd)
	extension_start=40
	extension_end=10

	curve_cmd_js1=curve_js1[idx]
	curve_cmd_js2=curve_js2[idx]
	
	##add extension
	curve_cmd_js1_ext=et1.add_extension_egm_js(curve_cmd_js1,extension_start=extension_start,extension_end=extension_end)
	curve_cmd_js2_ext=et1.add_extension_egm_js(curve_cmd_js2,extension_start=extension_start,extension_end=extension_end)

	
	iteration=100
	adjust_weigt_it=20

	###jog both arm to start pose
	et1.jog_joint(curve_cmd_js1_ext[0])
	et2.jog_joint(curve_cmd_js2_ext[0])



	################################traverse curve for both arms#####################################

	curve_exe_js1=[]
	timestamp1=[]
	curve_exe_js2=[]
	timestamp2=[]
	print('traversing trajectory')
	try:
		for i in range(len(curve_cmd_js1_ext)):
			while True:
				res_i1, state_i1 = egm1.receive_from_robot()
				if res_i1:
					while True:
						res_i2, state_i2 = egm2.receive_from_robot()
						if res_i2:
							send_res1 = egm1.send_to_robot(curve_cmd_js1_ext[i])
							#save joint angles
							curve_exe_js1.append(np.radians(state_i1.joint_angles))
							timestamp1.append(state_i1.robot_message.header.tm)

							

							# res_i2, state_i2 = egm2.receive_from_robot()
							send_res2 = egm2.send_to_robot(curve_cmd_js2_ext[i])
							#save joint angles
							curve_exe_js2.append(np.radians(state_i2.joint_angles))
							timestamp2.append(state_i2.robot_message.header.tm)

							break
					break
	except KeyboardInterrupt:
		raise

	timestamp1=np.array(timestamp1)/1000
	timestamp2=np.array(timestamp2)/1000

	##############################calcualte error########################################
	relative_path_exe,relative_path_exe_R=form_relative_path(robot1,robot2,curve_exe_js1[extension_num+et1.idx_delay:-extension_num+et1.idx_delay],curve_exe_js2[extension_num+et1.idx_delay:-extension_num+et1.idx_delay],base2_R,base2_p)
	lam=calc_lam_cs(relative_path_exe)
	speed=np.gradient(lam)/np.gradient(timestamp1[extension_num+et1.idx_delay:-extension_num+et1.idx_delay])
	error,angle_error=calc_all_error_w_normal(relative_path_exe,relative_path[:,:3],relative_path_exe_R[:,:,-1],relative_path[:,3:])
	print(max(error))

	plt.plot(timestamp1)
	plt.show()
	plt.plot(timestamp2)
	plt.show()
	##############################plot error#####################################

	fig, ax1 = plt.subplots()
	ax2 = ax1.twinx()
	ax1.plot(lam, speed, 'g-', label='Speed')
	ax2.plot(lam, error, 'b-',label='Error')
	ax2.plot(lam, np.degrees(angle_error), 'y-',label='Normal Error')
	ax2.axis(ymin=0,ymax=100)

	ax1.set_xlabel('lambda (mm)')
	ax1.set_ylabel('Speed/lamdot (mm/s)', color='g')
	ax2.set_ylabel('Error/Normal Error (mm/deg)', color='b')
	plt.title("Speed and Error Plot")
	ax1.legend(loc=0)

	ax2.legend(loc=0)

	plt.legend()
	plt.show()

if __name__ == '__main__':
	main()